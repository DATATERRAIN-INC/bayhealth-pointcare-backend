"""Merge Retell AI transcript with Twilio human↔human STT into one Call.transcript."""

from __future__ import annotations

import io
import json
import logging
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

_STT_PROMPT = (
    "Two-person live phone call. Transcribe only the words actually spoken. "
    "Do not invent terms. Ignore hold music and automated voice."
)
# A short list of words ("Yes. Okay. Thank you. Bye.") made Whisper "hear" those
# words in silence. Context-only prompt instead.
_WHISPER_PROMPT = (
    "A phone call between a patient and a Bay Area Community Health care coordinator."
)
_PROMPT_ECHO_MARKERS = (
    "transcribe only the words actually spoken",
    "do not invent",
    "ignore hold music",
    "a phone call between",
)

# Our own Twilio announcements (twilio_bridge / ai_sms.agent_dial). They are
# recorded with the call but nobody said them, so they never belong in the
# patient / live-agent transcript.
_SYSTEM_PHRASES = (
    "You can speak now.",
    "Great, I will connect you with a coordinator now. Please hold.",
    "That agent is unavailable. I am connecting you to the next agent now. Please hold.",
    "Please hold. A live agent is helping another caller. You are still in line.",
    "Thank you for waiting. I am connecting you to a live agent now.",
    "I am sorry, none of our enrollment agents are available right now. "
    "What day and time works best for a callback?",
    "For example, you can say tomorrow at 3 PM, or Friday morning.",
    "Sorry, I did not catch a time. We will follow up another day. Goodbye.",
    "Sorry, I did not catch that. Please tell me a day and time for your callback.",
    "I am sorry, I could not reach the coordinator. Goodbye.",
    "Thank you for calling Bay Area Community Health. Goodbye.",
    "Okay. Please continue to hold. You are still in line.",
    "Sorry for the wait. All of our agents are still helping other callers. "
    "Would you like us to call you back instead? Press 1 or say yes for a callback. "
    "Press 2 or say no to keep holding.",
)
# Announcements with a patient name / service inside them.
_SYSTEM_TEMPLATE_PATTERNS = (
    # Live-agent briefing on answer (twilio_bridge.agent_briefing_text).
    re.compile(
        r"bay area community health transfer\W+connecting you with [^.?!]*[.?!]?",
        re.I,
    ),
    re.compile(
        r"hi\W+this is a bay area community health care coordinator\W+"
        r"i am connecting .*? then leave\W*",
        re.I | re.S,
    ),
    re.compile(
        r"(hello\W+)?transfer from bay area community health\W+"
        r"connecting you now with .*? please hold\W*",
        re.I | re.S,
    ),
)
# Whisper's classic inventions on silence / hold audio when the whole turn is only this.
_HALLUCINATED_TURNS = {"you", "thank you for watching", "thanks for watching", "please subscribe"}


def _phrase_pattern(phrase: str) -> "re.Pattern[str]":
    words = re.findall(r"[a-z0-9]+", phrase.lower())
    return re.compile(r"\b" + r"\W+".join(map(re.escape, words)) + r"\W*", re.I)


_SYSTEM_PATTERNS = [_phrase_pattern(p) for p in _SYSTEM_PHRASES]


def strip_system_speech(text: str) -> str:
    """Remove our own announcements and Whisper's silence inventions from a turn."""
    cleaned = str(text or "")
    for pattern in (*_SYSTEM_TEMPLATE_PATTERNS, *_SYSTEM_PATTERNS):
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = " ".join(re.findall(r"[a-z0-9']+", cleaned.lower()))
    if not words or words in _HALLUCINATED_TURNS:
        return ""
    return cleaned


def clean_human_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Patient / live-agent turns without system announcements or empty turns."""
    cleaned: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = strip_system_speech(item.get("text"))
        if not text:
            continue
        cleaned.append({**item, "text": text})
    return _collapse_adjacent_same_speaker(cleaned) if cleaned else []


def _looks_like_prompt_echo(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(marker in lowered for marker in _PROMPT_ECHO_MARKERS)


def _openai_key() -> str:
    return (getattr(settings, "OPENAI_API_KEY", "") or "").strip()


def _stt_models(*, want_segments: bool = False) -> List[str]:
    configured = (getattr(settings, "CALL_TRANSCRIPTION_MODEL", "") or "").strip()
    models: List[str] = []
    # whisper-1 is the most reliable for verbose_json segment timestamps.
    preferred = (
        ["whisper-1", configured, "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
        if want_segments
        else [configured, "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]
    )
    for model in preferred:
        if model and model not in models:
            models.append(model)
    return models


def _parse_verbose_segments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for row in payload.get("segments") or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(row.get("start") if row.get("start") is not None else 0.0)
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(row.get("end") if row.get("end") is not None else start)
        except (TypeError, ValueError):
            end = start
        if _whisper_says_no_speech(row):
            continue
        segment = {"start": start, "end": end, "text": text}
        try:
            segment["confidence"] = float(row.get("avg_logprob"))
        except (TypeError, ValueError):
            pass
        segments.append(segment)
    return segments


_PROMPT_FIXED_WORDS = set(
    re.findall(
        r"[a-z0-9']+",
        "a phone call between a patient and a bay area community health care coordinator about",
    )
)


def _drop_prompt_echo_segments(
    segments: List[Dict[str, Any]], prompt: str
) -> List[Dict[str, Any]]:
    """Whisper sometimes "hears" its own prompt in trailing silence
    ("Bay Area Community Health"). Drop short segments made only of prompt words."""
    # Only the fixed wording — never the patient's name or service, which are
    # real answers ("John Smith.", "About a mammogram.").
    prompt_words = set(re.findall(r"[a-z0-9']+", (prompt or "").lower())) & _PROMPT_FIXED_WORDS
    kept = []
    for seg in segments:
        words = re.findall(r"[a-z0-9']+", seg["text"].lower())
        if words and len(words) <= 6 and set(words) <= prompt_words:
            continue
        kept.append(seg)
    return kept


def _whisper_says_no_speech(row: Dict[str, Any]) -> bool:
    """Whisper's silence rule: likely no speech AND low confidence → invented text."""
    try:
        no_speech = float(row.get("no_speech_prob"))
        avg_logprob = float(row.get("avg_logprob"))
    except (TypeError, ValueError):
        return False
    return no_speech > 0.6 and avg_logprob < -1.0


def _parse_diarized_segments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse gpt-4o-transcribe-diarize / diarized_json style responses."""
    segments: List[Dict[str, Any]] = []
    rows = payload.get("segments") or payload.get("utterances") or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        speaker = str(
            row.get("speaker") or row.get("speaker_label") or row.get("role") or ""
        ).strip()
        try:
            start = float(row.get("start") if row.get("start") is not None else 0.0)
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(row.get("end") if row.get("end") is not None else start)
        except (TypeError, ValueError):
            end = start
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
            }
        )
    return segments


_STT_RETRY_WAITS = (2, 6)  # seconds between attempts on temporary failures
_STT_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _post_transcription(key: str, filename: str, audio: bytes, data: Dict[str, Any]):
    """POST to OpenAI speech-to-text, retrying network errors, 429 and 5xx.

    A one-off hiccup must not leave the live-agent transcript empty for good.
    """
    attempt = 0
    while True:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (filename, audio)},
                data=data,
                timeout=120,
            )
            if resp.status_code not in _STT_RETRY_STATUS or attempt >= len(_STT_RETRY_WAITS):
                return resp
            reason = f"status={resp.status_code}"
        except requests.RequestException as exc:
            if attempt >= len(_STT_RETRY_WAITS):
                raise
            reason = f"error={exc}"
        logger.warning(
            "STT retry model=%s file=%s attempt=%s %s",
            data.get("model"),
            filename,
            attempt + 1,
            reason,
        )
        time.sleep(_STT_RETRY_WAITS[attempt])
        attempt += 1


_REFINE_PAD_SECONDS = 0.2
_REFINE_WORKERS = 4


def _refine_model() -> str:
    return (getattr(settings, "CALL_TRANSCRIPTION_REFINE_MODEL", "") or "").strip()


def _wav_clip(audio: bytes, start: float, end: float) -> Optional[bytes]:
    """The [start, end] seconds of a WAV as its own WAV (None if not a WAV)."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as src:
            params = src.getparams()
            rate = src.getframerate()
            total = src.getnframes()
            first = max(0, int(start * rate))
            last = min(total, int(end * rate))
            if last <= first:
                return None
            src.setpos(first)
            frames = src.readframes(last - first)
    except (wave.Error, EOFError, ValueError):
        return None
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
    return out.getvalue()


# gpt-4o-transcribe sometimes leaks prompt markup into noisy clips ("context: ###").
_REFINE_JUNK = re.compile(r"\bcontext:|#+", re.IGNORECASE)
# Better model heard nothing: drop Whisper's sentence only when it is long enough
# to be an invented one (short "Yes." / "Okay." answers are always kept).
_REFINE_DROP_MIN_WORDS = 4
_HEARD_NOTHING = None


def _refine_one(key: str, model: str, clip: bytes, prompt: str, whisper_text: str) -> Optional[str]:
    """
    Better model's words for one sentence.

    "" = keep Whisper's text (request failed / answer not usable),
    None (_HEARD_NOTHING) = the better model heard no speech in this clip.
    """
    data = {"model": model, "response_format": "json", "prompt": prompt, "temperature": "0"}
    try:
        resp = _post_transcription(key, "segment.wav", clip, data)
    except requests.RequestException:
        return ""
    if resp.status_code >= 400:
        return ""
    body = (resp.text or "").strip()
    try:
        text = str(json.loads(body).get("text") or "").strip() if body.startswith("{") else body
    except ValueError:
        text = body
    if _REFINE_JUNK.search(text):
        text = re.sub(r"\s+", " ", _REFINE_JUNK.sub(" ", text)).strip()
    if not re.search(r"[A-Za-z0-9]", text):
        return _HEARD_NOTHING
    if _looks_like_prompt_echo(text):
        return ""
    # Far longer than what Whisper heard in the same audio = invented text.
    if len(text.split()) > 3 * len(whisper_text.split()) + 5:
        return ""
    return text


def refine_segment_text(
    audio: bytes, segments: List[Dict[str, Any]], *, key: str, prompt: str
) -> List[Dict[str, Any]]:
    """
    Re-transcribe each Whisper sentence with the more accurate model.

    Timing (start/end/confidence) stays Whisper's, so patient/agent order is
    unchanged. Any sentence the better model cannot do keeps Whisper's text.
    """
    model = _refine_model()
    if not model or not segments or not key:
        return segments
    jobs = []
    for i, seg in enumerate(segments):
        # A little padding, but never into the neighbouring sentence (no doubled words).
        prev_end = segments[i - 1]["end"] if i > 0 else 0.0
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else seg["end"] + 10
        start = max(prev_end, seg["start"] - _REFINE_PAD_SECONDS)
        end = min(next_start, seg["end"] + _REFINE_PAD_SECONDS)
        jobs.append(_wav_clip(audio, start, end))
    if not any(jobs):
        return segments  # not a WAV we can cut: keep Whisper's text
    with ThreadPoolExecutor(max_workers=_REFINE_WORKERS) as pool:
        texts = list(
            pool.map(
                lambda pair: (
                    _refine_one(key, model, pair[1], prompt, pair[0]["text"]) if pair[1] else ""
                ),
                zip(segments, jobs),
            )
        )
    refined = []
    improved = dropped = 0
    for seg, text in zip(segments, texts):
        if text is _HEARD_NOTHING and len(seg["text"].split()) >= _REFINE_DROP_MIN_WORDS:
            # Whisper wrote a whole sentence where the better model hears nothing:
            # Whisper invented it (noise / silence), so it is not stored.
            dropped += 1
            continue
        if text:
            improved += 1
            refined.append({**seg, "text": text, "whisper_text": seg["text"]})
        else:
            refined.append(seg)
    logger.info(
        "STT refine model=%s segments=%s refined=%s dropped=%s",
        model, len(segments), improved, dropped,
    )
    return refined


def transcribe_recording_detailed(
    audio: bytes, filename: str = "call.wav", *, context: str = ""
) -> Dict[str, Any]:
    """
    Transcribe audio. Prefer segment timestamps for turn interleaving.

    Returns {"text": str, "segments": [{"start","end","text"}, ...]}.
    """
    key = _openai_key()
    if not key or not audio:
        return {"text": "", "segments": []}

    for model in _stt_models(want_segments=True):
        # whisper-1 supports verbose_json + segment timestamps.
        # Names in the prompt (patient, service) help Whisper spell them right.
        prompt = (context or "").strip() or (_WHISPER_PROMPT if "whisper" in model else _STT_PROMPT)
        if "whisper" in model:
            data: Dict[str, Any] = {
                "model": model,
                "response_format": "verbose_json",
                "prompt": prompt,
                "temperature": "0",
            }
        else:
            # gpt-4o-transcribe family: try verbose_json first, then text.
            data = {
                "model": model,
                "response_format": "verbose_json",
                "prompt": prompt,
                "temperature": "0",
            }
        try:
            resp = _post_transcription(key, filename, audio, data)
            if resp.status_code >= 400:
                # Retry same model as plain text if verbose_json unsupported.
                if data.get("response_format") == "verbose_json":
                    data = {
                        "model": model,
                        "response_format": "text",
                        "prompt": prompt,
                        "temperature": "0",
                    }
                    resp = _post_transcription(key, filename, audio, data)
                if resp.status_code >= 400:
                    logger.warning(
                        "STT model=%s failed status=%s body=%s",
                        model,
                        resp.status_code,
                        (resp.text or "")[:300],
                    )
                    continue

            body = (resp.text or "").strip()
            if not body:
                continue

            if body.startswith("{"):
                try:
                    payload = json.loads(body)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    text = str(payload.get("text") or "").strip()
                    segments = _drop_prompt_echo_segments(
                        _parse_verbose_segments(payload), prompt
                    )
                    if _looks_like_prompt_echo(text):
                        continue
                    if payload.get("segments"):
                        if "whisper" in model:
                            # Keep Whisper's timing, take the better model's words.
                            segments = refine_segment_text(audio, segments, key=key, prompt=prompt)
                        # Only what survived the no-speech filter.
                        text = " ".join(s["text"] for s in segments).strip()
                    # A parsed response is the answer, even when empty (silence).
                    return {"text": text, "segments": segments}

            if _looks_like_prompt_echo(body):
                continue
            return {"text": body, "segments": []}
        except Exception:
            logger.exception("OpenAI transcription failed model=%s", model)
    return {"text": "", "segments": []}


def _token_set(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _flip_patient_live_agent(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flipped = []
    for item in items:
        row = dict(item)
        sp = str(row.get("speaker") or "")
        if sp == "patient":
            row["speaker"] = "live_agent"
            row["name"] = "Live agent"
        elif sp == "live_agent":
            row["speaker"] = "patient"
            row["name"] = "Patient"
        flipped.append(row)
    return flipped


def remap_diarized_with_provider_anchor(
    items: List[Dict[str, Any]],
    provider_text: str = "",
    patient_text: str = "",
) -> List[Dict[str, Any]]:
    """Align diarized clusters to patient / live_agent for any topic.

    Uses Twilio leg STT as voice anchors (not appointment keywords):
    - cluster closest to provider-leg text → live_agent
    - cluster closest to patient-leg text → patient
    Works for scheduling, general questions, callbacks, etc.
    """
    if not items:
        return items
    by_speaker: Dict[str, str] = {"patient": "", "live_agent": ""}
    for item in items:
        sp = str(item.get("speaker") or "")
        if sp in by_speaker:
            by_speaker[sp] = (
                f"{by_speaker[sp]} {item.get('text') or ''}".strip()
            )
    patient_tok = _token_set(by_speaker["patient"])
    agent_tok = _token_set(by_speaker["live_agent"])
    if not patient_tok or not agent_tok:
        return items

    provider_tok = _token_set(provider_text)
    patient_leg_tok = _token_set(patient_text)

    # Prefer two-anchor assignment when both legs have text.
    if len(provider_tok) >= 3 and len(patient_leg_tok) >= 3:
        # Score: provider→agent + patient→patient  vs  swapped.
        score_ok = _jaccard(provider_tok, agent_tok) + _jaccard(
            patient_leg_tok, patient_tok
        )
        score_swap = _jaccard(provider_tok, patient_tok) + _jaccard(
            patient_leg_tok, agent_tok
        )
        if score_swap > score_ok + 0.05:
            logger.info(
                "Remapping diarized speakers (two-leg anchors) "
                "ok=%.3f swap=%.3f",
                score_ok,
                score_swap,
            )
            return _flip_patient_live_agent(items)
        return items

    # Provider-only anchor: matching cluster is always live_agent.
    if len(provider_tok) >= 3:
        patient_score = _jaccard(provider_tok, patient_tok)
        agent_score = _jaccard(provider_tok, agent_tok)
        if patient_score >= agent_score + 0.08 and patient_score >= 0.12:
            logger.info(
                "Remapping diarized speakers (provider→live_agent) "
                "patient_score=%.3f agent_score=%.3f",
                patient_score,
                agent_score,
            )
            return _flip_patient_live_agent(items)
        return items

    # Patient-only anchor: matching cluster is always patient.
    if len(patient_leg_tok) >= 3:
        patient_score = _jaccard(patient_leg_tok, patient_tok)
        agent_score = _jaccard(patient_leg_tok, agent_tok)
        if agent_score >= patient_score + 0.08 and agent_score >= 0.12:
            logger.info(
                "Remapping diarized speakers (patient-leg→patient) "
                "patient_score=%.3f agent_score=%.3f",
                patient_score,
                agent_score,
            )
            return _flip_patient_live_agent(items)
    return items


def transcribe_mix_with_diarization(
    audio: bytes,
    filename: str = "mix.wav",
    provider_text: str = "",
    patient_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Best-effort speaker turns from a mixed recording.
    Topic-agnostic: works for any patient ↔ live-agent conversation.
    """
    key = _openai_key()
    if not key or not audio:
        return []

    models = []
    for model in (
        "gpt-4o-transcribe-diarize",
        (getattr(settings, "CALL_TRANSCRIPTION_MODEL", "") or "").strip(),
    ):
        if model and model not in models:
            models.append(model)

    for model in models:
        if "diarize" not in model:
            continue
        # OpenAI now requires chunking_strategy for diarization models.
        data = {
            "model": model,
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
        }
        try:
            resp = _post_transcription(key, filename, audio, data)
            if resp.status_code >= 400:
                logger.warning(
                    "Diarize STT model=%s failed status=%s body=%s",
                    model,
                    resp.status_code,
                    (resp.text or "")[:300],
                )
                continue
            payload = resp.json()
            if not isinstance(payload, dict):
                continue
            raw_segments = _parse_diarized_segments(payload)
            if not raw_segments:
                continue
            items = _diarized_segments_to_items(raw_segments)
            return remap_diarized_with_provider_anchor(
                items,
                provider_text=provider_text,
                patient_text=patient_text,
            )
        except Exception:
            logger.exception("Diarized transcription failed model=%s", model)
    return []


def _map_diarized_speaker(raw: str, speaker_map: Dict[str, str]) -> str:
    """Map diarization labels onto patient / live_agent only (2-party transfer)."""
    label = (raw or "").strip()
    if not label:
        return "unknown"
    if label in speaker_map:
        return speaker_map[label]
    lowered = label.lower().replace(" ", "_")
    if lowered in {"speaker_0", "a", "spk_0", "0", "speaker0", "patient"}:
        assigned = "patient"
    elif lowered in {
        "speaker_1",
        "b",
        "spk_1",
        "1",
        "speaker1",
        "live_agent",
        "agent",
        "provider",
    }:
        assigned = "live_agent"
    elif len(speaker_map) == 0:
        assigned = "patient"
    elif "live_agent" not in speaker_map.values():
        assigned = "live_agent"
    else:
        # Extra clusters on a 2-party call → attach to live_agent.
        assigned = "live_agent"
    speaker_map[label] = assigned
    return assigned


def _diarized_segments_to_items(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    speaker_map: Dict[str, str] = {}
    items: List[Dict[str, Any]] = []
    for seg in sorted(segments, key=lambda s: (s.get("start") or 0.0, s.get("end") or 0.0)):
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        speaker = _map_diarized_speaker(str(seg.get("speaker") or ""), speaker_map)
        if speaker not in {"patient", "live_agent"}:
            speaker = "live_agent"
        items.append(
            {
                "speaker": speaker,
                "text": text,
                "segment": "human",
                "name": _speaker_display_name(speaker),
                "at": round(float(seg.get("start") or 0.0), 3),
            }
        )
    return _collapse_adjacent_same_speaker(items)


def _speaker_display_name(speaker: str) -> str:
    sp = (speaker or "").strip().lower()
    if sp in {"patient", "user"}:
        return "Patient"
    if sp in {"live_agent", "provider"}:
        return "Live agent"
    if sp in {"agent", "ai", "ai_agent"}:
        return "AI agent"
    return "Live agent" if sp else "Patient"


def _labeled_speaker_item(speaker: str, text: str, at=None) -> Optional[Dict[str, Any]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    return {
        "speaker": speaker,
        "text": cleaned,
        "segment": "human",
        "name": _speaker_display_name(speaker),
        "at": at,
    }


def labeled_leg_items(
    *,
    patient_text: str = "",
    provider_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Guaranteed speaker-separated items from per-leg STT text.

    Inbound Twilio tracks are already single-speaker, so labels are reliable
    even when OpenAI returns no timestamps.
    """
    items: List[Dict[str, Any]] = []
    patient_item = _labeled_speaker_item("patient", patient_text)
    provider_item = _labeled_speaker_item("live_agent", provider_text)
    if patient_item:
        items.append(patient_item)
    if provider_item:
        items.append(provider_item)
    return items


def _absolute(seg: Dict[str, Any], offset: float) -> Dict[str, Any]:
    start = float(offset) + float(seg.get("start") or 0.0)
    end = float(offset) + float(seg.get("end") or seg.get("start") or 0.0)
    return {**seg, "start": start, "end": end, "text": str(seg.get("text") or "").strip()}


_ECHO_MIN_WORDS = 3
_ECHO_SIMILARITY = 0.75
_ECHO_TIME_SLACK_SECONDS = 2.0


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _is_echo_pair(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Same words heard on both channels at the same moment = one voice leaking."""
    if a["start"] > b["end"] + _ECHO_TIME_SLACK_SECONDS:
        return False
    if b["start"] > a["end"] + _ECHO_TIME_SLACK_SECONDS:
        return False
    wa, wb = _words(a["text"]), _words(b["text"])
    if min(len(wa), len(wb)) < _ECHO_MIN_WORDS:
        return False
    return SequenceMatcher(None, wa, wb).ratio() >= _ECHO_SIMILARITY


def drop_cross_talk(
    patient: List[Dict[str, Any]], agent: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Transcript-level echo cancellation between the two speaker channels.

    A speakerphone or open mic makes one person's words leak into the other
    person's recording, so Whisper writes them twice under both names. Keep
    the copy Whisper was more confident about (the real voice is clearer than
    its echo); without scores, keep the agent copy (the agent's line is clean).
    """
    drop_patient, drop_agent = set(), set()
    for pi, p in enumerate(patient):
        for ai, a in enumerate(agent):
            if ai in drop_agent or not _is_echo_pair(p, a):
                continue
            p_conf, a_conf = p.get("confidence"), a.get("confidence")
            if p_conf is not None and a_conf is not None and p_conf > a_conf + 0.3:
                drop_agent.add(ai)
            else:
                drop_patient.add(pi)
            break
    kept_agent = [s for i, s in enumerate(agent) if i not in drop_agent]
    kept_patient = []
    for i, seg in enumerate(patient):
        if i in drop_patient:
            continue
        seg = _strip_echoed_sentences(seg, kept_agent)
        if seg["text"]:
            kept_patient.append(seg)
    return kept_patient, kept_agent


def _sentences(text: str) -> List[str]:
    # Only split before a capital letter, so "3 p.m. works" / "Dr. John" stay whole.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text or "") if s.strip()]


def _strip_echoed_sentences(seg: Dict[str, Any], agent: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Whisper can glue an echo tail and the patient's own words into one segment
    ("Have a good day. Thank you. Bye."). Drop sentences that repeat what the
    agent was saying at that same moment. A reply after the agent stops
    talking ("You too, have a good day") does not overlap and is kept.
    """
    # Echo starts while the agent is still talking (Whisper timestamps can be ~1s early).
    overlapping = [a for a in agent if a["start"] - 1.0 <= seg["start"] <= a["end"] + 0.3]
    if not overlapping:
        return seg
    agent_sentences = [_words(x) for a in overlapping for x in _sentences(a["text"])]
    kept = []
    for sentence in _sentences(seg["text"]):
        words = _words(sentence)
        echoed = len(words) >= _ECHO_MIN_WORDS and any(
            SequenceMatcher(None, words, other).ratio() >= 0.8 for other in agent_sentences
        )
        if not echoed:
            kept.append(sentence)
    return {**seg, "text": " ".join(kept).strip()}


def interleave_speaker_segments(
    *,
    patient_segments: Optional[List[Dict[str, Any]]] = None,
    provider_segments: Optional[List[Dict[str, Any]]] = None,
    patient_offset: float = 0.0,
    provider_offset: float = 0.0,
    patient_text: str = "",
    provider_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Build turn-by-turn human items from per-leg STT segments.

    Absolute time = recording_start_offset + segment.start so late-joining
    provider audio lines up with the patient leg.

    Fallback (no segments): two labeled blobs — still 100% speaker-separated.
    """
    patient_segs = [s for s in (patient_segments or []) if str(s.get("text") or "").strip()]
    provider_segs = [s for s in (provider_segments or []) if str(s.get("text") or "").strip()]

    # No timestamps → labeled per-leg blobs (never one mixed unknown wall).
    if not patient_segs and not provider_segs:
        return labeled_leg_items(
            patient_text=patient_text,
            provider_text=provider_text,
        )

    patient_abs = [_absolute(seg, patient_offset) for seg in patient_segs]
    provider_abs = [_absolute(seg, provider_offset) for seg in provider_segs]
    patient_abs, provider_abs = drop_cross_talk(patient_abs, provider_abs)

    timed: List[Tuple[float, float, int, str, str]] = []
    for seg in patient_abs:
        timed.append((seg["start"], seg["end"], 0, "patient", seg["text"]))
    for seg in provider_abs:
        timed.append((seg["start"], seg["end"], 1, "live_agent", seg["text"]))

    timed.sort(key=lambda row: (row[0], row[2], row[1]))
    items = [
        {
            "speaker": speaker,
            "text": text,
            "segment": "human",
            "at": round(start, 3),
        }
        for start, _end, _rank, speaker, text in timed
    ]

    # Append plain-text-only leg so we never drop a speaker.
    if not patient_segs and (patient_text or "").strip():
        item = _labeled_speaker_item("patient", patient_text)
        if item:
            items.append(item)
    if not provider_segs and (provider_text or "").strip():
        item = _labeled_speaker_item("live_agent", provider_text)
        if item:
            items.append(item)

    return _collapse_adjacent_same_speaker(items)

def _collapse_adjacent_same_speaker(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive same-speaker fragments into readable turns."""
    merged: List[Dict[str, Any]] = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if (
            merged
            and merged[-1].get("speaker") == item.get("speaker")
            and merged[-1].get("segment") == item.get("segment")
        ):
            prev = merged[-1]["text"]
            # Avoid double spaces / duplicated joins.
            joiner = "" if prev.endswith((" ", "\n")) or text.startswith((" ", "\n")) else " "
            merged[-1]["text"] = f"{prev}{joiner}{text}".strip()
            continue
        merged.append(
            {
                "speaker": item.get("speaker") or "unknown",
                "text": text,
                "segment": item.get("segment") or "human",
                "at": item.get("at"),
            }
        )
    return merged


def humans_items_from_texts(
    *,
    patient_text: str = "",
    provider_text: str = "",
    mix_text: str = "",
    patient_segments: Optional[List[Dict[str, Any]]] = None,
    provider_segments: Optional[List[Dict[str, Any]]] = None,
    patient_offset: float = 0.0,
    provider_offset: float = 0.0,
    mix_items: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build human transcript items.

    Priority:
    1) Diarized mix_items with both speakers (most reliable for mono legs)
    2) Per-leg patient/provider text or segments
    3) Raw mix_text as speaker=unknown (last resort only)
    """
    if mix_items:
        mix_speakers = {
            str(i.get("speaker") or "")
            for i in mix_items
            if isinstance(i, dict) and str(i.get("text") or "").strip()
        }
        if {"patient", "live_agent"}.issubset(mix_speakers):
            return list(mix_items)

    has_legs = bool(
        (patient_text or "").strip()
        or (provider_text or "").strip()
        or (patient_segments or [])
        or (provider_segments or [])
    )
    if has_legs:
        return interleave_speaker_segments(
            patient_segments=patient_segments,
            provider_segments=provider_segments,
            patient_offset=patient_offset,
            provider_offset=provider_offset,
            patient_text=patient_text,
            provider_text=provider_text,
        )

    if mix_items:
        return list(mix_items)

    if (mix_text or "").strip():
        return [
            {
                "speaker": "unknown",
                "text": mix_text.strip(),
                "segment": "human",
                "at": None,
            }
        ]
    return []


def assert_speakers_separated(items: List[Dict[str, Any]]) -> bool:
    """True when human items never mix both people under speaker=unknown."""
    if not items:
        return False
    speakers = {str(i.get("speaker") or "") for i in items}
    if speakers == {"unknown"}:
        return False
    # Any labeled patient/live_agent means separation worked.
    return bool(speakers & {"patient", "live_agent"})

def tag_ai_segment(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tagged = []
    for item in items or []:
        row = dict(item)
        row["segment"] = "ai"
        tagged.append(row)
    return tagged


def is_log_url(url: str) -> bool:
    """Retell's public_log_url (…/public.log) is a debug log, not audio."""
    return (url or "").split("?", 1)[0].strip().lower().endswith(".log")


def merge_ai_and_humans(ai_items, humans_items) -> List[Dict[str, Any]]:
    return list(tag_ai_segment(ai_items or [])) + list(humans_items or [])


def _display_name_for_transcript_row(item: Dict[str, Any]) -> str:
    """UI label: AI agent vs Live agent vs Patient.

    Prefer explicit `name` when already set. Never treat all human
    segments as Live agent — patient turns are also segment=human.
    """
    existing = str(item.get("name") or "").strip()
    if existing in {"AI agent", "Live agent", "Patient"}:
        return existing
    speaker = str(item.get("speaker") or "").strip().lower()
    segment = str(item.get("segment") or "").strip().lower()
    if speaker in {"patient", "user"}:
        return "Patient"
    if speaker in {"live_agent", "provider"}:
        return "Live agent"
    if segment == "human":
        return "Live agent"
    return "AI agent"


def transcript_for_api(items) -> List[Dict[str, Any]]:
    """Public transcript rows: no `at`, plus display `name` for the UI."""
    cleaned: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        row = {
            key: value
            for key, value in item.items()
            if key not in {"at", "name"}
        }
        if not row:
            continue
        row["name"] = _display_name_for_transcript_row(row)
        cleaned.append(row)
    return cleaned


def save_humans_transcript_for_call(
    *,
    retell_call_id: str,
    patient_text: str = "",
    provider_text: str = "",
    mix_text: str = "",
    recording_url: str = "",
    live_agent_recording_url: str = "",
    twilio_call_sid: str = "",
    patient_segments: Optional[List[Dict[str, Any]]] = None,
    provider_segments: Optional[List[Dict[str, Any]]] = None,
    patient_offset: float = 0.0,
    provider_offset: float = 0.0,
    mix_items: Optional[List[Dict[str, Any]]] = None,
):
    from apps.ai_caller.models import Call

    call_id = (retell_call_id or "").strip()
    if not call_id:
        return None

    call = Call.objects.filter(retell_call_id=call_id).first()
    if not call:
        logger.warning("No Call row for retell_call_id=%s (humans transcript)", call_id)
        return None

    humans = humans_items_from_texts(
        patient_text=patient_text,
        provider_text=provider_text,
        mix_text=mix_text,
        patient_segments=patient_segments,
        provider_segments=provider_segments,
        patient_offset=patient_offset,
        provider_offset=provider_offset,
        mix_items=mix_items,
    )
    humans = clean_human_items(humans)
    if not humans:
        # Still persist Twilio Call SID / recording URLs if we have them.
        update_fields = []
        sid = (twilio_call_sid or "").strip()
        if sid and call.warm_transfer_session_id != sid:
            call.warm_transfer_session_id = sid[:64]
            update_fields.append("warm_transfer_session_id")
        # recording_url holds the AI (Retell) audio; only fill it when empty.
        if recording_url and not (call.recording_url or "").strip():
            call.recording_url = recording_url[:1024]
            update_fields.append("recording_url")
        if live_agent_recording_url and live_agent_recording_url != (
            call.live_agent_recording_url or ""
        ):
            call.live_agent_recording_url = live_agent_recording_url[:1024]
            update_fields.append("live_agent_recording_url")
        if update_fields:
            update_fields.append("updated_at")
            call.save(update_fields=update_fields)
        return call

    existing = call.live_agent_transcript or []
    existing_speakers = {
        str(i.get("speaker") or "") for i in existing if isinstance(i, dict)
    }
    new_speakers = {str(i.get("speaker") or "") for i in humans}
    existing_split = {"patient", "live_agent"}.issubset(existing_speakers)
    new_split = {"patient", "live_agent"}.issubset(new_speakers)
    # Never replace a good patient+live_agent split with a single wall.
    if existing and existing_split and not new_split:
        logger.info(
            "Keeping existing split humans transcript call_id=%s "
            "(new had speakers=%s)",
            call_id,
            sorted(new_speakers),
        )
        humans = existing
    elif new_split:
        # Always accept a real split (mix/diarize may refine labels).
        call.live_agent_transcript = humans
    elif not existing:
        call.live_agent_transcript = humans
    else:
        if len(humans) > len(existing):
            call.live_agent_transcript = humans
        else:
            humans = existing

    # recording_url holds the AI (Retell) audio; only fill it when empty.
    if recording_url and not (call.recording_url or "").strip():
        call.recording_url = recording_url[:1024]
    if live_agent_recording_url:
        call.live_agent_recording_url = live_agent_recording_url[:1024]
    sid = (twilio_call_sid or "").strip()
    if sid:
        call.warm_transfer_session_id = sid[:64]
    call.transcript = merge_ai_and_humans(call.retell_transcript or [], humans)
    call.save(
        update_fields=[
            "live_agent_transcript",
            "recording_url",
            "live_agent_recording_url",
            "warm_transfer_session_id",
            "transcript",
            "updated_at",
        ]
    )
    logger.info(
        "Merged humans transcript call_id=%s humans=%s total=%s twilio_call_sid=%s",
        call_id,
        len(humans),
        len(call.transcript or []),
        sid or "-",
    )
    return call
