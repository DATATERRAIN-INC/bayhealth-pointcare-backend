"""Merge Retell AI transcript with Twilio human↔human STT into one Call.transcript."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

_STT_PROMPT = (
    "Two-person live phone call. Transcribe only the words actually spoken. "
    "Do not invent terms. Ignore hold music and automated voice."
)
_WHISPER_PROMPT = "Hi. Yes. Okay. Thursday. four PM. Thank you. Bye."
_PROMPT_ECHO_MARKERS = (
    "transcribe only the words actually spoken",
    "do not invent",
    "ignore hold music",
)
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
        segments.append({"start": start, "end": end, "text": text})
    return segments


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


def transcribe_recording_detailed(
    audio: bytes, filename: str = "call.wav"
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
        if "whisper" in model:
            data: Dict[str, Any] = {
                "model": model,
                "response_format": "verbose_json",
                "prompt": _WHISPER_PROMPT,
            }
        else:
            # gpt-4o-transcribe family: try verbose_json first, then text.
            data = {
                "model": model,
                "response_format": "verbose_json",
                "prompt": _STT_PROMPT,
            }
        try:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (filename, audio)},
                data=data,
                timeout=120,
            )
            if resp.status_code >= 400:
                # Retry same model as plain text if verbose_json unsupported.
                if data.get("response_format") == "verbose_json":
                    data = {
                        "model": model,
                        "response_format": "text",
                        "prompt": _WHISPER_PROMPT if "whisper" in model else _STT_PROMPT,
                    }
                    resp = requests.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {key}"},
                        files={"file": (filename, audio)},
                        data=data,
                        timeout=120,
                    )
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
                    segments = _parse_verbose_segments(payload)
                    if _looks_like_prompt_echo(text):
                        continue
                    if text or segments:
                        if not text and segments:
                            text = " ".join(s["text"] for s in segments).strip()
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
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (filename, audio, "audio/wav")},
                data=data,
                timeout=180,
            )
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

    timed: List[Tuple[float, float, int, str, str]] = []
    for seg in patient_segs:
        text = str(seg.get("text") or "").strip()
        start = float(patient_offset) + float(seg.get("start") or 0.0)
        end = float(patient_offset) + float(seg.get("end") or seg.get("start") or 0.0)
        timed.append((start, end, 0, "patient", text))
    for seg in provider_segs:
        text = str(seg.get("text") or "").strip()
        start = float(provider_offset) + float(seg.get("start") or 0.0)
        end = float(provider_offset) + float(seg.get("end") or seg.get("start") or 0.0)
        timed.append((start, end, 1, "live_agent", text))

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
    if not humans:
        # Still persist Twilio Call SID / recording URLs if we have them.
        update_fields = []
        sid = (twilio_call_sid or "").strip()
        if sid and call.warm_transfer_session_id != sid:
            call.warm_transfer_session_id = sid[:64]
            update_fields.append("warm_transfer_session_id")
        if recording_url and recording_url != (call.recording_url or ""):
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

    if recording_url:
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
