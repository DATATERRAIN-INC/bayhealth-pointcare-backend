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
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


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


def transcribe_recording_bytes(audio: bytes, filename: str = "call.wav") -> str:
    return str(transcribe_recording_detailed(audio, filename=filename).get("text") or "")


def transcribe_mix_with_diarization(
    audio: bytes, filename: str = "mix.wav"
) -> List[Dict[str, Any]]:
    """
    Best-effort speaker turns from a mixed recording.
    Returns human transcript items (may be empty).
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
        data = {
            "model": model,
            "response_format": "diarized_json",
        }
        try:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (filename, audio)},
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
            return _diarized_segments_to_items(raw_segments)
        except Exception:
            logger.exception("Diarized transcription failed model=%s", model)
    return []


def _map_diarized_speaker(raw: str, speaker_map: Dict[str, str]) -> str:
    label = (raw or "").strip()
    if not label:
        return "unknown"
    if label in speaker_map:
        return speaker_map[label]
    lowered = label.lower().replace(" ", "_")
    if lowered in {"speaker_0", "a", "spk_0", "0", "speaker0"}:
        assigned = "patient"
    elif lowered in {"speaker_1", "b", "spk_1", "1", "speaker1"}:
        assigned = "live_agent"
    elif len(speaker_map) == 0:
        assigned = "patient"
    elif len(speaker_map) == 1:
        assigned = "live_agent"
    else:
        assigned = f"speaker_{len(speaker_map) + 1}"
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
        items.append(
            {
                "speaker": speaker,
                "text": text,
                "segment": "human",
                "at": round(float(seg.get("start") or 0.0), 3),
            }
        )
    return _collapse_adjacent_same_speaker(items)


def _segments_from_plain_text(text: str) -> List[Dict[str, Any]]:
    """Fallback: split a speaker blob into sentence-ish turns (no real timestamps)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(cleaned) if p and p.strip()]
    if not parts:
        parts = [cleaned]
    segments = []
    cursor = 0.0
    for part in parts:
        # Fake sequential offsets so patient/provider can still be interleaved
        # roughly if both sides only have plain text (better than one wall).
        duration = max(1.5, min(12.0, len(part.split()) * 0.35))
        segments.append(
            {
                "start": cursor,
                "end": cursor + duration,
                "text": part,
            }
        )
        cursor += duration
    return segments


def _labeled_speaker_item(speaker: str, text: str, at=None) -> Optional[Dict[str, Any]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    return {
        "speaker": speaker,
        "text": cleaned,
        "segment": "human",
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
    1) Per-leg patient/provider text or segments (always speaker-labeled)
    2) Diarized mix_items
    3) Raw mix_text as speaker=unknown (last resort only)
    """
    has_legs = bool(
        (patient_text or "").strip()
        or (provider_text or "").strip()
        or (patient_segments or [])
        or (provider_segments or [])
    )
    if has_legs:
        # Never fall through to unlabeled mix when per-leg audio exists.
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


def save_humans_transcript_for_call(
    *,
    retell_call_id: str,
    patient_text: str = "",
    provider_text: str = "",
    mix_text: str = "",
    recording_url: str = "",
    session_id: str = "",
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

    call = Call.all_objects.filter(retell_call_id=call_id).first()
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
        return call

    call.live_agent_transcript = humans
    if recording_url:
        call.recording_url = recording_url[:1024]
    if session_id and not call.warm_transfer_session_id:
        call.warm_transfer_session_id = session_id[:64]
    call.transcript = merge_ai_and_humans(call.retell_transcript or [], humans)
    call.save(
        update_fields=[
            "live_agent_transcript",
            "recording_url",
            "warm_transfer_session_id",
            "transcript",
            "updated_at",
        ]
    )
    logger.info(
        "Merged humans transcript call_id=%s humans=%s total=%s",
        call_id,
        len(humans),
        len(call.transcript or []),
    )
    return call
