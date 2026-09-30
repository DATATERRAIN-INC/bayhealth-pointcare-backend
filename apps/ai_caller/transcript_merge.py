"""Merge Retell AI transcript with Twilio human↔human STT into one Call.transcript."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

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


def transcribe_recording_bytes(audio: bytes, filename: str = "call.wav") -> str:
    key = (getattr(settings, "OPENAI_API_KEY", "") or "").strip()
    if not key or not audio:
        return ""

    configured = (getattr(settings, "CALL_TRANSCRIPTION_MODEL", "") or "").strip()
    models = []
    for model in (configured, "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"):
        if model and model not in models:
            models.append(model)

    for model in models:
        data = {"model": model, "response_format": "text"}
        data["prompt"] = _WHISPER_PROMPT if "whisper" in model else _STT_PROMPT
        try:
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
            text = (resp.text or "").strip()
            if text.startswith("{") and '"text"' in text:
                try:
                    text = str(json.loads(text).get("text") or "").strip()
                except Exception:
                    pass
            if _looks_like_prompt_echo(text):
                continue
            if text:
                return text
        except Exception:
            logger.exception("OpenAI transcription failed model=%s", model)
    return ""


def humans_items_from_texts(*, patient_text: str = "", provider_text: str = "", mix_text: str = "") -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if (patient_text or "").strip():
        items.append(
            {
                "speaker": "patient",
                "text": patient_text.strip(),
                "segment": "human",
                "at": None,
            }
        )
    if (provider_text or "").strip():
        items.append(
            {
                "speaker": "live_agent",
                "text": provider_text.strip(),
                "segment": "human",
                "at": None,
            }
        )
    if not items and (mix_text or "").strip():
        items.append(
            {
                "speaker": "unknown",
                "text": mix_text.strip(),
                "segment": "human",
                "at": None,
            }
        )
    return items


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
