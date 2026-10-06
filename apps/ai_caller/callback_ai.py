"""OpenAI helper to resolve patient callback times from free-form speech."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, List, Optional, Tuple

import requests
from django.conf import settings
from django.utils import timezone

dialer_logger = logging.getLogger("ai_caller.dialer")


def _openai_key() -> str:
    return (getattr(settings, "OPENAI_API_KEY", "") or "").strip()


def _resolve_timezone(tz_name: str):
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # Python < 3.9
        from backports.zoneinfo import ZoneInfo

    return ZoneInfo((tz_name or "America/New_York").strip())


def _transcript_to_lines(transcript) -> str:
    if not isinstance(transcript, list):
        return str(transcript or "").strip()
    lines: List[str] = []
    for entry in transcript:
        if not isinstance(entry, dict):
            continue
        speaker = str(entry.get("speaker") or entry.get("role") or "unknown").strip()
        text = str(entry.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def openai_resolve_callback_datetime(
    *,
    timezone_name: str = "America/New_York",
    now: Optional[datetime] = None,
    raw_text: str = "",
    transcript: Any = None,
) -> Tuple[Optional[datetime], str, str]:
    """
    Ask OpenAI for a concrete local callback datetime.
    Returns (aware_utc_datetime_or_None, raw_time_text, error_message).
    """
    key = _openai_key()
    if not key:
        return None, "", "OPENAI_API_KEY is not configured."

    try:
        tz = _resolve_timezone(timezone_name)
    except Exception:
        tz = _resolve_timezone("America/New_York")

    local_now = (now or timezone.now()).astimezone(tz)
    transcript_text = _transcript_to_lines(transcript)
    spoken = (raw_text or "").strip()
    if not spoken and not transcript_text:
        return None, "", "No callback text or transcript provided."

    system = (
        "You extract PATIENT callback scheduling requests from healthcare outbound call transcripts.\n"
        "Return ONLY valid JSON with keys:\n"
        "- wants_callback (boolean)\n"
        "- scheduled_at (string|null) local wall time as YYYY-MM-DDTHH:MM:SS with NO timezone suffix\n"
        "- raw_time_text (string) short phrase from the patient if possible\n"
        "- reason (string) brief explanation\n\n"
        "Decision rules:\n"
        "1) wants_callback=true ONLY if the PATIENT asked to be called later / is busy and wants a callback / gave a reschedule time.\n"
        "2) Prefer the PATIENT's words over the agent. Ignore agent promises unless the patient clearly requested a callback.\n"
        "3) If the patient only declines, hangs up, or says no without asking to reschedule, wants_callback=false.\n"
        "4) Convert relative phrases using current_local_time and timezone:\n"
        "   - tomorrow / today / after N days / in N days / in N hours / next Monday / this weekend / next week\n"
        "   - morning=>10:00, afternoon=>14:00, evening=>18:00 when no exact clock is given\n"
        "   - if only a day/date with no clock, use 10:00 local\n"
        "5) scheduled_at MUST be strictly after current_local_time. If the computed time is already past, move it forward appropriately (e.g. next day same clock).\n"
        "6) Handle English and Spanish (and mixed) the same way (e.g. 'mañana', 'en dos días', 'después de mañana').\n"
        "7) If multiple times are mentioned, use the latest clear patient preference.\n"
        "8) If unsure, wants_callback=false and scheduled_at=null. Do not invent a callback."
    )
    user_payload = {
        "timezone": timezone_name,
        "current_local_time": local_now.replace(microsecond=0).isoformat(),
        "callback_time_phrase": spoken,
        "transcript": transcript_text,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
            },
            timeout=45,
        )
    except Exception as exc:
        dialer_logger.exception("OpenAI callback resolve failed: %s", exc)
        return None, spoken, "OpenAI callback resolve request failed."

    if response.status_code >= 400:
        dialer_logger.warning(
            "OpenAI callback resolve status=%s body=%s",
            response.status_code,
            (response.text or "")[:300],
        )
        return None, spoken, f"OpenAI callback resolve failed ({response.status_code})."

    try:
        content = (
            ((response.json().get("choices") or [{}])[0].get("message") or {}).get(
                "content"
            )
            or ""
        ).strip()
        data = json.loads(content)
    except Exception:
        dialer_logger.warning(
            "OpenAI callback resolve bad JSON body=%s",
            (response.text or "")[:300],
        )
        return None, spoken, "OpenAI callback resolve returned invalid JSON."

    if not isinstance(data, dict) or not data.get("wants_callback"):
        return None, spoken, "No callback request detected."

    raw_out = str(data.get("raw_time_text") or spoken or "").strip()
    stamp = str(data.get("scheduled_at") or "").strip()
    if not stamp:
        return None, raw_out, "OpenAI did not return scheduled_at."

    # Accept "YYYY-MM-DDTHH:MM:SS" or with space; ignore trailing Z/offset if present.
    stamp = re.sub(r"(Z|[+-]\d{2}:\d{2})$", "", stamp).strip()
    parsed = None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            parsed = datetime.strptime(stamp, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None, raw_out, f"Could not parse OpenAI scheduled_at: {stamp}"

    when_local = parsed.replace(tzinfo=tz)
    if when_local <= local_now:
        when_local = when_local + timedelta(days=1)
        if when_local <= local_now:
            when_local = (local_now + timedelta(hours=1)).replace(
                second=0, microsecond=0
            )
    return when_local.astimezone(dt_timezone.utc), raw_out or spoken, ""
