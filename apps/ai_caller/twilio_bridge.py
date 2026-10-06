# -*- coding: utf-8 -*-
"""Record the two humans after a warm transfer.

Retell drops its recording when the AI transfers, so this flow uses Twilio:
call the first person, confirm they have a moment, dial the second person
with call recording, then transcribe that audio into humans_transcript.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from xml.sax.saxutils import escape

from django.conf import settings

from apps.ai_caller.retell import _digits, normalize_phone
from apps.ai_caller.transcript_merge import (
    save_humans_transcript_for_call,
    transcribe_mix_with_diarization,
    transcribe_recording_bytes,
    transcribe_recording_detailed,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 5
_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|alright|please|go ahead|i do)\b",
    re.I,
)
_NO_RE = re.compile(
    r"\b(no|nope|not now|busy|later|cannot|can't)\b",
    re.I,
)


def _backend_base() -> str:
    return (getattr(settings, "BACKEND_URL", "") or "").rstrip("/")


def _from_number() -> str:
    return (
        (getattr(settings, "TWILIO_PHONE_NUMBER", "") or "").strip()
        or (getattr(settings, "TOLLFREE_TWILIO_PHONE_NUMBER", "") or "").strip()
    )


def _voice() -> str:
    raw = (getattr(settings, "TWILIO_VOICE", None) or "Polly.Joanna").strip()
    if not raw.startswith("Polly."):
        raw = f"Polly.{raw}"
    return raw


def sessions_dir() -> Path:
    return Path(settings.BASE_DIR) / "apps" / "ai_caller" / "data" / "warm_transfer" / "sessions"


def _session_path(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)[:80]
    return sessions_dir() / f"{safe}.json"


def load_session(session_id: str) -> Dict[str, Any]:
    path = _session_path(session_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_pending_session() -> Dict[str, Any]:
    pending = sessions_dir() / "pending.json"
    if not pending.exists():
        return {}
    try:
        data = json.loads(pending.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    session_id = str((data or {}).get("session_id") or "").strip()
    return load_session(session_id) if session_id else {}


def save_session(session: Dict[str, Any]) -> Dict[str, Any]:
    session = dict(session or {})
    session_id = str(session.get("session_id") or "").strip()
    if not session_id:
        return session
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    folder = sessions_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = _session_path(session_id)
    path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
    return session


def _twilio_client():
    from twilio.rest import Client

    sid = (getattr(settings, "TWILIO_ACCOUNT_SID", "") or "").strip()
    token = (getattr(settings, "TWILIO_AUTH_TOKEN", "") or "").strip()
    if not sid or not token:
        return None
    return Client(sid, token)


def _abs_url(path: str, session_id: str, **extra: str) -> str:
    base = _backend_base()
    params = {"session_id": session_id, **{k: v for k, v in extra.items() if v}}
    return f"{base}{path}?{urlencode(params)}"


def _say(text: str) -> str:
    return (
        f'<Say voice="{escape(_voice())}" language="en-US">'
        f"{escape(text)}</Say>"
    )


def answer_twiml(session: Dict[str, Any]) -> str:
    session_id = str(session.get("session_id") or "")
    name = str(session.get("name") or "there")
    service = str(session.get("service_name") or "care")
    gather_url = escape(_abs_url("/api/ai-call/twilio/warm-transfer/gather/", session_id))
    greeting = (
        f"Hi {name}, this is a Bay Area Community Health care coordinator. "
        f"I have a team member ready to speak with you about {service}. "
        f"Do you have a moment?"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say(greeting)}"
        f'<Gather input="speech dtmf" timeout="6" speechTimeout="auto" '
        f'numDigits="1" action="{gather_url}" method="POST">'
        f"{_say('Please say yes if you have a moment, or no if you would like us to call back.')}"
        "</Gather>"
        f"{_say('Sorry, I did not catch that. We will try again another time. Goodbye.')}"
        "<Hangup/>"
        "</Response>"
    )


def declined_twiml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say('No problem. We will call back later. Goodbye.')}"
        "<Hangup/>"
        "</Response>"
    )


def conference_name(session_id: str) -> str:
    return f"care-wt-{session_id}"[:40]


def join_conference_twiml(session: Dict[str, Any], *, is_host: bool) -> str:
    session_id = str(session.get("session_id") or "")
    conf = escape(conference_name(session_id))
    status_cb = escape(
        _abs_url("/api/ai-call/twilio/warm-transfer/conference-status/", session_id)
    )
    mix_cb = escape(
        _abs_url(
            "/api/ai-call/twilio/warm-transfer/recording/",
            session_id,
            leg="mix",
        )
    )
    end_on_exit = "true" if is_host else "false"
    record_attrs = ""
    if is_host:
        record_attrs = (
            f'record="record-from-start" '
            f'recordingStatusCallback="{mix_cb}" '
            f'recordingStatusCallbackEvent="completed" '
            f'recordingStatusCallbackMethod="POST" '
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Dial>"
        f'<Conference beep="false" waitUrl="" startConferenceOnEnter="true" '
        f'endConferenceOnExit="{end_on_exit}" {record_attrs}'
        f'statusCallback="{status_cb}" statusCallbackEvent="start join end" '
        f'statusCallbackMethod="POST">{conf}</Conference>'
        "</Dial>"
        "</Response>"
    )


def connect_twiml(session: Dict[str, Any], *, speak_intro: bool = True) -> str:
    intro = (
        _say("Great, I will connect you with a coordinator now. Please hold.")
        if speak_intro
        else ""
    )
    body = join_conference_twiml(session, is_host=True)
    if not intro:
        return body
    return body.replace("<Response>", f"<Response>{intro}", 1)


def inbound_connect_twiml(session: Dict[str, Any]) -> str:
    """Person A already spoke with Retell; join the recorded conference."""
    return join_conference_twiml(session, is_host=True)


def participant_join_twiml(session: Dict[str, Any]) -> str:
    """Person B joins the same conference."""
    return join_conference_twiml(session, is_host=False)


def speak_now_twiml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say('You can speak now.')}"
        "</Response>"
    )


def start_coordinator_leg(session: Dict[str, Any]) -> None:
    """Dial person B into the conference after A has joined."""
    session = dict(session or {})
    session_id = str(session.get("session_id") or "")
    if not session_id or session.get("b_dialed"):
        return
    client = _twilio_client()
    destination = str(session.get("transfer_number") or "").strip()
    from_number = _from_number()
    if client is None or not destination or not from_number:
        return
    session["b_dialed"] = True
    save_session(session)
    try:
        call = client.calls.create(
            to=destination,
            from_=from_number,
            url=_abs_url("/api/ai-call/twilio/warm-transfer/join/", session_id),
            method="POST",
            timeout=30,
            status_callback=_abs_url(
                "/api/ai-call/twilio/warm-transfer/status/", session_id
            ),
            status_callback_event=["answered", "completed"],
            status_callback_method="POST",
        )
        session["dial_call_sid"] = str(getattr(call, "sid", "") or "")
        session["status"] = "dialing_coordinator"
        save_session(session)
        logger.info(
            "Dialed coordinator session=%s number=%s sid=%s index=%s",
            session_id,
            destination,
            session["dial_call_sid"],
            session.get("transfer_number_index", 0),
        )
        dialer_logger = logging.getLogger("ai_caller.dialer")
        dialer_logger.info(
            "AGENT_DIAL session=%s number=%s index=%s sid=%s",
            session_id,
            destination,
            session.get("transfer_number_index", 0),
            session.get("dial_call_sid"),
        )
    except Exception:
        session["b_dialed"] = False
        save_session(session)
        logger.exception("Failed to dial coordinator session=%s", session_id)
        # Try next agent if this dial could not be created.
        failover_to_next_live_agent(session, reason="dial_create_failed")


_AGENT_FAIL_STATUSES = frozenset(
    {"busy", "failed", "no-answer", "canceled", "cancelled"}
)


def failover_to_next_live_agent(session: Dict[str, Any], *, reason: str = "") -> bool:
    """
    If the current live agent did not answer, dial the next active number.
    Returns True when another dial was started.
    """
    session = load_session(str((session or {}).get("session_id") or "")) or dict(
        session or {}
    )
    session_id = str(session.get("session_id") or "")
    numbers = [
        str(n).strip()
        for n in (session.get("transfer_numbers") or session.get("live_agent_numbers") or [])
        if str(n).strip()
    ]
    if len(numbers) < 2:
        return False

    try:
        idx = int(session.get("transfer_number_index") or 0)
    except (TypeError, ValueError):
        idx = 0
    next_idx = idx + 1
    if next_idx >= len(numbers):
        dialer_logger = logging.getLogger("ai_caller.dialer")
        dialer_logger.info(
            "AGENT_FAILOVER_EXHAUSTED session=%s reason=%s tried=%s",
            session_id,
            reason or "unknown",
            len(numbers),
        )
        session["status"] = "agent_unreachable"
        save_session(session)
        return False

    previous = numbers[idx] if idx < len(numbers) else session.get("transfer_number")
    nxt = numbers[next_idx]
    attempts = list(session.get("failover_attempts") or [])
    attempts.append(
        {
            "from": previous,
            "to": nxt,
            "reason": reason or "no_answer",
            "index": next_idx,
        }
    )
    session["failover_attempts"] = attempts
    session["transfer_number_index"] = next_idx
    session["transfer_number"] = nxt
    session["live_agent_number"] = nxt
    session["b_dialed"] = False
    session["dial_call_sid"] = ""
    session["provider_recording_started"] = False
    session["status"] = "failover_dialing"
    save_session(session)

    dialer_logger = logging.getLogger("ai_caller.dialer")
    dialer_logger.info(
        "AGENT_FAILOVER session=%s reason=%s from=%s to=%s index=%s",
        session_id,
        reason or "no_answer",
        previous,
        nxt,
        next_idx,
    )
    logger.info(
        "Failing over live agent session=%s %s -> %s (%s)",
        session_id,
        previous,
        nxt,
        reason,
    )

    # Keep Call.transfer_number in sync with the number we are trying.
    retell_call_id = str(session.get("retell_call_id") or "").strip()
    if retell_call_id:
        try:
            from apps.ai_caller.models import Call

            Call.objects.filter(retell_call_id=retell_call_id).update(
                transfer_number=nxt
            )
        except Exception:
            logger.exception(
                "Failed to update Call.transfer_number on failover session=%s",
                session_id,
            )

    start_coordinator_leg(session)
    return True


def maybe_failover_on_agent_status(
    session: Dict[str, Any], *, call_status: str, call_sid: str = ""
) -> bool:
    """Failover when the outbound agent leg fails to connect."""
    status = (call_status or "").strip().lower()
    if status not in _AGENT_FAIL_STATUSES:
        return False
    session = load_session(str((session or {}).get("session_id") or "")) or dict(
        session or {}
    )
    dial_sid = str(session.get("dial_call_sid") or "").strip()
    call_sid = (call_sid or "").strip()
    # Only react to the agent (dial) leg, not the patient inbound leg.
    if not dial_sid:
        return False
    if call_sid and call_sid != dial_sid:
        return False
    if session.get("spoke_now") or session.get("status") == "connected":
        return False
    return failover_to_next_live_agent(session, reason=status)


def start_leg_recording(session: Dict[str, Any], call_sid: str, leg: str) -> None:
    """Record only that person's mic (inbound track) for speaker-split transcripts."""
    session_id = str((session or {}).get("session_id") or "")
    call_sid = (call_sid or "").strip()
    leg = (leg or "").strip().lower()
    if not session_id or not call_sid or leg not in {"patient", "provider"}:
        return
    session = load_session(session_id) or dict(session or {})
    if session.get(f"{leg}_recording_started"):
        return
    client = _twilio_client()
    if client is None:
        return
    session[f"{leg}_recording_started"] = True
    # Wall-clock anchor so patient/provider segment times can be interleaved.
    session[f"{leg}_recording_started_at"] = time.time()
    save_session(session)
    try:
        rec = client.calls(call_sid).recordings.create(
            recording_channels="mono",
            recording_track="inbound",
            recording_status_callback=_abs_url(
                "/api/ai-call/twilio/warm-transfer/recording/",
                session_id,
                leg=leg,
            ),
            recording_status_callback_event=["completed"],
            recording_status_callback_method="POST",
        )
        session[f"{leg}_twilio_recording_sid"] = str(getattr(rec, "sid", "") or "")
        save_session(session)
        logger.info(
            "Started %s inbound recording session=%s call=%s rec=%s",
            leg,
            session_id,
            call_sid,
            session.get(f"{leg}_twilio_recording_sid"),
        )
    except Exception:
        session[f"{leg}_recording_started"] = False
        save_session(session)
        logger.exception(
            "Failed to start %s recording session=%s call=%s",
            leg,
            session_id,
            call_sid,
        )


def start_leg_recording_for_call(session: Dict[str, Any], call_sid: str) -> None:
    call_sid = (call_sid or "").strip()
    if not call_sid:
        return
    session = load_session(str(session.get("session_id") or "")) or dict(session or {})
    inbound_sid = str(session.get("call_sid") or "").strip()
    dial_sid = str(session.get("dial_call_sid") or "").strip()
    inbound_speaker = str(session.get("inbound_speaker") or "patient").strip() or "patient"
    dial_speaker = str(session.get("dial_speaker") or "provider").strip() or "provider"
    if dial_sid and call_sid == dial_sid:
        start_leg_recording(session, call_sid, dial_speaker)
    elif inbound_sid and call_sid == inbound_sid:
        start_leg_recording(session, call_sid, inbound_speaker)


def announce_you_can_speak_now(session: Dict[str, Any], conference_sid: str) -> None:
    session = dict(session or {})
    if session.get("spoke_now") or not conference_sid:
        return
    client = _twilio_client()
    if client is None:
        return
    session_id = str(session.get("session_id") or "")
    try:
        client.conferences(conference_sid).update(
            announce_url=_abs_url(
                "/api/ai-call/twilio/warm-transfer/speak-now/", session_id
            ),
            announce_method="POST",
        )
        session["spoke_now"] = True
        session["conference_sid"] = conference_sid
        save_session(session)
        logger.info(
            "Announced You can speak now session=%s conf=%s",
            session_id,
            conference_sid,
        )
    except Exception:
        logger.exception("Failed to announce connected session=%s", session_id)


def handle_conference_status(session: Dict[str, Any], payload: Dict[str, str]) -> None:
    session = dict(session or {})
    event = (
        payload.get("StatusCallbackEvent") or payload.get("EventName") or ""
    ).strip().lower()
    conference_sid = (payload.get("ConferenceSid") or "").strip()
    join_call = (payload.get("CallSid") or "").strip()
    if conference_sid:
        session["conference_sid"] = conference_sid
        save_session(session)
    if event == "join" and join_call:
        start_leg_recording_for_call(session, join_call)
        session = load_session(str(session.get("session_id") or "")) or session
    if event != "join" or session.get("spoke_now") or not conference_sid:
        return
    count = 0
    try:
        count = int(payload.get("ParticipantCount") or 0)
    except (TypeError, ValueError):
        count = 0
    if count < 2:
        client = _twilio_client()
        if client is None:
            return
        try:
            parts = client.conferences(conference_sid).participants.list(limit=10)
            count = len(list(parts))
        except Exception:
            logger.exception("Failed to count conference participants")
            return
    if count >= 2:
        announce_you_can_speak_now(session, conference_sid)


def whisper_twiml(session: Dict[str, Any]) -> str:
    name = str(session.get("name") or "the caller")
    service = str(session.get("service_name") or "care")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say('Hi, this is a Bay Area Community Health care coordinator. I am connecting ' + name + ' about ' + service + '. I will introduce you both, then leave.')}"
        "</Response>"
    )


def dial_status_twiml(dial_status: str) -> str:
    status = (dial_status or "").strip().lower()
    if status in {"completed", "answered"}:
        xml = "<Hangup/>"
    else:
        xml = (
            f"{_say('I am sorry, I could not reach the coordinator. Goodbye.')}"
            "<Hangup/>"
        )
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{xml}</Response>'


def is_affirmative(speech: str, digits: str) -> bool:
    if (digits or "").strip() in {"1"}:
        return True
    return bool(_YES_RE.search(speech or ""))


def is_negative(speech: str, digits: str) -> bool:
    if (digits or "").strip() in {"2"}:
        return True
    return bool(_NO_RE.search(speech or ""))


def _download_twilio_recording(recording_url: str) -> bytes:
    url = (recording_url or "").strip()
    if not url:
        return b""
    if not url.lower().endswith((".wav", ".mp3", ".ogg", ".m4a")):
        url = f"{url}.wav"
    sid = (getattr(settings, "TWILIO_ACCOUNT_SID", "") or "").strip()
    token = (getattr(settings, "TWILIO_AUTH_TOKEN", "") or "").strip()
    if not sid or not token:
        return b""
    import requests

    try:
        resp = requests.get(url, auth=(sid, token), timeout=90)
        resp.raise_for_status()
    except Exception:
        logger.exception("Failed to download Twilio warm-transfer recording")
        return b""
    return resp.content or b""


def _combined_speaker_transcript(patient: str, provider: str) -> str:
    parts = []
    if (patient or "").strip():
        parts.append(f"Patient: {patient.strip()}")
    if (provider or "").strip():
        parts.append(f"Provider: {provider.strip()}")
    return "\n\n".join(parts)


def _recording_offset(session: Dict[str, Any], leg: str) -> float:
    """Seconds since epoch when that leg's inbound recording started."""
    raw = session.get(f"{leg}_recording_started_at")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _persist_to_db(session: Dict[str, Any]) -> None:
    """Write humans STT into Call and merge with AI transcript."""
    session = dict(session or {})
    retell_call_id = str(session.get("retell_call_id") or "").strip()
    if not retell_call_id:
        return
    patient = str(session.get("patient_transcript") or "")
    provider = str(session.get("provider_transcript") or "")
    mix = ""
    mix_items = session.get("mix_diarized_items") or None
    if not patient and not provider and not mix_items:
        mix = str(session.get("humans_transcript") or "")
    recording_url = (
        str(session.get("patient_recording_url") or "")
        or str(session.get("provider_recording_url") or "")
        or str(session.get("mix_recording_url") or "")
    )
    patient_offset = _recording_offset(session, "patient")
    provider_offset = _recording_offset(session, "provider")
    # If one leg has no wall-clock, keep relative-only merge (both 0).
    if not patient_offset or not provider_offset:
        patient_offset = 0.0
        provider_offset = 0.0
    try:
        save_humans_transcript_for_call(
            retell_call_id=retell_call_id,
            patient_text=patient,
            provider_text=provider,
            mix_text=mix,
            recording_url=recording_url,
            session_id=str(session.get("session_id") or ""),
            patient_segments=session.get("patient_segments") or [],
            provider_segments=session.get("provider_segments") or [],
            patient_offset=patient_offset,
            provider_offset=provider_offset,
            mix_items=mix_items if isinstance(mix_items, list) else None,
        )
    except Exception:
        logger.exception(
            "Failed to persist humans transcript to DB retell_call_id=%s",
            retell_call_id,
        )


def _persist_mix_transcript(
    session: Dict[str, Any],
    *,
    recording_url: str,
    recording_sid: str = "",
) -> Dict[str, Any]:
    """Fallback when per-speaker inbound recordings never appear."""
    session = dict(session or {})
    session_id = str(session.get("session_id") or "")
    if session.get("patient_transcript") or session.get("provider_transcript"):
        return session
    if session.get("mix_saved"):
        return session
    time.sleep(4)
    audio = _download_twilio_recording(recording_url)
    diarized_items = (
        transcribe_mix_with_diarization(audio, filename="mix.wav") if audio else []
    )
    text = ""
    if diarized_items:
        session["mix_diarized_items"] = diarized_items
        text = " ".join(
            str(item.get("text") or "").strip()
            for item in diarized_items
            if str(item.get("text") or "").strip()
        )
    elif audio:
        text = transcribe_recording_bytes(audio, filename="mix.wav")
    session["humans_transcript"] = text
    session["mix_recording_url"] = recording_url
    session["mix_recording_sid"] = recording_sid
    session["mix_saved"] = True
    session["humans_saved"] = bool(diarized_items or (text or "").strip())
    session["call_status"] = "ended"
    save_session(session)
    if session["humans_saved"]:
        _persist_to_db(session)
        restore_inbound_voice_url(session)
    logger.info(
        "Saved mixed humans transcript session=%s chars=%s diarized_turns=%s",
        session_id,
        len(text or ""),
        len(diarized_items or []),
    )
    return session


def persist_humans_transcript(
    session: Dict[str, Any],
    *,
    recording_url: str,
    recording_sid: str = "",
    leg: str = "",
) -> Dict[str, Any]:
    session = dict(session or {})
    session_id = str(session.get("session_id") or "")
    speaker = (leg or "").strip().lower()
    session = load_session(session_id) or session
    if speaker == "mix":
        return _persist_mix_transcript(
            session,
            recording_url=recording_url,
            recording_sid=recording_sid,
        )
    if speaker not in {"patient", "provider"}:
        logger.info("Skipping unlabeled warm-transfer recording session=%s", session_id)
        return session
    if session.get(f"{speaker}_stt_started") or session.get(f"{speaker}_transcript"):
        patient = str(session.get("patient_transcript") or "")
        provider = str(session.get("provider_transcript") or "")
        if patient or provider:
            session["humans_saved"] = bool(patient and provider)
            save_session(session)
            _persist_to_db(session)
            if patient and provider:
                restore_inbound_voice_url(session)
        return session
    session[f"{speaker}_stt_started"] = True
    save_session(session)
    time.sleep(4)
    audio = _download_twilio_recording(recording_url)
    detailed = (
        transcribe_recording_detailed(audio, filename=f"{speaker}.wav")
        if audio
        else {"text": "", "segments": []}
    )
    text = str(detailed.get("text") or "")
    segments = detailed.get("segments") or []
    session[f"{speaker}_transcript"] = text
    session[f"{speaker}_segments"] = segments
    session[f"{speaker}_recording_url"] = recording_url
    session[f"{speaker}_recording_sid"] = recording_sid
    session[f"{speaker}_saved"] = True
    patient = str(session.get("patient_transcript") or "")
    provider = str(session.get("provider_transcript") or "")
    session["humans_transcript"] = _combined_speaker_transcript(patient, provider)
    both_ready = bool(patient and provider)
    # Persist as soon as any leg has text; refresh again when the second arrives.
    session["humans_saved"] = both_ready
    session["call_status"] = "ended" if both_ready else (
        session.get("call_status") or "ended"
    )
    save_session(session)
    if patient or provider:
        _persist_to_db(session)
    if both_ready:
        restore_inbound_voice_url(session)
    logger.info(
        "Saved %s transcript session=%s chars=%s segments=%s both=%s",
        speaker,
        session_id,
        len(text),
        len(segments),
        both_ready,
    )
    return session


def _poll_twilio_recording(session_id: str) -> None:
    deadline = time.time() + 25 * 60
    client = _twilio_client()
    if not client:
        return
    while time.time() < deadline:
        session = load_session(session_id)
        if not session:
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue
        if session.get("patient_transcript") and session.get("provider_transcript"):
            return
        call_sid = str(session.get("call_sid") or "").strip()
        dial_sid = str(session.get("dial_call_sid") or "").strip()
        conference_sid = str(session.get("conference_sid") or "").strip()
        if not call_sid and not conference_sid:
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue
        recordings = []
        try:
            if call_sid:
                recordings.extend(list(client.recordings.list(call_sid=call_sid, limit=10)))
            if dial_sid:
                recordings.extend(list(client.recordings.list(call_sid=dial_sid, limit=10)))
            if conference_sid:
                recordings.extend(
                    list(client.conferences(conference_sid).recordings.list(limit=10))
                )
        except Exception:
            logger.exception("Twilio recording list failed call_sid=%s", call_sid)
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue
        for rec in recordings:
            rec_status = str(getattr(rec, "status", "") or "").lower()
            duration = int(getattr(rec, "duration", 0) or 0)
            if rec_status != "completed" or duration < 3:
                continue
            uri = str(getattr(rec, "uri", "") or "").split(".json")[0]
            account = (getattr(settings, "TWILIO_ACCOUNT_SID", "") or "").strip()
            recording_url = (
                str(getattr(rec, "media_url", "") or "").strip()
                or f"https://api.twilio.com{uri}"
            )
            if not recording_url.startswith("http"):
                recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{account}/Recordings/{rec.sid}"
            rec_call = str(getattr(rec, "call_sid", "") or "")
            if rec_call == dial_sid:
                rec_leg = "provider"
            elif rec_call == call_sid:
                rec_leg = "patient"
            else:
                rec_leg = "mix"
            persist_humans_transcript(
                session,
                recording_url=recording_url,
                recording_sid=str(getattr(rec, "sid", "") or ""),
                leg=rec_leg,
            )
            session = load_session(session_id)
            if session.get("patient_transcript") and session.get("provider_transcript"):
                return
        time.sleep(_POLL_INTERVAL_SECONDS)
    logger.warning("Timed out waiting for two-human recording session=%s", session_id)


def _start_recording_poller(session_id: str) -> None:
    thread = threading.Thread(
        target=_poll_twilio_recording,
        args=(session_id,),
        name=f"care-wt-rec-{session_id[:8]}",
        daemon=True,
    )
    thread.start()


def _incoming_number(client, phone: str):
    want = _digits(phone)
    if not want:
        return None
    try:
        matches = client.incoming_phone_numbers.list(phone_number=phone, limit=5)
        if matches:
            return matches[0]
    except Exception:
        logger.exception("Twilio incoming-number lookup failed")
    try:
        for item in client.incoming_phone_numbers.list(limit=50):
            if _digits(getattr(item, "phone_number", "") or "") == want:
                return item
    except Exception:
        logger.exception("Twilio incoming-number list failed")
    return None


def restore_inbound_voice_url(session: Dict[str, Any]) -> None:
    session = dict(session or {})
    if session.get("voice_url_restored"):
        return
    sid = str(session.get("incoming_sid") or "").strip()
    if not sid:
        return
    client = _twilio_client()
    if client is None:
        return
    previous = session.get("previous_voice_url")
    method = session.get("previous_voice_method") or "POST"
    try:
        kwargs: Dict[str, Any] = {"voice_method": method}
        if previous is not None:
            kwargs["voice_url"] = previous
        client.incoming_phone_numbers(sid).update(**kwargs)
        session["voice_url_restored"] = True
        save_session(session)
        logger.info("Restored Twilio voice URL sid=%s", sid)
    except Exception:
        logger.exception("Failed to restore Twilio voice URL sid=%s", sid)


def _schedule_voice_url_restore(session_id: str) -> None:
    def _run() -> None:
        time.sleep(40 * 60)
        restore_inbound_voice_url(load_session(session_id))

    threading.Thread(
        target=_run,
        name=f"care-wt-restore-{session_id[:8]}",
        daemon=True,
    ).start()


def attach_retell_call(session_id: str, retell_call_id: str) -> None:
    session = load_session(session_id)
    if not session:
        return
    session["retell_call_id"] = retell_call_id
    session["status"] = "retell_placed"
    save_session(session)
    _start_recording_poller(session_id)


def prepare_retell_bridge(
    *,
    phone_number: str,
    name: str,
    transfer_number: str,
    service_name: str,
    extra: Optional[Dict[str, Any]] = None,
    transfer_numbers: Optional[list] = None,
) -> Dict[str, Any]:
    """Point our Twilio number at inbound TwiML so Retell can transfer A to us."""
    bridge_number = _from_number()
    if not bridge_number:
        return {
            "ok": False,
            "error": "Twilio from-number is not configured.",
            "status_code": 503,
        }
    if not _backend_base():
        return {
            "ok": False,
            "error": "BACKEND_URL is not configured. Twilio needs a public URL for TwiML.",
            "status_code": 503,
        }
    client = _twilio_client()
    if client is None:
        return {
            "ok": False,
            "error": "Twilio is not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN.",
            "status_code": 503,
        }
    incoming = _incoming_number(client, bridge_number)
    if incoming is None:
        return {
            "ok": False,
            "error": "Twilio number was not found in this account, so Retell cannot transfer into the recorder.",
            "status_code": 503,
        }

    session_id = uuid.uuid4().hex[:16]
    inbound_url = _abs_url("/api/ai-call/twilio/warm-transfer/inbound/", session_id)
    previous_url = getattr(incoming, "voice_url", None) or ""
    previous_method = getattr(incoming, "voice_method", None) or "POST"
    try:
        incoming.update(voice_url=inbound_url, voice_method="POST")
    except Exception as exc:
        logger.exception("Failed to point Twilio number at warm-transfer inbound")
        return {
            "ok": False,
            "error": f"Failed to prepare Twilio recorder: {exc}",
            "status_code": 502,
        }

    numbers = [
        str(n).strip()
        for n in (transfer_numbers or ([transfer_number] if transfer_number else []) or [])
        if str(n).strip()
    ]
    primary = numbers[0] if numbers else str(transfer_number or "").strip()

    session_data: Dict[str, Any] = {
        "session_id": session_id,
        "name": name,
        "service_name": service_name,
        "phone_number": normalize_phone(phone_number),
        "transfer_number": primary,
        "transfer_numbers": numbers,
        "transfer_number_index": 0,
        "bridge_number": bridge_number,
        "status": "awaiting_retell_transfer",
        "humans_saved": False,
        "incoming_sid": incoming.sid,
        "previous_voice_url": previous_url,
        "previous_voice_method": previous_method,
    }
    if extra:
        session_data.update(extra)
        # Keep ordered list authoritative if caller also passed live_agent_numbers.
        extra_numbers = [
            str(n).strip()
            for n in (extra.get("live_agent_numbers") or extra.get("transfer_numbers") or [])
            if str(n).strip()
        ]
        if extra_numbers:
            session_data["transfer_numbers"] = extra_numbers
            session_data["transfer_number"] = extra_numbers[0]
            session_data["transfer_number_index"] = 0
    session = save_session(session_data)
    (sessions_dir() / "pending.json").write_text(
        json.dumps({"session_id": session_id}, indent=2) + "\n",
        encoding="utf-8",
    )
    _schedule_voice_url_restore(session_id)
    logger.info(
        "Retell bridge ready session=%s inbound=%s agents=%s",
        session_id,
        inbound_url,
        session_data.get("transfer_numbers"),
    )
    return {
        "ok": True,
        "session_id": session_id,
        "session": session,
        "bridge_number": bridge_number,
    }
