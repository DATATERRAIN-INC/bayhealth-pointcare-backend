# -*- coding: utf-8 -*-
"""Record the two humans after a warm transfer.

Retell drops its recording when the AI transfers, so this flow uses Twilio:
call the first person, confirm they have a moment, dial the second person
with call recording, then transcribe that audio into humans_transcript.
"""

from __future__ import annotations

import audioop
import io
import logging
import re
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode
from xml.sax.saxutils import escape

from django.conf import settings
from django.core.cache import caches

from apps.ai_caller.retell import _digits, normalize_phone
from apps.ai_caller.transcript_merge import (
    save_humans_transcript_for_call,
    transcribe_mix_with_diarization,
    transcribe_recording_detailed,
)

logger = logging.getLogger(__name__)
wt_logger = logging.getLogger("ai_caller.warm_transfer")

_SESSION_TTL_SECONDS = 60 * 60 * 24
_PENDING_CACHE_KEY = "warm_transfer:pending"

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


def _session_cache():
    """Prefer Redis; fall back to shared file cache (never LocMem)."""
    try:
        cache = caches["warm_transfer"]
        cache.get("_wt_ping")
        return cache
    except Exception:
        wt_logger.warning(
            "TRANSFER_SESSION_CACHE_FALLBACK using file cache (redis unavailable)"
        )
        return caches["warm_transfer_file"]


def _session_cache_key(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", (session_id or "").strip())[:80]
    return f"warm_transfer:session:{safe}"


def load_session(session_id: str) -> Dict[str, Any]:
    session_id = (session_id or "").strip()
    if not session_id:
        return {}
    try:
        data = _session_cache().get(_session_cache_key(session_id))
    except Exception:
        data = caches["warm_transfer_file"].get(_session_cache_key(session_id))
    return data if isinstance(data, dict) else {}


def load_pending_session() -> Dict[str, Any]:
    try:
        pending = _session_cache().get(_PENDING_CACHE_KEY)
    except Exception:
        pending = caches["warm_transfer_file"].get(_PENDING_CACHE_KEY)
    if not isinstance(pending, dict):
        return {}
    session_id = str(pending.get("session_id") or "").strip()
    return load_session(session_id) if session_id else {}


def save_session(session: Dict[str, Any]) -> Dict[str, Any]:
    """Persist warm-transfer state in Redis (or shared file cache)."""
    session = dict(session or {})
    session_id = str(session.get("session_id") or "").strip()
    if not session_id:
        return session
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    cache = _session_cache()
    try:
        cache.set(_session_cache_key(session_id), session, timeout=_SESSION_TTL_SECONDS)
        cache.set(
            _PENDING_CACHE_KEY,
            {"session_id": session_id},
            timeout=_SESSION_TTL_SECONDS,
        )
    except Exception:
        fallback = caches["warm_transfer_file"]
        fallback.set(
            _session_cache_key(session_id), session, timeout=_SESSION_TTL_SECONDS
        )
        fallback.set(
            _PENDING_CACHE_KEY,
            {"session_id": session_id},
            timeout=_SESSION_TTL_SECONDS,
        )
        wt_logger.warning(
            "TRANSFER_SESSION_CACHE_FALLBACK session=%s (file cache)",
            session_id,
        )
    return session


def split_stereo_wav(audio: bytes) -> Tuple[bytes, bytes]:
    """
    Split Twilio dual-channel WAV into (left/inbound, right/outbound) mono WAVs.

    On the patient call leg: left = patient mic, right = what patient hears
    (live agent in conference).
    """
    if not audio:
        return b"", b""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except Exception:
        logger.exception("Failed to parse WAV for stereo split")
        return audio, b""

    if channels < 2:
        return audio, b""

    try:
        left = audioop.tomono(frames, sampwidth, 1, 0)
        right = audioop.tomono(frames, sampwidth, 0, 1)
    except Exception:
        logger.exception("Failed to split stereo channels")
        return audio, b""

    def _mono_wav(mono_frames: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(sampwidth)
            out.setframerate(framerate)
            out.writeframes(mono_frames)
        return buf.getvalue()

    return _mono_wav(left), _mono_wav(right)


def patch_session(session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge fields into the latest cached session.

    Avoids lost updates when patient + provider STT finish in parallel
    (each process must not overwrite the other leg's transcript).
    """
    session_id = (session_id or "").strip()
    if not session_id:
        return dict(updates or {})
    latest = load_session(session_id) or {"session_id": session_id}
    latest.update(dict(updates or {}))
    latest["session_id"] = session_id
    return save_session(latest)


def _mix_items_are_split(mix_items: Any) -> bool:
    """True when diarized mix has both patient and live_agent turns."""
    if not isinstance(mix_items, list) or len(mix_items) < 2:
        return False
    speakers = {
        str(i.get("speaker") or "")
        for i in mix_items
        if isinstance(i, dict) and str(i.get("text") or "").strip()
    }
    return {"patient", "live_agent"}.issubset(speakers)


def _has_speaker_split(session: Dict[str, Any]) -> bool:
    """True only for a real channel/diarized split — not mono leg text.

    Patient inbound mono usually contains both voices; treating plain
    patient+provider STT as a split caused mix skip and fused walls.
    """
    s = session or {}
    if s.get("dual_channel_split") or s.get("diarized_from_patient_mono"):
        return True
    return _mix_items_are_split(s.get("mix_diarized_items"))


def extend_call_ended_at(
    session: Dict[str, Any],
    *,
    ended_at: Optional[datetime] = None,
) -> None:
    """Push Call.ended_at forward for full wall-clock duration.

    Retell often ends when the AI transfers; the patient + live agent keep
    talking on Twilio. Duration must be Retell start → last Twilio hangup
    (not Retell-only, and not a naive sum of both legs).
    """
    from django.utils import timezone as dj_tz

    from apps.ai_caller.models import Call

    retell_call_id = str((session or {}).get("retell_call_id") or "").strip()
    if not retell_call_id:
        return
    call = Call.objects.filter(retell_call_id=retell_call_id).first()
    if not call:
        return
    end = ended_at or dj_tz.now()
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if call.ended_at and call.ended_at >= end:
        return
    call.ended_at = end
    call.save(update_fields=["ended_at", "updated_at"])
    wt_logger.info(
        "TRANSFER_DURATION_EXTEND retell_call_id=%s ended_at=%s duration_seconds=%s",
        retell_call_id,
        end.isoformat(),
        call.duration_seconds,
    )


def _sync_call_twilio_ids(
    session: Dict[str, Any],
    *,
    recording_url: str = "",
    live_agent_recording_url: str = "",
) -> None:
    """Store Twilio CA SID + patient/live-agent recording URLs on Call."""
    from apps.ai_caller.models import Call

    retell_call_id = str((session or {}).get("retell_call_id") or "").strip()
    if not retell_call_id:
        return
    call = Call.objects.filter(retell_call_id=retell_call_id).first()
    if not call:
        return

    twilio_call_sid = str(
        (session or {}).get("call_sid")
        or (session or {}).get("dial_call_sid")
        or ""
    ).strip()
    patient_url = (
        recording_url
        or str((session or {}).get("patient_recording_s3_url") or "")
        or str((session or {}).get("patient_recording_url") or "")
    ).strip()
    agent_url = (
        live_agent_recording_url
        or str((session or {}).get("provider_recording_s3_url") or "")
        or str((session or {}).get("provider_recording_url") or "")
    ).strip()
    update_fields = []
    if twilio_call_sid and call.warm_transfer_session_id != twilio_call_sid:
        call.warm_transfer_session_id = twilio_call_sid[:64]
        update_fields.append("warm_transfer_session_id")
    if patient_url and patient_url != (call.recording_url or ""):
        call.recording_url = patient_url[:1024]
        update_fields.append("recording_url")
    if agent_url and agent_url != (call.live_agent_recording_url or ""):
        call.live_agent_recording_url = agent_url[:1024]
        update_fields.append("live_agent_recording_url")
    if update_fields:
        update_fields.append("updated_at")
        call.save(update_fields=update_fields)
        wt_logger.info(
            "TRANSFER_CALL_IDS retell_call_id=%s twilio_call_sid=%s "
            "patient_recording=%s live_agent_recording=%s",
            retell_call_id,
            twilio_call_sid or "-",
            bool(patient_url),
            bool(agent_url),
        )


def _upload_recording_audio(
    audio: bytes, *, retell_call_id: str, leg: str, recording_sid: str = ""
) -> str:
    """Upload Twilio recording bytes to S3; return public/media URL."""
    if not audio:
        return ""
    try:
        from common.s3 import build_s3_url, upload_bytes

        filename = f"{leg}_{(recording_sid or retell_call_id or 'call')[:32]}.wav"
        key = upload_bytes(
            audio,
            filename=filename,
            folder="warm-transfer-recordings",
            content_type="audio/wav",
        )
        return build_s3_url(key)
    except Exception:
        wt_logger.exception(
            "TRANSFER_RECORDING_UPLOAD_FAILED retell_call_id=%s leg=%s",
            retell_call_id,
            leg,
        )
        return ""


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
    """Start per-leg Twilio recording for speaker-split transcripts.

    Patient: try dual-channel first (left=patient, right=agent audio patient hears),
    then fall back to mono inbound. Provider: inbound mono.

    Must be called once the call is in-conference — starting too early causes
    Twilio error 21220 (not eligible for recording).
    """
    session_id = str((session or {}).get("session_id") or "")
    call_sid = (call_sid or "").strip()
    leg = (leg or "").strip().lower()
    if not session_id or not call_sid or leg not in {"patient", "provider"}:
        return
    session = load_session(session_id) or dict(session or {})
    if session.get(f"{leg}_recording_started") and session.get(
        f"{leg}_twilio_recording_sid"
    ):
        return
    client = _twilio_client()
    if client is None:
        return

    attempts = []
    if leg == "patient":
        attempts.append({"channels": "dual", "track": "both", "dual": True})
    attempts.append({"channels": "mono", "track": "inbound", "dual": False})

    session[f"{leg}_recording_started"] = True
    session[f"{leg}_recording_started_at"] = time.time()
    save_session(session)

    last_err = None
    for attempt in attempts:
        try:
            rec = client.calls(call_sid).recordings.create(
                recording_channels=attempt["channels"],
                recording_track=attempt["track"],
                recording_status_callback=_abs_url(
                    "/api/ai-call/twilio/warm-transfer/recording/",
                    session_id,
                    leg=leg,
                ),
                recording_status_callback_event=["completed"],
                recording_status_callback_method="POST",
            )
            patch_session(
                session_id,
                {
                    f"{leg}_twilio_recording_sid": str(getattr(rec, "sid", "") or ""),
                    f"{leg}_recording_dual": bool(attempt["dual"]),
                    f"{leg}_recording_started": True,
                },
            )
            logger.info(
                "Started %s recording session=%s call=%s rec=%s dual=%s channels=%s",
                leg,
                session_id,
                call_sid,
                getattr(rec, "sid", ""),
                attempt["dual"],
                attempt["channels"],
            )
            return
        except Exception as exc:
            last_err = exc
            err_text = str(exc)
            # 21220 = not eligible yet / unsupported mode — try next attempt.
            if "21220" in err_text or "not eligible" in err_text.lower():
                wt_logger.warning(
                    "TRANSFER_RECORDING_RETRY session=%s leg=%s dual=%s err=%s",
                    session_id,
                    leg,
                    attempt["dual"],
                    err_text[:180],
                )
                continue
            break

    patch_session(
        session_id,
        {
            f"{leg}_recording_started": False,
            f"{leg}_twilio_recording_sid": "",
        },
    )
    wt_logger.error(
        "TRANSFER_RECORDING_FAIL session=%s leg=%s call=%s err=%s",
        session_id,
        leg,
        call_sid,
        (str(last_err) if last_err else "unknown")[:300],
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


def _delayed_start_leg_recording(session_id: str, call_sid: str, leg: str) -> None:
    """Wait briefly so Twilio marks the call in-conference (avoids 21220)."""
    time.sleep(1.5)
    session = load_session(session_id) or {"session_id": session_id}
    start_leg_recording(session, call_sid, leg)


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
    # Conference over → live-agent talk ended; extend full-call duration.
    if event in {"end", "conference-end"}:
        extend_call_ended_at(session)
        return
    if event == "join" and join_call:
        session = load_session(str(session.get("session_id") or "")) or session
        inbound_sid = str(session.get("call_sid") or "").strip()
        dial_sid = str(session.get("dial_call_sid") or "").strip()
        leg = ""
        if dial_sid and join_call == dial_sid:
            leg = str(session.get("dial_speaker") or "provider")
        elif inbound_sid and join_call == inbound_sid:
            leg = str(session.get("inbound_speaker") or "patient")
        if leg:
            threading.Thread(
                target=_delayed_start_leg_recording,
                args=(str(session.get("session_id") or ""), join_call, leg),
                daemon=True,
            ).start()
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
        wt_logger.error(
            "TRANSFER_TRANSCRIPT_FAIL session=%s reason=missing_retell_call_id",
            session.get("session_id"),
        )
        return
    patient = str(session.get("patient_transcript") or "")
    provider = str(session.get("provider_transcript") or "")
    mix = ""
    mix_items = session.get("mix_diarized_items") or None
    # Prefer diarized mix whenever it has both speakers — even if per-leg
    # text exists. Per-leg mono often captures a fused wall and used to
    # overwrite a good mix split (humans=1 after diarized=13).
    patient_segments = session.get("patient_segments") or []
    provider_segments = session.get("provider_segments") or []
    dual_ok = bool(
        session.get("dual_channel_split") or session.get("diarized_from_patient_mono")
    )
    if _mix_items_are_split(mix_items):
        # Diarized mix wins — clear ALL leg inputs (text + segments).
        # Leaving segments set used to make humans_items_from_texts ignore mix.
        patient = ""
        provider = ""
        mix = ""
        patient_segments = []
        provider_segments = []
        wt_logger.info(
            "TRANSFER_PERSIST_PREFER_MIX retell_call_id=%s turns=%s",
            retell_call_id,
            len(mix_items or []),
        )
    elif dual_ok and patient and provider:
        mix = ""
        mix_items = None
    elif patient and provider and not dual_ok:
        # Mono legs only — do not persist a fused interleave; wait for mix.
        wt_logger.info(
            "TRANSFER_PERSIST_WAIT_MIX retell_call_id=%s reason=mono_legs_only",
            retell_call_id,
        )
        return
    elif not patient and not provider:
        mix = str(session.get("humans_transcript") or "")
        mix_items = None
    elif provider and not patient:
        # Provider-only is safe (live agent inbound). Persist it.
        mix = ""
        mix_items = None
        patient_segments = []
    else:
        # Patient-only mono is usually a fused two-person wall — wait for
        # provider + mix diarize instead of flashing humans=1 in the UI.
        wt_logger.info(
            "TRANSFER_PERSIST_WAIT_MIX retell_call_id=%s reason=patient_only_mono",
            retell_call_id,
        )
        return
    # Prefer S3 URLs we uploaded; fall back to Twilio media URLs.
    recording_url = (
        str(session.get("patient_recording_s3_url") or "")
        or str(session.get("patient_recording_url") or "")
        or (
            str(session.get("mix_recording_s3_url") or "")
            if not patient and not provider
            else ""
        )
        or (
            str(session.get("mix_recording_url") or "")
            if not patient and not provider
            else ""
        )
    )
    live_agent_recording_url = (
        str(session.get("provider_recording_s3_url") or "")
        or str(session.get("provider_recording_url") or "")
    )
    twilio_call_sid = str(
        session.get("call_sid") or session.get("dial_call_sid") or ""
    ).strip()
    patient_offset = _recording_offset(session, "patient")
    provider_offset = _recording_offset(session, "provider")
    # If one leg has no wall-clock, keep relative-only merge (both 0).
    if not patient_offset or not provider_offset:
        patient_offset = 0.0
        provider_offset = 0.0
    try:
        call = save_humans_transcript_for_call(
            retell_call_id=retell_call_id,
            patient_text=patient,
            provider_text=provider,
            mix_text=mix,
            recording_url=recording_url,
            live_agent_recording_url=live_agent_recording_url,
            twilio_call_sid=twilio_call_sid,
            patient_segments=patient_segments,
            provider_segments=provider_segments,
            patient_offset=patient_offset,
            provider_offset=provider_offset,
            mix_items=mix_items if isinstance(mix_items, list) else None,
        )
        if call and (call.live_agent_transcript or []):
            wt_logger.info(
                "TRANSFER_TRANSCRIPT_OK retell_call_id=%s twilio_call_sid=%s "
                "humans=%s merged=%s patient_recording=%s live_agent_recording=%s",
                retell_call_id,
                twilio_call_sid or "-",
                len(call.live_agent_transcript or []),
                len(call.transcript or []),
                (call.recording_url or "")[:120],
                (call.live_agent_recording_url or "")[:120],
            )
        else:
            wt_logger.warning(
                "TRANSFER_TRANSCRIPT_EMPTY retell_call_id=%s twilio_call_sid=%s "
                "patient_chars=%s provider_chars=%s",
                retell_call_id,
                twilio_call_sid or "-",
                len(patient),
                len(provider),
            )
    except Exception:
        wt_logger.exception(
            "TRANSFER_TRANSCRIPT_FAIL retell_call_id=%s twilio_call_sid=%s",
            retell_call_id,
            twilio_call_sid or "-",
        )
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
    """Fallback only when per-speaker inbound recordings never appear."""
    session_id = str((session or {}).get("session_id") or "")
    session = load_session(session_id) or dict(session or {})
    if _has_speaker_split(session):
        wt_logger.info(
            "TRANSFER_MIX_SKIP session=%s reason=speaker_split_ready",
            session_id,
        )
        return session
    if session.get("mix_saved") and session.get("mix_diarized_items"):
        return session
    time.sleep(4)
    audio = _download_twilio_recording(recording_url)
    if not audio:
        wt_logger.error(
            "TRANSFER_RECORDING_DOWNLOAD_FAIL session=%s leg=mix url=%s",
            session_id,
            (recording_url or "")[:160],
        )
        return load_session(session_id) or session
    # Re-check after download wait — dual/per-leg STT may have finished.
    session = load_session(session_id) or session
    if _has_speaker_split(session):
        wt_logger.info(
            "TRANSFER_MIX_SKIP session=%s reason=speaker_split_ready_after_wait",
            session_id,
        )
        return session
    s3_url = _upload_recording_audio(
        audio,
        retell_call_id=str(session.get("retell_call_id") or ""),
        leg="mix",
        recording_sid=recording_sid,
    )
    latest_for_stt = load_session(session_id) or session
    provider_anchor = str(latest_for_stt.get("provider_transcript") or "")
    patient_anchor = str(latest_for_stt.get("patient_transcript") or "")
    diarized_items = (
        transcribe_mix_with_diarization(
            audio,
            filename="mix.wav",
            provider_text=provider_anchor,
            patient_text=patient_anchor,
        )
        if audio
        else []
    )
    # Never store an undiarized mixed wall as the humans transcript.
    if not diarized_items:
        patch_session(
            session_id,
            {
                "mix_saved": True,
                "mix_recording_url": recording_url,
                "mix_recording_sid": recording_sid,
                **({"mix_recording_s3_url": s3_url} if s3_url else {}),
            },
        )
        wt_logger.error(
            "TRANSFER_FAIL session=%s leg=mix reason=no_diarization",
            session_id,
        )
        return load_session(session_id) or session

    text = " ".join(
        str(item.get("text") or "").strip()
        for item in diarized_items
        if str(item.get("text") or "").strip()
    )
    session = patch_session(
        session_id,
        {
            "mix_diarized_items": diarized_items,
            "humans_transcript": text,
            "mix_recording_url": recording_url,
            "mix_recording_sid": recording_sid,
            **({"mix_recording_s3_url": s3_url} if s3_url else {}),
            "mix_saved": True,
            "humans_saved": True,
            "db_persisted_complete": True,
            "call_status": "ended",
        },
    )
    _sync_call_twilio_ids(session, recording_url=s3_url or recording_url)
    _persist_to_db(session)
    restore_inbound_voice_url(session)
    wt_logger.info(
        "TRANSFER_SUCCESS session=%s leg=mix chars=%s diarized=%s",
        session_id,
        len(text or ""),
        len(diarized_items or []),
    )
    return session


def _finalize_leg_persist(session_id: str, *, speaker: str = "") -> Dict[str, Any]:
    """Reload both legs from cache and write merged transcript once ready."""
    session = load_session(session_id) or {}
    # Never let per-leg finalize clobber a completed diarized mix split.
    if _mix_items_are_split(session.get("mix_diarized_items")):
        if not session.get("db_persisted_complete"):
            _persist_to_db(session)
            patch_session(session_id, {"db_persisted_complete": True, "humans_saved": True})
        return load_session(session_id) or session
    patient = str(session.get("patient_transcript") or "")
    provider = str(session.get("provider_transcript") or "")
    both_ready = bool(patient and provider)
    if not patient and not provider:
        return session
    if session.get("db_persisted_complete") and both_ready:
        return session

    dual_ok = bool(
        session.get("dual_channel_split") or session.get("diarized_from_patient_mono")
    )
    # Mono patient+provider STT is not a reliable split — wait for mix diarize.
    if both_ready and not dual_ok:
        patient_recording = (
            str(session.get("patient_recording_s3_url") or "")
            or str(session.get("patient_recording_url") or "")
        )
        live_agent_recording = (
            str(session.get("provider_recording_s3_url") or "")
            or str(session.get("provider_recording_url") or "")
        )
        _sync_call_twilio_ids(
            session,
            recording_url=patient_recording,
            live_agent_recording_url=live_agent_recording,
        )
        wt_logger.info(
            "TRANSFER_WAIT_MIX session=%s leg=%s patient_chars=%s provider_chars=%s",
            session_id,
            speaker or "-",
            len(patient),
            len(provider),
        )
        return session

    patient_recording = (
        str(session.get("patient_recording_s3_url") or "")
        or str(session.get("patient_recording_url") or "")
    )
    live_agent_recording = (
        str(session.get("provider_recording_s3_url") or "")
        or str(session.get("provider_recording_url") or "")
    )
    updates: Dict[str, Any] = {
        "humans_transcript": _combined_speaker_transcript(patient, provider),
        "humans_saved": both_ready,
    }
    if both_ready and dual_ok:
        updates["call_status"] = "ended"
        updates["db_persisted_complete"] = True
    session = patch_session(session_id, updates)
    _sync_call_twilio_ids(
        session,
        recording_url=patient_recording,
        live_agent_recording_url=live_agent_recording,
    )
    _persist_to_db(session)
    if both_ready and dual_ok:
        restore_inbound_voice_url(session)
        wt_logger.info(
            "TRANSFER_SUCCESS session=%s leg=%s patient_chars=%s provider_chars=%s both=1 "
            "twilio_call_sid=%s",
            session_id,
            speaker or "-",
            len(patient),
            len(provider),
            str(session.get("call_sid") or session.get("dial_call_sid") or "-"),
        )
    elif session.get("patient_saved") and session.get("provider_saved") and dual_ok:
        wt_logger.warning(
            "TRANSFER_PARTIAL session=%s patient_chars=%s provider_chars=%s",
            session_id,
            len(patient),
            len(provider),
        )
        restore_inbound_voice_url(session)
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
    # Stop webhook/poller re-entry once a real speaker split is persisted.
    if session.get("db_persisted_complete") and (
        _mix_items_are_split(session.get("mix_diarized_items"))
        or (
            session.get("patient_transcript")
            and session.get("provider_transcript")
        )
    ):
        return session
    if speaker == "mix":
        return _persist_mix_transcript(
            session,
            recording_url=recording_url,
            recording_sid=recording_sid,
        )
    if speaker not in {"patient", "provider"}:
        logger.info("Skipping unlabeled warm-transfer recording session=%s", session_id)
        return session

    # Already transcribed this leg — merge from latest cache, do not re-STT.
    if session.get(f"{speaker}_saved") or (
        session.get(f"{speaker}_transcript") and session.get(f"{speaker}_stt_started")
    ):
        if session.get("db_persisted_complete"):
            return session
        return _finalize_leg_persist(session_id, speaker=speaker)

    # Another worker is already STT'ing this leg.
    if session.get(f"{speaker}_stt_started"):
        return session

    patch_session(session_id, {f"{speaker}_stt_started": True})
    time.sleep(4)
    audio = _download_twilio_recording(recording_url)
    if not audio:
        wt_logger.error(
            "TRANSFER_RECORDING_DOWNLOAD_FAIL session=%s leg=%s url=%s",
            session_id,
            speaker,
            (recording_url or "")[:160],
        )
        patch_session(session_id, {f"{speaker}_stt_started": False})
        return load_session(session_id) or session

    retell_call_id = str(
        (load_session(session_id) or session).get("retell_call_id") or ""
    )
    s3_url = _upload_recording_audio(
        audio,
        retell_call_id=retell_call_id,
        leg=speaker,
        recording_sid=recording_sid,
    )

    # Patient dual-channel: left=patient, right=live agent (what patient hears).
    left_audio, right_audio = (b"", b"")
    if speaker == "patient":
        left_audio, right_audio = split_stereo_wav(audio)

    leg_updates: Dict[str, Any] = {
        f"{speaker}_recording_url": recording_url,
        f"{speaker}_recording_sid": recording_sid,
        f"{speaker}_saved": True,
        f"{speaker}_stt_started": True,
    }
    if s3_url:
        leg_updates[f"{speaker}_recording_s3_url"] = s3_url

    if speaker == "patient" and right_audio:
        patient_detailed = transcribe_recording_detailed(
            left_audio or audio, filename="patient.wav"
        )
        provider_detailed = transcribe_recording_detailed(
            right_audio, filename="provider_from_dual.wav"
        )
        patient_text = str(patient_detailed.get("text") or "")
        provider_text = str(provider_detailed.get("text") or "")
        leg_updates.update(
            {
                "patient_transcript": patient_text,
                "patient_segments": patient_detailed.get("segments") or [],
                "provider_transcript": provider_text,
                "provider_segments": provider_detailed.get("segments") or [],
                "provider_saved": True,
                "provider_stt_started": True,
                "dual_channel_split": True,
            }
        )
        # Keep separate agent audio URL on Call when we upload right channel.
        agent_s3 = _upload_recording_audio(
            right_audio,
            retell_call_id=retell_call_id,
            leg="provider",
            recording_sid=f"{recording_sid}_R",
        )
        if agent_s3:
            leg_updates["provider_recording_s3_url"] = agent_s3
        wt_logger.info(
            "TRANSFER_DUAL_SPLIT session=%s patient_chars=%s provider_chars=%s",
            session_id,
            len(patient_text),
            len(provider_text),
        )
        text = patient_text
        segments = patient_detailed.get("segments") or []
    elif speaker == "patient":
        # Mono mix fallback: diarize so we don't store one Patient wall.
        detailed = transcribe_recording_detailed(audio, filename="patient.wav")
        text = str(detailed.get("text") or "")
        segments = detailed.get("segments") or []
        latest_for_stt = load_session(session_id) or session
        provider_anchor = str(latest_for_stt.get("provider_transcript") or "")
        # Plain mono STT of this same file is a weak patient anchor only;
        # provider-leg text is the reliable live-agent voice sample.
        patient_anchor = str(latest_for_stt.get("patient_transcript") or text or "")
        diarized = transcribe_mix_with_diarization(
            audio,
            filename="patient_mix.wav",
            provider_text=provider_anchor,
            patient_text=patient_anchor,
        )
        if _mix_items_are_split(diarized):
            patient_parts = [
                str(i.get("text") or "").strip()
                for i in diarized
                if i.get("speaker") == "patient" and str(i.get("text") or "").strip()
            ]
            provider_parts = [
                str(i.get("text") or "").strip()
                for i in diarized
                if i.get("speaker") == "live_agent" and str(i.get("text") or "").strip()
            ]
            leg_updates.update(
                {
                    "patient_transcript": " ".join(patient_parts),
                    "provider_transcript": " ".join(provider_parts),
                    "patient_segments": [
                        {"start": i.get("at") or 0, "end": i.get("at") or 0, "text": i.get("text")}
                        for i in diarized
                        if i.get("speaker") == "patient"
                    ],
                    "provider_segments": [
                        {"start": i.get("at") or 0, "end": i.get("at") or 0, "text": i.get("text")}
                        for i in diarized
                        if i.get("speaker") == "live_agent"
                    ],
                    "mix_diarized_items": diarized,
                    "provider_saved": True,
                    "provider_stt_started": True,
                    "diarized_from_patient_mono": True,
                }
            )
            wt_logger.info(
                "TRANSFER_DIARIZE_FALLBACK session=%s turns=%s",
                session_id,
                len(diarized),
            )
        else:
            leg_updates["patient_transcript"] = text
            leg_updates["patient_segments"] = segments
    else:
        detailed = (
            transcribe_recording_detailed(audio, filename=f"{speaker}.wav")
            if audio
            else {"text": "", "segments": []}
        )
        text = str(detailed.get("text") or "")
        segments = detailed.get("segments") or []
        # Do not overwrite dual-split provider text with empty inbound STT.
        latest = load_session(session_id) or {}
        if speaker == "provider" and latest.get("dual_channel_split") and (
            latest.get("provider_transcript") or ""
        ).strip():
            leg_updates = {
                "provider_recording_url": recording_url,
                "provider_recording_sid": recording_sid,
                "provider_saved": True,
                "provider_stt_started": True,
            }
            if s3_url and not latest.get("provider_recording_s3_url"):
                leg_updates["provider_recording_s3_url"] = s3_url
            text = str(latest.get("provider_transcript") or "")
            segments = latest.get("provider_segments") or []
        else:
            leg_updates[f"{speaker}_transcript"] = text
            leg_updates[f"{speaker}_segments"] = segments

    patch_session(session_id, leg_updates)

    session = _finalize_leg_persist(session_id, speaker=speaker)
    if not text and speaker == "patient":
        wt_logger.error(
            "TRANSFER_FAIL session=%s leg=%s reason=empty_stt",
            session_id,
            speaker,
        )
    logger.info(
        "Saved %s transcript session=%s chars=%s segments=%s both=%s dual=%s",
        speaker,
        session_id,
        len(text),
        len(segments),
        bool(
            session.get("patient_transcript") and session.get("provider_transcript")
        ),
        bool(session.get("dual_channel_split")),
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
        if session.get("db_persisted_complete"):
            return
        if _has_speaker_split(session):
            _finalize_leg_persist(session_id, speaker="poll")
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
        processed = set(session.get("processed_recording_sids") or [])
        for rec in recordings:
            rec_status = str(getattr(rec, "status", "") or "").lower()
            duration = int(getattr(rec, "duration", 0) or 0)
            if rec_status != "completed" or duration < 3:
                continue
            rec_sid = str(getattr(rec, "sid", "") or "")
            if rec_sid and rec_sid in processed:
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
            if rec_leg == "mix" and _has_speaker_split(session):
                continue
            persist_humans_transcript(
                session,
                recording_url=recording_url,
                recording_sid=rec_sid,
                leg=rec_leg,
            )
            if rec_sid:
                processed.add(rec_sid)
                patch_session(
                    session_id, {"processed_recording_sids": list(processed)}
                )
            session = load_session(session_id) or {}
            if session.get("db_persisted_complete"):
                return
            if _has_speaker_split(session):
                _finalize_leg_persist(session_id, speaker="poll")
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
    _schedule_voice_url_restore(session_id)
    wt_logger.info(
        "TRANSFER_BRIDGE_READY session=%s inbound=%s agents=%s",
        session_id,
        inbound_url,
        session_data.get("transfer_numbers"),
    )
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
