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
import math
import re
import struct
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
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
_OPEN_SESSIONS_KEY = "warm_transfer:open_sessions"
_ORIGINAL_VOICE_KEY = "warm_transfer:original_voice_url"

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
        _index_session(fallback, session)
        return session
    _index_session(cache, session)
    return session


def _phone_digits(value: str) -> str:
    return _digits(normalize_phone(value or "") or (value or ""))


def _phone_cache_key(digits: str) -> str:
    return f"warm_transfer:phone:{digits}"


def _queue_scope(session: Dict[str, Any]) -> str:
    # Fixed when the session is created: failover changes transfer_number.
    stored = str(session.get("queue_scope") or "")
    if stored:
        return stored
    user_id = session.get("user_id")
    if user_id:
        return f"user:{user_id}"
    number = _phone_digits(str(session.get("transfer_number") or ""))
    return f"agent:{number or 'default'}"


def _queue_key(scope: str) -> str:
    return f"warm_transfer:queue:{scope}"


def _agent_key(scope: str, number: str) -> str:
    """Which patient session holds this live agent right now."""
    return f"warm_transfer:agent:{scope}:{_phone_digits(number)}"


# A live-agent hold older than this is a leftover (e.g. a crashed call) — free it.
_AGENT_HOLD_MAX_SECONDS = 3 * 60 * 60


def _index_session(cache, session: Dict[str, Any]) -> None:
    """Remember this session by patient phone so concurrent transfers stay apart."""
    sid = str(session.get("session_id") or "").strip()
    if not sid:
        return
    phone = _phone_digits(str(session.get("phone_number") or ""))
    if phone:
        cache.set(_phone_cache_key(phone), sid, timeout=_SESSION_TTL_SECONDS)
    retell_id = str(session.get("retell_call_id") or "").strip()
    if retell_id:
        cache.set(
            f"warm_transfer:retell:{retell_id}",
            sid,
            timeout=_SESSION_TTL_SECONDS,
        )
    ended = bool(session.get("conference_ended") or session.get("agent_released"))
    # get()-then-set() on the shared open-sessions list is not atomic — two
    # concurrent transfers (e.g. patients requesting a live agent at the same
    # moment) can both read the same snapshot and the second write silently
    # clobbers the first, permanently dropping that patient from the queue.
    # Guard the read-modify-write with the same short spin-lock used for the
    # active/waiting agent slot elsewhere in this module.
    lock = "warm_transfer:open_sessions_lock"
    locked = False
    for _ in range(20):
        if cache.add(lock, sid, timeout=5):
            locked = True
            break
        time.sleep(0.02)
    try:
        try:
            ids = list(cache.get(_OPEN_SESSIONS_KEY) or [])
        except Exception:
            ids = []
        if ended:
            ids = [item for item in ids if item != sid]
        elif sid not in ids:
            ids.append(sid)
        cache.set(_OPEN_SESSIONS_KEY, ids, timeout=_SESSION_TTL_SECONDS)
    finally:
        if locked:
            cache.delete(lock)


def _open_sessions() -> list:
    try:
        ids = list(_session_cache().get(_OPEN_SESSIONS_KEY) or [])
    except Exception:
        ids = []
    sessions = []
    for sid in ids:
        session = load_session(str(sid))
        if session:
            sessions.append(session)
    return sessions


def _same_phone(left: str, right: str) -> bool:
    a = _phone_digits(left)
    b = _phone_digits(right)
    if not a or not b:
        return False
    if a == b:
        return True
    return len(a) >= 10 and len(b) >= 10 and a[-10:] == b[-10:]


def _party_numbers(*values: str) -> list:
    bridge = _phone_digits(_from_number())
    found = []
    for value in values:
        digits = _phone_digits(value)
        if not digits or digits == bridge or digits in found:
            continue
        found.append(digits)
    return found


# A transfer happens during the call that created its session. Older unclaimed
# sessions (calls that never transferred) must not grab a new inbound transfer.
_INBOUND_SESSION_MAX_AGE_SECONDS = 60 * 60


def _session_started_at(session: Dict[str, Any]) -> float:
    """Epoch seconds when the session was created (updated_at for older rows)."""
    created = session.get("created_at")
    if isinstance(created, (int, float)):
        return float(created)
    try:
        return datetime.fromisoformat(str(session.get("updated_at") or "")).timestamp()
    except ValueError:
        return 0.0


def resolve_inbound_session(
    *,
    call_sid: str = "",
    from_number: str = "",
    to_number: str = "",
    caller: str = "",
    called: str = "",
) -> Dict[str, Any]:
    """
    Pick this inbound transfer's own session.

    The Twilio number has one voice URL, so every transfer used to land in
    the latest conference. Match the patient phone, and if that is missing
    hand out unclaimed sessions one at a time.
    """
    call_sid = (call_sid or "").strip()
    cache = _session_cache()
    lock = "warm_transfer:inbound_lock"
    if not cache.add(lock, call_sid or "1", timeout=10):
        # Another transfer is being assigned. Wait briefly and try once.
        time.sleep(0.2)
        if not cache.add(lock, call_sid or "1", timeout=10):
            wt_logger.warning("TRANSFER_INBOUND_LOCK_BUSY call=%s", call_sid or "-")
            return {}
    try:
        sessions = _open_sessions()
        if call_sid:
            for session in sessions:
                if str(session.get("call_sid") or "") == call_sid:
                    return session
        # Only sessions from the last hour, so a new transfer never binds to an
        # earlier call's leftover session (e.g. a rehearsal call to the same phone).
        cutoff = time.time() - _INBOUND_SESSION_MAX_AGE_SECONDS
        sessions = [s for s in sessions if _session_started_at(s) >= cutoff]
        parties = _party_numbers(from_number, to_number, caller, called)
        for party in parties:
            # Same phone on two recent sessions: the newest call is the one transferring.
            for session in sorted(sessions, key=_session_started_at, reverse=True):
                if session.get("conference_ended") or session.get("agent_released"):
                    continue
                bound = str(session.get("call_sid") or "")
                if bound and bound != call_sid:
                    continue
                if _same_phone(str(session.get("phone_number") or ""), party):
                    if call_sid:
                        session["call_sid"] = call_sid
                        save_session(session)
                    return session
        unclaimed = [
            session
            for session in sessions
            if not session.get("call_sid")
            and not session.get("conference_ended")
            and not session.get("agent_released")
        ]
        if not unclaimed:
            return {}
        chosen = unclaimed[0]
        if call_sid:
            chosen["call_sid"] = call_sid
            save_session(chosen)
        if len(unclaimed) > 1 and not parties:
            wt_logger.info(
                "TRANSFER_INBOUND_FIFO session=%s call=%s waiting=%s",
                chosen.get("session_id"),
                call_sid or "-",
                len(unclaimed) - 1,
            )
        return chosen
    finally:
        cache.delete(lock)


def _slot_busy(session_id: str) -> bool:
    session = load_session(session_id)
    if not session:
        return False
    if session.get("agent_released") or session.get("conference_ended"):
        return False
    return str(session.get("queue_state") or "") == "active"


def _enqueue(cache, scope: str, session_id: str) -> None:
    key = _queue_key(scope)
    rows = [str(item) for item in (cache.get(key) or []) if str(item) != session_id]
    rows.append(session_id)
    cache.set(key, rows, timeout=_SESSION_TTL_SECONDS)


def _dequeue(cache, scope: str, session_id: str) -> None:
    key = _queue_key(scope)
    rows = [str(item) for item in (cache.get(key) or []) if str(item) != session_id]
    cache.set(key, rows, timeout=_SESSION_TTL_SECONDS)


def _agent_numbers(session: Dict[str, Any]) -> List[str]:
    """This account's live agents, in priority order."""
    numbers = [
        str(n).strip()
        for n in (
            (session or {}).get("transfer_numbers")
            or (session or {}).get("live_agent_numbers")
            or [(session or {}).get("transfer_number")]
        )
        if str(n or "").strip()
    ]
    unique: List[str] = []
    for number in numbers:
        if _phone_digits(number) not in {_phone_digits(u) for u in unique}:
            unique.append(number)
    return unique


def _agent_holder(cache, scope: str, number: str) -> Dict[str, Any]:
    """The live session holding this agent, or {} when the agent is free.

    Self-healing: a hold whose session ended, moved to another agent, or is
    hours old is cleared, so an agent never stays "busy" forever.
    """
    key = _agent_key(scope, number)
    holder_sid = str(cache.get(key) or "")
    if not holder_sid:
        return {}
    holder = load_session(holder_sid)
    stale = (
        not holder
        or holder.get("agent_released")
        or holder.get("conference_ended")
        or _phone_digits(str(holder.get("assigned_agent") or "")) != _phone_digits(number)
        or (time.time() - _session_started_at(holder)) > _AGENT_HOLD_MAX_SECONDS
    )
    if stale:
        if cache.get(key) == holder_sid:
            cache.delete(key)
        return {}
    return holder


def _reserve_free_agent(cache, scope: str, session: Dict[str, Any], *, skip=()) -> str:
    """Atomically take the first free agent (in priority order). "" when all are busy."""
    sid = str(session.get("session_id") or "")
    skipped = {_phone_digits(n) for n in skip}
    for number in _agent_numbers(session):
        if _phone_digits(number) in skipped:
            continue
        if _agent_holder(cache, scope, number):
            continue
        if cache.add(_agent_key(scope, number), sid, timeout=_SESSION_TTL_SECONDS):
            return number
    return ""


def _release_agent(cache, scope: str, session: Dict[str, Any]) -> None:
    number = str(session.get("assigned_agent") or "")
    sid = str(session.get("session_id") or "")
    if number and str(cache.get(_agent_key(scope, number)) or "") == sid:
        cache.delete(_agent_key(scope, number))


def _tried_numbers(session: Dict[str, Any]) -> List[str]:
    tried = set(session.get("tried_agents") or [])
    return [n for n in _agent_numbers(session) if _phone_digits(n) in tried]


def _assign_agent(session: Dict[str, Any], number: str) -> Dict[str, Any]:
    """Mark `number` as this patient's agent (the next dial goes to them)."""
    numbers = _agent_numbers(session)
    digits = [_phone_digits(n) for n in numbers]
    tried = list(session.get("tried_agents") or [])
    if _phone_digits(number) not in tried:
        tried.append(_phone_digits(number))
    session.update(
        {
            "queue_state": "active",
            "agent_claimed": True,
            "assigned_agent": number,
            "transfer_number": number,
            "live_agent_number": number,
            "transfer_number_index": (
                digits.index(_phone_digits(number)) if _phone_digits(number) in digits else 0
            ),
            "tried_agents": tried,
            "b_dialed": False,
            "dial_call_sid": "",
        }
    )
    return session


def _lock(cache, scope: str, owner: str, *, tries: int = 40) -> bool:
    lock = f"warm_transfer:admit:{scope}"
    for _ in range(tries):
        if cache.add(lock, owner, timeout=10):
            return True
        time.sleep(0.05)
    return False


def _unlock(cache, scope: str) -> None:
    cache.delete(f"warm_transfer:admit:{scope}")


def _busy_agent_sessions(cache, scope: str, session: Dict[str, Any]) -> List[Dict[str, Any]]:
    holders = []
    for number in _agent_numbers(session):
        holder = _agent_holder(cache, scope, number)
        if holder and holder.get("session_id") != session.get("session_id"):
            holders.append(holder)
    return holders


def _notify_busy_agents_of_new_waiter(cache, scope: str, session: Dict[str, Any]) -> None:
    """Private beep to every agent who is on a call: a patient is now waiting.

    Call outside the queue lock — each beep is a Twilio API request."""
    for holder in _busy_agent_sessions(cache, scope, session):
        notify_agent_waiting_tone(holder)


def _wait_for_agent(cache, scope: str, session: Dict[str, Any], *, front: bool = False) -> None:
    sid = str(session.get("session_id") or "")
    rows = [str(item) for item in (cache.get(_queue_key(scope)) or []) if str(item) != sid]
    rows = [sid, *rows] if front else [*rows, sid]
    cache.set(_queue_key(scope), rows, timeout=_SESSION_TTL_SECONDS)
    session["queue_state"] = "waiting"
    session["agent_claimed"] = False
    session["assigned_agent"] = ""
    session["waiting_since"] = time.time()
    session["wait_token"] = uuid.uuid4().hex


# All agents busy this long → offer the patient a callback instead of holding.
def _max_wait_seconds() -> int:
    return int(getattr(settings, "WARM_TRANSFER_MAX_WAIT_SECONDS", 180) or 180)


def _schedule_long_wait_offer(session: Dict[str, Any]) -> None:
    """Check back after the max wait (Celery). Without Celery the patient just keeps holding."""
    sid = str(session.get("session_id") or "")
    token = str(session.get("wait_token") or "")
    if not sid or not token:
        return
    try:
        from apps.ai_caller.tasks import offer_callback_after_long_wait

        offer_callback_after_long_wait.apply_async(
            args=[sid, token], countdown=_max_wait_seconds()
        )
    except Exception:
        wt_logger.exception("LONG_WAIT_OFFER_SCHEDULE_FAILED session=%s", sid)


def admit_patient_to_agent(session: Dict[str, Any]) -> str:
    """
    Give this patient the first free live agent; with every agent busy, the
    patient waits on hold and each busy agent hears a short beep.

    Returns "active" or "waiting".
    """
    session = load_session(str((session or {}).get("session_id") or "")) or dict(
        session or {}
    )
    sid = str(session.get("session_id") or "").strip()
    if not sid:
        return "waiting"
    if session.get("queue_state") == "active" and session.get("agent_claimed"):
        return "active"
    if session.get("queue_state") == "waiting":
        return "waiting"
    scope = _queue_scope(session)
    cache = _session_cache()
    locked = _lock(cache, scope, sid)
    busy_agents: List[Dict[str, Any]] = []
    try:
        number = _reserve_free_agent(cache, scope, session) if locked else ""
        if number:
            _assign_agent(session, number)
            save_session(session)
            wt_logger.info("AGENT_QUEUE_ACTIVE session=%s scope=%s agent=%s", sid, scope, number)
            return "active"
        busy_agents = _busy_agent_sessions(cache, scope, session)
        _wait_for_agent(cache, scope, session)
        save_session(session)
        wt_logger.info(
            "AGENT_QUEUE_WAIT session=%s scope=%s reason=%s",
            sid,
            scope,
            "all_agents_busy" if locked else "lock",
        )
    finally:
        if locked:
            _unlock(cache, scope)
    if session.get("queue_state") == "waiting":
        for holder in busy_agents:
            notify_agent_waiting_tone(holder)  # outside the lock (Twilio API calls)
        _schedule_long_wait_offer(session)
    if not locked:
        # Could not check the agents just now — try again so a free agent is used.
        promote_next_waiting(scope)
        session = load_session(sid) or session
        if session.get("queue_state") == "active":
            return "active"  # an agent was free after all (already being dialed)
    return "waiting"


def transfer_call_should_stay_open(retell_call_id: str) -> bool:
    """True while this patient is still with the agent or waiting for one."""
    retell_call_id = (retell_call_id or "").strip()
    if not retell_call_id:
        return False
    try:
        sid = str(_session_cache().get(f"warm_transfer:retell:{retell_call_id}") or "")
    except Exception:
        sid = ""
    session = load_session(sid) if sid else {}
    if not session or session.get("conference_ended") or session.get("agent_released"):
        return False
    state = str(session.get("queue_state") or "")
    if state in {"waiting", "active"}:
        return True
    return bool(session.get("call_sid"))


def set_transfer_call_status(session: Dict[str, Any], status: str) -> None:
    """Publish waiting / in_progress / completed on the call so the queue updates."""
    from apps.ai_caller.models import Call

    session = dict(session or {})
    retell_call_id = str(session.get("retell_call_id") or "").strip()
    call_sid = str(session.get("call_sid") or "").strip()
    call = None
    if retell_call_id:
        call = Call.objects.filter(retell_call_id=retell_call_id).first()
    if call is None and call_sid:
        call = Call.objects.filter(warm_transfer_session_id=call_sid).first()
    if call is None or call.status == status:
        return
    if call.status in {Call.Status.CALLBACK, Call.Status.CANCEL}:
        return
    call.status = status
    call.save(update_fields=["status", "updated_at"])


def release_agent_slot(session: Dict[str, Any]) -> None:
    """Free this patient's live agent and give them to the next waiting patient."""
    session = load_session(str((session or {}).get("session_id") or "")) or dict(
        session or {}
    )
    sid = str(session.get("session_id") or "").strip()
    if not sid or session.get("agent_released"):
        return
    scope = _queue_scope(session)
    cache = _session_cache()
    had_agent = bool(session.get("assigned_agent")) or bool(session.get("agent_claimed"))
    locked = _lock(cache, scope, sid)
    try:
        _dequeue(cache, scope, sid)
        _release_agent(cache, scope, session)
    finally:
        if locked:
            _unlock(cache, scope)
    dial_sid = str(session.get("dial_call_sid") or "").strip()
    if dial_sid and not session.get("spoke_now"):
        # Patient left while their agent's phone was still ringing: stop it, so the
        # agent is free and no late "no-answer" re-opens this finished call.
        _hangup_call(dial_sid)
    session["agent_released"] = True
    session["conference_ended"] = True
    session["agent_claimed"] = False
    session["queue_state"] = "ended"
    save_session(session)
    # Also after a callback offer: a scheduled callback sets CALLBACK, which
    # set_transfer_call_status never overwrites; a hang-up must not leave the
    # call "in progress" forever (that blocks every future dial).
    set_transfer_call_status(session, "completed")
    if had_agent:
        promote_next_waiting(scope)
    restore_inbound_voice_url(session)


def promote_next_waiting(scope: str) -> None:
    """Connect waiting patients (longest wait first) to every free live agent."""
    scope = (scope or "").strip()
    if not scope:
        return
    cache = _session_cache()
    if not _lock(cache, scope, "promote", tries=100):
        wt_logger.warning("AGENT_QUEUE_PROMOTE_LOCK_BUSY scope=%s", scope)
        return
    promoted: List[Dict[str, Any]] = []
    try:
        rows = [str(item) for item in (cache.get(_queue_key(scope)) or [])]
        remaining: List[str] = []
        for sid in rows:
            session = load_session(sid)
            if (
                not session
                or session.get("agent_released")
                or session.get("conference_ended")
                or not session.get("call_sid")
            ):
                continue  # hung up while waiting
            if session.get("hold_offer_in_progress"):
                remaining.append(sid)  # answering the callback offer: keep their place
                continue
            # Never re-ring an agent who already missed this patient on this call.
            number = _reserve_free_agent(
                cache, scope, session, skip=_tried_numbers(session)
            )
            if not number:
                remaining.append(sid)  # no suitable agent free: keep their place in line
                continue
            _assign_agent(session, number)
            save_session(session)
            promoted.append(session)
            wt_logger.info(
                "AGENT_QUEUE_PROMOTE session=%s scope=%s agent=%s", sid, scope, number
            )
        cache.set(_queue_key(scope), remaining, timeout=_SESSION_TTL_SECONDS)
    finally:
        _unlock(cache, scope)
    for chosen in promoted:
        set_transfer_call_status(chosen, "in_progress")
        announce_connecting(chosen)
        start_coordinator_leg(chosen)


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


def _human_recording_url(session: Dict[str, Any]) -> str:
    """Best audio of the patient <-> live agent talk, both voices first.

    Order: patient dual-channel (both voices) > conference mix (both voices)
    > live-agent leg > patient mono leg.
    """
    s = session or {}
    candidates = []
    if s.get("patient_recording_dual"):
        candidates += [s.get("patient_recording_s3_url"), s.get("patient_recording_url")]
    candidates += [
        s.get("mix_recording_s3_url"),
        s.get("mix_recording_url"),
        s.get("provider_recording_s3_url"),
        s.get("provider_recording_url"),
        s.get("patient_recording_s3_url"),
        s.get("patient_recording_url"),
    ]
    for url in candidates:
        if str(url or "").strip():
            return str(url).strip()
    return ""


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
        _human_recording_url(session)
        or live_agent_recording_url
        or str((session or {}).get("provider_recording_s3_url") or "")
        or str((session or {}).get("provider_recording_url") or "")
    ).strip()
    update_fields = []
    if twilio_call_sid and call.warm_transfer_session_id != twilio_call_sid:
        call.warm_transfer_session_id = twilio_call_sid[:64]
        update_fields.append("warm_transfer_session_id")
    # recording_url is the AI (Retell) call audio. Only use a Twilio leg when
    # Retell has not provided one; Retell's own recording replaces it later.
    if patient_url and not (call.recording_url or "").strip():
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


def _hangup_call(call_sid: str) -> None:
    """End one Twilio call so it cannot enter a conference it no longer belongs to."""
    call_sid = (call_sid or "").strip()
    if not call_sid:
        return
    client = _twilio_client()
    if client is None:
        return
    try:
        client.calls(call_sid).update(status="completed")
    except Exception:
        logger.exception("Failed to hang up call %s", call_sid)


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
    """One conference per patient session. Names are never shared."""
    safe = re.sub(r"[^A-Za-z0-9]", "", (session_id or "").strip())[:32]
    return f"care-wt-{safe}"[:40]


def agent_join_allowed(session: Dict[str, Any], call_sid: str = "") -> bool:
    """Only the agent dialed for the active patient may enter that conference."""
    if not session:
        return False
    if session.get("conference_ended") or session.get("agent_released"):
        return False
    if str(session.get("queue_state") or "") != "active" or not session.get("agent_claimed"):
        return False
    allowed = str(session.get("dial_call_sid") or "").strip()
    call_sid = (call_sid or "").strip()
    if allowed and call_sid and call_sid != allowed:
        return False
    return True


def shared_inbound_url() -> str:
    """One webhook for every transfer. The caller is matched to their own session."""
    return f"{_backend_base()}/api/ai-call/twilio/warm-transfer/inbound/"


def _ring_url() -> str:
    """Audio played to the patient while the live agent line is ringing."""
    configured = (getattr(settings, "WARM_TRANSFER_RING_URL", "") or "").strip()
    if configured:
        return configured
    # Twilio Client outgoing ringtone — works as a transfer ringback.
    return "https://sdk.twilio.com/js/client/sounds/releases/1.0.0/outgoing.mp3"


# Twilio re-requests the waitUrl when its TwiML finishes, so a finite ring makes
# the hold message repeat (every ~30-40s) and follows a change from "agent
# being dialed" to "waiting for a free agent".
_WAIT_RING_LOOPS = 8


def conference_wait_twiml(session: Optional[Dict[str, Any]] = None) -> str:
    """Conference waitUrl, played while the patient is alone in their room.

    Agent being dialed → ringback. All agents busy → the hold message, then
    ringback, repeated until an agent is free.
    """
    session_id = str((session or {}).get("session_id") or "")
    if not session_id:
        ring_forever = f'<Play loop="0">{escape(_ring_url())}</Play>'
        return '<?xml version="1.0" encoding="UTF-8"?>' f"<Response>{ring_forever}</Response>"
    ring = f'<Play loop="{_WAIT_RING_LOOPS}">{escape(_ring_url())}</Play>'
    hold = ""
    if str((session or {}).get("queue_state") or "") == "waiting":
        hold = _say(HOLD_MESSAGE)
    # Explicitly come back here, so the hold message repeats and follows state changes.
    again = escape(_abs_url("/api/ai-call/twilio/warm-transfer/wait/", session_id))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response>{hold}{ring}<Redirect method="POST">{again}</Redirect></Response>'
    )


def join_conference_twiml(session: Dict[str, Any], *, is_host: bool) -> str:
    session_id = str(session.get("session_id") or "")
    conf = escape(conference_name(session_id))
    status_cb = escape(
        _abs_url("/api/ai-call/twilio/warm-transfer/conference-status/", session_id)
    )
    wait_url = escape(
        _abs_url("/api/ai-call/twilio/warm-transfer/wait/", session_id)
    )
    mix_cb = escape(
        _abs_url(
            "/api/ai-call/twilio/warm-transfer/recording/",
            session_id,
            leg="mix",
        )
    )
    # Either side leaving ends the call: the agent hanging up must not leave the
    # patient in a silent room (and frees the agent for the next patient).
    end_on_exit = "true"
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
        f'<Conference beep="false" waitUrl="{wait_url}" waitMethod="POST" '
        f'startConferenceOnEnter="true" '
        f'endConferenceOnExit="{end_on_exit}" {record_attrs}'
        f'statusCallback="{status_cb}" statusCallbackEvent="start join end" '
        f'statusCallbackMethod="POST">{conf}</Conference>'
        "</Dial>"
        f"{_say(CALL_ENDED_MESSAGE) if is_host else ''}"
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


def agent_briefing_text(session: Dict[str, Any]) -> str:
    """What the live agent hears on answering, before meeting the patient."""
    name = str((session or {}).get("name") or "").strip()
    if not name or name.lower() == "there":
        name = "a patient"
    service = str((session or {}).get("service_name") or "").strip()
    about = f" about {service}" if service and service.lower() != "care" else ""
    return f"Bay Area Community Health transfer. Connecting you with {name}{about}."


def participant_join_twiml(session: Dict[str, Any]) -> str:
    """Person B (live agent) hears who is waiting, then joins the same conference.

    The <Say> runs on the agent's call leg only, so the patient never hears it.
    """
    body = join_conference_twiml(session, is_host=False)
    return body.replace("<Response>", f"<Response>{_say(agent_briefing_text(session))}", 1)


def speak_now_twiml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say('You can speak now.')}"
        "</Response>"
    )


_QUEUE_TONE_WAV: Optional[bytes] = None


def generate_queue_tone_wav() -> bytes:
    """Short single beep — no external audio asset needed. Cached after first build."""
    global _QUEUE_TONE_WAV
    if _QUEUE_TONE_WAV is not None:
        return _QUEUE_TONE_WAV
    sample_rate = 8000
    duration = 0.3
    freq = 900.0
    amplitude = 12000
    n_samples = int(sample_rate * duration)
    frames = bytearray()
    for i in range(n_samples):
        value = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        frames += struct.pack("<h", value)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    _QUEUE_TONE_WAV = buf.getvalue()
    return _QUEUE_TONE_WAV


def queue_tone_twiml(session_id: str = "") -> str:
    tone_url = escape(
        _abs_url("/api/ai-call/twilio/warm-transfer/queue-tone-audio/", session_id)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Play>{tone_url}</Play>"
        "</Response>"
    )


def notify_agent_waiting_tone(active_session: Dict[str, Any]) -> None:
    """Private one-way beep to the live agent only — a caller is now waiting.

    Uses the Conference Participant-level announce: it plays only to the
    agent's leg, the patient on the call never hears it.
    """
    conference_sid = str(active_session.get("conference_sid") or "").strip()
    agent_call_sid = str(active_session.get("dial_call_sid") or "").strip()
    if not conference_sid or not agent_call_sid:
        return
    client = _twilio_client()
    if client is None:
        return
    session_id = str(active_session.get("session_id") or "")
    try:
        client.conferences(conference_sid).participants(agent_call_sid).update(
            announce_url=_abs_url(
                "/api/ai-call/twilio/warm-transfer/queue-tone/", session_id
            ),
            announce_method="POST",
        )
        wt_logger.info(
            "AGENT_QUEUE_TONE session=%s conf=%s agent_call=%s",
            session_id,
            conference_sid,
            agent_call_sid,
        )
    except Exception:
        wt_logger.exception(
            "Failed to play queue-waiting tone conf=%s agent_call=%s",
            conference_sid,
            agent_call_sid,
        )


def next_agent_twiml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say('That agent is unavailable. I am connecting you to the next agent now. Please hold.')}"
        "</Response>"
    )


HOLD_MESSAGE = "Please hold. A live agent is helping another caller. You are still in line."
# Played to the patient after the live agent hangs up (the conference has ended).
CALL_ENDED_MESSAGE = "Thank you for calling Bay Area Community Health. Goodbye."


def hold_twiml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say(HOLD_MESSAGE)}"
        "</Response>"
    )


def connecting_twiml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say('Thank you for waiting. I am connecting you to a live agent now.')}"
        "</Response>"
    )


def ask_callback_twiml(session: Dict[str, Any]) -> str:
    session_id = str((session or {}).get("session_id") or "")
    gather_url = escape(
        _abs_url("/api/ai-call/twilio/warm-transfer/callback-gather/", session_id)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say('I am sorry, none of our enrollment agents are available right now. What day and time works best for a callback?')}"
        f'<Gather input="speech" timeout="8" speechTimeout="auto" '
        f'action="{gather_url}" method="POST">'
        f"{_say('For example, you can say tomorrow at 3 PM, or Friday morning.')}"
        "</Gather>"
        f"{_say('Sorry, I did not catch a time. We will follow up another day. Goodbye.')}"
        "<Hangup/>"
        "</Response>"
    )


def callback_scheduled_twiml(when_text: str = "") -> str:
    when = (when_text or "").strip()
    if when:
        line = (
            f"Got it. I have scheduled your callback for {when}. "
            "Thank you for your time. Goodbye."
        )
    else:
        line = (
            "Got it. I have scheduled your callback. "
            "Thank you for your time. Goodbye."
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say(line)}"
        "<Hangup/>"
        "</Response>"
    )


def callback_unclear_twiml(session: Dict[str, Any]) -> str:
    session_id = str((session or {}).get("session_id") or "")
    gather_url = escape(
        _abs_url("/api/ai-call/twilio/warm-transfer/callback-gather/", session_id)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{_say('Sorry, I did not catch that. Please tell me a day and time for your callback.')}"
        f'<Gather input="speech" timeout="8" speechTimeout="auto" '
        f'action="{gather_url}" method="POST"></Gather>'
        f"{_say('We will follow up another day. Goodbye.')}"
        "<Hangup/>"
        "</Response>"
    )


def _announce_conference(session: Dict[str, Any], path: str, label: str) -> None:
    session = dict(session or {})
    conference_sid = str(session.get("conference_sid") or "").strip()
    session_id = str(session.get("session_id") or "")
    if not conference_sid or not session_id:
        return
    client = _twilio_client()
    if client is None:
        return
    try:
        client.conferences(conference_sid).update(
            announce_url=_abs_url(path, session_id),
            announce_method="POST",
        )
        logger.info(
            "Announced %s session=%s conf=%s", label, session_id, conference_sid
        )
    except Exception:
        logger.exception("Failed to announce %s session=%s", label, session_id)


def announce_hold(session: Dict[str, Any]) -> None:
    _announce_conference(
        session, "/api/ai-call/twilio/warm-transfer/hold/", "hold"
    )


def announce_connecting(session: Dict[str, Any]) -> None:
    _announce_conference(
        session, "/api/ai-call/twilio/warm-transfer/connecting/", "connecting"
    )


def announce_next_agent(session: Dict[str, Any]) -> None:
    """Tell the patient on the conference that we are dialing the next live agent."""
    session = dict(session or {})
    conference_sid = str(session.get("conference_sid") or "").strip()
    session_id = str(session.get("session_id") or "")
    if not conference_sid or not session_id:
        return
    client = _twilio_client()
    if client is None:
        return
    try:
        client.conferences(conference_sid).update(
            announce_url=_abs_url(
                "/api/ai-call/twilio/warm-transfer/next-agent/", session_id
            ),
            announce_method="POST",
        )
        logger.info(
            "Announced next-agent failover session=%s conf=%s",
            session_id,
            conference_sid,
        )
    except Exception:
        logger.exception(
            "Failed to announce next-agent failover session=%s", session_id
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
        logger.error(
            "Cannot dial live agent session=%s client=%s destination=%s from=%s",
            session_id,
            client is not None,
            bool(destination),
            bool(from_number),
        )
        failover_to_next_live_agent(session, reason="dial_not_configured")
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
    This patient's agent did not answer (or could not be dialed): free that
    agent and dial the next FREE agent this patient has not tried yet.
    All other agents busy → back to the front of the waiting line.
    Every agent tried → ask the patient for a callback time.
    Returns True when the patient is being handled (dial, wait, or callback).
    """
    session = load_session(str((session or {}).get("session_id") or "")) or dict(
        session or {}
    )
    session_id = str(session.get("session_id") or "")
    if session.get("agent_released") or session.get("conference_ended"):
        return False  # the patient already left
    numbers = _agent_numbers(session)
    scope = _queue_scope(session)
    cache = _session_cache()
    previous = str(session.get("assigned_agent") or session.get("transfer_number") or "")

    # Drop the agent who did not answer so they cannot join late.
    _hangup_call(str(session.get("dial_call_sid") or "").strip())
    session["dial_call_sid"] = ""
    session["b_dialed"] = False
    session["provider_recording_started"] = False
    tried = list(session.get("tried_agents") or [])
    if previous and _phone_digits(previous) not in tried:
        tried.append(_phone_digits(previous))
    session["tried_agents"] = tried

    if len(numbers) < 2:
        # No other agent to try: ask the patient for a callback time.
        session["status"] = "agent_unreachable"
        save_session(session)
        return offer_agent_unreachable_callback(session)

    nxt = ""
    untried = [n for n in numbers if _phone_digits(n) not in tried]
    locked = _lock(cache, scope, session_id)
    try:
        _release_agent(cache, scope, session)
        if locked:
            already_tried = [n for n in numbers if _phone_digits(n) in tried]
            nxt = _reserve_free_agent(cache, scope, session, skip=already_tried)
        if nxt:
            attempts = list(session.get("failover_attempts") or [])
            attempts.append(
                {"from": previous, "to": nxt, "reason": reason or "no_answer"}
            )
            session["failover_attempts"] = attempts
            _assign_agent(session, nxt)
            session["status"] = "failover_dialing"
            save_session(session)
        elif untried:
            # Agents left to try, but all are on calls: wait for the first free one.
            _wait_for_agent(cache, scope, session, front=True)
            session["status"] = "waiting_for_agent"
            save_session(session)
        else:
            session["status"] = "agent_unreachable"
            save_session(session)
    finally:
        if locked:
            _unlock(cache, scope)

    dialer_logger = logging.getLogger("ai_caller.dialer")
    if not nxt and not untried:
        dialer_logger.info(
            "AGENT_FAILOVER_EXHAUSTED session=%s reason=%s tried=%s",
            session_id,
            reason or "unknown",
            len(tried),
        )
        offer_agent_unreachable_callback(session)
        return True
    if not nxt:
        dialer_logger.info(
            "AGENT_FAILOVER_WAIT session=%s reason=%s from=%s busy_untried=%s",
            session_id,
            reason or "no_answer",
            previous,
            len(untried),
        )
        set_transfer_call_status(session, "waiting")
        announce_hold(session)
        _notify_busy_agents_of_new_waiter(cache, scope, session)
        _schedule_long_wait_offer(session)
        # The agent who did not answer is free again — offer them to others in line.
        promote_next_waiting(scope)
        return True

    dialer_logger.info(
        "AGENT_FAILOVER session=%s reason=%s from=%s to=%s",
        session_id,
        reason or "no_answer",
        previous,
        nxt,
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

    # Patient is already in the conference — announce before dialing the next agent.
    announce_next_agent(session)
    start_coordinator_leg(session)
    return True


def offer_agent_unreachable_callback(session: Dict[str, Any]) -> bool:
    """
    When no live agent answers, redirect the patient leg to ask for a callback time.
    Returns True when the redirect (or prior offer) is in progress.
    """
    session = load_session(str((session or {}).get("session_id") or "")) or dict(
        session or {}
    )
    session_id = str(session.get("session_id") or "")
    if not session_id:
        return False
    if session.get("callback_offer_started") or session.get("status") in {
        "agent_unreachable_ask_callback",
        "callback_scheduled",
    }:
        return True

    patient_sid = str(session.get("call_sid") or "").strip()
    session["status"] = "agent_unreachable_ask_callback"
    session["callback_offer_started"] = True
    save_session(session)
    # This patient is no longer with the agent. Let the next waiting caller through.
    release_agent_slot(session)

    dialer_logger = logging.getLogger("ai_caller.dialer")
    dialer_logger.info(
        "AGENT_UNREACHABLE_ASK_CALLBACK session=%s patient_sid=%s",
        session_id,
        patient_sid or "-",
    )

    client = _twilio_client()
    if client is None or not patient_sid:
        logger.warning(
            "Cannot ask callback — missing twilio client or patient call_sid session=%s",
            session_id,
        )
        return False
    try:
        client.calls(patient_sid).update(
            url=_abs_url(
                "/api/ai-call/twilio/warm-transfer/ask-callback/", session_id
            ),
            method="POST",
        )
        return True
    except Exception:
        logger.exception(
            "Failed to redirect patient for callback ask session=%s", session_id
        )
        return False


def offer_callback_to_long_waiter(session_id: str, wait_token: str) -> bool:
    """Still waiting after the max wait → step the patient out to a callback offer."""
    session = load_session(session_id)
    if (
        not session
        or session.get("queue_state") != "waiting"
        or session.get("agent_released")
        or session.get("conference_ended")
        or session.get("hold_offer_in_progress")
        or str(session.get("wait_token") or "") != str(wait_token or "")
    ):
        return False  # connected, hung up, or a newer wait started
    patient_sid = str(session.get("call_sid") or "").strip()
    client = _twilio_client()
    if client is None or not patient_sid:
        return False
    scope = _queue_scope(session)
    cache = _session_cache()
    if not _lock(cache, scope, session_id):
        return False
    try:
        session = load_session(session_id) or session
        if session.get("queue_state") != "waiting" or session.get("hold_offer_in_progress"):
            return False  # promoted or already being offered meanwhile
        session["hold_offer_in_progress"] = True
        session["stepped_out_conference_sid"] = session.get("conference_sid") or ""
        save_session(session)
    finally:
        _unlock(cache, scope)
    try:
        client.calls(patient_sid).update(
            url=_abs_url("/api/ai-call/twilio/warm-transfer/long-wait-offer/", session_id),
            method="POST",
            # Know if they hang up while answering (frees their place in line).
            status_callback=_abs_url("/api/ai-call/twilio/warm-transfer/status/", session_id),
            status_callback_method="POST",
        )
    except Exception:
        wt_logger.exception("LONG_WAIT_OFFER_FAILED session=%s", session_id)
        session["hold_offer_in_progress"] = False
        save_session(session)
        return False
    wt_logger.info("LONG_WAIT_OFFER session=%s waited=%ss", session_id, _max_wait_seconds())
    return True


def long_wait_offer_twiml(session: Dict[str, Any]) -> str:
    session_id = str((session or {}).get("session_id") or "")
    choice_url = escape(_abs_url("/api/ai-call/twilio/warm-transfer/long-wait-choice/", session_id))
    keep_holding = escape(
        _abs_url("/api/ai-call/twilio/warm-transfer/long-wait-choice/", session_id, choice="hold")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Gather input="speech dtmf" numDigits="1" timeout="8" speechTimeout="auto" '
        f'action="{choice_url}" method="POST">'
        f"{_say('Sorry for the wait. All of our agents are still helping other callers. Would you like us to call you back instead? Press 1 or say yes for a callback. Press 2 or say no to keep holding.')}"
        "</Gather>"
        f'<Redirect method="POST">{keep_holding}</Redirect>'
        "</Response>"
    )


def _spoken_callback_time(when: datetime, tz) -> str:
    local = when.astimezone(tz)
    today = datetime.now(tz).date()
    clock = local.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")
    if local.date() == today:
        return f"today at {clock}"
    if (local.date() - today).days == 1:
        return f"tomorrow at {clock}"
    return f"on {local.strftime('%A')} at {clock}"


def schedule_long_wait_callback(session: Dict[str, Any]) -> str:
    """Book the callback at the Settings time. Returns what to tell the patient ("" on failure)."""
    from apps.ai_caller.models import Call, CallerSettings
    from apps.ai_caller.scheduling import schedule_callback_request
    from apps.ai_caller.services import next_callback_time, resolve_timezone

    retell_call_id = str(session.get("retell_call_id") or "").strip()
    call = Call.objects.filter(retell_call_id=retell_call_id).first() if retell_call_id else None
    if call is None or call.patient is None:
        return ""
    user = call.user or getattr(call.patient, "user", None)
    if user is None:
        return ""
    settings_obj = CallerSettings.load(user)
    when = next_callback_time(settings_obj)
    row, error = schedule_callback_request(
        call=call,
        raw_time_text="Callback offered after a long hold",
        scheduled_at=when,
    )
    if error or not row:
        wt_logger.warning("LONG_WAIT_CALLBACK_FAILED session=%s err=%s", session.get("session_id"), error)
        return ""
    return _spoken_callback_time(when, resolve_timezone(settings_obj.timezone))


_CALLBACK_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|alright|call me back|call back|callback)\b", re.I
)
_KEEP_HOLDING_RE = re.compile(r"\b(no|nope|hold|wait|keep|stay)\b", re.I)


def wants_long_wait_callback(speech: str, digits: str) -> bool:
    """Press 1 / "yes, call me back" → True. Press 2 / "no, I'll hold" → False.
    Mixed answers ("no, please keep me on hold") favour holding."""
    digits = (digits or "").strip()
    if digits:
        return digits == "1"
    speech = speech or ""
    return bool(_CALLBACK_YES_RE.search(speech)) and not _KEEP_HOLDING_RE.search(speech)


def handle_long_wait_choice(session: Dict[str, Any], *, speech: str, digits: str, choice: str) -> str:
    """Patient answered the offer: book the callback, or go back on hold in the same place."""
    session_id = str(session.get("session_id") or "")
    wants_callback = choice != "hold" and wants_long_wait_callback(speech, digits)
    if wants_callback:
        when_text = schedule_long_wait_callback(session)
        if when_text:
            session["hold_offer_in_progress"] = False
            session["status"] = "callback_scheduled"
            save_session(session)
            release_agent_slot(session)  # leaves the line; CALLBACK stays (never overwritten)
            wt_logger.info("LONG_WAIT_CALLBACK session=%s when=%s", session_id, when_text)
            return (
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<Response>"
                f"{_say(f'Thank you. We will call you back {when_text}. Goodbye.')}"
                "<Hangup/>"
                "</Response>"
            )
    # Keep holding (said no, pressed 2, no answer, or booking failed).
    if session.get("agent_released") or session.get("conference_ended"):
        # Their place was lost (e.g. a delayed hang-up signal): never leave them
        # holding with no line — book the callback instead.
        when_text = schedule_long_wait_callback(session)
        message = (
            f"We are sorry for the wait. We will call you back {when_text}. Goodbye."
            if when_text
            else "We are sorry for the wait. Our team will call you back. Goodbye."
        )
        return f'<?xml version="1.0" encoding="UTF-8"?><Response>{_say(message)}<Hangup/></Response>'
    scope = _queue_scope(session)
    cache = _session_cache()
    locked = _lock(cache, scope, session_id)
    try:
        rows = [str(r) for r in (cache.get(_queue_key(scope)) or [])]
        if session_id not in rows:
            cache.set(_queue_key(scope), [session_id, *rows], timeout=_SESSION_TTL_SECONDS)
        session = load_session(session_id) or session
        session["hold_offer_in_progress"] = False
        session["wait_token"] = uuid.uuid4().hex  # restart the max-wait clock
        save_session(session)
    finally:
        if locked:
            _unlock(cache, scope)
    _schedule_long_wait_offer(session)
    promote_next_waiting(_queue_scope(session))  # an agent may have freed meanwhile
    body = inbound_connect_twiml(load_session(session_id) or session)
    return body.replace(
        "<Response>",
        f"<Response>{_say('Okay. Please continue to hold. You are still in line.')}",
        1,
    )


def schedule_callback_from_warm_transfer_speech(
    session: Dict[str, Any], *, speech: str
) -> tuple:
    """
    Parse spoken callback time and create ScheduledOutreach.
    Returns (ok, message_for_patient, error).
    """
    from apps.ai_caller.models import Call
    from apps.ai_caller.scheduling import schedule_callback_request

    session = dict(session or {})
    raw = (speech or "").strip()
    if not raw:
        return False, "", "empty"

    retell_call_id = str(session.get("retell_call_id") or "").strip()
    patient_sid = str(session.get("call_sid") or "").strip()
    call = None
    if retell_call_id:
        call = Call.objects.filter(retell_call_id=retell_call_id).first()
    if not call and patient_sid:
        call = Call.objects.filter(warm_transfer_session_id=patient_sid).first()
    if not call:
        return False, "", "call_not_found"

    row, error = schedule_callback_request(call=call, raw_time_text=raw)
    if error or not row:
        return False, "", error or "schedule_failed"

    when = (getattr(row, "raw_time_text", None) or raw).strip()
    session["status"] = "callback_scheduled"
    session["callback_raw_time"] = when
    save_session(session)
    return True, when, ""


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
    if session.get("agent_released") or session.get("conference_ended"):
        # The patient's call is over: never re-open it or ring another agent.
        _hangup_call(str(session.get("dial_call_sid") or "").strip())
        return False
    dial_sid = str(session.get("dial_call_sid") or "").strip()
    call_sid = (call_sid or "").strip()
    # Only react to the agent (dial) leg, not the patient inbound leg.
    if not dial_sid:
        return False
    if call_sid and call_sid != dial_sid:
        return False
    if session.get("spoke_now") or session.get("status") == "connected":
        return False
    if session.get("callback_offer_started") or session.get("status") in {
        "agent_unreachable_ask_callback",
        "callback_scheduled",
    }:
        return True

    numbers = [
        str(n).strip()
        for n in (
            session.get("transfer_numbers") or session.get("live_agent_numbers") or []
        )
        if str(n).strip()
    ]
    # Single agent (or empty list): no failover target — ask for callback.
    if len(numbers) < 2:
        session["status"] = "agent_unreachable"
        save_session(session)
        return offer_agent_unreachable_callback(session)

    if failover_to_next_live_agent(session, reason=status):
        return True
    return offer_agent_unreachable_callback(session)


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
    from apps.ai_caller.models import Call

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
        return

    try:
        set_transfer_call_status(session, Call.Status.CONNECT_LIVE_AGENT)
    except Exception:
        logger.exception(
            "Failed to set connect_live_agent status session=%s", session_id
        )


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
    # Twilio sends "participant-join" / "conference-end"; accept the short forms too.
    event = {
        "participant-join": "join",
        "participant-leave": "leave",
        "conference-start": "start",
        "conference-end": "end",
    }.get(event, event)
    conference_sid = (payload.get("ConferenceSid") or "").strip()
    join_call = (payload.get("CallSid") or "").strip()
    if conference_sid:
        session["conference_sid"] = conference_sid
        save_session(session)
    # Conference over → live-agent talk ended; extend full-call duration.
    latest = load_session(str(session.get("session_id") or "")) or session
    if event == "end" and (
        latest.get("hold_offer_in_progress")
        or (conference_sid and conference_sid == latest.get("stepped_out_conference_sid"))
    ):
        return  # the hold room they left to answer the callback offer
    if event == "end":
        extend_call_ended_at(session)
        release_agent_slot(session)
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
        # A waiting patient hears the hold message from the waitUrl (repeating).
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
    elif patient and provider and not dual_ok and session.get("mix_failed"):
        # Mix diarization failed — per-leg text is the best we will get.
        mix = ""
        mix_items = None
        wt_logger.info(
            "TRANSFER_PERSIST_LEGS_AFTER_MIX_FAIL retell_call_id=%s",
            retell_call_id,
        )
    elif patient and provider and not dual_ok:
        # Mono legs only — do not persist a fused interleave; wait for mix.
        wt_logger.info(
            "TRANSFER_PERSIST_WAIT_MIX retell_call_id=%s reason=mono_legs_only",
            retell_call_id,
        )
        return
    elif not patient and not provider:
        if mix_items:
            # Diarization produced structured, timestamped turns but didn't
            # confidently split into patient + live_agent (e.g. only one
            # voice cluster detected). Keep those turns intact instead of
            # flattening everything into a single unlabeled wall of text —
            # a partially-labeled transcript is still better than one blob.
            mix = ""
        else:
            mix = str(session.get("humans_transcript") or "")
            mix_items = None
    elif provider and not patient:
        # Provider-only is safe (live agent inbound). Persist it.
        mix = ""
        mix_items = None
        patient_segments = []
    elif session.get("mix_failed"):
        # Patient-only mono and the mix could not be diarized — keep the text.
        mix = ""
        mix_items = None
        provider_segments = []
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
    live_agent_recording_url = _human_recording_url(session)
    twilio_call_sid = str(
        session.get("call_sid") or session.get("dial_call_sid") or ""
    ).strip()
    patient_offset = _recording_offset(session, "patient")
    provider_offset = _recording_offset(session, "provider")
    if dual_ok:
        # Both speakers came from the patient recording (left/right channel or
        # diarized), so their times share the patient recording's clock.
        provider_offset = patient_offset
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
        session = patch_session(
            session_id,
            {
                "mix_saved": True,
                "mix_failed": True,
                "mix_recording_url": recording_url,
                "mix_recording_sid": recording_sid,
                **({"mix_recording_s3_url": s3_url} if s3_url else {}),
            },
        )
        wt_logger.error(
            "TRANSFER_FAIL session=%s leg=mix reason=no_diarization",
            session_id,
        )
        # Per-leg transcripts were held back waiting for this mix; save them now
        # so the live-agent conversation is not lost.
        if session.get("patient_transcript") or session.get("provider_transcript"):
            _persist_to_db(session)
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
    # Mono patient+provider STT is not a reliable split — wait for mix diarize
    # (unless the mix already failed; then the legs are all we have).
    if both_ready and not dual_ok and not session.get("mix_failed"):
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


def _stt_context(session: Dict[str, Any]) -> str:
    """Whisper prompt with the real names so they are spelled right."""
    name = str((session or {}).get("name") or "").strip()
    service = str((session or {}).get("service_name") or "").strip()
    who = f"{name}, a patient," if name and name != "there" else "a patient"
    about = f" about a {service}" if service else ""
    return f"A phone call between {who} and a Bay Area Community Health care coordinator{about}."


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
            left_audio or audio, filename="patient.wav", context=_stt_context(session)
        )
        provider_detailed = transcribe_recording_detailed(
            right_audio, filename="provider_from_dual.wav", context=_stt_context(session)
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
        detailed = transcribe_recording_detailed(
            audio, filename="patient.wav", context=_stt_context(session)
        )
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
            transcribe_recording_detailed(
                audio, filename=f"{speaker}.wav", context=_stt_context(session)
            )
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
    """Put the Twilio number back only after every transfer has finished.

    Restoring earlier points the next patient at another patient's webhook.
    """
    session = dict(session or {})
    if session.get("voice_url_restored"):
        return
    this_id = str(session.get("session_id") or "").strip()
    for other in _open_sessions():
        other_id = str(other.get("session_id") or "")
        if other_id == this_id:
            continue
        if other.get("conference_ended") or other.get("agent_released"):
            continue
        return
    sid = str(session.get("incoming_sid") or "").strip()
    if not sid:
        return
    client = _twilio_client()
    if client is None:
        return
    original = {}
    try:
        original = _session_cache().get(_ORIGINAL_VOICE_KEY) or {}
    except Exception:
        original = {}
    if not isinstance(original, dict):
        original = {}
    previous = original.get("voice_url")
    if previous is None:
        previous = session.get("previous_voice_url")
    if isinstance(previous, str) and "/warm-transfer/inbound" in previous:
        previous = ""
    method = original.get("voice_method") or session.get("previous_voice_method") or "POST"
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
    inbound_url = shared_inbound_url()
    current_url = getattr(incoming, "voice_url", None) or ""
    current_method = getattr(incoming, "voice_method", None) or "POST"
    original = {}
    try:
        original = _session_cache().get(_ORIGINAL_VOICE_KEY) or {}
    except Exception:
        original = {}
    if not isinstance(original, dict):
        original = {}
    # Never treat a warm-transfer webhook (any tunnel) as the "original"
    # production voice URL — restoring that later causes Twilio app errors.
    current_is_wt = "/twilio/warm-transfer/inbound" in str(current_url)
    needs_voice_update = (
        str(current_url).rstrip("/") != inbound_url.rstrip("/")
        or str(current_method).upper() != "POST"
    )
    if needs_voice_update:
        if not original and current_url and not current_is_wt:
            original = {"voice_url": current_url, "voice_method": current_method}
            try:
                _session_cache().set(
                    _ORIGINAL_VOICE_KEY, original, timeout=_SESSION_TTL_SECONDS
                )
            except Exception:
                logger.exception("Failed to remember original Twilio voice URL")
        try:
            incoming.update(voice_url=inbound_url, voice_method="POST")
            logger.info(
                "Twilio voice URL refreshed to current BACKEND_URL inbound=%s (was %s)",
                inbound_url,
                current_url,
            )
        except Exception as exc:
            logger.exception("Failed to point Twilio number at warm-transfer inbound")
            return {
                "ok": False,
                "error": f"Failed to prepare Twilio recorder: {exc}",
                "status_code": 502,
            }
    previous_url = original.get("voice_url") or (
        "" if current_is_wt else current_url
    )
    if "/warm-transfer/inbound" in str(previous_url):
        previous_url = ""
    previous_method = original.get("voice_method") or (
        current_method if not current_is_wt else "POST"
    )

    numbers = [
        str(n).strip()
        for n in (transfer_numbers or ([transfer_number] if transfer_number else []) or [])
        if str(n).strip()
    ]
    primary = numbers[0] if numbers else str(transfer_number or "").strip()

    session_data: Dict[str, Any] = {
        "session_id": session_id,
        "created_at": time.time(),
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
    session_data["queue_scope"] = _queue_scope(session_data)
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
