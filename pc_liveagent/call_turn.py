"""Server-owned Collect Mode turn controller.

Pipeline: receive → normalize → validate → Playwright (serialized) → verify →
commit → return exact next speech for Retell.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import config
from .answer_parse import (
    extract_email,
    is_email_choice,
    is_fast_path_resolve,
    is_google_choice,
    is_language_ack,
    normalize_utterance,
    resolve_answer,
    resolve_with_claude,
    sanitize_person_name,
)
from .control_intent import detect_control_intent
from .elig_meta import format_question, get_field_meta
from .live_fill_prompt import LIVE_FILL_GREETING
from .live_form import (
    ELIG_QUESTIONS,
    _age_option_labels,
    _child_instance_id,
    correct_name_field,
    get_session,
    navigate_back_one,
    peek_choice_labels,
    promote_signed_in_to_eligibility,
    skip_current_optional,
    submit_field,
)

# Steps where Playwright often takes seconds — ack then deliver next via interrupt.
_SLOW_ELIG_STEPS = frozenset(
    {
        "marital",
        "children",
        "child_first",
        "child_last",
        "child_rel",
        "child_age",
        "child_more",
        "student",
        "disabled",
        "relatives",
        "support",
        "spouse_first",
        "spouse_last",
        "spouse_role",
        "spouse_age",
        "person_first",
        "person_last",
        "person_rel",
        "person_age",
        "income_when",
        "income_range",
        "special",
        "special_person",
        "programs",
    }
)
_elig_browser_tasks: dict[str, asyncio.Task] = {}
ELIG_PROGRESS_ACK = "One moment."

logger = logging.getLogger(__name__)

SIGN_IN_QUESTION = "How would you like to sign in, with Google or with email?"
EMAIL_QUESTION = "What email address should I use?"
GOOGLE_EMAIL_QUESTION = "Which Google email should I use?"
SSO_ACK = "One moment while I sign you in with Google."
SSO_BUSY = "One moment while I finish signing you in."
CLOSING = (
    "You're all set for household eligibility. Pointcare will follow up if they "
    "need anything. Thank you for your time. Goodbye."
)
# Stage E (application) reserved — after programs we close for now.
# When enabled, start_stage_e() should drive application_answer fills.
STAGE_E_ENABLED = False

_PENDING_SIGNIN_RE = re.compile(
    r"complete the visible google|say google again|has not finished|"
    r"sign-in page is not ready|which google email",
    re.I,
)
_signin_watch_tasks: dict[str, asyncio.Task] = {}
_sso_tasks: dict[str, asyncio.Task] = {}


async def start_stage_e(call_id: str, form_url: str = "") -> TurnResult:
    """Hook for Stage E application workflow after household eligibility."""
    return TurnResult(
        speech=(
            "Your household eligibility answers are saved. The full application "
            "will continue in a later update. Thank you. Goodbye."
        ),
        end_call=True,
        phase="closing",
        detail="stage_e_not_enabled",
    )


@dataclass
class TurnResult:
    speech: str
    end_call: bool = False
    phase: str = ""
    turn_id: str = ""
    ok: bool = True
    detail: str = ""


@dataclass
class CallController:
    call_id: str
    phase: str = "language"  # language|sign_in|email|google_email|eligibility|closing
    turn_seq: int = 0
    # Dedupe key includes phase/step so consecutive Yes/No across questions apply.
    last_user_fp: str = ""
    last_dedupe_ctx: str = ""
    last_speech: str = LIVE_FILL_GREETING
    pending_op_id: str = ""
    committed: list[dict[str, Any]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: float = field(default_factory=time.time)
    # Post-SSO delivery guards (Custom LLM WebSocket).
    sso_in_progress: bool = False
    sso_ack_sent: bool = False
    eligibility_screen_verified: bool = False
    eligibility_question_sent: bool = False
    active_response_id: int | None = None
    active_websocket_status: str = "unknown"  # connected|disconnected|unknown
    interrupt_seq: int = 0
    # Pending correction: {"field": "first_name"|"last_name"|"age"|..., "prompt": str}
    pending_correction: dict[str, str] | None = None
    # Browser eligibility op in flight — ignore answers until it finishes.
    elig_op_in_progress: bool = False
    last_question_delivery_key: str = ""
    # Outbound speech that failed because the socket was down — retry on reconnect.
    pending_outbound_speech: str = ""
    pending_outbound_key: str = ""


_controllers: dict[str, CallController] = {}
_controllers_lock = asyncio.Lock()


async def get_controller(call_id: str) -> CallController:
    async with _controllers_lock:
        ctrl = _controllers.get(call_id)
        if ctrl is None:
            ctrl = CallController(call_id=call_id)
            _controllers[call_id] = ctrl
        return ctrl


async def drop_controller(call_id: str) -> None:
    for bucket in (_signin_watch_tasks, _sso_tasks, _elig_browser_tasks):
        task = bucket.pop(call_id, None)
        if task and not task.done():
            task.cancel()
    async with _controllers_lock:
        _controllers.pop(call_id, None)


def _correction_prompt(field: str) -> str:
    labels = {
        "first_name": "first name",
        "last_name": "last name",
        "name": "name",
        "age": "age",
        "relationship": "relationship",
        "answer": "answer",
    }
    nice = labels.get(field, field.replace("_", " ") or "value")
    return f"Of course. What should the {nice} be?"


def _step_for_correction_field(field: str, current_step: str) -> str:
    """Best target elig_step for a correction field given where we are."""
    if field in ("first_name", "name"):
        if current_step.startswith("spouse_"):
            return "spouse_first"
        if current_step.startswith("person_"):
            return "person_first"
        return "child_first"
    if field == "last_name":
        if current_step.startswith("spouse_"):
            return "spouse_last"
        if current_step.startswith("person_"):
            return "person_last"
        return "child_last"
    if field == "age":
        if current_step.startswith("spouse_"):
            return "spouse_age"
        if current_step.startswith("person_"):
            return "person_age"
        return "child_age"
    if field == "relationship":
        if current_step.startswith("person_"):
            return "person_rel"
        return "child_rel"
    return current_step


async def _deliver_question_once(call_id: str, speech: str, *, key: str) -> bool:
    """Speak via agent_interrupt at most once per delivery key."""
    text = (speech or "").strip()
    if not text:
        return False
    ctrl = await get_controller(call_id)
    if ctrl.last_question_delivery_key == key:
        logger.info(
            "question_delivery skipped duplicate call_id=%s key=%s", call_id, key
        )
        return False
    ok = await _speak_interrupt(call_id, text)
    if ok:
        ctrl.last_question_delivery_key = key
        ctrl.last_speech = text
        ctrl.pending_outbound_speech = ""
        ctrl.pending_outbound_key = ""
        logger.info(
            "question_delivery_outcome call_id=%s key=%s outcome=ws_sent",
            call_id,
            key,
        )
    else:
        # Preserve verified next question for reconnect recovery — do not mark spoken.
        ctrl.pending_outbound_speech = text
        ctrl.pending_outbound_key = key
        ctrl.last_speech = text
        logger.warning(
            "question_delivery_outcome call_id=%s key=%s outcome=ws_failed "
            "queued_for_recovery=1",
            call_id,
            key,
        )
    return ok


async def recover_pending_outbound(call_id: str) -> bool:
    """After WebSocket reconnect, deliver any queued next-question once."""
    ctrl = await get_controller(call_id)
    speech = (ctrl.pending_outbound_speech or "").strip()
    key = (ctrl.pending_outbound_key or "").strip()
    if not speech or not key:
        return False
    if ctrl.active_websocket_status != "connected":
        return False
    logger.info(
        "question_delivery_recovery call_id=%s key=%s speech=%r",
        call_id,
        key,
        speech[:80],
    )
    return await _deliver_question_once(call_id, speech, key=key)


async def _run_eligibility_browser_background(
    call_id: str,
    *,
    answer: str,
    step: str,
    turn_id: str,
    form_url: str,
    commit_key: str,
) -> None:
    """Playwright eligibility apply off the WebSocket response path."""
    ctrl = await get_controller(call_id)
    logger.info(
        "elig_browser_started call_id=%s step=%s turn_id=%s", call_id, step, turn_id
    )
    try:
        result = await submit_field(
            call_id,
            "eligibility_answer",
            answer,
            url=form_url or config.POINTCARE_URL,
            request_id=f"ws-bg:{turn_id}",
        )
        spoken = _strip_ask(str(result.get("spoken") or result.get("message") or ""))
        session = await get_session(call_id)
        next_step = str(result.get("step") or (session.elig_step if session else "") or "")
        if not result.get("ok"):
            # Prefer live session step (special_person stays special_person on clarify).
            live_step = (
                str(result.get("step") or "")
                or (session.elig_step if session else "")
                or step
            )
            speech = spoken or format_question(live_step)
            ctrl.last_speech = speech
            logger.info(
                "elig_browser_completed call_id=%s step=%s live_step=%s outcome=failed",
                call_id,
                step,
                live_step,
            )
            await _deliver_question_once(
                call_id, speech, key=f"fail:{live_step}:{turn_id}"
            )
            return

        ctrl.committed.append(
            {
                "key": commit_key,
                "turn_id": turn_id,
                "step": step,
                "answer": answer,
                "at": time.time(),
            }
        )
        if not next_step:
            speech = spoken if spoken and "all set" in spoken.lower() else CLOSING
            ctrl.phase = "closing"
            ctrl.last_speech = speech
            await _deliver_question_once(
                call_id, speech, key=f"close:{turn_id}"
            )
            logger.info(
                "elig_browser_completed call_id=%s step=%s outcome=closing",
                call_id,
                step,
            )
            return

        speech = spoken or format_question(next_step)
        ctrl.last_speech = speech
        logger.info(
            "elig_browser_completed call_id=%s step=%s next=%s browser_verified=1",
            call_id,
            step,
            next_step,
        )
        await _deliver_question_once(
            call_id, speech, key=f"next:{next_step}:{turn_id}"
        )
    except asyncio.CancelledError:
        logger.info("elig_browser_cancelled call_id=%s", call_id)
        raise
    except Exception:
        logger.exception("elig_browser_failed call_id=%s step=%s", call_id, step)
        try:
            await _deliver_question_once(
                call_id,
                format_question(step),
                key=f"err:{step}:{turn_id}",
            )
        except Exception:
            pass
    finally:
        async with ctrl.lock:
            ctrl.elig_op_in_progress = False
            ctrl.pending_op_id = ""
        task = _elig_browser_tasks.get(call_id)
        if task is asyncio.current_task():
            _elig_browser_tasks.pop(call_id, None)


def _schedule_eligibility_browser(
    call_id: str,
    *,
    answer: str,
    step: str,
    turn_id: str,
    form_url: str,
    commit_key: str,
) -> None:
    prev = _elig_browser_tasks.get(call_id)
    if prev and not prev.done():
        logger.info("elig_browser already running call_id=%s", call_id)
        return
    _elig_browser_tasks[call_id] = asyncio.create_task(
        _run_eligibility_browser_background(
            call_id,
            answer=answer,
            step=step,
            turn_id=turn_id,
            form_url=form_url,
            commit_key=commit_key,
        ),
        name=f"elig-browser-{call_id}",
    )



def _user_fingerprint(text: str) -> str:
    return normalize_utterance(text).lower()[:80]


def _strip_ask(spoken: str) -> str:
    text = (spoken or "").strip()
    if text.startswith("ASK:"):
        text = text[4:].strip()
    return text


async def begin_speech(call_id: str) -> str:
    """Return the opening line, or resume the current question after WS reconnect.

    Retell auto_reconnect opens a new socket mid-call. Resetting to language here
    caused a second sign-in ask while the browser was already on eligibility.
    """
    async with _controllers_lock:
        existing = _controllers.get(call_id)
        if existing is not None and (
            existing.turn_seq > 0 or existing.phase not in ("", "language")
        ):
            speech = existing.last_speech or LIVE_FILL_GREETING
            logger.info(
                "WS resume call_id=%s phase=%s turn_seq=%s speech=%r",
                call_id,
                existing.phase,
                existing.turn_seq,
                speech[:80],
            )
            return speech
        ctrl = existing or CallController(call_id=call_id)
        _controllers[call_id] = ctrl
    ctrl.phase = "language"
    ctrl.last_speech = LIVE_FILL_GREETING
    return LIVE_FILL_GREETING


def _eligibility_question_for_session(session: Any) -> str:
    step = (session.elig_step if session else "") or "marital"
    return ELIG_QUESTIONS.get(step) or ELIG_QUESTIONS.get("marital", "")


async def sync_controller_to_eligibility(call_id: str) -> str | None:
    """If Sign-in is done in the browser, move Collect Mode to Stage D speech."""
    ctrl = await get_controller(call_id)
    if ctrl.sso_in_progress:
        return None
    if ctrl.phase == "eligibility":
        session = await get_session(call_id)
        if session and session.elig_step:
            ctrl.eligibility_screen_verified = True
            speech = _eligibility_question_for_session(session)
            if speech and speech != ctrl.last_speech:
                ctrl.last_speech = speech
            return ctrl.last_speech or speech
        return ctrl.last_speech or None

    speech = await promote_signed_in_to_eligibility(call_id)
    if not speech:
        session = await get_session(call_id)
        if session and session.signed_in and session.elig_step:
            speech = _eligibility_question_for_session(session)
    if not speech:
        return None
    ctrl.phase = "eligibility"
    ctrl.pending_op_id = ""
    ctrl.eligibility_screen_verified = True
    ctrl.last_speech = speech
    logger.info(
        "browser_screen_verified call_id=%s speech=%r",
        call_id,
        speech[:80],
    )
    return speech


def schedule_signin_watch(call_id: str) -> None:
    """Backup poll when SSO is pending outside the background task (e.g. chooser)."""
    prev = _signin_watch_tasks.get(call_id)
    if prev and not prev.done():
        return
    _signin_watch_tasks[call_id] = asyncio.create_task(
        _watch_signin_then_speak(call_id),
        name=f"signin-watch-{call_id}",
    )


def _schedule_google_sso(
    call_id: str,
    *,
    email: str = "",
    form_url: str = "",
    turn_id: str = "",
) -> None:
    """Run Playwright Google SSO off the WebSocket response path."""
    prev = _sso_tasks.get(call_id)
    if prev and not prev.done():
        logger.info("SSO already running call_id=%s", call_id)
        return
    _sso_tasks[call_id] = asyncio.create_task(
        _run_google_sso_background(
            call_id, email=email, form_url=form_url, turn_id=turn_id
        ),
        name=f"google-sso-{call_id}",
    )
    logger.info("SSO started call_id=%s turn_id=%s", call_id, turn_id)


async def _speak_interrupt(call_id: str, content: str) -> bool:
    """Send agent_interrupt on the active Custom LLM socket only."""
    text = (content or "").strip()
    if not text:
        return False
    ctrl = await get_controller(call_id)
    if ctrl.active_websocket_status == "disconnected":
        logger.warning(
            "interrupt skipped socket_disconnected call_id=%s speech=%r",
            call_id,
            text[:80],
        )
        return False
    ctrl.interrupt_seq += 1
    interrupt_id = ctrl.interrupt_seq
    try:
        from .retell_llm_ws import speak_now, websocket_connected

        if not websocket_connected(call_id):
            ctrl.active_websocket_status = "disconnected"
            logger.warning(
                "interrupt skipped no_active_ws call_id=%s speech=%r",
                call_id,
                text[:80],
            )
            return False
        ok = await speak_now(call_id, text, interrupt_id=interrupt_id)
        logger.info(
            "interrupt_sent call_id=%s interrupt_id=%s outcome=%s speech=%r",
            call_id,
            interrupt_id,
            "ok" if ok else "failed",
            text[:80],
        )
        return ok
    except Exception:
        logger.exception("interrupt failed call_id=%s", call_id)
        return False


async def deliver_first_eligibility_question(
    call_id: str,
    *,
    speech: str | None = None,
) -> bool:
    """Speak the first Stage D question once via agent_interrupt after screen verify.

    A successful WebSocket send is logged as delivery_attempted — not as confirmed
    audio heard by the caller.
    """
    ctrl = await get_controller(call_id)
    if ctrl.eligibility_question_sent:
        logger.info("eligibility_question skipped already_sent call_id=%s", call_id)
        return False
    if ctrl.sso_in_progress:
        logger.info("eligibility_question deferred sso_in_progress call_id=%s", call_id)
        return False

    text = (speech or "").strip()
    if not ctrl.eligibility_screen_verified:
        promoted = await sync_controller_to_eligibility(call_id)
        if not promoted and not ctrl.eligibility_screen_verified:
            session = await get_session(call_id)
            if session and session.elig_step:
                ctrl.eligibility_screen_verified = True
                promoted = _eligibility_question_for_session(session)
            else:
                logger.info(
                    "eligibility_question deferred screen_unverified call_id=%s",
                    call_id,
                )
                return False
        text = text or (promoted or "")

    if not text:
        text = ctrl.last_speech or ELIG_QUESTIONS.get("marital", "")
    if not text:
        return False

    ctrl.phase = "eligibility"
    ctrl.last_speech = text
    logger.info(
        "eligibility_question_delivery_attempt call_id=%s speech=%r",
        call_id,
        text[:80],
    )
    ok = await _speak_interrupt(call_id, text)
    if ok:
        ctrl.eligibility_question_sent = True
        logger.info(
            "eligibility_question_delivery_outcome call_id=%s outcome=ws_sent "
            "(not confirmation caller heard audio)",
            call_id,
        )
    else:
        logger.warning(
            "eligibility_question_delivery_outcome call_id=%s outcome=ws_failed",
            call_id,
        )
    return ok


# Back-compat alias used by reconnect / older call sites.
async def announce_eligibility(call_id: str, speech: str) -> bool:
    return await deliver_first_eligibility_question(call_id, speech=speech)


async def _run_google_sso_background(
    call_id: str,
    *,
    email: str = "",
    form_url: str = "",
    turn_id: str = "",
) -> None:
    """Playwright Google SSO; never blocks the Custom LLM receive loop."""
    ctrl = await get_controller(call_id)
    try:
        # Browser work stays off the WebSocket turn; do not hold ctrl.lock here.
        result = await submit_field(
            call_id,
            "sign_in_google",
            email or "",
            url=form_url or config.POINTCARE_URL,
            request_id=f"ws-sso:{turn_id or uuid.uuid4().hex[:8]}",
        )
        spoken = _strip_ask(str(result.get("spoken") or result.get("message") or ""))
        lowered = spoken.lower()

        if not result.get("ok"):
            async with ctrl.lock:
                ctrl.sso_in_progress = False
                ctrl.last_speech = spoken or SIGN_IN_QUESTION
            logger.info(
                "SSO completed call_id=%s outcome=failed detail=%s",
                call_id,
                str(result.get("error") or "")[:120],
            )
            if _should_watch_signin(spoken):
                schedule_signin_watch(call_id)
            await _speak_interrupt(call_id, spoken or SIGN_IN_QUESTION)
            return

        if "which google" in lowered or "which email" in lowered:
            async with ctrl.lock:
                ctrl.sso_in_progress = False
                ctrl.phase = "google_email"
                ctrl.last_speech = spoken or GOOGLE_EMAIL_QUESTION
            logger.info("SSO completed call_id=%s outcome=need_google_email", call_id)
            schedule_signin_watch(call_id)
            await _speak_interrupt(call_id, spoken or GOOGLE_EMAIL_QUESTION)
            return

        session = await get_session(call_id)
        speech = spoken or ELIG_QUESTIONS.get("marital", "")
        if session and session.elig_step:
            speech = _eligibility_question_for_session(session)
            logger.info(
                "browser_screen_verified call_id=%s step=%s",
                call_id,
                session.elig_step,
            )
        async with ctrl.lock:
            ctrl.sso_in_progress = False
            ctrl.phase = "eligibility"
            ctrl.pending_op_id = ""
            ctrl.eligibility_screen_verified = True
            ctrl.committed.append({"op": "sign_in_google", "turn_id": turn_id})
            ctrl.last_speech = speech
        logger.info(
            "SSO completed call_id=%s outcome=ok browser_screen_verified=1",
            call_id,
        )
        await deliver_first_eligibility_question(call_id, speech=speech)
    except asyncio.CancelledError:
        async with ctrl.lock:
            ctrl.sso_in_progress = False
        logger.info("SSO cancelled call_id=%s", call_id)
        raise
    except Exception:
        async with ctrl.lock:
            ctrl.sso_in_progress = False
        logger.exception("SSO background failed call_id=%s", call_id)
        try:
            await _speak_interrupt(call_id, SIGN_IN_QUESTION)
        except Exception:
            pass
    finally:
        task = _sso_tasks.get(call_id)
        if task is asyncio.current_task():
            _sso_tasks.pop(call_id, None)


async def _watch_signin_then_speak(call_id: str) -> None:
    """Backup: wait for browser eligibility, then interrupt once (no HTTP nudge)."""
    try:
        for _ in range(90):
            ctrl = await get_controller(call_id)
            if ctrl.eligibility_question_sent:
                return
            if ctrl.sso_in_progress:
                await asyncio.sleep(0.5)
                continue
            speech = await sync_controller_to_eligibility(call_id)
            ctrl = await get_controller(call_id)
            if ctrl.eligibility_screen_verified or ctrl.phase == "eligibility" or speech:
                await deliver_first_eligibility_question(
                    call_id, speech=speech or ctrl.last_speech
                )
                return
            await asyncio.sleep(1.0)
        logger.warning("Sign-in watch timed out call_id=%s", call_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Sign-in watch failed call_id=%s", call_id)
    finally:
        task = _signin_watch_tasks.get(call_id)
        if task is asyncio.current_task():
            _signin_watch_tasks.pop(call_id, None)


def _should_watch_signin(spoken: str) -> bool:
    return bool(_PENDING_SIGNIN_RE.search(spoken or ""))

async def process_user_turn(
    call_id: str,
    utterance: str,
    *,
    response_id: int | None = None,
    form_url: str = "",
) -> TurnResult:
    """Primary answer pipeline for Custom LLM WebSocket (and optional tool path)."""
    ctrl = await get_controller(call_id)
    async with ctrl.lock:
        ctrl.turn_seq += 1
        turn_id = f"{call_id}:{ctrl.turn_seq}:{uuid.uuid4().hex[:8]}"
        text = normalize_utterance(utterance)
        fp = _user_fingerprint(text)
        logger.info(
            "Turn start call_id=%s turn_id=%s phase=%s response_id=%s utterance=%r",
            call_id,
            turn_id,
            ctrl.phase,
            response_id,
            text[:80],
        )

        if response_id is not None:
            ctrl.active_response_id = response_id

        if ctrl.sso_in_progress:
            # Do not treat caller speech as an eligibility answer during SSO.
            speech = SSO_BUSY if text else SSO_ACK
            ctrl.last_speech = speech
            return TurnResult(
                speech=speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="sso_in_progress",
            )

        if not text:
            # Silence / reminder path: if SSO finished, start Stage D automatically.
            if ctrl.phase in ("sign_in", "google_email", "language"):
                promoted = await sync_controller_to_eligibility(call_id)
                if promoted:
                    ctrl.eligibility_screen_verified = True
                    # Prefer interrupt delivery; response content stays empty if sent.
                    if not ctrl.eligibility_question_sent:
                        await deliver_first_eligibility_question(
                            call_id, speech=promoted
                        )
                        return TurnResult(
                            speech="",
                            phase="eligibility",
                            turn_id=turn_id,
                            detail="auto_start_after_signin_interrupt",
                        )
                    return TurnResult(
                        speech=promoted,
                        phase="eligibility",
                        turn_id=turn_id,
                        detail="auto_start_after_signin",
                    )
            return TurnResult(
                speech=ctrl.last_speech or "Could you say that again?",
                phase=ctrl.phase,
                turn_id=turn_id,
            )

        # If the browser already finished Google SSO, never re-ask sign-in.
        if ctrl.phase in ("sign_in", "google_email", "language"):
            promoted = await sync_controller_to_eligibility(call_id)
            if promoted:
                session_now = await get_session(call_id)
                step_now = (session_now.elig_step if session_now else "") or "marital"
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = f"eligibility:{step_now}"
                ctrl.phase = "eligibility"
                ctrl.eligibility_screen_verified = True
                ctrl.last_speech = promoted
                # Spoken on this response turn — mark sent so interrupt does not double.
                ctrl.eligibility_question_sent = True
                return TurnResult(
                    speech=promoted,
                    phase="eligibility",
                    turn_id=turn_id,
                    detail="auto_start_after_signin",
                )

        # Dedupe only within the same phase+step context (not global Yes/No).
        elig_step = ""
        if ctrl.phase == "eligibility":
            session0 = await get_session(call_id)
            elig_step = (session0.elig_step if session0 else "") or "marital"
        dedupe_ctx = f"{ctrl.phase}:{elig_step}"
        if (
            fp
            and fp == ctrl.last_user_fp
            and dedupe_ctx == ctrl.last_dedupe_ctx
            and ctrl.phase not in ("closing",)
        ):
            return TurnResult(
                speech=ctrl.last_speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="duplicate_utterance",
            )

        if ctrl.phase == "language":
            if not is_language_ack(text):
                # Still treat most early replies as continue-in-English.
                if len(text) > 40 and not is_language_ack(text):
                    speech = (
                        "Is it okay to continue in English, or would you prefer "
                        "another language?"
                    )
                    ctrl.last_speech = speech
                    return TurnResult(speech=speech, phase=ctrl.phase, turn_id=turn_id)
            ctrl.last_user_fp = fp
            ctrl.last_dedupe_ctx = dedupe_ctx
            ctrl.phase = "sign_in"
            ctrl.last_speech = SIGN_IN_QUESTION
            return TurnResult(speech=SIGN_IN_QUESTION, phase=ctrl.phase, turn_id=turn_id)

        if ctrl.phase == "sign_in":
            if is_google_choice(text):
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                ctrl.pending_op_id = turn_id
                # Ack immediately on this response_id; Playwright SSO runs in background.
                ctrl.sso_in_progress = True
                ctrl.sso_ack_sent = True
                ctrl.eligibility_question_sent = False
                ctrl.eligibility_screen_verified = False
                ctrl.last_speech = SSO_ACK
                logger.info(
                    "acknowledgment_sent call_id=%s response_id=%s speech=%r",
                    call_id,
                    response_id,
                    SSO_ACK,
                )
                _schedule_google_sso(
                    call_id,
                    email="",
                    form_url=form_url or config.POINTCARE_URL,
                    turn_id=turn_id,
                )
                return TurnResult(
                    speech=SSO_ACK,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="sso_ack",
                )

            if is_email_choice(text):
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                ctrl.phase = "email"
                ctrl.last_speech = EMAIL_QUESTION
                return TurnResult(speech=EMAIL_QUESTION, phase=ctrl.phase, turn_id=turn_id)

            # Keep the full sign-in question (not a clipped "say Google or email").
            ctrl.last_speech = SIGN_IN_QUESTION
            return TurnResult(
                speech=SIGN_IN_QUESTION,
                phase=ctrl.phase,
                turn_id=turn_id,
            )

        if ctrl.phase == "email":
            email = extract_email(text)
            if not email:
                return TurnResult(
                    speech="What email address should I use?",
                    phase=ctrl.phase,
                    turn_id=turn_id,
                )
            ctrl.last_user_fp = fp
            ctrl.last_dedupe_ctx = dedupe_ctx
            result = await submit_field(
                call_id,
                "email",
                email,
                url=form_url or config.POINTCARE_URL,
                request_id=f"ws:{turn_id}",
            )
            spoken = _strip_ask(str(result.get("spoken") or ""))
            ctrl.phase = "closing"
            ctrl.last_speech = spoken or "Your email was entered. Goodbye."
            return TurnResult(
                speech=ctrl.last_speech,
                end_call=True,
                phase=ctrl.phase,
                turn_id=turn_id,
            )

        if ctrl.phase == "google_email":
            email = extract_email(text)
            if not email:
                return TurnResult(
                    speech=GOOGLE_EMAIL_QUESTION,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                )
            ctrl.last_user_fp = fp
            ctrl.last_dedupe_ctx = dedupe_ctx
            ctrl.pending_op_id = turn_id
            ctrl.sso_in_progress = True
            ctrl.sso_ack_sent = True
            ctrl.eligibility_question_sent = False
            ctrl.eligibility_screen_verified = False
            ctrl.last_speech = SSO_BUSY
            logger.info(
                "acknowledgment_sent call_id=%s response_id=%s speech=%r",
                call_id,
                response_id,
                SSO_BUSY,
            )
            _schedule_google_sso(
                call_id,
                email=email,
                form_url=form_url or config.POINTCARE_URL,
                turn_id=turn_id,
            )
            return TurnResult(
                speech=SSO_BUSY,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="sso_ack",
            )

        if ctrl.phase == "eligibility":
            if ctrl.sso_ack_sent and not ctrl.eligibility_question_sent:
                speech = ctrl.last_speech or format_question("marital")
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="eligibility_question_not_active",
                )
            if ctrl.elig_op_in_progress:
                return TurnResult(
                    speech=ELIG_PROGRESS_ACK,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="elig_op_in_progress",
                )

            session = await get_session(call_id)
            # Prefer live browser step when available so resume stays synced.
            step = (session.elig_step if session else "") or elig_step or "marital"
            meta = get_field_meta(step)

            # Pending correction consumes the next utterance as the replacement value.
            if ctrl.pending_correction and text:
                intent_peek = detect_control_intent(text)
                if intent_peek.kind == "repeat":
                    prompt = (
                        ctrl.pending_correction.get("prompt")
                        or _correction_prompt(
                            ctrl.pending_correction.get("field") or "first_name"
                        )
                    )
                    ctrl.last_speech = prompt
                    return TurnResult(
                        speech=prompt,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="correction_prompt_repeat",
                    )
                if intent_peek.kind == "change" and not intent_peek.value:
                    field = intent_peek.target or ctrl.pending_correction.get(
                        "field"
                    ) or "first_name"
                    prompt = _correction_prompt(field)
                    ctrl.pending_correction = {"field": field, "prompt": prompt}
                    ctrl.last_speech = prompt
                    return TurnResult(
                        speech=prompt,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="correction_prompt",
                    )
                if intent_peek.kind in ("go_back", "skip", "next"):
                    ctrl.pending_correction = None
                    # Fall through to normal intent handling below.
                else:
                    field = ctrl.pending_correction.get("field") or "first_name"
                    if intent_peek.kind == "change" and intent_peek.value:
                        field = intent_peek.target or field
                        value = intent_peek.value
                    else:
                        value = text
                    if field in ("first_name", "last_name", "name"):
                        value = sanitize_person_name(value) or value
                        ctrl.pending_correction = None
                        ctrl.last_user_fp = fp
                        ctrl.last_dedupe_ctx = dedupe_ctx
                        edit = await correct_name_field(call_id, field, value)
                        speech = _strip_ask(str(edit.get("spoken") or ""))
                        ctrl.last_speech = speech
                        if edit.get("ok"):
                            session = await get_session(call_id)
                            step = (
                                str(edit.get("step") or "")
                                or (session.elig_step if session else step)
                                or step
                            )
                        return TurnResult(
                            speech=speech or format_question(step),
                            phase=ctrl.phase,
                            turn_id=turn_id,
                            ok=bool(edit.get("ok")),
                            detail="name_corrected"
                            if edit.get("ok")
                            else "name_correct_failed",
                        )
                    ctrl.pending_correction = None
                    target_step = _step_for_correction_field(field, step)
                    # Non-name corrections may still need a verified Back.
                    for _ in range(4):
                        session = await get_session(call_id)
                        cur = (session.elig_step if session else "") or step
                        if cur == target_step:
                            step = cur
                            break
                        nav = await navigate_back_one(call_id)
                        if not nav.get("ok"):
                            break
                        step = str(nav.get("step") or target_step)
                    meta = get_field_meta(step)
                    text = value
                    # Fall through to normal resolve/submit for target_step.

            intent = detect_control_intent(text)
            if intent.kind == "repeat":
                labels = await peek_choice_labels(call_id)
                speech = format_question(step, options=labels or None)
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech, phase=ctrl.phase, turn_id=turn_id, detail="repeat"
                )
            if intent.kind == "next":
                if ctrl.elig_op_in_progress:
                    return TurnResult(
                        speech=ELIG_PROGRESS_ACK,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="next_waiting_browser",
                    )
                unanswered = not any(
                    item.get("step") == step for item in ctrl.committed[-8:]
                )
                session_now = await get_session(call_id)
                current = (session_now.elig_step if session_now else step) or step
                if unanswered and current == step:
                    labels = await peek_choice_labels(call_id)
                    speech = (
                        format_question(step, options=labels or None)
                        + " We still need this answer before moving on."
                    )
                    ctrl.last_speech = speech
                    return TurnResult(
                        speech=speech,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="next_blocked_unanswered",
                    )
                labels = await peek_choice_labels(call_id)
                speech = format_question(current, options=labels or None)
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech, phase=ctrl.phase, turn_id=turn_id, detail="next"
                )
            if intent.kind == "go_back":
                ctrl.pending_correction = None
                nav = await navigate_back_one(call_id)
                speech = _strip_ask(str(nav.get("spoken") or ""))
                if nav.get("ok"):
                    ctrl.last_speech = speech
                    return TurnResult(
                        speech=speech,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="go_back",
                    )
                ctrl.last_speech = speech or format_question(step)
                return TurnResult(
                    speech=ctrl.last_speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    ok=False,
                    detail="go_back_failed",
                )
            if intent.kind == "skip":
                skip_result = await skip_current_optional(call_id)
                speech = _strip_ask(str(skip_result.get("spoken") or ""))
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    ok=bool(skip_result.get("ok")),
                    detail="skip" if skip_result.get("ok") else "skip_blocked",
                )
            if intent.kind == "redo":
                # Re-answer the current step without inventing navigation.
                labels = await peek_choice_labels(call_id)
                speech = (
                    format_question(step, options=labels or None)
                    + " Go ahead and give the corrected answer."
                )
                ctrl.last_speech = speech
                # Drop commits for this step/instance so the new answer can apply.
                inst = _child_instance_id(session)
                ctrl.committed = [
                    item
                    for item in ctrl.committed
                    if not (
                        item.get("step") == step
                        and (
                            not step.startswith("child_")
                            or f":{inst}:" in str(item.get("key") or "")
                        )
                    )
                ]
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="redo_prompt",
                )
            if intent.kind == "change":
                field = intent.target or ""
                if not field:
                    prompt = (
                        "Which field should I change — first name, last name, or age?"
                    )
                    ctrl.pending_correction = {
                        "field": "first_name",
                        "prompt": prompt,
                        "need_field": True,
                    }
                    ctrl.last_speech = prompt
                    return TurnResult(
                        speech=prompt,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="correction_clarify_field",
                    )
                if field == "name":
                    field = "first_name"
                if not intent.value:
                    # Start a correction dialog — do not fall through to answer parse.
                    prompt = _correction_prompt(field)
                    ctrl.pending_correction = {"field": field, "prompt": prompt}
                    ctrl.last_speech = prompt
                    return TurnResult(
                        speech=prompt,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="correction_prompt",
                    )
                if field in ("first_name", "last_name", "name"):
                    value = sanitize_person_name(intent.value) or intent.value
                    ctrl.last_user_fp = fp
                    ctrl.last_dedupe_ctx = dedupe_ctx
                    edit = await correct_name_field(call_id, field, value)
                    speech = _strip_ask(str(edit.get("spoken") or ""))
                    ctrl.last_speech = speech
                    return TurnResult(
                        speech=speech or format_question(step),
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        ok=bool(edit.get("ok")),
                        detail="name_corrected"
                        if edit.get("ok")
                        else "name_correct_failed",
                    )
                # Non-name change with value: verified Back then apply.
                target_step = _step_for_correction_field(field, step)
                for _ in range(4):
                    session = await get_session(call_id)
                    cur = (session.elig_step if session else "") or step
                    if cur == target_step:
                        step = cur
                        break
                    nav = await navigate_back_one(call_id)
                    if not nav.get("ok"):
                        break
                    step = str(nav.get("step") or target_step)
                meta = get_field_meta(step)
                text = intent.value

            labels = await peek_choice_labels(call_id)
            # Age steps: never expose First Name / Spouse captions as choices.
            resolve_opts = labels or None
            if step in ("spouse_age", "child_age", "person_age"):
                resolve_opts = _age_option_labels(labels) or None
            resolved = resolve_answer(
                step, text, options=resolve_opts, meta=meta
            )
            answer, clarify = resolved.answer, resolved.clarify
            resolver_used = "python"
            if clarify == "__REPEAT__":
                speech = format_question(step, options=resolve_opts)
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech, phase=ctrl.phase, turn_id=turn_id, detail="repeat"
                )

            # Fast path: clear Yes/No, exact option, mapped age/income, names.
            # Otherwise ask Haiku for structured resolution (validated).
            if not is_fast_path_resolve(resolved):
                step_before = step
                seq_before = ctrl.turn_seq
                committed_before = len(ctrl.committed)
                member_ctx = ""
                try:
                    session_ctx = await get_session(call_id)
                    if session_ctx:
                        member_ctx = str(
                            session_ctx.filled.get("household_names")
                            or session_ctx.filled.get("special_person_assigned")
                            or ""
                        )
                except Exception:
                    member_ctx = ""
                answer_type = str(getattr(meta, "answer_type", None) or "text")
                claude_opts = resolve_opts
                claude = await resolve_with_claude(
                    step,
                    text,
                    question=format_question(step, options=claude_opts),
                    answer_type=answer_type,
                    options=claude_opts,
                    member_context=member_ctx,
                    turn_id=turn_id,
                )
                # Stale guard: ignore Claude if a newer turn committed, or this
                # step was already answered by another operation while we waited.
                session_now = await get_session(call_id)
                step_now = (session_now.elig_step if session_now else step) or step
                step_already_committed = any(
                    item.get("step") == step_before
                    and item.get("turn_id") != turn_id
                    for item in ctrl.committed[-8:]
                )
                superseded = (
                    ctrl.turn_seq != seq_before
                    or len(ctrl.committed) > committed_before
                    or step_already_committed
                    or (
                        ctrl.pending_op_id
                        and ctrl.pending_op_id != turn_id
                        and ctrl.elig_op_in_progress
                    )
                )
                if superseded:
                    logger.warning(
                        "claude_result_discarded_stale call_id=%s turn_id=%s "
                        "step=%s step_now=%s",
                        call_id,
                        turn_id,
                        step_before,
                        step_now,
                    )
                    answer = None
                    clarify = format_question(step_now, options=resolve_opts)
                elif claude.accepted and claude.answer:
                    answer = claude.answer
                    clarify = None
                    resolver_used = "claude"
                    logger.info(
                        "answer_resolver=claude call_id=%s step=%s confidence=%.2f "
                        "latency_ms=%.0f verified=1 answer=%r",
                        call_id,
                        step,
                        claude.confidence,
                        claude.latency_ms,
                        answer[:60],
                    )
                elif claude.needs_clarify or claude.reject_reason in (
                    "not_in_options",
                    "low_confidence",
                    "condition_not_person",
                    "timeout",
                    "error",
                ):
                    # Timeout/error: keep Python clarify; never invent options from
                    # unrelated form labels, and never click a low-confidence guess.
                    clarify = (
                        claude.clarify_prompt
                        or clarify
                        or format_question(step, options=resolve_opts)
                    )
                    if (
                        resolve_opts
                        and "Please choose" not in (clarify or "")
                        and "Options include" not in (clarify or "")
                    ):
                        shown = ", ".join(resolve_opts[:6])
                        clarify = f"{clarify} Options include: {shown}."
                    answer = None
                    logger.info(
                        "answer_resolver=claude call_id=%s step=%s verified=0 "
                        "reason=%s latency_ms=%.0f",
                        call_id,
                        step,
                        claude.reject_reason or "needs_clarify",
                        claude.latency_ms,
                    )
                # else: fall through to Python clarify / free-text answer

            if clarify and not answer:
                ctrl.last_speech = clarify
                return TurnResult(
                    speech=clarify,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail=resolved.detail or "clarify",
                )

            assert answer is not None
            if resolver_used == "python":
                logger.info(
                    "answer_resolver=python call_id=%s step=%s detail=%s answer=%r",
                    call_id,
                    step,
                    resolved.detail,
                    answer[:60],
                )
            # Idempotency: step + child-instance (when relevant) + answer.
            # Do not use bare step:answer — that blocks a second child saying
            # "Child" / "Seven" / "No" after a prior child used the same words.
            session_for_key = session or await get_session(call_id)
            if step.startswith("child_") or step in ("children", "child_more"):
                commit_key = (
                    f"{step}:{_child_instance_id(session_for_key)}:{answer.lower()}"
                )
            else:
                commit_key = f"{step}:{answer.lower()}"
            prior = [
                item
                for item in ctrl.committed[-12:]
                if item.get("key") == commit_key and item.get("step") == step
            ]
            if prior:
                session2 = await get_session(call_id)
                next_q = (session2.elig_step if session2 else "") or step
                # If we are still on the same step, the prior "commit" did not
                # produce a lasting transition — drop it and retry Playwright.
                if next_q == step:
                    ctrl.committed = [
                        item
                        for item in ctrl.committed
                        if not (
                            item.get("key") == commit_key and item.get("step") == step
                        )
                    ]
                    logger.warning(
                        "already_committed_retry call_id=%s step=%s key=%s",
                        call_id,
                        step,
                        commit_key,
                    )
                else:
                    # Truly advanced — speak the live next question.
                    speech = format_question(next_q) or ctrl.last_speech
                    return TurnResult(
                        speech=speech or ctrl.last_speech,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="already_committed",
                    )

            ctrl.last_user_fp = fp
            ctrl.last_dedupe_ctx = dedupe_ctx
            ctrl.pending_op_id = turn_id

            # Slow browser steps: ack immediately, apply in background, interrupt next Q.
            if step in _SLOW_ELIG_STEPS:
                ctrl.elig_op_in_progress = True
                _schedule_eligibility_browser(
                    call_id,
                    answer=answer,
                    step=step,
                    turn_id=turn_id,
                    form_url=form_url or config.POINTCARE_URL,
                    commit_key=commit_key,
                )
                ctrl.last_speech = ELIG_PROGRESS_ACK
                logger.info(
                    "elig_ack_sent call_id=%s response_id=%s step=%s",
                    call_id,
                    response_id,
                    step,
                )
                return TurnResult(
                    speech=ELIG_PROGRESS_ACK,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="elig_ack",
                )

            result = await submit_field(
                call_id,
                "eligibility_answer",
                answer,
                url=form_url or config.POINTCARE_URL,
                request_id=f"ws:{turn_id}",
            )
            ctrl.pending_op_id = ""
            spoken = _strip_ask(str(result.get("spoken") or result.get("message") or ""))
            if not result.get("ok"):
                session2 = await get_session(call_id)
                step2 = (session2.elig_step if session2 else step) or step
                speech = spoken or format_question(step2) or ctrl.last_speech
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    ok=False,
                    detail=str(result.get("error") or "")[:200],
                )

            ctrl.committed.append(
                {
                    "key": commit_key,
                    "turn_id": turn_id,
                    "step": step,
                    "answer": answer,
                    "at": time.time(),
                }
            )
            next_step = str(result.get("step") or "")
            if not next_step:
                if STAGE_E_ENABLED:
                    ctrl.phase = "closing"
                    stage_e = await start_stage_e(call_id, form_url)
                    ctrl.last_speech = stage_e.speech
                    return TurnResult(
                        speech=stage_e.speech,
                        end_call=stage_e.end_call,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="stage_e",
                    )
                ctrl.phase = "closing"
                speech = spoken if spoken and "all set" in spoken.lower() else CLOSING
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech,
                    end_call=True,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                )

            speech = spoken or format_question(next_step)
            ctrl.last_speech = speech
            ctrl.last_question_delivery_key = f"sync:{next_step}:{turn_id}"
            return TurnResult(speech=speech, phase=ctrl.phase, turn_id=turn_id)

        # closing / unknown
        return TurnResult(
            speech=ctrl.last_speech or CLOSING,
            end_call=True,
            phase=ctrl.phase,
            turn_id=turn_id,
        )


def latest_user_utterance(transcript: list[Any] | None) -> str:
    if not isinstance(transcript, list):
        return ""
    for item in reversed(transcript):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").lower()
        if role != "user":
            continue
        content = item.get("content")
        if isinstance(content, list):
            bits = []
            for part in content:
                if isinstance(part, dict):
                    bits.append(str(part.get("text") or part.get("word") or ""))
                else:
                    bits.append(str(part))
            text = " ".join(bits).strip()
        else:
            text = str(content or "").strip()
        if text and text != "(unintelligible audio)":
            return text
    return ""
