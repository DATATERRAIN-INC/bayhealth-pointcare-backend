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
    ResolveResult,
    extract_email,
    is_apple_choice,
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
from .call_trace import clear_call_trace, trace_event
from .live_form import (
    ELIG_QUESTIONS,
    EMAIL_SECURE_COMPLETE,
    EMAIL_SECURE_RETRY,
    _age_option_labels,
    _child_instance_id,
    _detect_income_period_switch,
    _is_no,
    _is_yes,
    _wants_income_period_reselect,
    correct_member_field,
    correct_name_field,
    get_session,
    navigate_back_one,
    peek_choice_labels,
    poll_email_auth_until_signed_in,
    promote_signed_in_to_eligibility,
    skip_current_optional,
    spoken_for_auth_screen,
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
        "child_confirm",
        "child_more",
        "student",
        "disabled",
        "relatives",
        "support",
        "spouse_first",
        "spouse_last",
        "spouse_role",
        "spouse_age",
        "spouse_confirm",
        "person_first",
        "person_last",
        "person_rel",
        "person_age",
        "person_confirm",
        "income_when",
        "income_range",
        "special",
        "special_person",
        "special_more",
        "programs",
        "proceed_application",
        "confirm_start_application",
    }
)
_elig_browser_tasks: dict[str, asyncio.Task] = {}
ELIG_PROGRESS_ACK = "One moment."

logger = logging.getLogger(__name__)

CALL_PURPOSE = (
    "I'm Kyle from Bay Area Community Health, calling to help you with "
    "Pointcare registration and check household eligibility."
)
SIGN_IN_QUESTION = (
    "How would you like to sign in, with Google, with Apple, or with email?"
)
EMAIL_QUESTION = "What email address should I use?"
EMAIL_CLARIFY = (
    "I didn't catch a full email address. Please say it slowly, like "
    "name at gmail dot com."
)
EMAIL_GO_BACK = (
    "Okay. How would you like to sign in — with Google, with Apple, "
    "or with a different email?"
)
GOOGLE_EMAIL_QUESTION = "Which Google email should I use?"
SSO_ACK = "One moment while I sign you in with Google."
SSO_BUSY = "One moment while I finish signing you in."
# Short ack only — screen-specific verify/password guidance is spoken once after fill.
EMAIL_ACK = "Okay, I'm entering that email now."
EMAIL_BUSY = (
    "Please finish the sign-in steps in the Pointcare window. "
    "I'll continue when you're signed in."
)
CLOSING = (
    "I've finished filling your household eligibility form. "
    "Thank you for your time. Goodbye."
)
# Stage E Application Collect Mode — entered only after Yes on program-summary.
STAGE_E_ENABLED = True

_PENDING_SIGNIN_RE = re.compile(
    r"complete the visible google|say google again|has not finished|"
    r"sign-in page is not ready|which google email|"
    r"finish signing in in the pointcare window|password or verification code",
    re.I,
)
_signin_watch_tasks: dict[str, asyncio.Task] = {}
_sso_tasks: dict[str, asyncio.Task] = {}
_email_auth_tasks: dict[str, asyncio.Task] = {}
_app_browser_tasks: dict[str, asyncio.Task] = {}

_SLOW_APP_STEPS = frozenset(
    {
        "app_agent_offer",
        "app_citizenship",
        "app_immigration",
        "app_insurance",
        "app_dob_review",
        "app_address_review",
        "app_marital_review",
        "app_childcare_review",
        "app_circumstances",
        "app_tax",
        "app_income",
        "app_income_employed",
        "app_income_self",
        "app_income_more",
        "app_expenses",
        "app_documents",
        "app_submit_confirm",
    }
)


async def start_stage_e(call_id: str, form_url: str = "") -> TurnResult:
    """Begin Stage E when application was already started (idempotent recovery)."""
    from .app_stage import app_question_for_step
    from .live_form import get_session

    session = await get_session(call_id)
    if session is None or not (session.app_step or "").strip():
        return TurnResult(
            speech=CLOSING,
            end_call=True,
            phase="closing",
            detail="stage_e_not_ready",
        )
    step = session.app_step
    speech = app_question_for_step(step, session)
    ctrl = await get_controller(call_id)
    ctrl.phase = "application"
    ctrl.last_speech = speech
    return TurnResult(
        speech=speech,
        end_call=False,
        phase="application",
        detail="stage_e",
    )


@dataclass
class TurnResult:
    speech: str
    end_call: bool = False
    phase: str = ""
    turn_id: str = ""
    ok: bool = True
    detail: str = ""


def _speakable_email(addr: str) -> str:
    """TTS-friendly reading of an address (at / dot), no secrets beyond the address."""
    value = (addr or "").strip().lower()
    if "@" not in value:
        return value
    local, _, domain = value.partition("@")
    # Speak digit runs individually so "003" is not mumbled as a large number.
    local = re.sub(r"\d", lambda m: f" {m.group(0)} ", local)
    local = re.sub(r"\s+", " ", local).strip()
    domain = domain.replace(".", " dot ")
    return f"{local} at {domain}"


def _email_confirm_speech(addr: str) -> str:
    return (
        f"I heard {_speakable_email(addr)}. Is that correct? "
        "Say yes to continue, or say go back to enter a different email."
    )


_EMAIL_GO_BACK_RE = re.compile(
    r"(?:"
    r"\bgo\s*-?\s*backs?\b|"
    r"\bgoback\b|"
    r"\bgo\s+backwards?\b|"
    r"\bgoing\s+back\b|"
    r"\bwent\s+back\b|"
    r"\bplease\s+go\s+back\b|"
    r"\bback\s+to\s+(?:email|the\s+email)\b"
    r")",
    re.I,
)


def _wants_email_go_back(text: str) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    # Substring-friendly: ASR may attach punctuation or glue tokens.
    compact = re.sub(r"[\s\-_.]+", "", lowered)
    if "goback" in compact or "gobacks" in compact:
        return True
    if _EMAIL_GO_BACK_RE.search(lowered):
        return True
    intent = detect_control_intent(text)
    if intent.kind in ("go_back", "redo"):
        return True
    # A newly dictated address is a correction, not go-back
    # (e.g. local-part "wrong@…" must not match the word "wrong").
    if extract_email(text):
        return False
    if intent.kind == "change":
        return True
    if _is_no(text):
        return True
    return bool(
        re.search(
            r"\b("
            r"wrong|incorrect|not right|not correct|different email|"
            r"change (the )?email|start over|enter again|re-?enter|"
            r"try again|another email|new email"
            r")\b",
            lowered,
        )
    )


@dataclass
class CallController:
    call_id: str
    # language|sign_in|email|email_confirm|email_await_secure|google_email|eligibility|application|closing
    phase: str = "language"
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
    # Native email auth (password/OTP completed in the browser window).
    email_auth_in_progress: bool = False
    email_fill_retries: int = 0
    # Bumped on every go-back / new auth start so stale Playwright tasks cannot
    # overwrite phase after the caller corrects a wrong email.
    email_auth_epoch: int = 0
    # Last successfully extracted address for this call (never logged).
    email_captured: bool = False
    # Held only until confirm/go-back; cleared after auth starts or go-back.
    pending_email: str = ""
    eligibility_screen_verified: bool = False
    eligibility_question_sent: bool = False
    active_response_id: int | None = None
    active_websocket_status: str = "unknown"  # connected|disconnected|unknown
    interrupt_seq: int = 0
    # Pending correction: {"field": "first_name"|"last_name"|"age"|..., "prompt": str}
    pending_correction: dict[str, str] | None = None
    # Browser eligibility op in flight — ignore answers until it finishes.
    elig_op_in_progress: bool = False
    # Bumped on go-back / superseded browser ops so late Playwright results die.
    elig_nav_epoch: int = 0
    last_question_delivery_key: str = ""
    # Outbound speech that failed because the socket was down — retry on reconnect.
    pending_outbound_speech: str = ""
    pending_outbound_key: str = ""
    # Retell sends a cumulative transcript on every response_required. Track the
    # tip of the user transcript so unchanged historical lines are not re-run.
    last_transcript_user_index: int = -1
    last_transcript_user_fp: str = ""
    # User-turn index already applied to a field. A later step must not consume
    # that same transcript event again. A higher index is a new spoken turn,
    # even when the words match.
    consumed_transcript_user_index: int = -1
    # True after an eligibility browser attempt failed. A later Retell turn
    # (higher user index) may repeat the same words, such as Yes after a
    # failed confirmation. A replay of the consumed index is still dropped.
    last_elig_attempt_failed: bool = False
    # Last response_id we already answered (provider-level event dedupe).
    last_handled_response_id: int | None = None


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
    for bucket in (
        _signin_watch_tasks,
        _sso_tasks,
        _elig_browser_tasks,
        _email_auth_tasks,
        _app_browser_tasks,
    ):
        task = bucket.pop(call_id, None)
        if task and not task.done():
            task.cancel()
    try:
        from .form_bridge import invalidate_transcript_answers

        invalidate_transcript_answers(call_id)
    except Exception:
        pass
    clear_call_trace(call_id)
    async with _controllers_lock:
        _controllers.pop(call_id, None)


def _cancel_elig_browser_task(call_id: str) -> None:
    task = _elig_browser_tasks.pop(call_id, None)
    if task and not task.done():
        task.cancel()
        logger.info("elig_browser_cancelled_for_nav call_id=%s", call_id)


def _begin_elig_navigation(ctrl: CallController) -> int:
    """Invalidate in-flight eligibility browser work and stale transcript answers."""
    ctrl.elig_nav_epoch += 1
    ctrl.elig_op_in_progress = False
    ctrl.pending_op_id = ""
    _cancel_elig_browser_task(ctrl.call_id)
    try:
        from .form_bridge import invalidate_transcript_answers

        invalidate_transcript_answers(ctrl.call_id)
    except Exception:
        pass
    return ctrl.elig_nav_epoch


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


def _same_member_form_step(step: str) -> bool:
    """True when the active step belongs to an open spouse/child/person form."""
    s = str(step or "")
    return s.startswith(("spouse_", "child_", "person_"))


def _step_group_safe(step: str) -> str:
    s = str(step or "")
    if s.startswith("spouse_"):
        return "spouse_form"
    if s.startswith("child_"):
        return "child_form"
    if s.startswith("person_"):
        return "person_form"
    return s


_FIELD_ONLY_RE = re.compile(
    r"^\s*(?:the\s+)?(?P<field>first\s+name|last\s+name|name|age|relationship)\s*"
    r"(?:only|please|field)?[.!]?\s*$",
    re.I,
)


def _field_from_confirm_utterance(text: str) -> str:
    """Map a short confirm-correction reply onto a field name."""
    m = _FIELD_ONLY_RE.match(text or "")
    if not m:
        return ""
    raw = re.sub(r"\s+", "_", m.group("field").lower())
    if raw == "name":
        return "first_name"
    return raw


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
        if current_step.startswith("spouse_"):
            return "spouse_role"
        if current_step.startswith("person_"):
            return "person_rel"
        return "child_rel"
    return current_step


async def _deliver_question_once(
    call_id: str, speech: str, *, key: str, end_call: bool = False
) -> bool:
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
    ok = await _speak_interrupt(call_id, text, end_call=end_call)
    if ok:
        ctrl.last_question_delivery_key = key
        ctrl.last_speech = text
        ctrl.pending_outbound_speech = ""
        ctrl.pending_outbound_key = ""
        logger.info(
            "question_delivery_outcome call_id=%s key=%s outcome=ws_sent end=%s",
            call_id,
            key,
            end_call,
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
    epoch: int,
) -> None:
    """Playwright eligibility apply off the WebSocket response path."""
    ctrl = await get_controller(call_id)

    def _aborted() -> bool:
        return ctrl.elig_nav_epoch != epoch

    logger.info(
        "elig_browser_started call_id=%s step=%s turn_id=%s epoch=%s answer=%r",
        call_id,
        step,
        turn_id,
        epoch,
        (answer or "")[:40],
    )
    trace_event(
        call_id,
        "elig_browser_started",
        step=step,
        turn_id=turn_id,
        epoch=epoch,
        op_id=turn_id,
        answer=(answer or "")[:40],
    )
    try:
        if _aborted():
            logger.info(
                "elig_browser_aborted_before_submit call_id=%s epoch=%s", call_id, epoch
            )
            return
        result = await submit_field(
            call_id,
            "eligibility_answer",
            answer,
            url=form_url or config.POINTCARE_URL,
            request_id=f"ws-bg:{turn_id}:e{epoch}",
        )
        if _aborted():
            logger.info(
                "elig_browser_aborted_after_submit call_id=%s epoch=%s (go-back won)",
                call_id,
                epoch,
            )
            trace_event(
                call_id,
                "elig_browser_stale",
                turn_id=turn_id,
                epoch=epoch,
                reason="nav_epoch",
            )
            return
        spoken = _strip_ask(str(result.get("spoken") or result.get("message") or ""))
        session = await get_session(call_id)
        # Explicit empty step means Stage D is complete — do not fall back to a
        # stale session.elig_step (empty string is falsy in `or` chains).
        if "step" in result:
            next_step = str(result.get("step") or "")
        else:
            next_step = str((session.elig_step if session else "") or "")
        if not result.get("ok"):
            ctrl.last_elig_attempt_failed = True
            # Prefer live session step (special_person stays special_person on clarify).
            live_step = (
                str(result.get("step") or "")
                or (session.elig_step if session else "")
                or step
            )
            speech = spoken or format_question(live_step)
            if _aborted():
                return
            ctrl.last_speech = speech
            logger.info(
                "elig_browser_completed call_id=%s step=%s live_step=%s outcome=failed "
                "reason=%s",
                call_id,
                step,
                live_step,
                str(result.get("error") or "")[:80],
            )
            trace_event(
                call_id,
                "elig_browser_failed",
                step=step,
                live_step=live_step,
                turn_id=turn_id,
                reason=str(result.get("error") or "")[:80],
            )
            await _deliver_question_once(
                call_id, speech, key=f"fail:{live_step}:{turn_id}"
            )
            return

        if _aborted():
            return
        ctrl.committed.append(
            {
                "key": commit_key,
                "turn_id": turn_id,
                "step": step,
                "answer": answer,
                "epoch": epoch,
                "at": time.time(),
            }
        )
        if not next_step:
            speech = _closing_speech(spoken)
            ctrl.phase = "closing"
            ctrl.last_speech = speech
            await _deliver_question_once(
                call_id, speech, key=f"close:{turn_id}", end_call=True
            )
            logger.info(
                "elig_browser_completed call_id=%s step=%s outcome=closing end_call=1",
                call_id,
                step,
            )
            return

        # Programs Yes → Stage E app_* step (do not close).
        if str(next_step).startswith("app_") or result.get("phase") == "application":
            ctrl.phase = "application"
            speech = spoken or format_question(next_step)
            # format_question may miss app_* — prefer spoken from browser result.
            if not spoken and str(next_step).startswith("app_"):
                from .app_stage import app_question_for_step

                speech = app_question_for_step(str(next_step))
            elif spoken:
                speech = spoken
            ctrl.last_speech = speech
            logger.info(
                "elig_browser_completed call_id=%s step=%s next=%s outcome=stage_e",
                call_id,
                step,
                next_step,
            )
            await _deliver_question_once(
                call_id, speech, key=f"stage_e:{next_step}:{turn_id}"
            )
            return

        ctrl.last_elig_attempt_failed = False
        speech = spoken or format_question(next_step)
        ctrl.last_speech = speech
        if result.get("await_field_correction"):
            ctrl.pending_correction = {
                "field": "",
                "prompt": speech,
                "need_field": True,
            }
        logger.info(
            "elig_browser_completed call_id=%s step=%s next=%s browser_verified=1 epoch=%s",
            call_id,
            step,
            next_step,
            epoch,
        )
        trace_event(
            call_id,
            "elig_browser_ok",
            step=step,
            next_step=next_step,
            turn_id=turn_id,
            epoch=epoch,
            screen=str(result.get("screen") or ""),
        )
        await _deliver_question_once(
            call_id, speech, key=f"next:{next_step}:{turn_id}"
        )
    except asyncio.CancelledError:
        # Disconnect / superseded nav — never treat as a successful person save.
        logger.info(
            "elig_browser_cancelled call_id=%s step=%s epoch=%s "
            "(no commit; browser state must be re-verified on resume)",
            call_id,
            step,
            epoch,
        )
        trace_event(
            call_id,
            "elig_browser_cancelled",
            step=step,
            turn_id=turn_id,
            epoch=epoch,
            committed=False,
        )
        raise
    except Exception:
        logger.exception("elig_browser_failed call_id=%s step=%s", call_id, step)
        if not _aborted():
            ctrl.last_elig_attempt_failed = True
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
            if not _aborted():
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
    epoch: int,
) -> None:
    """Replace any in-flight eligibility browser task — never silently drop answers."""
    prev = _elig_browser_tasks.get(call_id)
    if prev and not prev.done():
        prev.cancel()
        logger.info(
            "elig_browser_replaced call_id=%s turn_id=%s epoch=%s",
            call_id,
            turn_id,
            epoch,
        )
    _elig_browser_tasks[call_id] = asyncio.create_task(
        _run_eligibility_browser_background(
            call_id,
            answer=answer,
            step=step,
            turn_id=turn_id,
            form_url=form_url,
            commit_key=commit_key,
            epoch=epoch,
        ),
        name=f"elig-browser-{call_id}-e{epoch}",
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
    session = None
    if config.APPLICATION_TEST_MODE:
        session = await get_session(call_id)
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
    opening = _application_test_opening(session)
    if opening:
        if session is not None and session.app_step:
            ctrl.phase = "application"
        else:
            ctrl.phase = "eligibility"
            ctrl.eligibility_screen_verified = True
        ctrl.eligibility_question_sent = True
        ctrl.last_speech = opening
        logger.info("application_test_opening call_id=%s", call_id)
        return opening
    ctrl.phase = "language"
    ctrl.last_speech = LIVE_FILL_GREETING
    return LIVE_FILL_GREETING


def _application_test_opening(session: Any) -> str:
    if not config.APPLICATION_TEST_MODE or session is None:
        return ""
    if (
        session.elig_step
        not in ("programs", "proceed_application", "confirm_start_application")
        and not session.app_step
    ):
        return ""
    return (
        (session.filled.get("application_test_opening") or "").strip()
        or (session.filled.get("application_gate_speech") or "").strip()
    )


async def announce_application_test_opening(call_id: str) -> bool:
    """Replace a greeting already spoken with the saved-household question.

    No-op unless application test mode is on and the browser has already landed
    on the program summary or an in-progress application.
    """
    if not config.APPLICATION_TEST_MODE:
        return False
    session = await get_session(call_id)
    opening = _application_test_opening(session)
    if not opening or session is None:
        return False
    ctrl = await get_controller(call_id)
    greeted = ctrl.phase == "language"
    if session.app_step:
        ctrl.phase = "application"
    else:
        ctrl.phase = "eligibility"
        ctrl.eligibility_screen_verified = True
    ctrl.last_speech = opening
    ctrl.eligibility_question_sent = True
    if not greeted:
        return True
    from .retell_llm_ws import speak_now, websocket_connected

    if not websocket_connected(call_id):
        return True
    return await speak_now(call_id, opening, interrupt_id=2)


def _speech_after_signin(session: Any, spoken: str) -> str:
    """Question after auth. Prefer the browser tool's spoken text (includes Stage D intro)."""
    speech = _strip_ask(spoken or "")
    # Never replace a non-empty post-auth spoken line — _begin_eligibility already
    # built the household-eligibility purpose intro + first question. Rebuilding
    # via _eligibility_question_for_session used to drop that intro.
    if speech:
        return speech
    if session is not None:
        stored = (session.filled.get("eligibility_opening_speech") or "").strip()
        if stored:
            return stored
        if session.elig_step:
            rebuilt = _eligibility_question_for_session(session)
            if rebuilt:
                return rebuilt
    if config.APPLICATION_AFTER_SIGNIN:
        return "I could not open the saved Pointcare application page."
    return ELIG_QUESTIONS.get("marital", "")


def _eligibility_question_for_session(session: Any) -> str:
    step = (session.elig_step if session else "") or "marital"
    if session is not None:
        stored = (session.filled.get("eligibility_opening_speech") or "").strip()
        # First delivery after Stage D open — keep the purpose intro.
        if (
            step == "marital"
            and stored
            and not session.filled.get("eligibility_opening_delivered")
        ):
            return stored
        from .live_form import _spoken_question_for_step

        return _spoken_question_for_step(session, step) or ELIG_QUESTIONS.get(
            "marital", ""
        )
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


async def _promote_email_auth_success(
    call_id: str,
    result: dict[str, Any],
    *,
    turn_id: str = "",
    epoch: int | None = None,
) -> str:
    """Move controller to eligibility after native email auth succeeds."""
    ctrl = await get_controller(call_id)
    if epoch is not None and ctrl.email_auth_epoch != epoch:
        logger.info(
            "Email auth promote skipped stale epoch call_id=%s task_epoch=%s live=%s",
            call_id,
            epoch,
            ctrl.email_auth_epoch,
        )
        return ""
    session = await get_session(call_id)
    speech = _speech_after_signin(session, str(result.get("spoken") or ""))
    async with ctrl.lock:
        if epoch is not None and ctrl.email_auth_epoch != epoch:
            return ""
        ctrl.email_auth_in_progress = False
        ctrl.sso_in_progress = False
        ctrl.phase = "eligibility"
        ctrl.pending_op_id = ""
        ctrl.eligibility_screen_verified = True
        ctrl.committed.append({"op": "sign_in_email", "turn_id": turn_id})
        ctrl.last_speech = speech
    logger.info(
        "Email auth completed call_id=%s outcome=ok browser_screen_verified=1",
        call_id,
    )
    await deliver_first_eligibility_question(call_id, speech=speech)
    return speech


async def _run_email_auth_background(
    call_id: str,
    *,
    email: str,
    form_url: str = "",
    turn_id: str = "",
    epoch: int = 0,
) -> None:
    """Fill email via Playwright, then poll while the caller finishes secure steps."""
    ctrl = await get_controller(call_id)

    def _aborted() -> bool:
        return ctrl.email_auth_epoch != epoch

    try:
        if _aborted():
            logger.info("Email auth aborted before fill call_id=%s epoch=%s", call_id, epoch)
            return
        result = await submit_field(
            call_id,
            "email",
            email,
            url=form_url or config.POINTCARE_URL,
            request_id=f"ws-email:{turn_id or uuid.uuid4().hex[:8]}:e{epoch}",
        )
        if _aborted():
            logger.info(
                "Email auth aborted after fill call_id=%s epoch=%s (go-back won)",
                call_id,
                epoch,
            )
            return
        spoken = _strip_ask(str(result.get("spoken") or result.get("message") or ""))
        auth_screen = str(result.get("auth_screen") or "")
        awaiting = bool(result.get("awaiting_secure"))

        if auth_screen == "google_popup" or "which google" in spoken.lower():
            async with ctrl.lock:
                if _aborted():
                    return
                ctrl.email_auth_in_progress = False
                ctrl.sso_in_progress = False
                ctrl.phase = "google_email"
                ctrl.last_speech = spoken or GOOGLE_EMAIL_QUESTION
            schedule_signin_watch(call_id)
            await _speak_interrupt(call_id, spoken or GOOGLE_EMAIL_QUESTION)
            return

        if result.get("ok") and auth_screen == "signed_in":
            await _promote_email_auth_success(
                call_id, result, turn_id=turn_id, epoch=epoch
            )
            return

        if not result.get("ok") and not awaiting:
            async with ctrl.lock:
                if _aborted():
                    return
                ctrl.email_auth_in_progress = False
                ctrl.sso_in_progress = False
                ctrl.email_captured = False
                ctrl.phase = "email"
                ctrl.email_fill_retries += 1
                # Useful clarification — not the same first-ask prompt on a loop.
                ctrl.last_speech = spoken or EMAIL_CLARIFY
            logger.info(
                "Email auth completed call_id=%s outcome=failed screen=%s",
                call_id,
                auth_screen,
            )
            if not _aborted():
                await _speak_interrupt(call_id, spoken or EMAIL_CLARIFY)
            return

        # Email filled; speak one screen-specific secure step (verify link / password).
        guide = spoken or spoken_for_auth_screen(auth_screen)
        async with ctrl.lock:
            if _aborted():
                return
            ctrl.phase = "email_await_secure"
            ctrl.email_captured = True
            # Replace the short EMAIL_ACK with the real instruction (once).
            ctrl.last_speech = guide
        logger.info(
            "Email auth awaiting secure step call_id=%s screen=%s epoch=%s",
            call_id,
            auth_screen or "unknown",
            epoch,
        )
        if guide and not _aborted():
            await _speak_interrupt(call_id, guide)

        if _aborted():
            return
        polled = await poll_email_auth_until_signed_in(call_id)
        if _aborted():
            logger.info(
                "Email auth aborted after poll call_id=%s epoch=%s", call_id, epoch
            )
            return
        if polled and polled.get("ok") and polled.get("auth_screen") == "signed_in":
            await _promote_email_auth_success(
                call_id, polled, turn_id=turn_id, epoch=epoch
            )
            return

        async with ctrl.lock:
            if _aborted():
                return
            ctrl.email_auth_in_progress = False
            ctrl.sso_in_progress = False
            # Keep email_captured — do not re-ask for the address on timeout;
            # guide them to finish the window or say the address only if needed.
            if ctrl.phase == "email_await_secure":
                ctrl.last_speech = (
                    (polled or {}).get("spoken") or EMAIL_SECURE_RETRY
                )
        if not _aborted():
            await _speak_interrupt(
                call_id, (polled or {}).get("spoken") or EMAIL_SECURE_RETRY
            )
    except asyncio.CancelledError:
        async with ctrl.lock:
            if not _aborted():
                ctrl.email_auth_in_progress = False
                ctrl.sso_in_progress = False
        logger.info("Email auth cancelled call_id=%s epoch=%s", call_id, epoch)
        raise
    except Exception:
        async with ctrl.lock:
            if not _aborted():
                ctrl.email_auth_in_progress = False
                ctrl.sso_in_progress = False
                ctrl.email_captured = False
                ctrl.phase = "email"
        logger.exception("Email auth background failed call_id=%s", call_id)
        if not _aborted():
            try:
                await _speak_interrupt(call_id, EMAIL_CLARIFY)
            except Exception:
                pass
    finally:
        task = _email_auth_tasks.get(call_id)
        if task is asyncio.current_task():
            _email_auth_tasks.pop(call_id, None)


def _schedule_email_auth(
    call_id: str,
    *,
    email: str,
    form_url: str = "",
    turn_id: str = "",
    epoch: int = 0,
) -> None:
    """Run native email fill + secure-auth poll off the WebSocket response path."""
    prev = _email_auth_tasks.get(call_id)
    if prev and not prev.done():
        prev.cancel()
        logger.info("Email auth replaced prior task call_id=%s", call_id)
    _email_auth_tasks[call_id] = asyncio.create_task(
        _run_email_auth_background(
            call_id, email=email, form_url=form_url, turn_id=turn_id, epoch=epoch
        ),
        name=f"email-auth-{call_id}-e{epoch}",
    )
    logger.info(
        "Email auth started call_id=%s turn_id=%s epoch=%s", call_id, turn_id, epoch
    )


def _cancel_email_auth_task(call_id: str) -> None:
    task = _email_auth_tasks.pop(call_id, None)
    if task and not task.done():
        task.cancel()


def _begin_email_confirm(ctrl: CallController, email: str) -> str:
    """Store address and return the reconfirm prompt (does not start Playwright)."""
    # Invalidate any in-flight auth from a prior confirmation.
    ctrl.email_auth_epoch += 1
    _cancel_email_auth_task(ctrl.call_id)
    ctrl.pending_email = (email or "").strip().lower()
    ctrl.email_captured = True
    ctrl.email_auth_in_progress = False
    ctrl.sso_in_progress = False
    ctrl.phase = "email_confirm"
    speech = _email_confirm_speech(ctrl.pending_email)
    ctrl.last_speech = speech
    return speech


def _reset_email_entry(ctrl: CallController) -> str:
    """Go back to sign-in chooser (Google / Apple / different email) and reset browser UI.

    Idempotent when already on the chooser with no pending email auth — do not
    bump epoch or navigate again for a repeated/stale go-back utterance.
    """
    already_on_chooser = (
        ctrl.phase == "sign_in"
        and not (ctrl.pending_email or "").strip()
        and not ctrl.email_auth_in_progress
        and not ctrl.sso_in_progress
        and not ctrl.email_captured
    )
    if already_on_chooser:
        ctrl.last_speech = EMAIL_GO_BACK
        logger.info(
            "email_go_back_noop call_id=%s phase=%s reason=already_on_chooser",
            ctrl.call_id,
            ctrl.phase,
        )
        trace_event(
            ctrl.call_id,
            "email_go_back_noop",
            phase=ctrl.phase,
            reason="already_on_chooser",
        )
        return EMAIL_GO_BACK

    ctrl.email_auth_epoch += 1
    _cancel_email_auth_task(ctrl.call_id)
    ctrl.pending_email = ""
    ctrl.email_captured = False
    ctrl.email_auth_in_progress = False
    ctrl.sso_in_progress = False
    ctrl.phase = "sign_in"
    ctrl.last_speech = EMAIL_GO_BACK
    logger.info(
        "email_go_back_applied call_id=%s epoch=%s",
        ctrl.call_id,
        ctrl.email_auth_epoch,
    )
    trace_event(
        ctrl.call_id,
        "email_go_back",
        epoch=ctrl.email_auth_epoch,
        phase=ctrl.phase,
        op="browser_reset",
    )
    # Navigate browser off verify/password back to Google/Apple/email options.
    try:
        asyncio.create_task(
            _reset_email_sign_in_browser(ctrl.call_id),
            name=f"reset-email-ui-{ctrl.call_id}",
        )
    except Exception:
        pass
    return EMAIL_GO_BACK


async def _reset_email_sign_in_browser(call_id: str) -> None:
    try:
        from .live_form import reset_email_sign_in_ui

        await reset_email_sign_in_ui(call_id)
    except Exception:
        logger.exception("reset_email_sign_in_ui failed call_id=%s", call_id)


def _start_confirmed_email_auth(
    ctrl: CallController,
    *,
    call_id: str,
    form_url: str,
    turn_id: str,
    response_id: int | None,
) -> TurnResult:
    email = (ctrl.pending_email or "").strip()
    if not email:
        speech = EMAIL_QUESTION
        ctrl.phase = "email"
        ctrl.email_captured = False
        ctrl.last_speech = speech
        return TurnResult(
            speech=speech,
            phase=ctrl.phase,
            turn_id=turn_id,
            detail="email_address_needed",
            end_call=False,
        )
    ctrl.email_auth_epoch += 1
    epoch = ctrl.email_auth_epoch
    ctrl.pending_op_id = turn_id
    ctrl.email_captured = True
    ctrl.email_auth_in_progress = True
    ctrl.sso_in_progress = True
    ctrl.sso_ack_sent = True
    ctrl.eligibility_question_sent = False
    ctrl.eligibility_screen_verified = False
    ctrl.phase = "email_await_secure"
    ctrl.last_speech = EMAIL_ACK
    logger.info(
        "acknowledgment_sent call_id=%s response_id=%s speech=%r epoch=%s",
        call_id,
        response_id,
        EMAIL_ACK,
        epoch,
    )
    trace_event(
        call_id,
        "email_auth_ack",
        turn_id=turn_id,
        epoch=epoch,
        phase=ctrl.phase,
        response_id=response_id,
    )
    _schedule_email_auth(
        call_id,
        email=email,
        form_url=form_url or config.POINTCARE_URL,
        turn_id=turn_id,
        epoch=epoch,
    )
    # Keep pending_email for go-back / reconfirm until auth completes.
    return TurnResult(
        speech=EMAIL_ACK,
        phase=ctrl.phase,
        turn_id=turn_id,
        detail="email_ack",
        end_call=False,
    )


def _closing_speech(spoken: str = "") -> str:
    """Prefer a server closing line that acknowledges the household form is done."""
    text = (spoken or "").strip()
    lowered = text.lower()
    if text and (
        "household eligibility" in lowered
        or "finished filling" in lowered
        or ("thank" in lowered and "goodbye" in lowered)
        or "all set" in lowered
    ):
        return text
    return CLOSING


async def _speak_interrupt(
    call_id: str, content: str, *, end_call: bool = False
) -> bool:
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
        ok = await speak_now(
            call_id, text, interrupt_id=interrupt_id, end_call=end_call
        )
        logger.info(
            "interrupt_sent call_id=%s interrupt_id=%s outcome=%s end=%s speech=%r",
            call_id,
            interrupt_id,
            "ok" if ok else "failed",
            end_call,
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
        session = await get_session(call_id)
        if session is not None and (session.filled.get("eligibility_opening_speech") or ""):
            session.filled["eligibility_opening_delivered"] = "1"
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
        speech = _speech_after_signin(session, spoken)
        if session and session.elig_step:
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


async def _run_apple_sso_background(
    call_id: str,
    *,
    form_url: str = "",
    turn_id: str = "",
) -> None:
    """Playwright Apple SSO; same delivery pattern as Google."""
    ctrl = await get_controller(call_id)
    try:
        result = await submit_field(
            call_id,
            "continue_with_apple",
            "",
            url=form_url or config.POINTCARE_URL,
            request_id=f"ws-apple:{turn_id or uuid.uuid4().hex[:8]}",
        )
        spoken = _strip_ask(str(result.get("spoken") or result.get("message") or ""))
        if not result.get("ok"):
            async with ctrl.lock:
                ctrl.sso_in_progress = False
                ctrl.last_speech = spoken or SIGN_IN_QUESTION
            await _speak_interrupt(call_id, spoken or SIGN_IN_QUESTION)
            return
        session = await get_session(call_id)
        speech = _speech_after_signin(session, spoken)
        async with ctrl.lock:
            ctrl.sso_in_progress = False
            ctrl.phase = "eligibility"
            ctrl.pending_op_id = ""
            ctrl.eligibility_screen_verified = True
            ctrl.committed.append({"op": "sign_in_apple", "turn_id": turn_id})
            ctrl.last_speech = speech
        await deliver_first_eligibility_question(call_id, speech=speech)
    except asyncio.CancelledError:
        async with ctrl.lock:
            ctrl.sso_in_progress = False
        raise
    except Exception:
        async with ctrl.lock:
            ctrl.sso_in_progress = False
        logger.exception("Apple SSO background failed call_id=%s", call_id)
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


async def _repeat_with_context(call_id: str, ctrl: CallController) -> str:
    """Re-speak the current step with purpose context — never advance the flow."""
    phase = ctrl.phase or ""
    if phase == "language":
        return LIVE_FILL_GREETING
    if phase == "sign_in":
        return f"{CALL_PURPOSE} {SIGN_IN_QUESTION}"
    if phase == "google_email":
        return f"{CALL_PURPOSE} {GOOGLE_EMAIL_QUESTION}"
    if phase == "email":
        return f"{CALL_PURPOSE} {EMAIL_QUESTION}"
    if phase == "email_confirm":
        return ctrl.last_speech or (
            f"{CALL_PURPOSE} Please confirm the email address I should use."
        )
    if phase == "email_await_secure":
        return ctrl.last_speech or (
            "Please finish the password or verification steps in the Pointcare "
            "window. I'll continue when you're signed in."
        )
    if phase == "eligibility":
        session = await get_session(call_id)
        step = (session.elig_step if session else "") or "marital"
        if ctrl.pending_correction:
            return (
                ctrl.pending_correction.get("prompt")
                or _correction_prompt(
                    ctrl.pending_correction.get("field") or "first_name"
                )
            )
        if step == "programs" and session is not None:
            stored = (session.filled.get("programs_permission_speech") or "").strip()
            if stored:
                return stored
        labels = await peek_choice_labels(call_id)
        question = format_question(step, options=labels or None)
        from .live_form import _section_context_for_step

        context = _section_context_for_step(step)
        if context and step.startswith(("spouse_", "child_")):
            # Explain section context without restarting or losing answers.
            return f"{context} {question}"
        return question
    if phase == "application":
        session = await get_session(call_id)
        from .app_stage import app_question_for_step

        step = (session.app_step if session else "") or ""
        return app_question_for_step(step) if step else (ctrl.last_speech or CLOSING)
    if phase == "closing":
        return ctrl.last_speech or CLOSING
    return ctrl.last_speech or f"{CALL_PURPOSE} Could you say that again?"


async def _process_application_turn(
    call_id: str,
    ctrl: CallController,
    *,
    text: str,
    fp: str,
    dedupe_ctx: str,
    turn_id: str,
    form_url: str,
) -> TurnResult:
    """Collect Mode Stage E — one spoken answer → Playwright → next question."""
    from .app_stage import app_question_for_step

    session = await get_session(call_id)
    step = (session.app_step if session else "") or ""
    if not step:
        ctrl.phase = "closing"
        speech = ctrl.last_speech or CLOSING
        return TurnResult(
            speech=speech, end_call=True, phase="closing", turn_id=turn_id, detail="app_done"
        )

    intent = detect_control_intent(text) if text else detect_control_intent("")
    if intent.kind == "repeat":
        speech = app_question_for_step(step)
        ctrl.last_speech = speech
        return TurnResult(
            speech=speech, phase="application", turn_id=turn_id, detail="repeat"
        )

    if not text:
        speech = app_question_for_step(step)
        ctrl.last_speech = speech
        return TurnResult(speech=speech, phase="application", turn_id=turn_id)

    commit_key = f"app:{step}:{fp}"
    if any(item.get("key") == commit_key for item in ctrl.committed[-12:]):
        return TurnResult(
            speech=ctrl.last_speech or app_question_for_step(step),
            phase="application",
            turn_id=turn_id,
            detail="duplicate_app_answer",
        )

    ctrl.last_user_fp = fp
    ctrl.last_dedupe_ctx = f"application:{step}"
    ctrl.pending_op_id = turn_id

    if step in _SLOW_APP_STEPS:
        ctrl.elig_op_in_progress = True
        epoch = ctrl.elig_nav_epoch

        async def _run_app() -> None:
            try:
                result = await submit_field(
                    call_id,
                    "application_answer",
                    text,
                    url=form_url or config.POINTCARE_URL,
                    request_id=f"ws-app:{turn_id}",
                )
                if _controllers.get(call_id) is not ctrl:
                    return
                if ctrl.elig_nav_epoch != epoch:
                    return
                spoken = _strip_ask(
                    str(result.get("spoken") or result.get("message") or "")
                )
                next_step = str(result["step"]) if "step" in result else ""
                if not result.get("ok"):
                    speech = spoken or app_question_for_step(step)
                    ctrl.last_speech = speech
                    await _deliver_question_once(
                        call_id, speech, key=f"app-fail:{step}:{turn_id}"
                    )
                    return
                ctrl.committed.append(
                    {
                        "key": commit_key,
                        "turn_id": turn_id,
                        "step": step,
                        "answer": text,
                        "at": time.time(),
                    }
                )
                if not next_step:
                    ctrl.phase = "closing"
                    speech = _closing_speech(spoken)
                    ctrl.last_speech = speech
                    await _deliver_question_once(
                        call_id,
                        speech,
                        key=f"app-close:{turn_id}",
                        end_call=bool(result.get("end_call", True)),
                    )
                    return
                ctrl.phase = "application"
                speech = spoken or app_question_for_step(next_step)
                ctrl.last_speech = speech
                await _deliver_question_once(
                    call_id, speech, key=f"app-next:{next_step}:{turn_id}"
                )
            except Exception:
                logger.exception("app_browser_failed call_id=%s step=%s", call_id, step)
                try:
                    await _deliver_question_once(
                        call_id,
                        app_question_for_step(step),
                        key=f"app-err:{step}:{turn_id}",
                    )
                except Exception:
                    pass
            finally:
                async with ctrl.lock:
                    if ctrl.elig_nav_epoch == epoch:
                        ctrl.elig_op_in_progress = False
                        ctrl.pending_op_id = ""
                task = _app_browser_tasks.get(call_id)
                if task is asyncio.current_task():
                    _app_browser_tasks.pop(call_id, None)

        prev = _app_browser_tasks.get(call_id)
        if prev and not prev.done():
            prev.cancel()
        _app_browser_tasks[call_id] = asyncio.create_task(
            _run_app(), name=f"app-browser-{call_id}"
        )
        return TurnResult(
            speech=ELIG_PROGRESS_ACK,
            phase="application",
            turn_id=turn_id,
            detail="app_ack",
        )

    # Fast path (unlikely for Stage E — most steps are slow).
    result = await submit_field(
        call_id,
        "application_answer",
        text,
        url=form_url or config.POINTCARE_URL,
        request_id=f"ws-app:{turn_id}",
    )
    spoken = _strip_ask(str(result.get("spoken") or result.get("message") or ""))
    next_step = str(result["step"]) if "step" in result else ""
    if not result.get("ok"):
        speech = spoken or app_question_for_step(step)
        ctrl.last_speech = speech
        return TurnResult(
            speech=speech,
            phase="application",
            turn_id=turn_id,
            ok=False,
            detail=str(result.get("error") or "")[:200],
        )
    ctrl.committed.append(
        {
            "key": commit_key,
            "turn_id": turn_id,
            "step": step,
            "answer": text,
            "at": time.time(),
        }
    )
    if not next_step:
        ctrl.phase = "closing"
        speech = _closing_speech(spoken)
        ctrl.last_speech = speech
        return TurnResult(
            speech=speech,
            end_call=bool(result.get("end_call", True)),
            phase="closing",
            turn_id=turn_id,
            detail="application_complete",
        )
    speech = spoken or app_question_for_step(next_step)
    ctrl.last_speech = speech
    return TurnResult(speech=speech, phase="application", turn_id=turn_id)


async def process_user_turn(
    call_id: str,
    utterance: str,
    *,
    response_id: int | None = None,
    form_url: str = "",
    transcript_user_index: int | None = None,
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

        # Go-back for wrong email must win even while Playwright auth is running,
        # and also while we are only waiting on confirm (auth flags may be false).
        if (
            text
            and ctrl.phase in ("email", "email_confirm", "email_await_secure")
            and _wants_email_go_back(text)
        ):
            ctrl.last_user_fp = fp
            speech = _reset_email_entry(ctrl)
            return TurnResult(
                speech=speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="email_go_back",
                end_call=False,
            )

        if ctrl.sso_in_progress or ctrl.email_auth_in_progress:
            # Context/repeat during auth: explain what is happening, do not advance.
            if text and detect_control_intent(text).kind == "repeat":
                if ctrl.email_auth_in_progress or ctrl.phase in (
                    "email",
                    "email_confirm",
                    "email_await_secure",
                ):
                    speech = (
                        f"{CALL_PURPOSE} I'm entering your email now. "
                        "Please finish any password or verification steps in the "
                        "Pointcare window."
                    )
                    detail = "email_auth_repeat"
                else:
                    speech = (
                        f"{CALL_PURPOSE} I'm signing you in with Google now. "
                        "Please finish any prompts in the Pointcare window."
                    )
                    detail = "sso_repeat"
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail=detail,
                )
            # Do not treat caller speech as an eligibility answer during auth.
            if ctrl.email_auth_in_progress or ctrl.phase in (
                "email",
                "email_confirm",
                "email_await_secure",
            ):
                speech = EMAIL_BUSY if text else EMAIL_ACK
                detail = "email_auth_in_progress"
            else:
                speech = SSO_BUSY if text else SSO_ACK
                detail = "sso_in_progress"
            ctrl.last_speech = speech
            return TurnResult(
                speech=speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail=detail,
            )

        _pre_auth_phases = (
            "sign_in",
            "google_email",
            "language",
            "email",
            "email_confirm",
            "email_await_secure",
        )

        if not text:
            # Silence / reminder path: if SSO/email auth finished, start Stage D.
            if ctrl.phase in _pre_auth_phases:
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
            if ctrl.phase == "closing":
                return TurnResult(
                    speech=ctrl.last_speech or CLOSING,
                    phase="closing",
                    turn_id=turn_id,
                    detail="closing",
                    end_call=True,
                )
            return TurnResult(
                speech=ctrl.last_speech or "Could you say that again?",
                phase=ctrl.phase,
                turn_id=turn_id,
            )

        # Application test mode: a saved household is already on programs or
        # Stage E. Do not treat the caller's first words as a language answer.
        if config.APPLICATION_TEST_MODE:
            test_session = await get_session(call_id)
            if test_session and (
                test_session.elig_step == "programs" or test_session.app_step
            ):
                if test_session.app_step:
                    ctrl.phase = "application"
                else:
                    ctrl.phase = "eligibility"
                    ctrl.eligibility_screen_verified = True
                opening = (test_session.filled.get("application_test_opening") or "").strip()
                if opening and (
                    not ctrl.last_speech or ctrl.last_speech == LIVE_FILL_GREETING
                ):
                    ctrl.last_speech = opening
                ctrl.eligibility_question_sent = True

        # If the browser already finished sign-in, never re-ask Google/email.
        if ctrl.phase in _pre_auth_phases:
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
        # Same transcript event must not fill the next field after the step moves.
        # A higher user index is a new Retell turn and may repeat the same word.
        if (
            ctrl.phase == "eligibility"
            and transcript_user_index is not None
            and ctrl.consumed_transcript_user_index >= 0
            and transcript_user_index <= ctrl.consumed_transcript_user_index
        ):
            speech = ctrl.last_speech or format_question(elig_step)
            return TurnResult(
                speech=speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="stale_transcript_turn",
            )
        dedupe_ctx = f"{ctrl.phase}:{elig_step}"
        same_words = bool(
            fp
            and fp == ctrl.last_user_fp
            and dedupe_ctx == ctrl.last_dedupe_ctx
            and ctrl.phase not in ("closing",)
        )
        # A new spoken turn may repeat Yes after that confirmation failed.
        # The same Retell user index is already returned as stale_transcript_turn.
        fresh_after_failed = (
            ctrl.phase == "eligibility"
            and ctrl.last_elig_attempt_failed
            and transcript_user_index is not None
            and transcript_user_index > ctrl.consumed_transcript_user_index
        )
        if same_words and not fresh_after_failed:
            return TurnResult(
                speech=ctrl.last_speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="duplicate_utterance",
            )

        # "For what?", "what?", "repeat", etc. — re-explain current step; never advance.
        if detect_control_intent(text).kind == "repeat":
            speech = await _repeat_with_context(call_id, ctrl)
            ctrl.last_speech = speech
            ctrl.last_user_fp = fp
            ctrl.last_dedupe_ctx = dedupe_ctx
            trace_event(
                call_id,
                "context_repeat",
                turn_id=turn_id,
                phase=ctrl.phase,
                step=elig_step or "",
            )
            return TurnResult(
                speech=speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="repeat",
            )

        if ctrl.phase == "language":
            if not is_language_ack(text):
                # Confusion / non-language replies: stay on language with context.
                # Only clear continue-acks (English / yes / ok) advance to sign-in.
                speech = (
                    f"{CALL_PURPOSE} Is it okay to continue in English, or would "
                    "you prefer another language?"
                )
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="language_clarify",
                )
            ctrl.last_user_fp = fp
            ctrl.last_dedupe_ctx = dedupe_ctx
            ctrl.phase = "sign_in"
            ctrl.last_speech = SIGN_IN_QUESTION
            return TurnResult(speech=SIGN_IN_QUESTION, phase=ctrl.phase, turn_id=turn_id)

        if ctrl.phase == "sign_in":
            # Stale / repeated go-back while already on the chooser: re-ask once,
            # never reset the browser or infer a provider.
            if _wants_email_go_back(text):
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                speech = _reset_email_entry(ctrl)
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="email_go_back",
                    end_call=False,
                )

            email_in_utterance = extract_email(text)
            # Address present (e.g. "my email is jane@gmail.com") → confirm before fill.
            # Do not treat "gmail.com" inside an address as Google SSO.
            if email_in_utterance and (
                is_email_choice(text)
                or not is_google_choice(text)
                or "@" in text
                or " at " in f" {text.lower()} "
            ):
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                speech = _begin_email_confirm(ctrl, email_in_utterance)
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="email_confirm",
                    end_call=False,
                )

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

            if is_apple_choice(text):
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                ctrl.pending_op_id = turn_id
                ctrl.sso_in_progress = True
                ctrl.sso_ack_sent = True
                ctrl.eligibility_question_sent = False
                ctrl.eligibility_screen_verified = False
                speech = "One moment while I sign you in with Apple."
                ctrl.last_speech = speech
                prev = _sso_tasks.get(call_id)
                if prev and not prev.done():
                    prev.cancel()
                _sso_tasks[call_id] = asyncio.create_task(
                    _run_apple_sso_background(
                        call_id,
                        form_url=form_url or config.POINTCARE_URL,
                        turn_id=turn_id,
                    ),
                    name=f"apple-sso-{call_id}",
                )
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="sso_ack_apple",
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
            if _wants_email_go_back(text):
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                speech = _reset_email_entry(ctrl)
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="email_go_back",
                    end_call=False,
                )
            # Already confirmed and handed to the browser — do not re-prompt.
            if ctrl.email_captured and ctrl.email_auth_in_progress:
                speech = ctrl.last_speech or EMAIL_BUSY
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech,
                    phase="email_await_secure",
                    turn_id=turn_id,
                    detail="email_already_captured",
                    end_call=False,
                )
            email = extract_email(text)
            if not email:
                # "use email" alone is not an address — keep the first question.
                if is_email_choice(text) and "at" not in text.lower() and "@" not in text:
                    speech = EMAIL_QUESTION
                    detail = "email_address_needed"
                else:
                    speech = EMAIL_CLARIFY
                    detail = "email_address_unclear"
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail=detail,
                    end_call=False,
                )
            if ctrl.email_fill_retries >= 3:
                speech = (
                    "I'm having trouble with email sign-in. "
                    "You can say Google to try Google sign-in instead."
                )
                ctrl.last_speech = speech
                ctrl.phase = "sign_in"
                ctrl.email_captured = False
                ctrl.pending_email = ""
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="email_retries_exhausted",
                    end_call=False,
                )
            ctrl.last_user_fp = fp
            ctrl.last_dedupe_ctx = dedupe_ctx
            speech = _begin_email_confirm(ctrl, email)
            return TurnResult(
                speech=speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="email_confirm",
                end_call=False,
            )

        if ctrl.phase == "email_confirm":
            if _wants_email_go_back(text):
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                speech = _reset_email_entry(ctrl)
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="email_go_back",
                    end_call=False,
                )
            # Caller dictated a different address instead of yes/no.
            new_email = extract_email(text)
            if new_email and new_email != (ctrl.pending_email or "").lower():
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                speech = _begin_email_confirm(ctrl, new_email)
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="email_confirm_updated",
                    end_call=False,
                )
            if _is_yes(text) or re.search(
                r"\b(correct|confirm|continue|that(?:'s| is) (?:right|correct)|go ahead)\b",
                text,
                re.I,
            ):
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                return _start_confirmed_email_auth(
                    ctrl,
                    call_id=call_id,
                    form_url=form_url or config.POINTCARE_URL,
                    turn_id=turn_id,
                    response_id=response_id,
                )
            # Unclear — repeat the confirm prompt with the pending address.
            speech = _email_confirm_speech(ctrl.pending_email or "")
            if not ctrl.pending_email:
                speech = EMAIL_QUESTION
                ctrl.phase = "email"
            ctrl.last_speech = speech
            return TurnResult(
                speech=speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="email_confirm_repeat",
                end_call=False,
            )

        if ctrl.phase == "email_await_secure":
            # Wrong email after fill started — cancel and re-ask.
            if _wants_email_go_back(text):
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                speech = _reset_email_entry(ctrl)
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="email_go_back",
                    end_call=False,
                )
            # Caller may still be finishing password/OTP in the browser.
            # Accept a new email only if they clearly dictate one again.
            email = extract_email(text)
            if email and not ctrl.email_auth_in_progress:
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                speech = _begin_email_confirm(ctrl, email)
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="email_confirm",
                    end_call=False,
                )
            speech = ctrl.last_speech or EMAIL_BUSY
            ctrl.last_speech = speech
            return TurnResult(
                speech=speech,
                phase=ctrl.phase,
                turn_id=turn_id,
                detail="email_await_secure",
                end_call=False,
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

            # Navigation / repeat must win over a busy browser fill.
            early_intent = detect_control_intent(text) if text else detect_control_intent("")
            # Income period lives on the same page as income_range — "go back to
            # monthly" must not click browser Back; fall through to answer resolve.
            income_period_nav = bool(
                text
                and elig_step == "income_range"
                and (
                    _detect_income_period_switch(text)
                    or _wants_income_period_reselect(text)
                )
            )
            if early_intent.kind == "go_back" and not income_period_nav:
                epoch = _begin_elig_navigation(ctrl)
                ctrl.pending_correction = None
                ctrl.last_user_fp = fp
                ctrl.last_dedupe_ctx = dedupe_ctx
                nav = await navigate_back_one(call_id)
                speech = _strip_ask(str(nav.get("spoken") or ""))
                if nav.get("ok"):
                    ctrl.last_speech = speech
                    trace_event(
                        call_id,
                        "elig_go_back",
                        turn_id=turn_id,
                        epoch=epoch,
                        step=str(nav.get("step") or ""),
                        screen=str(nav.get("screen") or ""),
                        ok=True,
                    )
                    return TurnResult(
                        speech=speech,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="go_back",
                    )
                ctrl.last_speech = speech or format_question(
                    str(nav.get("step") or elig_step or "marital")
                )
                trace_event(
                    call_id,
                    "elig_go_back",
                    turn_id=turn_id,
                    epoch=epoch,
                    ok=False,
                    reason=str(nav.get("error") or "failed"),
                )
                return TurnResult(
                    speech=ctrl.last_speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    ok=False,
                    detail="go_back_failed",
                )

            if ctrl.elig_op_in_progress:
                if early_intent.kind in ("repeat", "skip", "next", "redo", "change"):
                    pass  # Fall through — these do not mutate the form yet.
                else:
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
                # "age" / "first name" alone after "which field?" — pin the field.
                if ctrl.pending_correction.get("need_field") or not (
                    ctrl.pending_correction.get("field") or ""
                ).strip():
                    named = _field_from_confirm_utterance(text) or (
                        intent_peek.target
                        if intent_peek.kind == "change"
                        else ""
                    )
                    if named and not intent_peek.value:
                        prompt = _correction_prompt(named)
                        ctrl.pending_correction = {
                            "field": named,
                            "prompt": prompt,
                        }
                        ctrl.last_speech = prompt
                        return TurnResult(
                            speech=prompt,
                            phase=ctrl.phase,
                            turn_id=turn_id,
                            detail="correction_prompt",
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
                    # Same-form corrections (child/spouse/person): never click Back.
                    if _same_member_form_step(step) or field in (
                        "first_name",
                        "last_name",
                        "name",
                        "age",
                        "relationship",
                    ):
                        if field in ("name",):
                            field = "first_name"
                        if field in ("first_name", "last_name"):
                            value = sanitize_person_name(value) or value
                        ctrl.last_user_fp = fp
                        ctrl.last_dedupe_ctx = dedupe_ctx
                        edit = await correct_member_field(call_id, field, value)
                        speech = _strip_ask(str(edit.get("spoken") or ""))
                        if edit.get("ok"):
                            ctrl.pending_correction = None
                            session = await get_session(call_id)
                            step = (
                                str(edit.get("step") or "")
                                or (session.elig_step if session else step)
                                or step
                            )
                        else:
                            # Keep pending so the caller can retry the value.
                            ctrl.pending_correction = {
                                "field": field,
                                "prompt": speech
                                or _correction_prompt(field),
                            }
                        ctrl.last_speech = speech
                        return TurnResult(
                            speech=speech or _correction_prompt(field),
                            phase=ctrl.phase,
                            turn_id=turn_id,
                            ok=bool(edit.get("ok")),
                            detail="member_corrected"
                            if edit.get("ok")
                            else "member_correct_failed",
                        )
                    ctrl.pending_correction = None
                    target_step = _step_for_correction_field(field, step)
                    # Only leave the form when the target is a different page group.
                    if _step_group_safe(step) != _step_group_safe(target_step):
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
                    else:
                        step = target_step
                        session = await get_session(call_id)
                        if session is not None:
                            session.elig_step = target_step
                    meta = get_field_meta(step)
                    text = value
                    # Fall through to normal resolve/submit for target_step.

            intent = detect_control_intent(text)
            if intent.kind in ("repeat", "list_options"):
                if step == "programs":
                    session_p = await get_session(call_id)
                    speech = (
                        (
                            (session_p.filled.get("programs_permission_speech") or "")
                            if session_p
                            else ""
                        ).strip()
                        or format_question("programs")
                    )
                elif intent.kind == "list_options" or step in (
                    "child_rel",
                    "person_rel",
                    "marital",
                    "spouse_role",
                    "income_when",
                ):
                    from .live_form import build_options_help_speech

                    labels = await peek_choice_labels(call_id)
                    if step == "income_range":
                        period_for_q = (
                            (session.filled.get("income_when") or "Annual")
                            if session
                            else "Annual"
                        )
                        speech = format_question(
                            step, period=period_for_q
                        )
                    else:
                        speech = build_options_help_speech(step, labels or None)
                else:
                    labels = await peek_choice_labels(call_id)
                    period_for_q = None
                    if step == "income_range" and session is not None:
                        period_for_q = session.filled.get("income_when") or "Annual"
                    speech = format_question(
                        step, options=labels or None, period=period_for_q
                    )
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech,
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail="list_options" if intent.kind == "list_options" else "repeat",
                )
            # Program-summary contextual questions — never Start Applications.
            if step == "programs" and text:
                from .app_stage import (
                    programs_context_reply,
                    summary_from_session_filled,
                )

                session_p = await get_session(call_id)
                summary = summary_from_session_filled(
                    session_p.filled if session_p else {}
                )
                ctx = programs_context_reply(text, summary)
                if ctx:
                    ctrl.last_speech = ctx
                    return TurnResult(
                        speech=ctx,
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        detail="programs_context",
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
                # Same-page income period change is not browser Back.
                if step == "income_range" and (
                    _detect_income_period_switch(text)
                    or _wants_income_period_reselect(text)
                ):
                    pass  # Fall through to resolve_answer / submit.
                else:
                    # Handled earlier so pending browser fills cannot block navigation.
                    epoch = _begin_elig_navigation(ctrl)
                    ctrl.pending_correction = None
                    nav = await navigate_back_one(call_id)
                    speech = _strip_ask(str(nav.get("spoken") or ""))
                    if nav.get("ok"):
                        ctrl.last_speech = speech
                        trace_event(
                            call_id,
                            "elig_go_back",
                            turn_id=turn_id,
                            epoch=epoch,
                            step=str(nav.get("step") or ""),
                            ok=True,
                        )
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
                        "Which field should I change — first name, last name, "
                        "relationship, or age?"
                    )
                    ctrl.pending_correction = {
                        "field": "",
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
                # In-place member-form correction — never Back through children/marital.
                if _same_member_form_step(step) or field in (
                    "first_name",
                    "last_name",
                    "age",
                    "relationship",
                ):
                    value = intent.value
                    if field in ("first_name", "last_name"):
                        value = sanitize_person_name(value) or value
                    ctrl.last_user_fp = fp
                    ctrl.last_dedupe_ctx = dedupe_ctx
                    edit = await correct_member_field(call_id, field, value)
                    speech = _strip_ask(str(edit.get("spoken") or ""))
                    if edit.get("ok"):
                        ctrl.pending_correction = None
                    else:
                        ctrl.pending_correction = {
                            "field": field,
                            "prompt": speech or _correction_prompt(field),
                        }
                    ctrl.last_speech = speech
                    return TurnResult(
                        speech=speech or _correction_prompt(field),
                        phase=ctrl.phase,
                        turn_id=turn_id,
                        ok=bool(edit.get("ok")),
                        detail="member_corrected"
                        if edit.get("ok")
                        else "member_correct_failed",
                    )
                # Outside member forms: map to the target step without blind Back loops
                # when already on the same questionnaire group.
                target_step = _step_for_correction_field(field, step)
                if _step_group_safe(step) != _step_group_safe(target_step):
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
                else:
                    step = target_step
                    session = await get_session(call_id)
                    if session is not None:
                        session.elig_step = target_step
                meta = get_field_meta(step)
                text = intent.value

            labels = await peek_choice_labels(call_id)
            # Age steps: never expose First Name / Spouse captions as choices.
            resolve_opts = labels or None
            if step in ("spouse_age", "child_age", "person_age"):
                resolve_opts = _age_option_labels(labels) or None
            # Application permission gate is Yes/No only — ignore dashboard CTAs.
            if step in (
                "proceed_application",
                "confirm_start_application",
                "programs",
                "spouse_confirm",
                "child_confirm",
                "person_confirm",
            ):
                resolve_opts = None
            # Custom "Other" relationship text — accept free text, never rematch chips.
            awaiting_other = (
                (session.filled.get("rel_other_awaiting") or "")
                if session is not None
                else ""
            )
            if awaiting_other in ("person_rel", "child_rel") and step == awaiting_other:
                cleaned_other = re.sub(r"\s+", " ", (text or "").strip()).strip(" .")
                if cleaned_other and detect_control_intent(cleaned_other).kind == "none":
                    resolved = ResolveResult(
                        answer=cleaned_other,
                        detail="relationship_other_text",
                    )
                else:
                    resolved = resolve_answer(
                        step, text, options=resolve_opts, meta=meta
                    )
            else:
                resolved = resolve_answer(
                    step, text, options=resolve_opts, meta=meta
                )
            answer, clarify = resolved.answer, resolved.clarify
            resolver_used = "python"
            # Prefer the selected income period in voice clarifies / questions.
            if step == "income_range" and session is not None:
                period_now = session.filled.get("income_when") or "Annual"
                if resolved.detail == "income_scale_unclear":
                    from .live_form import _income_scale_clarify_speech

                    clarify = _income_scale_clarify_speech(text, period_now)
                elif resolved.detail in (
                    "income_range_ambiguous",
                    "income_range_unclear",
                ):
                    from .live_form import _income_range_natural_clarify

                    clarify = _income_range_natural_clarify(
                        text, period=period_now
                    )
            if clarify == "__REPEAT__":
                period_for_q = None
                if step == "income_range" and session is not None:
                    period_for_q = session.filled.get("income_when") or "Annual"
                speech = format_question(
                    step, options=resolve_opts, period=period_for_q
                )
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
                    # Timeout/error: deterministic local fallback — never invent
                    # options, never append "Options include" to required-field chrome,
                    # never click a low-confidence guess.
                    from .live_form import (
                        _income_range_natural_clarify,
                        build_options_help_speech,
                    )

                    if step == "income_range":
                        period_now = (
                            (session.filled.get("income_when") or "Annual")
                            if session
                            else "Annual"
                        )
                        local_fallback = _income_range_natural_clarify(
                            text, period=period_now
                        )
                    else:
                        local_fallback = build_options_help_speech(
                            step, resolve_opts
                        )
                    if claude.reject_reason in ("timeout", "error"):
                        # Prefer local contextual speech over Claude's delayed prompt.
                        clarify = local_fallback or clarify or format_question(step)
                    else:
                        clarify = (
                            claude.clarify_prompt
                            or local_fallback
                            or clarify
                            or format_question(step)
                        )
                    # Strip any leaked internal validation wording.
                    clarify = re.sub(
                        r"\s*This field is required to continue\.?",
                        "",
                        clarify or "",
                        flags=re.I,
                    ).strip()
                    clarify = re.sub(
                        r"\s*Options include:.*$",
                        "",
                        clarify,
                        flags=re.I,
                    ).strip() or local_fallback or format_question(step)
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
            ctrl.last_elig_attempt_failed = False
            if transcript_user_index is not None and transcript_user_index >= 0:
                ctrl.consumed_transcript_user_index = transcript_user_index

            # Slow browser steps: apply in background. Speak "One moment" only when
            # the operation is still running after a short wait; otherwise the
            # interrupt already delivered the next question (avoid a dead ack).
            if step in _SLOW_ELIG_STEPS:
                ctrl.elig_nav_epoch += 1
                epoch = ctrl.elig_nav_epoch
                ctrl.elig_op_in_progress = True
                _schedule_eligibility_browser(
                    call_id,
                    answer=answer,
                    step=step,
                    turn_id=turn_id,
                    form_url=form_url or config.POINTCARE_URL,
                    commit_key=commit_key,
                    epoch=epoch,
                )
                task = _elig_browser_tasks.get(call_id)
                fast_done = False
                if task is not None:
                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=0.45)
                        fast_done = task.done() and not task.cancelled()
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        fast_done = False
                    except Exception:
                        fast_done = False
                if fast_done and ctrl.elig_nav_epoch == epoch:
                    # Background already interrupted the next/clarify speech.
                    speech = (ctrl.last_speech or "").strip()
                    if speech and speech != ELIG_PROGRESS_ACK:
                        logger.info(
                            "elig_fast_complete call_id=%s step=%s turn_id=%s "
                            "epoch=%s speech=%r",
                            call_id,
                            step,
                            turn_id,
                            epoch,
                            speech[:60],
                        )
                        trace_event(
                            call_id,
                            "elig_fast_complete",
                            turn_id=turn_id,
                            step=step,
                            epoch=epoch,
                            phase=ctrl.phase,
                        )
                        return TurnResult(
                            speech="",
                            phase=ctrl.phase,
                            turn_id=turn_id,
                            detail="elig_fast_complete",
                        )
                ctrl.last_speech = ELIG_PROGRESS_ACK
                logger.info(
                    "elig_ack_sent call_id=%s response_id=%s step=%s turn_id=%s epoch=%s "
                    "answer=%r",
                    call_id,
                    response_id,
                    step,
                    turn_id,
                    epoch,
                    (answer or "")[:40],
                )
                trace_event(
                    call_id,
                    "elig_ack",
                    turn_id=turn_id,
                    step=step,
                    epoch=epoch,
                    answer=(answer or "")[:40],
                    phase=ctrl.phase,
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
            # Explicit empty step closes (decline / submitted). app_* starts Stage E.
            next_step = str(result["step"]) if "step" in result else ""
            if str(next_step).startswith("app_") or result.get("phase") == "application":
                ctrl.phase = "application"
                if not spoken and str(next_step).startswith("app_"):
                    from .app_stage import app_question_for_step

                    speech = app_question_for_step(str(next_step))
                else:
                    speech = spoken or format_question(next_step)
                ctrl.last_speech = speech
                ctrl.last_question_delivery_key = f"sync:stage_e:{next_step}:{turn_id}"
                return TurnResult(
                    speech=speech,
                    phase="application",
                    turn_id=turn_id,
                    detail="stage_e",
                )
            if not next_step:
                ctrl.phase = "closing"
                speech = _closing_speech(spoken)
                ctrl.last_speech = speech
                return TurnResult(
                    speech=speech,
                    end_call=bool(result.get("end_call", True)),
                    phase=ctrl.phase,
                    turn_id=turn_id,
                    detail=str(result.get("action") or "eligibility_complete"),
                )

            speech = spoken or format_question(next_step)
            ctrl.last_speech = speech
            if result.get("await_field_correction"):
                ctrl.pending_correction = {
                    "field": "",
                    "prompt": speech,
                    "need_field": True,
                }
            ctrl.last_question_delivery_key = f"sync:{next_step}:{turn_id}"
            return TurnResult(speech=speech, phase=ctrl.phase, turn_id=turn_id)

        if ctrl.phase == "application":
            return await _process_application_turn(
                call_id,
                ctrl,
                text=text,
                fp=fp,
                dedupe_ctx=dedupe_ctx,
                turn_id=turn_id,
                form_url=form_url,
            )

        # closing / unknown
        return TurnResult(
            speech=ctrl.last_speech or CLOSING,
            end_call=True,
            phase=ctrl.phase,
            turn_id=turn_id,
        )


def _transcript_item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, list):
        bits = []
        for part in content:
            if isinstance(part, dict):
                bits.append(str(part.get("text") or part.get("word") or ""))
            else:
                bits.append(str(part))
        return " ".join(bits).strip()
    return str(content or "").strip()


def latest_user_utterances(transcript: list[Any] | None, *, limit: int = 5) -> list[str]:
    """Return newest-first user utterances from a Retell transcript."""
    if not isinstance(transcript, list):
        return []
    found: list[str] = []
    for item in reversed(transcript):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").lower()
        if role != "user":
            continue
        text = _transcript_item_text(item)
        if text and text != "(unintelligible audio)":
            found.append(text)
            if len(found) >= limit:
                break
    return found


def latest_user_turn(transcript: list[Any] | None) -> tuple[str, int]:
    """Return (newest_user_text, zero-based_user_index) from a cumulative transcript.

    Index counts only user roles so a new spoken turn always advances the index
    even when the spoken text is identical to a prior turn.
    """
    if not isinstance(transcript, list):
        return "", -1
    last_text = ""
    last_index = -1
    user_i = -1
    for item in transcript:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").lower()
        if role != "user":
            continue
        text = _transcript_item_text(item)
        if not text or text == "(unintelligible audio)":
            continue
        user_i += 1
        last_text = text
        last_index = user_i
    return last_text, last_index


def latest_user_utterance(transcript: list[Any] | None) -> str:
    """Newest user line only — never prefer an older go-back over a newer answer."""
    text, _idx = latest_user_turn(transcript)
    return text


@dataclass
class TranscriptTurnDecision:
    """Result of classifying a Retell cumulative transcript for one WS event."""

    utterance: str
    user_index: int
    transcript_kind: str  # cumulative | empty
    dedupe: str  # process | stale_tip | duplicate_response_id | empty
    response_id: int | None = None


def classify_transcript_turn(
    ctrl: CallController,
    transcript: list[Any] | None,
    *,
    response_id: int | None = None,
) -> TranscriptTurnDecision:
    """Decide whether this response_required carries a genuinely new user turn.

    Retell Custom LLM sends the full live transcript (cumulative) on every
    response_required / reminder_required. A new response_id alone is not proof
    of new speech — only an advanced user-turn index (or changed tip text) is.
    """
    if (
        response_id is not None
        and ctrl.last_handled_response_id is not None
        and response_id == ctrl.last_handled_response_id
    ):
        return TranscriptTurnDecision(
            utterance="",
            user_index=ctrl.last_transcript_user_index,
            transcript_kind="cumulative",
            dedupe="duplicate_response_id",
            response_id=response_id,
        )

    text, idx = latest_user_turn(transcript)
    if not text or idx < 0:
        return TranscriptTurnDecision(
            utterance="",
            user_index=-1,
            transcript_kind="empty",
            dedupe="empty",
            response_id=response_id,
        )

    fp = _user_fingerprint(text)
    if (
        idx == ctrl.last_transcript_user_index
        and fp == ctrl.last_transcript_user_fp
        and ctrl.last_transcript_user_index >= 0
    ):
        return TranscriptTurnDecision(
            utterance=text,
            user_index=idx,
            transcript_kind="cumulative",
            dedupe="stale_tip",
            response_id=response_id,
        )

    return TranscriptTurnDecision(
        utterance=text,
        user_index=idx,
        transcript_kind="cumulative",
        dedupe="process",
        response_id=response_id,
    )


def mark_transcript_turn_handled(
    ctrl: CallController,
    decision: TranscriptTurnDecision,
    *,
    response_id: int | None = None,
) -> None:
    """Record that this provider event / transcript tip was handled."""
    rid = response_id if response_id is not None else decision.response_id
    if rid is not None:
        ctrl.last_handled_response_id = rid
    if decision.dedupe == "process" and decision.user_index >= 0:
        ctrl.last_transcript_user_index = decision.user_index
        ctrl.last_transcript_user_fp = _user_fingerprint(decision.utterance)
