"""Per-call Playwright session that types Retell answers into Pointcare in real time."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page

from . import config
from .browser import is_playwright_closed_error, open_page
from .form_fields import ACTION_BUTTONS, SSO_BUTTON_NEEDLES, WELCOME_CLICKS, get_field

logger = logging.getLogger(__name__)


@dataclass
class LiveFormSession:
    call_id: str
    url: str
    page: Page | None = None
    filled: dict[str, str] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    error: str = ""
    welcome_done: bool = False
    popup: Page | None = None
    last_heard: str = ""
    signed_in: bool = False
    elig_step: str = ""
    # Stage E Application Collect Mode step (empty until Start Applications succeeds).
    app_step: str = ""
    after_person: str = ""
    last_applied: str = ""
    last_applied_at: float = 0
    last_result: dict[str, Any] | None = None
    processed_requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_step_answer: tuple[str, str] | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_sessions: dict[str, LiveFormSession] = {}
_sessions_lock = asyncio.Lock()
_active_call_id = ""
# Serialize shared-Chrome startup/welcome so concurrent calls cannot tear down
# each other's Playwright page mid-automation.
_startup_lock = asyncio.Lock()


def _safe_url_for_log(url: str) -> str:
    """Log host+path only — drop query strings that may carry onboarding codes."""
    text = (url or "").strip()
    if not text:
        return ""
    return text.split("?", 1)[0][:160]


def session_ids_with_pages() -> list[str]:
    """Call IDs that currently hold a Playwright Page on the shared CDP client."""
    return [
        cid
        for cid, session in _sessions.items()
        if session.page is not None and session.status not in ("ended", "superseded")
    ]


async def detach_all_pages_for_browser_restart(*, reason: str = "") -> None:
    """Clear session page refs before a forced CDP disconnect (idempotent).

    Do not wait forever for ``session.lock``. ``start_live_form`` already holds
    that lock when it calls ``open_page``, and a Chrome restart from there would
    deadlock and never launch the window.
    """
    async with _sessions_lock:
        items = list(_sessions.items())
    for call_id, session in items:
        acquired = False
        try:
            await asyncio.wait_for(session.lock.acquire(), timeout=0.3)
            acquired = True
        except asyncio.TimeoutError:
            logger.warning(
                "Detach without waiting for session lock call_id=%s reason=%s",
                call_id,
                reason or "-",
            )
        try:
            if session.page is None and session.popup is None:
                continue
            logger.info(
                "Detaching page refs call_id=%s status=%s reason=%s",
                call_id,
                session.status,
                reason or "-",
            )
            session.page = None
            session.popup = None
            if session.status in ("open", "pending"):
                session.status = "error"
                session.error = session.error or f"browser_restart:{reason or 'unknown'}"
        finally:
            if acquired:
                session.lock.release()


async def _detach_session_page(session: LiveFormSession, *, reason: str = "") -> None:
    """Drop this session's page references without closing shared CDP."""
    async with session.lock:
        if session.page is None and session.popup is None:
            return
        logger.info(
            "Detaching session page call_id=%s reason=%s",
            session.call_id,
            reason or "-",
        )
        session.page = None
        session.popup = None


async def _page_wait(page: Page, timeout_ms: int, *, call_id: str = "") -> None:
    """wait_for_timeout with a clear error when the Playwright handler is gone."""
    try:
        if page.is_closed():
            raise RuntimeError(
                f"BROWSER_CLOSED: page already closed"
                + (f" call_id={call_id}" if call_id else "")
            )
        await page.wait_for_timeout(timeout_ms)
    except Exception as exc:
        if is_playwright_closed_error(exc) or (
            isinstance(exc, RuntimeError) and str(exc).startswith("BROWSER_CLOSED:")
        ):
            raise RuntimeError(
                "BROWSER_CLOSED: Playwright page/connection closed during wait"
                + (f" call_id={call_id}" if call_id else "")
            ) from exc
        raise

# Stage D Collect Mode — spoken after each Playwright fill. Server owns order/branching.
# Stage D Collect Mode wording (ask, don't guide/tap).
# Stage E Application continues after the program-summary permission ask (see app_stage).
# Spoken once when Stage D opens — purpose first, then the first question.
ELIGIBILITY_SECTION_INTRO = (
    "Let's start your household eligibility check. "
    "I'll ask a few short questions about your household so Pointcare can see "
    "what programs you may qualify for. "
)

ELIG_QUESTIONS: dict[str, str] = {
    "marital": (
        "What is your marital status? Never Married, Married, Separated, Divorced, "
        "Widowed, Registered Domestic Partner, or Common Law Marriage?"
    ),
    "spouse_first": "What is their first name?",
    "spouse_last": "What is their last name?",
    "spouse_role": "Are they your spouse or your partner?",
    "spouse_age": "What is their age range, or date of birth?",
    "spouse_confirm": (
        "Please confirm those details. Say Yes to continue, or No to change them."
    ),
    "children": (
        "Do you live with and mainly support any children under 19 who live with you "
        "more than half the year? This includes relatives who are not your biological "
        "children. Please say Yes or No."
    ),
    "child_first": "What is the child's first name?",
    "child_last": "What is the child's last name?",
    "child_rel": (
        "What is their relationship to you? The options are Child, Stepchild, "
        "Grandchild, Foster Child, or Other. Which best describes their "
        "relationship to you?"
    ),
    "child_age": "What is the child's age range?",
    "child_confirm": (
        "Please confirm those details. Say Yes to continue, or No to change them."
    ),
    "child_more": "Is there another child to add? Please say Yes or No.",
    "student": (
        "Are you the parent or legal guardian of a full-time student aged 18 to 21 "
        "whom you financially support? Please say Yes or No."
    ),
    "disabled": (
        "Do you financially support a disabled person of any age who lives with you "
        "year-round? Please say Yes or No."
    ),
    "relatives": (
        "Do you support any other relatives who live with you, including in-laws or "
        "step-family? Please say Yes or No."
    ),
    "support": (
        "Does someone provide more than half of your annual financial support? "
        "Please say No, or Yes and which situation fits — student supported by a "
        "parent, you live with the person who supports you, disabled and live with "
        "them year-round, or a relative supports you."
    ),
    "person_first": "What is their first name?",
    "person_last": "What is their last name?",
    "person_rel": (
        "What is their relationship to you? The options are Child, Stepchild, "
        "Grandchild, Foster Child, or Other. Which best describes their "
        "relationship to you?"
    ),
    "person_age": "What is their age range, or date of birth?",
    "person_confirm": (
        "Please confirm those details. Say Yes to continue, or No to change them."
    ),
    "income_when": (
        "For household income, which period do you know best — Annual, Monthly, or "
        "Bi-weekly?"
    ),
    "income_range": (
        "About how much does your household earn before taxes? "
        "You can say an approximate amount, like forty thousand, or a range like "
        "forty to fifty thousand."
    ),
    "special": (
        "Do any of these apply: pregnant, breast cancer, cervical cancer, blind, "
        "disabled on SSDI, was in foster care at 18, or none of the above?"
    ),
    "special_person": "Which household member does each selected condition apply to?",
    "special_more": (
        "Are there any other conditions to add? Please say Yes or No."
    ),
    "babies_count": "How many babies are expected?",
    "due_date": "What is the due date?",
    "programs": (
        "Your household eligibility check is complete. Would you like to continue "
        "and start the application? Please say Yes or No."
    ),
    "proceed_application": (
        "Your household eligibility check is finished. "
        "Can I start the application now? Please say Yes to start, or No to end the call."
    ),
    "confirm_start_application": (
        "Should I start the application now? Please say Yes to start, or No to stop."
    ),
}
YES_NO_ELIG_STEPS = frozenset(
    {
        "children",
        "student",
        "disabled",
        "relatives",
        "support",
        "spouse_confirm",
        "child_confirm",
        "person_confirm",
        "child_more",
        "special_more",
        "programs",
        "proceed_application",
        "confirm_start_application",
    }
)


def classify_yes_no(answer: str) -> str | None:
    """Return 'yes', 'no', or None when mixed/ambiguous (clarify — do not guess)."""
    text = (answer or "").strip().lower()
    if not text:
        return None
    # Navigation / control is never a Yes/No answer.
    if re.search(
        r"\b(go\s*back|goback|previous question|next question|repeat|"
        r"say that again|skip)\b",
        text,
    ):
        return None

    starts_yes = bool(re.match(r"^(yes|yeah|yep|yup|yee|yea)\b", text))
    starts_no = bool(re.match(r"^(no|nope|nah|negative)\b", text))
    if starts_yes and starts_no:
        return None
    # Leading Yes with a later No cue (or contradiction) → clarify.
    if starts_yes:
        rest = text.split(None, 1)[1] if " " in text else ""
        if rest and re.search(r"\b(no|nope|nah|not really|never)\b", rest):
            return None
        # ASR often drops the apostrophe ("dont" / "do not").
        if rest and re.search(r"\bi (do not|don'?t|dont)\b", rest):
            return None
        return "yes"
    if starts_no:
        rest = text.split(None, 1)[1] if " " in text else ""
        if rest and re.search(r"\b(yes|yeah|yep|yup)\b", rest):
            return None
        return "no"

    if re.search(
        r"\b("
        r"(i|he|she|they|we) said no|"
        r"none of (those|them|the above)|"
        r"not really|no thanks"
        r")\b",
        text,
    ):
        if re.search(r"\b(yes|yeah|yep|yup)\b", text):
            return None
        return "no"
    # Bare negation without a leading Yes — conservative "I don't" / "I do not".
    if re.search(r"^(i (do not|don'?t|dont))\b", text):
        return "no"

    if re.search(
        r"\b("
        r"that(?:'s|s)?\s+right|"
        r"that(?:'s|s)?\s+correct|"
        r"affirmative"
        r")\b",
        text,
    ):
        if re.search(r"\b(no|not|don't|do not|never)\b", text):
            return None
        return "yes"
    if re.search(r"\b((i|he|she|they|we) said yes)\b", text):
        return "yes"
    # Soft affirmations only when no negation is present.
    if re.search(r"\b(i do|i have|sure(?:\s+do)?|absolutely|of course)\b", text):
        if re.search(r"\b(no|not|don't|do not|never)\b", text):
            return None
        return "yes"
    if re.fullmatch(r"correct[.!]?", text):
        return "yes"
    return None


def _is_no(answer: str) -> bool:
    return classify_yes_no(answer) == "no"


def _is_yes(answer: str) -> bool:
    return classify_yes_no(answer) == "yes"

def _is_partnered(answer: str) -> bool:
    text = (answer or "").lower()
    if re.search(r"never|separated|divorced|widowed", text):
        return False
    return re.search(r"married|partner|domestic|common law", text) is not None


def _next_elig_step(session: LiveFormSession, answer: str) -> str:
    step = session.elig_step
    yes = _is_yes(answer)
    if step == "marital":
        return "spouse_first" if _is_partnered(answer) else "children"
    if step == "spouse_first":
        return "spouse_last"
    if step == "spouse_last":
        return "spouse_role"
    if step == "spouse_role":
        return "spouse_age"
    if step == "spouse_age":
        return "spouse_confirm"
    if step == "spouse_confirm":
        # No stays on confirm so the caller can name which field to fix.
        return "children" if yes else "spouse_confirm"
    if step == "children":
        return "child_first" if yes else "student"
    if step == "child_first":
        return "child_last"
    if step == "child_last":
        return "child_rel"
    if step == "child_rel":
        return "child_age"
    if step == "child_age":
        return "child_confirm"
    if step == "child_confirm":
        return "child_more" if yes else "child_confirm"
    if step == "child_more":
        return "child_first" if yes else "student"
    if step == "student":
        return _person_or("disabled", session, yes, kind="student")
    if step == "disabled":
        return _person_or("relatives", session, yes, kind="disabled")
    if step == "relatives":
        return _person_or("support", session, yes, kind="relative")
    if step == "support":
        return _person_or("income_when", session, yes, kind="support")
    if step == "person_first":
        return "person_last"
    if step == "person_last":
        return "person_rel"
    if step == "person_rel":
        return "person_age"
    if step == "person_age":
        return "person_confirm"
    if step == "person_confirm":
        return (session.after_person or "income_when") if yes else "person_confirm"
    if step == "income_when":
        return "income_range"
    if step == "income_range":
        return "special"
    if step == "special":
        return "special_person"
    if step == "special_person":
        return "special_more"
    if step == "special_more":
        if yes:
            return "special"
        return (
            "babies_count"
            if "pregnant" in (session.filled.get("special_conditions") or "").lower()
            else "programs"
        )
    if step == "babies_count":
        return "due_date"
    if step == "due_date":
        return "programs"
    return ""


def _person_or(
    after: str, session: LiveFormSession, yes: bool, *, kind: str = ""
) -> str:
    if not yes:
        return after
    session.after_person = after
    if kind:
        session.filled["person_kind"] = kind
    return "person_first"


def _set_person_kind_from_gate(session: LiveFormSession, gate_step: str) -> None:
    """Remember which Yes/No gate opened the generic person form."""
    mapping = {
        "student": "student",
        "disabled": "disabled",
        "relatives": "relative",
        "support": "support",
    }
    kind = mapping.get(gate_step, "")
    if kind:
        session.filled["person_kind"] = kind


def _person_section_intro(session: LiveFormSession) -> str:
    kind = (session.filled.get("person_kind") or "").strip().lower()
    if kind == "disabled":
        return "Let's learn more about the disabled individual you support. "
    if kind == "relative":
        return "Let's learn more about the relatives you support. "
    if kind == "student":
        return "Let's learn more about the student you support. "
    if kind == "support":
        return "Let's learn more about the person who supports you. "
    return "Let's learn more about this household member. "


def _spoken_age_range(label: str) -> str:
    """Natural age speech from a verified chip — never invent an exact age."""
    text = (label or "").strip()
    if not text:
        return "unknown"
    lo, hi = _parse_range_bounds(text)
    name = re.sub(r"^\s*\([^)]+\)\s*", "", text).strip(" -–—")
    if lo is not None and hi is not None:
        band = f"the {lo}-to-{hi} age range"
    elif lo is not None:
        band = f"the {lo}-and-up age range"
    else:
        band = text
    if name and name.lower() not in band.lower():
        return f"{band}, labelled {name}"
    return band


def _member_confirm_speech(session: LiveFormSession, kind: str) -> str:
    """Read back first name, last name, relationship, and age before Continue."""
    if kind == "spouse":
        first = (session.filled.get("spouse_first") or "").strip() or "unknown"
        last = (session.filled.get("spouse_last") or "").strip() or "unknown"
        rel = (session.filled.get("spouse_role") or "").strip() or "spouse or partner"
        age_raw = (session.filled.get("spouse_age") or "").strip()
        who = "your spouse or partner"
    elif kind == "child":
        first = (session.filled.get("child_first") or "").strip() or "unknown"
        last = (session.filled.get("child_last") or "").strip() or "unknown"
        rel = (session.filled.get("child_rel_other") or "").strip() or (
            (session.filled.get("child_rel") or "").strip() or "child"
        )
        age_raw = (session.filled.get("child_age") or "").strip()
        who = "the child"
    else:
        first = (session.filled.get("person_first") or "").strip() or "unknown"
        last = (session.filled.get("person_last") or "").strip() or "unknown"
        rel = (session.filled.get("person_rel_other") or "").strip() or (
            (session.filled.get("person_rel") or "").strip() or "relative"
        )
        age_raw = (session.filled.get("person_age") or "").strip()
        who = "this household member"
        person_kind = (session.filled.get("person_kind") or "").strip().lower()
        if person_kind == "disabled":
            who = "the disabled individual"
        elif person_kind == "relative":
            who = "the relative"
        elif person_kind == "student":
            who = "the student"
        elif person_kind == "support":
            who = "the person who supports you"
    age = _spoken_age_range(age_raw) if age_raw else "unknown"
    return (
        f"Just to confirm for {who}: first name {first}, last name {last}, "
        f"relationship {rel}, age {age}. Is that correct? Please say Yes or No."
    )


def _confirm_which_field_speech() -> str:
    return (
        "Okay. Which detail should I change — first name, last name, "
        "relationship, or age?"
    )


def _member_kind_from_step(step: str) -> str:
    if str(step or "").startswith("spouse_"):
        return "spouse"
    if str(step or "").startswith("person_"):
        return "person"
    return "child"


def _confirm_step_for_kind(kind: str) -> str:
    return {
        "spouse": "spouse_confirm",
        "person": "person_confirm",
        "child": "child_confirm",
    }.get(kind, "child_confirm")


def _result_with_next(session: LiveFormSession, answer: str, result: dict[str, Any]) -> dict[str, Any]:
    nxt = _after_clicked(session, answer)
    if nxt:
        result["spoken"] = "ASK: " + nxt
    return result


def _after_clicked(session: LiveFormSession, answer: str) -> str:
    """Next spoken question. Empty until eligibility has started."""
    if not session.elig_step or session.elig_step not in ELIG_QUESTIONS:
        return ""
    nxt = _next_elig_step(session, answer)
    session.elig_step = nxt
    if not nxt:
        return (
            "I've finished filling your household eligibility form. "
            "Thank you for your time. Goodbye."
        )
    return _spoken_question_for_step(session, nxt)


async def start_live_form(call_id: str, url: str = "") -> LiveFormSession:
    """Open/reuse the shared agent Chrome for one call and advance welcome screens.

    Startup is serialized: only one call may open/navigate/welcome on the shared
    CDP tab at a time. Superseded sessions drop page refs without closing CDP.
    """
    global _active_call_id
    if not call_id:
        raise ValueError("call_id is required")
    form_url = (url or "").strip() or config.POINTCARE_URL

    async with _startup_lock:
        previous: LiveFormSession | None = None
        async with _sessions_lock:
            session = _sessions.get(call_id)
            if session is None:
                if _active_call_id and _active_call_id != call_id:
                    previous = _sessions.get(_active_call_id)
                    if previous is not None:
                        previous.status = "superseded"
                session = LiveFormSession(call_id=call_id, url=form_url)
                _sessions[call_id] = session
                _active_call_id = call_id
                logger.info(
                    "Session registered call_id=%s previous=%s",
                    call_id,
                    previous.call_id if previous else "-",
                )
            elif session.status == "superseded":
                raise RuntimeError("This call was superseded by a newer browser session")
            elif form_url and not session.url:
                session.url = form_url

        if previous is not None:
            await _detach_session_page(previous, reason=f"superseded_by:{call_id}")

        async with session.lock:
            if session.status == "superseded":
                raise RuntimeError("This call was superseded by a newer browser session")
            # Once welcome finished for this call, never clear cookies or re-click the
            # 4 onboarding CTAs — even if the tab drifted back to Get Started or the
            # Page ref was dropped (CDP restart). That loop is what callers see as
            # "landing refreshes and does the 4 clicks again while the call is on".
            if session.welcome_done and session.ready.is_set():
                page = session.page
                if page is not None:
                    try:
                        if page.is_closed():
                            page = None
                            session.page = None
                    except Exception:
                        page = None
                        session.page = None
                if page is None:
                    page = await open_page(None)
                    session.page = page
                    logger.info(
                        "Session reattached without re-welcome call_id=%s url=%s",
                        call_id,
                        _safe_url_for_log(page.url or ""),
                    )
                await _recover_signin_surface_if_needed(page, call_id=call_id)
                session.status = "open"
                session.error = ""
                session.url = page.url or session.url
                logger.info("Session already ready call_id=%s", call_id)
                return session
            page: Page | None = session.page
            try:
                if page is None:
                    target = session.url or config.POINTCARE_URL
                    if not target:
                        raise RuntimeError("POINTCARE_URL is empty. Set it in .env.")
                    logger.info(
                        "Opening Pointcare call_id=%s url=%s",
                        call_id,
                        _safe_url_for_log(target),
                    )
                    # Full call: one navigation to the onboarding page, then the
                    # four welcome clicks. Do not reload that page before sign-in.
                    # Application test / after-signin keep the saved account path.
                    if (
                        not config.APPLICATION_TEST_MODE
                        and not config.APPLICATION_AFTER_SIGNIN
                    ):
                        page = await open_page(None)
                        session.page = page
                        await _clear_pointcare_session(page)
                        await page.goto(
                            target,
                            wait_until="domcontentloaded",
                            timeout=45000,
                        )
                    else:
                        page = await open_page(target)
                        session.page = page
                        await _page_wait(page, 800, call_id=call_id)
                    session.url = page.url or target
                if (
                    config.APPLICATION_AFTER_SIGNIN
                    and not config.APPLICATION_TEST_MODE
                    and not _on_welcome_landing(page.url)
                ):
                    await _wait_for_saved_household(page, call_id=call_id)
                if (
                    config.APPLICATION_AFTER_SIGNIN
                    and not config.APPLICATION_TEST_MODE
                    and not _on_welcome_landing(page.url)
                    and await _application_test_authenticated(page)
                ):
                    # Saved account stays put. Spoken intro and sign-in still run.
                    # Navigation to the application happens only after sign-in.
                    session.error = ""
                    session.welcome_done = True
                    session.status = "open"
                    session.url = page.url or session.url
                    session.ready.set()
                    logger.info(
                        "Saved household ready for post-sign-in application call_id=%s url=%s",
                        call_id,
                        _safe_url_for_log(session.url),
                    )
                    return session
                if config.APPLICATION_TEST_MODE:
                    await _wait_for_saved_household(page, call_id=call_id)
                    if await _application_test_authenticated(page):
                        return await _finish_application_test_session(session, page)
                if not session.welcome_done:
                    # Full call must never begin welcome clicks from the saved
                    # household dashboard — force welcome page 1 first.
                    if (
                        not config.APPLICATION_TEST_MODE
                        and not config.APPLICATION_AFTER_SIGNIN
                        and await _on_saved_household_surface(page)
                    ):
                        raise RuntimeError(
                            "Pointcare opened the saved household (Quick Apply) "
                            "instead of welcome/sign-in. "
                            + _safe_url_for_log(page.url or "")
                        )
                    logger.info("Welcome screens start call_id=%s", call_id)
                    clicked = await advance_welcome_screens(page, call_id=call_id)
                    if "/auth" in (page.url or ""):
                        await settle_on_signin_page(page, call_id=call_id)
                    elif (
                        config.APPLICATION_AFTER_SIGNIN
                        and await _application_test_authenticated(page)
                    ):
                        session.welcome_done = True
                        session.status = "open"
                        session.url = page.url or session.url
                        session.ready.set()
                        return session
                    elif config.APPLICATION_TEST_MODE and await _application_test_authenticated(
                        page
                    ):
                        return await _finish_application_test_session(session, page)
                    elif await _already_signed_in(page) or await _on_saved_household_surface(
                        page
                    ):
                        raise RuntimeError(
                            "Pointcare bypassed the fresh sign-in page with a saved session"
                        )
                    else:
                        raise RuntimeError(
                            "Welcome flow did not reach the sign-in page: "
                            + _safe_url_for_log(page.url or "")
                        )
                    session.error = ""
                    session.history.append(
                        {
                            "field_id": "welcome",
                            "value": " → ".join(clicked)
                            or _safe_url_for_log(page.url or ""),
                            "action": "wait_for_caller",
                            "ok": True,
                        }
                    )
                    session.welcome_done = True
                    session.url = page.url or session.url
                session.status = "open"
                session.ready.set()
                logger.info(
                    "Live Pointcare session opened call_id=%s url=%s welcome_done=%s err=%s",
                    call_id,
                    _safe_url_for_log(session.url),
                    session.welcome_done,
                    session.error or "-",
                )
            except Exception as exc:
                session.error = f"welcome: {exc}"
                session.status = "error"
                logger.exception(
                    "Session startup failed call_id=%s url=%s",
                    call_id,
                    _safe_url_for_log(
                        getattr(page, "url", "") if page is not None else session.url
                    ),
                )
                # Drop dead page refs so APIs do not advertise a live browser.
                if is_playwright_closed_error(exc) or "BROWSER_CLOSED" in str(exc):
                    session.page = None
                    session.popup = None
                else:
                    try:
                        if page is not None and page.is_closed():
                            session.page = None
                            session.popup = None
                    except Exception:
                        session.page = None
                        session.popup = None
                raise
        return session


def _past_welcome(url: str) -> bool:
    lowered = (url or "").lower()
    return any(
        part in lowered
        for part in ("/auth", "/registration", "/dashboard", "/eligibility", "/application")
    )


def _on_welcome_landing(url: str) -> bool:
    """True on the marketing onboarding URL, before Get Started reaches sign-in.

    That page mentions health insurance, so it must not be treated as a saved
    application screen.
    """
    lowered = (url or "").lower()
    return "/onboarding" in lowered and not _past_welcome(lowered)


async def advance_welcome_screens(page: Page, *, call_id: str = "") -> list[str]:
    """Click through the 4 pre-profile pages. Page 3 (frustration) is skipped via Continue."""
    clicked: list[str] = []
    if _past_welcome(page.url):
        logger.info(
            "Already past welcome call_id=%s url=%s",
            call_id or "-",
            _safe_url_for_log(page.url),
        )
        return clicked
    for names in WELCOME_CLICKS:
        label = names[0]
        if _past_welcome(page.url):
            break
        try:
            logger.info("Welcome CTA call_id=%s label=%s", call_id or "-", label)
            await _click_cta(page, label, attempts=3)
        except Exception as exc:
            if is_playwright_closed_error(exc):
                raise RuntimeError(
                    f"BROWSER_CLOSED: Playwright closed while clicking {label!r}"
                    + (f" call_id={call_id}" if call_id else "")
                ) from exc
            if _past_welcome(page.url) or clicked:
                logger.info(
                    "Welcome button %s not shown call_id=%s url=%s",
                    label,
                    call_id or "-",
                    _safe_url_for_log(page.url),
                )
                break
            raise
        clicked.append(label)
        await _page_wait(page, 1400, call_id=call_id)
        logger.info(
            "Welcome step clicked call_id=%s label=%s url=%s",
            call_id or "-",
            label,
            _safe_url_for_log(page.url),
        )
    return clicked


async def settle_on_signin_page(page: Page, *, call_id: str = "") -> None:
    """Stop after Create My Profile. Do not click Google / Apple / saved email."""
    await _page_wait(page, 800, call_id=call_id)
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass
    await _dismiss_account_picker(page)
    await _clear_autofilled_email(page)
    logger.info(
        "Sign-in page ready call_id=%s url=%s",
        call_id or "-",
        _safe_url_for_log(page.url),
    )


async def _welcome_get_started_visible(page: Page) -> bool:
    """True on marketing welcome page 1 (Get Started) — not Quick Apply /auth.

    Uses content only (no ``_already_signed_in``) so detectors cannot recurse.
    """
    if not _on_welcome_landing(page.url):
        return False
    try:
        buttons = (await _visible_button_text(page)).lower()
    except Exception:
        buttons = ""
    body = await _page_body_lower_snippet(page)
    if _text_looks_like_saved_dashboard(body, buttons):
        return False
    return "get started" in buttons or "get started" in body


async def _recover_signin_surface_if_needed(page: Page, *, call_id: str = "") -> None:
    """If welcome already ran but the tab is back on Get Started, open /auth once.

    Never clears cookies or re-runs the 4 welcome CTAs mid-call.
    """
    if not await _welcome_get_started_visible(page):
        return
    logger.warning(
        "Welcome landing reappeared mid-call; opening /auth without re-welcome "
        "call_id=%s url=%s",
        call_id or "-",
        _safe_url_for_log(page.url),
    )
    try:
        await page.goto(
            "https://my.pointcare.com/auth",
            wait_until="domcontentloaded",
            timeout=45000,
        )
        await settle_on_signin_page(page, call_id=call_id)
    except Exception:
        logger.exception(
            "Could not recover sign-in surface call_id=%s", call_id or "-"
        )


async def _dismiss_account_picker(page: Page) -> None:
    """Close Chrome/Google 'Use this account' chips without choosing one."""
    dismiss = [
        page.get_by_role("button", name=re.compile(r"^(close|dismiss|not now|no thanks|use another)$", re.I)),
        page.locator("[aria-label='Close']"),
        page.locator("[aria-label='Dismiss']"),
    ]
    for loc in dismiss:
        try:
            target = loc.first
            if await target.count() and await target.is_visible():
                await target.click(timeout=1500)
                await page.wait_for_timeout(200)
        except Exception:
            continue


async def _clear_autofilled_email(page: Page) -> None:
    try:
        await page.evaluate(
            """() => {
              const boxes = [...document.querySelectorAll('input')].filter((el) => {
                const type = (el.getAttribute('type') || 'text').toLowerCase();
                const name = ((el.getAttribute('name') || '') + (el.getAttribute('id') || '') + (el.getAttribute('placeholder') || '') + (el.getAttribute('autocomplete') || '')).toLowerCase();
                return type === 'email' || name.includes('email');
              });
              for (const el of boxes) {
                el.setAttribute('autocomplete', 'off');
                el.value = '';
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
              }
            }"""
        )
    except Exception:
        pass


def _is_sso_label(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(needle in lowered for needle in SSO_BUTTON_NEEDLES)


def _normalize_email(text: str) -> str:
    """Delegate to answer_parse (lazy import avoids circular import at load)."""
    from .answer_parse import normalize_spoken_email

    return normalize_spoken_email(text)


def _extract_email(text: str) -> str:
    from .answer_parse import extract_email as _extract_email_shared

    return _extract_email_shared(text)


def _front_page(session: LiveFormSession) -> Page:
    popup = session.popup
    if popup is not None:
        try:
            if not popup.is_closed():
                return popup
        except Exception:
            pass
        session.popup = None
    if session.page is None:
        raise RuntimeError("form page not ready")
    return session.page


def _ok(call_id: str, field_key: str, text: str, act: str, session: LiveFormSession, msg: str) -> dict[str, Any]:
    return {
        "ok": True,
        "call_id": call_id,
        "field_id": field_key,
        "value": text,
        "action": act,
        "filled": dict(session.filled),
        "message": msg,
        "spoken": "Saved.",
    }


def _provider_popup(page: Page) -> Page | None:
    for candidate in page.context.pages:
        if candidate == page:
            continue
        try:
            if candidate.is_closed():
                continue
        except Exception:
            continue
        url = candidate.url or ""
        if "accounts.google." in url or "appleid.apple.com" in url or "/__/auth/handler" in url:
            return candidate
    return None


async def _post_auth_session_ready(page: Page) -> bool:
    """True when the main page is signed in enough to leave the Google chooser.

    Production uses the strict detector. Application test mode also accepts the
    saved dashboard / program / application surfaces (Quick Apply, etc.).
    """
    if await _already_signed_in(page):
        return True
    if config.APPLICATION_TEST_MODE or config.APPLICATION_AFTER_SIGNIN:
        return await _application_test_authenticated(page)
    return False


async def _sign_in_with_provider(session: LiveFormSession, button_name: str, email: str) -> str:
    """Click Continue with Google/Apple and pick the account they named."""
    page = session.page
    if page is None:
        raise RuntimeError("form page not ready")
    # OAuth cancel / session drop can bounce the shared tab to Get Started. Go to
    # /auth once — do not clear cookies or replay the 4 welcome clicks.
    if session.welcome_done and await _welcome_get_started_visible(page):
        await _recover_signin_surface_if_needed(page, call_id=session.call_id)
        session.url = page.url or session.url
    if config.APPLICATION_AFTER_SIGNIN:
        session.filled["signin_choice_done"] = "1"
        if session.elig_step in ("proceed_application", "confirm_start_application"):
            speech = (session.filled.get("application_gate_speech") or "").strip()
            return "ASK: " + speech
        if session.app_step:
            from .app_stage import app_question_for_step

            return "ASK: " + app_question_for_step(session.app_step, session)
        if await _post_auth_session_ready(page):
            return await _open_saved_application(session, page)
    if session.signed_in and session.elig_step:
        screen = await _eligibility_screen(page)
        recovered = _screen_first_step(screen, session)
        session.elig_step = recovered
        return "ASK: " + ELIG_QUESTIONS[recovered]
    if await _post_auth_session_ready(page):
        return await _begin_eligibility(session, page)
    popup = _provider_popup(page)
    if popup is None:
        try:
            async with page.expect_popup(timeout=12000) as info:
                await _click_labeled(page, button_name)
            popup = await info.value
        except Exception:
            logger.exception("Provider popup did not open for %s on %s", button_name, page.url)
            popup = _provider_popup(page)
    if popup is None:
        if await _post_auth_session_ready(page):
            if config.APPLICATION_AFTER_SIGNIN:
                return await _open_saved_application(session, page)
            return await _begin_eligibility(session, page)
        visible = await _visible_button_text(page)
        logger.error("Sign-in button missing url=%s buttons=%s", page.url, visible)
        raise RuntimeError(f"{button_name} was not on the page. Buttons: {visible}")
    session.popup = popup
    try:
        await popup.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception:
        pass
    await popup.wait_for_timeout(800)
    for _ in range(10):
        if "accounts.google." in (popup.url or ""):
            break
        try:
            await popup.wait_for_timeout(300)
        except Exception:
            break
    wanted = _extract_email(email)
    if "accounts.google." in (popup.url or "") and not wanted:
        accounts: list[str] = []
        for _ in range(8):
            accounts = await _google_account_emails(popup)
            if accounts:
                break
            try:
                await popup.wait_for_timeout(400)
            except Exception:
                break
        logger.info("Google chooser accounts=%s", accounts)
        if len(accounts) == 1:
            wanted = accounts[0]
            await _choose_google_account(popup, wanted)
        elif len(accounts) > 1:
            return "ASK: Which Google email should we use? " + " or ".join(accounts)
        else:
            logger.info("Google chooser has no readable accounts yet; waiting for sign-in state")
    elif "accounts.google." in (popup.url or "") and wanted:
        await _choose_google_account(popup, wanted)

    # Do not report success merely because an account was selected. OAuth can
    # leave the chooser open while the main page is still unauthenticated.
    for attempt in range(50):
        if await _post_auth_session_ready(page):
            session.popup = None
            logger.info("Google sign-in verified on main page url=%s", page.url)
            if config.APPLICATION_AFTER_SIGNIN:
                return await _open_saved_application(session, page)
            return await _begin_eligibility(session, page)
        try:
            if popup.is_closed():
                session.popup = None
                popup = None
            else:
                url = popup.url or ""
                if "accounts.google." in url and not wanted:
                    accounts = await _google_account_emails(popup)
                    if len(accounts) == 1:
                        wanted = accounts[0]
                        await _choose_google_account(popup, wanted)
                    elif len(accounts) > 1:
                        return "ASK: Which Google email should we use? " + " or ".join(accounts)
                    elif attempt >= 20:
                        email_box = popup.locator(
                            "#identifierId, input[type='email']"
                        ).first
                        if await email_box.count() and await email_box.is_visible():
                            return "ASK: Which Google email should we use?"
                if (
                    "accounts.google." in url
                    or "appleid.apple.com" in url
                    or "/__/auth/handler" in url
                ):
                    for pattern in (r"^\s*Continue\s*$", r"^\s*Allow\s*$"):
                        try:
                            btn = popup.get_by_role(
                                "button", name=re.compile(pattern, re.I)
                            ).first
                            if await btn.count() and await btn.is_visible():
                                await btn.click(timeout=2000)
                                break
                        except Exception:
                            continue
        except Exception:
            session.popup = None
        try:
            await page.wait_for_timeout(500)
        except Exception:
            break

    if await _post_auth_session_ready(page):
        if config.APPLICATION_AFTER_SIGNIN:
            return await _open_saved_application(session, page)
        return await _begin_eligibility(session, page)
    logger.error(
        "Google account selected but sign-in was not verified main_url=%s popup_url=%s",
        page.url,
        popup.url if popup is not None else "closed",
    )
    raise RuntimeError(
        "Google sign-in has not finished. Please complete the visible Google prompt, then say Google again."
    )


async def _open_saved_application(session: LiveFormSession, page: Page) -> str:
    """Continue the saved account. Does not open or reset Stage D."""
    from .app_stage import begin_application_after_signin

    return await begin_application_after_signin(session, page)


async def _begin_eligibility(session: LiveFormSession, page: Page) -> str:
    """Open Check Household Eligibility, then return the first question."""
    if config.APPLICATION_AFTER_SIGNIN:
        raise RuntimeError(
            "APPLICATION_AFTER_SIGNIN: refused to start Stage D after sign-in. "
            "The saved application route must be used instead."
        )
    if config.APPLICATION_TEST_MODE:
        from .app_stage import prepare_application_test

        return await prepare_application_test(session, page)
    opened = await _open_household_eligibility(page)
    if not opened:
        visible = await _visible_button_text(page)
        logger.error("Eligibility page did not open url=%s buttons=%s", page.url, visible)
        raise RuntimeError(
            "Logged in, but Check Household Eligibility did not open. "
            f"Visible actions: {visible or 'none'}"
        )
    reset = await _reset_eligibility_to_first_question(page)
    if not reset:
        logger.error("Could not reset eligibility to marital status url=%s", page.url)
        raise RuntimeError(
            "Logged in, but the eligibility form could not reach its first question."
        )
    session.signed_in = True
    session.elig_step = "marital"
    session.after_person = ""
    session.last_applied = ""
    session.last_result = None
    session.filled["eligibility_intro_pending"] = "1"
    opening = _spoken_question_for_step(session, session.elig_step)
    # Persist so post-sign-in speech helpers cannot drop the purpose intro.
    session.filled["eligibility_opening_speech"] = opening
    logger.info("Eligibility open url=%s step=%s", page.url, session.elig_step)
    return "ASK: " + opening


# Spoken guidance for native email auth — never request passwords/codes aloud.
EMAIL_SECURE_COMPLETE = (
    "Please finish signing in in the Pointcare window on this computer. "
    "I will not ask you to say a password or verification code. "
    "I'll continue when you're signed in."
)
EMAIL_VERIFY_LINK = (
    "Please open the verification link we sent to your email, finish verifying "
    "in the Pointcare window, then come back. I will not ask you to read a code "
    "aloud. I'll continue when you're signed in."
)
EMAIL_PASSWORD_SECURE = (
    "Please finish creating or entering your password in the Pointcare window. "
    "I will not ask you to say it. I'll continue when you're signed in."
)
EMAIL_OTP_SECURE = (
    "Please enter the verification code in the Pointcare window on this computer. "
    "I will not ask you to say the code. I'll continue when you're signed in."
)
EMAIL_SECURE_RETRY = (
    "I could not finish the email sign-in yet. Please complete any steps in the "
    "Pointcare window, or say go back to try a different email, Google, or Apple."
)


def spoken_for_auth_screen(screen: str) -> str:
    """One clear instruction for the current secure auth step (no secrets aloud)."""
    key = (screen or "").strip().lower()
    if key == "need_confirm":
        return EMAIL_VERIFY_LINK
    if key == "need_password":
        return EMAIL_PASSWORD_SECURE
    if key in ("need_otp", "need_mobile"):
        return EMAIL_OTP_SECURE
    return EMAIL_SECURE_COMPLETE


async def _detect_auth_screen(page: Page) -> str:
    """Classify the current Pointcare auth UI without logging secrets."""
    if await _already_signed_in(page):
        return "signed_in"
    try:
        body = (await page.locator("body").inner_text(timeout=2500)).lower()
    except Exception:
        body = ""
    try:
        has_password = await page.locator("input[type='password']:visible").count() > 0
    except Exception:
        has_password = False
    try:
        has_email = await page.locator("input[type='email']:visible").count() > 0
    except Exception:
        has_email = False
    try:
        has_tel = await page.locator("input[type='tel']:visible").count() > 0
    except Exception:
        has_tel = False

    # Prefer verify-link / inbox screens before password (copy often mentions both).
    if any(
        needle in body
        for needle in (
            "confirm your email",
            "check your email",
            "verify your email",
            "verification link",
            "verify link",
            "sent you an email",
            "sent an email",
            "check your inbox",
            "magic link",
            "click the link",
            "open the link",
            "we emailed you",
            "email we sent",
        )
    ):
        return "need_confirm"
    if has_password or "create a password" in body or "enter your password" in body:
        return "need_password"
    if any(
        needle in body
        for needle in (
            "verification code",
            "one-time",
            "enter the code",
            "security code",
            "otp",
        )
    ):
        return "need_otp"
    if has_tel or any(
        needle in body for needle in ("mobile number", "phone number", "verify your phone")
    ):
        return "need_mobile"
    if has_email:
        return "need_email"
    return "unknown"


async def _email_field_matches(page: Page, email: str) -> bool:
    wanted = (email or "").strip().lower()
    if not wanted:
        return False
    try:
        return bool(
            await page.evaluate(
                """(want) => {
                  const nodes = [...document.querySelectorAll('input')];
                  return nodes.some((el) => {
                    const type = (el.getAttribute('type') || 'text').toLowerCase();
                    const name = (
                      (el.getAttribute('name') || '') +
                      (el.getAttribute('id') || '') +
                      (el.getAttribute('placeholder') || '') +
                      (el.getAttribute('autocomplete') || '')
                    ).toLowerCase();
                    const isEmail =
                      type === 'email' || name.includes('email') || type === 'text';
                    if (!isEmail) return false;
                    return ((el.value || '').trim().toLowerCase()).includes(want);
                  });
                }""",
                wanted,
            )
        )
    except Exception:
        return False


async def _ensure_native_email_form(page: Page) -> None:
    """Reveal the native email input if Pointcare shows an email CTA first."""
    try:
        if await page.locator("input[type='email']:visible").count():
            return
    except Exception:
        pass
    patterns = (
        r"continue with email",
        r"sign\s*in with email",
        r"use email",
        r"^email$",
        r"email address",
    )
    for pattern in patterns:
        try:
            btn = page.get_by_role("button", name=re.compile(pattern, re.I)).first
            if await btn.count() and await btn.is_visible():
                label = (await btn.inner_text()).strip().lower()
                if "google" in label or "apple" in label:
                    continue
                await btn.click(timeout=4000)
                await page.wait_for_timeout(500)
                if await page.locator("input[type='email']:visible").count():
                    return
        except Exception:
            continue


async def _click_continue_if_present(page: Page) -> bool:
    for action in ("continue", "next", "submit"):
        try:
            await _click_action(page, action)
            return True
        except Exception:
            continue
    return False


async def _sign_in_with_email(session: LiveFormSession, email: str) -> dict[str, Any]:
    """Fill native Pointcare email, advance Continue, wait for secure auth or eligibility.

    Never collects passwords or OTPs from the caller — those stay in the browser window.
    """
    page = session.page
    if page is None:
        raise RuntimeError("form page not ready")
    address = _extract_email(email) or _normalize_email(email)
    if not address or "@" not in address:
        return {
            "ok": False,
            "auth_screen": "need_email",
            "spoken": "What email address should I use?",
            "error": "invalid_email",
        }

    if config.APPLICATION_AFTER_SIGNIN:
        session.filled["signin_choice_done"] = "1"
        if session.elig_step in ("proceed_application", "confirm_start_application"):
            speech = (session.filled.get("application_gate_speech") or "").strip()
            return {
                "ok": True,
                "auth_screen": "signed_in",
                "spoken": "ASK: " + speech,
                "step": session.elig_step,
            }
        if await _already_signed_in(page) or await _application_test_authenticated(page):
            spoken = await _open_saved_application(session, page)
            return {
                "ok": True,
                "auth_screen": "signed_in",
                "spoken": spoken,
                "step": session.elig_step,
            }
    if session.signed_in and session.elig_step:
        screen = await _eligibility_screen(page)
        recovered = _screen_first_step(screen, session)
        session.elig_step = recovered
        return {
            "ok": True,
            "auth_screen": "signed_in",
            "spoken": "ASK: " + ELIG_QUESTIONS[recovered],
            "step": recovered,
        }
    if await _already_signed_in(page):
        spoken = await _begin_eligibility(session, page)
        return {"ok": True, "auth_screen": "signed_in", "spoken": spoken, "step": "marital"}

    # Preserve Google chooser path if a Google popup is already open.
    popup = session.popup or _provider_popup(page)
    if popup is not None and "accounts.google." in (popup.url or ""):
        session.popup = popup
        msg = await _sign_in_with_provider(session, "Continue with Google", address)
        return {
            "ok": True,
            "auth_screen": "google_popup",
            "spoken": msg if str(msg).startswith("ASK:") else "ASK: " + str(msg),
            "action": "sso",
        }

    await _ensure_native_email_form(page)
    spec = get_field("email")
    labels = spec.labels if spec else ("email", "email address")
    placeholders = spec.placeholders if spec else ("email",)
    input_types = spec.input_types if spec else ("email", "text")

    await _fill_text(page, labels, placeholders, input_types, address)
    if not await _email_field_matches(page, address):
        logger.warning("Native email fill not verified call_id=%s", session.call_id)
        return {
            "ok": False,
            "auth_screen": "need_email",
            "spoken": EMAIL_SECURE_RETRY,
            "error": "email_not_filled",
        }

    # Store that we filled email without logging the address.
    session.filled["email"] = address
    session.history.append(
        {"field_id": "email", "value": "[redacted]", "action": "fill", "ok": True}
    )
    clicked = await _click_continue_if_present(page)
    logger.info(
        "Native email filled call_id=%s continue_clicked=%s",
        session.call_id,
        clicked,
    )
    try:
        await page.wait_for_timeout(900)
    except Exception:
        pass

    # Poll briefly: either we land on eligibility or a secure auth screen.
    last_screen = "unknown"
    for attempt in range(40):
        if await _already_signed_in(page):
            if config.APPLICATION_AFTER_SIGNIN:
                spoken = await _open_saved_application(session, page)
            else:
                spoken = await _begin_eligibility(session, page)
            return {
                "ok": True,
                "auth_screen": "signed_in",
                "spoken": spoken,
                "step": session.elig_step or (
                    "" if config.APPLICATION_AFTER_SIGNIN else "marital"
                ),
            }
        last_screen = await _detect_auth_screen(page)
        if last_screen in ("need_password", "need_otp", "need_mobile", "need_confirm"):
            logger.info(
                "Native email auth awaiting secure step call_id=%s screen=%s attempt=%s",
                session.call_id,
                last_screen,
                attempt,
            )
            # Keep polling briefly — caller completes password/link in Chrome.
            if attempt >= 4:
                return {
                    "ok": True,
                    "auth_screen": last_screen,
                    "spoken": spoken_for_auth_screen(last_screen),
                    "awaiting_secure": True,
                }
        elif last_screen == "need_email" and attempt >= 6:
            return {
                "ok": False,
                "auth_screen": "need_email",
                "spoken": EMAIL_SECURE_RETRY,
                "error": "still_on_email",
            }
        try:
            await page.wait_for_timeout(500)
        except Exception:
            break

    if await _already_signed_in(page) or (
        config.APPLICATION_AFTER_SIGNIN and await _application_test_authenticated(page)
    ):
        if config.APPLICATION_AFTER_SIGNIN:
            spoken = await _open_saved_application(session, page)
        else:
            spoken = await _begin_eligibility(session, page)
        return {
            "ok": True,
            "auth_screen": "signed_in",
            "spoken": spoken,
            "step": session.elig_step or (
                "" if config.APPLICATION_AFTER_SIGNIN else "marital"
            ),
        }

    logger.warning(
        "Native email auth incomplete call_id=%s screen=%s",
        session.call_id,
        last_screen,
    )
    return {
        "ok": True,
        "auth_screen": last_screen or "unknown",
        "spoken": spoken_for_auth_screen(last_screen or "unknown"),
        "awaiting_secure": True,
    }


async def poll_email_auth_until_signed_in(
    call_id: str, *, max_attempts: int = 90
) -> dict[str, Any] | None:
    """Background poll while the caller finishes password/OTP in the browser."""
    session = await get_session(call_id)
    if session is None or session.page is None:
        return None
    page = session.page
    for _ in range(max_attempts):
        try:
            if session.signed_in and session.elig_step:
                if session.elig_step in (
                    "proceed_application",
                    "confirm_start_application",
                ):
                    spoken = (
                        session.filled.get("application_gate_speech") or ""
                    ).strip()
                else:
                    spoken = ELIG_QUESTIONS.get(session.elig_step, "")
                return {
                    "ok": True,
                    "auth_screen": "signed_in",
                    "spoken": "ASK: " + spoken,
                    "step": session.elig_step,
                }
            signed = await _already_signed_in(page)
            if not signed and config.APPLICATION_AFTER_SIGNIN:
                signed = await _application_test_authenticated(page)
            if signed:
                async with session.lock:
                    if config.APPLICATION_AFTER_SIGNIN:
                        spoken = await _open_saved_application(session, page)
                    else:
                        spoken = await _begin_eligibility(session, page)
                return {
                    "ok": True,
                    "auth_screen": "signed_in",
                    "spoken": spoken,
                    "step": session.elig_step or (
                        "" if config.APPLICATION_AFTER_SIGNIN else "marital"
                    ),
                }
        except Exception as exc:
            if config.APPLICATION_AFTER_SIGNIN and "APPLICATION_AFTER_SIGNIN" in str(exc):
                detail = str(exc).split(":", 1)[-1].strip()
                return {
                    "ok": False,
                    "auth_screen": "application_page",
                    "spoken": "ASK: " + detail[:400],
                    "error": "application_after_signin",
                }
            logger.exception("email auth poll failed call_id=%s", call_id)
            return None
        await asyncio.sleep(1.0)
    return {
        "ok": False,
        "auth_screen": "timeout",
        "spoken": EMAIL_SECURE_RETRY,
        "error": "auth_timeout",
    }


async def _on_marital_status(page: Page) -> bool:
    url = (page.url or "").lower()
    if "marital-status" in url or "marital_status" in url:
        return True
    try:
        body = (await page.locator("body").inner_text(timeout=3000)).lower()
    except Exception:
        body = ""
    return "what is your marital status" in body


async def _reset_eligibility_to_first_question(page: Page) -> bool:
    """Pointcare resumes unfinished checks; land on marital status."""
    for _ in range(20):
        if await _on_marital_status(page):
            return True
        before = await _screen_signature(page)
        back = page.get_by_role("button", name=re.compile(r"^\s*Back\s*$", re.I)).first
        try:
            if await back.count() and await back.is_visible():
                await back.click(timeout=4000)
                for _ in range(20):
                    await page.wait_for_timeout(250)
                    if await _screen_signature(page) != before:
                        break
                continue
        except Exception:
            logger.exception("Could not step back in eligibility")
        break
    # Direct route when Back is missing or the prior check resumed mid-flow.
    try:
        await page.goto(
            "https://my.pointcare.com/eligibility/marital-status",
            wait_until="domcontentloaded",
            timeout=60000,
        )
        await page.wait_for_timeout(800)
    except Exception:
        logger.exception("Could not open marital-status eligibility URL")
        return False
    for _ in range(12):
        if await _on_marital_status(page):
            return True
        await page.wait_for_timeout(400)
    return False


async def _open_household_eligibility(page: Page) -> bool:
    """Existing accounts land on the dashboard. Open Check Household Eligibility."""
    try:
        await page.bring_to_front()
    except Exception:
        pass
    if "/eligibility/" in (page.url or "").lower():
        return True
    for label in (
        "check household eligibility",
        "household eligibility",
        "check eligibility",
    ):
        try:
            btn = page.get_by_role(
                "button", name=re.compile(re.escape(label), re.I)
            ).first
            if await btn.count() and await btn.is_visible():
                await btn.click(timeout=5000)
                await page.wait_for_timeout(900)
                if "/eligibility/" in (page.url or "").lower():
                    logger.info("Opened household eligibility at %s", page.url)
                    return True
        except Exception:
            pass
    for _ in range(8):
        clicked = False
        try:
            clicked = await page.evaluate(
                """() => {
                  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                  const el = [...document.querySelectorAll('button, [role=button], a')].find((node) => {
                    const r = node.getBoundingClientRect();
                    return r.width > 2 && r.height > 2 && (
                      norm(node.innerText).includes('household eligibility')
                      || norm(node.innerText).includes('check eligibility')
                    );
                  });
                  if (!el) return false;
                  el.click();
                  return true;
                }"""
            )
        except Exception:
            clicked = False
        try:
            await page.wait_for_timeout(900)
        except Exception:
            return False
        if "/eligibility/" in (page.url or "").lower():
            logger.info("Opened household eligibility at %s", page.url)
            return True
        if not clicked:
            try:
                await page.wait_for_timeout(400)
            except Exception:
                return False
    if "/eligibility/" not in (page.url or "").lower():
        try:
            await page.goto(
                "https://my.pointcare.com/eligibility/marital-status",
                wait_until="domcontentloaded",
                timeout=60000,
            )
            await page.wait_for_timeout(800)
        except Exception:
            logger.exception("Could not navigate directly to eligibility")
    return "/eligibility/" in (page.url or "").lower()


async def _visible_button_text(page: Page) -> str:
    try:
        texts = await page.evaluate(
            """() => [...document.querySelectorAll('button, [role=button], a')]
              .filter((node) => node.getBoundingClientRect().width > 2)
              .map((node) => (node.innerText || '').replace(/\\s+/g, ' ').trim())
              .filter(Boolean)
              .slice(0, 12)"""
        )
    except Exception:
        return ""
    return " | ".join(texts or [])


def _is_pointcare_cookie(cookie: dict) -> bool:
    domain = str(cookie.get("domain") or "").lower().lstrip(".")
    return domain == "pointcare.com" or domain.endswith(".pointcare.com")


def _cookie_for_restore(cookie: dict) -> dict | None:
    """Cookie fields Playwright will accept when putting a login back."""
    name = cookie.get("name")
    domain = cookie.get("domain")
    if not name or cookie.get("value") is None or not domain:
        return None
    restored = {
        "name": str(name),
        "value": str(cookie.get("value")),
        "domain": str(domain),
        "path": str(cookie.get("path") or "/"),
    }
    expires = cookie.get("expires")
    if isinstance(expires, (int, float)) and expires > 0:
        restored["expires"] = expires
    if "httpOnly" in cookie:
        restored["httpOnly"] = bool(cookie.get("httpOnly"))
    if "secure" in cookie:
        restored["secure"] = bool(cookie.get("secure"))
    same_site = cookie.get("sameSite")
    if same_site in ("Strict", "Lax", "None"):
        restored["sameSite"] = same_site
    return restored


async def _clear_pointcare_session(page: Page) -> None:
    """Drop the Pointcare site session. Keep the Chrome profile's other logins."""
    try:
        context = page.context
        existing = await context.cookies()
        kept: list[dict] = []
        for cookie in existing:
            if _is_pointcare_cookie(cookie):
                continue
            restored = _cookie_for_restore(cookie)
            if restored is not None:
                kept.append(restored)
        await context.clear_cookies()
        if kept:
            await context.add_cookies(kept)
    except Exception:
        logger.exception("Could not clear Pointcare cookies for fresh sign-in")
    try:
        if "pointcare.com" in ((page.url or "").lower()):
            await page.evaluate(
                """() => {
                  try { localStorage.clear(); } catch (e) {}
                  try { sessionStorage.clear(); } catch (e) {}
                }"""
            )
    except Exception:
        pass


async def _page_body_lower_snippet(page: Page, *, limit: int = 4000) -> str:
    try:
        return (await page.locator("body").inner_text(timeout=2500)).lower()[:limit]
    except Exception:
        return ""


def _text_looks_like_saved_dashboard(body: str, buttons: str = "") -> bool:
    """True when page copy matches Next steps / Quick Apply household dashboard."""
    blob = f"{body or ''} {buttons or ''}".lower()
    if any(
        needle in blob
        for needle in (
            "continue quick apply",
            "new quick apply",
            "recheck eligibility",
            "coverage unknown",
        )
    ):
        return True
    if "quick apply" in blob and "household" in blob:
        return True
    return "next steps" in blob and "household" in blob


async def _on_saved_household_surface(page: Page) -> bool:
    """True on Next steps / Quick Apply dashboard, programs, or in-progress app.

    Content wins over URL: an authenticated redirect can keep /onboarding in the
    address bar while rendering the saved household dashboard.
    """
    url = (page.url or "").lower()
    if "accounts.google." in url or "/auth" in url:
        return False
    buttons = ""
    try:
        buttons = await _visible_button_text(page)
    except Exception:
        buttons = ""
    body = await _page_body_lower_snippet(page)
    # Content first: authenticated SPA can keep /onboarding while showing Quick Apply.
    if _text_looks_like_saved_dashboard(body, buttons):
        return True
    # Marketing welcome mentions "health insurance" — must not become app_insurance.
    if _on_welcome_landing(url):
        return False
    if await _already_signed_in(page):
        return True
    try:
        from .app_stage import classify_saved_household_page

        surface = await classify_saved_household_page(page)
    except Exception:
        surface = ""
    return surface in ("dashboard", "programs") or str(surface).startswith("app_")


async def _confirm_welcome_page_one(page: Page) -> bool:
    """True only on welcome page 1 — not the Quick Apply dashboard."""
    if not _on_welcome_landing(page.url):
        return False
    if await _on_saved_household_surface(page):
        return False
    try:
        buttons = (await _visible_button_text(page)).lower()
    except Exception:
        buttons = ""
    body = await _page_body_lower_snippet(page)
    if _text_looks_like_saved_dashboard(body, buttons):
        return False
    return "get started" in buttons or "get started" in body


async def _ui_logout_saved_session(page: Page) -> None:
    """Click Exit/Logout on a saved household so cookie clear can stick."""
    url = (page.url or "").lower()
    if "/eligibility/" in url:
        try:
            await asyncio.wait_for(_click_labeled(page, "Exit"), timeout=4)
            for _ in range(12):
                await page.wait_for_timeout(200)
                if "/eligibility/" not in (page.url or "").lower():
                    break
        except Exception:
            logger.warning("Could not exit eligibility before logout")
    # Open account/menu chrome if Logout is not already visible.
    try:
        buttons = (await _visible_button_text(page)).lower()
    except Exception:
        buttons = ""
    if "logout" not in buttons and "log out" not in buttons:
        for opener in ("Account", "Profile", "Menu", "Settings"):
            try:
                await asyncio.wait_for(_click_labeled(page, opener), timeout=3)
                await page.wait_for_timeout(400)
                break
            except Exception:
                continue
    try:
        await asyncio.wait_for(_click_labeled(page, "Logout"), timeout=5)
    except Exception:
        try:
            await asyncio.wait_for(_click_labeled(page, "Log out"), timeout=4)
        except Exception:
            logger.warning("Logout click failed or timed out")
            return
    await page.wait_for_timeout(600)
    for label in ("Log out", "Logout", "Yes"):
        try:
            await asyncio.wait_for(_click_labeled(page, label), timeout=3)
            await page.wait_for_timeout(400)
            break
        except Exception:
            continue


async def _logout_for_fresh_signin(page: Page, onboarding_url: str) -> None:
    """Land on welcome page 1 once, then the caller clicks through to sign-in.

    Must not leave the tab on the saved household dashboard (Next steps /
    Continue Quick Apply). The onboarding URL is opened once. It is opened
    again only when that first page is still the saved household.
    """
    target = (onboarding_url or "").strip()
    if not target:
        return

    async def _open_welcome() -> None:
        await page.goto(target, wait_until="domcontentloaded", timeout=45000)
        # Allow SPA auth redirect onto Quick Apply before we decide "welcome ready".
        await page.wait_for_timeout(1200)

    logger.info(
        "Fresh start: clear session, then open welcome page 1 url=%s",
        _safe_url_for_log(target),
    )
    await _clear_pointcare_session(page)
    try:
        await _open_welcome()
    except Exception:
        logger.exception("Could not open onboarding for fresh start")
        raise
    # Drop site storage now that the tab is on Pointcare. Do not navigate again.
    await _clear_pointcare_session(page)

    for attempt in range(4):
        if await _confirm_welcome_page_one(page):
            logger.info(
                "Welcome landing ready attempt=%s url=%s",
                attempt + 1,
                _safe_url_for_log(page.url),
            )
            return
        if await _on_saved_household_surface(page) or await _already_signed_in(page):
            logger.warning(
                "Saved household still open attempt=%s url=%s — UI logout then clear",
                attempt + 1,
                _safe_url_for_log(page.url),
            )
            try:
                await asyncio.wait_for(_ui_logout_saved_session(page), timeout=12)
            except Exception:
                logger.warning("UI logout timed out; clearing cookies anyway")
            await _clear_pointcare_session(page)
            await _open_welcome()
            continue
        await page.wait_for_timeout(800)
        if await _confirm_welcome_page_one(page):
            logger.info(
                "Welcome landing ready attempt=%s url=%s",
                attempt + 1,
                _safe_url_for_log(page.url),
            )
            return
        logger.info(
            "Not on welcome landing attempt=%s url=%s; reopening onboarding",
            attempt + 1,
            _safe_url_for_log(page.url),
        )
        await _open_welcome()

    if await _on_saved_household_surface(page) or await _already_signed_in(page):
        raise RuntimeError(
            "Pointcare stayed on the saved household (Quick Apply / Next steps) "
            "instead of the welcome pages. Sign out of Pointcare in this Chrome "
            "profile once, then start the call again."
        )
    if not await _confirm_welcome_page_one(page):
        raise RuntimeError(
            "Could not open the Pointcare welcome page for a fresh sign-in: "
            + _safe_url_for_log(page.url or "")
        )


async def _wait_for_saved_household(page: Page, *, call_id: str = "") -> None:
    """Give a saved Pointcare session a moment to leave the onboarding URL."""
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if await _application_test_authenticated(page):
            return
        url = (page.url or "").lower()
        if "/auth" in url:
            return
        await _page_wait(page, 400, call_id=call_id)


async def _finish_application_test_session(
    session: LiveFormSession, page: Page
) -> LiveFormSession:
    """Skip logout and Stage D. Land on the saved household application path."""
    from .app_stage import prepare_application_test

    spoken = await prepare_application_test(session, page)
    opening = spoken[4:].strip() if spoken.startswith("ASK:") else spoken
    session.filled["application_test_opening"] = opening
    session.history.append(
        {
            "field_id": "application_test",
            "value": session.elig_step or session.app_step or "unknown",
            "action": "resume_saved_household",
            "ok": True,
        }
    )
    session.error = ""
    session.welcome_done = True
    session.signed_in = True
    session.status = "open"
    session.url = page.url or session.url
    session.ready.set()
    logger.info(
        "Application test session opened call_id=%s step=%s app=%s",
        session.call_id,
        session.elig_step or "-",
        session.app_step or "-",
    )
    return session


async def _application_test_authenticated(page: Page) -> bool:
    """Saved-account check used only by application test mode.

    The household dashboard's Quick Apply control is often not a button, so the
    production sign-in detector can miss it. This does not change that detector.
    The onboarding welcome page is not a saved application, even when its copy
    mentions health insurance — unless the saved Quick Apply dashboard is showing.
    """
    if await _on_saved_household_surface(page):
        return True
    if _on_welcome_landing(page.url):
        return False
    if await _already_signed_in(page):
        return True
    from .app_stage import classify_saved_household_page

    surface = await classify_saved_household_page(page)
    return surface in ("dashboard", "programs") or surface.startswith("app_")


async def _already_signed_in(page: Page) -> bool:
    url = (page.url or "").lower()
    if "accounts.google." in url or "/auth" in url:
        return False
    # Marketing welcome with Get Started is never a signed-in session. Treating it
    # as signed-in made promote open eligibility, fail, and left the tab on the
    # landing page so later retries looked like a welcome loop.
    if await _welcome_get_started_visible(page):
        return False
    if "/eligibility/" in url:
        return True
    text = (await _visible_button_text(page)).lower()
    if "household eligibility" in text or "logout" in text:
        return True
    # Saved household dashboard (Next steps / Quick Apply) is still signed in —
    # do not treat it as a fresh welcome/sign-in start.
    body = await _page_body_lower_snippet(page)
    return _text_looks_like_saved_dashboard(body, text)


async def _google_account_emails(popup: Page) -> list[str]:
    try:
        found = await popup.evaluate(
            """() => {
              const fromAttr = [...document.querySelectorAll('[data-identifier]')]
                .map((node) => (node.getAttribute('data-identifier') || '').trim().toLowerCase());
              const fromText = [...(document.body.innerText || '').matchAll(/[\\w.+-]+@[\\w.-]+\\.[a-z]{2,}/gi)]
                .map((match) => match[0].toLowerCase());
              return [...fromAttr, ...fromText];
            }"""
        )
    except Exception:
        return []
    emails: list[str] = []
    for item in found or []:
        if item not in emails:
            emails.append(item)
    return emails


async def _list_google_accounts(popup: Page) -> str:
    loc = popup.locator("[role='link'][data-identifier]")
    try:
        count = await loc.count()
    except Exception:
        return "could not read them"
    names: list[str] = []
    for i in range(min(count, 6)):
        try:
            text = (await loc.nth(i).inner_text()).strip().replace("\n", " ")
        except Exception:
            text = ""
        if text and "use another account" not in text.lower():
            names.append(text)
    return "; ".join(names) or "none listed yet"


async def _choose_google_account(popup: Page, email: str) -> None:
    wanted = _extract_email(email)
    if not wanted:
        return
    await popup.wait_for_timeout(400)
    clicked = await popup.evaluate(
        """(wanted) => {
          const nodes = [...document.querySelectorAll('[data-identifier]')];
          const el = nodes.find((node) => (node.getAttribute('data-identifier') || '').toLowerCase() === wanted);
          if (!el) return false;
          (el.closest('[role=link]') || el).click();
          return true;
        }""",
        wanted,
    )
    if clicked:
        logger.info("Chose Google account %s", wanted)
        return
    if wanted:
        try:
            other = popup.get_by_text(re.compile(r"use another account", re.I)).first
            if await other.count() and await other.is_visible():
                await other.click(timeout=4000)
                await popup.wait_for_timeout(700)
        except Exception:
            pass
        box = popup.locator("#identifierId, input[type='email']").first
        try:
            if await box.count() and await box.is_visible():
                await box.click(timeout=3000)
                await box.fill(wanted, timeout=5000)
                nxt = popup.get_by_role("button", name=re.compile(r"^\s*Next\s*$", re.I)).first
                if await nxt.count():
                    await nxt.click(timeout=5000)
                logger.info("Typed Google email %s", wanted)
                return
        except Exception:
            logger.exception("Could not type Google email")
    raise RuntimeError(
        f"Google account {wanted!r} is not on the chooser. "
        "Ask the caller to say the email again."
    )


async def _pick_address_suggestion(page: Page, value: str) -> None:
    await page.wait_for_timeout(800)
    hint = re.escape((value or "").strip()[:18])
    if not hint:
        return
    option = page.locator("[role='option'], .pac-item, li").filter(
        has_text=re.compile(hint, re.I)
    ).first
    try:
        if await option.count() and await option.is_visible():
            await option.click(timeout=3000)
    except Exception:
        pass


async def _click_labeled(page: Page, label: str) -> None:
    """Click a button whose visible text is exactly this label, including Google."""
    clicked = await page.evaluate(
        """(label) => {
          const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
          const want = norm(label);
          const nodes = [...document.querySelectorAll('button, [role=button], a')].filter((node) => {
            const r = node.getBoundingClientRect();
            if (r.width < 2 || r.height < 2) return false;
            const text = norm(node.innerText);
            return text === want || text.includes(want);
          });
          nodes.sort((a, b) => norm(a.innerText).length - norm(b.innerText).length);
          const el = nodes[0];
          if (!el) return false;
          el.click();
          return true;
        }""",
        label,
    )
    if not clicked:
        await _click_cta(page, label, allow_sso=True, attempts=3)


async def _click_cta(page: Page, label: str, allow_sso: bool = False, attempts: int = 8) -> None:
    """Click a Pointcare button with an exact label."""
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*$", re.I)
    last_exc: Exception | None = None
    locators = [
        page.get_by_role("button", name=pattern),
        page.locator("button").filter(has_text=pattern),
        page.locator("[role='button']").filter(has_text=pattern),
        page.get_by_role("link", name=pattern),
    ]
    for _attempt in range(attempts):
        for loc in locators:
            try:
                target = loc.first
                await target.wait_for(state="visible", timeout=2500)
                name = (await target.inner_text()).strip()
                if _is_sso_label(name) and not allow_sso:
                    continue
                if not pattern.search(name) and name.lower() != label.lower():
                    continue
                await target.scroll_into_view_if_needed()
                await target.click(timeout=8000)
                return
            except Exception as exc:
                if is_playwright_closed_error(exc):
                    raise
                last_exc = exc
                continue
        try:
            await _page_wait(page, 500)
        except RuntimeError:
            if last_exc and is_playwright_closed_error(last_exc):
                raise RuntimeError(
                    f"BROWSER_CLOSED: Playwright closed while clicking {label!r}"
                ) from last_exc
            raise
    raise RuntimeError(f"Could not click {label!r} after retries: {last_exc}")


async def _clear_email_inputs_on_page(page: Page) -> None:
    try:
        await page.evaluate(
            """() => {
              const boxes = [...document.querySelectorAll('input')].filter((el) => {
                const type = (el.getAttribute('type') || 'text').toLowerCase();
                const name = (
                  (el.getAttribute('name') || '') +
                  (el.getAttribute('id') || '') +
                  (el.getAttribute('placeholder') || '') +
                  (el.getAttribute('autocomplete') || '')
                ).toLowerCase();
                return type === 'email' || name.includes('email');
              });
              for (const el of boxes) {
                el.value = '';
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
              }
              return boxes.length;
            }"""
        )
    except Exception:
        pass


async def _on_sign_in_chooser(page: Page) -> bool:
    """True when Google / Apple / email sign-in options are available."""
    try:
        visible = (await _visible_button_text(page)).lower()
    except Exception:
        visible = ""
    if "continue with google" in visible or "continue with apple" in visible:
        return True
    if "google" in visible and ("apple" in visible or "email" in visible):
        return True
    return False


async def _click_sign_in_back_control(page: Page) -> bool:
    """Click common Pointcare controls that leave verify/password back to chooser."""
    patterns = (
        r"use (a )?different email",
        r"change email",
        r"edit email",
        r"use another email",
        r"sign in another way",
        r"other (sign[- ]?in )?options",
        r"try another method",
        r"^back$",
        r"go back",
        r"previous",
    )
    for pattern in patterns:
        for role in ("button", "link"):
            try:
                loc = page.get_by_role(role, name=re.compile(pattern, re.I)).first
                if await loc.count() and await loc.is_visible():
                    label = (await loc.inner_text()).strip().lower()
                    if "google" in label or "apple" in label:
                        continue
                    await loc.click(timeout=3000)
                    await page.wait_for_timeout(600)
                    return True
            except Exception:
                continue
    return False


async def clear_native_email_input(call_id: str) -> bool:
    """Clear visible email inputs after go-back so a corrected address can be typed."""
    session = await get_session(call_id)
    if session is None or session.page is None:
        return False
    page = session.page
    try:
        await _clear_email_inputs_on_page(page)
        session.filled.pop("email", None)
        logger.info("Cleared native email input call_id=%s", call_id)
        return True
    except Exception:
        logger.exception("clear_native_email_input evaluate failed call_id=%s", call_id)
        return False


async def reset_email_sign_in_ui(call_id: str) -> bool:
    """After go-back: leave verify/password screens and land on Google/Apple/email chooser."""
    session = await get_session(call_id)
    if session is None or session.page is None:
        return False
    page = session.page
    try:
        for _ in range(5):
            if await _on_sign_in_chooser(page):
                await _clear_email_inputs_on_page(page)
                session.filled.pop("email", None)
                session.popup = None
                logger.info(
                    "Email sign-in UI reset to chooser call_id=%s url=%s",
                    call_id,
                    page.url,
                )
                return True
            if await _click_sign_in_back_control(page):
                continue
            try:
                await page.go_back(timeout=4000)
                await page.wait_for_timeout(700)
            except Exception:
                break
        # Last resort: clear whatever email field is visible on the current page.
        await _clear_email_inputs_on_page(page)
        session.filled.pop("email", None)
        session.popup = None
        logger.warning(
            "Email sign-in UI reset partial call_id=%s url=%s",
            call_id,
            page.url,
        )
        return await _on_sign_in_chooser(page)
    except Exception:
        logger.exception("reset_email_sign_in_ui failed call_id=%s", call_id)
        return False


async def promote_signed_in_to_eligibility(call_id: str) -> str | None:
    """If the browser is already signed in, open Stage D and return the first question.

    Used when OAuth finishes after the voice turn already returned a waiting message.
    Application-after-signin waits until the caller chooses a sign-in method, then
    never opens Stage D.
    """
    session = await get_session(call_id)
    if session is None or session.page is None:
        return None
    if config.APPLICATION_AFTER_SIGNIN:
        if session.filled.get("signin_choice_done") != "1":
            return None
        if session.elig_step in ("proceed_application", "confirm_start_application"):
            return (session.filled.get("application_gate_speech") or "").strip() or None
        if session.app_step:
            from .app_stage import app_question_for_step

            return app_question_for_step(session.app_step, session)
    if session.signed_in and session.elig_step:
        stored = (session.filled.get("eligibility_opening_speech") or "").strip()
        if stored and not session.filled.get("eligibility_opening_delivered"):
            return stored
        return _spoken_question_for_step(session, session.elig_step) or None
    page = session.page
    try:
        if await _welcome_get_started_visible(page):
            logger.info(
                "promote_signed_in skipped: still on welcome Get Started call_id=%s",
                call_id,
            )
            return None
        if not await _post_auth_session_ready(page):
            return None
    except Exception:
        logger.exception("promote_signed_in: signed-in check failed call_id=%s", call_id)
        return None
    async with session.lock:
        if session.signed_in and session.elig_step:
            stored = (session.filled.get("eligibility_opening_speech") or "").strip()
            if stored and not session.filled.get("eligibility_opening_delivered"):
                return stored
            return _spoken_question_for_step(session, session.elig_step) or None
        # Re-check under the lock: SSO can bounce the tab to Get Started between
        # the outer ready check and navigation.
        if await _welcome_get_started_visible(page):
            logger.info(
                "promote_signed_in aborted under lock: welcome Get Started call_id=%s",
                call_id,
            )
            return None
        if not await _post_auth_session_ready(page):
            return None
        try:
            if config.APPLICATION_AFTER_SIGNIN:
                msg = await _open_saved_application(session, page)
            else:
                msg = await _begin_eligibility(session, page)
        except Exception as exc:
            logger.exception("promote_signed_in: post-auth navigation failed call_id=%s", call_id)
            if config.APPLICATION_AFTER_SIGNIN or config.APPLICATION_TEST_MODE:
                # Never speak APPLICATION_TEST / stack diagnostics to the caller.
                detail = str(exc)
                logger.error(
                    "promote_signed_in diagnostic call_id=%s detail=%s",
                    call_id,
                    detail[:300],
                )
                from .app_stage import POST_AUTH_RETRY_SPEECH

                session.elig_step = "proceed_application"
                session.filled["application_gate"] = "proceed"
                session.filled["application_gate_speech"] = POST_AUTH_RETRY_SPEECH
                session.filled["application_test_opening"] = POST_AUTH_RETRY_SPEECH
                session.filled["awaiting_quick_apply"] = "1"
                session.filled["application_started"] = "0"
                session.signed_in = True
                session.welcome_done = True
                return POST_AUTH_RETRY_SPEECH
            return None
    text = str(msg or "")
    if text.startswith("ASK:"):
        text = text[4:].strip()
    return text or ELIG_QUESTIONS.get(session.elig_step) or None


async def get_session(call_id: str) -> LiveFormSession | None:
    async with _sessions_lock:
        return _sessions.get(call_id)


async def describe_live_browser(
    session: LiveFormSession | None,
    *,
    call_id: str = "",
    open_base_url: str = "",
) -> dict[str, Any]:
    """Snapshot of the agent Chrome tab for API / Postman responses.

    Includes the live page URL, Chrome profile path, CDP endpoint, and a
    clickable local page that focuses the agent window and links DevTools.
    """
    from .browser import (
        CDP_PORT,
        CDP_URL,
        cdp_target_for_url,
        local_devtools_url,
    )
    from .config import (
        CHROME_AGENT_DATA,
        CHROME_PROFILE_DIRECTORY,
        PORT,
        PUBLIC_BASE_URL,
        chrome_profile_label,
    )

    cid = (call_id or (session.call_id if session else "") or "").strip()
    live_url = ""
    opened = False
    if session is not None:
        try:
            page_live = session.page is not None and not (
                getattr(session.page, "is_closed", lambda: False)()
            )
        except Exception:
            page_live = False
        # Failed/superseded/ended sessions must not look "opened".
        opened = bool(
            page_live and session.status == "open" and session.welcome_done
        )
        if page_live and session.page is not None:
            try:
                live_url = str(session.page.url or "")
                if live_url:
                    session.url = live_url
            except Exception:
                live_url = session.url or ""
        else:
            live_url = session.url or ""

    target = await asyncio.to_thread(cdp_target_for_url, live_url)
    if not live_url:
        live_url = str((target or {}).get("url") or "")
    devtools = local_devtools_url(target) if target else ""
    base = (
        (open_base_url or "").rstrip("/")
        or (PUBLIC_BASE_URL or "").rstrip("/")
        or f"http://127.0.0.1:{PORT}"
    )
    open_url = f"{base}/form-filler/sessions/{cid}/chrome" if cid else ""

    return {
        "opened": opened,
        "welcome_done": bool(session.welcome_done) if session else False,
        "url": live_url,
        "open_url": open_url,
        "devtools_url": devtools,
        "cdp_url": CDP_URL,
        "cdp_port": CDP_PORT,
        "chrome_profile": chrome_profile_label(),
        "chrome_profile_directory": CHROME_PROFILE_DIRECTORY,
        "chrome_user_data": str(CHROME_AGENT_DATA),
        "title": str((target or {}).get("title") or ""),
        "error": (session.error if session else "") or "",
    }


async def end_live_form(call_id: str) -> None:
    """End one call session. Idempotent; does not disconnect shared CDP/Chrome."""
    global _active_call_id
    async with _sessions_lock:
        session = _sessions.pop(call_id, None)
        if _active_call_id == call_id:
            _active_call_id = ""
    if not session:
        logger.info("end_live_form no-op call_id=%s (already gone)", call_id)
        return
    async with session.lock:
        session.status = "ended"
        session.page = None
        session.popup = None
        session.ready.set()
    logger.info("Session ended call_id=%s (shared CDP left connected)", call_id)


async def peek_choice_labels(call_id: str) -> list[str]:
    """Read visible choice labels without mutating questionnaire state."""
    session = await get_session(call_id)
    if session is None or session.page is None:
        return []
    async with session.lock:
        try:
            return await _visible_choice_labels(session.page)
        except Exception:
            logger.exception("peek_choice_labels failed call_id=%s", call_id)
            return []


async def _yes_no_selection_state(page: Page) -> str:
    """Return 'yes', 'no', 'both', or 'none' from visible selected Yes/No controls."""
    try:
        state = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const nodes = [...document.querySelectorAll(
                'button, [role=button], [role=radio], [role=option], label, input'
              )];
              let yes = false;
              let no = false;
              for (const el of nodes) {
                const text = norm(el.innerText || el.getAttribute('aria-label') || el.value);
                if (text !== 'yes' && text !== 'no') continue;
                const r = el.getBoundingClientRect();
                if (r.width < 2 || r.height < 2) continue;
                const selected =
                  el.getAttribute('aria-checked') === 'true' ||
                  el.getAttribute('aria-pressed') === 'true' ||
                  el.getAttribute('aria-selected') === 'true' ||
                  (el instanceof HTMLInputElement && el.checked) ||
                  (el.className || '').toLowerCase().includes('selected') ||
                  (el.className || '').toLowerCase().includes('active') ||
                  (el.className || '').toLowerCase().includes('checked');
                if (!selected) continue;
                if (text === 'yes') yes = true;
                if (text === 'no') no = true;
              }
              if (yes && no) return 'both';
              if (yes) return 'yes';
              if (no) return 'no';
              return 'none';
            }"""
        )
    except Exception:
        return "none"
    return str(state or "none")


async def navigate_back_one(call_id: str) -> dict[str, Any]:
    """Click Pointcare Back when available and reconcile elig_step to the screen."""
    from .call_trace import trace_event

    session = await get_session(call_id)
    if session is None or session.page is None:
        return {
            "ok": False,
            "error": "no_session",
            "spoken": "I can't go back right now because the form is not open.",
        }
    async with session.lock:
        page = session.page
        before = await _screen_signature(page)
        before_screen = await _eligibility_screen(page)
        before_step = session.elig_step
        back = page.get_by_role("button", name=re.compile(r"^\s*Back\s*$", re.I)).first
        try:
            if not await back.count() or not await back.is_visible():
                from .elig_meta import format_question

                speech = (
                    "I can't go back from this screen. There is no Back button "
                    "available. You can still correct the current question — "
                    + format_question(session.elig_step or before_step or "marital")
                )
                trace_event(
                    call_id,
                    "go_back_rejected",
                    reason="no_back",
                    screen_before=before_screen,
                    step=session.elig_step,
                )
                return {
                    "ok": False,
                    "error": "no_back",
                    "spoken": speech,
                    "step": session.elig_step,
                    "screen": before_screen,
                }
            await back.click(timeout=5000)
        except Exception:
            logger.exception("Back click failed call_id=%s", call_id)
            return {
                "ok": False,
                "error": "back_failed",
                "spoken": (
                    "I wasn't able to go back on the website. Let's stay on the "
                    "current question."
                ),
                "step": session.elig_step,
                "screen": before_screen,
            }
        landed = before_screen
        for _ in range(24):
            await page.wait_for_timeout(250)
            if await _screen_signature(page) != before:
                landed = await _eligibility_screen(page)
                break
        else:
            trace_event(
                call_id,
                "go_back_rejected",
                reason="no_transition",
                screen_before=before_screen,
                step=session.elig_step,
            )
            return {
                "ok": False,
                "error": "no_transition",
                "spoken": (
                    "I pressed Back, but the form did not move. Let's stay on the "
                    "current question."
                ),
                "step": session.elig_step,
                "screen": before_screen,
            }
        recovered = _screen_first_step(landed, session)
        if recovered and recovered in ELIG_QUESTIONS:
            session.elig_step = recovered
        # Drop last answer cache so a stale Yes/No cannot be replayed.
        session.last_step_answer = None
        session.last_result = None
        from .elig_meta import format_question

        speech = format_question(session.elig_step)
        trace_event(
            call_id,
            "go_back_ok",
            screen_before=before_screen,
            screen_after=landed,
            step_before=before_step,
            step_after=session.elig_step,
            selected=await _yes_no_selection_state(page),
        )
        return {
            "ok": True,
            "step": session.elig_step,
            "screen": landed,
            "screen_before": before_screen,
            "spoken": speech,
        }


async def skip_current_optional(call_id: str) -> dict[str, Any]:
    """Skip only when metadata allows; verify the browser actually advances."""
    from .elig_meta import format_question, get_field_meta

    session = await get_session(call_id)
    if session is None or session.page is None:
        return {
            "ok": False,
            "error": "no_session",
            "spoken": "I can't skip right now because the form is not open.",
        }
    step = session.elig_step or ""
    meta = get_field_meta(step)
    if meta.required is True or not meta.can_skip:
        return {
            "ok": False,
            "error": "required",
            "spoken": (
                format_question(step)
                + " This field is required to continue, so I can't skip it."
            ),
            "step": step,
        }
    async with session.lock:
        page = session.page
        before = await _screen_signature(page)
        behavior = meta.skip_behavior or "continue"
        try:
            if behavior == "skip_button":
                await _click_cta(page, "Skip", attempts=2)
            elif behavior == "next":
                await _click_cta(page, "Next", attempts=2)
            else:
                await _click_cta(page, "Continue", attempts=2)
        except Exception:
            return {
                "ok": False,
                "error": "skip_click_failed",
                "spoken": (
                    format_question(step)
                    + " I couldn't find a skip control on this screen."
                ),
                "step": step,
            }
        advanced = False
        for _ in range(20):
            await page.wait_for_timeout(250)
            if await _screen_signature(page) != before:
                advanced = True
                break
        if not advanced:
            return {
                "ok": False,
                "error": "skip_blocked",
                "spoken": (
                    format_question(step)
                    + " The website did not let me skip this field."
                ),
                "step": step,
            }
        landed = await _eligibility_screen(page)
        recovered = _screen_first_step(landed, session)
        if recovered and recovered in ELIG_QUESTIONS:
            session.elig_step = recovered
        return {
            "ok": True,
            "step": session.elig_step,
            "spoken": format_question(session.elig_step),
        }


async def submit_field(
    call_id: str,
    field_id: str,
    value: str = "",
    action: str = "",
    url: str = "",
    request_id: str = "",
) -> dict[str, Any]:
    """Fill one field (and optional Continue/Next) for this call's browser session."""
    if not call_id:
        return {"ok": False, "error": "missing call_id"}
    field_key = (field_id or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not field_key:
        return {"ok": False, "error": "missing field_id"}

    started = time.perf_counter()
    _mark(call_id, "answer_received", started)
    try:
        session = await start_live_form(call_id, url)
    except Exception as exc:
        logger.exception("Browser session could not start call_id=%s", call_id)
        return {
            "ok": False,
            "call_id": call_id,
            "error": str(exc),
            "spoken": "The Pointcare sign-in page is not ready. Please wait a moment and try again.",
        }
    try:
        await asyncio.wait_for(session.ready.wait(), timeout=45)
    except asyncio.TimeoutError:
        return {"ok": False, "error": "browser session did not become ready in time"}

    async with session.lock:
        if session.page is None:
            return {"ok": False, "error": "form page not ready"}
        page = session.page
        spec = get_field(field_key)
        text = (value or "").strip()
        act = (action or "").strip().lower() or (spec.after_action if spec else "")
        if request_id and request_id in session.processed_requests:
            logger.info(
                "latency call_id=%s stage=request_replayed request_id=%s",
                call_id,
                request_id,
            )
            return session.processed_requests[request_id]
        _mark(call_id, "mapped", started)

        try:
            email_value = _extract_email(text)
            google_popup = session.popup or (_provider_popup(page) if page else None)
            on_google = google_popup is not None and "accounts.google." in (google_popup.url or "")
            if field_key in ("google", "gmail", "continue_with_google", "sign_in_with_google", "sign_in_google") or (
                email_value and on_google
            ):
                if (
                    field_key in ("google", "gmail", "continue_with_google", "sign_in_with_google", "sign_in_google")
                    and text
                    and not email_value
                    and not re.search(r"\b(google|gmail)\b", text, re.I)
                ):
                    logger.warning(
                        "Rejected non-Google value for Google sign-in call_id=%s value=%r",
                        call_id,
                        text[:80],
                    )
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": field_key,
                        "error": "caller did not choose Google",
                        "spoken": "How would you like to sign in, with Google or with email?",
                    }
                _mark(call_id, "browser_action", started)
                msg = await _sign_in_with_provider(session, "Continue with Google", email_value or text)
                _mark(call_id, "verified", started)
                entry = {"field_id": "continue_with_google", "value": email_value, "action": "sso", "ok": True}
                session.history.append(entry)
                spoken = msg[4:].strip() if str(msg).startswith("ASK:") else msg
                _mark(call_id, "tool_response", started)
                result = {
                    "ok": True,
                    "call_id": call_id,
                    "field_id": "continue_with_google",
                    "value": email_value,
                    "action": "sso",
                    "message": spoken,
                    "spoken": spoken if spoken.startswith("What") or spoken.startswith("Which") or "marital" in spoken.lower() else "ASK: " + spoken,
                }
                session.last_applied = text.lower()
                session.last_applied_at = time.time()
                session.last_result = result
                if request_id:
                    session.processed_requests[request_id] = result
                return result
            if field_key == "eligibility_answer" and not session.elig_step and not session.app_step:
                # Language / pre-sign-in utterances must not click Pointcare options.
                logger.warning(
                    "Ignored eligibility_answer before sign-in call_id=%s value=%r",
                    call_id,
                    text[:80],
                )
                spoken = "How would you like to sign in, with Google or with email?"
                result = {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": field_key,
                    "error": "eligibility not started",
                    "spoken": spoken,
                    "message": spoken,
                }
                if request_id:
                    session.processed_requests[request_id] = result
                return result
            # Stage E Application Collect Mode (after Start Applications).
            if session.app_step or field_key == "application_answer":
                from .app_stage import (
                    app_question_for_step,
                    stage_e_explain,
                    submit_application_answer,
                )

                if not session.app_step and field_key == "application_answer":
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "error": "application not started",
                        "spoken": (
                            "The application has not been started yet. "
                            "Please finish household eligibility first."
                        ),
                    }
                stage_e_text = text.lower().strip()
                if re.fullmatch(
                    r"(repeat|again|next|next question|what'?s the next question|"
                    r"what is the next question|what was the question)[.?!]?",
                    stage_e_text,
                ):
                    spoken = app_question_for_step(session.app_step, session)
                    result = {
                        "ok": True,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "value": text,
                        "action": "repeat",
                        "spoken": "ASK: " + spoken,
                        "message": "repeat",
                        "step": session.app_step,
                        "phase": "application",
                    }
                    session.last_result = result
                    if request_id:
                        session.processed_requests[request_id] = result
                    return result
                # Explain / why — stay on the active Stage E field.
                if re.search(
                    r"\b("
                    r"what\??|what does that mean|what do you mean|"
                    r"why (are you|do you) ask|why (are you|do you) calling|"
                    r"for what|pardon|huh"
                    r")\b",
                    stage_e_text,
                ) and not re.search(
                    r"\b(yes|no|change|fix|edit|submit|skip|continue)\b",
                    stage_e_text,
                ):
                    spoken = stage_e_explain(session.app_step)
                    result = {
                        "ok": True,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "value": text,
                        "action": "explain",
                        "spoken": "ASK: " + spoken,
                        "message": "explain",
                        "step": session.app_step,
                        "phase": "application",
                    }
                    session.last_result = result
                    if request_id:
                        session.processed_requests[request_id] = result
                    return result
                step_key = (session.app_step, text.lower())
                if session.last_step_answer == step_key and session.last_result:
                    return session.last_result
                result = await submit_application_answer(
                    call_id, session, page, text, started
                )
                if result.get("ok"):
                    session.last_step_answer = step_key
                    session.last_result = result
                    session.last_applied = text.lower()
                    session.last_applied_at = time.time()
                if request_id:
                    session.processed_requests[request_id] = result
                return result
            if session.elig_step:
                # "Next question" / repeat: speak the current question, do not click.
                if re.fullmatch(
                    r"(repeat|again|next|next question|what'?s the next question|"
                    r"what is the next question|what was the question)[.?!]?",
                    text.lower().strip(),
                ):
                    spoken = ELIG_QUESTIONS.get(session.elig_step, "Please answer the current question.")
                    result = {
                        "ok": True,
                        "call_id": call_id,
                        "field_id": "eligibility_answer",
                        "value": text,
                        "action": "repeat",
                        "spoken": "ASK: " + spoken,
                        "message": "repeat",
                        "step": session.elig_step,
                    }
                    session.last_result = result
                    if request_id:
                        session.processed_requests[request_id] = result
                    return result
                # Stale marital (etc.) arriving after we already advanced to a Yes/No step.
                # Programs decision accepts continue/start phrasing — do not stale-ignore it.
                if (
                    session.elig_step in YES_NO_ELIG_STEPS
                    and session.elig_step != "programs"
                    and not _is_yes(text)
                    and not _is_no(text)
                ):
                    spoken = ELIG_QUESTIONS[session.elig_step]
                    logger.info(
                        "Ignoring stale eligibility value on Yes/No step call_id=%s step=%s value=%r",
                        call_id,
                        session.elig_step,
                        text[:80],
                    )
                    result = {
                        "ok": True,
                        "call_id": call_id,
                        "field_id": "eligibility_answer",
                        "value": text,
                        "action": "stale_ignored",
                        "spoken": "ASK: " + spoken,
                        "message": "stale answer ignored",
                        "step": session.elig_step,
                    }
                    session.last_result = result
                    if request_id:
                        session.processed_requests[request_id] = result
                    return result
                step_key = (session.elig_step, text.lower())
                if session.last_step_answer == step_key and session.last_result:
                    return session.last_result
                result = await _submit_eligibility_answer(
                    call_id, session, page, text, started
                )
                if result.get("ok"):
                    session.last_step_answer = step_key
                    session.last_result = result
                    session.last_applied = text.lower()
                    session.last_applied_at = time.time()
                if request_id:
                    session.processed_requests[request_id] = result
                return result
            if field_key in (
                "spoken_choice",
                "marital_status",
                "has_children",
                "household_income",
                "special_conditions",
            ):
                target = _front_page(session)
                if session.elig_step == "marital":
                    body = (await target.locator("body").inner_text()).lower()
                    if "what is your marital status" not in body:
                        if not await _reset_eligibility_to_first_question(target):
                            raise RuntimeError(
                                "website is not on the marital-status question"
                            )
                kind = _answer_kind(field_key, text)
                if kind == "ambiguous":
                    raise RuntimeError(f"ambiguous answer {text!r}")
                before_url = target.url
                before_screen = await _screen_signature(target)
                before_choice = await _choice_signature(target, text)
                _mark(call_id, "browser_action", started)
                await _click_answer(target, text)
                if not await _verify_update(
                    target, before_url, text, kind, before_screen, before_choice
                ):
                    raise RuntimeError(f"click not verified for {text!r} on {target.url}")
                _mark(call_id, "verified", started)
                msg = f"Clicked {text!r}."
                session.history.append(
                    {"field_id": field_key, "value": text, "action": "click", "ok": True}
                )
                result = _result_with_next(session, text, _ok(call_id, field_key, text, "click", session, msg))
                _mark(call_id, "next_question", started)
                _mark(call_id, "tool_response", started)
                session.last_applied = text.lower()
                session.last_applied_at = time.time()
                session.last_result = result
                return result
            if field_key == "phone":
                digits = re.sub(r"\D", "", text)
                if len(digits) >= 10:
                    text = digits[-10:]
            if field_key == "otp":
                digits = re.sub(r"\D", "", text)
                if digits:
                    text = digits
            if field_key == "dob":
                text = _normalize_dob(text)
            if field_key in ("google", "gmail", "continue_with_google", "sign_in_with_google", "sign_in_google"):
                msg = await _sign_in_with_provider(session, "Continue with Google", text)
                entry = {"field_id": "continue_with_google", "value": text, "action": "sso", "ok": True}
                session.history.append(entry)
                spoken = msg[4:].strip() if str(msg).startswith("ASK:") else "Saved."
                return {
                    "ok": True,
                    "call_id": call_id,
                    "field_id": "continue_with_google",
                    "value": text,
                    "action": "wait" if not text else "sso",
                    "message": spoken,
                    "spoken": spoken,
                }
            if field_key in ("apple", "continue_with_apple", "sign_in_with_apple"):
                msg = await _sign_in_with_provider(session, "Continue with Apple", text)
                entry = {"field_id": "continue_with_apple", "value": text, "action": "sso", "ok": True}
                session.history.append(entry)
                return _ok(call_id, "continue_with_apple", text, "sso", session, msg)
            if field_key == "email":
                text = _normalize_email(text)
                if not text:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "email",
                        "error": "email requires the caller's address",
                        "spoken": "What email address should I use?",
                        "auth_screen": "need_email",
                    }
                _mark(call_id, "browser_action", started)
                email_result = await _sign_in_with_email(session, text)
                _mark(call_id, "verified", started)
                spoken = str(email_result.get("spoken") or EMAIL_SECURE_COMPLETE)
                result = {
                    "ok": bool(email_result.get("ok")),
                    "call_id": call_id,
                    "field_id": "email",
                    "value": "[redacted]",
                    "action": email_result.get("action") or "email_auth",
                    "message": spoken,
                    "spoken": spoken,
                    "auth_screen": email_result.get("auth_screen") or "",
                    "awaiting_secure": bool(email_result.get("awaiting_secure")),
                    "step": email_result.get("step") or "",
                    "error": email_result.get("error") or "",
                }
                session.last_result = result
                if request_id:
                    session.processed_requests[request_id] = result
                _mark(call_id, "tool_response", started)
                return result

            if spec and spec.kind == "action":
                await _click_action(_front_page(session), field_key if field_key in ACTION_BUTTONS else "continue")
                msg = f"Clicked {field_key}."
            elif spec and spec.kind == "click":
                await _click_named(_front_page(session), spec.button_names or spec.labels)
                msg = f"Clicked {field_key}."
            elif spec and spec.kind == "choice":
                if not text:
                    return {"ok": False, "error": f"{field_key} needs a choice value"}
                await _click_choice(_front_page(session), text)
                msg = f"Selected {text!r} for {field_key}."
            else:
                if "first" in field_key and "name" in field_key:
                    field_key = "first_name"
                    spec = get_field(field_key)
                elif "last" in field_key and "name" in field_key:
                    field_key = "last_name"
                    spec = get_field(field_key)
                labels = spec.labels if spec else (field_key.replace("_", " "),)
                placeholders = spec.placeholders if spec else ()
                input_types = spec.input_types if spec else ()
                if not text and act:
                    await _click_action(page, act)
                    msg = f"Clicked {act}."
                elif not text:
                    return {"ok": False, "error": f"{field_key} needs a value"}
                else:
                    target = _front_page(session)
                    kind = _answer_kind(field_key, text)
                    if kind == "ambiguous":
                        raise RuntimeError(f"ambiguous answer {text!r}")
                    before_url = target.url
                    before_screen = await _screen_signature(target)
                    before_choice = await _choice_signature(target, text)
                    _mark(call_id, "browser_action", started)
                    if kind == "yesno":
                        await _click_answer(target, text)
                    elif kind == "first_name":
                        await _fill_text(target, ("first name",), ("first name",), ("text",), text)
                    elif kind == "last_name":
                        await _fill_text(target, ("last name",), ("last name",), ("text",), text)
                    elif kind == "date":
                        await _fill_text(target, ("date of birth", "birth"), ("date", "mm"), ("text",), text)
                    elif kind == "email_text":
                        await _fill_text(
                            target,
                            labels,
                            placeholders,
                            input_types or ("email", "text"),
                            text,
                        )
                    elif kind == "income":
                        await _click_choice(target, text)
                    else:
                        await _click_choice(target, text)
                    if not await _verify_update(
                        target, before_url, text, kind, before_screen, before_choice
                    ):
                        raise RuntimeError(f"update not verified for {text!r}")
                    _mark(call_id, "verified", started)
                    if field_key == "address":
                        await _pick_address_suggestion(target, text)
                    session.filled[field_key] = text
                    msg = f"Entered {field_key}."
                    if act:
                        await target.wait_for_timeout(350)
                        await _click_action(target, act)
                        msg += f" Clicked {act}."

            entry = {"field_id": field_key, "value": text, "action": act, "ok": True}
            session.history.append(entry)
            logger.info(
                "Live filled call_id=%s field=%s action=%s",
                call_id,
                field_key,
                act or "-",
            )
            result = _result_with_next(
                session,
                text,
                {
                    "ok": True,
                    "call_id": call_id,
                    "field_id": field_key,
                    "value": text,
                    "action": act,
                    "filled": dict(session.filled),
                    "message": msg,
                    "spoken": "Saved.",
                },
            )
            _mark(call_id, "next_question", started)
            _mark(call_id, "tool_response", started)
            session.last_applied = text.lower()
            session.last_applied_at = time.time()
            session.last_result = result
            return result
        except Exception as exc:
            logger.exception("submit_field failed call_id=%s field=%s", call_id, field_key)
            session.error = str(exc)
            _mark(call_id, "tool_response", started)
            err_text = str(exc)
            if "APPLICATION_AFTER_SIGNIN:" in err_text:
                detail = err_text.split(":", 1)[1].strip()
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": field_key,
                    "error": "application_after_signin",
                    "spoken": "ASK: " + detail[:400],
                    "message": detail[:400],
                    "step": session.elig_step,
                }
            # Clarifications must keep the active step (never treat special_person
            # failures as a screen advance back to special).
            if err_text.startswith(
                (
                    "SUPPORT_CLARIFY:",
                    "INCOME_CLARIFY:",
                    "AGE_CLARIFY:",
                    "REL_CLARIFY:",
                    "CHILD_FORM:",
                    "CHILD_MORE_CLARIFY:",
                    "SPECIAL_CLARIFY:",
                    "SPECIAL_PERSON_CLARIFY:",
                    "SPECIAL_MORE_CLARIFY:",
                )
            ):
                clarify = err_text.split(":", 1)[1].strip()
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": field_key,
                    "error": "clarify",
                    "spoken": "ASK: " + clarify,
                    "message": clarify,
                    "step": session.elig_step,
                }
            # If the browser clearly advanced to a *different* page group, ask that
            # question. Never treat "still on the same form" as success (e.g.
            # child_more Continue stuck on child_form must not restart at child_first
            # or commit the failed No).
            try:
                if page is not None and (
                    field_key == "eligibility_answer" or session.elig_step
                ):
                    screen = await _eligibility_screen(page)
                    failed_group = _step_group(session.elig_step)
                    recovered = _screen_first_step(screen, session)
                    same_form = (
                        screen == failed_group
                        or screen == "unknown"
                        or (
                            recovered
                            and _step_group(recovered) == failed_group
                        )
                    )
                    if (
                        not same_form
                        and recovered
                        and recovered in ELIG_QUESTIONS
                        and recovered != session.elig_step
                        # Never roll special_person / detail steps back to special.
                        and not (
                            session.elig_step
                            in (
                                "special_person",
                                "special_more",
                                "babies_count",
                                "due_date",
                            )
                            and recovered == "special"
                        )
                    ):
                        session.elig_step = recovered
                        return _elig_ok(
                            call_id,
                            session,
                            text,
                            recovered,
                            f"Recovered after browser advance despite: {exc}",
                        )
                    if same_form and session.elig_step == "child_more":
                        hints = await _visible_validation_hints(page)
                        clarify = (
                            "I couldn't leave the child form after Continue"
                            + (f" ({hints})" if hints else "")
                            + ". Please say No again once the form looks complete, "
                            "or say Yes if you need to add another child."
                        )
                        return {
                            "ok": False,
                            "call_id": call_id,
                            "field_id": field_key,
                            "error": "clarify",
                            "spoken": "ASK: " + clarify,
                            "message": clarify,
                            "step": session.elig_step,
                        }
            except Exception:
                logger.exception("Could not recover eligibility screen after failure")
            repeat = ELIG_QUESTIONS.get(session.elig_step)
            if not repeat and field_key in (
                "google",
                "gmail",
                "continue_with_google",
                "sign_in_with_google",
                "sign_in_google",
            ):
                repeat = "Google sign-in could not be completed. Please say Google again when the sign-in page is ready."
            if not repeat:
                repeat = "Please say that answer again."
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": field_key,
                "error": str(exc),
                "spoken": "ASK: " + repeat,
            }


def _step_group(step: str) -> str:
    if step.startswith("spouse_"):
        return "spouse_form"
    if step.startswith("child_"):
        return "child_form"
    if step.startswith("person_"):
        return "person_form"
    if step in ("income_when", "income_range"):
        return "income"
    if step in (
        "special",
        "special_person",
        "special_more",
        "babies_count",
        "due_date",
    ):
        return "special"
    return step


async def _eligibility_screen(page: Page) -> str:
    url = (page.url or "").lower()
    # Prefer stable Pointcare route segments when available.
    url_map = (
        ("marital-status", "marital"),
        ("spouse", "spouse_form"),
        ("partner", "spouse_form"),
        ("live-care-children", "children"),
        ("parent-guardian-student", "student"),
        ("support-disabled", "disabled"),
        ("other-relative", "relatives"),
        ("relative", "relatives"),
        ("half-support", "support"),
        ("your-support", "support"),
        ("household-income", "income"),
        ("income", "income"),
        ("special-condition", "special"),
        ("special", "special"),
        ("program", "programs"),
    )
    for needle, screen in url_map:
        if needle in url:
            # Detail forms can keep the prior route fragment; refine with visible UI.
            break
    else:
        needle, screen = "", ""

    try:
        body = re.sub(r"\s+", " ", (await page.locator("body").inner_text()).lower())
    except Exception:
        body = ""
    labels = {item.lower() for item in await _visible_choice_labels(page)}
    has_yes_no = "yes" in labels and "no" in labels
    relationship_chips = bool(
        labels & {"child", "stepchild", "grandchild", "foster child"}
    )
    age_chips = any(
        re.search(r"\(\d", item) or "young adult" in item or "youth" in item
        for item in labels
    )

    # Person/child detail forms must win over Yes/No question text that may remain
    # in headings after navigation (this caused student Yes to hang forever).
    # A bare "Child" relationship chip must NOT force child_form — person forms
    # for disabled/relatives also offer Child as a relationship option.
    child_form_heading = bool(
        re.search(
            r"tell us about (your |the |this )?child\b|"
            r"\babout your child\b|"
            r"\bchild'?s (first|last) name\b",
            body,
        )
    )
    if relationship_chips or age_chips or (
        "first name" in body and "last name" in body and not has_yes_no
    ):
        if child_form_heading:
            return "child_form"
        if ("spouse" in body or "partner" in body) and not relationship_chips:
            return "spouse_form"
        return "person_form"
    if child_form_heading and "first name" in body:
        return "child_form"
    if "tell us about" in body and "first name" in body and "relationship" in body:
        return "person_form"
    if "tell us about your spouse or partner" in body:
        return "spouse_form"

    if needle and screen in (
        "marital",
        "children",
        "student",
        "disabled",
        "relatives",
        "support",
        "income",
        "special",
        "programs",
        "spouse_form",
    ):
        # Only trust Yes/No route segments when Yes/No controls are still visible.
        if screen in ("children", "student", "disabled", "relatives", "support"):
            if has_yes_no or ("yes" in body and "no" in body):
                return screen
        else:
            return screen

    if not body:
        return "unknown"
    if "what is your marital status" in body:
        return "marital"
    # Support options literally contain "full-time student" / "disabled" text, so
    # detect this screen before student/disabled keyword matches.
    if (
        "more than half of your support" in body
        or "more than half of your annual financial support" in body
        or "half of your annual financial support" in body
    ):
        return "support"

    if has_yes_no or ("yes" in body and "no" in body):
        if (
            "children under the age of 19" in body
            or "children under 19" in body
            or "live with and care for any children" in body
            or "mainly support any children" in body
        ):
            return "children"
        if (
            ("full-time student" in body or "full time student" in body)
            and "more than half" not in body
            and "parent or legal guardian supports me" not in body
        ):
            return "student"
        if (
            "disabled individual" in body
            or "disabled person" in body
            or "support a disabled" in body
            or (
                "disabled person of any age" in body
                and "live year-round with the person that supports me" not in body
            )
        ):
            return "disabled"
        if "other relatives" in body or "including in-laws" in body:
            return "relatives"
    if "household income" in body or (
        "annual" in body and "monthly" in body and "every two weeks" in body
    ):
        return "income"
    if "breast cancer" in body or "cervical cancer" in body or "foster care at 18" in body:
        return "special"
    if (
        "great news" in body
        or "start applications" in body
        or "continue to applications" in body
        or ("programs" in body and ("qualify" in body or "eligible" in body))
        or ("not eligible at this time" in body and "eligible" in body)
    ):
        return "programs"
    return "unknown"


def _screen_first_step(screen: str, session: LiveFormSession) -> str:
    if screen == "spouse_form":
        if session.elig_step and str(session.elig_step).startswith("spouse_"):
            return session.elig_step
        return "spouse_first"
    if screen == "child_form":
        if session.elig_step and str(session.elig_step).startswith("child_"):
            return session.elig_step
        return "child_first"
    if screen == "person_form":
        if session.elig_step and str(session.elig_step).startswith("person_"):
            return session.elig_step
        return "person_first"
    if screen == "income":
        # Do not roll income_range back to income_when on failed clicks / reconnect.
        if session.elig_step == "income_range":
            return "income_range"
        return "income_when"
    if screen == "special":
        # Keep member-assignment / more-conditions / pregnancy detail once selected.
        if session.elig_step in (
            "special_person",
            "special_more",
            "babies_count",
            "due_date",
        ):
            return session.elig_step
        return "special"
    if screen in ELIG_QUESTIONS:
        return screen
    return session.elig_step


def _income_range_question_for_period(period: str) -> str:
    """Voice-friendly income amount ask based on the selected period.

    Never enumerates the visible dollar-band chips — callers are on the phone.
    """
    key = _income_period_choice(period or "")
    if key == "Monthly":
        return (
            "About how much does your household earn per month before taxes?"
        )
    if key == "Bi-weekly":
        return (
            "About how much does your household earn every two weeks before taxes?"
        )
    return "About how much does your household earn per year before taxes?"


_REL_OPTION_SKIP = frozenset(
    {
        "first name",
        "last name",
        "living with me",
        "same mailing address",
        "relationship to me",
        "date of birth (optional)",
        "age",
        "continue",
        "back",
        "exit",
        "add another",
    }
)


def _join_spoken_choices(labels: list[str]) -> str:
    clean = [item.strip() for item in labels if item and item.strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} or {clean[1]}"
    return ", ".join(clean[:-1]) + f", or {clean[-1]}"


def _relationship_option_labels(labels: list[str] | None) -> list[str]:
    """Keep relationship chips; drop name/age/form chrome."""
    out: list[str] = []
    for label in labels or []:
        low = (label or "").strip().lower().rstrip(" *")
        if not low or low in _REL_OPTION_SKIP or len(label) > 60:
            continue
        if re.match(r"^\$?\d", low) or re.search(r"\d+\s*[-–—]\s*\d+", low):
            continue
        if re.search(
            r"child|step|grand|foster|other|spouse|partner|parent|sibling|"
            r"brother|sister|relative|niece|nephew|cousin|aunt|uncle|in-?law|"
            r"guardian|friend|roommate",
            low,
            re.I,
        ):
            if label not in out:
                out.append(label)
    return out


def _default_relationship_options(step: str) -> list[str]:
    """Authoritative fallback options when the live control is not readable yet."""
    from .elig_meta import get_field_meta

    meta = get_field_meta(step)
    if meta.options_hint:
        return list(meta.options_hint)
    if step == "spouse_role":
        return ["Spouse", "Partner"]
    return ["Child", "Stepchild", "Grandchild", "Foster Child", "Other"]


def relationship_question_speech(
    step: str, options: list[str] | None = None
) -> str:
    """Initial relationship question that always reads the applicable options.

    Prefer live/visible labels; fall back to field metadata. Never invent Other
    when the live option list is present without it. Never lists name/age chrome.
    """
    live = _relationship_option_labels(options) if options is not None else []
    opts = live if live else _default_relationship_options(step)
    joined = _join_spoken_choices(opts[:8]) or "the options on the form"
    if step == "spouse_role":
        return (
            f"Are they your spouse or your partner? The options are {joined}. "
            "Which one applies?"
        )
    return (
        f"What is their relationship to you? The options are {joined}. "
        "Which best describes their relationship to you?"
    )


def build_options_help_speech(step: str, options: list[str] | None = None) -> str:
    """Natural spoken list of available choices for the active field.

    Never includes internal validation text. Never enumerates income bands.
    """
    from .elig_meta import get_field_meta

    meta = get_field_meta(step)
    if step == "income_range" or meta.answer_type == "income_range":
        # Caller may not have session; keep period-agnostic if unknown.
        return (
            "Please say an approximate household income amount for the period "
            "you selected — for example forty thousand, or a range like forty "
            "to fifty thousand. I won't read every dollar range aloud."
        )
    if step in ("child_rel", "person_rel", "spouse_role") or (
        meta.answer_type == "choice" and "rel" in step
    ):
        # Re-read options clearly without the generic "required field" chrome.
        return relationship_question_speech(step, options)
    if step in ("spouse_age", "child_age", "person_age") or meta.answer_type == "age":
        age_opts = _age_option_labels(options or [])
        if age_opts:
            joined = _join_spoken_choices(age_opts[:6])
            return f"The age choices are {joined}. Which one fits?"
        return (
            "Please say their age as a number, like twenty five, "
            "or the age range that matches."
        )
    if step == "income_when" or meta.answer_type == "income_period":
        return (
            "You can say Annual for yearly income, Monthly, or Bi-weekly "
            "for every two weeks. Which period do you know best?"
        )
    if step == "marital":
        opts = [o for o in (options or list(meta.options_hint)) if o][:7]
        joined = _join_spoken_choices(opts) if opts else (
            "Never Married, Married, Separated, Divorced, Widowed, "
            "Registered Domestic Partner, or Common Law Marriage"
        )
        return f"The marital status choices are {joined}. Which one applies?"
    if meta.answer_type == "yes_no" or step in YES_NO_ELIG_STEPS:
        base = (ELIG_QUESTIONS.get(step) or "").strip()
        return (base + " Please say Yes or No.").strip()
    if meta.answer_type == "choice":
        shown = [item for item in (options or list(meta.options_hint)) if item][:8]
        if shown:
            joined = _join_spoken_choices(shown)
            base = (ELIG_QUESTIONS.get(step) or "").strip()
            prefix = f"{base} " if base else ""
            return f"{prefix}The choices are {joined}. Which one applies?".strip()
    base = (ELIG_QUESTIONS.get(step) or "").strip()
    return base or "Could you say that again?"


def _spoken_question_for_step(session: LiveFormSession, step: str) -> str:
    """Build the server-controlled spoken question, with one-time section intros."""
    if step in ("proceed_application", "confirm_start_application"):
        stored = (session.filled.get("application_gate_speech") or "").strip()
        if stored:
            return stored
    if step == "programs":
        stored = (session.filled.get("programs_permission_speech") or "").strip()
        if stored:
            return stored
    if step == "income_range":
        return _income_range_question_for_period(
            session.filled.get("income_when") or "Annual"
        )
    # Relationship questions always announce the applicable options up front.
    if step in ("child_rel", "person_rel", "spouse_role"):
        cached = (session.filled.get(f"{step}_options_speech") or "").strip()
        if cached:
            return cached
        return relationship_question_speech(step)
    base = (ELIG_QUESTIONS.get(step) or "").strip()
    if not base:
        return "Could you say that again?"
    if step == "marital" and session.filled.get("eligibility_intro_pending") == "1":
        session.filled.pop("eligibility_intro_pending", None)
        session.filled["eligibility_intro_spoken"] = "1"
        return ELIGIBILITY_SECTION_INTRO + base
    if step == "spouse_confirm":
        return _member_confirm_speech(session, "spouse")
    if step == "child_confirm":
        return _member_confirm_speech(session, "child")
    if step == "person_confirm":
        return _member_confirm_speech(session, "person")
    if step == "spouse_first" and not session.filled.get("spouse_section_intro"):
        session.filled["spouse_section_intro"] = "1"
        return "Let's learn more about your spouse or partner. " + base
    if step == "child_first":
        inst = _ensure_child_instance(session)
        key = f"child_section_intro_{inst}"
        if not session.filled.get(key):
            session.filled[key] = "1"
            if inst == "1":
                return "Let's learn more about your child. " + base
            return "Let's learn more about the next child. " + base
    if step == "person_first" and not session.filled.get("person_section_intro"):
        session.filled["person_section_intro"] = "1"
        return _person_section_intro(session) + base
    return base


async def _relationship_speech_for_page(
    session: LiveFormSession, page: Page, step: str
) -> str:
    """Build relationship speech from live chips when visible; cache on session."""
    labels: list[str] = []
    try:
        labels = await _visible_choice_labels(page)
    except Exception:
        labels = []
    rel_opts = _relationship_option_labels(labels)
    speech = relationship_question_speech(step, rel_opts or None)
    session.filled[f"{step}_options_speech"] = speech
    return speech


async def _enter_programs_step(
    session: LiveFormSession, page: Page
) -> tuple[str, str]:
    """Land on program-summary, scrape live results, return (step, spoken_override)."""
    from .app_stage import (
        build_programs_permission_speech,
        store_program_summary,
    )

    try:
        await store_program_summary(session, page)
    except Exception:
        logger.exception("enter_programs_step scrape failed; using fallback speech")
    speech = (session.filled.get("programs_permission_speech") or "").strip()
    if not speech:
        # Never leave Stage D without a consent question after Next to programs.
        speech = build_programs_permission_speech({"programs": []})
        session.filled["programs_permission_speech"] = speech
    return "programs", speech


def _section_context_for_step(step: str) -> str:
    """Brief section context for explain/repeat-without-advance requests."""
    if step.startswith("spouse_"):
        return "These questions are about your spouse or partner."
    if step.startswith("child_"):
        return "These questions are about the child we're adding."
    if step.startswith("person_"):
        return "These questions are about the household member we're adding."
    return ""


def _elig_ok(
    call_id: str,
    session: LiveFormSession,
    answer: str,
    next_step: str,
    message: str,
    *,
    spoken_override: str | None = None,
    await_field_correction: bool = False,
) -> dict[str, Any]:
    session.elig_step = next_step
    if spoken_override is not None:
        spoken = spoken_override
    elif next_step:
        spoken = _spoken_question_for_step(session, next_step)
    else:
        spoken = (
            "I've finished filling your household eligibility form. "
            "Thank you for your time. Goodbye."
        )
    out: dict[str, Any] = {
        "ok": True,
        "call_id": call_id,
        "field_id": "eligibility_answer",
        "value": answer,
        "action": "verified",
        "message": message,
        "spoken": "ASK: " + spoken,
        "step": next_step,
    }
    if await_field_correction:
        out["await_field_correction"] = True
    return out


def _remember_household_name(session: LiveFormSession, name: str) -> None:
    names = [
        item.strip()
        for item in session.filled.get("household_names", "").split("|")
        if item.strip()
    ]
    if name.strip() and name.strip().lower() not in {item.lower() for item in names}:
        names.append(name.strip())
    session.filled["household_names"] = "|".join(names)


def _child_instance_id(session: LiveFormSession | None) -> str:
    """Stable id for the active child so duplicate answers across kids don't collide."""
    if session is None:
        return "1"
    raw = (session.filled.get("child_instance") or "1").strip()
    return raw if raw.isdigit() and int(raw) >= 1 else "1"


def _ensure_child_instance(session: LiveFormSession) -> str:
    inst = _child_instance_id(session)
    session.filled["child_instance"] = inst
    return inst


def _bump_child_instance(session: LiveFormSession) -> str:
    """Start a new child instance after Add Another succeeds."""
    current = int(_child_instance_id(session))
    nxt = str(current + 1)
    session.filled["child_instance"] = nxt
    # Clear per-child fill cache so the new blank form is not treated as done.
    for key in (
        "child_first",
        "child_last",
        "child_rel",
        "child_age",
    ):
        session.filled.pop(key, None)
    return nxt


async def _visible_validation_hints(page: Page) -> str:
    """Visible validation text, including plain copy outside alert/error nodes.

    Pointcare renders "Please select an option" as ordinary text under the
    field. Alert/error selectors alone missed that and let a click count as
    success while the relationship was still blank.
    """
    try:
        texts = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const out = [];
              const push = (raw) => {
                const t = norm(raw);
                if (!t || t.length > 160 || out.includes(t)) return;
                out.push(t);
              };
              const isVisible = (el) => {
                if (!el || !el.getBoundingClientRect) return false;
                const r = el.getBoundingClientRect();
                if (r.width <= 2 || r.height <= 2) return false;
                const style = window.getComputedStyle ? window.getComputedStyle(el) : null;
                if (style && (style.visibility === 'hidden' || style.display === 'none')) {
                  return false;
                }
                return true;
              };
              const demandsSelection = (t) => {
                const low = t.toLowerCase();
                return low.includes('please select')
                  || low.includes('select an option')
                  || low.includes('this field is required')
                  || low.includes('field is required');
              };
              const alertNodes = [...document.querySelectorAll(
                '[role=alert], [aria-invalid=true], .error, .invalid, '
                + '[class*="error"], [class*="invalid"], [class*="validation"]'
              )];
              for (const node of alertNodes) {
                if (!isVisible(node)) continue;
                push(node.innerText || node.textContent || '');
                if (out.length >= 8) return out;
              }
              const leaves = [...document.querySelectorAll(
                'p, span, div, label, li, small, strong, em'
              )];
              for (const el of leaves) {
                if (!isVisible(el)) continue;
                let own = '';
                for (const n of el.childNodes) {
                  if (n.nodeType === 3) own += n.textContent || '';
                }
                own = norm(own);
                if (own && demandsSelection(own)) push(own);
                if (out.length >= 8) break;
              }
              return out;
            }"""
        )
    except Exception:
        return ""
    return "; ".join(texts or [])


async def _body_contains(page: Page, phrases: tuple[str, ...]) -> bool:
    try:
        body = re.sub(r"\s+", " ", (await page.locator("body").inner_text()).lower())
    except Exception:
        return False
    return any(phrase.lower() in body for phrase in phrases)


async def _wait_for_screen(
    page: Page,
    screens: tuple[str, ...],
    timeout_s: float = 12,
) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = await _eligibility_screen(page)
        if current in screens:
            return current
        await page.wait_for_timeout(250)
    raise RuntimeError(
        f"Pointcare did not reach {screens}; current screen is "
        f"{await _eligibility_screen(page)}"
    )


def _sanitize_fill_value(label: str, value: str) -> str:
    """Normalize ASR noise for name fields before typing into Pointcare."""
    text = (value or "").strip()
    if re.search(r"name", label or "", re.I):
        from .answer_parse import sanitize_person_name

        text = sanitize_person_name(text)
    return text


async def _locate_named_input(page: Page, label: str):
    """Find a visible text input by accessible name, placeholder, or label text."""
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*\*?\s*$", re.I)
    loose = re.compile(re.escape(label), re.I)
    candidates = [
        page.get_by_role("textbox", name=pattern).first,
        page.get_by_label(pattern).first,
        page.get_by_placeholder(loose).first,
        page.locator("label").filter(has_text=pattern).locator("input, textarea").first,
        page.locator("label").filter(has_text=pattern).locator("..").locator(
            "input, textarea"
        ).first,
        page.locator(
            f"input[aria-label*='{label}' i], textarea[aria-label*='{label}' i]"
        ).first,
    ]
    # Pointcare sometimes uses name/id attributes without a wired label.
    attr_key = "first" if "first" in label.lower() else (
        "last" if "last" in label.lower() else label.lower().replace(" ", "")
    )
    if attr_key in {"first", "last"}:
        candidates.append(
            page.locator(
                f"input[name*='{attr_key}' i][type='text'], "
                f"input[id*='{attr_key}' i], "
                f"input[name*='{attr_key}Name' i], "
                f"input[name*='{attr_key}_name' i]"
            ).first
        )
    for loc in candidates:
        try:
            if await loc.count() and await loc.is_visible():
                # Prefer enabled, editable inputs.
                try:
                    if await loc.is_disabled():
                        continue
                except Exception:
                    pass
                return loc
        except Exception:
            continue
    return None


async def _name_field_visible(page: Page, label: str) -> bool:
    return (await _locate_named_input(page, label)) is not None


async def _wait_for_name_field(
    page: Page,
    label: str,
    *,
    timeout_s: float = 12,
) -> bool:
    """Condition-based wait until a named input is visible and actionable."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await _name_field_visible(page, label):
            return True
        await page.wait_for_timeout(200)
    return False


async def _wait_for_child_form_ready(page: Page, *, timeout_s: float = 14) -> bool:
    """Wait until Pointcare shows the child details form with First Name ready."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        screen = await _eligibility_screen(page)
        if screen == "child_form" and await _name_field_visible(page, "First Name"):
            return True
        # Some builds expose the inputs before the screen classifier settles.
        if await _name_field_visible(page, "First Name") and await _name_field_visible(
            page, "Last Name"
        ):
            body = ""
            try:
                body = re.sub(
                    r"\s+", " ", (await page.locator("body").inner_text()).lower()
                )
            except Exception:
                body = ""
            if "child" in body or "tell us about" in body:
                return True
        await page.wait_for_timeout(200)
    return False


async def _read_named_input(page: Page, label: str) -> str:
    loc = await _locate_named_input(page, label)
    if loc is None:
        return ""
    try:
        return (await loc.input_value()).strip()
    except Exception:
        return ""


async def _locate_other_relationship_input(page: Page):
    """Find the free-text field that appears after selecting Other relationship."""
    labels = (
        "Please specify",
        "Specify relationship",
        "Other relationship",
        "Describe relationship",
        "Relationship",
        "Relationship to me",
    )
    for label in labels:
        loc = await _locate_named_input(page, label)
        if loc is None:
            continue
        # Prefer a field that is not First/Last Name.
        try:
            aria = (
                (await loc.get_attribute("aria-label") or "")
                + " "
                + (await loc.get_attribute("placeholder") or "")
                + " "
                + (await loc.get_attribute("name") or "")
            ).lower()
        except Exception:
            aria = ""
        if "first" in aria or "last" in aria:
            continue
        return loc
    # Fallback: visible text input near "Other" that is not a name field.
    try:
        found = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const inputs = [...document.querySelectorAll('input, textarea')].filter((el) => {
                const r = el.getBoundingClientRect();
                if (r.width < 2 || r.height < 2) return false;
                if (el.disabled || el.readOnly) return false;
                const type = (el.getAttribute('type') || 'text').toLowerCase();
                if (type && type !== 'text' && type !== 'search' && el.tagName !== 'TEXTAREA') {
                  return false;
                }
                const meta = norm(
                  (el.getAttribute('aria-label') || '') + ' ' +
                  (el.getAttribute('placeholder') || '') + ' ' +
                  (el.getAttribute('name') || '') + ' ' +
                  (el.id || '')
                );
                if (meta.includes('first') || meta.includes('last') || meta.includes('email')) {
                  return false;
                }
                return meta.includes('relat') || meta.includes('other') ||
                  meta.includes('specify') || meta.includes('describe');
              });
              return inputs.length ? true : false;
            }"""
        )
        if found:
            return page.locator(
                "input[aria-label*='relat' i], input[placeholder*='relat' i], "
                "input[aria-label*='other' i], input[placeholder*='specify' i], "
                "input[name*='relat' i], textarea[aria-label*='relat' i]"
            ).first
    except Exception:
        pass
    return None


async def _other_relationship_input_visible(page: Page) -> bool:
    loc = await _locate_other_relationship_input(page)
    if loc is None:
        return False
    try:
        return bool(await loc.count() and await loc.is_visible())
    except Exception:
        return False


async def _fill_other_relationship_text(page: Page, value: str) -> str:
    """Fill the Other free-text relationship field and verify the value."""
    # Keep words like "in-law"; do not map onto predefined relationship chips.
    cleaned = re.sub(r"\s+", " ", (value or "").strip()).strip(" .")
    if not cleaned:
        raise RuntimeError(
            "REL_CLARIFY: What is their relationship to you?"
        )
    loc = await _locate_other_relationship_input(page)
    if loc is None:
        raise RuntimeError(
            "REL_CLARIFY: I selected Other, but I don't see a place to type the "
            "custom relationship. Please say a listed relationship, or try Other again."
        )
    await loc.click(timeout=4000)
    await loc.fill("", timeout=3000)
    await loc.fill(cleaned, timeout=6000)
    try:
        actual = (await loc.input_value()).strip()
    except Exception:
        actual = ""
    if actual.lower() != cleaned.lower() and cleaned.lower() not in actual.lower():
        raise RuntimeError(
            "REL_CLARIFY: I couldn't save that custom relationship. "
            "Please say their relationship again."
        )
    return actual or cleaned


def _name_reset_token(session: LiveFormSession | None, kind: str) -> str:
    """One reset per new spouse, child instance, or additional-person kind."""
    if kind == "child":
        return f"child:{_child_instance_id(session)}"
    if kind == "person":
        who = "person"
        if session is not None:
            who = (session.filled.get("person_kind") or "person").strip().lower() or "person"
        return f"person:{who}"
    return "spouse"


def _saved_name_blocks_reset(session: LiveFormSession | None, kind: str) -> bool:
    """Do not blank a person who already has a saved first name on this entry."""
    if session is None:
        return False
    if kind == "spouse":
        return bool((session.filled.get("spouse_first") or "").strip())
    if kind == "child":
        return bool((session.filled.get("child_first") or "").strip())
    return False


_ACTIVE_NAME_INDEX_JS = """
() => {
  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase()
    .replace(/[*:]/g, '').trim();
  const visible = (el) => {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const st = window.getComputedStyle(el);
    return st.visibility !== 'hidden' && st.display !== 'none';
  };
  const all = [...document.querySelectorAll('input, textarea')];
  const labelFor = (el) => {
    const aria = norm(el.getAttribute('aria-label') || '');
    if (aria) return aria;
    const ph = norm(el.getAttribute('placeholder') || '');
    if (ph) return ph;
    if (el.id) {
      const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (lab) return norm(lab.innerText || lab.textContent || '');
    }
    const wrap = el.closest('label');
    if (wrap) return norm(wrap.innerText || wrap.textContent || '');
    let prev = el.previousElementSibling;
    for (let i = 0; i < 3 && prev; i++) {
      const t = norm(prev.innerText || prev.textContent || '');
      if (t && t.length < 48) return t;
      prev = prev.previousElementSibling;
    }
    return '';
  };
  const kindOf = (text) => {
    if (text === 'first name' || text.startsWith('first name')) return 'first';
    if (text === 'last name' || text.startsWith('last name')) return 'last';
    return '';
  };
  const tagged = [];
  all.forEach((el, index) => {
    const type = (el.getAttribute('type') || 'text').toLowerCase();
    if (!visible(el) || el.disabled) return;
    if (!['text', 'search', 'tel', ''].includes(type)) return;
    const kind = kindOf(labelFor(el));
    if (kind) tagged.push({el, index, kind});
  });
  let best = null;
  for (const item of tagged) {
    if (item.kind !== 'first') continue;
    let node = item.el.parentElement;
    for (let depth = 0; depth < 8 && node; depth++) {
      const inside = tagged.filter((row) => node.contains(row.el));
      const firsts = inside.filter((row) => row.kind === 'first');
      const lasts = inside.filter((row) => row.kind === 'last');
      if (firsts.length === 1 && lasts.length === 1) {
        if (!best || best.node.contains(node)) {
          best = {node, first: firsts[0].index, last: lasts[0].index};
        }
        break;
      }
      node = node.parentElement;
    }
  }
  if (!best) return {first: -1, last: -1};
  return {first: best.first, last: best.last};
}
"""


async def _active_name_input_indexes(page: Page) -> dict[str, int] | None:
    """Indexes of First/Last Name inside the visible form that has both.

    None means the page could not be read, so the caller must not keep waiting.
    """
    try:
        found = await page.evaluate(_ACTIVE_NAME_INDEX_JS)
    except Exception:
        logger.exception("Could not locate active person name inputs")
        return None
    if not isinstance(found, dict):
        return None
    return {
        "first": int(found.get("first", -1)),
        "last": int(found.get("last", -1)),
    }


async def _blank_named_input_at(page: Page, index: int) -> None:
    if index < 0:
        return
    loc = page.locator("input, textarea").nth(index)
    try:
        current = (await loc.input_value()).strip()
        if current:
            await loc.fill("", timeout=3000)
    except Exception:
        logger.exception("Could not clear name input at index %s", index)


async def _clear_new_person_name_fields(
    page: Page,
    session: LiveFormSession | None = None,
    *,
    kind: str = "",
) -> None:
    """Blank First and Last Name once when a new person form is visible.

    Waits until both inputs are on the active form, clears them, then checks
    once more for a late autofill. Skips last-name entry, corrections, and a
    person who already has a saved first name.
    """
    token = _name_reset_token(session, kind or "spouse")
    if session is not None and session.filled.get("name_fields_reset_for") == token:
        return
    if _saved_name_blocks_reset(session, kind or "spouse"):
        return
    try:
        start_screen = await _eligibility_screen(page)
    except Exception:
        start_screen = ""
    found: dict[str, int] | None = {"first": -1, "last": -1}
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        try:
            now = await _eligibility_screen(page)
        except Exception:
            now = start_screen
        if (
            start_screen in ("spouse_form", "child_form", "person_form")
            and now not in (start_screen, "unknown", "")
        ):
            return
        found = await _active_name_input_indexes(page)
        if found is None:
            return
        if found["first"] >= 0 and found["last"] >= 0:
            break
        try:
            await page.wait_for_timeout(200)
        except Exception:
            return
    if found is None or found["first"] < 0 or found["last"] < 0:
        return
    await _blank_named_input_at(page, found["first"])
    await _blank_named_input_at(page, found["last"])
    try:
        await page.wait_for_timeout(400)
    except Exception:
        return
    try:
        now = await _eligibility_screen(page)
    except Exception:
        now = start_screen
    if (
        start_screen in ("spouse_form", "child_form", "person_form")
        and now not in (start_screen, "unknown", "")
    ):
        return
    if session is not None and session.filled.get("name_fields_reset_for") == token:
        return
    again = await _active_name_input_indexes(page)
    if again is not None and again["first"] >= 0 and again["last"] >= 0:
        await _blank_named_input_at(page, again["first"])
        await _blank_named_input_at(page, again["last"])
    if session is not None:
        session.filled["name_fields_reset_for"] = token


async def _strict_fill(page: Page, label: str, value: str) -> None:
    cleaned = _sanitize_fill_value(label, value)
    if not cleaned:
        raise RuntimeError(f"Empty value for {label!r} after normalization")
    loc = await _locate_named_input(page, label)
    if loc is None:
        raise RuntimeError(f"Could not find the {label!r} field on the current screen")
    try:
        # Always replace — never append to an existing value.
        await loc.click(timeout=4000)
        await loc.fill("", timeout=3000)
        await loc.fill(cleaned, timeout=6000)
        actual = (await loc.input_value()).strip()
        if re.search(r"name", label or "", re.I):
            if actual.lower().rstrip(".!?") != cleaned.lower().rstrip(".!?"):
                await loc.fill("", timeout=3000)
                await loc.press_sequentially(cleaned, delay=20)
                actual = (await loc.input_value()).strip()
                if actual.lower().rstrip(".!?") != cleaned.lower().rstrip(".!?"):
                    raise RuntimeError(
                        f"Name field {label!r} shows {actual!r}, expected {cleaned!r}"
                    )
        elif not actual and cleaned:
            raise RuntimeError(f"Field {label!r} remained empty after fill")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not fill the {label!r} field: {exc}") from exc


async def correct_name_field(
    call_id: str,
    field: str,
    value: str,
) -> dict[str, Any]:
    """Edit First/Last Name on the visible person/child form without Back-first.

    Prefer filling the on-screen field. Use Back only when the name field is
    not visible and a verified destination step is reached.
    """
    from .answer_parse import sanitize_person_name
    from .elig_meta import format_question

    session = await get_session(call_id)
    if session is None or session.page is None:
        return {
            "ok": False,
            "error": "no_session",
            "spoken": "I can't edit that name right now because the form is not open.",
        }
    key = (field or "first_name").lower().replace(" ", "_")
    if key in ("name", "first", "firstname"):
        key = "first_name"
    if key in ("last", "lastname"):
        key = "last_name"
    label = "First Name" if key == "first_name" else "Last Name"
    cleaned = sanitize_person_name(value) or (value or "").strip()
    if not cleaned:
        return {
            "ok": False,
            "error": "empty",
            "spoken": f"Of course. What should the {label.lower()} be?",
        }

    async with session.lock:
        page = session.page
        resume_step = session.elig_step or ""
        screen = await _eligibility_screen(page)

        # On an open child/person/spouse form, edit in place — no Back.
        if await _name_field_visible(page, label):
            await _strict_fill(page, label, cleaned)
            verified = await _read_named_input(page, label)
            if verified.lower().rstrip(".!?") != cleaned.lower().rstrip(".!?"):
                return {
                    "ok": False,
                    "error": "verify_failed",
                    "spoken": (
                        f"I couldn't update the {label.lower()} to {cleaned}. "
                        "Please say the name again."
                    ),
                    "step": resume_step,
                }
            # Stay on the current questionnaire step after an in-place edit.
            if screen == "child_form" and not resume_step.startswith("child_"):
                session.elig_step = "child_first" if key == "first_name" else "child_last"
            speech = format_question(session.elig_step or resume_step)
            return {
                "ok": True,
                "step": session.elig_step,
                "spoken": (
                    f"Updated the {label.lower()} to {cleaned}. {speech}"
                ).strip(),
                "value": cleaned,
                "in_place": True,
            }

        # Field not visible: Back only while we remain in the same form family.
        target_step = {
            "first_name": {
                "child_form": "child_first",
                "person_form": "person_first",
                "spouse_form": "spouse_first",
            }.get(screen, "child_first"),
            "last_name": {
                "child_form": "child_last",
                "person_form": "person_last",
                "spouse_form": "spouse_last",
            }.get(screen, "child_last"),
        }[key]

        # If we are already past the name on this form group, one verified Back
        # cycle is allowed — but stop if we leave the form family.
        form_group = screen if screen.endswith("_form") else _step_group(resume_step)
        for _ in range(3):
            back = page.get_by_role("button", name=re.compile(r"^\s*Back\s*$", re.I)).first
            try:
                if not await back.count() or not await back.is_visible():
                    break
                before = await _screen_signature(page)
                await back.click(timeout=5000)
                for _wait in range(20):
                    await page.wait_for_timeout(200)
                    if await _screen_signature(page) != before:
                        break
            except Exception:
                break
            landed = await _eligibility_screen(page)
            if form_group.endswith("_form") and landed != form_group:
                # Left the form — restore step from landed screen and abort edit.
                recovered = _screen_first_step(landed, session)
                if recovered:
                    session.elig_step = recovered
                return {
                    "ok": False,
                    "error": "left_form",
                    "spoken": (
                        "I couldn't reach that name field without leaving this form. "
                        + format_question(session.elig_step)
                    ),
                    "step": session.elig_step,
                }
            if await _name_field_visible(page, label):
                session.elig_step = target_step
                await _strict_fill(page, label, cleaned)
                verified = await _read_named_input(page, label)
                if verified.lower().rstrip(".!?") != cleaned.lower().rstrip(".!?"):
                    return {
                        "ok": False,
                        "error": "verify_failed",
                        "spoken": (
                            f"I couldn't update the {label.lower()} to {cleaned}. "
                            "Please say the name again."
                        ),
                        "step": session.elig_step,
                    }
                # After correcting first/last, resume at the next logical name/rel step.
                if key == "first_name":
                    next_step = {
                        "child_first": "child_last",
                        "person_first": "person_last",
                        "spouse_first": "spouse_last",
                    }.get(target_step, resume_step or target_step)
                else:
                    next_step = {
                        "child_last": "child_rel",
                        "person_last": "person_rel",
                        "spouse_last": "spouse_role",
                    }.get(target_step, resume_step or target_step)
                session.elig_step = next_step
                return {
                    "ok": True,
                    "step": next_step,
                    "spoken": (
                        f"Updated the {label.lower()} to {cleaned}. "
                        + format_question(next_step)
                    ),
                    "value": cleaned,
                }

        return {
            "ok": False,
            "error": "not_found",
            "spoken": (
                f"I can't see the {label.lower()} field on this screen. "
                + format_question(session.elig_step or resume_step)
            ),
            "step": session.elig_step,
        }


async def correct_member_field(
    call_id: str,
    field: str,
    value: str,
) -> dict[str, Any]:
    """Correct age/relationship/name on the open member form without Back.

    Always stays on the current spouse/child/person form. After a successful
    edit, returns to the confirm summary for that member.
    """
    from .answer_parse import sanitize_person_name
    from .elig_meta import format_question

    session = await get_session(call_id)
    if session is None or session.page is None:
        return {
            "ok": False,
            "error": "no_session",
            "spoken": "I can't edit that detail right now because the form is not open.",
        }
    key = (field or "").lower().replace(" ", "_")
    if key in ("name", "first", "firstname"):
        key = "first_name"
    if key in ("last", "lastname"):
        key = "last_name"
    if key in ("role", "rel"):
        key = "relationship"
    if key not in ("first_name", "last_name", "age", "relationship"):
        return {
            "ok": False,
            "error": "unknown_field",
            "spoken": _confirm_which_field_speech(),
        }

    async with session.lock:
        page = session.page
        screen = await _eligibility_screen(page)
        if screen not in ("child_form", "spouse_form", "person_form"):
            return {
                "ok": False,
                "error": "wrong_screen",
                "spoken": (
                    "That detail is not on this screen. "
                    + format_question(session.elig_step or "marital")
                ),
                "step": session.elig_step,
            }
        kind = {
            "child_form": "child",
            "spouse_form": "spouse",
            "person_form": "person",
        }[screen]
        confirm_step = _confirm_step_for_kind(kind)
        # Keep instance id stable while editing an existing child.
        if kind == "child":
            _ensure_child_instance(session)

        if key in ("first_name", "last_name"):
            label = "First Name" if key == "first_name" else "Last Name"
            cleaned = sanitize_person_name(value) or (value or "").strip()
            if not cleaned:
                return {
                    "ok": False,
                    "error": "empty",
                    "spoken": f"Of course. What should the {label.lower()} be?",
                }
            if not await _name_field_visible(page, label):
                return {
                    "ok": False,
                    "error": "not_visible",
                    "spoken": (
                        f"I can't see the {label.lower()} field on this form. "
                        "Please say the corrected value again."
                    ),
                    "step": session.elig_step,
                }
            await _strict_fill(page, label, cleaned)
            verified = await _read_named_input(page, label)
            if verified.lower().rstrip(".!?") != cleaned.lower().rstrip(".!?"):
                return {
                    "ok": False,
                    "error": "verify_failed",
                    "spoken": (
                        f"I couldn't update the {label.lower()} to {cleaned}. "
                        "Please say the name again."
                    ),
                    "step": session.elig_step,
                }
            filled_key = {
                ("child", "first_name"): "child_first",
                ("child", "last_name"): "child_last",
                ("spouse", "first_name"): "spouse_first",
                ("spouse", "last_name"): "spouse_last",
                ("person", "first_name"): "person_first",
                ("person", "last_name"): "person_last",
            }[(kind, key)]
            session.filled[filled_key] = cleaned
            if key == "first_name":
                _remember_household_name(session, cleaned)
            session.elig_step = confirm_step
            return {
                "ok": True,
                "step": confirm_step,
                "spoken": (
                    f"Updated the {label.lower()} to {cleaned}. "
                    + _member_confirm_speech(session, kind)
                ),
                "value": cleaned,
                "in_place": True,
                "field": key,
            }

        if key == "relationship":
            labels = await _visible_choice_labels(page)
            if kind == "spouse":
                lowered = (value or "").lower()
                mapped = "Partner" if "partner" in lowered else "Spouse"
                if "spouse" not in lowered and "partner" not in lowered:
                    # Prefer exact chip match when available.
                    mapped_rel = _map_relationship_label(value, labels)
                    mapped = mapped_rel or mapped
                try:
                    await _ensure_relationship_choice_selected(page, mapped)
                except RuntimeError as exc:
                    detail = str(exc)
                    if detail.startswith("REL_CLARIFY:"):
                        detail = detail.split(":", 1)[1].strip()
                    return {
                        "ok": False,
                        "error": "rel_not_selected",
                        "spoken": detail
                        or (
                            f"I couldn't select {mapped} on the form. "
                            "Please say the relationship again."
                        ),
                        "step": session.elig_step,
                    }
                session.filled["spouse_role"] = mapped
            else:
                rel_key = "child_rel" if kind == "child" else "person_rel"
                if session.filled.get("rel_other_awaiting") == rel_key or (
                    (session.filled.get(rel_key) or "").lower().rstrip(" *") == "other"
                    and await _other_relationship_input_visible(page)
                    and not _map_relationship_label(value, labels)
                ):
                    custom = await _fill_other_relationship_text(page, value)
                    session.filled[rel_key] = "Other"
                    session.filled[f"{rel_key}_other"] = custom
                    session.filled.pop("rel_other_awaiting", None)
                    mapped = custom
                else:
                    mapped = _map_relationship_label(value, labels)
                    if not mapped:
                        opts = _relationship_option_labels(labels)
                        return {
                            "ok": False,
                            "error": "rel_clarify",
                            "spoken": build_options_help_speech(rel_key, opts or labels),
                            "step": session.elig_step,
                        }
                    try:
                        await _ensure_relationship_choice_selected(page, mapped)
                    except RuntimeError as exc:
                        detail = str(exc)
                        if detail.startswith("REL_CLARIFY:"):
                            detail = detail.split(":", 1)[1].strip()
                        return {
                            "ok": False,
                            "error": "rel_not_selected",
                            "spoken": detail
                            or (
                                f"I couldn't select {mapped} on the form. "
                                "Please say the relationship again."
                            ),
                            "step": session.elig_step,
                        }
                    session.filled[rel_key] = mapped
                    if mapped.lower().rstrip(" *") == "other":
                        try:
                            await page.wait_for_timeout(500)
                        except Exception:
                            pass
                        if await _other_relationship_input_visible(page):
                            session.filled["rel_other_awaiting"] = rel_key
                            session.elig_step = rel_key
                            return {
                                "ok": True,
                                "step": rel_key,
                                "spoken": "What is the other relationship?",
                                "value": mapped,
                                "in_place": True,
                                "field": key,
                                "await_other_rel": True,
                            }
            session.elig_step = confirm_step
            return {
                "ok": True,
                "step": confirm_step,
                "spoken": (
                    f"Updated the relationship to {mapped}. "
                    + _member_confirm_speech(session, kind)
                ),
                "value": mapped,
                "in_place": True,
                "field": key,
            }

        # Age — require a real age/range; never Back off this form.
        age_labels = await _ensure_age_choices_visible(page)
        if not age_labels:
            return {
                "ok": False,
                "error": "no_age_chips",
                "spoken": (
                    "I can't see the age choices on this form yet. "
                    "Please say the age again in a moment."
                ),
                "step": session.elig_step,
            }
        if _is_relationship_only_age_utterance(value):
            return {
                "ok": False,
                "error": "not_an_age",
                "spoken": (
                    "AGE_CLARIFY: That sounds like a relationship, not an age. "
                    + _age_clarify_message(
                        "child_age" if kind == "child" else "spouse_age",
                        age_labels,
                    )
                ),
                "step": session.elig_step,
            }
        mapped = _map_age_label(value, age_labels)
        if not mapped:
            return {
                "ok": False,
                "error": "age_clarify",
                "spoken": "AGE_CLARIFY: " + _age_clarify_message(
                    "child_age" if kind == "child" else "spouse_age",
                    age_labels,
                ),
                "step": session.elig_step,
            }
        await _click_age_choice(page, mapped)
        filled_age = {
            "child": "child_age",
            "spouse": "spouse_age",
            "person": "person_age",
        }[kind]
        session.filled[filled_age] = mapped
        session.elig_step = confirm_step
        return {
            "ok": True,
            "step": confirm_step,
            "spoken": (
                f"Updated the age to {_spoken_age_range(mapped)}. "
                + _member_confirm_speech(session, kind)
            ),
            "value": mapped,
            "in_place": True,
            "field": key,
        }


async def _visible_choice_labels(page: Page) -> list[str]:
    try:
        labels = await page.evaluate(
            """() => {
              const nodes = [...document.querySelectorAll(
                'button, [role=button], [role=radio], [role=option], [role=listitem], a, label, li, div[tabindex], div[class*="choice"], div[class*="option"], span[class*="choice"]'
              )];
              const texts = nodes
                .filter((node) => {
                  const r = node.getBoundingClientRect();
                  return r.width > 2 && r.height > 2;
                })
                .map((node) => (node.innerText || node.textContent || '')
                  .replace(/\\s+/g, ' ').trim())
                .filter((text) => text
                  && text.length < 140
                  && !/^(exit|back|continue|next|logout|log out|done|add another)$/i.test(text));
              // Income chips are often plain clickable cards with "$" text only.
              const dollar = [...document.querySelectorAll('button, [role=button], div, span, li, a, label')]
                .filter((node) => {
                  const r = node.getBoundingClientRect();
                  if (r.width < 2 || r.height < 2) return false;
                  const t = (node.innerText || '').replace(/\\s+/g, ' ').trim();
                  return /^\\$?\\s*\\d/.test(t) && t.length < 60 && t.length > 2
                    && node.children.length <= 3;
                })
                .map((node) => (node.innerText || '').replace(/\\s+/g, ' ').trim());
              return [...texts, ...dollar];
            }"""
        )
    except Exception:
        return []
    out: list[str] = []
    for item in labels or []:
        if item not in out:
            out.append(item)
    return out


def _normalize_income_period_text(spoken: str) -> str:
    """Strip punctuation and collapse whitespace for income-period matching."""
    text = (spoken or "").strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _income_period_choice(spoken: str) -> str:
    """Map spoken income period onto the canonical Pointcare tab label.

    Important: do not use ``\\bbi[\\-\\s]?week\\b`` — that fails on \"bi-weekly\"
    because there is no word boundary between \"week\" and \"ly\".
    """
    text = _normalize_income_period_text(spoken)
    if not text:
        return spoken
    compact = re.sub(r"[\s\-]+", "", text)
    # bi-weekly / biweekly / bi weekly / every two weeks / STT by|buy weekly
    if (
        "biweekly" in compact
        or "biweek" == compact
        or compact in {"byweekly", "buyweekly", "byweek", "buyweek"}
        or re.search(
            r"\b(?:bi\s*week(?:ly|s)?|by\s*week(?:ly)?|buy\s*week(?:ly)?|"
            r"every\s*(?:two|2)\s*weeks?|(?:once\s+)?(?:every\s+)?two\s*weeks?|"
            r"fortnight(?:ly)?)\b",
            text,
        )
    ):
        return "Bi-weekly"
    if re.search(r"\bmonth", text):
        return "Monthly"
    if re.search(r"\b(year|annual)", text):
        return "Annual"
    return spoken


def _looks_like_income_period_label(label: str) -> bool:
    """True for a single income-period chip — not concatenated tab strips."""
    low = re.sub(r"\s+", " ", (label or "").strip().lower())
    if not low or len(low) > 24:
        return False
    if low in {"annual", "monthly", "bi-weekly", "biweekly", "every two weeks"}:
        return True
    return bool(
        re.fullmatch(
            r"(annual|monthly|bi[\s\-]*week(?:ly)?|every\s*(?:two|2)\s*weeks?)",
            low,
        )
    )


def _resolve_income_period_label(
    spoken: str, labels: list[str] | None = None
) -> str:
    """Map spoken period onto an exact visible Pointcare option (no first-option fallback)."""
    period = _income_period_choice(spoken)
    if period not in {"Annual", "Monthly", "Bi-weekly"}:
        return ""
    pool = [item for item in (labels or []) if (item or "").strip()]
    if not pool:
        return period
    # Prefer an exact visible label that resolves to the same period.
    for label in pool:
        if not _looks_like_income_period_label(label):
            continue
        if _income_period_choice(label) == period or label.lower().strip() == period.lower():
            return label
    period_like = [label for label in pool if _looks_like_income_period_label(label)]
    # Period chips are visible but none matched the spoken answer — clarify, don't guess.
    if period_like:
        return ""
    # Unrelated labels on screen (e.g. marital) — do not force an income period.
    return ""


_NUMBER_WORDS = {
    "zero": 0,
    "oh": 0,
    "a": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _spoken_amount_to_int(raw: str) -> int | None:
    """Parse amounts like '12000', 'twelve thousand', '40 thousand', '40k', '1 million'."""
    text = (raw or "").lower().replace(",", "").replace("$", "").strip()
    text = re.sub(r"[^\w\s.]+$", "", text).strip()
    if not text:
        return None
    # "40k" / "50 k" / "100k." (punctuation already stripped)
    k_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*k", text)
    if k_match:
        return int(float(k_match.group(1)) * 1000)
    # Embedded "100k" inside a longer residual phrase.
    embedded_k = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text)
    if embedded_k and not re.search(r"[a-z]", text.replace(embedded_k.group(0), "")):
        return int(float(embedded_k.group(1)) * 1000)
    # "40 thousand" / "50 hundred" / "1 million" with leading digits + scale word.
    digit_scale = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(thousand|hundred|million|k)\b",
        text,
    )
    if digit_scale:
        base = float(digit_scale.group(1))
        scale = digit_scale.group(2)
        if scale == "million":
            mult = 1_000_000
        elif scale in {"thousand", "k"}:
            mult = 1000
        else:
            mult = 100
        return int(base * mult)
    digits = re.search(r"(\d+)\s*\+?", text)
    letterish = re.sub(r"\b(dollars?|and)\b", "", text)
    if digits and not re.search(r"[a-z]", letterish):
        return int(digits.group(1))
    # "twelve thousand", "thirty two thousand seven hundred", "one million", "100k"
    if (
        "thousand" in text
        or "million" in text
        or re.search(r"\d\s*k\b|\bk\b", text)
        or any(w in text for w in _NUMBER_WORDS)
    ):
        total = 0
        current = 0
        # Include digit tokens so "40 thousand" works inside longer phrases.
        tokens = re.findall(r"\d+|[a-z]+", text)
        for word in tokens:
            if word in {"dollar", "dollars", "and", "to", "through", "thru", "between"}:
                continue
            if word.isdigit():
                current += int(word)
            elif word in {"thousand", "k"}:
                total += max(current, 1) * 1000
                current = 0
            elif word == "million":
                total += max(current, 1) * 1_000_000
                current = 0
            elif word == "hundred":
                current = max(current, 1) * 100
            elif word in _NUMBER_WORDS:
                current += _NUMBER_WORDS[word]
        total += current
        if total or re.search(r"\bzero\b", text):
            return total
        return None
    if digits:
        return int(digits.group(1))
    return None


def _strip_income_period_filler(spoken: str) -> str:
    """Remove period / pay-frequency words so they do not inflate amount parsing."""
    text = (spoken or "").strip().lower()
    text = re.sub(
        r"\b("
        r"every\s*(?:two|2)\s*weeks?|bi[\s\-]?weekly|fortnight(?:ly)?|"
        r"per\s*(?:year|month|week)|a\s*(?:year|month|week)|"
        r"annually|annual|yearly|monthly|weekly|"
        r"before\s*taxes|after\s*taxes|pre[\s\-]?tax|"
        r"household\s*income|income|salary|earn(?:s|ed|ing)?"
        r")\b",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def _income_amount_scale_unclear(spoken: str) -> bool:
    """True when a bare number needs dollars-vs-thousands clarification.

    Never guess that \"100\" means $100 or $100,000.
    """
    text = _strip_income_period_filler(spoken or "")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("$", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False
    # Explicit dollars without a thousands scale → treat as literal dollars.
    if re.search(r"\bdollars?\b", text, re.I) and not re.search(
        r"\b(thousand|million)\b|\d\s*k\b|\bk\b",
        text,
        re.I,
    ):
        return False
    # thousand / million / k make the scale clear. Bare "hundred" does not
    # (one hundred vs one hundred thousand).
    if re.search(r"\b(thousand|million)\b|\d\s*k\b|\bk\b", text, re.I):
        return False
    if re.search(r"\b\d{4,}\b", text):
        return False
    lo, hi = _parse_spoken_income_interval(text)
    if lo is None:
        return False
    if hi is not None and hi != lo:
        return lo < 1000 and hi < 1000
    return lo < 1000


def _income_period_phrase(period: str) -> str:
    key = _income_period_choice(period or "")
    if key == "Monthly":
        return "per month"
    if key == "Bi-weekly":
        return "every two weeks"
    return "per year"


def _income_scale_clarify_speech(spoken: str, period: str = "Annual") -> str:
    """Ask dollars vs thousands without echoing the utterance or listing bands."""
    text = (spoken or "").strip()
    nums = re.findall(r"\d+", re.sub(r"(?<=\d),(?=\d)", "", text))
    spoken_num = nums[0] if nums else ""
    word = {
        "40": "forty",
        "50": "fifty",
        "100": "one hundred",
        "200": "two hundred",
    }.get(spoken_num, spoken_num or "that amount")
    per = _income_period_phrase(period)
    return (
        f"Do you mean about {word} dollars, or about {word} thousand dollars "
        f"{per}?"
    )


def _income_range_natural_clarify(
    spoken: str = "",
    *,
    period: str = "Annual",
) -> str:
    """Voice-friendly clarify when the amount does not uniquely match a band.

    Never echo the caller's full utterance or enumerate every income chip.
    """
    del spoken  # Kept for call-site compatibility; do not speak it back.
    per = _income_period_phrase(period)
    return (
        f"I couldn't match that to a single income amount {per}. "
        f"Please say an approximate amount {per}, like forty thousand, "
        f"or a narrower range like forty to fifty thousand."
    )


_INCOME_PERIOD_SWITCH_RE = re.compile(
    r"\b("
    r"change\s+(?:the\s+)?(?:period|income\s+period|pay\s+period|frequency)|"
    r"(?:go\s+back|switch|choose|select|use|pick)\s+"
    r"(?:to\s+)?(?:the\s+)?(?:annual|yearly|monthly|bi[\s\-]?weekly)|"
    r"(?:i\s+(?:want|need)\s+to\s+)?(?:choose|select|use|pick)\s+"
    r"(?:annual|yearly|monthly|bi[\s\-]?weekly)|"
    r"(?:annual|yearly|monthly|bi[\s\-]?weekly)\s+instead|"
    r"change\s+(?:it\s+)?to\s+(?:annual|yearly|monthly|bi[\s\-]?weekly)|"
    r"(?:make\s+it|set\s+(?:it\s+)?to)\s+(?:annual|yearly|monthly|bi[\s\-]?weekly)|"
    r"go\s+back\s+to\s+(?:annual|monthly|bi[\s\-]?weekly|the\s+period)"
    r")\b",
    re.I,
)


def _detect_income_period_switch(spoken: str) -> str:
    """Return Annual/Monthly/Bi-weekly when utterance requests a period change."""
    text = (spoken or "").strip()
    if not text or not _INCOME_PERIOD_SWITCH_RE.search(text):
        return ""
    period = _income_period_choice(text)
    if period in {"Annual", "Monthly", "Bi-weekly"}:
        return period
    # "go back to the period" without naming one.
    if re.search(r"\b(period|annual|monthly|bi[\s\-]?weekly)\b", text, re.I):
        return ""
    return ""


def _wants_income_period_reselect(spoken: str) -> bool:
    """True when caller wants the period chooser again (no specific period yet)."""
    text = (spoken or "").strip().lower()
    if not text:
        return False
    if _detect_income_period_switch(text):
        return False
    return bool(
        re.search(
            r"\b("
            r"go\s+back\s+to\s+(?:the\s+)?(?:period|income\s+period)|"
            r"change\s+(?:the\s+)?(?:period|income\s+period|pay\s+period)|"
            r"(?:pick|choose)\s+(?:the\s+)?period\s+again|"
            r"annual\s+or\s+monthly|"
            r"monthly\s+or\s+annual"
            r")\b",
            text,
            re.I,
        )
    )


def _parse_range_bounds(label: str) -> tuple[int | None, int | None]:
    text = (label or "").replace(",", "").replace("–", "-").replace("—", "-")
    text = text.replace("to", "-")
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if not nums:
        return None, None
    if "+" in (label or "") or "plus" in (label or "").lower():
        return nums[0], None
    if len(nums) == 1:
        return nums[0], nums[0]
    return nums[0], nums[1]


def _parse_spoken_income_interval(
    spoken: str,
) -> tuple[int | None, int | None]:
    """Normalize a spoken income amount or range into (low, high).

    Handles shared scale: "forty to fifty thousand" → (40000, 50000),
    "40k to 50k", "between 40 and 50 thousand", digits with commas, etc.
    Returns (None, None) when nothing numeric can be recovered.
    """
    text = _strip_income_period_filler(spoken or "")
    # Strip thousand-separator commas so "40,000" → "40000".
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("$", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None, None

    # "between 40 and 50 thousand" / "between forty and fifty thousand"
    between = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+(thousand|hundred|k))?\s*$",
        text,
        re.I,
    )
    trailing_scale = ""
    if between:
        left, right = between.group(1).strip(), between.group(2).strip()
        trailing_scale = (between.group(3) or "").lower()
    else:
        parts = re.split(
            r"\s*(?:\bto\b|\bthrough\b|\bthru\b|\buntil\b|-|–|—)\s*",
            text,
            maxsplit=1,
        )
        left = (parts[0] or "").strip()
        right = (parts[1] or "").strip() if len(parts) > 1 else ""

    # Strip leading filler on the left side.
    left = re.sub(
        r"^(?:from|about|around|approximately|like|salary|income|range|between)\s+",
        "",
        left,
    ).strip()

    # Apply a trailing shared scale onto both sides before parsing.
    if trailing_scale:
        if trailing_scale and not re.search(
            r"\b(thousand|hundred|k)\b", left, re.I
        ):
            left = f"{left} {trailing_scale}"
        if trailing_scale and not re.search(
            r"\b(thousand|hundred|k)\b", right, re.I
        ):
            right = f"{right} {trailing_scale}"

    low = _spoken_amount_to_int(left) if left else None
    high = _spoken_amount_to_int(right) if right else None

    # Shared "thousand"/"k" scale: "forty to fifty thousand" / "40 to 50k".
    if low is not None and high is not None and low < 1000:
        right_has_scale = bool(
            re.search(r"\b(thousand|k)\b", right or "", re.I)
        )
        left_has_scale = bool(
            re.search(r"\b(thousand|k)\b", left or "", re.I)
        )
        if right_has_scale and not left_has_scale:
            low = low * 1000
        elif (
            not left_has_scale
            and not right_has_scale
            and high < 1000
            and re.search(r"\b(thousand|k)\b", text)
        ):
            low, high = low * 1000, high * 1000

    if low is not None and high is None and not right:
        return low, low
    if low is not None and high is not None and low > high:
        low, high = high, low
    return low, high


def _income_range_labels(labels: list[str]) -> list[str]:
    return [
        item
        for item in labels
        if item and ("$" in item or re.search(r"\d", item))
    ]


def _interval_contains(
    lo: int | None,
    hi: int | None,
    spoken_lo: int,
    spoken_hi: int,
) -> bool:
    """True when the chip fully contains the spoken closed interval.

    Open-ended chips (hi is None) only count as containment when the spoken
    low is at/above the chip floor *and* the spoken span is not wildly larger
    than a typical band (avoids mapping 90k–100k onto $39,901+ just because
    both sit above the floor). Prefer open-ended only for single amounts or
    when spoken_lo is near the open floor.
    """
    if lo is None:
        return False
    if spoken_lo < lo:
        return False
    if hi is None:
        # Open-ended: contain point amounts, or ranges that start near the floor
        # (within 25% of the floor or $5k), not arbitrary higher bands.
        if spoken_lo == spoken_hi:
            return spoken_lo >= lo
        near_floor = abs(spoken_lo - lo) <= max(5000, int(lo * 0.25))
        return near_floor and spoken_hi >= lo
    return spoken_hi <= hi


def _interval_overlap(
    lo: int | None,
    hi: int | None,
    spoken_lo: int,
    spoken_hi: int,
) -> int:
    """Length of overlap between chip and spoken interval (0 if none)."""
    if lo is None:
        return 0
    if hi is None:
        # Open-ended chips: only score overlap when the spoken low is near the
        # floor; otherwise a $39,901+ chip would dominate every high range.
        if spoken_hi < lo:
            return 0
        near_floor = abs(spoken_lo - lo) <= max(5000, int(lo * 0.25))
        if not near_floor and spoken_lo != spoken_hi:
            return 0
        if spoken_lo < lo:
            return 0
        return max(0, spoken_hi - spoken_lo)
    start = max(lo, spoken_lo)
    end = min(hi, spoken_hi)
    return max(0, end - start)


def _map_income_range_label(spoken: str, labels: list[str]) -> str:
    """Map spoken income onto a unique visible $ range chip.

    Matching rule (documented):
    1. Prefer a single chip that fully contains the spoken closed interval.
    2. If several contain it, pick the uniquely tightest span (else clarify).
    3. Else allow a uniquely dominant overlap (≥60% of spoken span and ≥2×
       the next best). Open-ended chips only win for point amounts or ranges
       that start near the open floor.
    4. Never fall back to the first/nearest chip on a weak score.

    Returns "" when no unique, semantically valid match exists, or when the
    amount scale is ambiguous (bare \"100\").
    """
    range_labels = _income_range_labels(labels)
    if not range_labels:
        return ""
    text = (spoken or "").strip().lower()
    if not text:
        return ""
    # Period-change / navigation must not be treated as dollar amounts.
    if _detect_income_period_switch(text) or _wants_income_period_reselect(text):
        return ""
    if _income_amount_scale_unclear(text):
        return ""

    # Explicit ordinal only when the caller asked for first/last option.
    if re.search(r"\b(first|lowest|smallest|option one|option 1)\b", text):
        return range_labels[0]
    if re.search(r"\b(last|highest|largest|top)\b", text):
        return range_labels[-1]

    spoken_lo, spoken_hi = _parse_spoken_income_interval(text)
    if spoken_lo is None and spoken_hi is None:
        return ""

    # Point amount → unique containing chip.
    if spoken_lo is not None and (
        spoken_hi is None or spoken_hi == spoken_lo
    ):
        amount = spoken_lo
        hits = []
        for label in range_labels:
            lo, hi = _parse_range_bounds(label)
            if lo is None:
                continue
            if hi is None and amount >= lo:
                hits.append(label)
            elif hi is not None and lo <= amount <= hi:
                hits.append(label)
        if len(hits) == 1:
            return hits[0]
        return ""

    assert spoken_lo is not None and spoken_hi is not None
    # Prefer unique full containment of the spoken interval.
    contained = [
        label
        for label in range_labels
        if _interval_contains(
            *_parse_range_bounds(label), spoken_lo, spoken_hi
        )
    ]
    if len(contained) == 1:
        return contained[0]
    if len(contained) > 1:
        # Prefer the tightest containing band.
        def span(label: str) -> int:
            lo, hi = _parse_range_bounds(label)
            if lo is None:
                return 10**12
            if hi is None:
                return 10**12
            return hi - lo

        contained.sort(key=span)
        if span(contained[0]) < span(contained[1]):
            return contained[0]
        return ""

    # No full containment: unique dominant overlap only when clearly one winner.
    overlaps: list[tuple[int, str]] = []
    spoken_span = max(1, spoken_hi - spoken_lo)
    for label in range_labels:
        lo, hi = _parse_range_bounds(label)
        ov = _interval_overlap(lo, hi, spoken_lo, spoken_hi)
        if ov > 0:
            overlaps.append((ov, label))
    if not overlaps:
        return ""
    overlaps.sort(reverse=True)
    best_ov, best_label = overlaps[0]
    second_ov = overlaps[1][0] if len(overlaps) > 1 else 0
    # Require majority overlap of the spoken range and a clear margin.
    if best_ov >= int(spoken_span * 0.6) and best_ov >= second_ov * 2:
        return best_label
    return ""


_UI_FIELD_LABELS = frozenset(
    {
        "first name",
        "last name",
        "living with me",
        "same mailing address",
        "relationship to me",
        "date of birth (optional)",
        "number of babies",
        "due date",
    }
)


def _labels_look_like_age(labels: list[str]) -> bool:
    return any(_looks_like_age_label(item) for item in labels)


def _looks_like_age_label(label: str) -> bool:
    """True only for real age-range chips — not relationship or form captions.

    Relationship buttons like "Child" / "Spouse" and the bare "Age" field
    caption must never be treated as selectable age options.
    """
    text = (label or "").strip()
    if not text or len(text) > 80:
        return False
    key = text.lower().rstrip(" *")
    if key in _UI_FIELD_LABELS or key in {
        "age",
        "spouse",
        "partner",
        "child",
        "stepchild",
        "grandchild",
        "foster child",
        "other",
        "relationship to me",
        "enter date of birth instead",
    }:
        return False
    # Pointcare age chips always expose a numeric band in parentheses.
    if re.search(r"\(\s*\d{1,2}\s*[-–—]\s*\d{1,2}\s*\)", text):
        return True
    if re.search(r"\(\s*\d{1,2}\+?\s*\)", text):
        return True
    # Rare chips without parens still name an age band with digits.
    if re.search(
        r"\b(baby|toddler|infant|teen|young adult|turning senior|senior|adult)\b",
        text,
        re.I,
    ) and re.search(r"\d", text):
        return True
    return False


def _age_option_labels(labels: list[str] | None) -> list[str]:
    """Filter scraped UI text down to visible age-range chips only."""
    out: list[str] = []
    for item in labels or []:
        if _looks_like_age_label(item) and item not in out:
            out.append(item)
    return out


def _labels_look_like_income(labels: list[str]) -> bool:
    return any("$" in (item or "") for item in labels)


def _choice_labels_for_matching(labels: list[str]) -> list[str]:
    """Drop static form field captions that must never be treated as answers."""
    out: list[str] = []
    for label in labels:
        key = (label or "").lower().rstrip(" *")
        if not key or key in _UI_FIELD_LABELS:
            continue
        out.append(label)
    return out


def _fuzzy_choice_label(spoken: str, labels: list[str]) -> str:
    """Map ASR / spoken text onto a visible Pointcare button label."""
    labels = _choice_labels_for_matching(labels)
    # Income period tabs before generic phrase mapping.
    period_label = _resolve_income_period_label(spoken, labels)
    if period_label:
        return period_label
    # Income dollar ranges only — never treat age chips as income bands.
    if _labels_look_like_income(labels) or (
        not _labels_look_like_age(labels)
        and any(re.search(r"\d{3,}", label or "") for label in labels)
    ):
        mapped = _map_income_range_label(spoken, labels)
        if mapped:
            for label in labels:
                if label.lower() == mapped.lower():
                    return label
            return mapped
        # No unique income match — do not fuzzy onto the first dollar chip.
        return spoken
    # Age chips: require validated age mapping (no loose fuzzy to Baby/etc.).
    if _labels_look_like_age(labels):
        mapped = _map_age_label(spoken, labels)
        if mapped:
            return mapped
        return spoken
    wanted = _choice_phrase(spoken)
    if not labels:
        return wanted
    lowered = wanted.lower().strip()
    # Never treat form captions or "last name is …" phrases as choice labels.
    if lowered in _UI_FIELD_LABELS or re.search(
        r"\b(first|last)\s+name\s+is\b", lowered
    ):
        return wanted
    for label in labels:
        if label.lower() == lowered:
            return label
    # Exact yes/no must not fuzzy-match onto longer labels like "Never Married".
    if lowered in {"yes", "no"}:
        return wanted
    import difflib

    close = difflib.get_close_matches(
        lowered,
        [label.lower() for label in labels],
        n=1,
        cutoff=0.72,
    )
    if close:
        for label in labels:
            if label.lower() == close[0]:
                return label
    # Shared tokens for ASR near-misses ("Never matter" → "Never Married").
    spoken_words = set(re.findall(r"[a-z0-9]{3,}", lowered))
    best = ""
    best_hits = 0
    for label in labels:
        label_words = set(re.findall(r"[a-z0-9]{3,}", label.lower()))
        hits = len(spoken_words & label_words)
        if "never" in spoken_words and "never" in label_words:
            if spoken_words & {"matter", "marry", "married", "merit", "murder", "marriage"}:
                hits = max(hits, 2)
        if hits > best_hits:
            best = label
            best_hits = hits
    if best and best_hits >= 2:
        return best
    return wanted


def _map_support_yes_label(spoken: str, labels: list[str]) -> str:
    """Map a Yes answer onto one of the four half-support Pointcare buttons."""
    text = (spoken or "").lower()
    yes_labels = [
        item
        for item in labels
        if item.lower().startswith("yes") and item.lower().strip() != "yes"
    ]
    if not yes_labels:
        yes_labels = [item for item in labels if item.lower().startswith("yes")]

    def pick(*needles: str) -> str:
        for label in yes_labels:
            lowered = label.lower()
            if all(needle in lowered for needle in needles):
                return label
        return ""

    if re.search(r"\b(student|parent|guardian|school)\b", text):
        return pick("student") or pick("parent")
    if re.search(r"\bdisabled\b", text):
        return pick("disabled")
    if re.search(r"\brelative\b", text):
        return pick("relative")
    if re.search(r"\b(live|living|half the year|with (them|him|her|the person))\b", text):
        # Prefer the non-relative "person that supports me" option.
        person = pick("person that supports")
        if person and "relative" not in person.lower():
            return person
        return person or pick("half the year")
    # Bare Yes with exactly one visible Yes option.
    if len(yes_labels) == 1:
        return yes_labels[0]
    return ""


async def _click_support_choice(
    page: Page,
    answer: str,
    yes: bool,
    labels: list[str] | None = None,
) -> None:
    visible = labels if labels is not None else await _visible_choice_labels(page)
    if not yes:
        await _click_exact_choice(page, "No")
        return
    mapped = _map_support_yes_label(answer, visible)
    if not mapped:
        raise RuntimeError(
            "SUPPORT_CLARIFY: Please say which situation fits: student supported "
            "by a parent, you live with the person who supports you, disabled and "
            "live with them year-round, or a relative supports you."
        )
    await _click_exact_choice(page, mapped)


async def _ensure_age_choices_visible(page: Page) -> list[str]:
    """Open the Age control when needed and return visible age-range chips only."""
    age_labels = _age_option_labels(await _visible_choice_labels(page))
    if age_labels:
        return age_labels
    # Spouse/person forms keep chips behind an "Age" control.
    for opener in (
        page.get_by_role("button", name=re.compile(r"^\s*Age\s*$", re.I)).first,
        page.get_by_text(re.compile(r"^\s*Age\s*$", re.I)).first,
        page.locator("label").filter(has_text=re.compile(r"^\s*Age\s*$", re.I)).first,
    ):
        try:
            if await opener.count() and await opener.is_visible():
                await opener.click(timeout=4000)
                deadline = time.monotonic() + 6
                while time.monotonic() < deadline:
                    age_labels = _age_option_labels(await _visible_choice_labels(page))
                    if age_labels:
                        return age_labels
                    await page.wait_for_timeout(200)
                break
        except Exception:
            continue
    return _age_option_labels(await _visible_choice_labels(page))


async def _click_age_choice(page: Page, mapped: str) -> None:
    """Click a verified age chip. Never falls back to generic/income digit matching."""
    if not mapped or not _looks_like_age_label(mapped):
        raise RuntimeError("AGE_CLARIFY: Please say a valid age range from the form.")
    age_labels = await _ensure_age_choices_visible(page)
    if mapped not in age_labels:
        # Case-insensitive reconcile against freshly opened chips.
        lowered = mapped.lower()
        hit = next((item for item in age_labels if item.lower() == lowered), "")
        if not hit:
            raise RuntimeError(
                "AGE_CLARIFY: " + _age_clarify_message("spouse_age", age_labels)
            )
        mapped = hit
    pattern = re.compile(rf"^\s*{re.escape(mapped)}\s*$", re.I)
    before = await _choice_signature(page, mapped)
    candidates = [
        page.get_by_role("radio", name=pattern).first,
        page.get_by_role("button", name=pattern).first,
        page.get_by_text(pattern, exact=True).first,
    ]
    for loc in candidates:
        try:
            if await loc.count() and await loc.is_visible():
                await loc.click(timeout=6000)
                for _ in range(20):
                    await page.wait_for_timeout(150)
                    if await _choice_signature(page, mapped) != before:
                        return
                # Selection may not expose aria state; confirm chip still listed.
                still = _age_option_labels(await _visible_choice_labels(page))
                if mapped in still or any(item.lower() == mapped.lower() for item in still):
                    return
                return
        except Exception:
            continue
    raise RuntimeError(
        f"AGE_CLARIFY: Could not select age option {mapped!r}. "
        + _age_clarify_message("spouse_age", age_labels)
    )


# Shared DOM helpers for relationship chips (person/child "Relationship to me"
# and spouse Spouse|Partner). Exact leaf labels only — never startsWith on
# multi-chip wrappers (that false-advanced Grandchild on call_ef11).
_RELATIONSHIP_CHIP_JS = """
const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
const isVisible = (el) => {
  const r = el.getBoundingClientRect();
  return r.width > 2 && r.height > 2;
};
const exactLabel = (el, want) => {
  const full = norm(el.innerText || el.textContent || '');
  if (full === want) return true;
  return norm(el.getAttribute('aria-label') || '') === want;
};
const isRelCaption = (t) => {
  if (!t || t.length > 80) return false;
  if (t === 'relationship to me' || t === 'relationship to me*') return true;
  if (t.startsWith('relationship to me')) return true;
  if (t.includes('spouse or') && t.includes('partner')) return true;
  if (t === 'spouse or partner' || t.startsWith('are they your spouse')) return true;
  return false;
};
const findRelRoot = () => {
  const captions = [...document.querySelectorAll(
    'label, legend, h1, h2, h3, h4, p, span, div, strong'
  )];
  for (const el of captions) {
    const t = norm(el.innerText || el.textContent || '');
    if (!isRelCaption(t)) continue;
    let cur = el.parentElement;
    for (let depth = 0; depth < 8 && cur; depth++) {
      const group = [...cur.querySelectorAll(
        'button, [role=button], [role=radio], [role=option], [role=checkbox], label'
      )].filter((n) => {
        if (!isVisible(n)) return false;
        const nt = norm(n.innerText || '');
        return !!nt && nt.length < 48;
      });
      if (group.length >= 2) return cur;
      cur = cur.parentElement;
    }
    return el.parentElement || document;
  }
  return document;
};
const leafChips = (root, want) => {
  const candidates = [...root.querySelectorAll(
    'button, [role=button], [role=radio], [role=option], [role=checkbox], label'
  )].filter((n) => isVisible(n) && exactLabel(n, want));
  return candidates.filter(
    (n) => !candidates.some((other) => other !== n && n.contains(other))
  );
};
const selectedMark = (el) => {
  if (!el) return false;
  if (el.getAttribute('aria-checked') === 'true') return true;
  if (el.getAttribute('aria-pressed') === 'true') return true;
  if (el.getAttribute('aria-selected') === 'true') return true;
  if (el.checked) return true;
  const ds = (el.getAttribute('data-state') || '').toLowerCase();
  if (ds === 'checked' || ds === 'on' || ds === 'selected') return true;
  const cls = String(el.className || '').toLowerCase();
  if (/(?:^|\\s)(?:selected|checked|is-selected|is-checked|mui-selected)(?:\\s|$)/.test(cls)) {
    return true;
  }
  // Pointcare often only flips a visual active/pressed class on the leaf chip.
  // Exact-label leaf + active is safe; bare active on wrappers is not used here.
  if (/(?:^|\\s)(?:active|pressed)(?:\\s|$)/.test(cls)) {
    const aria = (
      el.getAttribute('aria-checked')
      || el.getAttribute('aria-pressed')
      || el.getAttribute('aria-selected')
      || ''
    ).toLowerCase();
    if (aria !== 'false') return true;
  }
  const input = el.control || (el.querySelector && el.querySelector(
    'input[type=checkbox], input[type=radio]'
  ));
  return !!(input && input.checked);
};
"""


async def _relationship_chip_selected(page: Page, value: str) -> bool:
    """True when the exact leaf relationship chip looks selected.

    Scopes to Relationship-to-me or Spouse/Partner captions when present.
    Accepts aria/data/class selected markers and visual ``active`` on an
    exact-label leaf (Pointcare often lacks aria). Rejects parent wrappers.
    """
    wanted = (value or "").strip()
    if not wanted:
        return False
    try:
        return bool(
            await page.evaluate(
                f"""(wanted) => {{
                  {_RELATIONSHIP_CHIP_JS}
                  const want = norm(wanted);
                  if (!want) return false;
                  const chips = leafChips(findRelRoot(), want);
                  return chips.some((el) => selectedMark(el));
                }}""",
                wanted,
            )
        )
    except Exception:
        logger.exception("relationship selected check failed for %r", wanted[:60])
        return False


async def _click_relationship_chip(page: Page, value: str) -> bool:
    """Click the exact leaf relationship chip (person/child/spouse fields)."""
    wanted = (value or "").strip()
    if not wanted:
        return False
    try:
        clicked = await page.evaluate(
            f"""(wanted) => {{
              {_RELATIONSHIP_CHIP_JS}
              const want = norm(wanted);
              if (!want) return false;
              const chips = leafChips(findRelRoot(), want);
              if (!chips.length) return false;
              const rank = (el) => {{
                const role = (el.getAttribute('role') || '').toLowerCase();
                const tag = (el.tagName || '').toLowerCase();
                if (role === 'radio' || tag === 'input') return 3;
                if (tag === 'button' || role === 'button') return 2;
                if (tag === 'label') return 1;
                return 0;
              }};
              chips.sort((a, b) => rank(b) - rank(a));
              // Never click an already-selected chip (toggle-off risk).
              const target = chips.find((el) => !selectedMark(el)) || null;
              if (!target) return true; // already selected
              target.scrollIntoView({{ block: 'center', inline: 'nearest' }});
              target.click();
              return true;
            }}""",
            wanted,
        )
        return bool(clicked)
    except Exception:
        logger.exception("DOM relationship chip click failed for %r", wanted[:60])
        return False


def _relationship_validation_blocking(hints: str) -> bool:
    """True when on-screen validation still demands a relationship selection."""
    low = (hints or "").lower()
    return "please select" in low or (
        "required" in low and "relationship" in low
    ) or "select an option" in low


async def _relationship_marked_selected(page: Page, value: str) -> bool:
    """True only when the exact chip shows a selected marker. Does not click."""
    if await _relationship_chip_selected(page, value):
        return True
    return await _choice_appears_selected(page, value)


async def _ensure_relationship_choice_selected(page: Page, value: str) -> None:
    """Select a relationship chip once.

    A selected marker is enough when the field is not asking for a choice.
    Pointcare Spouse/Partner chips often have no ARIA state. After one click,
    a field that no longer shows "Please select an option" is accepted.
    The chip is never clicked a second time. Visible required text still
    blocks advancement.
    """
    wanted = (value or "").strip()
    if not wanted:
        raise RuntimeError("REL_CLARIFY: Please say the relationship.")

    async def _blocking() -> bool:
        return _relationship_validation_blocking(await _visible_validation_hints(page))

    if await _relationship_marked_selected(page, wanted) and not await _blocking():
        return

    clicked = False
    saw_required = False
    if not await _relationship_marked_selected(page, wanted):
        clicked = await _click_relationship_chip(page, wanted)
        if not clicked:
            try:
                await _click_exact_choice(page, wanted)
                clicked = True
            except Exception:
                logger.exception(
                    "Playwright relationship click failed for %r", wanted[:60]
                )
        for _ in range(12):
            try:
                await page.wait_for_timeout(125)
            except Exception:
                break
            if await _blocking():
                saw_required = True
                break
            if await _relationship_marked_selected(page, wanted):
                return

    if await _relationship_marked_selected(page, wanted) and not await _blocking():
        return
    if clicked and not saw_required and not await _blocking():
        return

    raise RuntimeError(
        f"REL_CLARIFY: I couldn't select {wanted} on the form. "
        "Please say the relationship again."
    )


async def _apply_relationship_chip_for_kind(
    session: LiveFormSession,
    page: Page,
    kind: str,
    mapped: str,
) -> str | None:
    """Select a mapped relationship chip and update session.filled.

    Returns a spoken override when Other opens a custom text field (caller stays
    on the relationship step); otherwise None and the chip value is stored.
    """
    rel_key = {
        "child": "child_rel",
        "person": "person_rel",
        "spouse": "spouse_role",
    }.get(kind, "")
    if not rel_key or not mapped:
        raise RuntimeError("REL_CLARIFY: Please say the relationship.")
    await _ensure_relationship_choice_selected(page, mapped)
    session.filled[rel_key] = mapped
    if mapped.lower().rstrip(" *") == "other" and kind != "spouse":
        try:
            await page.wait_for_timeout(500)
        except Exception:
            pass
        if await _other_relationship_input_visible(page):
            session.filled["rel_other_awaiting"] = rel_key
            return "What is the other relationship?"
        session.filled.pop("rel_other_awaiting", None)
    else:
        session.filled.pop("rel_other_awaiting", None)
        session.filled.pop(f"{rel_key}_other", None)
    return None


async def _try_confirm_relationship_correction(
    session: LiveFormSession,
    page: Page,
    kind: str,
    answer: str,
) -> dict[str, Any] | None:
    """If confirm utterance is a relationship chip, correct in place (no Back).

    Returns an _elig_ok-shaped dict when handled, else None.
    """
    labels = await _visible_choice_labels(page)
    if kind == "spouse":
        lowered = (answer or "").lower()
        if not re.search(r"\b(partner|spouse)\b", lowered):
            return None
        mapped = "Partner" if "partner" in lowered else "Spouse"
    else:
        mapped = _map_relationship_label(answer, labels)
        if not mapped:
            return None
    other_prompt = await _apply_relationship_chip_for_kind(
        session, page, kind, mapped
    )
    confirm_step = _confirm_step_for_kind(kind)
    if other_prompt:
        rel_key = "child_rel" if kind == "child" else "person_rel"
        session.elig_step = rel_key
        return {
            "ok": True,
            "step": rel_key,
            "spoken": "ASK: " + other_prompt,
            "message": f"Updated relationship to {mapped}; awaiting custom text.",
            "value": mapped,
        }
    spoken = _member_confirm_speech(session, kind)
    session.elig_step = confirm_step
    return {
        "ok": True,
        "step": confirm_step,
        "spoken": "ASK: " + spoken,
        "message": f"Updated relationship to {mapped}.",
        "value": mapped,
    }


async def _try_confirm_age_correction(
    session: LiveFormSession,
    page: Page,
    kind: str,
    answer: str,
) -> dict[str, Any] | None:
    """Correct age in place from a confirmation step. Does not click a match."""
    age_key = {
        "child": "child_age",
        "person": "person_age",
        "spouse": "spouse_age",
    }.get(kind, "")
    if not age_key:
        return None
    age_labels = await _ensure_age_choices_visible(page)
    mapped = _map_age_label(answer, age_labels)
    if not mapped:
        return None
    saved = (session.filled.get(age_key) or "").strip()
    if saved.lower() != mapped.lower():
        await _click_age_choice(page, mapped)
        session.filled[age_key] = mapped
    confirm_step = _confirm_step_for_kind(kind)
    spoken = _member_confirm_speech(session, kind)
    session.elig_step = confirm_step
    return {
        "ok": True,
        "step": confirm_step,
        "spoken": "ASK: " + spoken,
        "message": f"Age is {mapped}.",
        "value": mapped,
    }


async def _reassert_member_relationship_before_continue(
    session: LiveFormSession, page: Page, kind: str
) -> None:
    """Read the relationship field before Continue. Never click the chip.

    Continue is blocked when the field still shows a required-option message.
    A missing ARIA marker alone does not block Continue when that message is gone.
    """
    rel_key = {
        "child": "child_rel",
        "person": "person_rel",
        "spouse": "spouse_role",
    }.get(kind, "")
    if not rel_key:
        return
    wanted = (session.filled.get(rel_key) or "").strip()
    if not wanted:
        raise RuntimeError(
            "REL_CLARIFY: Relationship is still blank on the form. "
            "What is their relationship to you?"
        )
    custom = (session.filled.get(f"{rel_key}_other") or "").strip()
    chip = "Other" if custom else wanted
    if chip.lower().rstrip(" *") == "other":
        chip = "Other"
    if not await _relationship_marked_selected(page, chip):
        if _relationship_validation_blocking(await _visible_validation_hints(page)):
            raise RuntimeError(
                "REL_CLARIFY: The form still asks for a relationship. "
                "Please say the relationship again."
            )
    elif _relationship_validation_blocking(await _visible_validation_hints(page)):
        raise RuntimeError(
            "REL_CLARIFY: The form still asks for a relationship. "
            "Please say the relationship again."
        )
    if custom and await _other_relationship_input_visible(page):
        current = ""
        loc = await _locate_other_relationship_input(page)
        if loc is not None:
            try:
                current = (await loc.input_value()).strip()
            except Exception:
                current = ""
        if current.lower() != custom.lower():
            await _fill_other_relationship_text(page, custom)


async def _assert_member_form_ready_before_continue(
    session: LiveFormSession, page: Page, kind: str
) -> None:
    """Block Continue when age is missing or the form still requires a choice.

    Does not click the age chip. A saved age with no required-selection
    message is enough, even when the chip has no ARIA or selected class.
    """
    await _reassert_member_relationship_before_continue(session, page, kind)
    age_key = {
        "child": "child_age",
        "person": "person_age",
        "spouse": "spouse_age",
    }.get(kind, "")
    age = (session.filled.get(age_key) or "").strip() if age_key else ""
    if not age:
        raise RuntimeError(
            "AGE_CLARIFY: Age is still blank on the form. "
            "What is their age range?"
        )
    hints = await _visible_validation_hints(page)
    if _relationship_validation_blocking(hints):
        raise RuntimeError(
            "AGE_CLARIFY: The form still asks for a selection. "
            "Please say the age again."
        )


async def _click_exact_choice(page: Page, value: str) -> None:
    labels = await _visible_choice_labels(page)
    # Age chip answers must use the dedicated verifier — never generic/income fallbacks.
    if _looks_like_age_label(value):
        await _click_age_choice(page, value)
        return
    wanted = _fuzzy_choice_label(value, labels)
    logger.info(
        "Choice map spoken=%r resolved=%r visible=%s",
        (value or "")[:80],
        wanted,
        labels[:10],
    )
    if _looks_like_age_label(wanted):
        await _click_age_choice(page, wanted)
        return
    pattern = re.compile(rf"^\s*{re.escape(wanted)}\s*$", re.I)
    before = await _choice_signature(page, wanted)
    candidates = [
        page.get_by_role("radio", name=pattern).first,
        page.get_by_role("checkbox", name=pattern).first,
        page.get_by_role("button", name=pattern).first,
        page.get_by_text(pattern, exact=True).first,
    ]
    for loc in candidates:
        try:
            if await loc.count() and await loc.is_visible():
                await loc.click(timeout=6000)
                for _ in range(20):
                    await page.wait_for_timeout(150)
                    if await _choice_signature(page, wanted) != before:
                        return
                # Some Pointcare choices navigate without exposing selected state.
                return
        except Exception:
            continue
    # Fuzzy DOM click as a second path for Yes/No and near-miss labels.
    try:
        await _click_choice(page, wanted)
        return
    except Exception:
        pass
    # Income chips may be plain divs; match by normalized dollar text.
    # Never use this path for age chips (digit prefix matching is unsafe).
    if "$" in wanted or (
        re.search(r"\d", wanted)
        and not _looks_like_age_label(wanted)
        and _labels_look_like_income(labels)
    ):
        try:
            clicked = await page.evaluate(
                """(wanted) => {
                  const norm = (s) => (s || '').replace(/[$,\\s–—-]/g, '').toLowerCase();
                  const target = norm(wanted);
                  const nodes = [...document.querySelectorAll('button, [role=button], div, span, li, a, label')];
                  for (const node of nodes) {
                    const t = (node.innerText || '').replace(/\\s+/g, ' ').trim();
                    if (!t || t.length > 60) continue;
                    if (norm(t) !== target && !norm(t).startsWith(target) && !target.startsWith(norm(t))) continue;
                    const r = node.getBoundingClientRect();
                    if (r.width < 2 || r.height < 2) continue;
                    node.click();
                    return t;
                  }
                  return '';
                }""",
                wanted,
            )
            if clicked:
                logger.info("Clicked income-like choice via DOM text=%r", clicked)
                await page.wait_for_timeout(400)
                return
        except Exception:
            logger.exception("DOM income choice click failed for %r", wanted)
    raise RuntimeError(
        f"Could not select exact choice {wanted!r} from spoken {value!r}; "
        f"visible={labels[:10]}"
    )


async def _click_all_exact_choices(page: Page, value: str) -> int:
    pattern = re.compile(rf"^\s*{re.escape(value)}\s*$", re.I)
    loc = page.get_by_role("button", name=pattern)
    count = await loc.count()
    clicked = 0
    for index in range(count):
        item = loc.nth(index)
        try:
            if await item.is_visible():
                await item.click(timeout=5000)
                clicked += 1
        except Exception:
            continue
    return clicked


_NON_MEMBER_LABELS = frozenset(
    {
        "exit",
        "back",
        "continue",
        "next",
        "done",
        "me",  # kept via allowlist below when exact
        "never married",
        "married",
        "separated",
        "divorced",
        "widowed",
        "registered domestic partner",
        "common law marriage",
        "first name",
        "last name",
        "living with me",
        "same mailing address",
        "relationship to me",
        "age",
        "annual",
        "monthly",
        "bi-weekly",
        "yes",
        "no",
    }
)


def _spoken_member_query(answer: str) -> str:
    """Normalize spoken member text for matching (does not change DOM labels).

    Examples: "Angel or" → "Angel"; "Angel r dot" → "Angel R"; "John." → "John".
    """
    from .answer_parse import sanitize_person_name

    text = sanitize_person_name(answer or "")
    # Spoken initials: "r dot" / "R. dot" / "d period" → single letter.
    text = re.sub(
        r"\b([A-Za-z])\s*(?:dot|period)\b\.?",
        lambda m: m.group(1).upper(),
        text,
        flags=re.I,
    )
    # "Angel R." style already fine; collapse "Angel R ." leftovers.
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(
        r"\s+\b(or|and|uh|um|ah|the|a|an)\b\s*$",
        "",
        text,
        flags=re.I,
    ).strip()
    if re.search(r"\b(me|myself|caller)\b", (answer or "").lower()):
        return "Me"
    return text


def _split_member_queries(answer: str) -> list[str]:
    """Split 'Murali and Agent H' / 'myself, my wife' into separate queries."""
    text = (answer or "").strip()
    if not text:
        return []
    # Keep "and" inside names rare; split on list connectors.
    parts = re.split(
        r"\s*(?:,|;|\band\b|\balso\b|\bplus\b|\bas well as\b)\s*",
        text,
        flags=re.I,
    )
    out: list[str] = []
    for part in parts:
        cleaned = _spoken_member_query(part)
        if cleaned and cleaned.lower() not in {"and", "also", "plus"}:
            out.append(cleaned)
    # If split produced nothing useful, fall back to the whole utterance.
    if not out:
        whole = _spoken_member_query(text)
        if whole:
            out.append(whole)
    return out


def _normalize_member_label(text: str) -> str:
    cleaned = _spoken_member_query(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    # "John. D" / "John D." / "Murali M." → comparable token string
    cleaned = re.sub(r"\.(?=\s|$)", "", cleaned)
    cleaned = cleaned.replace(".", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower().strip()


def _resolve_kinship_to_member(
    spoken: str,
    members: list[dict[str, Any]],
    session: LiveFormSession,
) -> str | None:
    """Map myself/wife/daughter to a unique visible member label, or None."""
    text = (spoken or "").strip().lower()
    labels = [str(m.get("label") or "") for m in members]

    if re.search(r"\b(me|myself|i am|caller)\b", text) or text == "me":
        me_hits = [label for label in labels if label.strip().lower() == "me"]
        if len(me_hits) == 1:
            return me_hits[0]
        # Prefer primary applicant-looking single unique first name from session.
        return None

    spouse_first = (session.filled.get("spouse_first") or "").strip()
    spouse_last = (session.filled.get("spouse_last") or "").strip()
    if re.search(r"\b(wife|husband|spouse|partner)\b", text):
        candidates: list[str] = []
        for label in labels:
            norm = _normalize_member_label(label)
            if spouse_first and spouse_first.lower() in norm:
                candidates.append(label)
            elif spouse_last and spouse_last.lower() in norm:
                candidates.append(label)
        # Unique only — never guess among several adults.
        uniq = list(dict.fromkeys(candidates))
        if len(uniq) == 1:
            return uniq[0]
        return None

    if re.search(r"\b(daughter|son|child|kid)\b", text):
        child_first = (session.filled.get("child_first") or "").strip()
        candidates = []
        for label in labels:
            norm = _normalize_member_label(label)
            if child_first and child_first.lower() in norm:
                candidates.append(label)
        uniq = list(dict.fromkeys(candidates))
        if len(uniq) == 1:
            return uniq[0]
        return None

    return None


def _is_plausible_member_label(label: str) -> bool:
    text = (label or "").strip()
    if not text or len(text) > 60:
        return False
    key = re.sub(r"\s+", " ", text).lower().rstrip(" *.")
    if key in _NON_MEMBER_LABELS and key != "me":
        return False
    if re.search(
        r"someone is |someone has |someone was |none of the above|pregnant|"
        r"breast cancer|cervical|\bblind\b|\bssdi\b|foster care|"
        r"who does this apply",
        text,
        re.I,
    ):
        return False
    if re.search(r"^\$?\d", text):
        return False
    # Prefer person-like labels: "Me", "Angel R.", "John. D."
    if re.fullmatch(r"me", text, re.I):
        return True
    # Reject aggregate row text: "Murali M. Agent H. David W. Mickey M."
    if len(re.findall(r"[A-Za-z][A-Za-z'\-]*\s+[A-Za-z]\.?", text)) > 1:
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z .'\-]{0,40}$", text))


def _special_condition_negated(text: str, cue: str) -> bool:
    """True when the cue appears only inside a negated phrase."""
    lowered = (text or "").lower()
    cue_re = re.escape(cue)
    # "not blind", "isn't pregnant", "no one is blind", "nobody has cancer"
    if re.search(
        rf"\b(not|never|no|n't)\b(?:\s+\w+){{0,4}}\s+\b{cue_re}\b",
        lowered,
    ):
        return True
    if re.search(
        rf"\b(no\s+one|nobody|neither)\b(?:\s+\w+){{0,6}}\s+\b{cue_re}\b",
        lowered,
    ):
        return True
    if re.search(
        rf"\b(isn'?t|aren'?t|wasn'?t|weren'?t|don'?t|doesn'?t|didn'?t)\b"
        rf"(?:\s+\w+){{0,4}}\s+\b{cue_re}\b",
        lowered,
    ):
        return True
    return False


def _short_special_condition_label(label: str) -> str:
    """Spoken-friendly short form of a Pointcare special-condition chip."""
    text = (label or "").strip()
    text = re.sub(
        r"^someone\s+(is|has|was)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*\(ssdi\)\s*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or (label or "").strip()


def _record_special_assignment_pairs(
    session: LiveFormSession, conditions: list[str], members: list[str]
) -> None:
    """Remember verified condition→member assignments for confirmation speech."""
    conds = [c.strip() for c in conditions if (c or "").strip()]
    mems = [m.strip() for m in members if (m or "").strip()]
    if not conds or not mems:
        return
    existing = session.filled.get("special_assignment_pairs", "")
    chunks = [c for c in existing.split("||") if c.strip()]
    by_cond = {}
    for chunk in chunks:
        if ">>>" not in chunk:
            continue
        cond, mem_part = chunk.split(">>>", 1)
        by_cond[cond] = [m for m in mem_part.split("&") if m]
    for cond in conds:
        prior = by_cond.get(cond, [])
        for mem in mems:
            if mem not in prior:
                prior.append(mem)
        by_cond[cond] = prior
    session.filled["special_assignment_pairs"] = "||".join(
        f"{cond}>>>{'&'.join(mems)}" for cond, mems in by_cond.items()
    )


def _special_more_confirm_speech(session: LiveFormSession) -> str:
    """Confirm verified condition/member selections, then ask about more conditions."""
    pairs_raw = session.filled.get("special_assignment_pairs", "")
    parts: list[str] = []
    if pairs_raw:
        for chunk in pairs_raw.split("||"):
            if ">>>" not in chunk:
                continue
            cond, mem_part = chunk.split(">>>", 1)
            short = _short_special_condition_label(cond)
            members = " and ".join(m for m in mem_part.split("&") if m)
            if short and members:
                parts.append(f"{short} selected for {members}")
            elif short:
                parts.append(f"{short} selected")
    if not parts:
        conds = [
            _short_special_condition_label(c.strip())
            for c in (session.filled.get("special_conditions") or "").split(",")
            if c.strip()
        ]
        members = [
            m
            for m in (session.filled.get("special_person_assigned") or "").split("|")
            if m
        ]
        cond_text = " and ".join(conds) if conds else "the selected condition"
        if members:
            parts.append(f"{cond_text} selected for {' and '.join(members)}")
        else:
            parts.append(f"{cond_text} selected")
    if len(parts) == 1:
        body = f"I have {parts[0]}."
    else:
        body = "I have " + "; ".join(parts) + "."
    return body + " Are there any other conditions to add? Please say Yes or No."


async def _verify_special_selections_before_next(
    page: Page, session: LiveFormSession
) -> None:
    """Ensure remembered conditions (and members when visible) are still selected."""
    conditions = [
        c.strip()
        for c in (session.filled.get("special_conditions") or "").split(",")
        if c.strip()
        and not re.search(r"none of the above", c, re.I)
    ]
    for cond in conditions:
        state = await _special_condition_checkbox_state(page, cond)
        if state.get("found") and not state.get("checked"):
            raise RuntimeError(
                "SPECIAL_MORE_CLARIFY: I no longer see "
                f"{_short_special_condition_label(cond)} selected. "
                "Please say the condition again."
            )
    assigned = [
        m
        for m in (session.filled.get("special_person_assigned") or "").split("|")
        if m
    ]
    if not assigned:
        return
    discovered = await _discover_special_person_members(page)
    members = list(discovered.get("members") or [])
    if not members:
        # Apply-to list may collapse after assignment; conditions already verified.
        return
    by_norm = {
        _normalize_member_label(str(m.get("label") or "")): m for m in members
    }
    for name in assigned:
        row = by_norm.get(_normalize_member_label(name))
        if row is not None and not row.get("checked"):
            raise RuntimeError(
                "SPECIAL_MORE_CLARIFY: I no longer see "
                f"{name} selected for the condition. "
                "Please say the household member again."
            )


def _map_special_condition_labels(spoken: str, labels: list[str]) -> list[str]:
    """Map spoken special-condition answer onto visible Pointcare labels.

    Supports natural phrasing ("I'm blind", "visual disability") and multi-select.
    Negated cues never select a condition. "None of the above" is exclusive.
    """
    text = re.sub(r"[.!?…]+$", "", (spoken or "").strip().lower())
    if not text:
        return []
    condition_labels = [
        label
        for label in labels
        if re.search(
            r"someone is |someone has |someone was |none of the above|"
            r"pregnant|cancer|blind|disabled|foster",
            label,
            re.I,
        )
    ]
    pool = condition_labels or list(labels)

    def _find(*needles: str) -> str:
        for label in pool:
            low = label.lower()
            if all(n in low for n in needles):
                return label
        return ""

    # Synonym / phrase cues → visible labels (order matters for multi-select).
    cue_checks = [
        (
            "none",
            ("none",),
            bool(re.search(r"\b(none of the above|none|neither|no conditions?)\b", text))
            and not re.search(
                r"\b(blind|pregnant|cancer|disabled|ssdi|foster|visual)\b", text
            ),
        ),
        (
            "pregnant",
            ("pregnant",),
            bool(re.search(r"\b(pregnant|pregnancy|expecting)\b", text)),
        ),
        (
            "breast",
            ("breast", "cancer"),
            bool(re.search(r"\bbreast\b", text) and re.search(r"\bcancer\b", text)),
        ),
        (
            "cervical",
            ("cervical",),
            bool(re.search(r"\bcervical\b", text)),
        ),
        (
            "blind",
            ("blind",),
            bool(
                re.search(
                    r"\b(blind|visually impaired|visual(?:ly)?\s+disab\w*|vision loss|"
                    r"can'?t see|cannot see)\b",
                    text,
                )
            ),
        ),
        (
            "disabled",
            ("disabled",),
            bool(
                re.search(r"\b(disabled|ssdi|disability|on disability)\b", text)
                and not re.search(r"\bvisual(?:ly)?\s+disab", text)
            ),
        ),
        (
            "foster",
            ("foster",),
            bool(re.search(r"\bfoster\b", text)),
        ),
    ]

    hits: list[str] = []
    for cue_key, needles, matched in cue_checks:
        if not matched:
            continue
        if cue_key != "none" and _special_condition_negated(text, cue_key):
            continue
        if cue_key == "blind" and _special_condition_negated(text, "visual"):
            continue
        label = _find(*needles) if cue_key != "none" else _find("none of the above")
        if not label and cue_key == "none":
            label = next(
                (item for item in pool if "none of the above" in item.lower()), ""
            )
        if label and label not in hits:
            hits.append(label)

    # Direct containment / equality against visible options (non-negated).
    for label in pool:
        low = label.lower()
        if label in hits:
            continue
        if text == low or (len(text) >= 8 and (text in low or low in text)):
            # Skip if the matching cue words are negated in the utterance.
            cue = next(
                (
                    word
                    for word in (
                        "blind",
                        "pregnant",
                        "disabled",
                        "foster",
                        "breast",
                        "cervical",
                    )
                    if word in low and word in text
                ),
                "",
            )
            if cue and _special_condition_negated(text, cue):
                continue
            hits.append(label)
            continue
        spoken_words = set(re.findall(r"[a-z0-9]{3,}", text))
        label_words = set(re.findall(r"[a-z0-9]{3,}", low))
        if spoken_words and spoken_words <= label_words:
            if any(_special_condition_negated(text, w) for w in spoken_words & label_words):
                continue
            hits.append(label)

    # "None of the above" is exclusive — drop it when real conditions were selected.
    none_labels = [label for label in hits if "none of the above" in label.lower()]
    real = [label for label in hits if "none of the above" not in label.lower()]
    if real and none_labels:
        hits = real
    elif none_labels and not real:
        hits = none_labels[:1]

    out: list[str] = []
    for label in hits:
        if label not in out:
            out.append(label)
    return out


def _map_relationship_label(spoken: str, labels: list[str]) -> str | None:
    """Map natural relationship phrases onto a unique visible option.

    Returns None when ambiguous or unsupported — never guesses the first option.
    """
    text = re.sub(r"[.!?…]+$", "", (spoken or "").strip().lower())
    text = re.sub(r"\b(my|our|the|a|an|their|his|her)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or not labels:
        return None
    pool = [
        label
        for label in labels
        if label
        and label.lower().rstrip(" *")
        not in {
            "first name",
            "last name",
            "living with me",
            "same mailing address",
            "relationship to me",
            "date of birth (optional)",
            "age",
        }
    ] or list(labels)

    for label in pool:
        if label.lower() == text:
            return label

    synonym_groups: list[tuple[str, tuple[str, ...]]] = [
        (
            "stepchild",
            (
                r"\bstep[\s-]?child\b",
                r"\bstep[\s-]?daughter\b",
                r"\bstep[\s-]?son\b",
                r"\bstep[\s-]?kid\b",
            ),
        ),
        (
            "grandchild",
            (
                r"\bgrand[\s-]?child\b",
                r"\bgrand[\s-]?daughter\b",
                r"\bgrand[\s-]?son\b",
                r"\bgrandkid\b",
            ),
        ),
        (
            "foster child",
            (r"\bfoster\b",),
        ),
        (
            "child",
            (
                r"\bdaughter\b",
                r"\bson\b",
                r"\bkid\b",
                r"\bchild\b",
                r"\bchildren\b",
            ),
        ),
        (
            "other",
            (
                r"\bother\b",
                r"\bsomething else\b",
                r"\banother relationship\b",
                r"\ba different relationship\b",
                r"\bnone of (?:those|these|the above)\b",
                r"\bnot (?:listed|any of (?:those|these))\b",
            ),
        ),
    ]
    def _norm_rel_key(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower().replace("-", " ")).strip()

    for target_key, patterns in synonym_groups:
        if not any(re.search(pat, text, re.I) for pat in patterns):
            continue
        # Exact normalized label only — "child" must not match Stepchild/Grandchild.
        hits = [
            label
            for label in pool
            if _norm_rel_key(label) == target_key
        ]
        if target_key != "other":
            hits = [label for label in hits if "other" not in label.lower()] or hits
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return None
    # Unique containment as last resort.
    contains = [
        label
        for label in pool
        if label.lower() in text or text in label.lower()
    ]
    if len(contains) == 1:
        return contains[0]
    return None


_SPECIAL_MEMBER_DISCOVER_JS = """(conditionLabel) => {
  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
  const conditionKey = (label) => {
    const low = norm(label).toLowerCase();
    if (!low) return '';
    if (low.includes('cervical')) return 'cervical';
    if (low.includes('breast')) return 'breast';
    if (low.includes('blind')) return 'blind';
    if (low.includes('pregnant')) return 'pregnant';
    if (low.includes('foster')) return 'foster';
    if (low.includes('disabled') || low.includes('ssdi')) return 'disabled';
    return low;
  };
  const findConditionCard = (label) => {
    const key = conditionKey(label);
    if (!key) return null;
    const others = ['cervical', 'breast', 'blind', 'pregnant', 'foster', 'disabled']
      .filter((item) => item !== key);
    // Prefer climbing from the condition's own control — shared ancestors with
    // other conditions are allowed; pick the smallest reliable apply-to block.
    const seeds = [];
    for (const el of document.querySelectorAll(
      'label, button, [role=checkbox], [role=button], li, div, span'
    )) {
      const t = norm(el.innerText || el.textContent || '');
      if (!t || t.length > 140) continue;
      if (!/someone is |someone has |someone was /i.test(t)) continue;
      if (conditionKey(t) !== key) continue;
      seeds.push({ el, len: t.length });
    }
    seeds.sort((a, b) => a.len - b.len);
    for (const seed of seeds) {
      let cur = seed.el;
      for (let depth = 0; depth < 14 && cur; depth++) {
        const t = norm(cur.innerText || '').toLowerCase();
        if (
          t
          && t.includes(key)
          && /who does this apply to/i.test(t)
          && cur.querySelector('input[type=checkbox], [role=checkbox]')
        ) {
          return cur;
        }
        cur = cur.parentElement;
      }
    }
    // Fallback: smallest container with this key + apply-to. Soft-penalize
    // other condition keywords; do not hard-reject shared parents.
    const hits = [];
    for (const el of document.querySelectorAll('section, article, li, fieldset, div')) {
      const t = norm(el.innerText || '').toLowerCase();
      if (!t || t.length > 2500 || !t.includes(key)) continue;
      if (!/who does this apply to/i.test(t)) continue;
      if (!el.querySelector('input[type=checkbox], [role=checkbox]')) continue;
      const otherCount = others.filter((item) => t.includes(item)).length;
      const r = el.getBoundingClientRect();
      hits.push({
        el,
        len: t.length,
        otherCount,
        area: Math.max(1, r.width * r.height),
      });
    }
    hits.sort((a, b) => a.otherCount - b.otherCount || a.len - b.len || a.area - b.area);
    return hits.length ? hits[0].el : null;
  };
  const bodyText = norm((document.body && document.body.innerText) || '').slice(0, 2500);
  const hasApplyTo = /who does this apply to/i.test(bodyText);
  const hasSpecial = /someone is blind|someone is pregnant|none of the above|special cases|someone is disabled/i.test(bodyText);
  const hasMarital = /never married|what is your marital status/i.test(bodyText);
  const isCondition = (t) => /who does this apply to|someone is |someone has |someone was |none of the above|pregnant|breast cancer|cervical|\\bblind\\b|\\bssdi\\b|foster/i.test(t || '');
  const isNav = (t) => /^(exit|back|continue|next|done|recheck eligibility|add member|never married|married|separated|divorced|widowed|registered domestic partner|common law marriage|annual|monthly|bi-weekly|yes|no)$/i.test(t || '');
  // Reject aggregates like "Murali M. Agent H. David W. Mickey M."
  const isMultiName = (t) => {
    const hits = (t || '').match(/[A-Za-z][A-Za-z'\\-]*\\s+[A-Za-z]\\.?/g);
    return !!(hits && hits.length > 1);
  };
  const isPersonLike = (t) => {
    if (!t || t.length > 60 || t.length < 1) return false;
    if (isCondition(t) || isNav(t) || isMultiName(t)) return false;
    if (/^\\$?\\d/.test(t)) return false;
    if (/^me$/i.test(t)) return true;
    return /^[A-Za-z][A-Za-z .'\\-]{0,40}$/.test(t);
  };
  const leafPersonTexts = (root) => {
    if (!root || !root.querySelectorAll) return [];
    const out = [];
    for (const el of root.querySelectorAll(
      'div, span, p, label, h1, h2, h3, h4, [role=heading]'
    )) {
      const text = norm(el.innerText || el.textContent || '');
      if (!isPersonLike(text)) continue;
      // Prefer leaf name nodes (Pointcare puts "Mickey M." in a nested div).
      const childPerson = [...el.querySelectorAll('div, span, p, label, h1, h2, h3, h4')]
        .some((c) => {
          const ct = norm(c.innerText || c.textContent || '');
          return c !== el && isPersonLike(ct) && ct.length < text.length;
        });
      if (childPerson) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) continue;
      out.push(text);
    }
    return out;
  };
  // Pointcare: empty button[role=checkbox] beside a sibling name div in the same row.
  const nameFromCheckbox = (box) => {
    if (!box) return '';
    let name = box.getAttribute('aria-label') || '';
    if (!name && box.id) {
      const lab = document.querySelector('label[for=\"' + box.id.replace(/\"/g, '') + '\"]');
      if (lab) name = lab.innerText || lab.textContent || '';
    }
    if (!name) {
      const wrap = box.closest('label');
      if (wrap) name = wrap.innerText || wrap.textContent || '';
    }
    if (isPersonLike(norm(name))) return norm(name);
    // Climb to the member row; parent of the box is often an empty padding div.
    let cur = box.parentElement;
    for (let depth = 0; depth < 8 && cur; depth++) {
      const leaves = leafPersonTexts(cur);
      if (leaves.length === 1) return leaves[0];
      if (leaves.length > 1) {
        // Same row usually has exactly one name; take shortest unique leaf.
        leaves.sort((a, b) => a.length - b.length);
        if (!isMultiName(leaves[0])) return leaves[0];
      }
      const own = norm(cur.innerText || cur.textContent || '');
      if (isPersonLike(own)) return own;
      cur = cur.parentElement;
    }
    return '';
  };
  const readChecked = (box) => !!(
    (box && box.tagName === 'INPUT' && box.checked)
    || (box && box.getAttribute && box.getAttribute('aria-checked') === 'true')
    || (box && /\\b(checked|selected|active)\\b/i.test((box.className || '').toString()))
  );
  const collect = (root) => {
    if (!root) return [];
    const out = [];
    const seenCounts = {};
    const push = (text, box) => {
      if (!isPersonLike(text)) return;
      const r = box && box.getBoundingClientRect
        ? box.getBoundingClientRect()
        : { width: 2, height: 2 };
      // Name may be visible while the empty checkbox button has zero text size —
      // still accept when the row text resolved.
      if (box && r.width < 1 && r.height < 1) {
        const row = box.closest('[tabindex], div');
        const rr = row && row.getBoundingClientRect && row.getBoundingClientRect();
        if (!rr || rr.width < 1 || rr.height < 1) return;
      }
      const key = text.toLowerCase();
      const dupCount = seenCounts[key] || 0;
      seenCounts[key] = dupCount + 1;
      out.push({
        label: text,
        index: out.length,
        dupIndex: dupCount,
        checked: readChecked(box),
      });
    };
    // Primary: pair each checkbox control with its sibling/ancestor name leaf.
    for (const box of root.querySelectorAll('input[type=checkbox], [role=checkbox]')) {
      const text = nameFromCheckbox(box);
      if (!text || isCondition(text)) continue;
      push(text, box);
    }
    // Dedupe by label keeping first (first checkbox wins).
    const uniq = [];
    const seen = {};
    for (const row of out) {
      const k = (row.label || '').toLowerCase();
      if (seen[k]) continue;
      seen[k] = true;
      uniq.push(row);
    }
    return uniq;
  };
  const checkboxCount = document.querySelectorAll(
    'input[type=checkbox], [role=checkbox]'
  ).length;
  const sampleLabels = [...document.querySelectorAll(
    'label, [role=checkbox], button, [role=button], h1, h2, h3, h4, [role=heading], div, span'
  )]
    .map((n) => norm(n.innerText || n.textContent || n.getAttribute('aria-label') || ''))
    .filter((t) => t && t.length < 80)
    .slice(0, 20);
  const meta = {
    hasApplyTo,
    hasSpecial,
    hasMarital,
    checkboxCount,
    visibleSample: sampleLabels,
    url: (location && location.href) || '',
  };

  // Active condition only. Never climb to a wrapper that also contains Blind
  // or any other condition card, and never fall back to the first page list.
  if (conditionLabel) {
    const card = findConditionCard(conditionLabel);
    if (!card) {
      return { members: [], strategy: 'condition_scope_missing', ...meta };
    }
    const scoped = collect(card);
    return {
      members: scoped,
      strategy: scoped.length ? 'condition_scope' : 'condition_members_missing',
      ...meta,
    };
  }

  // Wrong / unready page: marital chrome without Special Cases apply-to.
  if (hasMarital && !hasApplyTo && !hasSpecial) {
    return { members: [], strategy: 'wrong_screen_marital', ...meta };
  }

  // Strategy 1: heading "Who does this apply to?" — prefer the short heading node.
  let strategy = 'heading';
  let root = null;
  const headings = [...document.querySelectorAll(
    'h1,h2,h3,h4,div,span,p,legend,label,strong,b,[role=heading]'
  )];
  // Prefer exact/short heading text over wrappers that also include member names.
  const headingHits = [];
  for (const el of headings) {
    const t = norm(el.innerText || el.textContent);
    if (!t || t.length > 80) continue;
    if (!/who does this apply to/i.test(t)) continue;
    headingHits.push({ el, t, len: t.length });
  }
  headingHits.sort((a, b) => a.len - b.len);
  if (headingHits.length) {
    const el = headingHits[0].el;
    root = el.closest('section,fieldset,article,form,div') || el.parentElement;
    let cur = el.parentElement;
    for (let i = 0; i < 10 && cur; i++) {
      const boxes = cur.querySelectorAll('input[type=checkbox], [role=checkbox]').length;
      const names = leafPersonTexts(cur).length;
      // Expand until the apply-to block that holds member checkbox rows.
      if (boxes >= 1 && names >= 1) {
        root = cur;
        break;
      }
      cur = cur.parentElement;
    }
  }
  let members = collect(root);
  if (members.length) {
    return { members, strategy, ...meta };
  }

  // Strategy 2: open special-condition accordion (data-state=open) only.
  strategy = 'special_section_fallback';
  const openBlocks = [...document.querySelectorAll('[data-state=open], [aria-expanded=true]')]
    .filter((n) => {
      const t = norm(n.innerText || n.textContent || '');
      return t && /someone is |someone has |someone was /i.test(t)
        && /who does this apply to/i.test(t);
    });
  for (const block of openBlocks) {
    const found = collect(block);
    if (found.length >= 1) {
      members = found;
      root = block;
      break;
    }
  }
  if (members.length) {
    return { members, strategy, ...meta };
  }

  // Do NOT treat page-wide checkbox counts as member evidence (call_ccdc…).
  strategy = hasApplyTo ? 'apply_to_empty' : 'section_missing';
  return { members: [], strategy, ...meta };
}"""


async def _discover_special_person_members(
    page: Page,
    condition: str = "",
) -> dict[str, Any]:
    """Discover member rows for one condition card, or the open apply-to list."""
    try:
        # Omit the argument when no condition is active so the open-list path,
        # and older callers, stay unchanged.
        if condition:
            raw = await page.evaluate(_SPECIAL_MEMBER_DISCOVER_JS, condition)
        else:
            raw = await page.evaluate(_SPECIAL_MEMBER_DISCOVER_JS)
    except Exception:
        logger.exception("Could not list special-person member checkboxes")
        return {
            "members": [],
            "strategy": "error",
            "checkbox_count": 0,
            "visible_labels": [],
            "has_apply_to": False,
            "has_special": False,
            "has_marital": False,
            "url": "",
        }
    # Back-compat: unit tests may mock evaluate to return a bare list.
    if isinstance(raw, list):
        members_raw = raw
        strategy = "legacy_list"
        checkbox_count = 0
        visible_labels: list[str] = []
        has_apply_to = False
        has_special = False
        has_marital = False
        url = ""
    else:
        data = raw if isinstance(raw, dict) else {}
        members_raw = list(data.get("members") or [])
        strategy = str(data.get("strategy") or "unknown")
        checkbox_count = int(data.get("checkboxCount") or 0)
        visible_labels = [str(x) for x in (data.get("visibleSample") or [])][:20]
        has_apply_to = bool(data.get("hasApplyTo"))
        has_special = bool(data.get("hasSpecial"))
        has_marital = bool(data.get("hasMarital"))
        url = str(data.get("url") or "")[:160]

    filtered = [
        row
        for row in members_raw
        if isinstance(row, dict)
        and _is_plausible_member_label(str(row.get("label") or ""))
    ]
    # Re-index after Python filtering while preserving dupIndex per label.
    seen: dict[str, int] = {}
    members: list[dict[str, Any]] = []
    for row in filtered:
        label = str(row.get("label") or "")
        key = label.lower()
        dup = seen.get(key, 0)
        seen[key] = dup + 1
        members.append(
            {
                "label": label,
                "index": len(members),
                "dupIndex": int(row.get("dupIndex") if "dupIndex" in row else dup),
                "checked": bool(row.get("checked")),
            }
        )
    return {
        "members": members,
        "strategy": strategy,
        "checkbox_count": checkbox_count,
        "visible_labels": visible_labels,
        "has_apply_to": has_apply_to,
        "has_special": has_special,
        "has_marital": has_marital,
        "url": url,
    }


async def _wait_for_special_apply_to_members(
    page: Page,
    *,
    timeout_ms: int = 3500,
    call_id: str = "",
    condition: str = "",
) -> dict[str, Any]:
    """Poll until the apply-to section exposes household member options.

    First check is immediate; bounded retries only when the list is still empty.
    """
    started = time.time()
    deadline = started + max(0.35, timeout_ms / 1000.0)
    last: dict[str, Any] = {}
    while True:
        last = await _discover_special_person_members(page, condition)
        members = list(last.get("members") or [])
        if members:
            logger.info(
                "special_apply_to_ready call_id=%s strategy=%s count=%s "
                "wait_ms=%.0f condition=%r",
                call_id,
                last.get("strategy"),
                len(members),
                (time.time() - started) * 1000,
                (condition or "")[:40],
            )
            return last
        strategy = str(last.get("strategy") or "")
        if strategy.startswith("wrong_screen"):
            break
        if time.time() >= deadline:
            break
        try:
            await page.wait_for_timeout(120)
        except Exception:
            break
    logger.warning(
        "special_apply_to_not_ready call_id=%s strategy=%s wait_ms=%.0f "
        "has_apply_to=%s has_special=%s condition=%r",
        call_id,
        last.get("strategy"),
        (time.time() - started) * 1000,
        last.get("has_apply_to"),
        last.get("has_special"),
        (condition or "")[:40],
    )
    return last


async def _recover_special_apply_to_section(
    page: Page,
    session: LiveFormSession,
) -> dict[str, Any]:
    """If apply-to is missing, re-assert the selected condition once (no toggle loop)."""
    conditions = [
        part.strip()
        for part in (session.filled.get("special_conditions") or "").split(",")
        if part.strip() and not re.search(r"none of the above", part, re.I)
    ]
    if not conditions:
        return await _discover_special_person_members(page)

    # Scroll toward special content if marital chrome is polluting the view.
    try:
        await page.evaluate(
            """() => {
              const hit = [...document.querySelectorAll('h1,h2,h3,div,span,label,button')]
                .find((el) => /special cases|someone is blind|who does this apply to/i
                  .test(el.innerText || ''));
              if (hit && hit.scrollIntoView) hit.scrollIntoView({ block: 'center' });
            }"""
        )
        await page.wait_for_timeout(300)
    except Exception:
        pass

    discovered = await _discover_special_person_members(page)
    if discovered.get("members"):
        return discovered

    # Re-select the first saved condition once to reopen apply-to.
    label = conditions[0]
    state = await _special_condition_checkbox_state(page, label)
    if state.get("found") and state.get("checked"):
        # Already checked but list missing — click once to re-expand if UI allows.
        # Prefer not toggling off: only click when unchecked.
        pass
    elif not (state.get("found") and state.get("checked")):
        try:
            await _click_exact_choice(page, label)
            await page.wait_for_timeout(400)
        except Exception:
            logger.exception(
                "recover special apply-to click failed call_id=%s label=%r",
                session.call_id,
                label,
            )
    return await _wait_for_special_apply_to_members(
        page, timeout_ms=5000, call_id=session.call_id
    )


async def _list_special_person_members(page: Page) -> list[dict[str, Any]]:
    """Return member checkbox rows (heading-scoped, then resilient fallback)."""
    discovered = await _discover_special_person_members(page)
    return list(discovered.get("members") or [])


_SPECIAL_MEMBER_CLICK_JS = """({ label, dupIndex, mode, condition }) => {
  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase()
    .replace(/\\.(?=\\s|$)/g, '');
  const target = norm(label);
  const soft = (s) => (s || '').replace(/\\s+/g, ' ').trim();
  const isCondition = (t) => /who does this apply to|someone is |someone has |someone was |none of the above/i.test(t || '');
  const isMultiName = (t) => {
    const hits = (t || '').match(/[A-Za-z][A-Za-z'\\-]*\\s+[A-Za-z]\\.?/g);
    return !!(hits && hits.length > 1);
  };
  const isPersonLike = (t) => {
    if (!t || t.length > 60) return false;
    if (isCondition(t) || isMultiName(t)) return false;
    if (/^me$/i.test(t)) return true;
    return /^[A-Za-z][A-Za-z .'\\-]{0,40}$/.test(t);
  };
  const nameMatches = (text) => {
    const n = norm(text);
    return n === target
      || n.startsWith(target + ' ')
      || target.startsWith(n)
      || n === target.replace(/\\s+/g, '');
  };
  const conditionKey = (value) => {
    const low = soft(value).toLowerCase();
    if (!low) return '';
    if (low.includes('cervical')) return 'cervical';
    if (low.includes('breast')) return 'breast';
    if (low.includes('blind')) return 'blind';
    if (low.includes('pregnant')) return 'pregnant';
    if (low.includes('foster')) return 'foster';
    if (low.includes('disabled') || low.includes('ssdi')) return 'disabled';
    return low;
  };
  const findConditionCard = (value) => {
    const key = conditionKey(value);
    if (!key) return null;
    const others = ['cervical', 'breast', 'blind', 'pregnant', 'foster', 'disabled']
      .filter((item) => item !== key);
    const seeds = [];
    for (const el of document.querySelectorAll(
      'label, button, [role=checkbox], [role=button], li, div, span'
    )) {
      const t = soft(el.innerText || el.textContent || '');
      if (!t || t.length > 140) continue;
      if (!/someone is |someone has |someone was /i.test(t)) continue;
      if (conditionKey(t) !== key) continue;
      seeds.push({ el, len: t.length });
    }
    seeds.sort((a, b) => a.len - b.len);
    for (const seed of seeds) {
      let cur = seed.el;
      for (let depth = 0; depth < 14 && cur; depth++) {
        const t = soft(cur.innerText || '').toLowerCase();
        if (
          t
          && t.includes(key)
          && /who does this apply to/i.test(t)
          && cur.querySelector('input[type=checkbox], [role=checkbox]')
        ) {
          return cur;
        }
        cur = cur.parentElement;
      }
    }
    const hits = [];
    for (const el of document.querySelectorAll('section, article, li, fieldset, div')) {
      const t = soft(el.innerText || '').toLowerCase();
      if (!t || t.length > 2500 || !t.includes(key)) continue;
      if (!/who does this apply to/i.test(t)) continue;
      if (!el.querySelector('input[type=checkbox], [role=checkbox]')) continue;
      const otherCount = others.filter((item) => t.includes(item)).length;
      hits.push({ el, len: t.length, otherCount });
    }
    hits.sort((a, b) => a.otherCount - b.otherCount || a.len - b.len);
    return hits.length ? hits[0].el : null;
  };
  const findRoot = () => {
    if (condition) return findConditionCard(condition);
    const headings = [...document.querySelectorAll(
      'h1,h2,h3,h4,div,span,p,legend,label,strong,b,[role=heading]'
    )];
    const hits = [];
    for (const el of headings) {
      const t = soft(el.innerText || el.textContent || '');
      if (!t || t.length > 80) continue;
      if (/who does this apply to/i.test(t)) hits.push({ el, len: t.length });
    }
    hits.sort((a, b) => a.len - b.len);
    if (hits.length) {
      const el = hits[0].el;
      let root = el.closest('section,fieldset,div,form') || el.parentElement;
      let cur = el.parentElement;
      for (let i = 0; i < 10 && cur; i++) {
        if (cur.querySelectorAll('input[type=checkbox], [role=checkbox]').length) {
          root = cur;
          break;
        }
        cur = cur.parentElement;
      }
      return root;
    }
    const open = [...document.querySelectorAll(
      '[data-state=open], [aria-expanded=true]'
    )].find((n) => /who does this apply to/i.test(soft(n.innerText || '')));
    return open || null;
  };
  const readChecked = (box) => !!(
    (box && box.tagName === 'INPUT' && box.checked)
    || (box && box.getAttribute && box.getAttribute('aria-checked') === 'true')
  );
  const checkboxForNameNode = (nameNode) => {
    let cur = nameNode;
    for (let depth = 0; depth < 8 && cur; depth++) {
      const box = cur.querySelector
        && cur.querySelector('input[type=checkbox], [role=checkbox]');
      if (box) return box;
      cur = cur.parentElement;
    }
    return null;
  };
  const root = findRoot();
  if (!root) return { ok: false, reason: 'condition_scope_missing' };
  const nameNodes = [...root.querySelectorAll(
    'div, span, p, label, h1, h2, h3, h4, [role=heading]'
  )].filter((node) => {
    const text = soft(
      node.innerText || node.textContent || node.getAttribute('aria-label') || ''
    );
    if (!isPersonLike(text) || !nameMatches(text)) return false;
    const childHit = [...node.querySelectorAll(
      'div, span, p, label, h1, h2, h3, h4'
    )].some((c) => {
      const ct = soft(c.innerText || c.textContent || '');
      return c !== node && isPersonLike(ct) && ct.length < text.length;
    });
    return !childHit;
  });
  if (!nameNodes.length) return { ok: false, reason: 'not_found' };
  if (dupIndex >= nameNodes.length) return { ok: false, reason: 'dup_oob' };
  const nameNode = nameNodes[dupIndex];
  const box = checkboxForNameNode(nameNode);
  if (!box) {
    return {
      ok: false,
      reason: 'no_checkbox',
      label: soft(nameNode.innerText || ''),
    };
  }
  const already = readChecked(box);
  if (mode === 'verify') {
    return {
      ok: already,
      label: soft(nameNode.innerText || ''),
      strategy: root === document.body ? 'pagewide' : 'scoped',
      already,
    };
  }
  if (!already) {
    box.click();
  }
  return {
    ok: true,
    clicked: !already,
    already,
    label: soft(nameNode.innerText || ''),
    strategy: root === document.body ? 'pagewide' : 'scoped',
  };
}"""


async def _click_special_person_member(
    page: Page, label: str, *, dup_index: int = 0, condition: str = ""
) -> bool:
    """Click the member's checkbox control (not the name text) and verify checked.

    Pointcare renders an empty ``button[role=checkbox]`` beside a nested name
    ``div`` (e.g. ``Mickey M.``). Clicking the name node alone does not toggle
    or expose ``aria-checked`` — always target the row checkbox. React updates
    ``aria-checked`` after paint, so verify after a short wait (never a second click).
    """
    click_arg = {
        "label": label,
        "dupIndex": int(dup_index),
        "mode": "click",
    }
    if condition:
        click_arg["condition"] = condition
    result = await page.evaluate(_SPECIAL_MEMBER_CLICK_JS, click_arg)
    if not result:
        return False
    # Back-compat for unit tests that mock a simple {ok: True} payload.
    if result.get("ok") and result.get("reason") is None and "clicked" not in result and "already" not in result:
        await page.wait_for_timeout(50)
        return True
    if not result.get("ok"):
        logger.warning(
            "special_person click missed label=%r reason=%s",
            label,
            result.get("reason"),
        )
        return False
    if result.get("already"):
        return True
    # Allow the SPA to commit aria-checked — do not click again.
    await page.wait_for_timeout(350)
    verify_arg = {
        "label": label,
        "dupIndex": int(dup_index),
        "mode": "verify",
    }
    if condition:
        verify_arg["condition"] = condition
    verify = await page.evaluate(_SPECIAL_MEMBER_CLICK_JS, verify_arg)
    if verify and verify.get("ok"):
        return True
    logger.warning(
        "special_person click unverified label=%r reason=%s strategy=%s",
        label,
        (verify or {}).get("reason"),
        (verify or result).get("strategy"),
    )
    return False


def _match_member_rows(
    spoken: str,
    members: list[dict[str, Any]],
    *,
    raw_answer: str = "",
) -> list[dict[str, Any]]:
    """Return member rows matching a spoken query (exact preferred)."""
    target_norm = _normalize_member_label(spoken)
    if not target_norm:
        return []
    exact = [
        m
        for m in members
        if _normalize_member_label(str(m.get("label") or "")) == target_norm
    ]
    if exact:
        return exact
    # "Mickey Mouse" → "Mickey M." (first name + last initial)
    parts = target_norm.split()
    if len(parts) >= 2 and len(parts[-1]) >= 1:
        first, last = parts[0], parts[-1]
        initial = last[0]
        initial_hits = [
            m
            for m in members
            if _normalize_member_label(str(m.get("label") or ""))
            in {f"{first} {initial}", f"{first}{initial}"}
            or re.fullmatch(
                rf"{re.escape(first)} {re.escape(initial)}\.?",
                _normalize_member_label(str(m.get("label") or "")),
            )
        ]
        if len(initial_hits) == 1:
            return initial_hits
        if len(initial_hits) > 1:
            return initial_hits
    # Prefix / containment (John → John D. / Murali → Murali M.)
    contains = [
        m
        for m in members
        if target_norm in _normalize_member_label(str(m.get("label") or ""))
        or _normalize_member_label(str(m.get("label") or "")).startswith(target_norm)
    ]
    if contains:
        return contains
    first = target_norm.split()[0]
    if first and len(first) >= 2:
        return [
            m
            for m in members
            if _normalize_member_label(str(m.get("label") or "")).startswith(first)
        ]
    return []


async def _assign_one_special_person_member(
    page: Page,
    spoken: str,
    members: list[dict[str, Any]],
    session: LiveFormSession,
    *,
    raw_answer: str = "",
    condition: str = "",
) -> str:
    """Select one household member; return the visible label selected."""
    kin = _resolve_kinship_to_member(spoken, members, session)
    if kin:
        spoken = kin

    matches = _match_member_rows(spoken, members, raw_answer=raw_answer)
    if not matches:
        labels = ", ".join(
            str(m.get("label") or "")
            for m in members[:8]
            if _is_plausible_member_label(str(m.get("label") or ""))
        )
        raise RuntimeError(
            "SPECIAL_PERSON_CLARIFY: Please say which person: "
            + (labels or "the household member names on the screen")
        )

    unique_labels = {
        _normalize_member_label(str(m.get("label") or "")) for m in matches
    }
    if len(matches) > 1 and len(unique_labels) == 1:
        ordinal = None
        if re.search(r"\b(first|1st)\b", raw_answer or spoken, re.I):
            ordinal = 0
        elif re.search(r"\b(second|2nd)\b", raw_answer or spoken, re.I):
            ordinal = 1
        elif re.search(r"\b(third|3rd)\b", raw_answer or spoken, re.I):
            ordinal = 2
        if ordinal is None:
            raise RuntimeError(
                "SPECIAL_PERSON_CLARIFY: There are two people named "
                f"{matches[0].get('label')}. Please say the first or the second."
            )
        if ordinal >= len(matches):
            raise RuntimeError(
                "SPECIAL_PERSON_CLARIFY: Please say the first or the second listed person."
            )
        chosen = matches[ordinal]
    elif len(matches) > 1:
        # Distinct people sharing a first name / vague query — clarify.
        labels = ", ".join(str(m.get("label") or "") for m in matches[:5])
        raise RuntimeError(
            "SPECIAL_PERSON_CLARIFY: Several people match. Please say one of: " + labels
        )
    else:
        chosen = matches[0]

    label = str(chosen.get("label") or "")
    dup_index = int(chosen.get("dupIndex") or 0)
    # Already checked — preserve selection (do not toggle off).
    if chosen.get("checked"):
        logger.info(
            "special_person already checked call_id=%s label=%r",
            session.call_id,
            label,
        )
        return label

    # One click only — never retry-toggle when a generic reader fails.
    if condition:
        ok = await _click_special_person_member(
            page, label, dup_index=dup_index, condition=condition
        )
    else:
        ok = await _click_special_person_member(page, label, dup_index=dup_index)
    if not ok:
        # Re-read only this condition's list. Never click another condition.
        if condition:
            refreshed = await _discover_special_person_members(page, condition)
        else:
            refreshed = await _discover_special_person_members(page)
        still = next(
            (
                m
                for m in (refreshed.get("members") or [])
                if _normalize_member_label(str(m.get("label") or ""))
                == _normalize_member_label(label)
                and bool(m.get("checked"))
            ),
            None,
        )
        if still:
            return label
        raise RuntimeError(
            f"SPECIAL_PERSON_CLARIFY: I couldn't select {label}. "
            "Please say the person's name again."
        )
    return label


async def _assign_special_person_member(
    page: Page, answer: str, session: LiveFormSession
) -> None:
    """Assign the selected special condition to one or more household members."""
    raw = (answer or "").strip()
    # A condition name is not a household member.
    if _utterance_is_only_condition(raw):
        raise RuntimeError(
            "SPECIAL_PERSON_CLARIFY: That selects the condition. "
            "Who does this apply to? Please say the household member's name."
        )

    queries = _split_member_queries(raw)
    if not queries:
        raise RuntimeError(
            "SPECIAL_PERSON_CLARIFY: Which household member does this apply to?"
        )

    targets = _special_person_target_conditions(raw, session)
    if not targets:
        # No condition was recorded. Keep the previous open-list assignment.
        targets = [""]
    discovered: dict[str, Any] = {}
    members: list[dict[str, Any]] = []
    strategy = ""
    # Assign inside each active condition card. Do not reuse another card's checkbox.
    selected_labels: list[str] = []
    name_query = _member_query_without_condition(raw)
    queries = _split_member_queries(name_query or raw)
    if not queries:
        raise RuntimeError(
            "SPECIAL_PERSON_CLARIFY: Which household member does this apply to?"
        )
    for condition in targets:
        discovered = await _wait_for_special_apply_to_members(
            page,
            timeout_ms=4000,
            call_id=session.call_id,
            condition=condition,
        )
        members = list(discovered.get("members") or [])
        strategy = str(discovered.get("strategy") or "")
        if not members:
            logger.warning(
                "special_person_scope_missing call_id=%s condition=%r strategy=%s spoken=%r",
                session.call_id,
                condition,
                strategy,
                (answer or "")[:40],
            )
            if strategy.startswith("wrong_screen") or discovered.get("has_marital"):
                raise RuntimeError(
                    "SPECIAL_PERSON_CLARIFY: This is not the Special Cases "
                    "member list. Please say the condition again."
                )
            if condition:
                raise RuntimeError(
                    "SPECIAL_PERSON_CLARIFY: I couldn't find the member list for "
                    f"{_short_special_condition_label(condition)}. "
                    "Please say the person's name again."
                )
            raise RuntimeError(
                "SPECIAL_PERSON_CLARIFY: I couldn't find the 'Who does this apply to?' "
                "member list. Please say the condition again."
            )
        for query in queries:
            label = await _assign_one_special_person_member(
                page,
                query,
                members,
                session,
                raw_answer=raw,
                condition=condition,
            )
            if label not in selected_labels:
                selected_labels.append(label)
    logger.info(
        "special_person assigned call_id=%s labels=%r spoken=%r conditions=%r",
        session.call_id,
        selected_labels,
        (answer or "")[:40],
        targets,
    )
    remembered = session.filled.get("special_person_assigned", "")
    bits = [x for x in remembered.split("|") if x]
    for label in selected_labels:
        if label not in bits:
            bits.append(label)
    session.filled["special_person_assigned"] = "|".join(bits)
    return


def _utterance_is_only_condition(raw: str) -> bool:
    """True when the caller named a condition and no household member."""
    text = re.sub(r"[.!?…]+$", "", (raw or "").strip().lower())
    if not text:
        return False
    if not re.search(
        r"\b(someone is|someone has|someone was|none of the above|cervical|"
        r"breast cancer|pregnant|blind|foster|ssdi|disabled)\b",
        text,
    ):
        return False
    condition_words = {
        "someone",
        "has",
        "have",
        "was",
        "were",
        "the",
        "above",
        "none",
        "cervical",
        "cancer",
        "breast",
        "pregnant",
        "pregnancy",
        "blind",
        "foster",
        "care",
        "ssdi",
        "disabled",
        "disability",
        "condition",
        "this",
        "that",
        "please",
        "said",
    }
    names = [
        word
        for word in re.findall(r"[a-z][a-z']{2,}", text)
        if word not in condition_words
    ]
    return not names


def _condition_key_in_utterance(raw: str) -> str:
    text = (raw or "").lower()
    for key, pattern in (
        ("cervical", r"\bcervical\b"),
        ("breast", r"\bbreast\b"),
        ("blind", r"\bblind\b"),
        ("pregnant", r"\b(pregnant|pregnancy)\b"),
        ("foster", r"\bfoster\b"),
        ("disabled", r"\b(disabled|ssdi)\b"),
    ):
        if re.search(pattern, text):
            return key
    return ""


def _member_query_without_condition(raw: str) -> str:
    text = re.sub(
        r"\b(someone is|someone has|someone was|cervical cancer|breast cancer|"
        r"cervical|pregnant|pregnancy|blind|foster care|ssdi|disabled)\b",
        " ",
        raw or "",
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip(" ,.")


def _special_person_target_conditions(raw: str, session: LiveFormSession) -> list[str]:
    active = [
        item
        for item in (session.filled.get("special_active_conditions") or "").split("|")
        if item.strip()
    ]
    if not active:
        active = [
            item.strip()
            for item in (session.filled.get("special_conditions") or "").split(",")
            if item.strip() and not re.search(r"none of the above", item, re.I)
        ]
    named = _condition_key_in_utterance(raw)
    if named:
        matched = [item for item in active if named in item.lower()]
        if matched:
            return matched
        return [named]
    return active


_TENS_WORDS = frozenset(
    {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
)
_UNIT_WORDS = frozenset(
    {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
)
_AGE_RANGE_CONNECTOR = re.compile(
    r"\b(to|through|thru|between|and)\b|[-–—]",
    re.I,
)


def _extract_age_numbers(text: str) -> list[int]:
    """Pull ages from digits and number words, joining compounds like 'twenty five'."""
    lowered = (text or "").lower()
    # Collapse "twenty-five" / "twenty five" into a single digit token first so
    # we never treat them as the separate ages 20 and 5.
    tens_alt = "|".join(sorted(_TENS_WORDS, key=len, reverse=True))
    units_alt = "|".join(sorted(_UNIT_WORDS, key=len, reverse=True))

    def _compound(match: re.Match[str]) -> str:
        return str(_NUMBER_WORDS[match.group(1)] + _NUMBER_WORDS[match.group(2)])

    lowered = re.sub(
        rf"\b({tens_alt})[-\s]+({units_alt})\b",
        _compound,
        lowered,
    )
    word_alt = "|".join(sorted((_NUMBER_WORDS.keys()), key=len, reverse=True))
    pattern = re.compile(rf"\b(\d{{1,3}}|{word_alt})\b", re.I)
    out: list[int] = []
    for match in pattern.finditer(lowered):
        token = match.group(1).lower()
        if token.isdigit():
            value = int(token)
            # Ignore years that look like DOB years.
            if value >= 1900:
                continue
            out.append(value)
        elif token in _NUMBER_WORDS:
            out.append(_NUMBER_WORDS[token])
    return out


def _spoken_has_age_range_connector(text: str) -> bool:
    return bool(_AGE_RANGE_CONNECTOR.search(text or ""))


_RELATIONSHIP_ONLY_AGE_RE = re.compile(
    r"^(?:"
    r"step[\s-]?child|grand[\s-]?child|foster[\s-]?child|"
    r"child|children|son|daughter|kid|kids|"
    r"spouse|partner|other"
    r")$",
    re.I,
)


def _is_relationship_only_age_utterance(spoken: str) -> bool:
    """True when the utterance is only a relationship word — not an age."""
    text = re.sub(r"[.!?…]+$", "", (spoken or "").strip().lower())
    text = re.sub(r"\b(my|our|the|a|an|their|his|her|is|are)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return bool(text) and bool(_RELATIONSHIP_ONLY_AGE_RE.match(text))


def _age_choice(answer: str) -> str:
    """Fallback label when visible chips are unknown (adult/spouse-style bands)."""
    lowered = (answer or "").lower()
    if _is_relationship_only_age_utterance(answer):
        return ""
    ages = _extract_age_numbers(lowered)
    if ages:
        age = ages[0] if len(ages) == 1 else max(ages)
        # Range like "sixteen to eighteen" → use high end for adult bands.
        if len(ages) >= 2 and max(ages) - min(ages) <= 20:
            age = max(ages)
        if age <= 18:
            return "(0-18) Child"
        if age <= 20:
            return "(19-20) Young Adult"
        if age <= 63:
            return "(21-63) Adult"
        if age == 64:
            return "(64) Turning Senior"
        return "(65+) Senior"
    aliases = (
        ("young adult", "(19-20) Young Adult"),
        ("turning senior", "(64) Turning Senior"),
        ("senior", "(65+) Senior"),
        ("adult", "(21-63) Adult"),
        ("baby", "(0-1) Baby"),
        ("toddler", "(1-5)"),
        # Never map bare "child" / relationship words to an age band.
    )
    for cue, label in aliases:
        if cue in lowered:
            return label
    return answer


def _map_age_label(spoken: str, labels: list[str]) -> str | None:
    """Map spoken age onto a visible age chip. None if ambiguous/unrelated.

    Never returns an arbitrary first option — callers must clarify on None.
    Never invents a chip that is not in the provided age-option list.
    Relationship answers like "Child" / "Stepchild" never select an age chip.
    """
    age_labels = _age_option_labels(labels)
    if not age_labels:
        return None
    if _is_relationship_only_age_utterance(spoken):
        return None
    text = (spoken or "").strip().lower()
    text = re.sub(r"\b(years?\s+old|years?|yrs?|age(?:d)?|i'?m|i am)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    # Exact / near-exact label match first.
    for label in age_labels:
        if label.lower() == text:
            return label
    for label in age_labels:
        compact_label = re.sub(r"[\s()]", "", label.lower())
        compact_text = re.sub(r"[\s()]", "", text)
        if compact_label and compact_label == compact_text:
            return label

    # Keyword aliases only when the word uniquely names that chip.
    # Do NOT use "child" — it collides with relationship answers.
    keyword_map = (
        (r"\bbab(?:y|ies)\b", r"\bbaby\b"),
        (r"\btoddler\b", r"\btoddler\b"),
        (r"\byoung adult\b", r"young adult"),
        (r"\bturning senior\b", r"turning senior"),
        (r"\bsenior\b", r"\bsenior\b"),
        (r"\bteen(?:ager)?\b", r"\bteen\b"),
        (r"\badult\b", r"\badult\b"),
    )
    for spoken_re, label_re in keyword_map:
        if re.search(spoken_re, text, re.I):
            hits = [label for label in age_labels if re.search(label_re, label, re.I)]
            # "adult" must not also match "young adult" unless uniquely intended.
            if spoken_re == r"\badult\b":
                hits = [
                    label
                    for label in hits
                    if not re.search(r"young adult", label, re.I)
                ] or hits
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                return None

    ages = _extract_age_numbers(text)
    if not ages:
        return None

    # Garbled ASR like "twenty twenty two thirty" has multiple ages without a
    # range connector — never guess a chip from the first pair.
    if len(ages) >= 2 and not _spoken_has_age_range_connector(text):
        # A single compound already joined above; leftover multi-number noise
        # without "to"/"through"/"between" must clarify.
        if len(ages) > 2 or (len(ages) == 2 and ages[0] == ages[1]):
            return None
        # Two distinct numbers without a connector is also ambiguous.
        return None

    if len(ages) >= 2:
        low, high = min(ages[0], ages[1]), max(ages[0], ages[1])
        exact: list[str] = []
        containing: list[str] = []
        for label in age_labels:
            lo, hi = _parse_range_bounds(label)
            if lo is None:
                continue
            if hi is not None and lo == low and hi == high:
                exact.append(label)
            elif hi is not None and lo <= low and high <= hi:
                containing.append(label)
            elif hi is None and low >= lo and high >= lo:
                containing.append(label)
        if len(exact) == 1:
            return exact[0]
        if not exact and len(containing) == 1:
            return containing[0]
        return None

    age = ages[0]
    if age < 0 or age > 120:
        return None
    matches: list[str] = []
    for label in age_labels:
        lo, hi = _parse_range_bounds(label)
        if lo is None:
            continue
        if hi is None and age >= lo:
            matches.append(label)
        elif hi is not None and lo <= age <= hi:
            matches.append(label)
    if len(matches) == 1:
        return matches[0]
    return None


def _age_clarify_message(step: str, labels: list[str]) -> str:
    question = ELIG_QUESTIONS.get(step) or "What is the age range?"
    age_labels = _age_option_labels(labels)
    if age_labels:
        shown = ", ".join(age_labels[:6])
        return f"{question} Please say one of: {shown}."
    return (
        f"{question} Please say the age as a number, like twenty five, "
        "or the age range shown on the form."
    )


def _yes_no_transition(step: str, yes: bool, landed: str) -> str:
    """Map a Yes/No click + actual Pointcare screen onto the next elig_step.

    Pointcare may skip intermediate screens (e.g. relatives No → income).
    Originating gate wins over a misclassified child_form for person-support Yes.
    """
    landed = (landed or "").strip()
    if yes:
        if step == "children":
            return "child_first"
        # student / disabled / relatives / support always open the person form.
        if step in ("student", "disabled", "relatives", "support"):
            return "person_first"
        if landed == "child_form":
            return "child_first"
        return "person_first"
    # No-path defaults, with relatives allowed to skip support → income.
    defaults = {
        "children": "student",
        "student": "disabled",
        "disabled": "relatives",
        "relatives": "support",
        "support": "income_when",
        "child_more": "student",
    }
    if step == "relatives":
        if landed == "income":
            return "income_when"
        if landed == "support":
            return "support"
        if landed and landed not in ("support", "income", "unknown", ""):
            raise RuntimeError(
                f"Unexpected screen after relatives No: {landed}"
            )
        return "support"
    if step == "support" and landed == "income":
        return "income_when"
    return defaults.get(step, step)


def _yes_no_acceptable_screens(step: str, yes: bool) -> tuple[str, ...]:
    if yes:
        return ("child_form", "person_form")
    mapping = {
        "children": ("student",),
        "student": ("disabled",),
        "disabled": ("relatives",),
        # Income may appear when Pointcare skips the support question.
        "relatives": ("support", "income"),
        "support": ("income",),
    }
    return mapping.get(step, ())


async def _click_continue_to(page: Page, screens: tuple[str, ...]) -> str:
    await _click_cta(page, "Continue", attempts=3)
    return await _wait_for_screen(page, screens)


async def _submit_eligibility_answer(
    call_id: str,
    session: LiveFormSession,
    page: Page,
    answer: str,
    started: float,
) -> dict[str, Any]:
    """Apply one answer according to server state and the actual Pointcare screen."""
    if not answer:
        raise RuntimeError("eligibility answer is empty")
    step = session.elig_step
    if step in ("proceed_application", "confirm_start_application"):
        from .app_stage import handle_application_gate

        return await handle_application_gate(call_id, session, page, answer)
    screen = await _eligibility_screen(page)
    expected_group = _step_group(step)
    if screen != "unknown" and screen != expected_group:
        # Do NOT treat mismatch as proof the caller's answer was applied.
        # Sync step to the browser and ask that question again.
        recovered = _screen_first_step(screen, session)
        session.elig_step = recovered
        from .elig_meta import format_question

        logger.warning(
            "Eligibility step mismatch call_id=%s server_step=%s browser_screen=%s "
            "recovered=%s answer_not_applied=1",
            call_id,
            step,
            screen,
            recovered,
        )
        return {
            "ok": False,
            "call_id": call_id,
            "field_id": "eligibility_answer",
            "value": answer,
            "error": "step_mismatch",
            "spoken": "ASK: " + format_question(recovered),
            "message": "Browser step differed; answer not applied.",
            "step": recovered,
            "screen": screen,
        }

    _mark(call_id, "browser_action", started)
    lowered = answer.lower()
    spoken_override: str | None = None

    if step == "marital":
        await _click_exact_choice(page, answer)
        if _is_partnered(answer):
            await _wait_for_screen(page, ("spouse_form",), timeout_s=18)
            await _clear_new_person_name_fields(page, session, kind="spouse")
            next_step = "spouse_first"
        else:
            await _wait_for_screen(page, ("children",), timeout_s=18)
            next_step = "children"
    elif step == "spouse_first":
        await _strict_fill(page, "First Name", answer)
        session.filled["spouse_first"] = answer
        _remember_household_name(session, answer)
        next_step = "spouse_last"
    elif step == "spouse_last":
        await _strict_fill(page, "Last Name", answer)
        session.filled["spouse_last"] = answer
        next_step = "spouse_role"
    elif step == "spouse_role":
        role = "Partner" if "partner" in lowered else "Spouse"
        await _ensure_relationship_choice_selected(page, role)
        session.filled["spouse_role"] = role
        next_step = "spouse_age"
    elif step == "spouse_age":
        if re.search(r"\b(19|20)\d{2}\b", answer):
            await _click_exact_choice(page, "Enter date of birth instead")
            dob = _normalize_dob(answer)
            await _strict_fill(page, "Date of birth (optional)", dob)
            session.filled["spouse_age"] = dob or answer
        else:
            age_labels = await _ensure_age_choices_visible(page)
            mapped = _map_age_label(answer, age_labels)
            if not mapped:
                raise RuntimeError(
                    "AGE_CLARIFY: " + _age_clarify_message(step, age_labels)
                )
            await _click_age_choice(page, mapped)
            session.filled["spouse_age"] = mapped
        next_step = "spouse_confirm"
    elif step == "spouse_confirm":
        if _is_yes(answer):
            await _assert_member_form_ready_before_continue(session, page, "spouse")
            await _click_continue_to(page, ("children",))
            next_step = "children"
        elif _is_no(answer):
            next_step = "spouse_confirm"
            spoken_override = _confirm_which_field_speech()
            _mark(call_id, "verified", started)
            return _elig_ok(
                call_id,
                session,
                answer,
                next_step,
                "Awaiting which spouse field to correct.",
                spoken_override=spoken_override,
                await_field_correction=True,
            )
        else:
            corrected = await _try_confirm_relationship_correction(
                session, page, "spouse", answer
            )
            if not corrected:
                corrected = await _try_confirm_age_correction(
                    session, page, "spouse", answer
                )
            if corrected:
                _mark(call_id, "verified", started)
                corrected["call_id"] = call_id
                corrected["field_id"] = "eligibility_answer"
                corrected["action"] = "verified"
                return corrected
            raise RuntimeError(
                "Please say Yes to confirm those spouse or partner details, "
                "No to change them, or say Spouse or Partner to correct the role."
            )
    elif step in ("children", "student", "disabled", "relatives", "support"):
        visible_labels = await _visible_choice_labels(page)
        if step == "support":
            if _is_no(answer):
                yes = False
            elif _is_yes(answer) or _map_support_yes_label(answer, visible_labels):
                yes = True
            else:
                raise RuntimeError(
                    "SUPPORT_CLARIFY: Please say No, or Yes and which situation fits."
                )
        elif _is_no(answer):
            yes = False
        elif _is_yes(answer):
            yes = True
        else:
            raise RuntimeError(f"Expected Yes or No for {step}, got {answer!r}")
        logger.info(
            "Eligibility yes/no call_id=%s step=%s answer=%r mapped=%s screen=%s",
            call_id,
            step,
            answer,
            yes,
            screen,
        )
        labels = {item.lower() for item in visible_labels}
        # Support screen has four long Yes options + No, not a plain Yes button.
        support_yes_labels = [
            item
            for item in visible_labels
            if item.lower().startswith("yes") and item.lower() != "yes"
        ]
        has_plain_yes_no = "yes" in labels and "no" in labels
        has_support_choices = step == "support" and (
            bool(support_yes_labels) or "no" in labels
        )
        # Missing chips is NOT proof the caller's answer was applied. Reconcile
        # only when the browser already left this Yes/No step AND the open screen
        # matches the expected transition for the requested answer.
        if not has_plain_yes_no and not has_support_choices:
            landed = await _eligibility_screen(page)
            selected = await _yes_no_selection_state(page)
            expected_landed = _yes_no_acceptable_screens(step, yes)
            if landed in expected_landed or (
                yes
                and landed in ("child_form", "person_form")
                and step in ("children", "student", "disabled", "relatives", "support")
            ):
                recovered = _screen_first_step(landed, session)
                # Originating gate wins: disabled/relatives/student/support Yes
                # must not become child_first just because the detector saw a
                # Child relationship chip.
                if step in ("student", "disabled", "relatives", "support") and yes:
                    if not session.after_person:
                        session.after_person = {
                            "student": "disabled",
                            "disabled": "relatives",
                            "relatives": "support",
                            "support": "income_when",
                        }.get(step, "income_when")
                    _set_person_kind_from_gate(session, step)
                    session.filled.pop("person_section_intro", None)
                    await _clear_new_person_name_fields(page, session, kind="person")
                    next_step = "person_first"
                elif landed == "child_form":
                    if not await _wait_for_child_form_ready(page, timeout_s=8):
                        raise RuntimeError(
                            "CHILD_FORM: The child details form is still opening. "
                            "Please wait a moment, then say the child's first name."
                        )
                    _ensure_child_instance(session)
                    await _clear_new_person_name_fields(page, session, kind="child")
                    next_step = "child_first"
                elif landed == "person_form":
                    if step != "children" and not session.after_person:
                        session.after_person = {
                            "student": "disabled",
                            "disabled": "relatives",
                            "relatives": "support",
                            "support": "income_when",
                        }.get(step, "income_when")
                    if step != "children":
                        _set_person_kind_from_gate(session, step)
                        session.filled.pop("person_section_intro", None)
                    await _clear_new_person_name_fields(page, session, kind="person")
                    next_step = "person_first"
                else:
                    next_step = recovered
                logger.info(
                    "Yes/No screen already advanced call_id=%s step=%s landed=%s "
                    "next=%s selected=%s requested=%s",
                    call_id,
                    step,
                    landed,
                    next_step,
                    selected,
                    "yes" if yes else "no",
                )
                _mark(call_id, "verified", started)
                return _elig_ok(
                    call_id,
                    session,
                    answer,
                    next_step,
                    "Browser already on the expected next screen for this answer.",
                )
            # Chips gone but screen does not match the requested answer — do not commit.
            raise RuntimeError(
                f"YESNO_VERIFY: I couldn't find Yes/No controls for {step}, and the "
                f"open screen ({landed or 'unknown'}) does not match a "
                f"{'Yes' if yes else 'No'} answer. Please say yes or no again."
            )
        if step == "support":
            await _click_support_choice(page, answer, yes, visible_labels)
        else:
            await _click_exact_choice(page, "Yes" if yes else "No")
            # Prefer an observable selected state before navigation; do not require
            # it when Pointcare navigates immediately without aria state.
            selected = await _yes_no_selection_state(page)
            want = "yes" if yes else "no"
            if selected in ("yes", "no") and selected != want:
                logger.warning(
                    "Yes/No selection mismatch call_id=%s step=%s want=%s got=%s",
                    call_id,
                    step,
                    want,
                    selected,
                )
                raise RuntimeError(
                    f"YESNO_VERIFY: I tried to select {want}, but the form shows "
                    f"{selected}. Please say yes or no clearly again."
                )
            logger.info(
                "Yes/No click verified call_id=%s step=%s want=%s selected=%s",
                call_id,
                step,
                want,
                selected,
            )
        if yes:
            if step == "children":
                # Require the real child form + First Name before advancing state.
                if not await _wait_for_child_form_ready(page, timeout_s=14):
                    landed = await _eligibility_screen(page)
                    raise RuntimeError(
                        "CHILD_FORM: I answered Yes, but the child details form "
                        f"did not open (screen={landed or 'unknown'}). "
                        "Please say Yes again, or tell me if the form looks different."
                    )
                _ensure_child_instance(session)
                await _clear_new_person_name_fields(page, session, kind="child")
                next_step = "child_first"
            else:
                try:
                    landed = await _wait_for_screen(
                        page, ("child_form", "person_form"), timeout_s=10
                    )
                except RuntimeError:
                    landed = await _eligibility_screen(page)
                    if landed not in ("child_form", "person_form"):
                        raise
                if step in ("student", "disabled", "relatives", "support"):
                    session.after_person = {
                        "student": "disabled",
                        "disabled": "relatives",
                        "relatives": "support",
                        "support": "income_when",
                    }.get(step, "income_when")
                    _set_person_kind_from_gate(session, step)
                    session.filled.pop("person_section_intro", None)
                elif landed != "child_form":
                    session.after_person = {
                        "student": "disabled",
                        "disabled": "relatives",
                        "relatives": "support",
                        "support": "income_when",
                    }.get(step, "income_when")
                    _set_person_kind_from_gate(session, step)
                    session.filled.pop("person_section_intro", None)
                next_step = _yes_no_transition(step, True, landed)
                if next_step in ("person_first", "child_first"):
                    await _clear_new_person_name_fields(
                        page,
                        session,
                        kind="child" if next_step == "child_first" else "person",
                    )
        else:
            acceptable = _yes_no_acceptable_screens(step, False)
            try:
                landed = await _wait_for_screen(page, acceptable, timeout_s=14)
            except RuntimeError:
                landed = await _eligibility_screen(page)
                if landed not in acceptable:
                    raise
            next_step = _yes_no_transition(step, False, landed)
    elif step == "child_first":
        if not await _name_field_visible(page, "First Name"):
            # Child form may still be mounting after Yes — wait, then reconcile.
            if not await _wait_for_child_form_ready(page, timeout_s=10):
                screen = await _eligibility_screen(page)
                if screen == "children":
                    session.elig_step = "children"
                    raise RuntimeError(
                        "CHILD_FORM: The child form is not open yet. "
                        "Please say Yes to the children question again."
                    )
                raise RuntimeError(
                    "CHILD_FORM: I can't see the child's First Name field yet "
                    f"(screen={screen or 'unknown'}). Please wait a moment and "
                    "say the first name again."
                )
        await _strict_fill(page, "First Name", answer)
        session.filled["child_first"] = answer
        _remember_household_name(session, answer)
        next_step = "child_last"
    elif step == "child_last":
        if not await _name_field_visible(page, "Last Name"):
            if not await _wait_for_name_field(page, "Last Name", timeout_s=8):
                raise RuntimeError(
                    "CHILD_FORM: I can't see the child's Last Name field on this screen."
                )
        await _strict_fill(page, "Last Name", answer)
        session.filled["child_last"] = answer
        next_step = "child_rel"
        spoken_override = await _relationship_speech_for_page(
            session, page, "child_rel"
        )
    elif step == "child_rel":
        if session.filled.get("rel_other_awaiting") == "child_rel":
            custom = await _fill_other_relationship_text(page, answer)
            session.filled["child_rel"] = "Other"
            session.filled["child_rel_other"] = custom
            session.filled.pop("rel_other_awaiting", None)
            next_step = "child_age"
        else:
            labels = await _visible_choice_labels(page)
            mapped = _map_relationship_label(answer, labels)
            if not mapped:
                opts = _relationship_option_labels(labels)
                raise RuntimeError(
                    "REL_CLARIFY: " + build_options_help_speech("child_rel", opts or labels)
                )
            other_prompt = await _apply_relationship_chip_for_kind(
                session, page, "child", mapped
            )
            if other_prompt:
                next_step = "child_rel"
                spoken_override = other_prompt
            else:
                next_step = "child_age"
    elif step == "child_age":
        age_labels = await _ensure_age_choices_visible(page)
        mapped = _map_age_label(answer, age_labels)
        if not mapped:
            raise RuntimeError("AGE_CLARIFY: " + _age_clarify_message(step, age_labels))
        await _click_age_choice(page, mapped)
        session.filled["child_age"] = mapped
        next_step = "child_confirm"
    elif step == "child_confirm":
        if _is_yes(answer):
            await _assert_member_form_ready_before_continue(session, page, "child")
            next_step = "child_more"
        elif _is_no(answer):
            next_step = "child_confirm"
            spoken_override = _confirm_which_field_speech()
            _mark(call_id, "verified", started)
            return _elig_ok(
                call_id,
                session,
                answer,
                next_step,
                "Awaiting which child field to correct.",
                spoken_override=spoken_override,
                await_field_correction=True,
            )
        else:
            corrected = await _try_confirm_relationship_correction(
                session, page, "child", answer
            )
            if not corrected:
                corrected = await _try_confirm_age_correction(
                    session, page, "child", answer
                )
            if corrected:
                _mark(call_id, "verified", started)
                corrected["call_id"] = call_id
                corrected["field_id"] = "eligibility_answer"
                corrected["action"] = "verified"
                return corrected
            raise RuntimeError(
                "Please say Yes to confirm those child details, No to change them, "
                "or say the corrected relationship."
            )
    elif step == "child_more":
        if _is_yes(answer):
            await _click_cta(page, "Add Another", attempts=3)
            await _wait_for_screen(page, ("child_form",))
            if not await _wait_for_child_form_ready(page, timeout_s=10):
                raise RuntimeError(
                    "CHILD_FORM: Add Another did not open a blank child form yet. "
                    "Please say Yes again in a moment."
                )
            # New blank child form: new instance id + first-name step.
            _bump_child_instance(session)
            session.elig_step = "child_first"
            await _clear_new_person_name_fields(page, session, kind="child")
            next_step = "child_first"
        else:
            before = await _eligibility_screen(page)
            try:
                await _assert_member_form_ready_before_continue(session, page, "child")
                await _click_continue_to(page, ("student",))
            except RuntimeError as exc:
                landed = await _eligibility_screen(page)
                hints = await _visible_validation_hints(page)
                if landed in ("child_form", before, "unknown", ""):
                    raise RuntimeError(
                        "CHILD_MORE_CLARIFY: I couldn't leave the child form after "
                        "Continue"
                        + (f" ({hints})" if hints else "")
                        + ". Please make sure the child's name, relationship, and "
                        "age are filled, then say No again — or say Yes to add "
                        "another child."
                    ) from exc
                raise
            next_step = "student"
    elif step == "person_first":
        await _strict_fill(page, "First Name", answer)
        session.filled["person_first"] = answer
        _remember_household_name(session, answer)
        next_step = "person_last"
    elif step == "person_last":
        await _strict_fill(page, "Last Name", answer)
        session.filled["person_last"] = answer
        next_step = "person_rel"
        spoken_override = await _relationship_speech_for_page(
            session, page, "person_rel"
        )
    elif step == "person_rel":
        if session.filled.get("rel_other_awaiting") == "person_rel":
            custom = await _fill_other_relationship_text(page, answer)
            session.filled["person_rel"] = "Other"
            session.filled["person_rel_other"] = custom
            session.filled.pop("rel_other_awaiting", None)
            next_step = "person_age"
        else:
            labels = await _visible_choice_labels(page)
            mapped = _map_relationship_label(answer, labels)
            if not mapped:
                # Free-text relationship fields may not use chips — fill as given.
                if any(
                    label.lower() == (answer or "").strip().lower() for label in labels
                ):
                    mapped = next(
                        label
                        for label in labels
                        if label.lower() == (answer or "").strip().lower()
                    )
                elif labels:
                    opts = _relationship_option_labels(labels) or labels[:6]
                    raise RuntimeError(
                        "REL_CLARIFY: "
                        + build_options_help_speech("person_rel", opts)
                    )
                else:
                    # No chips and no Other text box — clarify rather than invent.
                    if await _other_relationship_input_visible(page):
                        custom = await _fill_other_relationship_text(page, answer)
                        session.filled["person_rel"] = custom
                        session.filled["person_rel_other"] = custom
                        next_step = "person_age"
                        mapped = ""
                    else:
                        raise RuntimeError(
                            "REL_CLARIFY: "
                            + relationship_question_speech("person_rel", labels or None)
                        )
            if mapped:
                other_prompt = await _apply_relationship_chip_for_kind(
                    session, page, "person", mapped
                )
                if other_prompt:
                    next_step = "person_rel"
                    spoken_override = other_prompt
                else:
                    next_step = "person_age"
    elif step == "person_age":
        age_labels = await _ensure_age_choices_visible(page)
        mapped = _map_age_label(answer, age_labels)
        if not mapped:
            raise RuntimeError(
                "AGE_CLARIFY: " + _age_clarify_message(step, age_labels)
            )
        await _click_age_choice(page, mapped)
        session.filled["person_age"] = mapped
        next_step = "person_confirm"
    elif step == "person_confirm":
        if _is_yes(answer):
            await _assert_member_form_ready_before_continue(session, page, "person")
            target = session.after_person or "income_when"
            target_screen = "income" if target == "income_when" else target
            await _click_continue_to(page, (target_screen,))
            next_step = target
        elif _is_no(answer):
            next_step = "person_confirm"
            spoken_override = _confirm_which_field_speech()
            _mark(call_id, "verified", started)
            return _elig_ok(
                call_id,
                session,
                answer,
                next_step,
                "Awaiting which member field to correct.",
                spoken_override=spoken_override,
                await_field_correction=True,
            )
        else:
            corrected = await _try_confirm_relationship_correction(
                session, page, "person", answer
            )
            if not corrected:
                corrected = await _try_confirm_age_correction(
                    session, page, "person", answer
                )
            if corrected:
                _mark(call_id, "verified", started)
                corrected["call_id"] = call_id
                corrected["field_id"] = "eligibility_answer"
                corrected["action"] = "verified"
                return corrected
            raise RuntimeError(
                "Please say Yes to confirm those details, No to change them, "
                "or say the corrected relationship."
            )
    elif step == "income_when":
        period = await _select_and_verify_income_period(page, answer)
        session.filled["income_when"] = period
        session.filled.pop("income_range", None)
        session.elig_step = "income_range"
        next_step = "income_range"
    elif step == "income_range":
        # Same Pointcare screen as income_when — never browser-Back for period changes.
        if (
            answer == "__INCOME_PERIOD_RESELECT__"
            or _wants_income_period_reselect(answer)
        ):
            session.filled.pop("income_range", None)
            session.elig_step = "income_when"
            next_step = "income_when"
            spoken_override = ELIG_QUESTIONS["income_when"]
            _mark(call_id, "verified", started)
            return _elig_ok(
                call_id,
                session,
                answer,
                next_step,
                "Income period reselect on same page.",
                spoken_override=spoken_override,
            )
        switch_period = ""
        if isinstance(answer, str) and answer.startswith("__INCOME_PERIOD_SWITCH__:"):
            switch_period = answer.split(":", 1)[1].strip()
        else:
            switch_period = _detect_income_period_switch(answer)
        if switch_period:
            period = await _select_and_verify_income_period(page, switch_period)
            session.filled["income_when"] = period
            # Prior range may no longer fit the new period bands — ask again.
            session.filled.pop("income_range", None)
            session.elig_step = "income_range"
            next_step = "income_range"
            spoken_override = _income_range_question_for_period(period)
            _mark(call_id, "verified", started)
            return _elig_ok(
                call_id,
                session,
                period,
                next_step,
                f"Income period switched to {period} in place.",
                spoken_override=spoken_override,
            )
        if _income_amount_scale_unclear(answer):
            period_key = session.filled.get("income_when") or "Annual"
            raise RuntimeError(
                "INCOME_CLARIFY: "
                + _income_scale_clarify_speech(answer, period_key)
            )
        labels = await _visible_choice_labels(page)
        mapped = _map_income_range_label(answer, labels)
        if not mapped:
            # Dollar chips may need a scroll; retry after scrolling the main column.
            try:
                await page.mouse.wheel(0, 600)
                await page.wait_for_timeout(300)
            except Exception:
                pass
            labels = await _visible_choice_labels(page)
            mapped = _map_income_range_label(answer, labels)
        income_opts = _income_range_labels(labels)
        period_key = session.filled.get("income_when") or "Annual"
        if not mapped or mapped not in income_opts:
            raise RuntimeError(
                "INCOME_CLARIFY: "
                + _income_range_natural_clarify(answer, period=period_key)
            )
        await _click_exact_choice(page, mapped)
        # Verify the intended chip is selected before advancing.
        verified_range = False
        for _ in range(16):
            await page.wait_for_timeout(125)
            if await _choice_appears_selected(page, mapped):
                verified_range = True
                break
            try:
                screen_now = await _eligibility_screen(page)
                if screen_now == "special":
                    verified_range = True
                    break
            except Exception:
                pass
        if not verified_range:
            try:
                await _wait_for_screen(page, ("special",))
                verified_range = True
            except RuntimeError:
                raise RuntimeError(
                    "INCOME_CLARIFY: I couldn't confirm that income amount. "
                    + _income_range_natural_clarify(answer, period=period_key)
                )
        session.filled["income_range"] = mapped
        next_step = "special"
    elif step == "special":
        labels = await _visible_choice_labels(page)
        selected = _map_special_condition_labels(answer, labels)
        if not selected:
            # Soft scroll in case chips are below the fold, then retry.
            try:
                await page.mouse.wheel(0, 400)
                await page.wait_for_timeout(250)
            except Exception:
                pass
            labels = await _visible_choice_labels(page)
            selected = _map_special_condition_labels(answer, labels)
        if not selected:
            shown = ", ".join(
                label
                for label in labels
                if re.search(r"someone |none of the above", label, re.I)
            )[:200]
            raise RuntimeError(
                "SPECIAL_CLARIFY: Please say which condition applies. "
                + (f"Options include: {shown}." if shown else "Say blind, disabled, or none.")
            )
        for label in selected:
            await _select_special_condition_checkbox(
                page,
                label,
                call_id=call_id,
                multi=len(selected) > 1,
            )
        # Preserve prior selections when adding more conditions after special_more.
        prior = [
            item.strip()
            for item in (session.filled.get("special_conditions") or "").split(",")
            if item.strip()
        ]
        merged = list(prior)
        for label in selected:
            if label not in merged:
                merged.append(label)
        session.filled["special_conditions"] = ", ".join(merged)
        none_selected = len(selected) == 1 and re.search(
            r"none of the above", selected[0], re.I
        )
        if none_selected:
            session.filled.pop("special_person_queue", None)
            session.filled.pop("special_active_conditions", None)
            await _click_cta(page, "Next", attempts=3)
            await _wait_for_screen(page, ("programs",))
            next_step, spoken_override = await _enter_programs_step(session, page)
        else:
            # Ask who for one condition at a time — never assign under Blind
            # when Pregnant was also named in the same utterance.
            need_person = [
                label
                for label in selected
                if not re.search(r"none of the above", label, re.I)
            ]
            session.filled["special_person_queue"] = "|".join(need_person)
            current = need_person[0]
            session.filled["special_active_conditions"] = current
            ready = await _wait_for_special_apply_to_members(
                page, timeout_ms=3500, call_id=call_id, condition=current
            )
            if not list(ready.get("members") or []):
                await _recover_special_apply_to_section(page, session)
                ready = await _wait_for_special_apply_to_members(
                    page, timeout_ms=2000, call_id=call_id, condition=current
                )
            if not list(ready.get("members") or []):
                raise RuntimeError(
                    "SPECIAL_CLARIFY: I selected the condition, but the "
                    "'Who does this apply to?' member list did not open. "
                    "Please say the condition again."
                )
            session.filled["special_member_labels"] = "|".join(
                str(m.get("label") or "")
                for m in (ready.get("members") or [])[:12]
                if m.get("label")
            )
            next_step = "special_person"
            if len(need_person) > 1:
                spoken_override = (
                    f"Okay. Who does {_short_special_condition_label(current)} "
                    "apply to?"
                )
    elif step == "special_person":
        before_assigned = [
            item
            for item in (session.filled.get("special_person_assigned") or "").split("|")
            if item
        ]
        await _assign_special_person_member(page, answer, session)
        after_assigned = [
            item
            for item in (session.filled.get("special_person_assigned") or "").split("|")
            if item
        ]
        newly = [item for item in after_assigned if item not in before_assigned]
        if not newly:
            newly = after_assigned
        active = [
            item
            for item in (session.filled.get("special_active_conditions") or "").split("|")
            if item
        ]
        if not active:
            active = [
                item.strip()
                for item in (session.filled.get("special_conditions") or "").split(",")
                if item.strip()
            ]
        # Record only against the current active condition(s), not the whole queue.
        _record_special_assignment_pairs(session, active[:1] or active, newly)
        queue = [
            item
            for item in (session.filled.get("special_person_queue") or "").split("|")
            if item.strip()
        ]
        current = (active[0] if active else "").strip()
        remaining = [
            item
            for item in queue
            if item.strip().lower() != current.lower()
        ]
        if remaining:
            nxt = remaining[0]
            session.filled["special_person_queue"] = "|".join(remaining)
            session.filled["special_active_conditions"] = nxt
            # Ensure the next condition is checked and its apply-to list is open.
            try:
                await _select_special_condition_checkbox(
                    page, nxt, call_id=call_id, multi=True
                )
            except RuntimeError:
                # Already checked is fine; still need the member list.
                pass
            ready = await _wait_for_special_apply_to_members(
                page, timeout_ms=3500, call_id=call_id, condition=nxt
            )
            if not list(ready.get("members") or []):
                raise RuntimeError(
                    "SPECIAL_CLARIFY: I need the member list for "
                    f"{_short_special_condition_label(nxt)}. "
                    "Please say that condition again."
                )
            session.filled["special_member_labels"] = "|".join(
                str(m.get("label") or "")
                for m in (ready.get("members") or [])[:12]
                if m.get("label")
            )
            next_step = "special_person"
            spoken_override = (
                f"Okay. Who does {_short_special_condition_label(nxt)} apply to?"
            )
        else:
            session.filled.pop("special_person_queue", None)
            spoken_override = _special_more_confirm_speech(session)
            next_step = "special_more"
    elif step == "special_more":
        if _is_yes(answer):
            next_step = "special"
        elif _is_no(answer):
            await _verify_special_selections_before_next(page, session)
            if "pregnant" in (session.filled.get("special_conditions") or "").lower():
                try:
                    await _click_cta(page, "Add details", attempts=2)
                except Exception:
                    pass
                next_step = "babies_count"
            else:
                await _click_cta(page, "Next", attempts=3)
                await _wait_for_screen(page, ("programs",))
                next_step, spoken_override = await _enter_programs_step(session, page)
        else:
            raise RuntimeError(
                "SPECIAL_MORE_CLARIFY: "
                + _special_more_confirm_speech(session)
            )
    elif step == "babies_count":
        await _strict_fill(page, "Number of babies", answer)
        next_step = "due_date"
    elif step == "due_date":
        await _strict_fill(page, "Due date", _normalize_dob(answer))
        await _click_cta(page, "Done", attempts=2)
        await _click_cta(page, "Next", attempts=3)
        await _wait_for_screen(page, ("programs",))
        next_step, spoken_override = await _enter_programs_step(session, page)
    elif step == "programs":
        from .app_stage import handle_programs_decision

        # Ensure summary speech exists even if we resumed mid-flow.
        if not (session.filled.get("programs_permission_speech") or "").strip():
            await _enter_programs_step(session, page)
        return await handle_programs_decision(call_id, session, page, answer)
    else:
        raise RuntimeError(f"Unsupported eligibility step {step!r}")

    _mark(call_id, "verified", started)
    session.history.append(
        {
            "field_id": "eligibility_answer",
            "step": step,
            "value": answer,
            "action": "verified",
            "ok": True,
        }
    )
    # When advancing onto a relationship step, always announce live options.
    if (
        spoken_override is None
        and next_step in ("child_rel", "person_rel", "spouse_role")
    ):
        try:
            spoken_override = await _relationship_speech_for_page(
                session, page, next_step
            )
        except Exception:
            spoken_override = relationship_question_speech(next_step)
    result = _elig_ok(
        call_id,
        session,
        answer,
        next_step,
        f"Applied and verified eligibility step {step}.",
        spoken_override=spoken_override,
    )
    _mark(call_id, "next_question", started)
    _mark(call_id, "tool_response", started)
    return result


async def _fill_text(
    page: Page,
    labels: tuple[str, ...],
    placeholders: tuple[str, ...],
    input_types: tuple[str, ...],
    value: str,
) -> None:
    if "email" in input_types or any("email" in label for label in labels):
        box = page.locator("input[type='email']:visible").first
        try:
            if await box.count():
                await box.click(timeout=4000)
                await box.fill("", timeout=2000)
                await box.fill(value, timeout=8000)
                got = (await box.input_value()).strip()
                if value.lower() not in got.lower():
                    await box.press_sequentially(value, delay=25)
                return
        except Exception:
            logger.exception("email input fill failed")

    # 1) Explicit label association
    for label in labels:
        try:
            loc = page.get_by_label(re.compile(re.escape(label), re.I)).first
            if await loc.count():
                await loc.click(timeout=5000)
                await loc.fill(value, timeout=8000)
                return
        except Exception:
            pass

    # 2) Placeholder
    for ph in placeholders or labels:
        try:
            loc = page.get_by_placeholder(re.compile(re.escape(ph), re.I)).first
            if await loc.count():
                await loc.click(timeout=5000)
                await loc.fill(value, timeout=8000)
                return
        except Exception:
            pass

    # 3) Nearby text / Google-Form style listitem
    for label in labels:
        pattern = re.compile(re.escape(label), re.I)
        for container in (
            page.locator('div[role="listitem"]').filter(has_text=pattern).first,
            page.locator("label").filter(has_text=pattern).first,
            page.locator("div, section, form").filter(has_text=pattern).first,
        ):
            try:
                if not await container.count():
                    continue
                box = container.locator("input:visible, textarea:visible").first
                if await box.count():
                    await box.click(timeout=5000)
                    await box.fill(value, timeout=8000)
                    return
            except Exception:
                continue

    # 4) First matching input type still empty
    type_selector = ", ".join(
        f'input[type="{t}"]' for t in (input_types or ("text", "email", "tel", "password"))
    )
    if type_selector:
        inputs = page.locator(f"{type_selector}:visible")
        count = await inputs.count()
        for i in range(min(count, 12)):
            el = inputs.nth(i)
            try:
                current = (await el.input_value(timeout=1500)).strip()
            except Exception:
                current = ""
            if current:
                continue
            try:
                await el.click(timeout=4000)
                await el.fill(value, timeout=8000)
                return
            except Exception:
                continue

    raise RuntimeError(f"Could not find an input for labels={labels!r}")


_DOB_MONTHS: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_DOB_CARDINAL: dict[str, int] = {
    "zero": 0,
    "oh": 0,
    "o": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
}

_DOB_ORDINAL: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "twenty first": 21,
    "twenty-first": 21,
    "twenty second": 22,
    "twenty-second": 22,
    "twenty third": 23,
    "twenty-third": 23,
    "twenty fourth": 24,
    "twenty-fourth": 24,
    "twenty fifth": 25,
    "twenty-fifth": 25,
    "twenty sixth": 26,
    "twenty-sixth": 26,
    "twenty seventh": 27,
    "twenty-seventh": 27,
    "twenty eighth": 28,
    "twenty-eighth": 28,
    "twenty ninth": 29,
    "twenty-ninth": 29,
    "thirty first": 31,
    "thirty-first": 31,
}


def _dob_word_number(text: str) -> int | None:
    """Parse a small spoken number / ordinal (1–31) or None."""
    raw = re.sub(r"\s+", " ", (text or "").strip().lower().replace("-", " "))
    raw = re.sub(r"\b(the|a|an)\b", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw in _DOB_ORDINAL:
        return _DOB_ORDINAL[raw]
    if raw in _DOB_CARDINAL:
        return _DOB_CARDINAL[raw]
    parts = raw.split()
    # "zero four" / "oh four" → 4 (spoken leading zero for month/day)
    if len(parts) == 2 and parts[0] in ("zero", "oh", "o") and parts[1] in _DOB_CARDINAL:
        val = _DOB_CARDINAL[parts[1]]
        if 0 <= val <= 9:
            return val
    # "twenty two" / "twenty second"
    if len(parts) == 2 and parts[0] in ("twenty", "thirty"):
        base = _DOB_CARDINAL[parts[0]]
        if parts[1] in _DOB_ORDINAL:
            keyed = _DOB_ORDINAL.get(f"{parts[0]} {parts[1]}")
            if keyed is not None:
                return keyed
            ones = _DOB_ORDINAL[parts[1]]
            if ones < 10:
                return base + ones
        if parts[1] in _DOB_CARDINAL and _DOB_CARDINAL[parts[1]] < 10:
            return base + _DOB_CARDINAL[parts[1]]
    # Strip ordinal suffix: 22nd
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)", raw)
    if m:
        return int(m.group(1))
    return None


def _parse_spoken_mdy_word_sequence(
    lowered: str,
) -> tuple[int | None, int | None, int | None, str | None]:
    """Parse 'zero four twenty five twenty twenty two' → 04/25/2022."""
    # Only number-words / digits — no month names.
    if any(re.search(rf"\b{re.escape(name)}\b", lowered) for name in _DOB_MONTHS):
        return None, None, None, None
    words = re.findall(r"[a-z0-9]+", lowered)
    skip = {"is", "the", "a", "an", "of", "and", "month", "day", "date", "year"}
    words = [w for w in words if w not in skip]
    if len(words) < 3:
        return None, None, None, None
    # Pure digit tokens belong to the slash/space numeric path (ambiguity rules).
    if all(w.isdigit() for w in words):
        return None, None, None, None
    # Prefer year suffixes from the right so "twenty twenty two" wins over
    # a false "twenty five ..." year parse.
    for year_start in range(len(words) - 1, 0, -1):
        year = _dob_spoken_year(" ".join(words[year_start:]))
        if year is None:
            continue
        prefix = words[:year_start]
        for split in range(1, len(prefix)):
            month = _dob_word_number(" ".join(prefix[:split]))
            day = _dob_word_number(" ".join(prefix[split:]))
            if (
                month is not None
                and day is not None
                and 1 <= month <= 12
                and 1 <= day <= 31
                and _calendar_date_ok(month, day, year)
            ):
                return month, day, year, None
    return None, None, None, "incomplete"


def _dob_spoken_year(text: str) -> int | None:
    """Parse a spoken or numeric year (1900–2100)."""
    raw = re.sub(r"\s+", " ", (text or "").strip().lower().replace("-", " "))
    raw = re.sub(r"\b(the|year|of)\b", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return None
    if re.fullmatch(r"\d{4}", raw):
        return int(raw)
    # twenty twenty four → 2024
    m = re.fullmatch(
        r"(nineteen|twenty)\s+(\w+)(?:\s+(\w+))?",
        raw,
    )
    if m:
        century = 1900 if m.group(1) == "nineteen" else 2000
        tens = _DOB_CARDINAL.get(m.group(2))
        if tens is None:
            return None
        if m.group(3):
            ones = _DOB_CARDINAL.get(m.group(3))
            if ones is None or ones > 9:
                return None
            if tens in (20, 30) or tens < 10:
                # twenty four → 2000+20+4; twenty → need second word
                if tens >= 20:
                    return century + tens + ones
                return None
            return century + tens + ones
        if tens >= 10:
            return century + tens
        return None
    # two thousand twenty four
    m = re.fullmatch(
        r"two thousand(?: and)?(?:\s+(\w+))?(?:\s+(\w+))?",
        raw,
    )
    if m:
        year = 2000
        if m.group(1):
            a = _DOB_CARDINAL.get(m.group(1))
            if a is None:
                return None
            if m.group(2):
                b = _DOB_CARDINAL.get(m.group(2))
                if b is None or b > 9:
                    return None
                if a >= 20:
                    year += a + b
                else:
                    return None
            else:
                year += a
        return year
    return None


def _calendar_date_ok(month: int, day: int, year: int) -> bool:
    if year < 1900 or year > 2100 or month < 1 or month > 12 or day < 1 or day > 31:
        return False
    limits = {
        1: 31,
        2: 29,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    if day > limits[month]:
        return False
    if month == 2 and day == 29:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        if not leap:
            return False
    return True


def extract_dob_ymd(
    text: str,
    *,
    prefer_us_when_ambiguous: bool = True,
) -> tuple[int | None, int | None, int | None, str | None]:
    """Extract (month, day, year, status).

    ``status`` is None on success, ``\"ambiguous\"`` when both numeric parts are
    ≤12 and unequal, or ``\"incomplete\"`` / ``\"invalid\"`` when parsing fails.
    Never invents missing parts.
    """
    raw = (text or "").strip()
    if not raw:
        return None, None, None, "incomplete"
    # ASR often inserts periods: "Month is October. Day is twenty two."
    lowered = raw.lower().replace(",", " ").replace(".", " ").replace(";", " ")
    lowered = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    month_name = "|".join(sorted(_DOB_MONTHS, key=len, reverse=True))

    # ISO YYYY-MM-DD or YYYY/MM/DD
    iso = re.search(r"\b(19|20)\d{2}\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{1,2})\b", lowered)
    if iso:
        year = int(iso.group(0)[:4])
        rest = re.search(
            rf"{year}\s*[/-]\s*(\d{{1,2}})\s*[/-]\s*(\d{{1,2}})",
            lowered,
        )
        if rest:
            month, day = int(rest.group(1)), int(rest.group(2))
            if _calendar_date_ok(month, day, year):
                return month, day, year, None
            return None, None, None, "invalid"

    # "month is October, date/day 22, year 2024"
    if "month" in lowered and "year" in lowered:
        mon_m = re.search(rf"\bmonth(?:\s+is)?\s+({month_name})\b", lowered)
        day_m = re.search(
            r"\b(?:date|day)(?:\s+is)?\s+(.+?)\s+year\b",
            lowered,
        )
        year_m = re.search(r"\byear(?:\s+is)?\s+(.+)$", lowered)
        if mon_m and day_m and year_m:
            month = _DOB_MONTHS[mon_m.group(1)]
            day = _dob_word_number(day_m.group(1).strip())
            year = _dob_spoken_year(year_m.group(1).strip())
            if day is None or year is None:
                return None, None, None, "incomplete"
            if _calendar_date_ok(month, day, year):
                return month, day, year, None
            return None, None, None, "invalid"

    # Fully spoken numeric: "zero four twenty five twenty twenty two"
    spoken_seq = _parse_spoken_mdy_word_sequence(lowered)
    if spoken_seq[0] is not None:
        return spoken_seq
    if spoken_seq[3] == "incomplete" and not any(
        name in lowered for name in ("january", "february", "march", "april", "may",
                                     "june", "july", "august", "september", "october",
                                     "november", "december", "jan", "feb", "mar", "apr",
                                     "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    ):
        # Word-number utterance that failed — do not fall through to junk matches.
        pass

    # "the twenty-second of October, 2024" / "22nd of October 2024"
    of_month = re.search(
        rf"\b(?:the\s+)?(\d{{1,2}}(?:st|nd|rd|th)?|[a-z][a-z\s-]+?)\s+of\s+"
        rf"({month_name})\s*,?\s+(.+)$",
        lowered,
    )
    if of_month:
        day = _dob_word_number(of_month.group(1).strip())
        month = _DOB_MONTHS[of_month.group(2)]
        year = _dob_spoken_year(of_month.group(3).strip())
        if day is None or year is None:
            return None, None, None, "incomplete"
        if _calendar_date_ok(month, day, year):
            return month, day, year, None
        return None, None, None, "invalid"

    # Month name + day + year (October 22 2024 / October twenty-second twenty twenty-four)
    named = re.search(
        rf"\b({month_name})\s+(\d{{1,2}}(?:st|nd|rd|th)?|[a-z][a-z\s-]+?)"
        rf"(?:\s*,)?\s+(.+)$",
        lowered,
    )
    if named and "month" not in named.group(0):
        month = _DOB_MONTHS[named.group(1)]
        day = _dob_word_number(named.group(2).strip())
        year = _dob_spoken_year(named.group(3).strip())
        if day is not None and year is not None:
            if _calendar_date_ok(month, day, year):
                return month, day, year, None
            return None, None, None, "invalid"

    # Day + month name + year (22 October 2024)
    named = re.search(
        rf"\b(\d{{1,2}})\s+({month_name})\s+(\d{{4}})\b",
        lowered,
    )
    if named:
        day, month, year = (
            int(named.group(1)),
            _DOB_MONTHS[named.group(2)],
            int(named.group(3)),
        )
        if _calendar_date_ok(month, day, year):
            return month, day, year, None
        return None, None, None, "invalid"

    # Month name + day + year numeric (retry simpler)
    named = re.search(
        rf"\b({month_name})\s+(\d{{1,2}})\s+(\d{{4}})\b",
        lowered,
    )
    if named:
        month, day, year = (
            _DOB_MONTHS[named.group(1)],
            int(named.group(2)),
            int(named.group(3)),
        )
        if _calendar_date_ok(month, day, year):
            return month, day, year, None
        return None, None, None, "invalid"

    # Numeric A/B/YYYY or A B YYYY
    numeric = re.search(
        r"\b(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{4})\b",
        lowered,
    )
    if not numeric:
        numeric = re.search(r"\b(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b", lowered)
    if numeric:
        a, b, year = int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3))
        if a > 12 and b <= 12:
            day, month = a, b  # DD/MM/YYYY
        elif b > 12 and a <= 12:
            month, day = a, b  # MM/DD/YYYY
        elif a <= 12 and b <= 12:
            if a != b and not prefer_us_when_ambiguous:
                return None, None, None, "ambiguous"
            month, day = a, b  # equal parts, or Stage D US preference
        else:
            return None, None, None, "invalid"
        if _calendar_date_ok(month, day, year):
            return month, day, year, None
        return None, None, None, "invalid"

    return None, None, None, "incomplete"


def _normalize_dob(text: str) -> str:
    """Normalize a DOB utterance to MM/DD/YYYY when possible (Stage D compatible).

    Ambiguous numeric dates prefer U.S. month-first. Stage E should call
    ``extract_dob_ymd(..., prefer_us_when_ambiguous=False)`` for clarification.
    """
    month, day, year, status = extract_dob_ymd(text, prefer_us_when_ambiguous=True)
    if status is None and month and day and year:
        return f"{month:02d}/{day:02d}/{year:04d}"
    return (text or "").strip()


async def _click_answer(page: Page, value: str) -> None:
    """Click the on-screen button or choice that matches what the caller said."""
    spoken = (value or "").strip()
    if "@" in spoken:
        raise RuntimeError(f"email {spoken!r} is not a choice button")
    if re.search(r"\b(google|gmail)\b", spoken, re.I):
        await _click_cta(page, "Continue with Google", allow_sso=True, attempts=3)
        return
    if re.search(r"\bapple\b", spoken, re.I):
        await _click_cta(page, "Continue with Apple", allow_sso=True, attempts=3)
        return
    exact = {
        "continue": "Continue",
        "next": "Next",
        "skip": "Skip",
        "yes": "Yes",
        "no": "No",
    }.get(spoken.lower())
    if exact:
        try:
            await _click_cta(page, exact, attempts=3)
            return
        except Exception:
            pass
    await _click_choice(page, spoken)


def _mark(call_id: str, stage: str, started: float) -> None:
    logger.info(
        "latency call_id=%s stage=%s t_ms=%d",
        call_id,
        stage,
        int((time.perf_counter() - started) * 1000),
    )


def _answer_kind(field_key: str, text: str) -> str:
    lowered = (text or "").strip().lower()
    if re.match(r"^(yes|yeah|yep|yup|no|nope|nah)\b", lowered) and len(lowered.split()) <= 6:
        return "yesno"
    if field_key == "dob" or re.search(r"\b(19|20)\d{2}\b", text or ""):
        return "date"
    if "first" in field_key and "name" in field_key:
        return "first_name"
    if "last" in field_key and "name" in field_key:
        return "last_name"
    # Never treat an email address as a choice chip / button label.
    if field_key == "email" or "@" in (text or ""):
        return "email_text"
    if field_key in ("household_income", "special_conditions") or re.search(
        r"\$|\byearly\b|\bmonthly\b|\bannual\b|every two weeks", lowered
    ):
        return "income"
    if len(lowered.split()) > 10:
        return "ambiguous"
    return "choice"


async def _verify_update(
    page: Page,
    before_url: str,
    value: str,
    kind: str,
    before_screen: str = "",
    before_choice: str = "",
) -> bool:
    if kind in ("first_name", "last_name", "date"):
        return await page.evaluate(
            """(value) => {
              const want = (value || '').trim().toLowerCase();
              if (!want) return false;
              return [...document.querySelectorAll('input, textarea')].some((el) =>
                (el.value || '').trim().toLowerCase().includes(want)
              );
            }""",
            value,
        )
    phrase = _choice_phrase(value).lower()
    if not before_screen:
        before_screen = await _screen_signature(page)
    for _ in range(40):
        if (page.url or "") != (before_url or ""):
            return True
        if await _screen_signature(page) != before_screen:
            return True
        if before_choice and await _choice_signature(page, value) != before_choice:
            return True
        pressed = await page.evaluate(
            """(phrase) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              return [...document.querySelectorAll('button, [role=button], [role=radio], [role=option]')].some((el) => {
                if (norm(el.innerText) !== phrase) return false;
                const aria = (el.getAttribute('aria-pressed') || el.getAttribute('aria-checked') || '').toLowerCase();
                return aria === 'true';
              });
            }""",
            phrase,
        )
        if pressed:
            return True
        try:
            await page.wait_for_timeout(250)
        except Exception:
            return False
    return False


def _is_none_of_the_above_label(label: str) -> bool:
    return bool(re.search(r"none of the above", label or "", re.I))


async def _special_condition_checkbox_state(
    page: Page, label: str
) -> dict[str, Any]:
    """Read Special Cases condition state from the real option DOM (special-only).

    Pointcare condition cards often do not expose checked/aria on the clicked
    outer text node. Walk ancestors for nested input[type=checkbox], aria, or
    selected classes. Prefer short exact condition chips — never match household
    member rows or the whole "Who does this apply to?" container.
    """
    phrase = (label or "").strip()
    if not phrase:
        return {"found": False, "checked": False, "source": "empty"}
    try:
        raw = await page.evaluate(
            """(phrase) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const want = norm(phrase);
              const textOf = (el) => norm(
                el.innerText || el.textContent || el.getAttribute('aria-label') || ''
              );
              const isPersonLike = (t) => {
                if (!t || t.length > 60) return false;
                if (/someone is |someone has |someone was |none of the above|who does this apply/i.test(t)) {
                  return false;
                }
                return /^[a-z][a-z .'\\-]{0,40}$/i.test(t);
              };
              const nodes = [...document.querySelectorAll(
                'label, [role=checkbox], button, [role=button], li, div, span'
              )];
              const candidates = [];
              for (const el of nodes) {
                const text = textOf(el);
                if (!text) continue;
                if (isPersonLike(text)) continue;
                if (text !== want && !text.startsWith(want)) continue;
                // Skip giant wrappers that include apply-to + member names.
                if (text.length > want.length + 48) continue;
                if (/who does this apply to/i.test(text)) continue;
                candidates.push(el);
              }
              candidates.sort((a, b) => textOf(a).length - textOf(b).length);

              const readInput = (root) => {
                if (!root) return null;
                if (root.tagName === 'INPUT' && root.type === 'checkbox') return root;
                if (root.tagName === 'LABEL' && root.control
                    && root.control.type === 'checkbox') {
                  return root.control;
                }
                if (root.querySelector) {
                  const nested = root.querySelector('input[type=checkbox]');
                  if (nested) return nested;
                }
                if (root.previousElementSibling
                    && root.previousElementSibling.matches
                    && root.previousElementSibling.matches('input[type=checkbox]')) {
                  return root.previousElementSibling;
                }
                return null;
              };

              for (const el of candidates) {
                let cur = el;
                for (let depth = 0; depth < 8 && cur; depth++) {
                  const input = readInput(cur);
                  if (input) {
                    return {
                      found: true,
                      checked: !!input.checked,
                      source: 'input',
                    };
                  }
                  const aria = cur.getAttribute && cur.getAttribute('aria-checked');
                  if (aria === 'true' || aria === 'false') {
                    return {
                      found: true,
                      checked: aria === 'true',
                      source: 'aria',
                    };
                  }
                  const cls = (cur.className || '').toString().toLowerCase();
                  if (/\\b(selected|checked|active|pressed)\\b/.test(cls)) {
                    return { found: true, checked: true, source: 'class' };
                  }
                  // Condition-scoped apply-to expansion under the same card.
                  const block = (cur.innerText || '').toLowerCase();
                  if (
                    block.includes(want)
                    && /who does this apply to/.test(block)
                    && !/none of the above/.test(want)
                  ) {
                    return {
                      found: true,
                      checked: true,
                      source: 'apply_to_near',
                    };
                  }
                  cur = cur.parentElement;
                }
              }
              return { found: false, checked: false, source: 'missing' };
            }""",
            phrase,
        )
        if isinstance(raw, dict):
            return {
                "found": bool(raw.get("found")),
                "checked": bool(raw.get("checked")),
                "source": str(raw.get("source") or "missing"),
            }
    except Exception:
        logger.exception("special checkbox state read failed label=%r", phrase[:60])
    return {"found": False, "checked": False, "source": "error"}


async def _special_next_enabled(page: Page) -> bool | None:
    """Read-only: whether the Special Cases Next CTA is enabled. None if unknown."""
    try:
        raw = await page.evaluate(
            """() => {
              const nodes = [...document.querySelectorAll(
                'button, [role=button], a, input[type=button], input[type=submit]'
              )];
              for (const el of nodes) {
                const text = (el.innerText || el.textContent || el.value || '')
                  .replace(/\\s+/g, ' ').trim();
                if (!/^next$/i.test(text)) continue;
                if (el.disabled) return false;
                if (el.getAttribute('aria-disabled') === 'true') return false;
                const cls = (el.className || '').toLowerCase();
                if (/\\bdisabled\\b/.test(cls)) return false;
                return true;
              }
              return null;
            }"""
        )
        if raw is None:
            return None
        return bool(raw)
    except Exception:
        return None


async def _special_apply_to_member_count(page: Page) -> int:
    """How many household-member checkboxes are currently discoverable."""
    try:
        discovered = await _discover_special_person_members(page)
        return len(list(discovered.get("members") or []))
    except Exception:
        return 0


async def _special_apply_to_heading_visible(page: Page) -> bool:
    try:
        return bool(
            await page.evaluate(
                """() => /who does this apply to/i.test(
                  (document.body && document.body.innerText) || ''
                )"""
            )
        )
    except Exception:
        return False


async def _special_condition_readonly_reinspect(
    page: Page, label: str
) -> dict[str, Any]:
    """Bounded read-only re-check of the condition card — never clicks.

    Walks every checkbox / role=checkbox and resolves its accessible name so we
    can verify selection when the outer card text node has no checked state.
    """
    phrase = (label or "").strip()
    if not phrase:
        return {"found": False, "checked": False, "source": "empty"}
    try:
        raw = await page.evaluate(
            """(phrase) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const want = norm(phrase);
              const isConditionName = (t) =>
                /someone is |someone has |someone was |none of the above/.test(t)
                || t === want
                || t.startsWith(want);
              const isPersonOnly = (t) =>
                !!t
                && t.length < 50
                && !isConditionName(t)
                && /^[a-z][a-z .'\\-]{0,40}$/i.test(t);

              const boxes = [...document.querySelectorAll(
                'input[type=checkbox], [role=checkbox]'
              )];
              for (const box of boxes) {
                let name = '';
                const aria = box.getAttribute('aria-label');
                if (aria) name = aria;
                else if (box.id) {
                  const lab = document.querySelector(
                    'label[for=\"' + box.id.replace(/\"/g, '') + '\"]'
                  );
                  if (lab) name = lab.innerText || lab.textContent || '';
                }
                if (!name) {
                  const wrap = box.closest('label');
                  if (wrap) name = wrap.innerText || wrap.textContent || '';
                }
                if (!name && box.parentElement) {
                  name = box.parentElement.innerText || box.parentElement.textContent || '';
                }
                const n = norm(name);
                if (!n || isPersonOnly(n)) continue;
                // Prefer short names; allow slightly longer when want is the lead text.
                const lead = n.slice(0, Math.max(want.length + 24, 80));
                if (n !== want && !n.startsWith(want) && !lead.startsWith(want)) {
                  continue;
                }
                if (!isConditionName(n) && n !== want && !n.startsWith(want)) {
                  continue;
                }
                const checked = !!(
                  box.checked
                  || box.getAttribute('aria-checked') === 'true'
                );
                return {
                  found: true,
                  checked,
                  source: 'reinspect_checkbox',
                };
              }
              return { found: false, checked: false, source: 'reinspect_missing' };
            }""",
            phrase,
        )
        if isinstance(raw, dict):
            return {
                "found": bool(raw.get("found")),
                "checked": bool(raw.get("checked")),
                "source": str(raw.get("source") or "reinspect_missing"),
            }
    except Exception:
        logger.exception(
            "special condition readonly reinspect failed label=%r", phrase[:60]
        )
    return {"found": False, "checked": False, "source": "reinspect_error"}


async def _condition_scoped_members_ready(
    page: Page, label: str
) -> dict[str, Any] | None:
    """Return discovery result when THIS condition's member list is visible."""
    if _is_none_of_the_above_label(label):
        return None
    discovered = await _discover_special_person_members(page, label)
    members = list(discovered.get("members") or [])
    strategy = str(discovered.get("strategy") or "")
    if members and strategy in {"condition_scope", "legacy_list"}:
        return discovered
    return None


async def _select_special_condition_checkbox(
    page: Page,
    label: str,
    *,
    call_id: str,
    multi: bool,
) -> None:
    """Select one Special Cases option with at most one click — never toggle off.

    Isolated to the special step. Does not modify shared choice helpers.
    Accept only when THIS condition's checkbox is verified or its own apply-to
    member section is discoverable. Never invent success without evidence.
    Do not require the page-wide member count to increase for a second condition.
    """
    is_none = _is_none_of_the_above_label(label)
    state = await _special_condition_checkbox_state(page, label)
    if state.get("found") and state.get("checked"):
        logger.info(
            "special condition already checked call_id=%s label=%r source=%s",
            call_id,
            label,
            state.get("source"),
        )
        return

    before_members = await _special_apply_to_member_count(page)
    started = time.time()
    await _click_exact_choice(page, label)

    # Poll briefly for condition-specific evidence — never click a second time.
    last_after: dict[str, Any] = {"found": False, "checked": False, "source": "pending"}
    reinspect: dict[str, Any] = {"found": False, "checked": False, "source": ""}
    for attempt in range(10):
        last_after = await _special_condition_checkbox_state(page, label)
        if last_after.get("found") and last_after.get("checked"):
            logger.info(
                "special condition checked call_id=%s label=%r source=%s "
                "wait_ms=%.0f attempt=%s",
                call_id,
                label,
                last_after.get("source"),
                (time.time() - started) * 1000,
                attempt,
            )
            return
        # Multi (or unreadable state): accept THIS condition's scoped member list
        # even when page-wide member count did not increase.
        if not is_none and (multi or not last_after.get("found")):
            scoped = await _condition_scoped_members_ready(page, label)
            if scoped is not None:
                logger.info(
                    "special condition accepted via scoped members call_id=%s "
                    "label=%r strategy=%s count=%s multi=%s wait_ms=%.0f",
                    call_id,
                    label,
                    scoped.get("strategy"),
                    len(scoped.get("members") or []),
                    multi,
                    (time.time() - started) * 1000,
                )
                return
        if attempt in (2, 5):
            reinspect = await _special_condition_readonly_reinspect(page, label)
            if reinspect.get("found") and reinspect.get("checked"):
                logger.info(
                    "special condition checked via reinspect call_id=%s label=%r "
                    "source=%s",
                    call_id,
                    label,
                    reinspect.get("source"),
                )
                return
        if last_after.get("found") and not last_after.get("checked") and attempt >= 2:
            # Stable unchecked after one click — do NOT click again.
            break
        # Single-select: page-wide apply-to growth is enough once state is unread.
        if not is_none and not multi and not last_after.get("found"):
            after_members = await _special_apply_to_member_count(page)
            apply_heading = await _special_apply_to_heading_visible(page)
            if after_members > before_members or (
                apply_heading and after_members > 0
            ):
                logger.info(
                    "special condition accepted via apply-to UI call_id=%s label=%r "
                    "members_before=%s members_after=%s heading=%s attempt=%s",
                    call_id,
                    label,
                    before_members,
                    after_members,
                    apply_heading,
                    attempt,
                )
                return
        try:
            await page.wait_for_timeout(100)
        except Exception:
            break

    if last_after.get("found") and not last_after.get("checked"):
        raise RuntimeError(
            f"SPECIAL_CLARIFY: I couldn't confirm {label}. "
            "Please say the condition again."
        )
    if reinspect.get("found") and not reinspect.get("checked"):
        raise RuntimeError(
            f"SPECIAL_CLARIFY: I couldn't confirm {label}. "
            "Please say the condition again."
        )

    after_members = await _special_apply_to_member_count(page)
    apply_heading = await _special_apply_to_heading_visible(page)
    # Single-select only: page-wide apply-to growth is enough when checkbox
    # state is unreadable. Multi must use THIS condition's scoped evidence above.
    if not is_none and not multi and (
        after_members > before_members
        or (apply_heading and after_members > 0)
    ):
        logger.info(
            "special condition accepted via apply-to UI call_id=%s label=%r "
            "members_before=%s members_after=%s heading=%s",
            call_id,
            label,
            before_members,
            after_members,
            apply_heading,
        )
        return

    if is_none:
        next_on = await _special_next_enabled(page)
        if next_on is True:
            logger.info(
                "special none accepted via Next enabled call_id=%s",
                call_id,
            )
            return
        others_checked = False
        try:
            labels_now = await _visible_choice_labels(page)
            for other in labels_now:
                if _is_none_of_the_above_label(other):
                    continue
                if not re.search(
                    r"someone is |someone has |someone was ", other, re.I
                ):
                    continue
                st = await _special_condition_checkbox_state(page, other)
                if st.get("found") and st.get("checked"):
                    others_checked = True
                    break
        except Exception:
            others_checked = True  # fail closed → ask again
        if not others_checked:
            logger.info(
                "special none accepted via no other conditions checked call_id=%s",
                call_id,
            )
            return
        raise RuntimeError(
            "SPECIAL_CLARIFY: I couldn't confirm None of the above. "
            "Please say none of the above again."
        )

    logger.warning(
        "special condition inconclusive after click call_id=%s label=%r "
        "multi=%s members_after=%s state=%s wait_ms=%.0f",
        call_id,
        label,
        multi,
        after_members,
        last_after.get("source"),
        (time.time() - started) * 1000,
    )
    raise RuntimeError(
        f"SPECIAL_CLARIFY: I couldn't confirm {label}. "
        "Please say the condition again."
    )


async def _choice_appears_selected(page: Page, value: str) -> bool:
    """True when the named choice/checkbox looks already selected.

    Requires an exact label match on the leaf control. Parent wrappers often
    concatenate every chip (``Child Stepchild …``); ``startsWith('child')`` on
    those wrappers plus a broad ``active`` class caused false positives so the
    agent skipped the real click and still said the relationship was updated.
    """
    phrase = _choice_phrase(value)
    try:
        return bool(
            await page.evaluate(
                """(phrase) => {
                  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                  const want = norm(phrase);
                  if (!want) return false;
                  const nodes = [...document.querySelectorAll(
                    'button, [role=button], [role=radio], [role=option], [role=checkbox], label, input'
                  )];
                  for (const el of nodes) {
                    const text = norm(
                      el.innerText || el.textContent || el.getAttribute('aria-label') || el.value || ''
                    );
                    // Exact label only — never startsWith on multi-chip wrappers.
                    if (text !== want) continue;
                    if (el.getAttribute('aria-checked') === 'true') return true;
                    if (el.getAttribute('aria-pressed') === 'true') return true;
                    if (el.getAttribute('aria-selected') === 'true') return true;
                    if (el.checked) return true;
                    const cls = String(el.className || '').toLowerCase();
                    // Prefer explicit selected/checked tokens; bare "active" alone is
                    // common on layout chrome and is only trusted with aria/checked.
                    if (/(?:^|\\s)(?:selected|checked|is-selected|is-checked)(?:\\s|$)/.test(cls)) {
                      return true;
                    }
                    if (/(?:^|\\s)(?:active|pressed)(?:\\s|$)/.test(cls)) {
                      const aria = (
                        el.getAttribute('aria-checked')
                        || el.getAttribute('aria-pressed')
                        || el.getAttribute('aria-selected')
                        || ''
                      ).toLowerCase();
                      // Some Pointcare chips only flip a visual class with no aria.
                      // Accept active/pressed on an exact-label leaf control.
                      if (aria !== 'false') return true;
                    }
                    const input = el.control || (el.querySelector && el.querySelector(
                      'input[type=checkbox], input[type=radio]'
                    ));
                    if (input && input.checked) return true;
                  }
                  return false;
                }""",
                phrase,
            )
        )
    except Exception:
        return False


async def _income_period_selected(page: Page, period: str) -> bool:
    """True when the income-period tab looks selected (aria/class/pressed)."""
    phrase = (period or "").strip()
    if not phrase:
        return False
    try:
        return bool(
            await page.evaluate(
                """(phrase) => {
                  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                  const want = norm(phrase);
                  const nodes = [...document.querySelectorAll(
                    'button, [role=button], [role=radio], [role=tab], [role=option]'
                  )];
                  const el = nodes.find((node) => norm(node.innerText) === want);
                  if (!el) return false;
                  const aria = (el.getAttribute('aria-checked')
                    || el.getAttribute('aria-pressed')
                    || el.getAttribute('aria-selected')
                    || '').toLowerCase();
                  if (aria === 'true') return true;
                  const cls = (el.className || '').toLowerCase();
                  if (/\\b(selected|active|checked|pressed|is-selected)\\b/.test(cls)) {
                    return true;
                  }
                  return false;
                }""",
                phrase,
            )
        )
    except Exception:
        return False


async def _select_and_verify_income_period(
    page: Page,
    spoken: str,
    *,
    labels: list[str] | None = None,
) -> str:
    """Select Annual/Monthly/Bi-weekly on the current income page (no browser Back).

    Returns the visible period label that was verified. Raises INCOME_CLARIFY
    when the period cannot be resolved or verified.
    """
    visible = labels if labels is not None else await _visible_choice_labels(page)
    period = _resolve_income_period_label(spoken, visible)
    if not period:
        canonical = _income_period_choice(spoken)
        if canonical in {"Annual", "Monthly", "Bi-weekly"} and not any(
            _looks_like_income_period_label(item) for item in visible
        ):
            period = canonical
        else:
            shown = ", ".join(
                item for item in visible if _looks_like_income_period_label(item)
            ) or "Annual, Monthly, or Bi-weekly"
            raise RuntimeError(
                "INCOME_CLARIFY: Please say which income period you know best — "
                f"{shown}."
            )
    before_sig = await _choice_signature(page, period)
    before_ranges = _income_range_labels(visible)
    await _click_exact_choice(page, period)
    verified = False
    for _ in range(24):
        await page.wait_for_timeout(125)
        after_sig = await _choice_signature(page, period)
        if after_sig != before_sig and after_sig not in ("", "__missing__"):
            verified = True
            break
        after_labels = await _visible_choice_labels(page)
        after_ranges = _income_range_labels(after_labels)
        # Selecting a period refreshes the dollar-range chips.
        if after_ranges and after_ranges != before_ranges:
            verified = True
            break
        if await _income_period_selected(page, period):
            verified = True
            break
    if not verified:
        raise RuntimeError(
            "INCOME_CLARIFY: I couldn't confirm that income period. "
            "Please say Annual, Monthly, or Bi-weekly."
        )
    return period


async def _choice_signature(page: Page, value: str) -> str:
    """Capture choice state so React class/style changes count as verification."""
    phrase = _choice_phrase(value)
    try:
        return await page.evaluate(
            """(phrase) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const want = norm(phrase);
              const nodes = [...document.querySelectorAll(
                'button, [role=button], [role=radio], [role=option], [role=checkbox]'
              )];
              const el = nodes.find((node) => norm(node.innerText) === want);
              if (!el) return '__missing__';
              return [
                el.className || '',
                el.getAttribute('style') || '',
                el.getAttribute('aria-checked') || '',
                el.getAttribute('aria-pressed') || '',
                String(el.checked || false),
              ].join('|');
            }""",
            phrase,
        )
    except Exception:
        return ""


async def _screen_signature(page: Page) -> str:
    """Identify the rendered question; Pointcare often advances without changing URL."""
    try:
        return await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const headings = [...document.querySelectorAll('h1, h2, h3, [role=heading]')]
                .map((el) => norm(el.innerText)).filter(Boolean).join('|');
              if (headings) return headings;
              return norm((document.body && document.body.innerText) || '').slice(0, 700);
            }"""
        )
    except Exception:
        return ""


def _choice_phrase(value: str) -> str:
    spoken = (value or "").strip()
    lowered = spoken.lower()
    if re.match(r"^(yes|yeah|yep|yup)\b", lowered):
        return "Yes"
    if re.match(r"^(no|nope|nah|negative)\b", lowered):
        return "No"
    # ASR often turns "Never Married" into never matter / never marry / never merit.
    if re.search(r"\bnever\b", lowered) and re.search(
        r"\b(married|marry|matter|merit|murder|marriage)\b", lowered
    ):
        return "Never Married"
    marital = (
        ("never married", "Never Married"),
        ("single", "Never Married"),
        ("not married", "Never Married"),
        ("common law", "Common Law Marriage"),
        ("domestic partner", "Registered Domestic Partner"),
        ("registered domestic", "Registered Domestic Partner"),
        ("married", "Married"),
        ("separated", "Separated"),
        ("divorced", "Divorced"),
        ("widowed", "Widowed"),
        ("widow", "Widowed"),
        ("partner", "Registered Domestic Partner"),
    )
    for cue, label in marital:
        if cue in lowered:
            return label
    return spoken


async def _click_choice(page: Page, value: str) -> None:
    wanted = _choice_phrase(value)
    clicked = await page.evaluate(
        """(wanted) => {
          const norm = (s) => (s || '').replace(/[\\u2010-\\u2015\\u2212–—−]/g, '-').replace(/\\s+/g, ' ').trim().toLowerCase();
          const compact = (s) => norm(s).replace(/[$,]/g, '').replace(/\\s+/g, '');
          const words = (s) => norm(s).split(' ').filter((w) => w.length > 2);
          const target = norm(wanted);
          const targetCompact = compact(wanted);
          const targetWords = new Set(words(wanted));
          const nodes = [...document.querySelectorAll('button, [role=button], [role=radio], [role=option]')];
          let best = null;
          let bestScore = 0;
          for (const el of nodes) {
            const text = norm(el.innerText);
            if (!text || text === 'exit' || text.length > 80) continue;
            const r = el.getBoundingClientRect();
            if (r.width < 2 || r.height < 2) continue;
            const compactText = compact(text);
            let score = 0;
            if (text === target || (targetCompact && compactText === targetCompact)) score = 100;
            else if (targetCompact.length >= 4 && compactText.includes(targetCompact)) score = 88;
            else if (compactText.length >= 4 && targetCompact.includes(compactText)) score = 86;
            else {
              const buttonWords = words(text);
              const hits = buttonWords.filter((w) => targetWords.has(w)).length;
              if (hits && buttonWords.length === 1 && targetWords.has(buttonWords[0])) score = 90;
              else if (hits) score = 60 + hits * 10 - (buttonWords.length - hits) * 15;
            }
            if (score > bestScore) {
              best = el;
              bestScore = score;
            }
          }
          if (!best || bestScore < 80) return false;
          best.click();
          return true;
        }""",
        wanted,
    )
    if clicked:
        await page.wait_for_timeout(400)
        return
    pattern = re.compile(rf"^\s*{re.escape(wanted)}\s*$", re.I)
    candidates = [
        page.get_by_role("radio", name=pattern).first,
        page.get_by_role("option", name=pattern).first,
        page.get_by_role("checkbox", name=pattern).first,
        page.get_by_role("button", name=pattern).first,
        page.get_by_text(wanted, exact=True).first,
    ]
    for loc in candidates:
        try:
            if await loc.count():
                await loc.click(timeout=8000)
                await page.wait_for_timeout(400)
                return
        except Exception:
            continue
    raise RuntimeError(f"Could not select choice {value!r}")


async def _click_named(page: Page, names: tuple[str, ...]) -> None:
    for name in names:
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*$", re.I)
        for loc in (
            page.get_by_role("button", name=pattern).first,
            page.get_by_role("link", name=pattern).first,
        ):
            try:
                if not await loc.count():
                    continue
                label = (await loc.inner_text()).strip()
                if _is_sso_label(label):
                    continue
                await loc.click(timeout=8000)
                await page.wait_for_timeout(500)
                return
            except Exception:
                continue
    raise RuntimeError(f"Could not click any of {names!r}")


async def _click_action(page: Page, action: str) -> None:
    names = ACTION_BUTTONS.get(action, (action,))
    last_exc: Exception | None = None
    for name in names:
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*$", re.I)
        for loc in (
            page.get_by_role("button", name=pattern),
            page.get_by_role("link", name=pattern),
            page.locator("button, [role='button'], input[type='submit']").filter(has_text=pattern),
        ):
            try:
                target = loc.first
                if not await target.count():
                    continue
                label = (await target.inner_text()).strip()
                if _is_sso_label(label):
                    continue
                if not pattern.search(label):
                    continue
                await target.click(timeout=8000)
                await page.wait_for_timeout(700)
                return
            except Exception as exc:
                last_exc = exc
                continue
    raise RuntimeError(f"Could not click action {action!r}: {last_exc}")
