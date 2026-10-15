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
    """Clear session page refs before a forced CDP disconnect (idempotent)."""
    async with _sessions_lock:
        items = list(_sessions.items())
    for call_id, session in items:
        async with session.lock:
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
# Stage E (full application) is not started here yet; extend after programs when ready.
ELIG_QUESTIONS: dict[str, str] = {
    "marital": (
        "What is your marital status? Never Married, Married, Separated, Divorced, "
        "Widowed, Registered Domestic Partner, or Common Law Marriage?"
    ),
    "spouse_first": "What is their first name?",
    "spouse_last": "What is their last name?",
    "spouse_role": "Are they your spouse or your partner?",
    "spouse_age": "What is their age range, or date of birth?",
    "children": (
        "Do you live with and mainly support any children under 19 who live with you "
        "more than half the year? This includes relatives who are not your biological "
        "children. Please say Yes or No."
    ),
    "child_first": "What is the child's first name?",
    "child_last": "What is the child's last name?",
    "child_rel": (
        "What is the relationship to the child? Child, Stepchild, Grandchild, "
        "Foster Child, or Other?"
    ),
    "child_age": "What is the child's age range?",
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
    "person_rel": "What is their relationship to you?",
    "person_age": "What is their age range, or date of birth?",
    "income_when": (
        "For household income, which period do you know best — Annual, Monthly, or "
        "Bi-weekly?"
    ),
    "income_range": (
        "Which approximate household income range applies before taxes? For example, "
        "zero to twelve thousand, or say the dollar range you see."
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
        "Your household may qualify for programs such as Medi-Cal, Medicare, or "
        "Covered California. This does not mean approval. Would you like to continue "
        "to the application? Please say Yes or No."
    ),
}
YES_NO_ELIG_STEPS = frozenset(
    {
        "children",
        "student",
        "disabled",
        "relatives",
        "support",
        "child_more",
        "special_more",
        "programs",
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
        return "children"
    if step == "children":
        return "child_first" if yes else "student"
    if step == "child_first":
        return "child_last"
    if step == "child_last":
        return "child_rel"
    if step == "child_rel":
        return "child_age"
    if step == "child_age":
        return "child_more"
    if step == "child_more":
        return "child_first" if yes else "student"
    if step == "student":
        return _person_or("disabled", session, yes)
    if step == "disabled":
        return _person_or("relatives", session, yes)
    if step == "relatives":
        return _person_or("support", session, yes)
    if step == "support":
        return _person_or("income_when", session, yes)
    if step == "person_first":
        return "person_last"
    if step == "person_last":
        return "person_rel"
    if step == "person_rel":
        return "person_age"
    if step == "person_age":
        return session.after_person or "income_when"
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


def _person_or(after: str, session: LiveFormSession, yes: bool) -> str:
    if not yes:
        return after
    session.after_person = after
    return "person_first"


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
    return ELIG_QUESTIONS[nxt]


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
            if session.page is not None and session.ready.is_set() and session.welcome_done:
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
                    # Opens agent Chrome, then forces a fresh sign-in page for this call.
                    page = await open_page(target)
                    session.page = page
                    await _page_wait(page, 800, call_id=call_id)
                    await _logout_for_fresh_signin(page, target)
                    session.url = page.url or target
                if not session.welcome_done:
                    logger.info("Welcome screens start call_id=%s", call_id)
                    clicked = await advance_welcome_screens(page, call_id=call_id)
                    if "/auth" in (page.url or ""):
                        await settle_on_signin_page(page, call_id=call_id)
                    elif await _already_signed_in(page):
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


async def _sign_in_with_provider(session: LiveFormSession, button_name: str, email: str) -> str:
    """Click Continue with Google/Apple and pick the account they named."""
    page = session.page
    if page is None:
        raise RuntimeError("form page not ready")
    if session.signed_in and session.elig_step:
        screen = await _eligibility_screen(page)
        recovered = _screen_first_step(screen, session)
        session.elig_step = recovered
        return "ASK: " + ELIG_QUESTIONS[recovered]
    if await _already_signed_in(page):
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
        if await _already_signed_in(page):
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
        if await _already_signed_in(page):
            session.popup = None
            logger.info("Google sign-in verified on main page url=%s", page.url)
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

    if await _already_signed_in(page):
        return await _begin_eligibility(session, page)
    logger.error(
        "Google account selected but sign-in was not verified main_url=%s popup_url=%s",
        page.url,
        popup.url if popup is not None else "closed",
    )
    raise RuntimeError(
        "Google sign-in has not finished. Please complete the visible Google prompt, then say Google again."
    )


async def _begin_eligibility(session: LiveFormSession, page: Page) -> str:
    """Open Check Household Eligibility, then return the first question."""
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
    logger.info("Eligibility open url=%s step=%s", page.url, session.elig_step)
    return "ASK: " + ELIG_QUESTIONS[session.elig_step]


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
            spoken = await _begin_eligibility(session, page)
            return {
                "ok": True,
                "auth_screen": "signed_in",
                "spoken": spoken,
                "step": session.elig_step or "marital",
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

    if await _already_signed_in(page):
        spoken = await _begin_eligibility(session, page)
        return {
            "ok": True,
            "auth_screen": "signed_in",
            "spoken": spoken,
            "step": session.elig_step or "marital",
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
                return {
                    "ok": True,
                    "auth_screen": "signed_in",
                    "spoken": "ASK: " + ELIG_QUESTIONS.get(session.elig_step, ""),
                    "step": session.elig_step,
                }
            if await _already_signed_in(page):
                async with session.lock:
                    spoken = await _begin_eligibility(session, page)
                return {
                    "ok": True,
                    "auth_screen": "signed_in",
                    "spoken": spoken,
                    "step": session.elig_step or "marital",
                }
        except Exception:
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


async def _clear_pointcare_session(page: Page) -> None:
    """Hard-reset the agent Chrome session when UI logout is slow or stuck."""
    try:
        await page.context.clear_cookies()
    except Exception:
        logger.exception("Could not clear cookies for fresh sign-in")
    try:
        await page.evaluate(
            """() => {
              try { localStorage.clear(); } catch (e) {}
              try { sessionStorage.clear(); } catch (e) {}
            }"""
        )
    except Exception:
        pass


async def _logout_for_fresh_signin(page: Page, onboarding_url: str) -> None:
    """Leave a saved login so the call starts on the sign-in page."""
    if not await _already_signed_in(page):
        return
    logger.info("Logging out so the call starts on a fresh sign-in page")

    async def _ui_logout() -> None:
        if "/eligibility/" in (page.url or "").lower():
            try:
                await asyncio.wait_for(_click_labeled(page, "Exit"), timeout=4)
                for _ in range(12):
                    await page.wait_for_timeout(200)
                    if "/eligibility/" not in (page.url or "").lower():
                        break
            except Exception:
                logger.warning("Could not exit eligibility before logout")
        try:
            await asyncio.wait_for(_click_labeled(page, "Logout"), timeout=5)
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

    try:
        await asyncio.wait_for(_ui_logout(), timeout=12)
    except Exception:
        logger.warning("UI logout timed out; clearing agent Chrome session cookies")
        await _clear_pointcare_session(page)

    try:
        await page.goto(onboarding_url, wait_until="domcontentloaded", timeout=45000)
    except Exception:
        logger.exception("Could not reopen the onboarding page after logout")
        raise
    await page.wait_for_timeout(700)
    if await _already_signed_in(page):
        logger.warning("Still signed in after logout; clearing cookies and retrying onboarding")
        await _clear_pointcare_session(page)
        await page.goto(onboarding_url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(700)
    if await _already_signed_in(page):
        raise RuntimeError("Pointcare remained signed in after Logout")


async def _already_signed_in(page: Page) -> bool:
    url = (page.url or "").lower()
    if "accounts.google." in url or "/auth" in url:
        return False
    if "/eligibility/" in url:
        return True
    text = (await _visible_button_text(page)).lower()
    return "household eligibility" in text or "logout" in text


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
    """
    session = await get_session(call_id)
    if session is None or session.page is None:
        return None
    if session.signed_in and session.elig_step:
        return ELIG_QUESTIONS.get(session.elig_step) or None
    page = session.page
    try:
        if not await _already_signed_in(page):
            return None
    except Exception:
        logger.exception("promote_signed_in: signed-in check failed call_id=%s", call_id)
        return None
    async with session.lock:
        if session.signed_in and session.elig_step:
            return ELIG_QUESTIONS.get(session.elig_step) or None
        try:
            msg = await _begin_eligibility(session, page)
        except Exception:
            logger.exception("promote_signed_in: begin eligibility failed call_id=%s", call_id)
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
            if field_key == "eligibility_answer" and not session.elig_step:
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
                if (
                    session.elig_step in YES_NO_ELIG_STEPS
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
            # Clarifications must keep the active step (never treat special_person
            # failures as a screen advance back to special).
            if err_text.startswith(
                (
                    "SUPPORT_CLARIFY:",
                    "INCOME_CLARIFY:",
                    "AGE_CLARIFY:",
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
    if relationship_chips or age_chips or (
        "first name" in body and "last name" in body and not has_yes_no
    ):
        if "tell us about" in body and "child" in body:
            return "child_form"
        if ("spouse" in body or "partner" in body) and not relationship_chips:
            return "spouse_form"
        return "person_form"
    if "tell us about" in body and "child" in body and "first name" in body:
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
    if "programs" in body and ("qualify" in body or "eligible" in body):
        return "programs"
    return "unknown"


def _screen_first_step(screen: str, session: LiveFormSession) -> str:
    if screen == "spouse_form":
        return "spouse_first"
    if screen == "child_form":
        return "child_first"
    if screen == "person_form":
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


def _spoken_question_for_step(session: LiveFormSession, step: str) -> str:
    """Build the server-controlled spoken question, with one-time section intros."""
    base = (ELIG_QUESTIONS.get(step) or "").strip()
    if not base:
        return "Could you say that again?"
    if step == "spouse_first" and not session.filled.get("spouse_section_intro"):
        session.filled["spouse_section_intro"] = "1"
        return (
            "Now, I'd like to ask a few questions about your spouse or partner. "
            + base
        )
    if step == "child_first":
        inst = _ensure_child_instance(session)
        key = f"child_section_intro_{inst}"
        if not session.filled.get(key):
            session.filled[key] = "1"
            if inst == "1":
                return "Now, let's get a few details about the child. " + base
            return "Now, let's get a few details about the next child. " + base
    return base


def _section_context_for_step(step: str) -> str:
    """Brief section context for explain/repeat-without-advance requests."""
    if step.startswith("spouse_"):
        return "These questions are about your spouse or partner."
    if step.startswith("child_"):
        return "These questions are about the child we're adding."
    return ""


def _elig_ok(
    call_id: str,
    session: LiveFormSession,
    answer: str,
    next_step: str,
    message: str,
    *,
    spoken_override: str | None = None,
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
    return {
        "ok": True,
        "call_id": call_id,
        "field_id": "eligibility_answer",
        "value": answer,
        "action": "verified",
        "message": message,
        "spoken": "ASK: " + spoken,
        "step": next_step,
    }


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
    """Best-effort scrape of on-screen validation / required-field messages."""
    try:
        texts = await page.evaluate(
            """() => {
              const nodes = [...document.querySelectorAll(
                '[role=alert], [aria-invalid=true], .error, .invalid, '
                + '[class*="error"], [class*="invalid"], [class*="validation"]'
              )];
              const out = [];
              for (const node of nodes) {
                const t = (node.innerText || node.textContent || '')
                  .replace(/\\s+/g, ' ').trim();
                if (t && t.length < 160 && !out.includes(t)) out.push(t);
                if (out.length >= 5) break;
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
    """Parse amounts like '12000', 'twelve thousand', '40 thousand', '40k'."""
    text = (raw or "").lower().replace(",", "").replace("$", "").strip()
    if not text:
        return None
    # "40k" / "50 k"
    k_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*k", text)
    if k_match:
        return int(float(k_match.group(1)) * 1000)
    # "40 thousand" / "50 hundred" with leading digits + scale word.
    digit_scale = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(thousand|hundred|k)\b",
        text,
    )
    if digit_scale:
        base = float(digit_scale.group(1))
        scale = digit_scale.group(2)
        mult = 1000 if scale in {"thousand", "k"} else 100
        return int(base * mult)
    digits = re.search(r"(\d+)\s*\+?", text)
    letterish = re.sub(r"\b(dollars?|and)\b", "", text)
    if digits and not re.search(r"[a-z]", letterish):
        return int(digits.group(1))
    # "twelve thousand", "thirty two thousand seven hundred"
    if "thousand" in text or re.search(r"\bk\b", text) or any(
        w in text for w in _NUMBER_WORDS
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
    text = (spoken or "").strip().lower()
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

    Returns "" when no unique, semantically valid match exists.
    Never defaults to the first/lowest chip on a weak score.
    """
    range_labels = _income_range_labels(labels)
    if not range_labels:
        return ""
    text = (spoken or "").strip().lower()
    if not text:
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
            # Only explicit "other" — niece/cousin/etc. are unsupported without UI labels.
            (r"\bother\b",),
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


_SPECIAL_MEMBER_DISCOVER_JS = """() => {
  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
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
) -> dict[str, Any]:
    """Discover household member rows inside the Special Cases apply-to section."""
    try:
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
    timeout_ms: int = 8000,
    call_id: str = "",
) -> dict[str, Any]:
    """Poll until the apply-to section exposes household member options."""
    deadline = time.time() + max(0.5, timeout_ms / 1000.0)
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = await _discover_special_person_members(page)
        members = list(last.get("members") or [])
        if members:
            logger.info(
                "special_apply_to_ready call_id=%s strategy=%s count=%s labels=%s",
                call_id,
                last.get("strategy"),
                len(members),
                [m.get("label") for m in members[:6]],
            )
            return last
        strategy = str(last.get("strategy") or "")
        if strategy.startswith("wrong_screen"):
            break
        try:
            await page.wait_for_timeout(250)
        except Exception:
            break
    logger.warning(
        "special_apply_to_not_ready call_id=%s strategy=%s has_apply_to=%s "
        "has_special=%s has_marital=%s checkbox_count=%s visible=%s url=%s",
        call_id,
        last.get("strategy"),
        last.get("has_apply_to"),
        last.get("has_special"),
        last.get("has_marital"),
        last.get("checkbox_count"),
        last.get("visible_labels"),
        (last.get("url") or "")[:80],
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


_SPECIAL_MEMBER_CLICK_JS = """({ label, dupIndex, mode }) => {
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
  const findRoot = () => {
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
    return open || document.body;
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
    page: Page, label: str, *, dup_index: int = 0
) -> bool:
    """Click the member's checkbox control (not the name text) and verify checked.

    Pointcare renders an empty ``button[role=checkbox]`` beside a nested name
    ``div`` (e.g. ``Mickey M.``). Clicking the name node alone does not toggle
    or expose ``aria-checked`` — always target the row checkbox. React updates
    ``aria-checked`` after paint, so verify after a short wait (never a second click).
    """
    result = await page.evaluate(
        _SPECIAL_MEMBER_CLICK_JS,
        {"label": label, "dupIndex": int(dup_index), "mode": "click"},
    )
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
    verify = await page.evaluate(
        _SPECIAL_MEMBER_CLICK_JS,
        {"label": label, "dupIndex": int(dup_index), "mode": "verify"},
    )
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
    ok = await _click_special_person_member(page, label, dup_index=dup_index)
    if not ok:
        # Reacquire from current DOM (SPA may have rerendered) and verify once.
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
    # Condition chips are a different step — never treat them as member assignment.
    if re.search(
        r"\bsomeone is\b|\bsomeone has\b|\bi said someone\b|"
        r"^\s*(?:none of the above|pregnant|breast cancer|cervical cancer|"
        r"blind|disabled(?:\s+on\s+ssdi)?|ssdi|was in foster care(?: at 18)?)\s*[.!]?\s*$",
        raw,
        re.I,
    ):
        raise RuntimeError(
            "SPECIAL_PERSON_CLARIFY: That selects the condition. "
            "Who does this apply to? Please say the household member's name."
        )

    queries = _split_member_queries(raw)
    if not queries:
        raise RuntimeError(
            "SPECIAL_PERSON_CLARIFY: Which household member does this apply to?"
        )

    discovered = await _wait_for_special_apply_to_members(
        page, timeout_ms=4000, call_id=session.call_id
    )
    members = list(discovered.get("members") or [])
    strategy = str(discovered.get("strategy") or "")
    if not members:
        # Wrong screen or collapsed apply-to — attempt one bounded recovery.
        discovered = await _recover_special_apply_to_section(page, session)
        members = list(discovered.get("members") or [])
        strategy = str(discovered.get("strategy") or strategy)
    if not members:
        logger.warning(
            "special_person_discovery_empty call_id=%s strategy=%s "
            "checkbox_count=%s has_apply_to=%s has_special=%s has_marital=%s "
            "visible=%s spoken=%r url=%s",
            session.call_id,
            strategy,
            discovered.get("checkbox_count"),
            discovered.get("has_apply_to"),
            discovered.get("has_special"),
            discovered.get("has_marital"),
            discovered.get("visible_labels"),
            (answer or "")[:40],
            (discovered.get("url") or "")[:80],
        )
        if discovered.get("has_marital") and not discovered.get("has_apply_to"):
            raise RuntimeError(
                "SPECIAL_PERSON_CLARIFY: The Special Cases member list is not on "
                "the screen right now. Please say the condition again so I can "
                "reopen who it applies to."
            )
        raise RuntimeError(
            "SPECIAL_PERSON_CLARIFY: The household member list for that condition "
            "is not open yet. Please wait a moment and say the person's name again, "
            "or say the condition again to reopen the list."
        )
    logger.info(
        "special_person_discovery call_id=%s strategy=%s count=%s labels=%s",
        session.call_id,
        strategy,
        len(members),
        [m.get("label") for m in members[:8]],
    )

    selected_labels: list[str] = []
    for query in queries:
        label = await _assign_one_special_person_member(
            page, query, members, session, raw_answer=raw
        )
        if label not in selected_labels:
            selected_labels.append(label)
        # Refresh checked state so subsequent selects see prior checks.
        discovered = await _discover_special_person_members(page)
        members = list(discovered.get("members") or []) or members

    logger.info(
        "special_person assigned call_id=%s labels=%r spoken=%r",
        session.call_id,
        selected_labels,
        (answer or "")[:40],
    )
    remembered = session.filled.get("special_person_assigned", "")
    bits = [x for x in remembered.split("|") if x]
    for label in selected_labels:
        if label not in bits:
            bits.append(label)
    session.filled["special_person_assigned"] = "|".join(bits)


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


def _age_choice(answer: str) -> str:
    """Fallback label when visible chips are unknown (adult/spouse-style bands)."""
    lowered = (answer or "").lower()
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
        ("child", "(0-18) Child"),
        ("baby", "(0-1) Baby"),
        ("toddler", "(1-5)"),
    )
    for cue, label in aliases:
        if cue in lowered:
            return label
    return answer


def _map_age_label(spoken: str, labels: list[str]) -> str | None:
    """Map spoken age onto a visible age chip. None if ambiguous/unrelated.

    Never returns an arbitrary first option — callers must clarify on None.
    Never invents a chip that is not in the provided age-option list.
    """
    age_labels = _age_option_labels(labels)
    if not age_labels:
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
    keyword_map = (
        (r"\bbab(?:y|ies)\b", r"\bbaby\b"),
        (r"\btoddler\b", r"\btoddler\b"),
        (r"\byoung adult\b", r"young adult"),
        (r"\bturning senior\b", r"turning senior"),
        (r"\bsenior\b", r"\bsenior\b"),
        (r"\bteen(?:ager)?\b", r"\bteen\b"),
        (r"\badult\b", r"\badult\b"),
        (r"\bchild\b", r"\bchild\b"),
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
    """
    landed = (landed or "").strip()
    if yes:
        if step == "children" or landed == "child_form":
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
        await _click_exact_choice(page, "Partner" if "partner" in lowered else "Spouse")
        next_step = "spouse_age"
    elif step == "spouse_age":
        if re.search(r"\b(19|20)\d{2}\b", answer):
            await _click_exact_choice(page, "Enter date of birth instead")
            await _strict_fill(page, "Date of birth (optional)", _normalize_dob(answer))
        else:
            age_labels = await _ensure_age_choices_visible(page)
            mapped = _map_age_label(answer, age_labels)
            if not mapped:
                raise RuntimeError(
                    "AGE_CLARIFY: " + _age_clarify_message(step, age_labels)
                )
            await _click_age_choice(page, mapped)
        await _click_continue_to(page, ("children",))
        next_step = "children"
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
                if landed == "child_form":
                    if not await _wait_for_child_form_ready(page, timeout_s=8):
                        raise RuntimeError(
                            "CHILD_FORM: The child details form is still opening. "
                            "Please wait a moment, then say the child's first name."
                        )
                    _ensure_child_instance(session)
                    next_step = "child_first"
                elif landed == "person_form":
                    if step != "children" and not session.after_person:
                        session.after_person = {
                            "student": "disabled",
                            "disabled": "relatives",
                            "relatives": "support",
                            "support": "income_when",
                        }.get(step, "income_when")
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
                next_step = "child_first"
            else:
                try:
                    landed = await _wait_for_screen(
                        page, ("child_form", "person_form"), timeout_s=14
                    )
                except RuntimeError:
                    landed = await _eligibility_screen(page)
                    if landed not in ("child_form", "person_form"):
                        raise
                if landed != "child_form":
                    session.after_person = {
                        "student": "disabled",
                        "disabled": "relatives",
                        "relatives": "support",
                        "support": "income_when",
                    }.get(step, "income_when")
                next_step = _yes_no_transition(step, True, landed)
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
        _remember_household_name(session, answer)
        next_step = "child_last"
    elif step == "child_last":
        if not await _name_field_visible(page, "Last Name"):
            if not await _wait_for_name_field(page, "Last Name", timeout_s=8):
                raise RuntimeError(
                    "CHILD_FORM: I can't see the child's Last Name field on this screen."
                )
        await _strict_fill(page, "Last Name", answer)
        next_step = "child_rel"
    elif step == "child_rel":
        labels = await _visible_choice_labels(page)
        mapped = _map_relationship_label(answer, labels)
        if not mapped:
            shown = ", ".join(
                label
                for label in labels
                if re.search(r"child|other|step|grand|foster", label, re.I)
            )[:160]
            raise RuntimeError(
                "REL_CLARIFY: Please say the relationship. "
                + (
                    f"Options include: {shown}."
                    if shown
                    else "For example, Child or Grandchild."
                )
            )
        await _click_exact_choice(page, mapped)
        next_step = "child_age"
    elif step == "child_age":
        age_labels = await _ensure_age_choices_visible(page)
        mapped = _map_age_label(answer, age_labels)
        if not mapped:
            raise RuntimeError("AGE_CLARIFY: " + _age_clarify_message(step, age_labels))
        await _click_age_choice(page, mapped)
        next_step = "child_more"
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
            if await _name_field_visible(page, "First Name"):
                try:
                    loc = await _locate_named_input(page, "First Name")
                    if loc is not None:
                        current = (await loc.input_value()).strip()
                        if current:
                            await loc.fill("", timeout=3000)
                except Exception:
                    logger.exception("Could not clear new child first name field")
            next_step = "child_first"
        else:
            before = await _eligibility_screen(page)
            try:
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
        _remember_household_name(session, answer)
        next_step = "person_last"
    elif step == "person_last":
        await _strict_fill(page, "Last Name", answer)
        next_step = "person_rel"
    elif step == "person_rel":
        labels = await _visible_choice_labels(page)
        mapped = _map_relationship_label(answer, labels)
        if not mapped:
            # Free-text relationship fields may not use chips — fill or click as given.
            if any(
                label.lower() == (answer or "").strip().lower() for label in labels
            ):
                mapped = next(
                    label
                    for label in labels
                    if label.lower() == (answer or "").strip().lower()
                )
            elif labels:
                shown = ", ".join(labels[:6])
                raise RuntimeError(
                    "REL_CLARIFY: Please say the relationship. "
                    f"Options include: {shown}."
                )
            else:
                mapped = answer
        await _click_exact_choice(page, mapped)
        next_step = "person_age"
    elif step == "person_age":
        age_labels = await _ensure_age_choices_visible(page)
        mapped = _map_age_label(answer, age_labels)
        if not mapped:
            raise RuntimeError(
                "AGE_CLARIFY: " + _age_clarify_message(step, age_labels)
            )
        await _click_age_choice(page, mapped)
        target = session.after_person or "income_when"
        target_screen = "income" if target == "income_when" else target
        await _click_continue_to(page, (target_screen,))
        next_step = target
    elif step == "income_when":
        labels = await _visible_choice_labels(page)
        period = _resolve_income_period_label(answer, labels)
        if not period:
            # Labels may still be loading — allow canonical when spoken maps cleanly.
            canonical = _income_period_choice(answer)
            if canonical in {"Annual", "Monthly", "Bi-weekly"} and not any(
                _looks_like_income_period_label(item) for item in labels
            ):
                period = canonical
            else:
                shown = ", ".join(
                    item
                    for item in labels
                    if _looks_like_income_period_label(item)
                ) or "Annual, Monthly, or Bi-weekly"
                raise RuntimeError(
                    "INCOME_CLARIFY: Please say which income period you know best. "
                    f"Options include: {shown}."
                )
        before_sig = await _choice_signature(page, period)
        before_ranges = _income_range_labels(labels)
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
        session.filled["income_when"] = period
        session.elig_step = "income_range"
        next_step = "income_range"
    elif step == "income_range":
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
        if not mapped or mapped not in income_opts:
            shown = ", ".join(income_opts[:8]) if income_opts else "the ranges on the form"
            raise RuntimeError(
                "INCOME_CLARIFY: Please say which income range matches what you "
                f"earn. Options include: {shown}."
            )
        await _click_exact_choice(page, mapped)
        # Verify the intended chip is selected / still the resolved label.
        after_labels = await _visible_choice_labels(page)
        # If we navigated away, screen wait confirms success.
        try:
            await _wait_for_screen(page, ("special",))
        except RuntimeError:
            # Still on income — confirm mapped label remains among selections.
            if mapped not in after_labels and mapped not in income_opts:
                raise RuntimeError(
                    "INCOME_CLARIFY: I couldn't confirm that income range. "
                    f"Please say one of: {', '.join(income_opts[:8])}."
                )
            raise
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
        session.filled["special_active_conditions"] = "|".join(selected)
        none_selected = len(selected) == 1 and re.search(
            r"none of the above", selected[0], re.I
        )
        if none_selected:
            await _click_cta(page, "Next", attempts=3)
            await _wait_for_screen(page, ("programs",))
            next_step = "programs"
        else:
            # Do not enter special_person until apply-to members are discoverable.
            ready = await _wait_for_special_apply_to_members(
                page, timeout_ms=8000, call_id=call_id
            )
            if not list(ready.get("members") or []):
                ready = await _recover_special_apply_to_section(page, session)
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
        _record_special_assignment_pairs(session, active, newly)
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
                next_step = "programs"
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
        next_step = "programs"
    elif step == "programs":
        if _is_yes(answer) or re.search(r"\b(continue|application|proceed)\b", lowered):
            await _click_labeled(page, "Continue to Applications")
            next_step = ""
        else:
            await _click_labeled(page, "Back")
            next_step = ""
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


def _normalize_dob(text: str) -> str:
    raw = (text or "").strip()
    lowered = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", raw.lower().replace(",", " "))
    months = {
        "jan": "01", "january": "01", "feb": "02", "february": "02",
        "mar": "03", "march": "03", "apr": "04", "april": "04",
        "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
        "aug": "08", "august": "08", "sep": "09", "sept": "09", "september": "09",
        "oct": "10", "october": "10", "nov": "11", "november": "11",
        "dec": "12", "december": "12",
    }
    month_name = "|".join(sorted(months, key=len, reverse=True))
    named = re.search(rf"({month_name})\s+(\d{{1,2}})\s+(\d{{4}})", lowered)
    if named:
        return f"{months[named.group(1)]}/{int(named.group(2)):02d}/{named.group(3)}"
    named = re.search(rf"(\d{{1,2}})\s+({month_name})\s+(\d{{4}})", lowered)
    if named:
        return f"{months[named.group(2)]}/{int(named.group(1)):02d}/{named.group(3)}"
    numeric = re.search(r"(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{4})", lowered)
    if not numeric:
        numeric = re.search(r"(\d{1,2})\s+(\d{1,2})\s+(\d{4})", lowered)
    if numeric:
        return f"{int(numeric.group(1)):02d}/{int(numeric.group(2)):02d}/{numeric.group(3)}"
    return raw


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


async def _select_special_condition_checkbox(
    page: Page,
    label: str,
    *,
    call_id: str,
    multi: bool,
) -> None:
    """Select one Special Cases option with at most one click — never toggle off.

    Isolated to the special step. Does not modify shared choice helpers.
    Accept only when checkbox is verified checked or the apply-to member section
    is confirmed expanded. Never invent success without evidence.
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
    await _click_exact_choice(page, label)
    # Allow the apply-to member section to expand after a successful select.
    await page.wait_for_timeout(450)

    after = await _special_condition_checkbox_state(page, label)
    if after.get("found") and after.get("checked"):
        logger.info(
            "special condition checked call_id=%s label=%r source=%s",
            call_id,
            label,
            after.get("source"),
        )
        return

    if after.get("found") and not after.get("checked"):
        # Explicit unchecked after one click — do NOT click again.
        raise RuntimeError(
            f"SPECIAL_CLARIFY: I couldn't confirm {label}. "
            "Please say the condition again."
        )

    # State unreadable on the outer card — never retry-click.
    after_members = await _special_apply_to_member_count(page)
    apply_heading = await _special_apply_to_heading_visible(page)
    if not is_none and (
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
        raise RuntimeError(
            "SPECIAL_CLARIFY: I couldn't confirm None of the above. "
            "Please say none of the above again."
        )

    # Bounded read-only re-inspection (no second click).
    await page.wait_for_timeout(350)
    reinspect = await _special_condition_readonly_reinspect(page, label)
    if reinspect.get("found") and reinspect.get("checked"):
        logger.info(
            "special condition checked via reinspect call_id=%s label=%r source=%s",
            call_id,
            label,
            reinspect.get("source"),
        )
        return
    if reinspect.get("found") and not reinspect.get("checked"):
        raise RuntimeError(
            f"SPECIAL_CLARIFY: I couldn't confirm {label}. "
            "Please say the condition again."
        )

    # Re-check apply-to after the extra wait — still no clicks.
    later_members = await _special_apply_to_member_count(page)
    later_heading = await _special_apply_to_heading_visible(page)
    if later_members > before_members or (later_heading and later_members > 0):
        logger.info(
            "special condition accepted via delayed apply-to call_id=%s label=%r "
            "members_before=%s members_after=%s",
            call_id,
            label,
            before_members,
            later_members,
        )
        return

    logger.warning(
        "special condition inconclusive after click call_id=%s label=%r "
        "multi=%s members_after=%s reinspect=%s",
        call_id,
        label,
        multi,
        later_members,
        reinspect.get("source"),
    )
    raise RuntimeError(
        f"SPECIAL_CLARIFY: I couldn't confirm {label}. "
        "Please say the condition again."
    )


async def _choice_appears_selected(page: Page, value: str) -> bool:
    """True when the named choice/checkbox looks already selected."""
    phrase = _choice_phrase(value)
    try:
        return bool(
            await page.evaluate(
                """(phrase) => {
                  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                  const want = norm(phrase);
                  const nodes = [...document.querySelectorAll(
                    'button, [role=button], [role=radio], [role=option], [role=checkbox], label, input'
                  )];
                  for (const el of nodes) {
                    const text = norm(
                      el.innerText || el.textContent || el.getAttribute('aria-label') || el.value || ''
                    );
                    if (text !== want && !text.startsWith(want)) continue;
                    if (el.getAttribute('aria-checked') === 'true') return true;
                    if (el.getAttribute('aria-pressed') === 'true') return true;
                    if (el.getAttribute('aria-selected') === 'true') return true;
                    if (el.checked) return true;
                    const cls = (el.className || '').toLowerCase();
                    if (/selected|checked|active|pressed/.test(cls)) return true;
                    const input = el.control || (el.querySelector && el.querySelector('input[type=checkbox]'));
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
