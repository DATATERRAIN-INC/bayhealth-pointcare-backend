"""Stage E Application Collect Mode + program-summary permission gate.

Stage D lands on the Pointcare program-summary screen. We scrape what the page
actually shows, ask permission, then either start the application (Yes) or close
politely without clicking Start Applications (No).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from playwright.async_api import Page

logger = logging.getLogger(__name__)

# Collect Mode questions — server owns order; live form may skip screens.
APP_QUESTIONS: dict[str, str] = {
    "app_agent_offer": (
        "Connecting with a licensed agent later is optional and not required to "
        "finish this application. Would you like that option? Please say Yes or No."
    ),
    "app_citizenship": (
        "Are all household members listed on this application U.S. citizens? "
        "Please say Yes or No."
    ),
    "app_immigration_who": (
        "Which household members are not U.S. citizens? "
        "Say the names, or say continue if everyone who needs a status is already set."
    ),
    "app_immigration": (
        "What is that person's citizenship or immigration status — "
        "U.S. Citizen, U.S. National, Satisfactory immigration status, "
        "or Prefer not to answer?"
    ),
    "app_insurance": (
        "Does anyone on this application have health insurance now? "
        "Please say Yes or No."
    ),
    "app_insurance_members": (
        "Which household members currently have health insurance? "
        "Say the names, or say continue when everyone with coverage is selected."
    ),
    "app_insurance_coverage_type": (
        "What type of coverage do they have — Medicare, Medicaid, health insurance "
        "through a job, TRICARE, VA Health Care, private insurance or ACA Marketplace, "
        "or Other? You can name more than one if needed."
    ),
    "app_dob_review": (
        "Please confirm each person's date of birth looks correct. "
        "Say Yes to continue, or tell me whose date of birth to change."
    ),
    "app_dob_collect": (
        "What is that person's date of birth? Please say the month, day, and year."
    ),
    "app_address_review": (
        "Please confirm each person's home and mailing address looks correct. "
        "Say Yes to continue, or tell me who needs an address update."
    ),
    "app_marital_review": (
        "Please confirm the marital status already filled in looks correct. "
        "Say Yes to continue, or say change if something is wrong."
    ),
    "app_childcare_review": (
        "For each adult, confirm whether they care for children under 19. "
        "Say Yes if that looks correct, or tell me what to change."
    ),
    "app_circumstances": (
        "Do any of these apply to anyone on the application: needs help with daily "
        "activities, veteran or military, American Indian or Alaska Native, "
        "full-time student 18 to 20, or none of the above? "
        "You can name more than one if needed."
    ),
    "app_circumstances_person": (
        "Which household member does that circumstance apply to? Say the name."
    ),
    "app_tax": (
        "The tax filing status on this screen is already filled in. "
        "Does it look correct? Say Yes to continue, or say edit if you need to change it."
    ),
    "app_income": (
        "For work and income, we go one person at a time. What income type should I "
        "enter for the next person — Employed, Self-employed, Unemployed, SSA, SSDI, "
        "Pension, Alimony, Retirement, Rental, Worker's comp, Other, or None?"
    ),
    "app_income_employed": (
        "What is the monthly pay before taxes?"
    ),
    "app_income_employed_hours": "How many hours per week?",
    "app_income_employed_frequency": (
        "What is the pay frequency — weekly, bi-weekly, or monthly?"
    ),
    "app_income_employed_employer": "What is the employer name?",
    "app_income_self": (
        "What is the monthly income after expenses?"
    ),
    "app_income_self_work": "What type of work is it?",
    "app_income_more": (
        "Does that person have another income source to add? Please say Yes or No. "
        "If every person already has income or None, say continue."
    ),
    "app_expenses": (
        "Household expenses are optional. Would you like to add an expense such as "
        "student loans, childcare, or medical, or say skip to continue?"
    ),
    "app_expenses_amount": (
        "What amount should I enter for that expense?"
    ),
    "app_documents": (
        "Documents can be provided now or later. Say skip to continue without "
        "uploading now, or say upload if you want help with documents later."
    ),
    "app_submit_confirm": (
        "The application is ready for final review. After you submit, answers may "
        "no longer be editable. Would you like me to submit the application now? "
        "Please say Yes or No."
    ),
}

# Sequential Collect Mode sub-fields for employed / self-employed income.
_INCOME_EMPLOYED_SEQ = (
    ("pay", "app_income_employed", "What is the monthly pay before taxes?"),
    ("hours", "app_income_employed_hours", "How many hours per week?"),
    (
        "frequency",
        "app_income_employed_frequency",
        "What is the pay frequency — weekly, bi-weekly, or monthly?",
    ),
    ("employer", "app_income_employed_employer", "What is the employer name?"),
)
_INCOME_SELF_SEQ = (
    ("pay", "app_income_self", "What is the monthly income after expenses?"),
    ("work", "app_income_self_work", "What type of work is it?"),
)

YES_NO_APP_STEPS = frozenset(
    {
        "app_agent_offer",
        "app_citizenship",
        "app_insurance",
        "app_dob_review",
        "app_address_review",
        "app_marital_review",
        "app_childcare_review",
        "app_tax",
        "app_income_more",
        "app_submit_confirm",
    }
)

_STAGE_E_EXPLAIN: dict[str, str] = {
    "app_agent_offer": (
        "A licensed agent can help later if you want, but you do not need one to "
        "complete this application."
    ),
    "app_citizenship": (
        "Pointcare needs to know whether everyone listed is a U.S. citizen so it "
        "can choose the right immigration questions."
    ),
    "app_immigration_who": (
        "After you say not everyone is a U.S. citizen, Pointcare needs a "
        "citizenship or immigration status for each household member shown."
    ),
    "app_immigration": (
        "For each household member, Pointcare asks which citizenship or "
        "immigration status to record from the options on the form. "
        "Preselected radios are not treated as confirmed answers."
    ),
    "app_dob_collect": (
        "Pointcare needs a date of birth for each household member who does not "
        "already have one saved on this screen."
    ),
    "app_insurance": (
        "Pointcare asks whether anyone already has health coverage so benefits "
        "can be calculated correctly."
    ),
    "app_insurance_members": (
        "When someone has coverage, Pointcare asks which household members to "
        "mark, then what type of coverage each selected person has."
    ),
    "app_insurance_coverage_type": (
        "Each selected household member can have one or more coverage types "
        "from the options shown under their name."
    ),
    "app_income": (
        "We collect each person's income type and details one short question at "
        "a time so Pointcare can fill the work section accurately."
    ),
    "app_submit_confirm": (
        "After submit, Pointcare may lock the answers. Say Yes only when you are "
        "ready for me to submit."
    ),
}

CLOSING_NO_APPLICATION = (
    "Your household eligibility check is complete, and I have not started an "
    "application. You can log back into Pointcare later if you want to continue. "
    "Thank you for your time. Goodbye."
)

CLOSING_SUBMITTED = (
    "You're all done. Pointcare will follow up if they need anything. You can "
    "upload documents later by logging in with the same account you used today. "
    "For questions, email support@pointcare.com. Thank you for your time. Goodbye."
)

CLOSING_NOT_SUBMITTED = (
    "Okay — I have not submitted the application. Your progress is saved in "
    "Pointcare, and you can finish later by logging in. Thank you for your time. "
    "Goodbye."
)

PROCEED_QUESTION = (
    "Your household eligibility check is finished. "
    "Can I start the application now? Please say Yes to start, or No to end the call."
)

CLOSING_NO_PROCEED = (
    "Okay. I will not start the application. Your saved household information "
    "stays in Pointcare. Thank you for your time. Goodbye."
)

# Friendly caller speech when post-auth navigation fails — never speak diagnostics.
POST_AUTH_RETRY_SPEECH = (
    "You're signed in, and I can see your household dashboard. "
    "Your household eligibility check is finished. "
    "Can I start the application now? Please say Yes to start, or No to end the call."
)

GATE_STEPS = frozenset({"proceed_application", "confirm_start_application"})

PROGRAMS_CLARIFY = (
    "Please say Yes if you want me to start the application, or No if you do not "
    "want to start it now."
)


def build_programs_permission_speech(summary: dict[str, Any]) -> str:
    """Spoken permission ask from scraped program-summary data — never invent programs.

    Speak only programs that may be eligible. Do not list Medicare/Marketplace
    (or any other card) marked \"Not eligible at this time\".
    """
    eligible = [
        p
        for p in (summary.get("programs") or [])
        if isinstance(p, dict) and p.get("may_be_eligible")
    ]
    parts: list[str] = ["Your household eligibility check is complete."]
    if eligible:
        names = [_short_program_name(str(p.get("name") or "")) for p in eligible]
        names = [n for n in names if n]
        if len(names) == 1:
            parts.append(
                f"Pointcare indicates that your household may be eligible for {names[0]}."
            )
        elif names:
            joined = ", ".join(names[:-1]) + f", and {names[-1]}"
            parts.append(
                f"Pointcare indicates that your household may be eligible for {joined}."
            )
        details = []
        for p in eligible:
            detail = (p.get("detail") or "").strip()
            # Skip page chrome like "Not eligible at this time" if it leaked in.
            if re.search(r"not eligible", detail, re.I):
                continue
            if detail and len(detail) < 160:
                details.append(detail)
        if details and len(details) == 1:
            # Soften approval language if the page used strong wording.
            soft = re.sub(
                r"\b(are eligible|is eligible|approved)\b",
                "may be eligible",
                details[0],
                flags=re.I,
            )
            if soft.lower() not in " ".join(parts).lower():
                parts.append(soft if soft.endswith(".") else soft + ".")
        parts.append(
            "Would you like to continue and start the application? "
            "Please say Yes or No."
        )
    else:
        parts.append(
            "Pointcare does not currently show a program marked as one your "
            "household may be eligible for. "
            "Would you like to continue and start the application? Please say Yes or No."
        )
    return " ".join(parts)


def _short_program_name(name: str) -> str:
    text = re.sub(r"\s+", " ", (name or "").strip())
    if not text:
        return ""
    # Prefer common public names when the card title is long.
    low = text.lower()
    if "medikids" in low or "healthy kids" in low:
        return text[:80].rstrip(" .,:;")
    if "medicaid" in low or "medi-cal" in low or "medi cal" in low:
        if "medi-cal" in low or "medi cal" in low:
            return "Medi-Cal"
        return "Medicaid"
    if "medicare" in low:
        return "Medicare"
    if "marketplace" in low or "covered california" in low or "aca" in low:
        if "covered california" in low:
            return "Covered California"
        return "ACA Marketplace"
    # Keep short titles; trim long card text.
    return text[:48].rstrip(" .,:;")


def programs_context_reply(utterance: str, summary: dict[str, Any] | None) -> str | None:
    """Brief contextual answers on the program-summary decision — never navigate."""
    text = (utterance or "").strip().lower()
    if not text:
        return None
    summary = summary or {}
    if re.search(
        r"what happens if i (say )?no|if i (say )?no what|what if i (don'?t|do not)",
        text,
    ):
        return (
            "If you say no, I will not start the application. Your household "
            "eligibility results stay saved in Pointcare, and you can continue later "
            "by logging in. "
            + PROGRAMS_CLARIFY
        )
    if re.search(
        r"what (is|does) (this|that) program|what does that mean|which program|"
        r"what (am i|are we) (eligible|applying)|explain (the )?program",
        text,
    ):
        speech = build_programs_permission_speech(summary)
        return (
            "These are the programs Pointcare listed from your household eligibility "
            "answers — not an approval. "
            + speech
        )
    return None


async def scrape_program_summary(page: Page) -> dict[str, Any]:
    """Read program names / eligibility messages / Start Applications from the live page."""
    try:
        raw = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const body = norm(document.body && document.body.innerText);
              const programs = [];
              const seen = new Set();
              const nodes = [...document.querySelectorAll(
                'label, li, [role="checkbox"], [role="listitem"], button, div, section'
              )];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 20) continue;
                let text = norm(el.innerText || el.textContent || '');
                if (!text || text.length < 4 || text.length > 320) continue;
                const low = text.toLowerCase();
                const looksProgram = (
                  /medicaid|medi-?cal|medicare|marketplace|covered california|\\baca\\b|medikids|healthy kids|florida/i.test(low)
                );
                if (!looksProgram) continue;
                // Prefer compact cards over giant page wrappers.
                if (text.length > 220) continue;
                const notEligible = /not eligible at this time|not eligible/i.test(low);
                const mayEligible = !notEligible && (
                  /may be eligible|might be eligible|eligible for/i.test(low)
                );
                // Extract a short name: first line or known keyword phrase.
                let name = text.split('\\n')[0].slice(0, 80);
                const nameMatch = text.match(
                  /(Medi-?Cal|Medicaid|Medicare|ACA Marketplace|Covered California)/i
                );
                if (nameMatch) name = nameMatch[1];
                const key = name.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                let checked = false;
                const input = el.querySelector && el.querySelector('input[type=checkbox]');
                if (input) checked = !!input.checked;
                const aria = el.getAttribute && el.getAttribute('aria-checked');
                if (aria === 'true') checked = true;
                if (aria === 'false') checked = false;
                programs.push({
                  name,
                  detail: text.slice(0, 200),
                  may_be_eligible: mayEligible || (checked && !notEligible),
                  not_eligible: notEligible,
                  checked,
                });
                if (programs.length >= 8) break;
              }
              const hasStart = /start applications/i.test(body);
              const hasContinue = /continue to applications/i.test(body);
              const headline = /great news/i.test(body) ? 'Great News' : '';
              return {
                programs,
                has_start_applications: hasStart,
                has_continue_to_applications: hasContinue,
                headline,
                body_sample: body.slice(0, 500),
              };
            }"""
        )
    except Exception:
        logger.exception("scrape_program_summary failed")
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    programs = raw.get("programs") if isinstance(raw.get("programs"), list) else []
    return {
        "programs": programs,
        "has_start_applications": bool(raw.get("has_start_applications")),
        "has_continue_to_applications": bool(raw.get("has_continue_to_applications")),
        "headline": str(raw.get("headline") or ""),
        "body_sample": str(raw.get("body_sample") or "")[:500],
    }


def summary_from_session_filled(filled: dict[str, str]) -> dict[str, Any]:
    raw = (filled.get("program_summary_json") or "").strip()
    if not raw:
        return {"programs": []}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"programs": []}
    except Exception:
        return {"programs": []}


async def store_program_summary(session: Any, page: Page) -> dict[str, Any]:
    summary = await scrape_program_summary(page)
    session.filled["program_summary_json"] = json.dumps(summary)
    speech = build_programs_permission_speech(summary)
    session.filled["programs_permission_speech"] = speech
    return summary


async def click_exact_visible_text(page: Page, label: str) -> bool:
    """Click the smallest visible node whose text is exactly ``label``.

    Pointcare's yellow actions are often a div, not a <button>.
    """
    try:
        clicked = await page.evaluate(
            """(label) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const want = norm(label);
              if (!want) return false;
              const nodes = [...document.querySelectorAll(
                'button, [role=button], a, div, span'
              )];
              const hits = [];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 8 || r.height < 8) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (text !== want || text.length > 80) continue;
                hits.push({ el, area: r.width * r.height });
              }
              hits.sort((a, b) => a.area - b.area);
              if (!hits.length) return false;
              hits[0].el.click();
              return true;
            }""",
            label,
        )
        return bool(clicked)
    except Exception:
        logger.exception("click_exact_visible_text failed label=%s", label)
        return False


async def click_quick_apply(page: Page) -> str:
    """Click the dashboard Quick Apply control once.

    Prefers ``Continue Quick Apply`` over ``New Quick Apply`` / bare ``Quick Apply``.
    """
    for label in ("Continue Quick Apply", "Quick Apply", "New Quick Apply"):
        if await click_exact_visible_text(page, label):
            return label
    try:
        clicked = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const nodes = [...document.querySelectorAll(
                'button, [role=button], a, div, span'
              )];
              const hits = [];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 8 || r.height < 8) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text || text.length > 60) continue;
                if (!text.includes('quick apply')) continue;
                if (text.includes('recheck') || text.includes('add member')) continue;
                if (text.includes('household eligibility')) continue;
                hits.push({ el, text, len: text.length, area: r.width * r.height });
              }
              // Prefer Continue Quick Apply over New / bare Quick Apply.
              hits.sort((a, b) => {
                const rank = (t) =>
                  t.includes('continue') ? 0 : t === 'quick apply' ? 1 : 2;
                const d = rank(a.text) - rank(b.text);
                if (d) return d;
                return a.len - b.len || a.area - b.area;
              });
              if (!hits.length) return '';
              hits[0].el.click();
              return hits[0].text;
            }"""
        )
        return str(clicked or "").strip()
    except Exception:
        logger.exception("click_quick_apply failed")
        return ""


async def wait_for_saved_household_surface(page: Page, timeout_s: float = 12.0) -> str:
    """Wait briefly for Quick Apply / programs / application screens after sign-in."""
    deadline = time.monotonic() + timeout_s
    last = "unknown"
    waiting_for_buttons = False
    while time.monotonic() < deadline:
        last = await classify_saved_household_page(page)
        if last.startswith("app_") or last == "programs":
            return last
        body = await _page_body_lower(page)
        if "quick apply" in body:
            return "dashboard"
        if "next steps" in body and "household" in body:
            # Dashboard shell visible; Quick Apply buttons may still be rendering.
            waiting_for_buttons = True
            await page.wait_for_timeout(400)
            continue
        if last == "dashboard":
            waiting_for_buttons = True
            await page.wait_for_timeout(400)
            continue
        if not waiting_for_buttons:
            return last
        await page.wait_for_timeout(400)
    return last


async def _page_body_lower(page: Page) -> str:
    try:
        text = await page.locator("body").inner_text()
    except Exception:
        return ""
    return re.sub(r"\s+", " ", text or "").strip().lower()


async def classify_saved_household_page(page: Page) -> str:
    """Distinguish dashboard (Quick Apply) from Great News (Start/Continue Applications)."""
    body = await _page_body_lower(page)
    has_start = (
        "start applications" in body
        or "continue to applications" in body
        or "continue to application" in body
    )
    has_quick = "quick apply" in body
    has_news = "great news" in body
    if has_news or (
        has_start
        and (
            "not eligible" in body
            or "may be eligible" in body
            or "applying to" in body
        )
    ):
        return "programs"
    # Quick Apply is the dashboard. Check it before application-step keywords
    # so household cards are not mistaken for an in-progress application.
    if has_quick and not has_start:
        return "dashboard"
    app_screen = await detect_application_screen(page)
    if app_screen and app_screen not in ("", "programs"):
        return app_screen
    if "next steps" in body and "household" in body:
        return "dashboard"
    return "unknown"


def _arm_start_permission(session: Any, speech: str) -> str:
    """Ask once before any Quick Apply / Start Applications click."""
    session.elig_step = "proceed_application"
    session.app_step = ""
    session.signed_in = True
    session.welcome_done = True
    session.filled["application_gate"] = "proceed"
    session.filled["application_gate_speech"] = speech
    session.filled["application_test_opening"] = speech
    session.filled["awaiting_quick_apply"] = "1"
    session.filled["application_started"] = "0"
    return "ASK: " + speech


async def prepare_application_test(session: Any, page: Page) -> str:
    """Land on the saved household and ask before starting the application.

    Flow: sign-in → acknowledge eligibility finished → ask to start → Yes/No.
    Does not click Quick Apply until the caller says Yes.
    """
    surface = await wait_for_saved_household_surface(page)
    logger.info(
        "application_test_surface call_id=%s surface=%s url=%s",
        getattr(session, "call_id", "-"),
        surface,
        (page.url or "")[:140],
    )
    if surface == "dashboard":
        return _arm_start_permission(session, PROCEED_QUESTION)
    if surface == "programs":
        await store_program_summary(session, page)
        speech = (session.filled.get("programs_permission_speech") or "").strip()
        if not speech:
            speech = build_programs_permission_speech(
                summary_from_session_filled(session.filled)
            )
        # Still one Yes/No gate — programs speech already asks to start.
        session.elig_step = "programs"
        session.app_step = ""
        session.signed_in = True
        session.welcome_done = True
        session.filled["application_test_opening"] = speech
        session.filled["application_started"] = "0"
        session.filled.pop("awaiting_quick_apply", None)
        return "ASK: " + speech
    if surface.startswith("app_"):
        # Application-test default: restart at question 1 via Back, do not resume
        # mid-flow from a prior call. Browser must actually show the first step.
        first = await restart_application_to_first_question(page)
        verified = await detect_application_screen(page)
        if verified not in _FIRST_STAGE_E_STEPS:
            raise RuntimeError(
                "APPLICATION_TEST: restart reported "
                f"{first!r} but the visible screen is {verified!r}."
            )
        step = verified
        question = app_question_for_step(step)
        _set_pending_question(session, step, question)
        session.app_step = step
        session.elig_step = ""
        session.signed_in = True
        session.welcome_done = True
        session.filled["application_started"] = "1"
        session.filled["resume_app_step"] = step
        session.filled.pop("awaiting_quick_apply", None)
        # Clear mid-flow collect queues from a prior process if any.
        session.filled.pop("app_dob_queue", None)
        session.filled.pop("app_dob_current", None)
        session.filled.pop("app_immigration_queue", None)
        session.filled.pop("app_immigration_current", None)
        session.filled.pop("app_citizenship_no", None)
        session.filled.pop("app_insurance_queue", None)
        session.filled.pop("app_insurance_current", None)
        speech = (
            "Okay, I'll start the application from the first question. " + question
        )
        session.filled["application_test_opening"] = speech
        return "ASK: " + speech
    body = (await _page_body_lower(page))[:300]
    raw_url = (getattr(page, "url", "") or "").lower()
    if "/application" in raw_url:
        raise RuntimeError(
            "APPLICATION_TEST: an application page is open but the active Stage E "
            f"step could not be identified reliably. url={page.url} surface={surface} "
            f"body={body!r}"
        )
    raise RuntimeError(
        "APPLICATION_TEST: expected the saved household dashboard (Quick Apply), "
        "the Great News program summary (Start Applications), or an application "
        f"already in progress. url={page.url} surface={surface} body={body!r}"
    )


def build_start_confirmation(summary: dict[str, Any]) -> str:
    """Explain eligible programs on the page, then ask before Start Applications.

    Do not announce programs marked not eligible.
    """
    eligible = [
        p
        for p in (summary.get("programs") or [])
        if isinstance(p, dict) and p.get("may_be_eligible")
    ]
    parts: list[str] = []
    names = [_short_program_name(str(p.get("name") or "")) for p in eligible]
    names = [n for n in names if n]
    if len(names) == 1:
        parts.append(f"Pointcare shows your household may be eligible for {names[0]}.")
    elif names:
        joined = ", ".join(names[:-1]) + f", and {names[-1]}"
        parts.append(f"Pointcare shows your household may be eligible for {joined}.")
    else:
        parts.append(
            "Pointcare does not currently show a program marked as one your "
            "household may be eligible for."
        )
    parts.append(
        "Should I start the application for the eligible programs? "
        "Please say Yes to start, or No to stop."
    )
    return " ".join(parts)


async def begin_application_after_signin(session: Any, page: Page) -> str:
    """After sign-in, ask before starting the application. Never starts Stage D."""
    session.filled["signin_choice_done"] = "1"
    if session.elig_step in GATE_STEPS:
        speech = (session.filled.get("application_gate_speech") or PROCEED_QUESTION).strip()
        return "ASK: " + speech
    if session.app_step:
        speech = (
            "Your application is already in progress. "
            + app_question_for_step(session.app_step)
        )
        return "ASK: " + speech

    surface = await wait_for_saved_household_surface(page)
    logger.info(
        "application_after_signin_surface call_id=%s surface=%s url=%s",
        getattr(session, "call_id", "-"),
        surface,
        (page.url or "")[:140],
    )
    if surface == "dashboard":
        return _arm_start_permission(session, PROCEED_QUESTION)
    if surface == "programs":
        await store_program_summary(session, page)
        return _arm_start_permission(session, PROCEED_QUESTION)
    if surface.startswith("app_"):
        # Remember the live step; wait for Yes before speaking Stage E questions.
        session.filled["resume_app_step"] = surface
        session.filled["application_started"] = "1"
        session.filled.pop("awaiting_quick_apply", None)
        session.elig_step = "proceed_application"
        session.app_step = ""
        session.signed_in = True
        session.welcome_done = True
        session.filled["application_gate"] = "proceed"
        session.filled["application_gate_speech"] = PROCEED_QUESTION
        return "ASK: " + PROCEED_QUESTION
    body = (await _page_body_lower(page))[:300]
    raw_url = (getattr(page, "url", "") or "").lower()
    if "/application" in raw_url:
        raise RuntimeError(
            "APPLICATION_AFTER_SIGNIN: an application page is open but the active "
            f"Stage E step could not be identified. url={page.url} surface={surface} "
            f"body={body!r}"
        )
    raise RuntimeError(
        "APPLICATION_AFTER_SIGNIN: expected the saved household dashboard "
        "(Quick Apply), the program summary (Start Applications), or an "
        f"application already in progress. url={page.url} surface={surface} "
        f"body={body!r}"
    )


async def _wait_surface_after_quick_apply(page: Page, timeout_s: float = 15.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await page.wait_for_timeout(400)
        surface = await classify_saved_household_page(page)
        if surface and surface != "dashboard":
            return surface
    return await classify_saved_household_page(page)


async def _start_application_from_permission(
    call_id: str,
    session: Any,
    page: Page,
    answer: str,
) -> dict[str, Any]:
    """Yes path: Continue Quick Apply (if needed) → Start Applications → Stage E."""
    surface = await classify_saved_household_page(page)
    if surface == "dashboard":
        clicked = await click_quick_apply(page)
        if not clicked:
            logger.error(
                "quick_apply_missing_after_yes call_id=%s url=%s",
                call_id,
                (page.url or "")[:120],
            )
            speech = (
                session.filled.get("application_gate_speech") or PROCEED_QUESTION
            ).strip()
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "eligibility_answer",
                "value": answer,
                "error": "quick_apply_missing",
                "spoken": "ASK: " + speech,
                "message": "Quick Apply control not found",
                "step": "proceed_application",
                "phase": "eligibility",
            }
        session.filled["quick_apply_clicked"] = clicked
        surface = await _wait_surface_after_quick_apply(page)
        logger.info(
            "quick_apply_after_yes call_id=%s clicked=%s surface=%s",
            call_id,
            clicked,
            surface,
        )
    session.filled.pop("awaiting_quick_apply", None)

    if surface.startswith("app_"):
        session.filled["application_started"] = "1"
        session.elig_step = ""
        # Application-test: never resume mid-flow after Quick Apply — Back to Q1.
        from . import config as _cfg

        if getattr(_cfg, "APPLICATION_TEST_MODE", False):
            first = await restart_application_to_first_question(page)
            verified = await detect_application_screen(page)
            if verified not in _FIRST_STAGE_E_STEPS:
                raise RuntimeError(
                    "APPLICATION_TEST: after Quick Apply, restart reported "
                    f"{first!r} but visible screen is {verified!r}."
                )
            step = verified
            question = app_question_for_step(step)
            _set_pending_question(session, step, question)
            session.app_step = step
            session.filled["resume_app_step"] = step
            session.filled.pop("app_dob_queue", None)
            session.filled.pop("app_dob_current", None)
            session.filled.pop("app_immigration_queue", None)
            session.filled.pop("app_immigration_current", None)
            session.filled.pop("app_citizenship_no", None)
            session.filled.pop("app_insurance_queue", None)
            session.filled.pop("app_insurance_current", None)
            intro = (
                "Okay, I'll start the application from the first question. "
                + question
            )
            return _app_ok(
                call_id,
                session,
                answer,
                step,
                "Restarted in-progress application at question 1",
                spoken_override=intro,
            )
        step, question = await _prepare_resume_speech(session, page, surface)
        session.app_step = step
        intro = "Okay, I've started the application. " + question
        return _app_ok(
            call_id,
            session,
            answer,
            step,
            "Opened in-progress application",
            spoken_override=intro,
        )

    if surface == "programs":
        await store_program_summary(session, page)
        clicked = await click_start_applications(page)
        started = await wait_for_application_started(page)
        if not started:
            logger.error(
                "start_applications_not_verified call_id=%s clicked=%s url=%s",
                call_id,
                clicked,
                (page.url or "")[:120],
            )
            speech = (
                session.filled.get("application_gate_speech") or PROCEED_QUESTION
            ).strip()
            session.elig_step = "proceed_application"
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "eligibility_answer",
                "value": answer,
                "error": "start_applications_failed",
                "spoken": (
                    "ASK: I couldn't confirm that the application started. "
                    "Please say Yes to try again, or No to end the call. "
                    + speech
                ),
                "message": "Start Applications click not verified",
                "step": "proceed_application",
                "phase": "eligibility",
            }
        session.filled["application_started"] = "1"
        session.filled["start_applications_clicked"] = clicked
        session.elig_step = ""
        screen = await detect_application_screen(page)
        next_step = screen if screen and screen != "programs" else "app_citizenship"
        session.app_step = next_step
        intro = "Okay, I've started the application. " + app_question_for_step(next_step)
        return _app_ok(
            call_id,
            session,
            answer,
            next_step,
            "Started application",
            spoken_override=intro,
        )

    speech = (session.filled.get("application_gate_speech") or PROCEED_QUESTION).strip()
    session.elig_step = "proceed_application"
    session.filled["awaiting_quick_apply"] = "1"
    return {
        "ok": False,
        "call_id": call_id,
        "field_id": "eligibility_answer",
        "value": answer,
        "error": "application_start_surface_missing",
        "spoken": "ASK: " + speech,
        "message": f"Unexpected surface after Yes: {surface}",
        "step": "proceed_application",
        "phase": "eligibility",
    }


async def handle_application_gate(
    call_id: str,
    session: Any,
    page: Page,
    answer: str,
) -> dict[str, Any]:
    """Proceed question, then a separate confirmation before Start Applications."""
    from .live_form import classify_yes_no

    step = session.elig_step or ""
    decision = classify_yes_no(answer)
    lowered = (answer or "").strip().lower()
    if decision is None and re.search(
        r"\b(start|continue|proceed|go ahead|quick apply)\b", lowered
    ):
        decision = "yes"
    if step == "proceed_application":
        if decision is None:
            speech = (session.filled.get("application_gate_speech") or PROCEED_QUESTION).strip()
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "eligibility_answer",
                "value": answer,
                "error": "proceed_clarify",
                "spoken": (
                    "ASK: Please say Yes to start the application, or No to end "
                    "the call. " + speech
                ),
                "message": "proceed clarify",
                "step": "proceed_application",
                "phase": "eligibility",
            }
        if decision == "no":
            session.elig_step = ""
            session.app_step = ""
            logger.info("application_proceed_declined call_id=%s", call_id)
            return {
                "ok": True,
                "call_id": call_id,
                "field_id": "eligibility_answer",
                "value": answer,
                "action": "declined_application",
                "message": "Caller declined to proceed",
                "spoken": "ASK: " + CLOSING_NO_PROCEED,
                "step": "",
                "phase": "closing",
                "end_call": True,
            }
        resume = (session.filled.get("resume_app_step") or "").strip()
        if resume or session.filled.get("application_started") == "1":
            from . import config as _cfg

            live = await detect_application_screen(page)
            if live and live.startswith("app_"):
                surface = live
            elif resume.startswith("app_"):
                surface = resume
            else:
                return _unidentified_screen_error(call_id, session, answer)
            # Application-test: Back to question 1 instead of resuming mid-flow.
            if getattr(_cfg, "APPLICATION_TEST_MODE", False):
                first = await restart_application_to_first_question(page)
                verified = await detect_application_screen(page)
                if verified not in _FIRST_STAGE_E_STEPS:
                    raise RuntimeError(
                        "APPLICATION_TEST: gate restart reported "
                        f"{first!r} but visible screen is {verified!r}."
                    )
                step = verified
                question = app_question_for_step(step)
                _set_pending_question(session, step, question)
                session.app_step = step
                session.elig_step = ""
                session.filled["application_started"] = "1"
                session.filled["resume_app_step"] = step
                session.filled.pop("awaiting_quick_apply", None)
                session.filled.pop("app_dob_queue", None)
                session.filled.pop("app_dob_current", None)
                intro = (
                    "Okay, I'll start the application from the first question. "
                    + question
                )
                return _app_ok(
                    call_id,
                    session,
                    answer,
                    step,
                    "Restarted in-progress application at question 1",
                    spoken_override=intro,
                )
            step, question = await _prepare_resume_speech(session, page, surface)
            session.app_step = step
            session.elig_step = ""
            session.filled["application_started"] = "1"
            session.filled["resume_app_step"] = step
            session.filled.pop("awaiting_quick_apply", None)
            intro = (
                "Okay, I'll continue the application that's already open. "
                + question
            )
            return _app_ok(
                call_id,
                session,
                answer,
                step,
                "Resumed in-progress application",
                spoken_override=intro,
            )
        # One Yes starts the application (Continue Quick Apply → Start Applications).
        return await _start_application_from_permission(
            call_id, session, page, answer
        )

    if step != "confirm_start_application":
        raise RuntimeError(f"APPLICATION_AFTER_SIGNIN: unexpected gate step {step!r}")

    if decision is None:
        speech = (session.filled.get("application_gate_speech") or "").strip()
        return {
            "ok": False,
            "call_id": call_id,
            "field_id": "eligibility_answer",
            "value": answer,
            "error": "confirm_clarify",
            "spoken": "ASK: Please say Yes to start the application, or No to stop. " + speech,
            "message": "start confirmation clarify",
            "step": "confirm_start_application",
            "phase": "eligibility",
        }
    if decision == "no":
        session.elig_step = ""
        session.app_step = ""
        session.filled["application_started"] = "0"
        return {
            "ok": True,
            "call_id": call_id,
            "field_id": "eligibility_answer",
            "value": answer,
            "action": "declined_application",
            "message": "Caller declined Start Applications",
            "spoken": "ASK: " + CLOSING_NO_PROCEED,
            "step": "",
            "phase": "closing",
            "end_call": True,
        }
    if session.filled.get("application_started") == "1" and session.app_step:
        return _app_ok(
            call_id,
            session,
            answer,
            session.app_step,
            "Application already started",
            spoken_override=app_question_for_step(session.app_step),
        )
    clicked = await click_start_applications(page)
    started = await wait_for_application_started(page)
    if not started:
        logger.error(
            "start_applications_not_verified call_id=%s clicked=%s url=%s",
            call_id,
            clicked,
            (page.url or "")[:120],
        )
        speech = (session.filled.get("application_gate_speech") or "").strip()
        return {
            "ok": False,
            "call_id": call_id,
            "field_id": "eligibility_answer",
            "value": answer,
            "error": "start_applications_failed",
            "spoken": (
                "ASK: I couldn't confirm that the application started. " + speech
            ),
            "message": "Start Applications click not verified",
            "step": "confirm_start_application",
            "phase": "eligibility",
        }
    session.filled["application_started"] = "1"
    session.filled["start_applications_clicked"] = clicked
    session.elig_step = ""
    screen = await detect_application_screen(page)
    next_step = screen if screen and screen != "programs" else "app_citizenship"
    session.app_step = next_step
    intro = "Okay, I've started the application. " + app_question_for_step(next_step)
    logger.info(
        "stage_e_started call_id=%s clicked=%s first_step=%s",
        call_id,
        clicked,
        next_step,
    )
    return _app_ok(
        call_id,
        session,
        answer,
        next_step,
        "Started application",
        spoken_override=intro,
    )


async def click_start_applications(page: Page) -> str:
    """Click the live Start Applications (or Continue to Applications) control once."""
    from .live_form import _click_cta

    labels = (
        "Start Applications",
        "Start Application",
        "Continue to Applications",
        "Continue to Application",
    )
    for label in labels:
        if await click_exact_visible_text(page, label):
            return label
    try:
        await _click_cta(page, "Start Applications", attempts=1)
        return "Start Applications"
    except Exception as exc:
        raise RuntimeError(
            "Could not find Start Applications on the program summary screen: "
            + str(exc)
        ) from exc


async def wait_for_application_started(page: Page, *, timeout_ms: int = 20000) -> bool:
    """True when Pointcare left the program-summary for an application step."""
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    while time.monotonic() < deadline:
        screen = await detect_application_screen(page)
        if screen and screen != "programs":
            return True
        url = (page.url or "").lower()
        if any(
            part in url
            for part in ("/application", "/apply", "step-", "eligibility/application")
        ):
            # URL moved even if heading scrape is slow.
            if "program" not in url or "application" in url:
                return True
        try:
            body = (await page.locator("body").inner_text()).lower()
        except Exception:
            body = ""
        if "start applications" not in body and (
            "u.s. citizen" in body
            or "us citizen" in body
            or "application eligibility" in body
            or "step 1 of" in body
            or "licensed agent" in body
        ):
            return True
        await page.wait_for_timeout(350)
    return False


async def detect_application_screen(page: Page) -> str:
    """Map the visible application UI to an app_* step key.

    Never invent ``app_citizenship`` from the URL alone — unknown application
    pages return ``\"\"`` so resume can report a recoverable error.
    """
    try:
        body = re.sub(
            r"\s+", " ", (await page.locator("body").inner_text()).lower()
        )
    except Exception:
        return ""
    if "great news" in body or (
        "start applications" in body and ("not eligible" in body or "may be eligible" in body)
    ):
        return "programs"
    if "continue to applications" in body and "eligible" in body:
        return "programs"
    if "licensed agent" in body or "clearguide" in body or "connect me" in body:
        return "app_agent_offer"
    # Immigration detail (member radios) vs everyone-citizens Yes/No.
    has_immig_options = (
        "prefer not to answer" in body
        and (
            "satisfactory immigration" in body
            or "u.s. national" in body
            or "us national" in body
        )
    )
    has_citizen_question = bool(
        re.search(
            r"(are|is)\s+(all|everyone|every)\b.{0,40}\bcitizen",
            body,
        )
        or "everyone listed is a u.s. citizen" in body
        or (
            "all household members" in body
            and "citizen" in body
        )
    )
    if has_immig_options and not has_citizen_question:
        return "app_immigration"
    if "u.s. citizen" in body or "us citizen" in body:
        if has_immig_options and has_citizen_question:
            # Yes/No still visible with expanded options — stay on citizenship
            # until No is verified; callers reconcile to immigration after No.
            return "app_citizenship"
        return "app_citizenship"
    if "immigration status" in body and has_immig_options:
        return "app_immigration"
    if (
        "health insurance" in body
        or ("insurance" in body and ("anyone" in body or "coverage" in body))
        or "someone has coverage" in body
        or "no one has coverage" in body
        or "nobody has coverage" in body
        or "no one has health" in body
    ):
        return "app_insurance"
    if (
        "date of birth" in body
        or re.search(r"\bdob\b", body)
        or "add date of birth" in body
        or "needs date of birth" in body
    ):
        return "app_dob_review"
    if "home address" in body or "mailing address" in body:
        # Review screens may already show saved addresses without Add/pencil.
        return "app_address_review"
    if "marital status" in body and ("pencil" in body or "check" in body):
        return "app_marital_review"
    if "children under 19" in body or "caring for children" in body:
        return "app_childcare_review"
    if (
        "daily activities" in body
        or "alaska native" in body
        or "american indian" in body
        or ("veteran" in body and "student 18" in body)
    ):
        return "app_circumstances"
    if "filing status" in body or "head of household" in body or (
        "tax" in body and "dependent" in body
    ):
        return "app_tax"
    if "add income" in body or ("income type" in body) or (
        "employed" in body and "self-employed" in body and "none" in body
    ):
        return "app_income"
    if "household expense" in body or (
        "student loan" in body and "childcare" in body and "optional" in body
    ):
        return "app_expenses"
    if "upload a file" in body or "take a photo" in body or "proof of" in body:
        return "app_documents"
    if (
        "you're all set" in body
        or "you are all set" in body
        or "submit applications" in body
        or "submit application" in body
    ):
        return "app_submit_confirm"
    if any(
        needle in body
        for needle in (
            "submitted",
            "thank you",
            "we received",
            "application has been",
            "all set",
        )
    ):
        # Post-submit confirmation — do not invent an earlier step.
        return ""
    return ""


def app_question_for_step(step: str, session: Any | None = None) -> str:
    """Return the spoken prompt for ``step``, preferring a pending personalized line."""
    if session is not None:
        pending = (session.filled.get("app_pending_question") or "").strip()
        pending_step = (session.filled.get("app_pending_step") or "").strip()
        if pending and (not pending_step or pending_step == step):
            return pending
    return (APP_QUESTIONS.get(step) or "").strip() or (
        "Please answer the current application question."
    )


def _set_pending_question(session: Any, step: str, speech: str) -> None:
    text = (speech or "").strip()
    if text:
        session.filled["app_pending_question"] = text
        session.filled["app_pending_step"] = step
    else:
        session.filled.pop("app_pending_question", None)
        session.filled.pop("app_pending_step", None)


def stage_e_explain(step: str) -> str:
    """Brief why-this-question context without inventing options."""
    base = app_question_for_step(step)
    extra = (_STAGE_E_EXPLAIN.get(step) or "").strip()
    if extra:
        return f"{extra} {base}"
    return base


def _income_seq_state(session: Any) -> dict[str, Any]:
    raw = session.filled.get("app_income_seq") or ""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _set_income_seq(session: Any, data: dict[str, Any] | None) -> None:
    if not data:
        session.filled.pop("app_income_seq", None)
        return
    session.filled["app_income_seq"] = json.dumps(data)


async def _fill_next_visible_input(page: Page, value: str) -> bool:
    """Fill the next empty visible text/number input (income detail Collect Mode)."""
    try:
        filled = await page.evaluate(
            """(value) => {
              const want = (value || '').trim();
              if (!want) return false;
              const inputs = [...document.querySelectorAll(
                'input:not([type=hidden]):not([type=checkbox]):not([type=radio]), textarea'
              )].filter((el) => {
                const r = el.getBoundingClientRect();
                return r.width > 2 && r.height > 2 && !el.disabled && !el.readOnly;
              });
              for (const el of inputs) {
                if ((el.value || '').trim()) continue;
                el.focus();
                el.value = want;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
              }
              // Fallback: overwrite the first visible empty-looking field.
              if (inputs.length) {
                const el = inputs[0];
                el.focus();
                el.value = want;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
              }
              return false;
            }""",
            value,
        )
        return bool(filled)
    except Exception:
        logger.exception("fill_next_visible_input failed")
        return False


async def _scrape_review_summary(page: Page, step: str) -> str:
    """Best-effort short summary of saved review-screen values for confirmation."""
    body = await _page_body_lower(page)
    if not body:
        return ""
    # Keep summaries short — never dump the whole page.
    snippet = re.sub(r"\s+", " ", body)[:280]
    if step == "app_dob_review" and "date of birth" in snippet:
        return "I see date of birth details already on this screen. "
    if step == "app_address_review" and (
        "address" in snippet or "mailing" in snippet
    ):
        return "I see home or mailing address details already on this screen. "
    if step == "app_marital_review" and "marital" in snippet:
        return "I see a marital status already filled in. "
    if step == "app_childcare_review" and (
        "children" in snippet or "care" in snippet
    ):
        return "I see childcare answers already on this screen. "
    if step == "app_tax" and ("filing" in snippet or "tax" in snippet):
        return "I see a tax filing status already filled in. "
    return ""


def _parse_dob_answer(text: str) -> tuple[str | None, str | None]:
    """Parse a DOB utterance for Stage E.

    Returns ``(mm_dd_yyyy, clarify)``. On success ``clarify`` is None. When the
    numeric date is ambiguous (both parts ≤12), ``clarify`` asks which order
    without guessing. Incomplete/invalid dates return ``(None, None)`` or a
    short clarify when useful. Never logs the raw DOB value.
    """
    from .live_form import extract_dob_ymd

    month, day, year, status = extract_dob_ymd(
        text or "", prefer_us_when_ambiguous=False
    )
    if status == "ambiguous":
        return None, (
            "That numeric date is ambiguous. Please say the month name, then "
            "the day, then the year — for example, October twenty-second, "
            "twenty twenty-four."
        )
    if status == "incomplete":
        return None, None
    if status == "invalid" or month is None or day is None or year is None:
        return None, None
    return f"{month:02d}/{day:02d}/{year:04d}", None


_FIRST_STAGE_E_STEPS = frozenset({"app_agent_offer", "app_citizenship"})


async def _click_application_back(page: Page) -> bool:
    """Click a visible Back control on the application. Returns True if clicked."""
    try:
        back = page.get_by_role("button", name=re.compile(r"^\s*Back\s*$", re.I)).first
        if await back.count() and await back.is_visible():
            await back.click(timeout=4000)
            await page.wait_for_timeout(450)
            return True
    except Exception:
        pass
    try:
        from .live_form import _click_cta

        await _click_cta(page, "Back", attempts=1)
        await page.wait_for_timeout(450)
        return True
    except Exception:
        return False


async def restart_application_to_first_question(page: Page) -> str:
    """Navigate the live application back to the first Stage E question.

    Uses Pointcare's Back control only — does not create a new draft, clear
    auth, or erase household eligibility. Returns the verified first-step key
    (``app_agent_offer`` or ``app_citizenship``). Raises RuntimeError when the
    UI cannot safely reach question 1.
    """
    screen = await detect_application_screen(page)
    if screen in _FIRST_STAGE_E_STEPS:
        logger.info("stage_e_already_at_first_question screen=%s", screen)
        return screen
    if not (screen or "").startswith("app_"):
        raise RuntimeError(
            "APPLICATION_TEST: expected an in-progress application screen to "
            f"restart from question 1, got {screen!r}."
        )

    for attempt in range(24):
        screen = await detect_application_screen(page)
        if screen in _FIRST_STAGE_E_STEPS:
            logger.info(
                "stage_e_restarted_to_first attempt=%s screen=%s",
                attempt,
                screen,
            )
            return screen
        if screen in ("programs",) or screen == "dashboard" or not screen:
            raise RuntimeError(
                "APPLICATION_TEST: navigated away from the application before "
                "reaching question 1. Pointcare has no safe restart-to-step-1 "
                "control here without risking a duplicate draft. Open the "
                "existing application to step 1 manually, or leave the draft "
                f"on the first question. last_screen={screen!r}"
            )
        before = screen
        clicked = await _click_application_back(page)
        if not clicked:
            raise RuntimeError(
                "APPLICATION_TEST: cannot reach question 1 from "
                f"{before!r}; Back control is missing or inactive. "
                "Leave the in-progress application on the first question "
                "(licensed agent or citizenship) before the next test call."
            )
        # Wait for the screen to change after Back.
        for _ in range(16):
            await page.wait_for_timeout(250)
            now = await detect_application_screen(page)
            if now != before:
                break

    final = await detect_application_screen(page)
    raise RuntimeError(
        "APPLICATION_TEST: too many Back clicks without reaching question 1 "
        f"(still on {final!r})."
    )


def _clean_member_display_name(name: str) -> str:
    """Strip trailing UI labels like Add that scrape sometimes glues onto names."""
    label = re.sub(r"\s+", " ", (name or "").strip())
    label = re.sub(
        r"\s+\b(add|edit|pencil|save|next|continue)\b\.?$",
        "",
        label,
        flags=re.I,
    ).strip(" .")
    return label or "that person"


def _dob_question_for(name: str) -> str:
    label = _clean_member_display_name(name)
    return (
        f"What is {label}'s date of birth? Please say the month, day, and year."
    )


def _immigration_status_question_for(name: str) -> str:
    label = _clean_member_display_name(name)
    return (
        f"What is {label}'s citizenship or immigration status — "
        "U.S. Citizen, U.S. National, Satisfactory immigration status, "
        "or Prefer not to answer?"
    )


def _insurance_coverage_question_for(name: str) -> str:
    label = _clean_member_display_name(name)
    return (
        f"What type of coverage does {label} have — Medicare, Medicaid, "
        "health insurance through a job, TRICARE, VA Health Care, private "
        "insurance or ACA Marketplace, or Other? You can name more than one."
    )


def _map_coverage_types(lowered: str) -> list[str]:
    """Map spoken coverage phrases to visible Pointcare option labels."""
    mapping = (
        ("Medicare", ("medicare",)),
        ("Medicaid", ("medicaid", "medi-cal", "medical")),
        (
            "Health insurance through a job (employer-sponsored)",
            ("through a job", "employer", "job", "work insurance", "work coverage"),
        ),
        ("TRICARE", ("tricare", "tri-care", "military health")),
        (
            "VA Health Care Program",
            ("va health", "veterans", "v a health", "va care"),
        ),
        (
            "Private insurance or ACA Marketplace",
            (
                "private",
                "marketplace",
                "aca",
                "obama",
                "exchange",
            ),
        ),
        ("Other", ("other", "something else")),
    )
    hits: list[str] = []
    for label, cues in mapping:
        if any(cue in lowered for cue in cues):
            if label not in hits:
                hits.append(label)
    return hits


async def _scrape_dob_cards(page: Page) -> list[dict[str, Any]]:
    """Inspect household DOB cards: name, saved DOB, and whether Add is needed.

    Pointcare often puts \"Date of birth\" only in the page heading, while each
    member row shows a name plus Add or a saved date — so rows without the DOB
    phrase in-card must still be collected when the page is a DOB review.
    """
    try:
        rows = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const bodyLow = norm(document.body && document.body.innerText || '')
                .toLowerCase();
              const pageIsDob =
                bodyLow.includes('date of birth') ||
                /\\bdob\\b/.test(bodyLow) ||
                bodyLow.includes('add date of birth');
              const cards = [];
              const nodes = [...document.querySelectorAll(
                '[class*="card"], [class*="Card"], [class*="member"], section, article, li, div'
              )];
              const seen = new Set();
              const isNameLine = (line) => {
                let t = (line || '').replace(/\\s+/g, ' ').trim();
                t = t.replace(/\\s+\\b(add|edit|pencil|save|next|continue)\\b\\.?$/i, '').trim();
                return /^[A-Za-z][A-Za-z .'-]{0,40}$/.test(t) &&
                  t.split(' ').length <= 4 &&
                  !/^(date of birth|dob|add|edit|save|next|continue|yes|no|pencil)$/i
                    .test(t);
              };
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 20 || r.height > 420) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text || text.length > 420) continue;
                const low = text.toLowerCase();
                const hasDobCue =
                  low.includes('date of birth') ||
                  /\\bdob\\b/.test(low) ||
                  low.includes('add date') ||
                  low.includes('needs date');
                const hasAdd = /(?:^|\\n|\\s)add(?:\\s|$)/i.test(text) ||
                  [...el.querySelectorAll('button, [role=button], a')].some((b) => {
                    const t = norm(b.innerText || b.textContent || '').toLowerCase();
                    return t === 'add' || t.startsWith('add ');
                  });
                const dobMatch = text.match(
                  /(?:date of birth|dob)\\s*[:\\-]?\\s*(\\d{1,2}[\\/-]\\d{1,2}[\\/-]\\d{2,4})/i
                );
                const dobAlt = text.match(/\\b(\\d{1,2}[\\/-]\\d{1,2}[\\/-]\\d{4})\\b/);
                const dob = (dobMatch && dobMatch[1]) || (dobAlt && dobAlt[1]) || '';
                // On a DOB page, name + Add/date is enough even without the phrase.
                if (!hasDobCue && !(pageIsDob && (hasAdd || dob))) continue;
                const lines = text.split('\\n').map(norm).filter(Boolean);
                let name = '';
                for (const line of lines.slice(0, 5)) {
                  if (/date of birth|\\bdob\\b|add |needs |pencil|edit|save|next|continue/i
                    .test(line)) {
                    continue;
                  }
                  if (isNameLine(line)) {
                    name = line.replace(/\\s+\\b(add|edit|pencil|save|next|continue)\\b\\.?$/i, '').trim();
                    break;
                  }
                }
                if (!name) continue;
                const key = name.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                const needsAdd = Boolean(
                  (hasAdd && !dob) ||
                  /needs date|add date of birth|missing/i.test(text)
                );
                cards.push({
                  name,
                  dob: dob || '',
                  needs_add: needsAdd,
                  index: cards.length,
                });
              }
              return cards;
            }"""
        )
        if isinstance(rows, list):
            parsed = [
                row for row in rows if isinstance(row, dict) and row.get("name")
            ]
            if parsed:
                return parsed
            # Real empty DOM scrape — try body text before giving up.
            return await _scrape_dob_cards_from_body(page)
        return []
    except Exception:
        logger.exception("scrape_dob_cards failed")
        try:
            return await _scrape_dob_cards_from_body(page)
        except Exception:
            return []
    return []


async def _scrape_dob_cards_from_body(page: Page) -> list[dict[str, Any]]:
    """Parse DOB member rows from visible body text when card DOM scrape misses."""
    try:
        raw = await page.locator("body").inner_text()
    except Exception:
        return []
    if not isinstance(raw, str) or not raw.strip():
        return []
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]
    body_low = " ".join(lines).lower()
    if not (
        "date of birth" in body_low
        or re.search(r"\bdob\b", body_low)
        or "add date" in body_low
    ):
        return []
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    name_re = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,40}$")
    skip_re = re.compile(
        r"date of birth|\bdob\b|^add$|^edit$|^save$|^next$|^continue$|"
        r"please confirm|looks correct|household",
        re.I,
    )
    for i, line in enumerate(lines):
        if not name_re.match(line) or skip_re.search(line):
            continue
        if len(line.split()) > 4:
            continue
        # Only consume lines until the next member-looking name.
        chunk: list[str] = []
        for follow in lines[i + 1 : i + 4]:
            if name_re.match(follow) and not skip_re.search(follow) and len(
                follow.split()
            ) <= 4:
                break
            chunk.append(follow)
        window = " ".join(chunk)
        window_low = window.lower()
        dob_match = re.search(
            r"(?:date of birth|dob)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            window,
            re.I,
        )
        dob_alt = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b", window)
        dob = (dob_match.group(1) if dob_match else "") or (
            dob_alt.group(1) if dob_alt else ""
        )
        has_add = bool(re.search(r"(?:^|\s)add(?:\s|$)", window_low))
        # Require nearby Add or a date so page headings are not members.
        if not (has_add or dob):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        cards.append(
            {
                "name": line,
                "dob": dob,
                "needs_add": bool(has_add and not dob),
                "index": len(cards),
            }
        )
    return cards


async def _scrape_immigration_members(page: Page) -> list[dict[str, Any]]:
    """Inspect expanded immigration-status controls per household member."""
    try:
        rows = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const out = [];
              const nodes = [...document.querySelectorAll(
                '[class*="card"], [class*="Card"], [class*="member"], section, article, li, div, fieldset'
              )];
              const seen = new Set();
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 40 || r.height > 520) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text || text.length > 500) continue;
                const low = text.toLowerCase();
                if (!(
                  low.includes('u.s. citizen') ||
                  low.includes('us citizen') ||
                  low.includes('immigration') ||
                  low.includes('u.s. national') ||
                  low.includes('prefer not to answer')
                )) {
                  continue;
                }
                const lines = text.split('\\n').map(norm).filter(Boolean);
                let name = '';
                for (const line of lines.slice(0, 5)) {
                  if (/citizen|immigration|national|prefer not|satisfactory|yes|no/i.test(line)) {
                    continue;
                  }
                  if (/^[A-Za-z][A-Za-z .'-]{1,40}$/.test(line) && line.split(' ').length <= 4) {
                    name = line;
                    break;
                  }
                }
                if (!name) continue;
                const key = name.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                let selected = '';
                const radios = [...el.querySelectorAll(
                  'input[type=radio], [role=radio], label, button, [role=button]'
                )];
                for (const ctrl of radios) {
                  const label = norm(
                    ctrl.getAttribute('aria-label') ||
                    ctrl.innerText ||
                    ctrl.textContent ||
                    ''
                  );
                  const checked =
                    ctrl.checked === true ||
                    ctrl.getAttribute('aria-checked') === 'true' ||
                    (ctrl.className || '').toLowerCase().includes('checked') ||
                    (ctrl.className || '').toLowerCase().includes('selected');
                  if (checked && label) {
                    selected = label;
                    break;
                  }
                }
                out.push({ name, selected, index: out.length });
              }
              return out;
            }"""
        )
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict) and row.get("name")]
    except Exception:
        logger.exception("scrape_immigration_members failed")
    return []


async def _set_member_immigration_status(
    page: Page, member_name: str, status: str
) -> bool:
    """Select ``status`` only within the named member's immigration section."""
    try:
        ok = await page.evaluate(
            """({ memberName, status }) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const wantName = norm(memberName);
              const wantStatus = norm(status);
              if (!wantName || !wantStatus) return false;
              const nodes = [...document.querySelectorAll(
                '[class*="card"], [class*="Card"], [class*="member"], section, article, li, div, fieldset'
              )];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 40) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text.includes(wantName)) continue;
                if (!(
                  text.includes('citizen') ||
                  text.includes('immigration') ||
                  text.includes('national') ||
                  text.includes('prefer not')
                )) {
                  continue;
                }
                const controls = [...el.querySelectorAll(
                  'label, button, [role=button], [role=radio], input[type=radio], span, div'
                )];
                for (const ctrl of controls) {
                  const cr = ctrl.getBoundingClientRect();
                  if (cr.width < 6 || cr.height < 6) continue;
                  const label = norm(
                    ctrl.getAttribute('aria-label') ||
                    ctrl.innerText ||
                    ctrl.textContent ||
                    ''
                  );
                  if (!label || label.length > 80) continue;
                  if (label === wantStatus || label.includes(wantStatus)) {
                    ctrl.click();
                    return true;
                  }
                }
              }
              return false;
            }""",
            {"memberName": member_name, "status": status},
        )
        return bool(ok)
    except Exception:
        logger.exception(
            "set_member_immigration_status failed member=%r status=%r",
            member_name,
            status,
        )
        return False


async def _verify_member_immigration_status(
    page: Page, member_name: str, status: str
) -> bool:
    """True when the named member card shows ``status`` selected."""
    want_name = (member_name or "").strip().lower()
    want_status = (status or "").strip().lower()
    if not want_name or not want_status:
        return False
    members = await _scrape_immigration_members(page)
    for row in members:
        name = str(row.get("name") or "").strip().lower()
        selected = str(row.get("selected") or "").strip().lower()
        if name == want_name or want_name in name or name in want_name:
            if want_status in selected or selected in want_status:
                return True
    return False


async def _scrape_insurance_members(page: Page) -> list[dict[str, Any]]:
    """Inspect expanded household-member coverage checkboxes and types."""
    try:
        rows = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const coverageCues = [
                'medicare', 'medicaid', 'tricare', 'va health', 'private',
                'employer', 'through a job', 'marketplace', 'other',
              ];
              const out = [];
              const nodes = [...document.querySelectorAll(
                '[class*="card"], [class*="Card"], [class*="member"], section, article, li, div, fieldset, label'
              )];
              const seen = new Set();
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 24 || r.height > 720) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text || text.length > 700) continue;
                const low = text.toLowerCase();
                const hasCoverage = coverageCues.some((c) => low.includes(c));
                const hasCheckbox = !!el.querySelector(
                  'input[type=checkbox], [role=checkbox]'
                );
                if (!(hasCoverage || hasCheckbox)) continue;
                const lines = text.split('\\n').map(norm).filter(Boolean);
                let name = '';
                for (const line of lines.slice(0, 6)) {
                  const ll = line.toLowerCase();
                  if (
                    /medicare|medicaid|tricare|coverage|insurance|employer|private|other|va health|marketplace|yes|no/.test(ll)
                  ) {
                    continue;
                  }
                  if (/^[A-Za-z][A-Za-z .'-]{1,40}$/.test(line) && line.split(' ').length <= 4) {
                    name = line;
                    break;
                  }
                }
                if (!name) continue;
                const key = name.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                let checked = false;
                const boxes = [...el.querySelectorAll(
                  'input[type=checkbox], [role=checkbox]'
                )];
                for (const box of boxes) {
                  if (
                    box.checked === true ||
                    box.getAttribute('aria-checked') === 'true' ||
                    (box.className || '').toLowerCase().includes('checked')
                  ) {
                    checked = true;
                    break;
                  }
                }
                const types = [];
                for (const cue of [
                  'Medicare', 'Medicaid',
                  'Health insurance through a job (employer-sponsored)',
                  'TRICARE', 'VA Health Care Program',
                  'Private insurance or ACA Marketplace', 'Other',
                ]) {
                  if (low.includes(cue.toLowerCase().slice(0, 12))) {
                    // detect selected coverage controls under this card
                  }
                }
                const selectedTypes = [];
                const controls = [...el.querySelectorAll(
                  'input[type=checkbox], [role=checkbox], label, button, [role=button]'
                )];
                for (const ctrl of controls) {
                  const label = norm(
                    ctrl.getAttribute('aria-label') ||
                    ctrl.innerText ||
                    ctrl.textContent ||
                    ''
                  );
                  if (!label || label.length > 90) continue;
                  const ll = label.toLowerCase();
                  const isCoverage = coverageCues.some((c) => ll.includes(c));
                  if (!isCoverage) continue;
                  const on =
                    ctrl.checked === true ||
                    ctrl.getAttribute('aria-checked') === 'true' ||
                    (ctrl.className || '').toLowerCase().includes('checked') ||
                    (ctrl.className || '').toLowerCase().includes('selected');
                  if (on) selectedTypes.push(label);
                }
                out.push({
                  name,
                  checked,
                  selected_types: selectedTypes,
                  index: out.length,
                });
              }
              return out;
            }"""
        )
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict) and row.get("name")]
    except Exception:
        logger.exception("scrape_insurance_members failed")
    return []


async def _set_member_insurance_checked(
    page: Page, member_name: str, checked: bool = True
) -> bool:
    """Toggle the named member's insurance checkbox without touching others."""
    try:
        ok = await page.evaluate(
            """({ memberName, wantChecked }) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const want = norm(memberName);
              if (!want) return false;
              const nodes = [...document.querySelectorAll(
                '[class*="card"], [class*="Card"], [class*="member"], section, article, li, div, fieldset, label'
              )];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 20) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text.includes(want)) continue;
                const boxes = [...el.querySelectorAll(
                  'input[type=checkbox], [role=checkbox], label'
                )];
                for (const box of boxes) {
                  const br = box.getBoundingClientRect();
                  if (br.width < 4 || br.height < 4) continue;
                  const label = norm(
                    box.getAttribute('aria-label') ||
                    box.innerText ||
                    box.textContent ||
                    ''
                  );
                  // Prefer the member-name checkbox, not a nested coverage type.
                  const isCoverage = /medicare|medicaid|tricare|va health|private|employer|job|marketplace|other/.test(label);
                  if (isCoverage && label && !label.includes(want)) continue;
                  const on =
                    box.checked === true ||
                    box.getAttribute('aria-checked') === 'true' ||
                    (box.className || '').toLowerCase().includes('checked');
                  if (wantChecked && on) return true;
                  if (!wantChecked && !on) return true;
                  box.click();
                  return true;
                }
              }
              return false;
            }""",
            {"memberName": member_name, "wantChecked": checked},
        )
        return bool(ok)
    except Exception:
        logger.exception(
            "set_member_insurance_checked failed member=%r", member_name
        )
        return False


async def _set_member_coverage_type(
    page: Page, member_name: str, coverage: str
) -> bool:
    """Select a coverage type only within the named member's insurance card."""
    try:
        ok = await page.evaluate(
            """({ memberName, coverage }) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const wantName = norm(memberName);
              const wantCov = norm(coverage);
              if (!wantName || !wantCov) return false;
              const nodes = [...document.querySelectorAll(
                '[class*="card"], [class*="Card"], [class*="member"], section, article, li, div, fieldset'
              )];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 40) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text.includes(wantName)) continue;
                if (!(
                  text.includes('medicare') ||
                  text.includes('medicaid') ||
                  text.includes('coverage') ||
                  text.includes('insurance') ||
                  text.includes('tricare')
                )) {
                  continue;
                }
                const controls = [...el.querySelectorAll(
                  'label, button, [role=button], [role=checkbox], input[type=checkbox], span, div'
                )];
                for (const ctrl of controls) {
                  const cr = ctrl.getBoundingClientRect();
                  if (cr.width < 6 || cr.height < 6) continue;
                  const label = norm(
                    ctrl.getAttribute('aria-label') ||
                    ctrl.innerText ||
                    ctrl.textContent ||
                    ''
                  );
                  if (!label || label.length > 100) continue;
                  if (!(label === wantCov || label.includes(wantCov) || wantCov.includes(label))) {
                    continue;
                  }
                  const on =
                    ctrl.checked === true ||
                    ctrl.getAttribute('aria-checked') === 'true' ||
                    (ctrl.className || '').toLowerCase().includes('checked') ||
                    (ctrl.className || '').toLowerCase().includes('selected');
                  // Never toggle an already-selected coverage off.
                  if (on) return true;
                  ctrl.click();
                  return true;
                }
              }
              return false;
            }""",
            {"memberName": member_name, "coverage": coverage},
        )
        return bool(ok)
    except Exception:
        logger.exception(
            "set_member_coverage_type failed member=%r coverage=%r",
            member_name,
            coverage,
        )
        return False


async def _verify_member_coverage_types(
    page: Page, member_name: str, expected: list[str]
) -> bool:
    """True when each expected coverage type is selected under the member."""
    want = (member_name or "").strip().lower()
    if not want or not expected:
        return False
    rows = await _scrape_insurance_members(page)
    for row in rows:
        name = str(row.get("name") or "").strip().lower()
        if not (name == want or want in name or name in want):
            continue
        selected = [
            str(t).strip().lower()
            for t in (row.get("selected_types") or [])
            if str(t).strip()
        ]
        for exp in expected:
            el = exp.strip().lower()
            if not any(el in s or s in el for s in selected):
                return False
        return True
    return False


async def _set_circumstance_member(
    page: Page, member_name: str, circumstance: str
) -> bool:
    """Select ``member_name`` only inside the named circumstance section.

    Prevents a Cervical-cancer member click from toggling Blind (or any other
    section that appears earlier on the page).
    """
    try:
        ok = await page.evaluate(
            """({ memberName, circumstance }) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const wantMember = norm(memberName);
              const wantCond = norm(circumstance);
              if (!wantMember) return false;
              const nodes = [...document.querySelectorAll(
                '[class*="card"], [class*="Card"], section, article, fieldset, li, div'
              )];
              // Prefer the smallest card that contains the circumstance text.
              const candidates = [];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 30 || r.height > 900) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text) continue;
                if (wantCond && !text.includes(wantCond.slice(0, Math.min(24, wantCond.length)))) {
                  // Also accept key tokens (veteran, student, indian, daily).
                  const tokens = wantCond.split(' ').filter((t) => t.length > 4);
                  if (!tokens.some((t) => text.includes(t))) continue;
                }
                if (!wantCond && !/apply|who does|member|household/.test(text)) continue;
                candidates.push({ el, area: r.width * r.height, text });
              }
              candidates.sort((a, b) => a.area - b.area);
              const scopes = candidates.length ? candidates : [{ el: document.body, area: 0, text: '' }];
              for (const scope of scopes) {
                const controls = [...scope.el.querySelectorAll(
                  'input[type=checkbox], [role=checkbox], label, button, [role=button], span, div'
                )];
                for (const ctrl of controls) {
                  const cr = ctrl.getBoundingClientRect();
                  if (cr.width < 4 || cr.height < 4) continue;
                  const label = norm(
                    ctrl.getAttribute('aria-label') ||
                    ctrl.innerText ||
                    ctrl.textContent ||
                    ''
                  );
                  if (!label || label.length > 60) continue;
                  if (!(
                    label === wantMember ||
                    label.includes(wantMember) ||
                    wantMember.includes(label)
                  )) {
                    continue;
                  }
                  // Avoid matching the circumstance title itself.
                  if (wantCond && (label.includes(wantCond) || wantCond.includes(label))) {
                    continue;
                  }
                  const on =
                    ctrl.checked === true ||
                    ctrl.getAttribute('aria-checked') === 'true' ||
                    (ctrl.className || '').toLowerCase().includes('checked') ||
                    (ctrl.className || '').toLowerCase().includes('selected');
                  if (on) return true;
                  ctrl.click();
                  return true;
                }
              }
              return false;
            }""",
            {"memberName": member_name, "circumstance": circumstance or ""},
        )
        return bool(ok)
    except Exception:
        logger.exception(
            "set_circumstance_member failed member=%r circumstance=%r",
            member_name,
            circumstance,
        )
        return False


async def _scrape_address_cards(page: Page) -> list[dict[str, Any]]:
    """Inspect household address cards for missing home/mailing entries."""
    try:
        rows = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const out = [];
              const nodes = [...document.querySelectorAll(
                '[class*="card"], [class*="Card"], [class*="member"], section, article, li, div'
              )];
              const seen = new Set();
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 24 || r.height > 600) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text || text.length > 600) continue;
                const low = text.toLowerCase();
                if (!(low.includes('home address') || low.includes('mailing address') || low.includes('address'))) {
                  continue;
                }
                const lines = text.split('\\n').map(norm).filter(Boolean);
                let name = '';
                for (const line of lines.slice(0, 5)) {
                  if (/address|add|edit|home|mailing|street|city|zip/i.test(line)) continue;
                  if (/^[A-Za-z][A-Za-z .'-]{1,40}$/.test(line) && line.split(' ').length <= 4) {
                    name = line;
                    break;
                  }
                }
                if (!name) continue;
                const key = name.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                const needsHome =
                  /needs home address|add home|no home address/i.test(text) ||
                  (low.includes('home address') && /\\badd\\b/i.test(text) && !/\\d{5}/.test(text));
                const needsMailing =
                  /needs mailing address|add mailing|no mailing address/i.test(text) ||
                  (low.includes('mailing address') && /\\badd\\b/i.test(text) && !/\\d{5}/.test(text));
                const hasAdd = /(?:^|\\s)add(?:\\s|$)/i.test(text);
                out.push({
                  name,
                  needs_home: needsHome || (hasAdd && low.includes('home')),
                  needs_mailing: needsMailing || (hasAdd && low.includes('mailing')),
                  needs_add: hasAdd && (needsHome || needsMailing || !/\\d{5}/.test(text)),
                  index: out.length,
                });
              }
              return out;
            }"""
        )
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict) and row.get("name")]
    except Exception:
        logger.exception("scrape_address_cards failed")
    return []


async def _click_address_option(page: Page, label: str) -> bool:
    """Click a visible address option such as No Home Address."""
    from .live_form import _click_cta, _click_exact_choice

    for candidate in (label, label.title(), label.upper()):
        try:
            await _click_exact_choice(page, candidate)
            return True
        except Exception:
            try:
                await _click_cta(page, candidate, attempts=1)
                return True
            except Exception:
                continue
    return False


async def _click_member_add_dob(page: Page, member_name: str) -> bool:
    """Click the Add control on the named member's DOB card."""
    try:
        ok = await page.evaluate(
            """(memberName) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase()
                .replace(/\\./g, '');
              const want = norm(memberName);
              if (!want) return false;
              const wantFirst = want.split(' ')[0];
              const nodes = [...document.querySelectorAll(
                '[class*="card"], [class*="Card"], [class*="member"], section, article, li, div'
              )];
              const pageIsDob = norm(
                document.body && document.body.innerText || ''
              ).includes('date of birth');
              const candidates = [];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 20) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (!text) continue;
                const nameHit =
                  text.includes(want) ||
                  (wantFirst && text.split(/\\n| /).includes(wantFirst));
                if (!nameHit) continue;
                const hasDobCue =
                  text.includes('date of birth') ||
                  /\\bdob\\b/.test(text) ||
                  text.includes('add date');
                const buttons = [...el.querySelectorAll(
                  'button, [role=button], a, span, div'
                )];
                for (const btn of buttons) {
                  const br = btn.getBoundingClientRect();
                  if (br.width < 8 || br.height < 8) continue;
                  const label = norm(btn.innerText || btn.textContent || '');
                  if (
                    label === 'add' ||
                    label.startsWith('add ') ||
                    label.includes('add date')
                  ) {
                    if (hasDobCue || pageIsDob) {
                      candidates.push({ btn, exact: text.includes(want), area: br.width * br.height });
                    }
                  }
                }
              }
              candidates.sort((a, b) => Number(b.exact) - Number(a.exact) || a.area - b.area);
              if (!candidates.length) return false;
              candidates[0].btn.click();
              return true;
            }""",
            member_name,
        )
        return bool(ok)
    except Exception:
        logger.exception("click_member_add_dob failed member=%r", member_name)
        return False


def _dob_parts(dob: str) -> tuple[str, str, str] | None:
    """Split ``MM/DD/YYYY`` into month, day, year strings."""
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", (dob or "").strip())
    if not m:
        return None
    return f"{int(m.group(1)):02d}", f"{int(m.group(2)):02d}", m.group(3)


def _dob_digits_match(expected: str, actual: str) -> bool:
    want = re.sub(r"\D", "", expected or "")
    got = re.sub(r"\D", "", actual or "")
    if not want or not got:
        return False
    return got == want or got.endswith(want) or want.endswith(got)


async def _inspect_dob_editor(page: Page) -> dict[str, Any]:
    """Describe the open DOB editor controls (masked / date / segmented)."""
    try:
        info = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const visible = (el) => {
                const r = el.getBoundingClientRect();
                return r.width > 2 && r.height > 2 && !el.disabled;
              };
              const inputs = [...document.querySelectorAll(
                'input:not([type=hidden]):not([type=checkbox]):not([type=radio]), textarea'
              )].filter(visible);
              const dateLike = inputs.filter((el) => {
                const t = (el.type || '').toLowerCase();
                const ph = norm(el.placeholder || el.getAttribute('aria-label') || '');
                const name = norm(el.name || el.id || '');
                return (
                  t === 'date' ||
                  /birth|dob|mm\\/dd|month|day|year/i.test(ph + ' ' + name)
                );
              });
              const pool = dateLike.length ? dateLike : inputs;
              const values = pool.map((el) => ({
                type: (el.type || 'text').toLowerCase(),
                value: el.value || '',
                placeholder: el.placeholder || '',
                name: el.name || el.id || '',
                maxLength: el.maxLength || 0,
                inputMode: el.inputMode || '',
              }));
              const hasSave = [...document.querySelectorAll(
                'button, [role=button], a, input[type=button], input[type=submit]'
              )].some((el) => {
                if (!visible(el)) return false;
                const t = norm(el.innerText || el.value || el.textContent || '')
                  .toLowerCase();
                return t === 'save' || t.startsWith('save ') || t.includes('save and');
              });
              return {
                count: pool.length,
                values,
                hasSave,
                kinds: values.map((v) => v.type),
              };
            }"""
        )
        return info if isinstance(info, dict) else {}
    except Exception:
        logger.exception("inspect_dob_editor failed")
        return {}


async def _read_dob_editor_value(page: Page) -> str:
    """Return concatenated digits currently shown in the open DOB editor."""
    try:
        raw = await page.evaluate(
            """() => {
              const visible = (el) => {
                const r = el.getBoundingClientRect();
                return r.width > 2 && r.height > 2 && !el.disabled;
              };
              const inputs = [...document.querySelectorAll(
                'input:not([type=hidden]):not([type=checkbox]):not([type=radio]), textarea'
              )].filter(visible);
              const dateLike = inputs.filter((el) => {
                const t = (el.type || '').toLowerCase();
                const ph = ((el.placeholder || '') + ' ' + (el.name || '') + ' ' +
                  (el.id || '') + ' ' + (el.getAttribute('aria-label') || ''))
                  .toLowerCase();
                return t === 'date' || /birth|dob|mm\\/dd|month|day|year/.test(ph);
              });
              const pool = dateLike.length ? dateLike : inputs.slice(0, 3);
              return pool.map((el) => el.value || '').join(' ');
            }"""
        )
        return str(raw or "")
    except Exception:
        return ""


async def _fill_dob_editor_fields(page: Page, dob: str) -> bool:
    """Fill the open DOB editor using the format Pointcare actually exposes."""
    parts = _dob_parts(dob)
    if not parts:
        return False
    mm, dd, yyyy = parts
    iso = f"{yyyy}-{mm}-{dd}"
    digits = f"{mm}{dd}{yyyy}"
    try:
        filled = await page.evaluate(
            """({ mm, dd, yyyy, iso, slash, digits }) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const visible = (el) => {
                const r = el.getBoundingClientRect();
                return r.width > 2 && r.height > 2 && !el.disabled && !el.readOnly;
              };
              const setVal = (el, value) => {
                el.focus();
                el.value = '';
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.value = value;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
              };
              const inputs = [...document.querySelectorAll(
                'input:not([type=hidden]):not([type=checkbox]):not([type=radio]), textarea'
              )].filter(visible);
              if (!inputs.length) return { ok: false, mode: 'none' };

              const dateInputs = inputs.filter((el) => (el.type || '').toLowerCase() === 'date');
              if (dateInputs.length === 1) {
                setVal(dateInputs[0], iso);
                return { ok: true, mode: 'html-date', value: dateInputs[0].value || '' };
              }

              const labeled = { month: null, day: null, year: null, combined: null };
              for (const el of inputs) {
                const hint = norm(
                  (el.placeholder || '') + ' ' + (el.name || '') + ' ' +
                  (el.id || '') + ' ' + (el.getAttribute('aria-label') || '') +
                  ' ' + (el.getAttribute('autocomplete') || '')
                );
                if (/month|mm\\b/.test(hint) && !labeled.month) labeled.month = el;
                else if (/\\bday\\b|dd\\b/.test(hint) && !labeled.day) labeled.day = el;
                else if (/year|yyyy|yy\\b/.test(hint) && !labeled.year) labeled.year = el;
                else if (/birth|dob|date/.test(hint) && !labeled.combined) labeled.combined = el;
              }
              if (labeled.month && labeled.day && labeled.year) {
                setVal(labeled.month, mm);
                setVal(labeled.day, dd);
                setVal(labeled.year, yyyy);
                return {
                  ok: true,
                  mode: 'segmented',
                  value: [labeled.month.value, labeled.day.value, labeled.year.value].join('/'),
                };
              }
              // Three short numeric fields without clear labels — treat as M/D/Y.
              const short = inputs.filter((el) => {
                const ml = el.maxLength || 0;
                const t = (el.type || '').toLowerCase();
                return (t === 'text' || t === 'tel' || t === 'number') && (ml > 0 && ml <= 4);
              });
              if (short.length >= 3) {
                setVal(short[0], mm);
                setVal(short[1], dd);
                setVal(short[2], yyyy);
                return {
                  ok: true,
                  mode: 'segmented-short',
                  value: [short[0].value, short[1].value, short[2].value].join('/'),
                };
              }
              const target = labeled.combined || inputs[0];
              setVal(target, slash);
              let got = target.value || '';
              if (!got || got.replace(/\\D/g, '').length < 8) {
                setVal(target, digits);
                got = target.value || '';
              }
              return { ok: true, mode: 'combined', value: got };
            }""",
            {
                "mm": mm,
                "dd": dd,
                "yyyy": yyyy,
                "iso": iso,
                "slash": f"{mm}/{dd}/{yyyy}",
                "digits": digits,
            },
        )
        if not isinstance(filled, dict) or not filled.get("ok"):
            return False
        got = str(filled.get("value") or "")
        if _dob_digits_match(dob, got):
            return True
        # Masked fields often need sequential key entry after clear.
        return await _type_dob_via_keyboard(page, dob)
    except Exception:
        logger.exception("fill_dob_editor_fields failed")
        return False


async def _type_dob_via_keyboard(page: Page, dob: str) -> bool:
    """Enter DOB digits sequentially into the focused/visible masked field."""
    parts = _dob_parts(dob)
    if not parts:
        return False
    mm, dd, yyyy = parts
    digits = f"{mm}{dd}{yyyy}"
    try:
        focused = await page.evaluate(
            """() => {
              const visible = (el) => {
                const r = el.getBoundingClientRect();
                return r.width > 2 && r.height > 2 && !el.disabled;
              };
              const inputs = [...document.querySelectorAll(
                'input:not([type=hidden]):not([type=checkbox]):not([type=radio]), textarea'
              )].filter(visible);
              const dateLike = inputs.filter((el) => {
                const t = (el.type || '').toLowerCase();
                const ph = ((el.placeholder || '') + ' ' + (el.name || '') + ' ' +
                  (el.id || '') + ' ' + (el.getAttribute('aria-label') || ''))
                  .toLowerCase();
                return t === 'date' || /birth|dob|mm\\/dd|date/.test(ph);
              });
              const el = dateLike[0] || inputs[0];
              if (!el) return false;
              el.focus();
              el.select && el.select();
              return true;
            }"""
        )
        if not focused:
            return False
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        await page.keyboard.type(digits, delay=40)
        await page.wait_for_timeout(150)
        got = await _read_dob_editor_value(page)
        return _dob_digits_match(dob, got)
    except Exception:
        logger.exception("type_dob_via_keyboard failed")
        return False


async def _click_dob_save(page: Page) -> bool:
    """Click Save in the open DOB editor. Failure is a real failure."""
    from .live_form import _click_cta

    for label in ("Save", "Save and Return", "Save & Return"):
        try:
            await _click_cta(page, label, attempts=1)
            return True
        except Exception:
            continue
    try:
        clicked = await page.evaluate(
            """() => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const nodes = [...document.querySelectorAll(
                'button, [role=button], a, input[type=button], input[type=submit]'
              )];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 4 || r.height < 4) continue;
                if (el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
                const t = norm(el.innerText || el.value || el.textContent || '');
                if (t === 'save' || t.startsWith('save ') || t.includes('save and')) {
                  el.click();
                  return true;
                }
              }
              return false;
            }"""
        )
        return bool(clicked)
    except Exception:
        logger.exception("click_dob_save failed")
        return False


async def _fill_and_save_dob(page: Page, dob: str) -> bool:
    """Fill the open DOB editor, verify the value, then Save.

    Never returns True merely because the editor opened. Save failure and a
    blank/malformed field are failures.
    """
    from .live_form import _strict_fill

    parts = _dob_parts(dob)
    if not parts:
        return False

    # Wait briefly for the editor controls after Add.
    editor: dict[str, Any] = {}
    for _ in range(8):
        editor = await _inspect_dob_editor(page)
        if int(editor.get("count") or 0) > 0:
            break
        await page.wait_for_timeout(150)
    if int(editor.get("count") or 0) <= 0:
        logger.warning("dob_editor_not_ready")
        return False

    filled = False
    # Prefer scoped editor fill over page-wide label lookup.
    filled = await _fill_dob_editor_fields(page, dob)
    if not filled:
        try:
            await _strict_fill(page, "Date of birth", dob)
            got = await _read_dob_editor_value(page)
            filled = _dob_digits_match(dob, got)
        except Exception:
            filled = False
    if not filled:
        filled = await _type_dob_via_keyboard(page, dob)
    if not filled:
        logger.warning("dob_fill_value_missing_after_entry")
        return False

    got_before_save = await _read_dob_editor_value(page)
    if not _dob_digits_match(dob, got_before_save):
        # One recovery: keyboard retype into the visible field.
        if not await _type_dob_via_keyboard(page, dob):
            logger.warning("dob_fill_malformed_before_save")
            return False
        got_before_save = await _read_dob_editor_value(page)
        if not _dob_digits_match(dob, got_before_save):
            return False

    saved = await _click_dob_save(page)
    if not saved:
        logger.warning("dob_save_click_failed")
        return False
    await page.wait_for_timeout(500)
    return True


async def _verify_member_dob(page: Page, member_name: str, dob: str) -> bool:
    """True when the member card shows ``dob`` and no longer needs Add."""
    cards = await _scrape_dob_cards(page)
    want = (member_name or "").strip().lower()
    want_digits = re.sub(r"\D", "", dob or "")
    matches = [
        card
        for card in cards
        if (card.get("name") or "").strip().lower() == want
        or want in (card.get("name") or "").strip().lower()
    ]
    if not matches:
        # Fallback: body contains the date near the name.
        body = await _page_body_lower(page)
        if want and want_digits and want in body and want_digits[-8:] in re.sub(
            r"\D", "", body
        ):
            return True
        return False
    for card in matches:
        got = re.sub(r"\D", "", str(card.get("dob") or ""))
        if got and want_digits and (
            got == want_digits or got.endswith(want_digits[-8:])
        ):
            if not card.get("needs_add"):
                return True
        if not card.get("needs_add") and got:
            return True
    return False


async def _next_button_enabled(page: Page) -> bool | None:
    """Return True/False when a Next/Continue control is found; None if missing."""
    try:
        state = await page.evaluate(
            """() => {
              const labels = ['next', 'continue'];
              const nodes = [...document.querySelectorAll(
                'button, [role=button], a, input[type=button], input[type=submit]'
              )];
              for (const el of nodes) {
                const text = ((el.innerText || el.value || el.textContent || '')
                  ).replace(/\\s+/g, ' ').trim().toLowerCase();
                if (!labels.includes(text)) continue;
                const r = el.getBoundingClientRect();
                if (r.width < 4 || r.height < 4) continue;
                const disabled =
                  el.disabled === true ||
                  el.getAttribute('aria-disabled') === 'true' ||
                  (el.className || '').toLowerCase().includes('disabled');
                return !disabled;
              }
              return null;
            }"""
        )
        if state is None:
            return None
        return bool(state)
    except Exception:
        return None


def _immigration_queue(session: Any) -> list[str]:
    raw = session.filled.get("app_immigration_queue") or ""
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception:
        pass
    return []


def _set_immigration_queue(session: Any, names: list[str] | None) -> None:
    if not names:
        session.filled.pop("app_immigration_queue", None)
        session.filled.pop("app_immigration_current", None)
        return
    session.filled["app_immigration_queue"] = json.dumps(names)


def _insurance_queue(session: Any) -> list[str]:
    raw = session.filled.get("app_insurance_queue") or ""
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception:
        pass
    return []


def _set_insurance_queue(session: Any, names: list[str] | None) -> None:
    if not names:
        session.filled.pop("app_insurance_queue", None)
        session.filled.pop("app_insurance_current", None)
        return
    session.filled["app_insurance_queue"] = json.dumps(names)


def _dob_queue(session: Any) -> list[str]:
    raw = session.filled.get("app_dob_queue") or ""
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception:
        pass
    return []


def _set_dob_queue(session: Any, names: list[str] | None) -> None:
    if not names:
        session.filled.pop("app_dob_queue", None)
        session.filled.pop("app_dob_current", None)
        session.filled.pop("app_dob_correcting", None)
        return
    session.filled["app_dob_queue"] = json.dumps(names)


# ASR often hears single-letter initials as short words ("H" → "edge" / "aitch").
_INITIAL_ASR_ALIASES: dict[str, str] = {
    "ay": "a",
    "bee": "b",
    "be": "b",
    "see": "c",
    "sea": "c",
    "dee": "d",
    "ee": "e",
    "ef": "f",
    "eff": "f",
    "gee": "g",
    "aitch": "h",
    "aitch.": "h",
    "edge": "h",
    "age": "h",
    "eyech": "h",
    "eye": "i",
    "jay": "j",
    "kay": "k",
    "el": "l",
    "ell": "l",
    "em": "m",
    "en": "n",
    "oh": "o",
    "pee": "p",
    "cue": "q",
    "queue": "q",
    "are": "r",
    "ess": "s",
    "as": "s",
    "tee": "t",
    "you": "u",
    "vee": "v",
    "doubleyou": "w",
    "ex": "x",
    "why": "y",
    "zee": "z",
    "zed": "z",
}


def _strip_dob_command_words(spoken: str) -> str:
    """Remove change/date-of-birth command words, leaving the spoken name."""
    text = (spoken or "").strip()
    text = re.sub(
        r"\b("
        r"change|correct|fix|edit|update|please|"
        r"date of birth|date|birth|birthday|dob|"
        r"for|of|the|to|whose|who|person|member"
        r")\b",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip(" .,!?")


def _normalize_spoken_initials(text: str) -> str:
    """Map ASR initial spellings onto single letters (\"Agent Edge\" → \"Agent H\").

    Only rewrites the final token of a multi-word name so ordinary words are
    not corrupted (e.g. \"are\" / \"as\" mid-sentence).
    """
    from .live_form import _spoken_member_query

    cleaned = _spoken_member_query(text or "")
    parts = cleaned.split()
    if len(parts) >= 2:
        key = parts[-1].lower().rstrip(".")
        alias = _INITIAL_ASR_ALIASES.get(key)
        if alias:
            parts[-1] = alias.upper()
    return " ".join(parts).strip()


def _title_member_label(label: str) -> str:
    parts = (label or "").split()
    out: list[str] = []
    for part in parts:
        if len(part) == 1:
            out.append(part.upper())
        else:
            out.append(part[:1].upper() + part[1:])
    return " ".join(out).strip()


def _extract_spoken_member_labels(spoken: str) -> list[str]:
    """Best-effort member labels from speech when the DOM scrape is empty."""
    from .live_form import _split_member_queries

    cleaned = _strip_dob_command_words(spoken)
    if not cleaned:
        return []
    cleaned = _normalize_spoken_initials(cleaned)
    queries = _split_member_queries(cleaned) or [cleaned]
    out: list[str] = []
    for query in queries:
        label = _title_member_label(_normalize_spoken_initials(query))
        if label and label not in out:
            out.append(label)
    return out


def _match_named_members(
    spoken: str, available: list[str]
) -> tuple[list[str], str | None]:
    """Match spoken names to available member labels. Clarify on duplicates.

    When ``available`` is empty, returns cleaned spoken labels so Collect Mode
    can still target \"Change Agent H\" without waiting for a perfect scrape.
    """
    from .live_form import _normalize_member_label, _split_member_queries

    cleaned = _strip_dob_command_words(spoken)
    cleaned = _normalize_spoken_initials(cleaned)
    if not cleaned:
        return [], None
    queries = _split_member_queries(cleaned) or [cleaned]
    queries = [_normalize_spoken_initials(q) for q in queries if q.strip()]
    if not queries:
        return [], None

    if not available:
        # No DOM labels — still accept the spoken name as the correction target.
        return _extract_spoken_member_labels(spoken), None

    hits: list[str] = []
    for query in queries:
        q_norm = _normalize_member_label(query)
        if not q_norm:
            continue
        matches: list[str] = []
        for name in available:
            n_norm = _normalize_member_label(name)
            if not n_norm:
                continue
            if (
                q_norm == n_norm
                or q_norm in n_norm
                or n_norm.startswith(q_norm)
                or n_norm.startswith(q_norm.split()[0])
            ):
                # Prefer first-token + initial matches: "agent h" ↔ "agent h."
                q_parts = q_norm.split()
                n_parts = n_norm.split()
                if len(q_parts) >= 2 and len(n_parts) >= 2:
                    if q_parts[0] == n_parts[0] and q_parts[1][:1] == n_parts[1][:1]:
                        if name not in matches:
                            matches.append(name)
                        continue
                if name not in matches:
                    matches.append(name)
        # Tight pass: first name only when unique among available.
        if not matches and " " not in q_norm:
            first_hits = [
                name
                for name in available
                if _normalize_member_label(name).split()[:1] == [q_norm]
            ]
            matches = list(dict.fromkeys(first_hits))
        uniq = list(dict.fromkeys(matches))
        if len(uniq) > 1:
            return [], (
                "I see more than one person matching "
                f"{query}. Please say the full name as shown on the application."
            )
        if len(uniq) == 1 and uniq[0] not in hits:
            hits.append(uniq[0])
    return hits, None


def _begin_dob_collect_for(
    session: Any, names: list[str]
) -> tuple[str, str]:
    """Arm Collect Mode for the first named member and return (step, speech)."""
    cleaned = [n.strip() for n in names if (n or "").strip()]
    if not cleaned:
        speech = app_question_for_step("app_dob_review")
        return "app_dob_review", speech
    current = cleaned[0]
    _set_dob_queue(session, cleaned)
    session.filled["app_dob_current"] = current
    session.filled["app_dob_correcting"] = current
    speech = _dob_question_for(current)
    _set_pending_question(session, "app_dob_collect", speech)
    return "app_dob_collect", speech


async def _prepare_dob_collect_speech(
    session: Any, page: Page
) -> tuple[str, str]:
    """Inspect DOB cards and return (step, spoken) for resume or entry."""
    cards = await _scrape_dob_cards(page)
    missing = [
        str(card.get("name") or "").strip()
        for card in cards
        if card.get("needs_add") and str(card.get("name") or "").strip()
    ]
    present = [
        card
        for card in cards
        if not card.get("needs_add") and str(card.get("dob") or "").strip()
    ]
    if missing:
        _set_dob_queue(session, missing)
        current = missing[0]
        session.filled["app_dob_current"] = current
        speech = _dob_question_for(current)
        if len(missing) > 1:
            speech = (
                f"I still need dates of birth for {len(missing)} people. " + speech
            )
        _set_pending_question(session, "app_dob_collect", speech)
        return "app_dob_collect", speech
    _set_dob_queue(session, None)
    if present:
        bits = []
        for card in present[:4]:
            bits.append(f"{card.get('name')} {card.get('dob')}")
        summary = "; ".join(bits)
        speech = (
            f"I see these dates of birth on the application: {summary}. "
            "Do they look correct? Say Yes to continue, or tell me whose "
            "date of birth to change."
        )
        _set_pending_question(session, "app_dob_review", speech)
        return "app_dob_review", speech
    speech = app_question_for_step("app_dob_review")
    _set_pending_question(session, "app_dob_review", speech)
    return "app_dob_review", speech


async def _prepare_resume_speech(
    session: Any, page: Page, surface: str
) -> tuple[str, str]:
    """Align resume step with live UI and return (step, spoken)."""
    if surface in ("app_dob_review", "app_dob_collect"):
        return await _prepare_dob_collect_speech(session, page)
    if surface in ("app_immigration", "app_immigration_who"):
        members = await _scrape_immigration_members(page)
        names = [
            str(m.get("name") or "").strip()
            for m in members
            if str(m.get("name") or "").strip()
        ]
        if names:
            # Do not trust preselected radios — walk every member.
            _set_immigration_queue(session, names)
            current = names[0]
            session.filled["app_immigration_current"] = current
            session.filled["app_citizenship_no"] = "1"
            speech = (
                "This application is on the citizenship and immigration screen. "
                + _immigration_status_question_for(current)
            )
            _set_pending_question(session, "app_immigration", speech)
            return "app_immigration", speech
        speech = app_question_for_step("app_immigration_who")
        _set_pending_question(session, "app_immigration_who", speech)
        return "app_immigration_who", speech
    if surface in ("app_insurance_members", "app_insurance_coverage_type"):
        current = (session.filled.get("app_insurance_current") or "").strip()
        queue = _insurance_queue(session)
        if surface == "app_insurance_coverage_type" or current or queue:
            target = current or (queue[0] if queue else "")
            if target:
                session.filled["app_insurance_current"] = target
                speech = _insurance_coverage_question_for(target)
                _set_pending_question(session, "app_insurance_coverage_type", speech)
                return "app_insurance_coverage_type", speech
        speech = app_question_for_step("app_insurance_members")
        _set_pending_question(session, "app_insurance_members", speech)
        return "app_insurance_members", speech
    if surface == "app_insurance":
        # Fresh insurance screen — ask Yes/No, not member selection.
        speech = app_question_for_step("app_insurance")
        _set_pending_question(session, "app_insurance", speech)
        return "app_insurance", speech
    speech = app_question_for_step(surface)
    _set_pending_question(session, surface, speech)
    return surface, speech


def _unidentified_screen_error(
    call_id: str,
    session: Any,
    answer: str = "",
) -> dict[str, Any]:
    return {
        "ok": False,
        "call_id": call_id,
        "field_id": "application_answer",
        "value": answer,
        "error": "stage_e_screen_unidentified",
        "spoken": (
            "ASK: I'm having trouble identifying the current application screen. "
            "Please stay on this page and say repeat in a moment so I can try again."
        ),
        "message": "Could not identify Stage E screen from live UI",
        "step": session.app_step or "",
        "phase": "application",
    }


def _map_immigration_status(lowered: str) -> str | None:
    if "prefer not" in lowered or "prefer not to answer" in lowered:
        return "Prefer not to answer"
    if (
        "satisfactory" in lowered
        or "green card" in lowered
        or "permanent resident" in lowered
        or re.search(r"\bvisa\b", lowered)
    ):
        return "Satisfactory immigration status"
    if re.search(r"\bnational\b", lowered):
        return "U.S. National"
    if re.search(r"\bcitizen\b", lowered):
        return "U.S. Citizen"
    return None


def _map_income_type(lowered: str) -> str | None:
    income_types = (
        "Self-employed",
        "Employed",
        "Unemployed",
        "SSA",
        "SSDI",
        "Pension",
        "Alimony",
        "Retirement",
        "Rental",
        "Worker's comp",
        "Other",
        "None",
    )
    if "worker" in lowered:
        return "Worker's comp"
    if "self" in lowered:
        return "Self-employed"
    for label in income_types:
        if label.lower() in lowered:
            return label
    return None


def _map_circumstance_labels(lowered: str) -> list[str]:
    mapping = (
        (
            "needs help with daily activities",
            ("daily activit", "daily help", "help with daily"),
        ),
        ("veteran or military", ("veteran", "military")),
        (
            "American Indian or Alaska Native",
            ("american indian", "alaska native", "native american"),
        ),
        ("full-time student", ("full-time student", "full time student", "student")),
        ("None of the above", ("none of the above", "none")),
    )
    hits: list[str] = []
    for label, cues in mapping:
        if any(cue in lowered for cue in cues):
            if label not in hits:
                hits.append(label)
    # Prefer explicit none alone.
    if "None of the above" in hits and len(hits) > 1:
        hits = [item for item in hits if item != "None of the above"]
    return hits


async def _start_income_seq(
    session: Any, kind: str
) -> tuple[str, str]:
    """Begin sequential income detail questions. Returns (step, spoken)."""
    seq = _INCOME_EMPLOYED_SEQ if kind == "employed" else _INCOME_SELF_SEQ
    first_key, first_step, first_q = seq[0]
    _set_income_seq(
        session,
        {"kind": kind, "index": 0, "values": {}, "keys": [item[0] for item in seq]},
    )
    return first_step, first_q


async def _continue_income_seq(
    call_id: str,
    session: Any,
    page: Page,
    answer: str,
) -> dict[str, Any]:
    """Store one income detail field, then ask the next or save."""
    from .live_form import _click_cta

    state = _income_seq_state(session)
    kind = str(state.get("kind") or "")
    if kind == "employed":
        seq = _INCOME_EMPLOYED_SEQ
    elif kind == "other":
        seq = (
            (
                "pay",
                "app_income_self",
                "What amount should I enter for that income?",
            ),
        )
    else:
        seq = _INCOME_SELF_SEQ
    index = int(state.get("index") or 0)
    values = dict(state.get("values") or {})
    if index < 0 or index >= len(seq):
        _set_income_seq(session, None)
        return _app_ok(
            call_id,
            session,
            answer,
            "app_income_more",
            "Income sequence reset",
            spoken_override=app_question_for_step("app_income_more"),
        )
    key, _step_id, _q = seq[index]
    text = (answer or "").strip()
    if not text:
        return {
            "ok": False,
            "call_id": call_id,
            "field_id": "application_answer",
            "spoken": "ASK: " + app_question_for_step(session.app_step or _step_id),
            "error": "income_field_empty",
            "step": session.app_step or _step_id,
            "phase": "application",
        }
    values[key] = text
    filled = await _fill_next_visible_input(page, text)
    if not filled and key in ("pay", "hours"):
        # Numeric fields should land in an input; ask again if nothing took.
        return {
            "ok": False,
            "call_id": call_id,
            "field_id": "application_answer",
            "spoken": (
                "ASK: I couldn't enter that value on the form. "
                + app_question_for_step(session.app_step or _step_id)
            ),
            "error": "income_field_not_filled",
            "step": session.app_step or _step_id,
            "phase": "application",
        }
    next_index = index + 1
    if next_index < len(seq):
        nkey, nstep, nq = seq[next_index]
        _set_income_seq(
            session,
            {
                "kind": kind,
                "index": next_index,
                "values": values,
                "keys": [item[0] for item in seq],
            },
        )
        return _app_ok(
            call_id,
            session,
            answer,
            nstep,
            f"Collected income field {key}",
            spoken_override=nq,
        )
    # All fields collected — save and ask about another source.
    try:
        await _click_cta(page, "Save", attempts=2)
    except Exception:
        pass
    _set_income_seq(session, None)
    return _app_ok(
        call_id,
        session,
        answer,
        "app_income_more",
        "Saved income details",
        spoken_override=app_question_for_step("app_income_more"),
    )


def _app_ok(
    call_id: str,
    session: Any,
    answer: str,
    next_step: str,
    message: str,
    *,
    spoken_override: str | None = None,
    end_call: bool = False,
) -> dict[str, Any]:
    session.app_step = next_step
    if next_step:
        session.elig_step = session.elig_step  # leave Stage D marker alone if any
    if spoken_override is not None:
        spoken = spoken_override
    elif next_step:
        spoken = app_question_for_step(next_step, session)
    else:
        spoken = CLOSING_SUBMITTED if end_call else CLOSING_NOT_SUBMITTED
    if next_step and spoken:
        _set_pending_question(session, next_step, spoken)
    elif not next_step:
        _set_pending_question(session, "", "")
    return {
        "ok": True,
        "call_id": call_id,
        "field_id": "application_answer",
        "value": answer,
        "action": "verified",
        "message": message,
        "spoken": "ASK: " + spoken,
        "step": next_step,
        "phase": "application" if next_step else "closing",
        "end_call": end_call and not next_step,
    }


_STAGE_E_YES_LEAD = re.compile(r"^\s*yes(?:\b|[\s—–\-:,.])", re.I)
_STAGE_E_NO_LEAD = re.compile(r"^\s*no(?:\b|[\s—–\-:,.])", re.I)


def resolve_stage_e_yes_no_label(yes: bool, labels: list[str]) -> str:
    """Map spoken Yes/No onto one visible Stage E option, including long cards.

    Matches exact ``Yes`` / ``No`` and descriptive cards such as
    ``Yes Everyone listed is a U.S. Citizen``. Does not change Stage D matching.
    """
    polarity = "yes" if yes else "no"
    lead = _STAGE_E_YES_LEAD if yes else _STAGE_E_NO_LEAD
    hits: list[str] = []
    for raw in labels or []:
        text = re.sub(r"\s+", " ", str(raw or "").strip())
        if not text or len(text) > 140:
            continue
        lowered = text.lower()
        if lowered == polarity or lead.match(text):
            if text not in hits:
                hits.append(text)
    if not hits:
        raise RuntimeError(
            f"YES_NO_CLARIFY: I could not find a visible {polarity!r} option "
            "for this question. Please say Yes or No again."
        )
    if len(hits) == 1:
        return hits[0]
    # Bare Yes/No plus one descriptive card for the same polarity → use the card.
    descriptive = [item for item in hits if item.lower() != polarity]
    if len(descriptive) == 1:
        return descriptive[0]
    raise RuntimeError(
        f"YES_NO_CLARIFY: Several {polarity!r} options are visible. "
        "Please say the full choice, or say Yes or No again."
    )


async def _click_stage_e_option_label(page: Page, label: str) -> bool:
    """Click the control whose full visible text matches ``label``.

    Does not use ``_choice_phrase`` / ``_click_exact_choice``, so descriptive
    Stage E cards keep their full text (e.g. citizenship Yes cards).
    """
    try:
        clicked = await page.evaluate(
            """(label) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const want = norm(label);
              if (!want) return false;
              const nodes = [...document.querySelectorAll(
                'button, [role=button], a, div, span, label'
              )];
              const hits = [];
              for (const el of nodes) {
                const r = el.getBoundingClientRect();
                if (r.width < 8 || r.height < 8) continue;
                const text = norm(el.innerText || el.textContent || '');
                if (text !== want || text.length > 160) continue;
                hits.push({ el, area: r.width * r.height });
              }
              hits.sort((a, b) => a.area - b.area);
              if (!hits.length) return false;
              hits[0].el.click();
              return true;
            }""",
            label,
        )
        if clicked:
            return True
    except Exception:
        logger.exception("stage_e option DOM click failed label=%r", label[:80])
    try:
        loc = page.get_by_text(label, exact=True).first
        if await loc.count() and await loc.is_visible():
            await loc.click(timeout=6000)
            return True
    except Exception:
        pass
    return False


async def _click_yes_no(page: Page, yes: bool) -> str:
    """Click the unique Stage E Yes/No option and verify it took effect."""
    from .live_form import _choice_signature, _visible_choice_labels

    labels = await _visible_choice_labels(page)
    target = resolve_stage_e_yes_no_label(yes, labels)
    before = await _choice_signature(page, target)
    screen_before = await detect_application_screen(page)
    logger.info(
        "stage_e_yes_no_click polarity=%s target=%r screen=%s",
        "yes" if yes else "no",
        target[:80],
        screen_before or "-",
    )
    clicked = await _click_stage_e_option_label(page, target)
    if not clicked:
        raise RuntimeError(
            "YES_NO_CLARIFY: I could not select that Yes or No option. "
            "Please say Yes or No again."
        )
    await page.wait_for_timeout(450)
    after = await _choice_signature(page, target)
    screen_after = await detect_application_screen(page)
    if screen_after and screen_before and screen_after != screen_before:
        return target
    if after != before and after != "__missing__":
        return target
    if after and after != "__missing__":
        return target
    # Successful navigation often removes the old card entirely.
    if after == "__missing__":
        await page.wait_for_timeout(500)
        screen_after = await detect_application_screen(page)
        if screen_after and screen_after != screen_before:
            return target
        new_labels = await _visible_choice_labels(page)
        if target not in new_labels:
            return target
    raise RuntimeError(
        "YES_NO_CLARIFY: That Yes or No choice did not register. "
        "Please say Yes or No again."
    )


def _yes_no_click_failure(
    call_id: str,
    session: Any,
    step: str,
    answer: str,
    exc: BaseException,
) -> dict[str, Any]:
    # Never speak raw targets, bodies, or exception dumps to the caller.
    _ = exc
    return {
        "ok": False,
        "call_id": call_id,
        "field_id": "application_answer",
        "value": answer,
        "error": "yes_no_click_failed",
        "spoken": (
            "ASK: Please say Yes or No again. " + app_question_for_step(step)
        ),
        "message": "yes_no_click_failed",
        "step": step,
        "phase": "application",
    }


# Backend-only Collect Mode sub-steps that share a parent Pointcare screen.
_APP_SUBSTEPS: dict[str, str] = {
    "app_immigration_who": "app_immigration",
    "app_insurance_members": "app_insurance",
    "app_insurance_coverage_type": "app_insurance",
    "app_dob_collect": "app_dob_review",
    "app_circumstances_person": "app_circumstances",
    "app_income_employed": "app_income",
    "app_income_employed_hours": "app_income",
    "app_income_employed_frequency": "app_income",
    "app_income_employed_employer": "app_income",
    "app_income_self": "app_income",
    "app_income_self_work": "app_income",
    "app_income_more": "app_income",
    "app_expenses_amount": "app_expenses",
}


async def _reconcile_app_step(session: Any, page: Page) -> str:
    """Align session.app_step with the visible Stage E screen before answering."""
    current = (session.app_step or "").strip()
    screen = await detect_application_screen(page)
    if screen and screen.startswith("app_") and screen != "programs":
        parent = _APP_SUBSTEPS.get(current, "")
        # Keep Collect Mode sub-steps while still on the parent Pointcare screen.
        if parent and (screen == parent or screen == current):
            return current
        # Immigration who/status stay while live screen reports immigration.
        if current in ("app_immigration_who", "app_immigration") and screen in (
            "app_immigration",
            "app_citizenship",
        ):
            # Citizenship Yes/No still visible after No expands members — keep
            # immigration collection once the caller already answered No.
            if current.startswith("app_immigration") and session.filled.get(
                "app_citizenship_no"
            ):
                return current
        # Insurance member/coverage sub-steps stay while Yes expands the form.
        if current in (
            "app_insurance_members",
            "app_insurance_coverage_type",
        ) and screen in ("app_insurance", "app_insurance_members"):
            return current
        if current != screen:
            logger.info(
                "stage_e_reconcile call_id=%s from=%s to=%s",
                getattr(session, "call_id", "-"),
                current or "-",
                screen,
            )
            session.app_step = screen
        return session.app_step
    # Never invent citizenship when the UI cannot be identified.
    return current or screen or ""


async def _recover_if_browser_advanced(
    call_id: str,
    session: Any,
    page: Page,
    answer: str,
    expected_step: str,
) -> dict[str, Any] | None:
    """If Pointcare already left ``expected_step``, return the next question."""
    screen = await detect_application_screen(page)
    if (
        screen
        and screen.startswith("app_")
        and screen not in ("programs", expected_step)
    ):
        logger.info(
            "stage_e_yes_no_recovered_after_nav call_id=%s from=%s to=%s",
            call_id,
            expected_step,
            screen,
        )
        step, speech = await _prepare_resume_speech(session, page, screen)
        return _app_ok(
            call_id,
            session,
            answer,
            step,
            "Recovered after page advanced",
            spoken_override=speech,
        )
    return None


async def _click_next_or_continue(page: Page) -> bool:
    """Click Next/Continue only when enabled. Returns True if a click was attempted."""
    from .live_form import _click_cta

    enabled = await _next_button_enabled(page)
    if enabled is False:
        logger.info("stage_e_next_disabled skip_click")
        return False
    for label in ("Next", "Continue", "Save and Return", "Save"):
        try:
            await _click_cta(page, label, attempts=1)
            await page.wait_for_timeout(600)
            return True
        except Exception:
            continue
    return False


async def _advance_after_answer(page: Page, session: Any) -> str:
    """Detect the next app step after a verified browser action."""
    await page.wait_for_timeout(700)
    screen = await detect_application_screen(page)
    if screen == "programs":
        # Still on summary — unexpected mid Stage E.
        return session.app_step or ""
    if screen:
        return screen
    # Try Next once only when enabled and the page looks stuck.
    try:
        await _click_next_or_continue(page)
    except Exception:
        pass
    screen = await detect_application_screen(page)
    return screen or (session.app_step or "")


async def handle_programs_decision(
    call_id: str,
    session: Any,
    page: Page,
    answer: str,
) -> dict[str, Any]:
    """Yes → Start Applications + first Stage E question; No → polite close; else clarify."""
    from .live_form import classify_yes_no

    lowered = (answer or "").strip().lower()
    decision = classify_yes_no(answer)
    wants_start = decision == "yes" or bool(
        re.search(r"\b(continue|start|proceed|application)\b", lowered)
        and decision != "no"
    )
    # "start the application" without a clear No counts as Yes.
    if decision is None and re.search(
        r"\b(start (the )?application|continue to (the )?application)\b", lowered
    ):
        wants_start = True
    if decision is None and not wants_start:
        summary = summary_from_session_filled(session.filled)
        ctx = programs_context_reply(answer, summary)
        speech = ctx or (
            (session.filled.get("programs_permission_speech") or "").strip()
            or build_programs_permission_speech(summary)
        )
        if not ctx:
            speech = f"{PROGRAMS_CLARIFY} {speech}"
        return {
            "ok": False,
            "call_id": call_id,
            "field_id": "eligibility_answer",
            "value": answer,
            "error": "programs_clarify",
            "spoken": "ASK: " + speech,
            "message": "programs clarify",
            "step": "programs",
            "phase": "eligibility",
        }

    if not wants_start or decision == "no":
        # Do not click Start Applications or Back — leave eligibility results intact.
        session.filled["application_started"] = "0"
        session.app_step = ""
        session.elig_step = ""
        logger.info(
            "programs_declined call_id=%s leave_results=1",
            call_id,
        )
        return {
            "ok": True,
            "call_id": call_id,
            "field_id": "eligibility_answer",
            "value": answer,
            "action": "declined_application",
            "message": "Caller declined to start the application",
            "spoken": "ASK: " + CLOSING_NO_APPLICATION,
            "step": "",
            "phase": "closing",
            "end_call": True,
        }

    # Yes — start application once.
    if session.filled.get("application_started") == "1" and session.app_step:
        return _app_ok(
            call_id,
            session,
            answer,
            session.app_step,
            "Application already started",
            spoken_override=app_question_for_step(session.app_step),
        )

    clicked = await click_start_applications(page)
    started = await wait_for_application_started(page)
    if not started:
        logger.error(
            "start_applications_not_verified call_id=%s clicked=%s url=%s",
            call_id,
            clicked,
            (page.url or "")[:120],
        )
        return {
            "ok": False,
            "call_id": call_id,
            "field_id": "eligibility_answer",
            "value": answer,
            "error": "start_applications_failed",
            "spoken": (
                "ASK: I couldn't confirm that the application started. "
                + PROGRAMS_CLARIFY
            ),
            "message": "Start Applications click not verified",
            "step": "programs",
            "phase": "eligibility",
        }

    session.filled["application_started"] = "1"
    session.filled["start_applications_clicked"] = clicked
    session.elig_step = ""
    screen = await detect_application_screen(page)
    next_step = screen if screen and screen != "programs" else "app_citizenship"
    session.app_step = next_step
    intro = (
        "Okay, I've started the application. "
        + app_question_for_step(next_step)
    )
    logger.info(
        "stage_e_started call_id=%s clicked=%s first_step=%s",
        call_id,
        clicked,
        next_step,
    )
    return _app_ok(
        call_id,
        session,
        answer,
        next_step,
        f"Started application via {clicked}",
        spoken_override=intro,
    )


async def submit_application_answer(
    call_id: str,
    session: Any,
    page: Page,
    answer: str,
    started: float,
) -> dict[str, Any]:
    """Apply one Collect Mode Stage E answer and return the next question."""
    from .live_form import (
        _click_cta,
        _click_exact_choice,
        _is_no,
        _is_yes,
        _mark,
        classify_yes_no,
    )

    # Browser may already have advanced (e.g. delayed Yes after citizenship).
    step = await _reconcile_app_step(session, page)
    if step == "programs":
        return _unidentified_screen_error(call_id, session, answer)
    if not step:
        # Keep a prior Collect Mode sub-step only if already set; otherwise fail
        # closed instead of inventing citizenship.
        prior = (session.app_step or "").strip()
        if prior.startswith("app_"):
            step = prior
        else:
            return _unidentified_screen_error(call_id, session, answer)
    session.app_step = step
    text = (answer or "").strip()
    lowered = text.lower()
    _mark(call_id, "browser_action", started)

    # Contextual / unclear on yes-no steps — do not navigate.
    if step in YES_NO_APP_STEPS:
        decision = classify_yes_no(text)
        if decision is None and step == "app_submit_confirm":
            if re.search(r"\b(review|check|go back|change|correct)\b", lowered):
                return _app_ok(
                    call_id,
                    session,
                    text,
                    step,
                    "submit deferred for review",
                    spoken_override=(
                        "Okay, I will not submit yet. Tell me what you'd like to "
                        "review or change, or say Yes when you want me to submit."
                    ),
                )
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "application_answer",
                "spoken": "ASK: " + app_question_for_step(step),
                "error": "submit_clarify",
                "step": step,
                "phase": "application",
            }

    next_step = step
    spoken_override: str | None = None
    end_call = False

    if step == "app_agent_offer":
        if _is_yes(text):
            try:
                await _click_exact_choice(page, "Yes, connect me")
            except Exception:
                try:
                    await _click_yes_no(page, True)
                except RuntimeError as exc:
                    return _yes_no_click_failure(call_id, session, step, text, exc)
        elif _is_no(text):
            clicked_no = False
            for label in ("No thanks", "No", "Continue without an agent"):
                try:
                    await _click_cta(page, label, attempts=1)
                    clicked_no = True
                    break
                except Exception:
                    continue
            if not clicked_no:
                try:
                    await _click_yes_no(page, False)
                except RuntimeError as exc:
                    return _yes_no_click_failure(call_id, session, step, text, exc)
        else:
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "application_answer",
                "spoken": "ASK: " + app_question_for_step(step),
                "error": "agent_offer_clarify",
                "step": step,
                "phase": "application",
            }
        try:
            await _click_cta(page, "Continue", attempts=2)
        except Exception:
            await _click_next_or_continue(page)
        next_step = await _advance_after_answer(page, session)

    elif step == "app_citizenship":
        if _is_yes(text):
            session.filled.pop("app_citizenship_no", None)
            try:
                await _click_yes_no(page, True)
            except RuntimeError as exc:
                recovered = await _recover_if_browser_advanced(
                    call_id, session, page, text, step
                )
                if recovered is not None:
                    return recovered
                return _yes_no_click_failure(call_id, session, step, text, exc)
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
        elif _is_no(text):
            try:
                await _click_yes_no(page, False)
            except RuntimeError as exc:
                recovered = await _recover_if_browser_advanced(
                    call_id, session, page, text, step
                )
                if recovered is not None:
                    return recovered
                return _yes_no_click_failure(call_id, session, step, text, exc)
            session.filled["app_citizenship_no"] = "1"
            await page.wait_for_timeout(500)
            members = await _scrape_immigration_members(page)
            # Do not treat preselected U.S. Citizen radios as verified answers.
            # Walk every household member shown by Pointcare.
            names = [
                str(m.get("name") or "").strip()
                for m in members
                if str(m.get("name") or "").strip()
            ]
            # Detect duplicate display names before starting the walk.
            lowered_names = [n.lower() for n in names]
            if len(set(lowered_names)) < len(lowered_names):
                next_step = "app_immigration_who"
                spoken_override = (
                    "I see more than one person with the same name on this "
                    "screen. Please say the full name as shown for each person "
                    "whose status I should set, one at a time."
                )
                _set_pending_question(session, next_step, spoken_override)
            elif names:
                _set_immigration_queue(session, names)
                current = names[0]
                session.filled["app_immigration_current"] = current
                next_step = "app_immigration"
                spoken_override = (
                    "Okay. I'll confirm each person's citizenship or immigration "
                    "status one at a time. "
                    + _immigration_status_question_for(current)
                )
                _set_pending_question(session, next_step, spoken_override)
            else:
                next_step = "app_immigration_who"
                spoken_override = app_question_for_step("app_immigration_who")
                _set_pending_question(session, next_step, spoken_override)
        else:
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "application_answer",
                "spoken": "ASK: " + app_question_for_step(step, session),
                "error": "citizenship_clarify",
                "step": step,
                "phase": "application",
            }

    elif step == "app_immigration_who":
        if re.search(r"\b(continue|next|done|that's all|finished)\b", lowered):
            enabled = await _next_button_enabled(page)
            if enabled is False:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: Pointcare still needs immigration status for one or "
                        "more people before I can continue. "
                        + app_question_for_step("app_immigration_who", session)
                    ),
                    "error": "immigration_incomplete",
                    "step": step,
                    "phase": "application",
                }
            clicked = await _click_next_or_continue(page)
            if not clicked and enabled is False:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: Next is still disabled. "
                        + app_question_for_step("app_immigration_who", session)
                    ),
                    "error": "immigration_next_disabled",
                    "step": step,
                    "phase": "application",
                }
            session.filled.pop("app_citizenship_no", None)
            _set_immigration_queue(session, None)
            next_step = await _advance_after_answer(page, session)
        else:
            members = await _scrape_immigration_members(page)
            available = [
                str(m.get("name") or "").strip()
                for m in members
                if str(m.get("name") or "").strip()
            ]
            if not available:
                # Fall back to free-form name tokens when scrape finds nothing.
                available = []
            hits, clarify = _match_named_members(text, available or [])
            if clarify:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + clarify,
                    "error": "immigration_who_ambiguous",
                    "step": step,
                    "phase": "application",
                }
            if not hits:
                # Accept spoken names even when scrape missed cards.
                tokens = [
                    tok
                    for tok in re.findall(r"[A-Za-z][A-Za-z'.-]{1,}", text)
                    if tok.lower()
                    not in {
                        "and",
                        "the",
                        "not",
                        "citizen",
                        "citizens",
                        "are",
                        "is",
                        "both",
                    }
                ]
                hits = tokens
            if not hits:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + app_question_for_step(step, session),
                    "error": "immigration_who_clarify",
                    "step": step,
                    "phase": "application",
                }
            _set_immigration_queue(session, hits)
            current = hits[0]
            session.filled["app_immigration_current"] = current
            next_step = "app_immigration"
            spoken_override = _immigration_status_question_for(current)
            _set_pending_question(session, next_step, spoken_override)

    elif step == "app_immigration":
        if re.search(r"\b(continue|next|done|that's all|finished)\b", lowered):
            enabled = await _next_button_enabled(page)
            if enabled is False:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: Pointcare still needs an immigration status before "
                        "I can continue. "
                        + app_question_for_step(
                            "app_immigration",
                            session,
                        )
                    ),
                    "error": "immigration_incomplete",
                    "step": step,
                    "phase": "application",
                }
            await _click_next_or_continue(page)
            session.filled.pop("app_citizenship_no", None)
            _set_immigration_queue(session, None)
            next_step = await _advance_after_answer(page, session)
        else:
            picked = _map_immigration_status(lowered)
            if not picked:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + app_question_for_step(step, session),
                    "error": "immigration_clarify",
                    "step": step,
                    "phase": "application",
                }
            current = (session.filled.get("app_immigration_current") or "").strip()
            queue = _immigration_queue(session)
            if current:
                # Duplicate-name guard: never click the wrong card.
                live_members = await _scrape_immigration_members(page)
                same = [
                    m
                    for m in live_members
                    if (m.get("name") or "").strip().lower() == current.lower()
                ]
                if len(same) > 1:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: I see more than one person named "
                            f"{current}. Please say the full name exactly as "
                            "shown on the card."
                        ),
                        "error": "immigration_duplicate_name",
                        "step": step,
                        "phase": "application",
                    }
                ok = await _set_member_immigration_status(page, current, picked)
                if not ok:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: I couldn't set that status on the form. "
                            + _immigration_status_question_for(current)
                        ),
                        "error": "immigration_status_not_set",
                        "step": step,
                        "phase": "application",
                    }
                await page.wait_for_timeout(250)
                verified = await _verify_member_immigration_status(
                    page, current, picked
                )
                if not verified:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: I couldn't verify that status was selected. "
                            + _immigration_status_question_for(current)
                        ),
                        "error": "immigration_status_not_verified",
                        "step": step,
                        "phase": "application",
                    }
                remaining = [name for name in queue if name.lower() != current.lower()]
                _set_immigration_queue(session, remaining or None)
                if remaining:
                    nxt = remaining[0]
                    session.filled["app_immigration_current"] = nxt
                    next_step = "app_immigration"
                    spoken_override = (
                        f"Okay, {picked} for {current}. "
                        + _immigration_status_question_for(nxt)
                    )
                    _set_pending_question(session, next_step, spoken_override)
                else:
                    session.filled.pop("app_immigration_current", None)
                    enabled = await _next_button_enabled(page)
                    if enabled is not False:
                        clicked = await _click_next_or_continue(page)
                        if clicked:
                            session.filled.pop("app_citizenship_no", None)
                            next_step = await _advance_after_answer(page, session)
                        else:
                            next_step = "app_immigration_who"
                            spoken_override = (
                                f"Okay, {picked} for {current}. "
                                "Say continue when you're ready, or name someone "
                                "to correct."
                            )
                            _set_pending_question(
                                session, next_step, spoken_override
                            )
                    else:
                        next_step = "app_immigration_who"
                        spoken_override = (
                            f"Okay, {picked} for {current}. "
                            "Pointcare still needs a status before Next is "
                            "available. Say a name to correct, or say continue "
                            "when ready."
                        )
                        _set_pending_question(session, next_step, spoken_override)
            else:
                # No active member — ask who first rather than clicking globally.
                next_step = "app_immigration_who"
                spoken_override = (
                    "Please tell me which household member that status is for. "
                    + app_question_for_step("app_immigration_who")
                )
                _set_pending_question(session, next_step, spoken_override)

    elif step == "app_insurance":
        if _is_yes(text):
            try:
                await _click_yes_no(page, True)
            except RuntimeError as exc:
                recovered = await _recover_if_browser_advanced(
                    call_id, session, page, text, step
                )
                if recovered is not None:
                    return recovered
                return _yes_no_click_failure(call_id, session, step, text, exc)
            await page.wait_for_timeout(400)
            # Some Pointcare flows ask which members have coverage next.
            screen = await detect_application_screen(page)
            if screen and screen not in ("", "app_insurance", "programs"):
                next_step = await _advance_after_answer(page, session)
            else:
                members = await _scrape_insurance_members(page)
                names = [
                    str(m.get("name") or "").strip()
                    for m in members
                    if str(m.get("name") or "").strip()
                ]
                next_step = "app_insurance_members"
                if names:
                    spoken_override = (
                        "Okay. I see household members listed for coverage. "
                        + app_question_for_step(next_step)
                    )
                else:
                    spoken_override = app_question_for_step(next_step)
                _set_pending_question(session, next_step, spoken_override)
        elif _is_no(text):
            try:
                await _click_yes_no(page, False)
            except RuntimeError as exc:
                recovered = await _recover_if_browser_advanced(
                    call_id, session, page, text, step
                )
                if recovered is not None:
                    return recovered
                return _yes_no_click_failure(call_id, session, step, text, exc)
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
        else:
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "application_answer",
                "spoken": "ASK: " + app_question_for_step(step),
                "error": "insurance_clarify",
                "step": step,
                "phase": "application",
            }

    elif step == "app_insurance_members":
        if re.search(r"\b(continue|next|done|that's all)\b", lowered) or _is_yes(text):
            rows = await _scrape_insurance_members(page)
            selected = [
                str(r.get("name") or "").strip()
                for r in rows
                if r.get("checked") and str(r.get("name") or "").strip()
            ]
            # Prefer the queue built while selecting names.
            queued = _insurance_queue(session)
            targets = queued or selected
            if targets:
                _set_insurance_queue(session, targets)
                current = targets[0]
                session.filled["app_insurance_current"] = current
                next_step = "app_insurance_coverage_type"
                spoken_override = _insurance_coverage_question_for(current)
                _set_pending_question(session, next_step, spoken_override)
            else:
                enabled = await _next_button_enabled(page)
                if enabled is False:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: Pointcare still needs at least one covered "
                            "member before I can continue. "
                            + app_question_for_step(step)
                        ),
                        "error": "insurance_members_incomplete",
                        "step": step,
                        "phase": "application",
                    }
                await _click_next_or_continue(page)
                next_step = await _advance_after_answer(page, session)
        else:
            members = await _scrape_insurance_members(page)
            available = [
                str(m.get("name") or "").strip()
                for m in members
                if str(m.get("name") or "").strip()
            ]
            hits, clarify = _match_named_members(text, available or [])
            if clarify:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + clarify,
                    "error": "insurance_members_ambiguous",
                    "step": step,
                    "phase": "application",
                }
            if not hits:
                tokens = [
                    tok
                    for tok in re.findall(r"[A-Za-z][A-Za-z'.-]{1,}", text)
                    if tok.lower()
                    not in {
                        "and",
                        "the",
                        "has",
                        "have",
                        "with",
                        "coverage",
                        "insurance",
                        "both",
                        "all",
                    }
                ]
                hits = tokens
            if not hits:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + app_question_for_step(step),
                    "error": "insurance_members_clarify",
                    "step": step,
                    "phase": "application",
                }
            # Resolve spoken names to scraped labels when possible.
            resolved: list[str] = []
            for hit in hits:
                match = next(
                    (
                        n
                        for n in available
                        if n.lower() == hit.lower()
                        or hit.lower() in n.lower()
                        or n.lower().startswith(hit.lower())
                    ),
                    hit,
                )
                if match not in resolved:
                    resolved.append(match)
            # Duplicate-name guard.
            for name in resolved:
                same = [n for n in available if n.lower() == name.lower()]
                if len(same) > 1:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: I see more than one person named "
                            f"{name}. Please say the full name as shown on the card."
                        ),
                        "error": "insurance_duplicate_name",
                        "step": step,
                        "phase": "application",
                    }
            clicked_any = False
            for name in resolved:
                ok = await _set_member_insurance_checked(page, name, True)
                if ok:
                    clicked_any = True
                else:
                    try:
                        await _click_exact_choice(page, name)
                        clicked_any = True
                    except Exception:
                        continue
            if not clicked_any:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + app_question_for_step(step),
                    "error": "insurance_members_not_set",
                    "step": step,
                    "phase": "application",
                }
            existing = _insurance_queue(session)
            merged = list(existing)
            for name in resolved:
                if name.lower() not in {m.lower() for m in merged}:
                    merged.append(name)
            _set_insurance_queue(session, merged)
            spoken_override = (
                "Okay. Say another name if needed, or say continue when "
                "everyone with coverage is selected."
            )
            next_step = "app_insurance_members"
            _set_pending_question(session, next_step, spoken_override)

    elif step == "app_insurance_coverage_type":
        current = (session.filled.get("app_insurance_current") or "").strip()
        queue = _insurance_queue(session)
        if re.search(r"\b(continue|next|done|that's all|finished)\b", lowered):
            if current:
                # Require at least one coverage type before leaving this member.
                rows = await _scrape_insurance_members(page)
                row = next(
                    (
                        r
                        for r in rows
                        if (r.get("name") or "").strip().lower() == current.lower()
                    ),
                    None,
                )
                selected_types = list((row or {}).get("selected_types") or [])
                if not selected_types:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: I still need a coverage type for "
                            f"{_clean_member_display_name(current)}. "
                            + _insurance_coverage_question_for(current)
                        ),
                        "error": "insurance_coverage_incomplete",
                        "step": step,
                        "phase": "application",
                    }
            remaining = [n for n in queue if n.lower() != current.lower()] if current else queue
            _set_insurance_queue(session, remaining or None)
            if remaining:
                nxt = remaining[0]
                session.filled["app_insurance_current"] = nxt
                next_step = "app_insurance_coverage_type"
                spoken_override = _insurance_coverage_question_for(nxt)
                _set_pending_question(session, next_step, spoken_override)
            else:
                session.filled.pop("app_insurance_current", None)
                enabled = await _next_button_enabled(page)
                if enabled is False:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: Pointcare still needs coverage details before "
                            "I can continue. "
                            + app_question_for_step("app_insurance_members")
                        ),
                        "error": "insurance_coverage_next_disabled",
                        "step": step,
                        "phase": "application",
                    }
                await _click_next_or_continue(page)
                next_step = await _advance_after_answer(page, session)
        else:
            types = _map_coverage_types(lowered)
            if not types:
                target = current or "that person"
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + _insurance_coverage_question_for(target),
                    "error": "insurance_coverage_clarify",
                    "step": step,
                    "phase": "application",
                }
            if not current:
                if queue:
                    current = queue[0]
                    session.filled["app_insurance_current"] = current
                else:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: Whose coverage type is that? "
                            + app_question_for_step("app_insurance_members")
                        ),
                        "error": "insurance_coverage_member_needed",
                        "step": "app_insurance_members",
                        "phase": "application",
                    }
            set_ok = True
            for cov in types:
                ok = await _set_member_coverage_type(page, current, cov)
                if not ok:
                    set_ok = False
                    break
            if not set_ok:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I couldn't select that coverage on the form. "
                        + _insurance_coverage_question_for(current)
                    ),
                    "error": "insurance_coverage_not_set",
                    "step": step,
                    "phase": "application",
                }
            await page.wait_for_timeout(250)
            verified = await _verify_member_coverage_types(page, current, types)
            # Soft-verify: scrape may miss selected_types; keep going if set_ok.
            if not verified:
                # Re-check by scraping; if still empty, ask once more.
                rows = await _scrape_insurance_members(page)
                row = next(
                    (
                        r
                        for r in rows
                        if (r.get("name") or "").strip().lower() == current.lower()
                    ),
                    None,
                )
                if row is not None and not list(row.get("selected_types") or []):
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: I couldn't verify that coverage was selected. "
                            + _insurance_coverage_question_for(current)
                        ),
                        "error": "insurance_coverage_not_verified",
                        "step": step,
                        "phase": "application",
                    }
            # Stay on this member so the caller can add more coverage types.
            label_list = ", ".join(types)
            others = [n for n in queue if n.lower() != current.lower()]
            if others:
                spoken_override = (
                    f"Okay, {label_list} for {current}. "
                    "Say another coverage type for them if needed, or say "
                    "continue to move to the next person."
                )
            else:
                spoken_override = (
                    f"Okay, {label_list} for {current}. "
                    "Say another coverage type if needed, or say continue."
                )
            session.filled["app_insurance_current"] = current
            if current.lower() not in {n.lower() for n in queue}:
                _set_insurance_queue(session, [current] + others)
            next_step = "app_insurance_coverage_type"
            _set_pending_question(session, next_step, spoken_override)

    elif step == "app_dob_review":
        cards = await _scrape_dob_cards(page)
        missing = [
            str(c.get("name") or "").strip()
            for c in cards
            if c.get("needs_add") and str(c.get("name") or "").strip()
        ]
        if missing:
            # Required DOBs still missing — never treat Yes as permission to advance.
            dob_step, dob_speech = await _prepare_dob_collect_speech(session, page)
            return _app_ok(
                call_id,
                session,
                text,
                dob_step,
                "dob missing required",
                spoken_override=dob_speech,
            )
        if _is_yes(text) or re.search(
            r"\b(continue|looks? (good|correct|fine)|next)\b", lowered
        ):
            enabled = await _next_button_enabled(page)
            if enabled is False:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: Next is still disabled, so a required date of birth "
                        "may still be missing. "
                        + app_question_for_step("app_dob_review", session)
                    ),
                    "error": "dob_next_disabled",
                    "step": step,
                    "phase": "application",
                }
            clicked = await _click_next_or_continue(page)
            if not clicked and enabled is False:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I can't continue yet because Next is disabled. "
                        + app_question_for_step("app_dob_review", session)
                    ),
                    "error": "dob_next_disabled",
                    "step": step,
                    "phase": "application",
                }
            next_step = await _advance_after_answer(page, session)
        elif _is_no(text) or re.search(
            r"\b(change|wrong|correct|fix|edit)\b", lowered
        ):
            available = [
                str(c.get("name") or "").strip()
                for c in cards
                if str(c.get("name") or "").strip()
            ]
            hits, clarify = _match_named_members(text, available)
            if clarify:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + clarify,
                    "error": "dob_member_ambiguous",
                    "step": step,
                    "phase": "application",
                }
            if hits:
                next_step, spoken_override = _begin_dob_collect_for(session, hits)
            else:
                spoken_override = (
                    "Okay — whose date of birth should I change? Say the name."
                )
                return _app_ok(
                    call_id,
                    session,
                    text,
                    step,
                    "dob change name needed",
                    spoken_override=spoken_override,
                )
        else:
            # Bare name like "Agent H" / "Agent Edge" while confirming DOBs.
            available = [
                str(c.get("name") or "").strip()
                for c in cards
                if str(c.get("name") or "").strip()
            ]
            hits, clarify = _match_named_members(text, available)
            if clarify:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + clarify,
                    "error": "dob_member_ambiguous",
                    "step": step,
                    "phase": "application",
                }
            name_like = bool(hits) or bool(_extract_spoken_member_labels(text))
            looks_like_name = bool(
                re.fullmatch(
                    r"[A-Za-z][A-Za-z'.\-]*(?:\s+[A-Za-z][A-Za-z'.\-]*){0,3}\.?",
                    (text or "").strip(),
                )
            )
            if name_like or looks_like_name:
                targets = hits or _extract_spoken_member_labels(text)
                if targets:
                    next_step, spoken_override = _begin_dob_collect_for(
                        session, targets
                    )
                else:
                    dob_step, dob_speech = await _prepare_dob_collect_speech(
                        session, page
                    )
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": "ASK: " + dob_speech,
                        "error": "dob_review_clarify",
                        "step": "app_dob_review",
                        "phase": "application",
                    }
            else:
                # Fresh entry / unclear — summarize or start collecting.
                dob_step, dob_speech = await _prepare_dob_collect_speech(
                    session, page
                )
                if dob_step == "app_dob_collect":
                    return _app_ok(
                        call_id,
                        session,
                        text,
                        dob_step,
                        "dob collect required",
                        spoken_override=dob_speech,
                    )
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + dob_speech,
                    "error": "dob_review_clarify",
                    "step": "app_dob_review",
                    "phase": "application",
                }

    elif step == "app_dob_collect":
        current = (session.filled.get("app_dob_current") or "").strip()
        queue = _dob_queue(session)
        # Allow "change Mickey" / name-only to retarget without inventing a date.
        parsed_early, _clarify_early = _parse_dob_answer(text)
        if re.search(r"\b(change|correct|fix|edit)\b", lowered) and not parsed_early:
            cards = await _scrape_dob_cards(page)
            available = [
                str(c.get("name") or "").strip()
                for c in cards
                if str(c.get("name") or "").strip()
            ]
            hits, clarify = _match_named_members(text, available)
            if clarify:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + clarify,
                    "error": "dob_member_ambiguous",
                    "step": step,
                    "phase": "application",
                }
            targets = hits or _extract_spoken_member_labels(text)
            if targets:
                next_step, spoken_override = _begin_dob_collect_for(session, targets)
                return _app_ok(
                    call_id,
                    session,
                    text,
                    next_step,
                    "dob retarget",
                    spoken_override=spoken_override,
                )
        dob, dob_clarify = _parse_dob_answer(text)
        if dob_clarify:
            # Keep the active member while clarifying ambiguous numeric dates.
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "application_answer",
                "spoken": "ASK: " + dob_clarify,
                "error": "dob_ambiguous",
                "step": step,
                "phase": "application",
            }
        if not dob:
            # Caller may have named another member instead of giving a date.
            alt_names = _extract_spoken_member_labels(text)
            looks_like_name = bool(
                re.fullmatch(
                    r"[A-Za-z][A-Za-z'.\-]*(?:\s+[A-Za-z][A-Za-z'.\-]*){0,3}\.?",
                    (text or "").strip(),
                )
            )
            if alt_names and looks_like_name:
                next_step, spoken_override = _begin_dob_collect_for(
                    session, alt_names
                )
            else:
                target = current or "that person"
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I need a clear month, day, and year. "
                        + _dob_question_for(target)
                    ),
                    "error": "dob_invalid",
                    "step": step,
                    "phase": "application",
                }
        else:
            if not current:
                cards = await _scrape_dob_cards(page)
                available = [
                    str(c.get("name") or "").strip()
                    for c in cards
                    if c.get("needs_add") and str(c.get("name") or "").strip()
                ]
                if len(available) == 1:
                    current = available[0]
                    session.filled["app_dob_current"] = current
                else:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: Whose date of birth is that? Please say the name."
                        ),
                        "error": "dob_member_needed",
                        "step": step,
                        "phase": "application",
                    }
            cards_before = await _scrape_dob_cards(page)
            same_name = [
                c
                for c in cards_before
                if (c.get("name") or "").strip().lower() == current.lower()
                or current.lower().split()[:1]
                == (c.get("name") or "").strip().lower().split()[:1]
            ]
            # Prefer exact label; fall back to first-name matches carefully.
            exact = [
                c
                for c in cards_before
                if (c.get("name") or "").strip().lower() == current.lower()
            ]
            if exact:
                same_name = exact
            if len(same_name) > 1:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I see more than one person named "
                        f"{current}. Please say the full name as shown on the card."
                    ),
                    "error": "dob_duplicate_name",
                    "step": step,
                    "phase": "application",
                }
            if same_name:
                current = str(same_name[0].get("name") or current)
                session.filled["app_dob_current"] = current
            needs_add = any(
                c.get("needs_add") for c in same_name
            ) or session.filled.get("app_dob_correcting")
            if needs_add or not same_name:
                opened = await _click_member_add_dob(page, current)
                if not opened and needs_add:
                    # Correction path may already have an open editor.
                    pass
            saved = await _fill_and_save_dob(page, dob)
            if not saved:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I couldn't enter that date on the form. "
                        + _dob_question_for(current)
                    ),
                    "error": "dob_fill_failed",
                    "step": step,
                    "phase": "application",
                }
            await page.wait_for_timeout(600)
            verified = await _verify_member_dob(page, current, dob)
            if not verified:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I couldn't verify that date of birth was saved. "
                        + _dob_question_for(current)
                    ),
                    "error": "dob_not_verified",
                    "step": step,
                    "phase": "application",
                }
            remaining = [name for name in queue if name.lower() != current.lower()]
            # Reinspect — Pointcare may reveal more missing cards after save.
            cards_after = await _scrape_dob_cards(page)
            still_missing = [
                str(c.get("name") or "").strip()
                for c in cards_after
                if c.get("needs_add") and str(c.get("name") or "").strip()
            ]
            for name in still_missing:
                if name.lower() != current.lower() and name not in remaining:
                    remaining.append(name)
            session.filled.pop("app_dob_correcting", None)
            if remaining:
                _set_dob_queue(session, remaining)
                nxt = remaining[0]
                session.filled["app_dob_current"] = nxt
                next_step = "app_dob_collect"
                spoken_override = (
                    f"Okay, I saved {dob} for {current}. "
                    + _dob_question_for(nxt)
                )
                _set_pending_question(session, next_step, spoken_override)
            else:
                _set_dob_queue(session, None)
                session.filled.pop("app_dob_current", None)
                next_step, spoken_override = await _prepare_dob_collect_speech(
                    session, page
                )

    elif step == "app_address_review":
        cards = await _scrape_address_cards(page)
        missing = [
            str(c.get("name") or "").strip()
            for c in cards
            if c.get("needs_add") and str(c.get("name") or "").strip()
        ]
        if re.search(r"\bno home address\b", lowered):
            ok = await _click_address_option(page, "No Home Address")
            if not ok:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I couldn't select No Home Address on the form. "
                        + app_question_for_step(step, session)
                    ),
                    "error": "address_no_home_failed",
                    "step": step,
                    "phase": "application",
                }
            spoken_override = (
                "Okay, No Home Address. "
                + app_question_for_step(step, session)
            )
            next_step = "app_address_review"
            _set_pending_question(session, next_step, spoken_override)
        elif re.search(r"\bno mailing address\b", lowered):
            ok = await _click_address_option(page, "No Mailing Address")
            if not ok:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I couldn't select No Mailing Address on the form. "
                        + app_question_for_step(step, session)
                    ),
                    "error": "address_no_mailing_failed",
                    "step": step,
                    "phase": "application",
                }
            spoken_override = (
                "Okay, No Mailing Address. "
                + app_question_for_step(step, session)
            )
            next_step = "app_address_review"
            _set_pending_question(session, next_step, spoken_override)
        elif _is_yes(text) or re.search(
            r"\b(continue|looks? (good|correct|fine)|next)\b", lowered
        ):
            if missing:
                names = ", ".join(missing[:4])
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: Pointcare still needs an address for "
                        f"{names}. Say the street address, or say "
                        "No Home Address or No Mailing Address if that applies."
                    ),
                    "error": "address_incomplete",
                    "step": step,
                    "phase": "application",
                }
            enabled = await _next_button_enabled(page)
            if enabled is False:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: Next is still disabled — an address may still be "
                        "missing. "
                        + app_question_for_step(step, session)
                    ),
                    "error": "address_next_disabled",
                    "step": step,
                    "phase": "application",
                }
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
        elif _is_no(text) or re.search(r"\b(change|wrong|correct|fix|add)\b", lowered):
            blurb = await _scrape_review_summary(page, step)
            target = ""
            if missing:
                target = missing[0]
            spoken_override = (
                blurb
                + (
                    f"What is the home or mailing address for {target}? "
                    if target
                    else "Whose address should I change, and what is the new street address? "
                )
                + "You can also say No Home Address or No Mailing Address."
            )
            return _app_ok(
                call_id,
                session,
                text,
                step,
                "address change requested",
                spoken_override=spoken_override,
            )
        else:
            # Treat a street-like utterance as an address to enter.
            looks_like_address = bool(
                re.search(
                    r"\d+\s+\w+|street|avenue|ave\b|road|rd\b|blvd|drive|dr\b|lane|ln\b|apt|suite|p\.?\s*o\.?\s*box",
                    lowered,
                )
            )
            if looks_like_address or missing:
                if missing:
                    # Open Add on the first missing member when possible.
                    try:
                        await _click_cta(page, "Add", attempts=1)
                    except Exception:
                        pass
                filled = await _fill_next_visible_input(page, text)
                if not filled:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: I couldn't enter that address. "
                            "Please say the street address again, or say "
                            "No Home Address or No Mailing Address."
                        ),
                        "error": "address_fill_failed",
                        "step": step,
                        "phase": "application",
                    }
                # Prefer a suggestion click when Pointcare offers autocomplete.
                try:
                    await page.wait_for_timeout(400)
                    await _click_exact_choice(page, text.split(",")[0].strip()[:40])
                except Exception:
                    pass
                saved = False
                for save_label in ("Save", "Save and Return", "Save & Return"):
                    try:
                        await _click_cta(page, save_label, attempts=1)
                        saved = True
                        break
                    except Exception:
                        continue
                if not saved:
                    return {
                        "ok": False,
                        "call_id": call_id,
                        "field_id": "application_answer",
                        "spoken": (
                            "ASK: I entered the address but couldn't save it. "
                            "Please say the address again."
                        ),
                        "error": "address_save_failed",
                        "step": step,
                        "phase": "application",
                    }
                spoken_override = (
                    "Okay, I saved that address. "
                    + app_question_for_step(step, session)
                )
                next_step = "app_address_review"
                _set_pending_question(session, next_step, spoken_override)
            else:
                blurb = await _scrape_review_summary(page, step)
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + blurb + app_question_for_step(step, session),
                    "error": "address_clarify",
                    "step": step,
                    "phase": "application",
                }

    elif step in (
        "app_marital_review",
        "app_childcare_review",
    ):
        if _is_yes(text) or re.search(
            r"\b(continue|looks? (good|correct|fine)|next)\b", lowered
        ):
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
        elif _is_no(text) or re.search(r"\b(change|wrong|correct|fix)\b", lowered):
            blurb = await _scrape_review_summary(page, step)
            return _app_ok(
                call_id,
                session,
                text,
                step,
                "change requested",
                spoken_override=(
                    blurb
                    + "Okay — tell me whose information to change and the new value, "
                    "or say Yes when it looks correct so I can continue."
                ),
            )
        else:
            blurb = await _scrape_review_summary(page, step)
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "application_answer",
                "spoken": "ASK: " + blurb + app_question_for_step(step, session),
                "error": f"{step}_clarify",
                "step": step,
                "phase": "application",
            }

    elif step == "app_circumstances":
        labels = _map_circumstance_labels(lowered)
        if not labels:
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "application_answer",
                "spoken": "ASK: " + app_question_for_step(step),
                "error": "circumstances_clarify",
                "step": step,
                "phase": "application",
            }
        for label in labels:
            try:
                await _click_exact_choice(page, label)
            except Exception:
                try:
                    await _click_cta(page, label, attempts=1)
                except Exception:
                    pass
        if labels == ["None of the above"]:
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
        else:
            # Queue every selected circumstance that needs a person assignment.
            session.filled["app_circumstance_queue"] = json.dumps(labels)
            session.filled["app_circumstance_pending"] = labels[0]
            next_step = "app_circumstances_person"
            spoken_override = (
                f"Okay, {labels[0]}. "
                + f"Which household member does {labels[0]} apply to?"
            )
            _set_pending_question(session, next_step, spoken_override)

    elif step == "app_circumstances_person":
        pending = (session.filled.get("app_circumstance_pending") or "").strip()
        queue_raw = session.filled.get("app_circumstance_queue") or "[]"
        try:
            queue = [
                str(x).strip()
                for x in json.loads(queue_raw)
                if str(x).strip()
            ]
        except Exception:
            queue = [pending] if pending else []
        if re.search(r"\b(continue|next|done|skip)\b", lowered):
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
            session.filled.pop("app_circumstance_pending", None)
            session.filled.pop("app_circumstance_queue", None)
        else:
            # Reject circumstance/condition phrases — need a household member.
            if _map_circumstance_labels(lowered) and not re.search(
                r"\b(applies?|for)\b", lowered
            ):
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: Please say the household member's name for "
                        f"{pending or 'that circumstance'}, not the condition again."
                    ),
                    "error": "circumstances_person_not_member",
                    "step": step,
                    "phase": "application",
                }
            member = ""
            for token in re.findall(
                r"[A-Za-z][A-Za-z'.\-]*(?:\s+[A-Za-z]\.?)?", text
            ):
                if token.lower() in {
                    "and",
                    "the",
                    "applies",
                    "to",
                    "for",
                    "this",
                    "that",
                }:
                    continue
                member = token.strip()
                if member:
                    break
            if not member:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: Which household member does "
                        f"{pending or 'that circumstance'} apply to?"
                    ),
                    "error": "circumstances_person_clarify",
                    "step": step,
                    "phase": "application",
                }
            # Scope the member click to the active circumstance card only —
            # never select Blind when Cervical cancer is the pending section.
            ok = await _set_circumstance_member(page, member, pending)
            if not ok:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I couldn't select that person under "
                        f"{pending or 'the current circumstance'}. "
                        "Please say the name as shown on the card."
                    ),
                    "error": "circumstances_person_not_set",
                    "step": step,
                    "phase": "application",
                }
            remaining = [c for c in queue if c.lower() != pending.lower()]
            if remaining:
                session.filled["app_circumstance_queue"] = json.dumps(remaining)
                session.filled["app_circumstance_pending"] = remaining[0]
                next_step = "app_circumstances_person"
                spoken_override = (
                    f"Okay. Which household member does {remaining[0]} apply to?"
                )
                _set_pending_question(session, next_step, spoken_override)
            else:
                session.filled.pop("app_circumstance_pending", None)
                session.filled.pop("app_circumstance_queue", None)
                await _click_next_or_continue(page)
                next_step = await _advance_after_answer(page, session)

    elif step == "app_tax":
        screen = await detect_application_screen(page)
        if screen and screen != "app_tax" and screen.startswith("app_"):
            # Tax screen absent — reconcile and continue.
            next_step = screen
            spoken_override = app_question_for_step(next_step)
        elif re.search(r"\b(edit|change)\b", lowered) or (
            _is_no(text) and "correct" not in lowered
        ):
            try:
                await _click_cta(page, "Edit", attempts=2)
            except Exception:
                pass
            spoken_override = (
                "Okay, the tax edit screen should be open. Say the filing status "
                "or dependent change you need, then say continue when you're ready "
                "for me to save and return."
            )
            next_step = "app_tax"
        elif _is_yes(text) or re.search(
            r"\b(continue|looks? (good|correct|fine)|next|save)\b", lowered
        ):
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
        else:
            blurb = await _scrape_review_summary(page, step)
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "application_answer",
                "spoken": "ASK: " + blurb + app_question_for_step(step),
                "error": "tax_clarify",
                "step": step,
                "phase": "application",
            }

    elif step == "app_income":
        if re.search(r"\b(continue|next|done|everyone)\b", lowered):
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
        else:
            picked = _map_income_type(lowered)
            if not picked:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + app_question_for_step(step),
                    "error": "income_clarify",
                    "step": step,
                    "phase": "application",
                }
            try:
                await _click_cta(page, "Add Income", attempts=1)
            except Exception:
                pass
            try:
                await _click_exact_choice(page, picked)
            except Exception:
                await _click_cta(page, picked, attempts=2)
            if picked == "Employed":
                next_step, spoken_override = await _start_income_seq(
                    session, "employed"
                )
            elif picked == "Self-employed":
                next_step, spoken_override = await _start_income_seq(session, "self")
            elif picked == "None":
                try:
                    await _click_cta(page, "Save", attempts=2)
                except Exception:
                    pass
                next_step = "app_income_more"
                spoken_override = app_question_for_step(next_step)
            else:
                _set_income_seq(
                    session,
                    {
                        "kind": "other",
                        "index": 0,
                        "values": {},
                        "keys": ["pay"],
                    },
                )
                next_step = "app_income_self"
                spoken_override = (
                    f"Okay, {picked}. What amount should I enter for that income?"
                )

    elif step in (
        "app_income_employed",
        "app_income_employed_hours",
        "app_income_employed_frequency",
        "app_income_employed_employer",
        "app_income_self",
        "app_income_self_work",
    ):
        return await _continue_income_seq(call_id, session, page, text)

    elif step == "app_income_more":
        if _is_yes(text):
            next_step = "app_income"
            spoken_override = app_question_for_step(next_step)
        else:
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)

    elif step == "app_expenses":
        if re.search(r"\b(skip|next|no|continue|none)\b", lowered) or _is_no(text):
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
        else:
            expense_cues = (
                ("student loan", "Student loans"),
                ("childcare", "Childcare"),
                ("child care", "Childcare"),
                ("medical", "Medical"),
            )
            picked = None
            for cue, label in expense_cues:
                if cue in lowered:
                    picked = label
                    break
            if picked:
                try:
                    await _click_exact_choice(page, picked)
                except Exception:
                    try:
                        await _click_cta(page, picked, attempts=1)
                    except Exception:
                        pass
                session.filled["app_expense_pending"] = picked
                next_step = "app_expenses_amount"
                spoken_override = app_question_for_step(next_step)
            else:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + app_question_for_step(step),
                    "error": "expenses_clarify",
                    "step": step,
                    "phase": "application",
                }

    elif step == "app_expenses_amount":
        filled = await _fill_next_visible_input(page, text)
        if not filled:
            return {
                "ok": False,
                "call_id": call_id,
                "field_id": "application_answer",
                "spoken": "ASK: " + app_question_for_step(step),
                "error": "expenses_amount_clarify",
                "step": step,
                "phase": "application",
            }
        try:
            await _click_cta(page, "Save", attempts=1)
        except Exception:
            pass
        session.filled.pop("app_expense_pending", None)
        await _click_next_or_continue(page)
        next_step = await _advance_after_answer(page, session)

    elif step == "app_documents":
        if re.search(r"\b(skip|next|later|continue|no)\b", lowered) or _is_no(text):
            await _click_next_or_continue(page)
            next_step = await _advance_after_answer(page, session)
        else:
            return _app_ok(
                call_id,
                session,
                text,
                step,
                "upload deferred",
                spoken_override=(
                    "I can't take document files over the phone. "
                    "Say skip to continue without uploading now. You can add "
                    "documents later by logging in with the same account — "
                    "I will not pretend a file was uploaded."
                ),
            )

    elif step == "app_submit_confirm":
        if _is_yes(text) or re.search(r"\b(submit|confirm)\b", lowered):
            clicked_submit = False
            for label in ("Submit Applications", "Submit Application", "Submit"):
                try:
                    await _click_cta(page, label, attempts=2)
                    clicked_submit = True
                    break
                except Exception:
                    continue
            if not clicked_submit:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": "ASK: " + app_question_for_step(step),
                    "error": "submit_button_missing",
                    "step": step,
                    "phase": "application",
                }
            await page.wait_for_timeout(1200)
            body = ""
            try:
                body = (await page.locator("body").inner_text()).lower()
            except Exception:
                pass
            confirmed = any(
                needle in body
                for needle in (
                    "submitted",
                    "thank you",
                    "all set",
                    "application has been",
                    "we received",
                )
            )
            if not confirmed:
                return {
                    "ok": False,
                    "call_id": call_id,
                    "field_id": "application_answer",
                    "spoken": (
                        "ASK: I couldn't confirm that Pointcare accepted the submission. "
                        + app_question_for_step(step)
                    ),
                    "error": "submit_not_verified",
                    "step": step,
                    "phase": "application",
                }
            session.filled["application_submitted"] = "1"
            session.app_step = ""
            return {
                "ok": True,
                "call_id": call_id,
                "field_id": "application_answer",
                "value": text,
                "action": "submitted",
                "message": "Application submitted and verified",
                "spoken": "ASK: " + CLOSING_SUBMITTED,
                "step": "",
                "phase": "closing",
                "end_call": True,
            }
        if _is_no(text):
            session.app_step = ""
            return {
                "ok": True,
                "call_id": call_id,
                "field_id": "application_answer",
                "value": text,
                "action": "not_submitted",
                "message": "Caller declined final submit",
                "spoken": "ASK: " + CLOSING_NOT_SUBMITTED,
                "step": "",
                "phase": "closing",
                "end_call": True,
            }
        return {
            "ok": False,
            "call_id": call_id,
            "field_id": "application_answer",
            "spoken": "ASK: " + app_question_for_step(step),
            "error": "submit_clarify",
            "step": step,
            "phase": "application",
        }
    else:
        # Unknown step — try Next and redetect.
        await _click_next_or_continue(page)
        next_step = await _advance_after_answer(page, session)

    if next_step == "programs":
        # Mid Stage E should not silently restart citizenship.
        next_step = step if step.startswith("app_") else ""
        if not next_step:
            return _unidentified_screen_error(call_id, session, answer)
    # Entering DOB review: inspect missing dates before asking a bare Yes.
    if (
        next_step == "app_dob_review"
        and spoken_override is None
        and step != "app_dob_review"
    ):
        next_step, spoken_override = await _prepare_dob_collect_speech(
            session, page
        )
    if not next_step:
        return _unidentified_screen_error(call_id, session, answer)
    _mark(call_id, "verified", started)
    session.history.append(
        {
            "field_id": "application_answer",
            "step": step,
            "value": answer,
            "action": "verified",
            "ok": True,
        }
    )
    result = _app_ok(
        call_id,
        session,
        answer,
        next_step,
        f"Applied application step {step}",
        spoken_override=spoken_override,
        end_call=end_call,
    )
    _mark(call_id, "next_question", started)
    _mark(call_id, "tool_response", started)
    return result
