"""Pointcare field ids Lauren reports via submit_field → Playwright locators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    labels: tuple[str, ...]
    kind: str = "text"  # text | password | tel | choice | click | action
    placeholders: tuple[str, ...] = ()
    input_types: tuple[str, ...] = ()
    button_names: tuple[str, ...] = ()
    after_action: str = ""  # continue | next | skip | ""


# Welcome screens (Stage A) before Sign in / Create Profile.
WELCOME_CLICKS: tuple[tuple[str, ...], ...] = (
    ("Get Started",),  # page 1 — coverage intro
    ("Continue",),  # page 2 — Pointcare support
    ("Continue",),  # page 3 — frustration (optional; Continue without picking)
    ("Create My Profile",),  # page 4 — start profile / sign-in
)

# Ordered roughly like the Pointcare voice script.
POINTCARE_FIELDS: dict[str, FieldSpec] = {
    "get_started": FieldSpec(
        "get_started",
        labels=("get started",),
        kind="click",
        button_names=("get started",),
    ),
    "create_my_profile": FieldSpec(
        "create_my_profile",
        labels=("create my profile",),
        kind="click",
        button_names=("create my profile",),
    ),
    "continue_with_google": FieldSpec(
        "continue_with_google",
        labels=("continue with google", "gmail", "google"),
        kind="sso",
        button_names=("Continue with Google",),
    ),
    "continue_with_apple": FieldSpec(
        "continue_with_apple",
        labels=("continue with apple", "apple"),
        kind="sso",
        button_names=("Continue with Apple",),
    ),
    "email": FieldSpec(
        "email",
        labels=("email", "email address"),
        placeholders=("email", "you@"),
        input_types=("email", "text"),
        after_action="continue",
    ),
    "password": FieldSpec(
        "password",
        labels=("password", "create password", "new password"),
        placeholders=("password",),
        input_types=("password",),
        after_action="continue",
    ),
    "phone": FieldSpec(
        "phone",
        labels=("phone", "phone number", "mobile", "mobile number"),
        placeholders=("phone", "mobile"),
        input_types=("tel", "text"),
        after_action="continue",
    ),
    "otp": FieldSpec(
        "otp",
        labels=("verification code", "code", "otp", "one-time"),
        placeholders=("code", "otp"),
        input_types=("text", "tel", "number"),
        after_action="continue",
    ),
    "first_name": FieldSpec(
        "first_name",
        labels=("first name", "given name"),
        placeholders=("first name",),
    ),
    "last_name": FieldSpec(
        "last_name",
        labels=("last name", "family name", "surname"),
        placeholders=("last name",),
    ),
    "dob": FieldSpec(
        "dob",
        labels=("date of birth", "birth date", "birthday", "dob"),
        placeholders=("mm", "dd", "yyyy", "date of birth"),
    ),
    "address": FieldSpec(
        "address",
        labels=("mailing address", "home address", "street address", "address"),
        placeholders=("address", "street", "start typing"),
    ),
    "apartment": FieldSpec(
        "apartment",
        labels=("apartment", "apt", "unit", "suite"),
        placeholders=("apt", "unit", "apartment"),
    ),
    "no_mailing_address": FieldSpec(
        "no_mailing_address",
        labels=("no mailing address", "no home address"),
        kind="click",
        button_names=("no mailing address", "no home address"),
    ),
    "ssn": FieldSpec(
        "ssn",
        labels=("social security", "ssn", "social security number"),
        placeholders=("ssn", "xxx-xx"),
        input_types=("text", "password", "tel"),
    ),
    "medicaid_id": FieldSpec(
        "medicaid_id",
        labels=("medicaid", "medicaid subscriber", "medicaid id"),
        placeholders=("medicaid",),
    ),
    "marital_status": FieldSpec(
        "marital_status",
        labels=("marital status",),
        kind="choice",
    ),
    "has_children": FieldSpec(
        "has_children",
        labels=("children under", "live with and care"),
        kind="choice",
    ),
    "household_income": FieldSpec(
        "household_income",
        labels=("household income", "income"),
        kind="choice",
    ),
    "special_conditions": FieldSpec(
        "special_conditions",
        labels=("special", "pregnant", "disabled", "none of the above"),
        kind="choice",
    ),
    "consent_sign": FieldSpec(
        "consent_sign",
        labels=("add signature", "signature", "sign"),
        kind="click",
        button_names=("add signature", "sign", "i agree"),
    ),
    "continue": FieldSpec(
        "continue",
        labels=("continue",),
        kind="action",
        button_names=("continue", "next", "submit"),
        after_action="",
    ),
    "next": FieldSpec(
        "next",
        labels=("next",),
        kind="action",
        button_names=("next", "continue"),
    ),
    "skip": FieldSpec(
        "skip",
        labels=("skip",),
        kind="action",
        button_names=("skip", "skip for now", "not now"),
    ),
    "check_eligibility": FieldSpec(
        "check_eligibility",
        labels=("check household eligibility", "check eligibility"),
        kind="click",
        button_names=("check household eligibility", "check eligibility"),
    ),
    "start_application": FieldSpec(
        "start_application",
        labels=("continue to application", "start application", "apply"),
        kind="click",
        button_names=("continue", "apply", "start application"),
    ),
    "submit_application": FieldSpec(
        "submit_application",
        labels=("submit", "submit application"),
        kind="action",
        button_names=("submit", "submit application", "finish"),
    ),
}


ACTION_BUTTONS: dict[str, tuple[str, ...]] = {
    # Exact labels only. "continue" must NOT match "Continue with Google/Apple".
    "continue": ("Continue",),
    "next": ("Next",),
    "skip": ("Skip", "Skip for now", "Not now"),
    "submit": ("Submit", "Submit Applications", "Submit Application"),
    "get_started": ("Get Started",),
    "create_my_profile": ("Create My Profile",),
}

# Never auto-click these on the Sign in page.
SSO_BUTTON_NEEDLES = (
    "google",
    "apple",
    "use this",
    "continue as",
    "continue with",
    "saved",
    "choose an account",
)


def get_field(field_id: str) -> FieldSpec | None:
    key = (field_id or "").strip().lower().replace(" ", "_").replace("-", "_")
    return POINTCARE_FIELDS.get(key)


def tool_webhook_url(webhook_url: str = "") -> str:
    raw = (webhook_url or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.endswith("/tool"):
        return raw
    return f"{raw}/tool"


def submit_field_tool_spec(tool_url: str) -> dict[str, Any]:
    return {
        "type": "custom",
        "name": "submit_field",
        "url": tool_url,
        "description": (
            "Collect Mode tool. The server owns Stage D Household Eligibility order "
            "and branching (Stage E Application is reserved for later). Submit exactly "
            "one caller answer, stay silent while it runs, then speak only the returned "
            "text. Use sign_in_google for Google, email for email sign-in, and "
            "eligibility_answer only after sign-in for household-eligibility answers "
            "(Yes, No, marital options, names, relationships, ages, income, special "
            "conditions). Never call eligibility_answer for language answers like "
            "English. If the caller answers mid-question, stop and submit immediately. "
            "A question you spoke is not an answer. Wait for a new caller utterance "
            "before calling again. Never invent options or speak JSON."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field_id": {
                    "type": "string",
                    # application_answer reserved when Stage E Collect Mode is enabled.
                    "enum": ["sign_in_google", "email", "eligibility_answer"],
                    "description": (
                        "sign_in_google | email | eligibility_answer "
                        "(future: application_answer for Stage E)."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "Confirmed value from the caller, or empty for click-only actions.",
                },
                "action": {
                    "type": "string",
                    "description": "Usually empty. The server determines browser navigation.",
                },
            },
            "required": ["field_id"],
        },
        "speak_during_execution": False,
        "speak_after_execution": True,
        "execution_message_description": "ok",
        "timeout_ms": 120000,
    }
