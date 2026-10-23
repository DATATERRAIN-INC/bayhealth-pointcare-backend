"""Stage D field metadata: required/optional, answer type, skip behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

from .live_form import ELIG_QUESTIONS


@dataclass(frozen=True)
class FieldMeta:
    required: bool | None = True  # True / False / None=unknown
    can_skip: bool = False
    skip_behavior: str = ""  # "" | "continue" | "next" | "skip_button"
    answer_type: str = "text"
    options_hint: tuple[str, ...] = field(default_factory=tuple)


def _meta(
    answer_type: str,
    *,
    required: bool | None = True,
    can_skip: bool = False,
    skip_behavior: str = "",
    options_hint: tuple[str, ...] = (),
) -> FieldMeta:
    return FieldMeta(
        required=required,
        can_skip=can_skip,
        skip_behavior=skip_behavior,
        answer_type=answer_type,
        options_hint=options_hint,
    )


# Stage D eligibility chips are required unless explicitly marked otherwise.
ELIG_FIELD_META: dict[str, FieldMeta] = {
    "marital": _meta(
        "choice",
        options_hint=(
            "Never Married",
            "Married",
            "Separated",
            "Divorced",
            "Widowed",
            "Registered Domestic Partner",
            "Common Law Marriage",
        ),
    ),
    "spouse_first": _meta("text"),
    "spouse_last": _meta("text"),
    "spouse_role": _meta("choice", options_hint=("Spouse", "Partner")),
    "spouse_age": _meta("age"),
    "spouse_confirm": _meta("yes_no"),
    "children": _meta("yes_no"),
    "child_first": _meta("text"),
    "child_last": _meta("text"),
    "child_rel": _meta(
        "choice",
        options_hint=("Child", "Stepchild", "Grandchild", "Foster Child", "Other"),
    ),
    "child_age": _meta("age"),
    "child_confirm": _meta("yes_no"),
    "child_more": _meta("yes_no"),
    "student": _meta("yes_no"),
    "disabled": _meta("yes_no"),
    "relatives": _meta("yes_no"),
    "support": _meta("yes_no"),
    "person_first": _meta("text"),
    "person_last": _meta("text"),
    "person_rel": _meta(
        "choice",
        # Default Pointcare person/relative form chips; live peek overrides when present.
        options_hint=(
            "Child",
            "Stepchild",
            "Grandchild",
            "Foster Child",
            "Other",
        ),
    ),
    "person_age": _meta("age"),
    "person_confirm": _meta("yes_no"),
    "income_when": _meta(
        "income_period",
        options_hint=("Annual", "Monthly", "Bi-weekly"),
    ),
    "income_range": _meta("income_range"),
    "special": _meta("choice"),
    "special_person": _meta("text"),
    "special_more": _meta("yes_no"),
    "babies_count": _meta("text"),
    "due_date": _meta("text", required=None, can_skip=False),
    "programs": _meta("yes_no"),
    "proceed_application": _meta("yes_no"),
    "confirm_start_application": _meta("yes_no"),
}


def get_field_meta(step: str) -> FieldMeta:
    return ELIG_FIELD_META.get(step) or _meta("text", required=None)


def format_question(
    step: str,
    *,
    options: list[str] | None = None,
    period: str | None = None,
    include_choices: bool = False,
) -> str:
    """Full spoken question for the active step.

    Never appends internal validation chrome such as
    \"This field is required to continue.\" Choice lists are opt-in via
    ``include_choices`` — prefer ``build_options_help_speech`` when the
    caller explicitly asks for options.
    """
    if step == "income_range":
        from .live_form import _income_range_question_for_period

        base = _income_range_question_for_period(period or "Annual").strip()
    elif step in ("child_rel", "person_rel", "spouse_role"):
        from .live_form import relationship_question_speech

        return relationship_question_speech(step, options)
    else:
        base = (ELIG_QUESTIONS.get(step) or "").strip()
    if not base:
        return "Could you say that again?"
    meta = get_field_meta(step)
    # Never list raw $ income chips for voice — callers cannot see the browser.
    extras: list[str] = []
    if (
        include_choices
        and options
        and meta.answer_type in ("choice", "age", "income_period")
        and step != "income_range"
    ):
        shown = [item for item in options if item and len(item) < 60][:8]
        if shown:
            joined = ", ".join(shown)
            if joined.lower() not in base.lower():
                extras.append(f" The choices are: {joined}.")
    return f"{base}{''.join(extras)}".strip()
