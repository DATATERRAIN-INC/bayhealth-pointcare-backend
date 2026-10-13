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
    "children": _meta("yes_no"),
    "child_first": _meta("text"),
    "child_last": _meta("text"),
    "child_rel": _meta(
        "choice",
        options_hint=("Child", "Stepchild", "Grandchild", "Foster Child", "Other"),
    ),
    "child_age": _meta("age"),
    "child_more": _meta("yes_no"),
    "student": _meta("yes_no"),
    "disabled": _meta("yes_no"),
    "relatives": _meta("yes_no"),
    "support": _meta("yes_no"),
    "person_first": _meta("text"),
    "person_last": _meta("text"),
    "person_rel": _meta("text"),
    "person_age": _meta("age"),
    "income_when": _meta(
        "income_period",
        options_hint=("Annual", "Monthly", "Bi-weekly"),
    ),
    "income_range": _meta("income_range"),
    "special": _meta("choice"),
    "special_person": _meta("text"),
    "babies_count": _meta("text"),
    "due_date": _meta("text", required=None, can_skip=False),
    "programs": _meta("yes_no"),
}


def get_field_meta(step: str) -> FieldMeta:
    return ELIG_FIELD_META.get(step) or _meta("text", required=None)


def format_question(step: str, *, options: list[str] | None = None) -> str:
    """Full spoken question with required/optional hint when known."""
    base = (ELIG_QUESTIONS.get(step) or "").strip()
    if not base:
        return "Could you say that again?"
    meta = get_field_meta(step)
    hint = ""
    if meta.required is True:
        hint = " This field is required to continue."
    elif meta.required is False or meta.can_skip:
        hint = (
            " This question is optional. You can answer it, or say skip if you prefer."
        )
    # Avoid duplicating "Please say Yes or No" style if already in base.
    extras: list[str] = []
    if options:
        shown = [item for item in options if item and len(item) < 60][:8]
        if shown and meta.answer_type in ("choice", "age", "income_range", "income_period"):
            joined = ", ".join(shown)
            if joined.lower() not in base.lower():
                extras.append(f" The choices are: {joined}.")
    return f"{base}{hint}{''.join(extras)}".strip()
