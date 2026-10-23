"""Normalize and validate caller answers.

Deterministic Python handles clear Yes/No, exact option matches, and simple
name/income/age mappings. Claude Haiku (low effort) assists only for
nontrivial or ambiguous utterances, and every Claude result is validated
against the active question and available options before Playwright acts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .live_form import (
    ELIG_QUESTIONS,
    YES_NO_ELIG_STEPS,
    _age_option_labels,
    _choice_phrase,
    _detect_income_period_switch,
    _extract_age_numbers,
    _income_amount_scale_unclear,
    _income_period_choice,
    _income_range_natural_clarify,
    _income_scale_clarify_speech,
    _is_no,
    _is_yes,
    _map_age_label,
    _map_income_range_label,
    _map_relationship_label,
    _map_special_condition_labels,
    _resolve_income_period_label,
    _spoken_has_age_range_connector,
    _wants_income_period_reselect,
    classify_yes_no,
)

logger = logging.getLogger(__name__)

# Deterministic resolver outcomes that must skip Claude (fast path).
_FAST_PATH_DETAILS = frozenset(
    {
        "yes",
        "no",
        "yes_no_clear",
        "choice_exact",
        "marital",
        "marital_single",
        "marital_option",
        "spouse_role",
        "confirm_rel_correction",
        "confirm_age_correction",
        "relationship_mapped",
        "relationship_other_text",
        "special_mapped",
        "special_multi",
        "age_mapped",
        "age_free",  # clear spoken age; Playwright opens chips and verifies
        "income_period",
        "income_period_switch",
        "income_period_reselect",
        "income_range_mapped",
        "name",
        "special_person",
        "support_no",
    }
)

_CLEAR_YES_NO = re.compile(r"^(yes|yeah|yep|yup|no|nope|nah)([.\s!]|$)", re.I)
_MARITAL_HINTS = re.compile(
    r"never married|single|married|separated|divorced|widowed|partner|common law|"
    r"i have never been married|never been married",
    re.I,
)
# Legacy repeat marker — control_intent now owns navigation; keep for back-compat.
_REPEAT_RE = re.compile(
    r"\b("
    r"repeat(?:\s+that|\s+the\s+question)?|"
    r"say\s+(?:that|it|the\s+question)\s+again|"
    r"what(?:'s|s|\s+is)\s+the\s+question|"
    r"what\s+was\s+the\s+question|"
    r"(?:i\s+)?(?:did(?:n't|\s+not)|cannot|can't)\s+hear"
    r"(?:\s+(?:the\s+)?question)?"
    r")\b",
    re.I,
)


@dataclass
class ResolveResult:
    answer: str | None = None
    clarify: str | None = None
    detail: str = ""
    resolver: str = "python"  # python | claude | none


@dataclass
class ClaudeResolveResult:
    intent: str = "unknown"  # answer | clarify | unknown | control
    answer: str | None = None
    matched_option: str | None = None
    confidence: float = 0.0
    needs_clarify: bool = True
    clarify_prompt: str | None = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    reject_reason: str = ""


def normalize_utterance(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"^(it is|it's|its|my name is|i am|i'm)\s+",
        "",
        text,
        flags=re.I,
    ).strip()
    return text


def _collapse_letter_by_letter_spelling(text: str) -> str:
    """Join clear letter-by-letter spelling into one name token.

    ``H u s s e y`` / ``H-U-S-S-E-Y`` → ``Hussey``.
    Does not join legitimate multi-word names like ``Anne Marie``.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    # Drop "spelled …" / "it's spelled as …" wrappers.
    raw = re.sub(
        r"^(?:(?:it'?s|that'?s|the\s+name\s+is)\s+)?"
        r"(?:spelled?|spelling)\s*(?:as\s+|is\s+|out\s+)?",
        "",
        raw,
        flags=re.I,
    ).strip()
    # Tokenize on spaces and hyphens used between letters.
    parts = [p for p in re.split(r"[\s\-–—]+", raw) if p]
    if len(parts) < 3:
        return text.strip()
    # Every token must be a single alphabetic letter (optional trailing apostrophe).
    if not all(re.fullmatch(r"[A-Za-z]'?", part) for part in parts):
        return text.strip()
    joined = "".join(part[0] for part in parts)
    # Preserve a trailing apostrophe only if a token carried one mid-name (rare).
    if any(part.endswith("'") for part in parts[:-1]):
        # Rebuild with apostrophes after those letters (e.g. O' B R I E N → O'Brien).
        bits: list[str] = []
        for part in parts:
            bits.append(part[0])
            if part.endswith("'"):
                bits.append("'")
        joined = "".join(bits)
    if "'" in joined:
        # Title-case around apostrophe: o'brien → O'Brien
        segs = joined.split("'")
        return "'".join(seg.capitalize() for seg in segs if seg is not None)
    return joined.capitalize()


def sanitize_person_name(raw: str) -> str:
    """Strip ASR sentence punctuation; keep apostrophes/hyphens/accents.

    Also unwrap phrases like "Last name is Martin" / "First name is John",
    and collapse letter-by-letter spelling (``H u s s e y`` → ``Hussey``).
    """
    text = normalize_utterance(raw)
    if not text:
        return ""
    wrapped = re.match(
        r"^(?:(?:the\s+|my\s+|their\s+|his\s+|her\s+)?"
        r"(?:first|last|given|family)\s+name\s+is|"
        r"name\s+is)\s+(.+)$",
        text,
        re.I,
    )
    if wrapped:
        text = wrapped.group(1).strip()
    # Drop trailing sentence enders from ASR ("John." → "John").
    text = re.sub(r"[.!?…]+$", "", text).strip()
    # Collapse "John ." mid-token leftovers from ASR.
    text = re.sub(r"\s+([.!?])\s*$", "", text).strip()
    # Spoken initials: "Angel r dot" → "Angel R"
    text = re.sub(
        r"\b([A-Za-z])\s*(?:dot|period)\b\.?",
        lambda m: m.group(1).upper(),
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+\.", ".", text)
    # Drop trailing ASR filler ("Angel or" → "Angel").
    text = re.sub(r"\s+\b(or|and|uh|um|ah)\b\s*$", "", text, flags=re.I).strip()
    # Letter-by-letter spelling from ASR must not keep spaces between letters.
    text = _collapse_letter_by_letter_spelling(text)
    return text


def is_language_ack(text: str) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    if re.search(r"\b(english|spanish|chinese|vietnamese|russian|arabic|tagalog)\b", lowered):
        return True
    return bool(
        re.fullmatch(
            r"(yes|yeah|yep|yup|sure|ok|okay|fine|that's fine|thats fine|continue|"
            r"go ahead|alright)[.!]?",
            lowered,
        )
    )


def is_google_choice(text: str) -> bool:
    return bool(re.search(r"\b(google|gmail|g-mail|g mail)\b", text or "", re.I))


def is_email_choice(text: str) -> bool:
    lowered = (text or "").lower()
    if is_google_choice(lowered):
        return False
    # Go-back / wrong-email corrections mention "email" but are not choosing email SSO.
    if re.search(
        r"\b("
        r"go\s*-?\s*back|goback|"
        r"wrong\s+e-?mail|incorrect\s+e-?mail|entered\s+the\s+wrong|"
        r"different\s+e-?mail|change\s+(the\s+)?e-?mail|"
        r"not\s+(my\s+)?e-?mail"
        r")\b",
        lowered,
    ):
        return False
    return bool(re.search(r"\be-?mail\b", lowered))


def is_apple_choice(text: str) -> bool:
    lowered = (text or "").lower()
    if is_google_choice(lowered) or re.search(r"\bemail\b", lowered):
        return False
    return bool(re.search(r"\bapple\b", lowered))


_EMAIL_ADDR_RE = re.compile(
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    re.I,
)

# Lead-ins Retell often prefixes before the address itself.
_EMAIL_LEADIN_RE = re.compile(
    r"^(?:(?:so|okay|ok|um|uh|well)\s+)?"
    r"(?:"
    r"(?:my|the|our)?\s*e-?mail(?:\s+address)?\s+is|"
    r"(?:my|the)?\s*e-?mail(?:\s+address)?\s*(?:should be|will be|to use)?|"
    r"it(?:'s| is)|use|"
    r"sign\s*in\s+with|"
    r"log\s*in\s+with"
    r")\s+",
    re.I,
)

# Spoken digits Retell may emit instead of "003" (including glued "zerozerothree").
_EMAIL_DIGIT_WORDS: tuple[tuple[str, str], ...] = (
    ("zero", "0"),
    ("nought", "0"),
    ("oh", "0"),
    ("one", "1"),
    ("two", "2"),
    ("three", "3"),
    ("four", "4"),
    ("five", "5"),
    ("six", "6"),
    ("seven", "7"),
    ("eight", "8"),
    ("nine", "9"),
)
_EMAIL_DIGIT_ALT = "|".join(
    sorted((w for w, _ in _EMAIL_DIGIT_WORDS), key=len, reverse=True)
)
# Glued/spaced digit runs not embedded inside other letters.
_EMAIL_DIGIT_RUN_RE = re.compile(
    rf"(?<![a-z])(?:(?:{_EMAIL_DIGIT_ALT})\s*)+(?![a-z])",
    re.I,
)


def _spoken_digits_to_chars(text: str) -> str:
    """Convert spoken digit words to characters; keep non-digit text intact."""

    def _run_to_digits(match: re.Match[str]) -> str:
        chunk = match.group(0).lower()
        digits: list[str] = []
        pos = 0
        ordered = sorted(_EMAIL_DIGIT_WORDS, key=lambda item: -len(item[0]))
        while pos < len(chunk):
            while pos < len(chunk) and chunk[pos].isspace():
                pos += 1
            if pos >= len(chunk):
                break
            hit = False
            for word, digit in ordered:
                if chunk.startswith(word, pos):
                    digits.append(digit)
                    pos += len(word)
                    hit = True
                    break
            if not hit:
                return match.group(0)
        return "".join(digits) if digits else match.group(0)

    # e.g. "zero zero three" / "zerozerothree" → "003"
    return _EMAIL_DIGIT_RUN_RE.sub(_run_to_digits, text or "")


def normalize_spoken_email(text: str) -> str:
    """Turn spoken email phrasing into a compact address candidate.

    Handles common ASR variants (at / at the rate / dot / period / underscore /
    hyphen / spoken digits) and spacing. Does not invent missing characters
    (e.g. will not insert @ between bare 'name gmail.com').
    """
    spoken = (text or "").strip()
    if not spoken:
        return ""
    spoken = _EMAIL_LEADIN_RE.sub("", spoken).strip()
    # Digits before collapsing spaces so "zero zero three" / "zerozerothree" → 003.
    spoken = _spoken_digits_to_chars(spoken)
    # Multi-word tokens first so "at the rate" is not partially eaten by "at".
    spoken = re.sub(r"\s+at\s+the\s+rate(?:\s+of)?\s+", "@", spoken, flags=re.I)
    spoken = re.sub(r"\bat\s+the\s+rate(?:\s+of)?\b", "@", spoken, flags=re.I)
    spoken = re.sub(r"\s+underscore\s+", "_", spoken, flags=re.I)
    spoken = re.sub(r"\bunderscore\b", "_", spoken, flags=re.I)
    spoken = re.sub(r"\s+(?:hyphen|dash)\s+", "-", spoken, flags=re.I)
    spoken = re.sub(r"\b(?:hyphen|dash)\b", "-", spoken, flags=re.I)
    spoken = re.sub(r"\s+at\s+", "@", spoken, flags=re.I)
    spoken = re.sub(r"\s+(?:dot|period)\s+", ".", spoken, flags=re.I)
    spoken = re.sub(r"\b(?:dot|period)\b", ".", spoken, flags=re.I)
    # "g mail" / "g-mail" → gmail (domain spelling, not a new character guess).
    spoken = re.sub(r"\bg\s*-?\s*mail\b", "gmail", spoken, flags=re.I)
    # Drop remaining whitespace and common ASR trailing punctuation.
    spoken = re.sub(r"\s+", "", spoken)
    spoken = spoken.strip(".,;:!?\"'()[]")
    return spoken


def extract_email(text: str) -> str:
    """Extract a usable email from typed or spoken Retell transcripts."""
    raw = (text or "").strip()
    if not raw:
        return ""
    match = _EMAIL_ADDR_RE.search(raw)
    if match:
        return match.group(0).lower()

    # Strip lead-in then search again before collapsing (keeps domain dots intact).
    stripped = _EMAIL_LEADIN_RE.sub("", raw).strip()
    match = _EMAIL_ADDR_RE.search(stripped)
    if match:
        return match.group(0).lower()

    compact = normalize_spoken_email(raw)
    match = _EMAIL_ADDR_RE.search(compact)
    if match:
        return match.group(0).lower()

    # If collapse glued a lead-in, search for the rightmost email-shaped token.
    matches = _EMAIL_ADDR_RE.findall(compact)
    if matches:
        return matches[-1].lower()
    return ""


def _with_full_question(step: str, clarify: str, *, options: list[str] | None = None) -> str:
    """Full question + specific clarify; never clarify alone for Yes/No steps.

    Does not append internal validation chrome. Does not enumerate income bands.
    """
    del options  # Kept for call-site compatibility; use build_options_help_speech.
    try:
        from .elig_meta import format_question

        question = format_question(step)
    except Exception:
        question = (ELIG_QUESTIONS.get(step) or "").strip()
    clarify = (clarify or "").strip()
    if not question:
        return clarify or "Could you say that again?"
    if not clarify:
        return question
    if clarify.lower() in question.lower():
        return question
    if question.lower().startswith(clarify.lower()):
        return question
    # Avoid stacking the same question when clarify already restates it.
    if "required to continue" in clarify.lower():
        clarify = re.sub(
            r"\s*This field is required to continue\.?",
            "",
            clarify,
            flags=re.I,
        ).strip()
    return f"{question} {clarify}".strip()


def _options_clarify(step: str, options: list[str] | None, fallback: str) -> str:
    """Speak available choices without internal validation or income-band lists."""
    if step == "income_range":
        return _income_range_natural_clarify("", period="Annual")
    try:
        from .live_form import build_options_help_speech

        spoken = build_options_help_speech(step, options)
        if spoken:
            return spoken
    except Exception:
        pass
    return _with_full_question(step, fallback)


def resolve_answer(
    step: str,
    utterance: str,
    *,
    options: list[str] | None = None,
    meta: Any = None,
) -> ResolveResult:
    """Question-aware answer normalization against optional live options."""
    text = normalize_utterance(utterance)
    if not text or text.lower() in {"(unintelligible audio)", "unintelligible"}:
        return ResolveResult(
            clarify=_with_full_question(
                step, "I didn't catch that. Could you say it again?", options=options
            ),
            detail="empty",
        )

    # Navigation is handled by control_intent; treat leftover repeat phrases as repeat.
    if _REPEAT_RE.search(text):
        return ResolveResult(clarify="__REPEAT__", detail="repeat")

    answer_type = getattr(meta, "answer_type", None) if meta is not None else None
    if answer_type is None:
        if step in YES_NO_ELIG_STEPS:
            answer_type = "yes_no"
        elif step in ("child_age", "spouse_age", "person_age"):
            answer_type = "age"
        elif step == "income_when":
            answer_type = "income_period"
        elif step == "income_range":
            answer_type = "income_range"
        elif step in ("marital", "spouse_role", "child_rel", "special"):
            answer_type = "choice"
        else:
            answer_type = "text"

    if step == "support":
        if _is_no(text):
            return ResolveResult(answer="No", detail="support_no")
        if _is_yes(text) or re.search(
            r"\b(student|parent|guardian|disabled|relative|half the year|"
            r"supports me|live with)\b",
            text,
            re.I,
        ):
            if _is_yes(text) and not re.search(
                r"\b(student|parent|guardian|disabled|relative|half|"
                r"live|living|school)\b",
                text,
                re.I,
            ):
                return ResolveResult(
                    clarify=_with_full_question(
                        step,
                        "Please say which situation fits: student supported by a "
                        "parent, you live with the person who supports you, disabled "
                        "and live with them year-round, or a relative supports you.",
                        options=options,
                    ),
                    detail="support_clarify",
                )
            return ResolveResult(answer=text, detail="support_yes")
        return ResolveResult(
            clarify=_with_full_question(
                step,
                "Please say No, or Yes and which support situation fits.",
                options=options,
            ),
            detail="support_unclear",
        )

    if answer_type == "yes_no" or step in YES_NO_ELIG_STEPS:
        yn = classify_yes_no(text)
        if yn == "no":
            return ResolveResult(answer="No", detail="no")
        if yn == "yes":
            return ResolveResult(answer="Yes", detail="yes")
        # Bare single-token Yes/No only — never peel the first word off a mixed phrase.
        if len((text or "").split()) == 1 and _CLEAR_YES_NO.match(text):
            token = (text or "").strip().lower().rstrip(".!,")
            return ResolveResult(
                answer=("No" if token.startswith("n") else "Yes"),
                detail="yes_no_clear",
            )
        # Application start gate: accept clear start/continue intent as Yes.
        if step in (
            "proceed_application",
            "confirm_start_application",
            "programs",
        ) and re.search(
            r"\b(start|continue|proceed|go ahead|quick apply)\b",
            text,
            re.I,
        ):
            return ResolveResult(answer="Yes", detail="yes")
        # Member confirm: a relationship chip name is an in-place correction,
        # not an unclear Yes/No (avoids Claude timeout → stuck confirm).
        if step in ("spouse_confirm", "person_confirm", "child_confirm"):
            low = (text or "").lower()
            if step == "spouse_confirm" and re.search(
                r"\b(spouse|partner)\b.{0,24}\b(correct|right)\b"
                r"|\b(correct|right)\b.{0,24}\b(spouse|partner)\b",
                low,
            ):
                return ResolveResult(answer="Yes", detail="yes")
            if step == "spouse_confirm":
                if re.search(r"\bpartner\b", low):
                    return ResolveResult(
                        answer="Partner", detail="confirm_rel_correction"
                    )
                if re.search(r"\bspouse\b", low):
                    return ResolveResult(
                        answer="Spouse", detail="confirm_rel_correction"
                    )
            else:
                from .live_form import _default_relationship_options

                rel_step = (
                    "person_rel" if step == "person_confirm" else "child_rel"
                )
                pool = list(options or []) or _default_relationship_options(
                    rel_step
                )
                mapped = _map_relationship_label(text, pool)
                if mapped:
                    return ResolveResult(
                        answer=mapped, detail="confirm_rel_correction"
                    )
            from .live_form import _is_relationship_only_age_utterance

            if not _is_relationship_only_age_utterance(text) and (
                _extract_age_numbers(text)
                or re.search(
                    r"\b(baby|babies|toddler|teen(?:ager)?|adult|senior|years?\s+old)\b",
                    low,
                )
            ):
                return ResolveResult(
                    answer=(text or "").strip(),
                    detail="confirm_age_correction",
                )
        # Never attach dashboard button labels (Continue Quick Apply, etc.) here.
        gate_opts = (
            None
            if step
            in ("proceed_application", "confirm_start_application", "programs")
            else options
        )
        return ResolveResult(
            clarify=_with_full_question(
                step,
                "I need a clear yes or no for this one.",
                options=gate_opts,
            ),
            detail="yes_no_unclear",
        )

    if answer_type == "age" or step in ("child_age", "spouse_age", "person_age"):
        # Only real age-range chips — never First Name / Spouse / Partner captions.
        age_opts = _age_option_labels(options)
        if age_opts:
            mapped = _map_age_label(text, age_opts)
            if mapped:
                return ResolveResult(answer=mapped, detail="age_mapped")
            return ResolveResult(
                clarify=_options_clarify(
                    step,
                    age_opts,
                    "Please say the age or the age range that matches the form.",
                ),
                detail="age_ambiguous",
            )
        ages = _extract_age_numbers(text)
        # Ambiguous multi-number ASR without a range connector must clarify —
        # never hand a low-confidence free-text age to Playwright/Claude fallback.
        if len(ages) >= 2 and not _spoken_has_age_range_connector(text):
            return ResolveResult(
                clarify=_with_full_question(
                    step,
                    "I didn't catch a clear age. Please say one age, like twenty five, "
                    "or an age range such as twenty one to sixty three.",
                ),
                detail="age_ambiguous",
            )
        # Without live age chips (Age dropdown still closed), accept a clear
        # spoken age for the Playwright path — which opens chips and verifies.
        if len(ages) == 1 or (
            len(ages) == 2 and _spoken_has_age_range_connector(text)
        ):
            return ResolveResult(answer=text, detail="age_free")
        if re.search(
            r"\b(baby|toddler|teen|young adult|senior|adult|child)\b",
            text,
            re.I,
        ):
            return ResolveResult(answer=text, detail="age_free")
        return ResolveResult(
            clarify=_with_full_question(
                step,
                "Please say an age or age range, like twenty five or sixteen to eighteen.",
            ),
            detail="age_unclear",
        )

    if answer_type == "income_period" or step == "income_when":
        # Do not send income-amount utterances to the period resolver.
        period_guess = _income_period_choice(text)
        looks_like_amount = bool(
            re.search(r"\$|\d|thousand|million|\bk\b|dollar", text, re.I)
        )
        if looks_like_amount and period_guess not in {
            "Annual",
            "Monthly",
            "Bi-weekly",
        }:
            return ResolveResult(
                clarify=_with_full_question(
                    step,
                    "Please say Annual, Monthly, or Bi-weekly for the income period.",
                    options=options,
                ),
                detail="income_period_unclear",
            )
        resolved = _resolve_income_period_label(text, options)
        if not resolved:
            canonical = _income_period_choice(text)
            if canonical in {"Annual", "Monthly", "Bi-weekly"}:
                # No visible options yet — accept canonical mapping.
                if not options:
                    resolved = canonical
                else:
                    # Options present but none matched (e.g. STT near-miss on chip).
                    resolved = ""
        if resolved:
            return ResolveResult(answer=resolved, detail="income_period")
        return ResolveResult(
            clarify=_options_clarify(
                step,
                options or ["Annual", "Monthly", "Bi-weekly"],
                "Please say Annual, Monthly, or Bi-weekly.",
            ),
            detail="income_period_unclear",
        )

    if answer_type == "income_range" or step == "income_range":
        # Classify period-change / reselect before treating the utterance as an amount.
        switch_to = _detect_income_period_switch(text)
        if switch_to:
            return ResolveResult(
                answer=f"__INCOME_PERIOD_SWITCH__:{switch_to}",
                detail="income_period_switch",
            )
        if _wants_income_period_reselect(text):
            return ResolveResult(
                answer="__INCOME_PERIOD_RESELECT__",
                detail="income_period_reselect",
            )
        if _income_amount_scale_unclear(text):
            return ResolveResult(
                clarify=_income_scale_clarify_speech(text, "Annual"),
                detail="income_scale_unclear",
            )
        if options:
            mapped = _map_income_range_label(text, options)
            if mapped:
                return ResolveResult(answer=mapped, detail="income_range_mapped")
            return ResolveResult(
                clarify=_income_range_natural_clarify(text, period="Annual"),
                detail="income_range_ambiguous",
            )
        # Without live options, pass free text through for Playwright remapping.
        if re.search(
            r"\$|\d|thousand|hundred|million|zero|first|lowest|option|dollar|\bk\b",
            text,
            re.I,
        ):
            return ResolveResult(answer=text, detail="income_range_free")
        return ResolveResult(
            clarify=_income_range_natural_clarify(text, period="Annual"),
            detail="income_range_unclear",
        )

    if step == "marital" or (answer_type == "choice" and step == "marital"):
        mapped = _choice_phrase(text)
        if _MARITAL_HINTS.search(text) or mapped != text:
            return ResolveResult(answer=mapped, detail="marital")
        if re.search(r"\bsingle\b", text, re.I):
            return ResolveResult(answer="Never Married", detail="marital_single")
        if options:
            lowered = text.lower()
            for label in options:
                if label.lower() == lowered or label.lower() in lowered:
                    return ResolveResult(answer=label, detail="marital_option")
        return ResolveResult(
            clarify=_options_clarify(
                step,
                options,
                "Please say your marital status: Never Married, Married, Separated, "
                "Divorced, Widowed, Registered Domestic Partner, or Common Law Marriage.",
            ),
            detail="marital_unclear",
        )

    if step == "spouse_role":
        if re.search(r"\bpartner\b", text, re.I):
            return ResolveResult(answer="Partner", detail="spouse_role")
        if re.search(r"\bspouse\b", text, re.I):
            return ResolveResult(answer="Spouse", detail="spouse_role")
        return ResolveResult(
            clarify=_with_full_question(step, "Please say Spouse or Partner."),
            detail="spouse_role_unclear",
        )

    if step in ("child_rel", "person_rel"):
        rel_opts = options or [
            "Child",
            "Stepchild",
            "Grandchild",
            "Foster Child",
            "Other",
        ]
        mapped_rel = _map_relationship_label(text, list(rel_opts))
        if mapped_rel:
            return ResolveResult(answer=mapped_rel, detail="relationship_mapped")
        # Explicit "Other" even when the chip label casing differs.
        if re.search(r"\bother\b", text, re.I) and not re.search(
            r"\b(mother|brother|father|grandmother|grandfather)\b", text, re.I
        ):
            other_hit = next(
                (label for label in rel_opts if label.lower().rstrip(" *") == "other"),
                "Other",
            )
            return ResolveResult(answer=other_hit, detail="relationship_mapped")
        return ResolveResult(
            clarify=_options_clarify(
                step,
                list(rel_opts),
                "Please say the relationship, like Child, Stepchild, or Other.",
            ),
            detail="relationship_unclear",
        )

    if step == "special_person":
        # Member assignment only — never treat condition chips as the person.
        if re.search(
            r"\bsomeone is\b|\bsomeone has\b|\bi said someone\b|"
            r"^\s*(blind|pregnant|breast cancer|cervical cancer|disabled|"
            r"none of the above|ssdi)\s*[.!]?\s*$",
            text,
            re.I,
        ):
            return ResolveResult(
                clarify=_with_full_question(
                    step,
                    "That selects the condition. Who does this apply to? "
                    "Please say the household member's name.",
                ),
                detail="special_person_got_condition",
            )
        cleaned = sanitize_person_name(text)
        if not cleaned:
            return ResolveResult(
                clarify=_with_full_question(
                    step, "Who does this apply to? Please say the person's name."
                ),
                detail="special_person_unclear",
            )
        return ResolveResult(answer=cleaned, detail="special_person")

    if step in (
        "child_first",
        "child_last",
        "spouse_first",
        "spouse_last",
        "person_first",
        "person_last",
    ) or answer_type == "text" and "name" in step:
        # Never route name answers through the choice clicker.
        if re.search(r"\b(first|last)\s+name\b", text, re.I) and not re.search(
            r"\b(first|last)\s+name\s+is\b", text, re.I
        ):
            # Literal "first name" / "last name" without a value — ask again.
            return ResolveResult(
                clarify=_with_full_question(step, "Please say the name."),
                detail="name_missing",
            )
        cleaned = sanitize_person_name(text)
        if not cleaned or cleaned.lower() in {"first name", "last name", "name"}:
            return ResolveResult(
                clarify=_with_full_question(step, "Please say the name."),
                detail="name_unclear",
            )
        return ResolveResult(answer=cleaned, detail="name")

    if answer_type == "choice" and options:
        choice_opts = [
            label
            for label in options
            if label
            and label.lower().rstrip(" *")
            not in {
                "first name",
                "last name",
                "living with me",
                "same mailing address",
                "relationship to me",
                "date of birth (optional)",
            }
        ]
        lowered = re.sub(r"[.!?…]+$", "", text.lower()).strip()
        # Free-text name phrases are not relationship/choice answers.
        if re.search(r"\b(first|last)\s+name\s+is\b", text, re.I):
            return ResolveResult(
                clarify=_with_full_question(
                    step,
                    "Please say the relationship option, such as Child or Grandchild.",
                ),
                detail="choice_got_name_phrase",
            )
        if step in ("child_rel", "person_rel"):
            mapped_rel = _map_relationship_label(text, choice_opts)
            if mapped_rel:
                return ResolveResult(answer=mapped_rel, detail="relationship_mapped")
            return ResolveResult(
                clarify=_options_clarify(
                    step,
                    choice_opts or options,
                    "Please say the relationship, like Child, Stepchild, or Grandchild.",
                ),
                detail="relationship_unclear",
            )
        if step == "special":
            mapped_special = _map_special_condition_labels(text, choice_opts)
            if len(mapped_special) == 1:
                return ResolveResult(
                    answer=mapped_special[0], detail="special_mapped"
                )
            if len(mapped_special) > 1:
                # Multi-select: keep the spoken utterance; Playwright maps again.
                return ResolveResult(answer=text, detail="special_multi")
            return ResolveResult(
                clarify=_options_clarify(
                    step,
                    choice_opts or options,
                    "Please say which condition applies, or none of the above.",
                ),
                detail="special_unclear",
            )
        exact = [label for label in choice_opts if label.lower() == lowered]
        if len(exact) == 1:
            return ResolveResult(answer=exact[0], detail="choice_exact")
        contains = [
            label
            for label in choice_opts
            if label.lower() in lowered or lowered in label.lower()
        ]
        if len(contains) == 1:
            return ResolveResult(answer=contains[0], detail="choice_contains")
        # Never silently pick the first option on ambiguous speech.
        return ResolveResult(
            clarify=_options_clarify(
                step, choice_opts or options, "Please say which option applies."
            ),
            detail="choice_ambiguous",
        )

    if len(text) > 120:
        return ResolveResult(
            clarify=_with_full_question(step, "Please give a shorter answer."),
            detail="too_long",
        )
    return ResolveResult(answer=text, detail="free_text")


def validate_for_step(step: str, utterance: str) -> tuple[str | None, str | None]:
    """Back-compat wrapper around resolve_answer."""
    result = resolve_answer(step, utterance)
    return result.answer, result.clarify


def is_fast_path_resolve(result: ResolveResult) -> bool:
    """True when Python already resolved a clear, safe answer — skip Claude."""
    return bool(result.answer) and result.detail in _FAST_PATH_DETAILS


def _normalize_option_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (label or "").lower())


def _match_option_against_list(
    candidate: str, options: list[str] | None
) -> str | None:
    """Return the exact visible option label, or None if not present."""
    if not candidate or not options:
        return None
    wanted = re.sub(r"[.!?…]+$", "", candidate.strip())
    for label in options:
        if label.lower() == wanted.lower():
            return label
    key = _normalize_option_key(wanted)
    hits = [label for label in options if _normalize_option_key(label) == key]
    if len(hits) == 1:
        return hits[0]
    # Allow short containment only when unique.
    contains = [
        label
        for label in options
        if wanted.lower() in label.lower() or label.lower() in wanted.lower()
    ]
    if len(contains) == 1:
        return contains[0]
    return None


def _special_condition_matches(spoken: str, options: list[str]) -> list[str]:
    """Map spoken special-condition cues onto visible labels (shared with Playwright)."""
    return _map_special_condition_labels(spoken, options or [])


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def validate_claude_result(
    step: str,
    result: ClaudeResolveResult,
    *,
    options: list[str] | None,
    answer_type: str,
    min_confidence: float,
) -> ClaudeResolveResult:
    """Accept Claude output only when it matches the active question/options."""
    if result.needs_clarify or result.intent == "clarify":
        result.accepted = False
        result.reject_reason = result.reject_reason or "needs_clarify"
        return result
    if result.intent not in ("answer", "control"):
        result.accepted = False
        result.reject_reason = "bad_intent"
        return result
    if result.confidence < min_confidence:
        result.accepted = False
        result.reject_reason = "low_confidence"
        result.needs_clarify = True
        return result

    candidate = (result.matched_option or result.answer or "").strip()
    if not candidate:
        result.accepted = False
        result.reject_reason = "empty"
        result.needs_clarify = True
        return result

    # Choice / age / income / yes-no: must map onto a real option when provided.
    option_bound = answer_type in (
        "choice",
        "age",
        "income_range",
        "income_period",
        "yes_no",
    ) or step in (
        "marital",
        "child_rel",
        "person_rel",
        "spouse_role",
        "special",
        "child_age",
        "spouse_age",
        "person_age",
        "income_when",
        "income_range",
        *YES_NO_ELIG_STEPS,
    )
    if option_bound and options:
        matched = _match_option_against_list(candidate, options)
        if not matched and answer_type == "yes_no":
            if _is_yes(candidate):
                matched = _match_option_against_list("Yes", options) or "Yes"
            elif _is_no(candidate):
                matched = _match_option_against_list("No", options) or "No"
        if not matched:
            # Re-run Python mappers as a second gate (never invent chips).
            if answer_type == "age" or step.endswith("_age"):
                matched = _map_age_label(candidate, _age_option_labels(options))
            elif answer_type == "income_range" or step == "income_range":
                mapped = _map_income_range_label(candidate, options)
                matched = mapped if mapped in options else None
            elif answer_type == "income_period" or step == "income_when":
                matched = _resolve_income_period_label(candidate, options) or None
                if not matched:
                    period = _income_period_choice(candidate)
                    if period in {"Annual", "Monthly", "Bi-weekly"}:
                        matched = _match_option_against_list(period, options)
            elif step == "special":
                mapped_list = _special_condition_matches(candidate, options)
                if len(mapped_list) == 1:
                    matched = mapped_list[0]
        if not matched:
            result.accepted = False
            result.reject_reason = "not_in_options"
            result.needs_clarify = True
            result.answer = None
            result.matched_option = None
            return result
        result.matched_option = matched
        result.answer = matched
    elif answer_type == "yes_no" or step in YES_NO_ELIG_STEPS:
        if _is_yes(candidate):
            result.answer = "Yes"
        elif _is_no(candidate):
            result.answer = "No"
        else:
            result.accepted = False
            result.reject_reason = "yes_no_unclear"
            result.needs_clarify = True
            return result
    elif step in (
        "child_first",
        "child_last",
        "spouse_first",
        "spouse_last",
        "person_first",
        "person_last",
        "special_person",
    ) or answer_type == "text":
        cleaned = sanitize_person_name(candidate) if "name" in step or step.endswith(
            ("_first", "_last")
        ) or step == "special_person" else candidate
        if step.endswith(("_first", "_last")) or step in (
            "child_first",
            "child_last",
            "spouse_first",
            "spouse_last",
            "person_first",
            "person_last",
        ):
            cleaned = sanitize_person_name(candidate)
        if not cleaned:
            result.accepted = False
            result.reject_reason = "empty_text"
            result.needs_clarify = True
            return result
        # Never accept condition phrases as special_person members.
        if step == "special_person" and re.search(
            r"\bsomeone is\b|^\s*(blind|pregnant|none of the above)\b",
            cleaned,
            re.I,
        ):
            result.accepted = False
            result.reject_reason = "condition_not_person"
            result.needs_clarify = True
            return result
        result.answer = cleaned
        result.matched_option = None
    else:
        result.answer = candidate

    result.accepted = True
    result.needs_clarify = False
    result.reject_reason = ""
    return result


async def resolve_with_claude(
    step: str,
    utterance: str,
    *,
    question: str = "",
    answer_type: str = "text",
    options: list[str] | None = None,
    member_context: str = "",
    turn_id: str = "",
) -> ClaudeResolveResult:
    """Ask Haiku for a compact structured mapping; validate before accepting.

    Returns a ClaudeResolveResult. Callers must not act on rejected results.
    """
    from . import config

    text = normalize_utterance(utterance)
    started = time.perf_counter()
    empty = ClaudeResolveResult(latency_ms=0.0, reject_reason="skipped")
    if not text or not config.ANTHROPIC_API_KEY:
        empty.reject_reason = "no_key_or_empty"
        return empty
    # Clear Yes/No stays on the Python fast path — do not call Claude.
    if (answer_type == "yes_no" or step in YES_NO_ELIG_STEPS) and _CLEAR_YES_NO.match(
        text
    ):
        empty.reject_reason = "fast_yes_no"
        return empty

    option_list = [item for item in (options or []) if item][:20]
    q = (question or ELIG_QUESTIONS.get(step) or step).strip()
    payload = {
        "step": step,
        "answer_type": answer_type,
        "question": q,
        "options": option_list,
        "member_context": (member_context or "")[:200],
        "utterance": text[:240],
    }
    system = (
        "You resolve spoken answers for a Pointcare eligibility phone form. "
        "Reply with ONLY one JSON object (no markdown) using keys: "
        "intent (answer|clarify|unknown), answer (string|null), "
        "matched_option (exact option string from options or null), "
        "confidence (0-1 number), needs_clarify (boolean), "
        "clarify_prompt (string|null). "
        "Rules: never invent options not in the options list; if options is "
        "non-empty and intent=answer, matched_option MUST be one of them; "
        "if unsure set needs_clarify=true and intent=clarify; never pick the "
        "first option by default; for names return the cleaned personal name; "
        "for ages/income return the matching displayed option text."
    )
    user = (
        "Resolve this caller utterance against the active question.\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    timeout_s = max(0.4, float(getattr(config, "ANTHROPIC_TIMEOUT_S", 1.5) or 1.5))
    min_conf = float(getattr(config, "ANTHROPIC_MIN_CONFIDENCE", 0.72) or 0.72)
    model = (config.ANTHROPIC_MODEL or "claude-haiku-5-5").strip()
    effort = (getattr(config, "ANTHROPIC_EFFORT", "low") or "low").strip()

    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

        async def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": 220,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            # Haiku 5.5: low effort + disabled thinking for latency.
            try:
                kwargs["output_config"] = {"effort": effort}
                kwargs["thinking"] = {"type": "disabled"}
                return await client.messages.create(**kwargs)
            except TypeError:
                kwargs.pop("output_config", None)
                kwargs.pop("thinking", None)
                return await client.messages.create(**kwargs)
            except Exception as first_exc:
                # Older SDK / account may reject thinking or effort — retry bare.
                msg = str(first_exc).lower()
                if "effort" in msg or "thinking" in msg or "output_config" in msg:
                    kwargs.pop("output_config", None)
                    kwargs.pop("thinking", None)
                    return await client.messages.create(**kwargs)
                raise

        resp = await asyncio.wait_for(_call(), timeout=timeout_s)
        out = ""
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", "") == "text":
                out += str(getattr(block, "text", "") or "")
        data = _extract_json_object(out) or {}
        latency_ms = (time.perf_counter() - started) * 1000.0
        result = ClaudeResolveResult(
            intent=str(data.get("intent") or "unknown").strip().lower(),
            answer=(
                str(data.get("answer")).strip()
                if data.get("answer") is not None
                else None
            ),
            matched_option=(
                str(data.get("matched_option")).strip()
                if data.get("matched_option") is not None
                else None
            ),
            confidence=float(data.get("confidence") or 0.0),
            needs_clarify=bool(data.get("needs_clarify", False)),
            clarify_prompt=(
                str(data.get("clarify_prompt")).strip()
                if data.get("clarify_prompt")
                else None
            ),
            latency_ms=latency_ms,
            raw=data,
        )
        result = validate_claude_result(
            step,
            result,
            options=option_list or None,
            answer_type=answer_type,
            min_confidence=min_conf,
        )
        logger.info(
            "claude_resolve step=%s turn_id=%s latency_ms=%.0f accepted=%s "
            "confidence=%.2f intent=%s answer=%r reason=%s",
            step,
            turn_id or "-",
            latency_ms,
            result.accepted,
            result.confidence,
            result.intent,
            (result.answer or "")[:60],
            result.reject_reason or "ok",
        )
        return result
    except asyncio.TimeoutError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.warning(
            "claude_resolve timeout step=%s turn_id=%s latency_ms=%.0f",
            step,
            turn_id or "-",
            latency_ms,
        )
        return ClaudeResolveResult(
            latency_ms=latency_ms,
            reject_reason="timeout",
            needs_clarify=True,
        )
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.exception(
            "claude_resolve failed step=%s turn_id=%s latency_ms=%.0f",
            step,
            turn_id or "-",
            latency_ms,
        )
        return ClaudeResolveResult(
            latency_ms=latency_ms,
            reject_reason="error",
            needs_clarify=True,
        )


async def maybe_clarify_with_claude(
    step: str,
    utterance: str,
    *,
    options: list[str] | None = None,
    meta: Any = None,
    question: str = "",
    member_context: str = "",
    turn_id: str = "",
) -> str | None:
    """Back-compat: return a validated Claude answer string, or None."""
    answer_type = getattr(meta, "answer_type", None) if meta is not None else None
    if not answer_type:
        if step in YES_NO_ELIG_STEPS:
            answer_type = "yes_no"
        elif step in ("child_age", "spouse_age", "person_age"):
            answer_type = "age"
        elif step == "income_when":
            answer_type = "income_period"
        elif step == "income_range":
            answer_type = "income_range"
        else:
            answer_type = "text"
    result = await resolve_with_claude(
        step,
        utterance,
        question=question or ELIG_QUESTIONS.get(step, ""),
        answer_type=str(answer_type),
        options=options,
        member_context=member_context,
        turn_id=turn_id,
    )
    if result.accepted and result.answer:
        return result.answer
    return None
