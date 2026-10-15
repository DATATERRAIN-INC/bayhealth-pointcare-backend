"""Control intents: repeat, next, go back, redo, change, skip — before answer parse."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .answer_parse import normalize_utterance


@dataclass(frozen=True)
class ControlIntent:
    kind: str  # none | repeat | next | go_back | redo | change | skip
    target: str = ""  # age | name | first_name | last_name | answer | ""
    value: str = ""  # optional new value embedded in the utterance


_NEXT_RE = re.compile(
    r"\b("
    r"what(?:'s|s|\s+is)\s+the\s+next\s+question|"
    r"next\s+question|"
    r"what(?:'s|s)\s+next|"
    r"move\s+on(?:\s+to\s+the\s+next)?"
    r")\b",
    re.I,
)
_REPEAT_RE = re.compile(
    r"\b("
    r"repeat(?:\s+that|\s+the\s+question)?|"
    r"say\s+(?:that|it|the\s+question)\s+again|"
    r"ask\s+(?:the\s+)?(?:question|that)\s+again|"
    r"ask\s+again|"
    r"what(?:'s|s|\s+is)\s+the\s+question|"
    r"what\s+was\s+the\s+question|"
    r"(?:i\s+)?(?:did(?:n't|\s+not)|cannot|can't)\s+hear"
    r"(?:\s+(?:the\s+)?question)?"
    r")\b",
    re.I,
)
# Context / confusion asks — re-explain current step; do not advance the flow.
_CONTEXT_RE = re.compile(
    r"(?:"
    r"^\s*(?:what|huh|pardon|come\s+again|sorry|excuse\s+me)\s*[?.!]?\s*$|"
    r"\bfor\s+what\b|"
    r"\bwhat\s+for\b|"
    r"\bwhat(?:'s|s|\s+is)\s+this(?:\s+(?:for|about))?\b|"
    r"\bwhat(?:'s|s|\s+is)\s+(?:the\s+)?(?:call|this\s+call)\s+about\b|"
    r"\bwhat do you mean\b|"
    r"\bwhy\s+(?:are\s+you|did\s+you)\s+call(?:ing)?\b|"
    r"\bwho\s+(?:is\s+this|are\s+you)\b|"
    r"\bcan you explain\b|"
    r"\bexplain\s+(?:that|this|please)\b|"
    r"\bi\s+(?:don't|do\s+not)\s+understand\b"
    r")",
    re.I,
)
_GO_BACK_RE = re.compile(
    r"\b("
    r"go\s*-?\s*back(?:\s+one\s+step)?|"
    r"goback|"
    r"go\s+backwards?|"
    r"previous\s+question|"
    r"back\s+(?:up|one\s+step|a\s+step)|"
    r"last\s+question\s+again"
    r")\b",
    re.I,
)
_REDO_RE = re.compile(
    r"\b("
    r"start\s+that\s+answer\s+again|"
    r"redo(?:\s+that)?|"
    r"start\s+again|"
    r"i\s+made\s+a\s+mistake|"
    r"let\s+me\s+(?:try|answer)\s+again|"
    r"start\s+over(?:\s+(?:on\s+)?(?:this|that))?"
    r")\b",
    re.I,
)
_CHANGE_RE = re.compile(
    r"\b("
    r"change\s+that|"
    r"change\s+(?:my\s+)?answer|"
    r"correct\s+(?:my\s+)?(?:answer|name|age)|"
    r"i\s+want\s+to\s+correct|"
    r"can\s+i\s+change|"
    r"change\s+(?:what\s+i\s+entered|the\s+(?:name|age|answer))|"
    r"change\s+(?:the\s+)?(?:age|name|first\s+name|last\s+name)"
    r")\b",
    re.I,
)
_SKIP_RE = re.compile(
    r"\b("
    r"skip\s+(?:this|it|the\s+question)?|"
    r"skip\s+this\s+if\s+(?:it\s+is\s+)?optional|"
    r"(?:prefer\s+to\s+)?skip(?:\s+it)?|"
    r"pass\s+on\s+this|"
    r"leave\s+(?:it|this)\s+blank"
    r")\b",
    re.I,
)

_CHANGE_VALUE_RE = re.compile(
    r"change\s+(?:the\s+)?(?P<target>age|name|first\s+name|last\s+name|answer)"
    r"\s+(?:to|into)\s+(?P<value>.+)$",
    re.I,
)
_CORRECT_NAME_RE = re.compile(
    r"(?:correct|change)\s+(?:my\s+)?(?:the\s+)?name\s+(?:to\s+)?(?P<value>.+)$",
    re.I,
)


def detect_control_intent(utterance: str) -> ControlIntent:
    """Classify navigation/correction requests. Prefer specific intents over answers."""
    text = normalize_utterance(utterance)
    if not text:
        return ControlIntent("none")

    # Order matters: next before repeat (both may contain "question").
    if _NEXT_RE.search(text):
        return ControlIntent("next")
    if _GO_BACK_RE.search(text):
        return ControlIntent("go_back")
    if _SKIP_RE.search(text):
        return ControlIntent("skip")

    change_val = _CHANGE_VALUE_RE.search(text)
    if change_val:
        target = re.sub(r"\s+", "_", change_val.group("target").lower())
        value = (change_val.group("value") or "").strip(" .")
        return ControlIntent("change", target=target, value=value)
    name_val = _CORRECT_NAME_RE.search(text)
    if name_val:
        return ControlIntent(
            "change",
            target="name",
            value=(name_val.group("value") or "").strip(" ."),
        )
    if _CHANGE_RE.search(text):
        target = ""
        if re.search(r"\bage\b", text, re.I):
            target = "age"
        elif re.search(r"\bfirst\s+name\b", text, re.I):
            target = "first_name"
        elif re.search(r"\blast\s+name\b", text, re.I):
            target = "last_name"
        elif re.search(r"\brelationship\b", text, re.I):
            target = "relationship"
        elif re.search(r"\bname\b", text, re.I):
            target = "first_name"
        return ControlIntent("change", target=target)
    if _REDO_RE.search(text):
        return ControlIntent("redo")
    if _REPEAT_RE.search(text) or _CONTEXT_RE.search(text):
        return ControlIntent("repeat")
    return ControlIntent("none")
