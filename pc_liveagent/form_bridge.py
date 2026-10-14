"""Retell webhook helpers: parse call payloads and route submit_field tool calls."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from . import config
from .live_form import (
    ELIG_QUESTIONS,
    YES_NO_ELIG_STEPS,
    end_live_form,
    get_session,
    start_live_form,
    submit_field,
)

logger = logging.getLogger(__name__)

# call_id -> { "step:answer" -> applied_at } — each question has its own key.
_elig_applied: dict[str, dict[str, float]] = {}
_call_snapshots: dict[str, dict[str, Any]] = {}
_fallback_tasks: dict[str, asyncio.Task] = {}
# Bumped on eligibility go-back so stale Yes/No from prior questions are ignored.
_transcript_nav_gen: dict[str, int] = {}
_TRANSCRIPT_KEYS = ("transcript", "transcript_object", "transcript_with_tool_calls")
# Let Retell's submit_field win so the tool result can speak the next question.
_FALLBACK_DELAY_S = 2.4
# Suppress duplicate applies for the same step+answer only (not across questions).
_STEP_APPLY_COOLDOWN_S = 6.0


def invalidate_transcript_answers(call_id: str) -> None:
    """After go-back: cancel delayed fallback and bump generation so old Yes/No die."""
    if not call_id:
        return
    _transcript_nav_gen[call_id] = _transcript_nav_gen.get(call_id, 0) + 1
    task = _fallback_tasks.pop(call_id, None)
    if task and not task.done():
        task.cancel()
    _heard.pop(call_id, None)
    logger.info(
        "transcript_answers_invalidated call_id=%s gen=%s",
        call_id,
        _transcript_nav_gen[call_id],
    )


def _apply_key(step: str, answer: str) -> str:
    return f"{step}:{answer.lower()}"


def _mark_applied(call_id: str, step: str, answer: str) -> None:
    bucket = _elig_applied.setdefault(call_id, {})
    bucket[_apply_key(step, answer)] = time.time()


def _skip_duplicate_apply(
    call_id: str,
    step: str,
    answer: str,
    session: Any,
) -> bool:
    """True if this exact step+answer was applied recently.

    Consecutive questions may both be Yes or both be No — keys include step, so
    children:no never blocks student:no.
    """
    if session.last_step_answer == (step, answer.lower()):
        return True
    bucket = _elig_applied.get(call_id) or {}
    applied_at = bucket.get(_apply_key(step, answer))
    if applied_at is None:
        return False
    if time.time() - applied_at >= _STEP_APPLY_COOLDOWN_S:
        return False
    return True


def _field_richness(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        return (1 if value.strip() else 0, len(value.strip()))
    if isinstance(value, list):
        chars = 0
        for item in value:
            if isinstance(item, dict):
                chars += len(_item_text(item))
            else:
                chars += len(str(item or ""))
        return (len(value), chars)
    return (0, 0)


def _prefer_richer(current: Any, incoming: Any) -> Any:
    if incoming is None or incoming == "" or incoming == []:
        return current
    if current is None or current == "" or current == []:
        return incoming
    return incoming if _field_richness(incoming) >= _field_richness(current) else current


def enrich_webhook_call(body: dict[str, Any] | None, call: dict[str, Any] | None) -> dict[str, Any]:
    """Merge top-level Retell transcript fields into the call object.

    transcript_updated payloads may put transcript_with_tool_calls on the webhook
    body itself, not only under call. Mid-call get-call has no transcript.
    """
    merged = dict(call or {})
    src = body if isinstance(body, dict) else {}
    for key in _TRANSCRIPT_KEYS:
        if key in src:
            merged[key] = _prefer_richer(merged.get(key), src.get(key))
        # Some payloads nest a second copy under data.
        data = src.get("data")
        if isinstance(data, dict) and key in data:
            merged[key] = _prefer_richer(merged.get(key), data.get(key))
    return merged


def remember_call_snapshot(call_id: str, call: dict[str, Any]) -> dict[str, Any]:
    """Keep the richest transcript seen for this call across empty follow-up webhooks."""
    if not call_id:
        return dict(call or {})
    prev = _call_snapshots.get(call_id) or {}
    merged = {**prev, **(call or {})}
    for key in _TRANSCRIPT_KEYS:
        merged[key] = _prefer_richer(prev.get(key), (call or {}).get(key))
    _call_snapshots[call_id] = merged
    return merged


def transcript_from_call(call: dict[str, Any]) -> str:
    text = str(call.get("transcript") or "").strip()
    if text:
        return text
    lines: list[str] = []
    for turn in call.get("transcript_with_tool_calls") or call.get("transcript_object") or []:
        if not isinstance(turn, dict):
            continue
        # Skip pure tool-invocation rows when building a readable transcript.
        role = str(turn.get("role") or turn.get("speaker") or "speaker").strip()
        if role.lower() in ("tool_call_invocation", "tool_call_result", "dtmf"):
            continue
        content = _item_text(turn)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _item_text(turn: dict[str, Any]) -> str:
    content = turn.get("content")
    if content is None:
        content = turn.get("content_text")
    text = _turn_text(content)
    if text:
        return text
    words = turn.get("words")
    if isinstance(words, list) and words:
        bits: list[str] = []
        for item in words:
            if isinstance(item, dict):
                bits.append(str(item.get("word") or item.get("text") or ""))
            else:
                bits.append(str(item or ""))
        return " ".join(bit for bit in bits if bit).strip()
    return ""


def target_url_from_call(call: dict[str, Any]) -> str:
    meta = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
    dyn = (
        call.get("retell_llm_dynamic_variables")
        if isinstance(call.get("retell_llm_dynamic_variables"), dict)
        else {}
    )
    return str(
        meta.get("pointcare_url")
        or meta.get("form_url")
        or dyn.get("pointcare_url")
        or dyn.get("form_url")
        or config.POINTCARE_URL
        or ""
    ).strip()


def event_kind(event: str) -> str:
    name = (event or "").lower().replace(" ", "_")
    if "transcript" in name:
        return "transcript"
    if name.endswith("_ended") or name.endswith("_analyzed") or name in ("call_ended", "call_analyzed"):
        return "ended"
    if "start" in name:
        return "start"
    return name


def parse_tool_args(body: dict[str, Any]) -> dict[str, Any]:
    args = body.get("args")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    # Some Retell payloads nest parameters at the top level.
    out: dict[str, Any] = {}
    for key in ("field_id", "value", "action", "answer"):
        if key in body and body[key] is not None:
            out[key] = body[key]
    return out


async def handle_submit_field_tool(
    call_id: str,
    args: dict[str, Any],
    form_url: str = "",
    request_id: str = "",
) -> str:
    field_id = str(args.get("field_id") or args.get("field") or "").strip()
    value = str(args.get("value") or args.get("answer") or "").strip()
    action = str(args.get("action") or "").strip()
    result = await submit_field(
        call_id=call_id,
        field_id=field_id,
        value=value,
        action=action,
        url=form_url,
        request_id=request_id,
    )
    spoken = str(result.get("spoken") or "").strip()
    if spoken.startswith("ASK:"):
        spoken = spoken[4:].strip()
    if not result.get("ok"):
        logger.warning(
            "submit_field not applied field=%s error=%s",
            field_id,
            result.get("error"),
        )
        return spoken or "Please say that answer again."
    if spoken in ("", "Saved.", "ok"):
        return "Please say that answer again."
    return spoken


_heard: dict[str, str] = {}
_pending: dict[str, str] = {}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")

# Agent question → field on the Pointcare page. First match wins.
_QUESTION_FIELDS: tuple[tuple[str, str, str], ...] = (
    (r"first name", "first_name", ""),
    (r"last name", "last_name", ""),
    (r"date of birth|birth date|birthday|\bdob\b", "dob", ""),
    (r"apartment|apt\b|unit number", "apartment", ""),
    (r"street address|home address|mailing address|\baddress\b", "address", ""),
    (r"password", "password", "continue"),
    (r"verification|one-time|\botp\b|code from the text|6 digit|six digit", "otp", "continue"),
    (r"phone|mobile", "phone", "continue"),
    (r"marital", "marital_status", ""),
    (r"children under|children who live", "has_children", ""),
    (r"student|disabled person|other relatives|more than half|provides more than half", "spoken_choice", ""),
    (r"relationship|spouse or partner|age range|living with", "spoken_choice", ""),
    (r"income|annual|monthly|every two weeks", "household_income", ""),
    (r"pregnant|breast cancer|cervical|foster care|special situation|none of the above", "special_conditions", ""),
    (r"social security|\bssn\b", "ssn", ""),
    (r"medicaid", "medicaid_id", ""),
    (r"signature|consent|add signature", "consent_sign", ""),
)


def _turn_text(content: Any) -> str:
    if isinstance(content, list):
        bits = []
        for item in content:
            if isinstance(item, dict):
                bits.append(str(item.get("text") or item.get("word") or ""))
            else:
                bits.append(str(item))
        return " ".join(bits).strip()
    return str(content or "").strip()


def _turns_from_object_list(raw: Any) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    if not isinstance(raw, list):
        return turns
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("speaker") or "").lower()
        if role in ("user", "customer", "caller"):
            kind = "user"
        elif role in ("agent", "assistant"):
            kind = "agent"
        else:
            continue
        text = _item_text(turn)
        if text:
            turns.append((kind, text))
    return turns


def _turns_from_transcript_text(transcript: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for line in (transcript or "").splitlines():
        match = re.match(
            r"^(agent|assistant|user|customer|caller)\s*:\s*(.+)$",
            line.strip(),
            re.I,
        )
        if not match:
            continue
        role = match.group(1).lower()
        kind = "user" if role in ("user", "customer", "caller") else "agent"
        turns.append((kind, match.group(2).strip()))
    return turns


def conversation_turns(call: dict[str, Any], transcript: str) -> list[tuple[str, str]]:
    """Prefer the richest transcript source.

    Live Retell payloads often have a stale transcript_object that is missing the
    latest Yes/No. Using the first non-empty source caused the fallback to miss
    answers for many seconds.
    """
    candidates = [
        _turns_from_object_list(call.get("transcript_object")),
        _turns_from_object_list(call.get("transcript_with_tool_calls")),
        _turns_from_transcript_text(transcript),
    ]
    candidates = [item for item in candidates if item]
    if not candidates:
        return []

    def _score(turns: list[tuple[str, str]]) -> tuple[int, int, int]:
        user_n = sum(1 for kind, _ in turns if kind == "user")
        chars = sum(len(text) for _, text in turns)
        return (user_n, len(turns), chars)

    return max(candidates, key=_score)


def latest_question_and_answer(call: dict[str, Any], transcript: str) -> tuple[str, str]:
    turns = conversation_turns(call, transcript)
    for index in range(len(turns) - 1, -1, -1):
        kind, text = turns[index]
        if kind != "user" or not text:
            continue
        question = ""
        for earlier in range(index - 1, -1, -1):
            if turns[earlier][0] == "agent":
                question = turns[earlier][1]
                break
        return question, text
    return "", ""


def _spoken_email(text: str) -> str:
    match = _EMAIL_RE.search(text or "")
    if match:
        return match.group(0)
    if re.search(r"\bat\b", text or "", flags=re.I) and re.search(r"\bdot\b", text or "", flags=re.I):
        spoken = re.sub(r"\s+at\s+", "@", text, flags=re.I)
        spoken = re.sub(r"\s+dot\s+", ".", spoken, flags=re.I)
        spoken = re.sub(r"\s+", "", spoken)
        match = _EMAIL_RE.search(spoken)
        if match:
            return match.group(0)
    return ""


def resolve_spoken(question: str, answer: str) -> tuple[str, str, str] | None:
    """Map the caller's answer, plus the question they were asked, to a page action."""
    asked = (question or "").lower()
    spoken = (answer or "").strip()
    spoken = re.sub(
        r"^(it is|it's|its|my name is|i am|i'm)\s+",
        "",
        spoken,
        flags=re.I,
    ).strip()
    lowered = spoken.lower()
    if not spoken:
        return None

    email = _spoken_email(spoken)
    says_google = re.search(r"\b(google|gmail|g-mail|g mail)\b", lowered) is not None
    says_apple = re.search(r"\bapple\b", lowered) is not None and not says_google

    if says_google and not email:
        return ("continue_with_google", "", "")
    if says_apple and not email:
        return ("continue_with_apple", "", "")
    if email and (says_google or re.search(r"google|gmail|which email|which account|sign in", asked)):
        return ("continue_with_google", email, "")
    if email and re.search(r"\bemail\b", asked):
        return ("email", email, "continue")
    if email and not asked:
        return ("email", email, "continue")

    if re.fullmatch(r"(skip|skip it|skip for now|not now|no thanks)", lowered):
        return ("skip", "", "")
    if re.fullmatch(r"(next|continue)", lowered):
        return (lowered, "", "")

    yes_no = re.fullmatch(r"(yes|yeah|yep|yup|no|nope|nah)", lowered)
    if yes_no and re.search(r"signature|consent|add signature", asked):
        if lowered.startswith("n"):
            return None
        return ("consent_sign", "", "")
    if yes_no and re.search(r"\b(yes|no)\b|children|student|disabled|insurance|citizen|pregnant", asked):
        return ("spoken_choice", "Yes" if lowered.startswith("y") else "No", "")

    for pattern, field_id, action in _QUESTION_FIELDS:
        if re.search(pattern, asked):
            if field_id == "consent_sign":
                return ("consent_sign", "", "")
            return (field_id, spoken, action)

    if re.fullmatch(r"(ok|okay|hello|hi|hey|thanks|thank you|sure|alright|good|fine)", lowered):
        return None
    return None


def _eligibility_answer_from_utterance(step: str, line: str) -> str | None:
    """Return a browser-ready answer, or None if this utterance should be ignored."""
    text = (line or "").strip()
    if not text:
        return None
    lowered = re.sub(r"[^\w\s']+", " ", text.lower())
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if re.search(
        r"\b(repeat|again|pardon|what did you|say that|hello|hold on|wait|next question)\b",
        lowered,
    ):
        return None
    if step in YES_NO_ELIG_STEPS:
        if re.search(
            r"\b((i|he|she|they|we) said no|no i (do not|don't|do))\b",
            lowered,
        ):
            return "No"
        if re.search(r"\b((i|he|she|they|we) said yes)\b", lowered):
            return "Yes"
        if re.match(r"^(no|nope|nah)\b", lowered):
            return "No"
        if re.match(r"^(yes|yeah|yep|yup)\b", lowered):
            return "Yes"
        return None
    if step == "marital":
        from .live_form import _choice_phrase

        mapped = _choice_phrase(text)
        if mapped != text or re.search(
            r"married|single|divorced|separated|widowed|partner|common law",
            lowered,
        ):
            return mapped
        return None
    # Free-text eligibility fields (names, ages, income, etc.).
    if len(text) > 120:
        return None
    return text


_STEP_QUESTION_CUES: dict[str, str] = {
    "children": r"children|live with and (care|mainly support)|under 19|under the age",
    "student": r"student|parent or legal guardian|18 to 21|18 - 21",
    "disabled": r"disabled|year-round|year round",
    "relatives": r"other relatives|in-laws|step-family|step family",
    "support": r"more than half of your support|half of your support",
    "child_more": r"another child",
    "programs": r"continue to the application|qualify for programs|Medi-Cal",
}

def _yes_no_from_recent_transcript(
    transcript: str,
    step: str,
    call: dict[str, Any] | None = None,
) -> str | None:
    """Best Yes/No for the current eligibility question.

    Retell often splits one spoken question across agent turns and may include an
    earlier ASR miss (Yes) before the caller repeats No. After a repeat request,
    prefer the latest clear Yes/No. Never reuse answers from before a go-back.
    """
    if step not in YES_NO_ELIG_STEPS:
        return None
    turns = conversation_turns(call or {}, transcript)
    cue = _STEP_QUESTION_CUES.get(step, r".")

    # Latest user turn is navigation → do not resolve a Yes/No from older turns.
    for kind, text in reversed(turns):
        if kind != "user":
            continue
        lowered = (text or "").lower()
        if re.search(r"\b(go\s*back|goback|previous question)\b", lowered):
            return None
        break

    # Start after the latest substantive question for this step (ignore tiny
    # fragments like "age of 19?").
    start = None
    for index in range(len(turns) - 1, -1, -1):
        kind, text = turns[index]
        if kind != "agent":
            continue
        stripped = (text or "").strip()
        if len(stripped) < 35:
            continue
        if re.search(cue, stripped, re.I):
            start = index + 1
            break
    # Do not reuse an earlier question's Yes/No (e.g. children No while on student).
    if start is None:
        return None

    # If the caller asked to go back or repeat, only use answers AFTER that request.
    reset_at = None
    for index in range(start, len(turns)):
        kind, text = turns[index]
        if kind != "user":
            continue
        lowered = (text or "").lower()
        if re.search(
            r"\b(go\s*back|goback|previous question|repeat|say that again|"
            r"what was the question)\b",
            lowered,
        ):
            reset_at = index + 1
    scan_from = reset_at if reset_at is not None else start

    answers: list[str] = []
    for kind, text in turns[scan_from:]:
        if kind != "user":
            continue
        lowered = (text or "").lower()
        if re.search(r"\b(go\s*back|goback|previous question)\b", lowered):
            answers.clear()
            continue
        mapped = _eligibility_answer_from_utterance(step, text)
        if mapped in ("Yes", "No"):
            answers.append(mapped)
    if answers:
        # Latest clear Yes/No for this question (handles barge-in and repeat).
        return answers[-1]
    return None


def _spoken_from_result(result: dict[str, Any]) -> str:
    spoken = str(result.get("spoken") or result.get("message") or "").strip()
    if spoken.startswith("ASK:"):
        spoken = spoken[4:].strip()
    return spoken


_fallback_locks: dict[str, asyncio.Lock] = {}


def _fallback_lock(call_id: str) -> asyncio.Lock:
    lock = _fallback_locks.get(call_id)
    if lock is None:
        lock = asyncio.Lock()
        _fallback_locks[call_id] = lock
    return lock


async def _apply_transcript_fallback(
    call_id: str,
    step: str,
    answer: str,
    form_url: str,
    line: str,
) -> None:
    async with _fallback_lock(call_id):
        session = await get_session(call_id)
        if session is None or not session.ready.is_set():
            return
        # Tool path already advanced past this step.
        if session.elig_step != step:
            return
        if _skip_duplicate_apply(call_id, step, answer, session):
            return

        # Claim before awaiting Playwright so delayed duplicates no-op.
        _mark_applied(call_id, step, answer)
        applied_key = _apply_key(step, answer)
        logger.info(
            "Transcript fallback applying eligibility call_id=%s step=%s answer=%r",
            call_id,
            step,
            answer,
        )
        result = await submit_field(
            call_id,
            "eligibility_answer",
            answer,
            url=form_url or session.url,
            request_id=f"transcript:{applied_key}",
        )
        if not result.get("ok"):
            # Allow a later retry if the browser never accepted this answer.
            bucket = _elig_applied.get(call_id)
            if bucket is not None:
                bucket.pop(applied_key, None)
            logger.warning(
                "Transcript fallback failed call_id=%s error=%s",
                call_id,
                result.get("error") or result.get("spoken") or result,
            )
            return

        next_step = str(
            result.get("step") or (await get_session(call_id) or session).elig_step
        )
        spoken = _spoken_from_result(result) or ELIG_QUESTIONS.get(next_step, "")
        # Retell never received a tool result, so nudge it to speak the next question.
        if spoken:
            try:
                from .retell_call import nudge_agent_speak

                await asyncio.to_thread(nudge_agent_speak, call_id, spoken)
            except Exception:
                logger.exception(
                    "Could not nudge Retell after transcript fallback call_id=%s", call_id
                )


async def on_caller_transcript(call_id: str, call: dict[str, Any], form_url: str = "") -> None:
    """Log transcript updates. Apply answers only when TRANSCRIPT_FALLBACK_APPLY is on.

    With Custom LLM WebSocket as primary, transcript webhooks must not compete with
    the server-owned answer pipeline. Keep snapshots for logging/recovery.
    """
    if call_id and call:
        remember_call_snapshot(call_id, call)

    session = await get_session(call_id)
    if session is None or not session.elig_step or not session.ready.is_set():
        return

    snapshot = remember_call_snapshot(call_id, call or {})
    transcript = transcript_from_call(snapshot)
    question, line = latest_question_and_answer(snapshot, transcript)
    step = session.elig_step
    if step in YES_NO_ELIG_STEPS:
        answer = _yes_no_from_recent_transcript(transcript, step, call=snapshot)
    else:
        answer = _eligibility_answer_from_utterance(step, line)
    twtc = snapshot.get("transcript_with_tool_calls")
    tobj = snapshot.get("transcript_object")
    logger.info(
        "Transcript probe call_id=%s step=%s chars=%s twtc=%s tobj=%s latest=%r resolved=%r apply=%s",
        call_id,
        step,
        len(transcript),
        len(twtc) if isinstance(twtc, list) else 0,
        len(tobj) if isinstance(tobj, list) else 0,
        (line or "")[:60],
        answer,
        config.TRANSCRIPT_FALLBACK_APPLY,
    )
    if not config.TRANSCRIPT_FALLBACK_APPLY:
        return
    if not answer:
        return
    # Never apply Yes/No from a go-back / control utterance.
    if re.search(r"\b(go\s*back|goback|previous question)\b", (line or "").lower()):
        return

    applied_key = _apply_key(step, answer)
    fingerprint = f"{applied_key}:{line.strip().lower()[:40]}"
    if _heard.get(call_id) == fingerprint:
        return
    if _skip_duplicate_apply(call_id, step, answer, session):
        return

    _heard[call_id] = fingerprint
    nav_gen = _transcript_nav_gen.get(call_id, 0)
    logger.info(
        "latency call_id=%s stage=transcript_received question=%r latest=%r resolved=%r step=%s gen=%s",
        call_id,
        (question or "")[:100],
        (line or "")[:80],
        answer,
        step,
        nav_gen,
    )

    # Delay so Custom LLM / submit_field can win first.
    prev = _fallback_tasks.pop(call_id, None)
    if prev and not prev.done():
        prev.cancel()

    async def _delayed() -> None:
        try:
            await asyncio.sleep(_FALLBACK_DELAY_S)
            if _transcript_nav_gen.get(call_id, 0) != nav_gen:
                logger.info(
                    "transcript_fallback_skipped_stale_nav call_id=%s gen=%s",
                    call_id,
                    nav_gen,
                )
                return
            await _apply_transcript_fallback(call_id, step, answer, form_url, line)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Delayed transcript fallback failed call_id=%s", call_id)
        finally:
            current = _fallback_tasks.get(call_id)
            if current is asyncio.current_task():
                _fallback_tasks.pop(call_id, None)

    _fallback_tasks[call_id] = asyncio.create_task(_delayed(), name=f"elig-fallback-{call_id}")


async def on_call_started(call_id: str, form_url: str = "") -> dict[str, Any]:
    from .live_form import describe_live_browser

    session = await start_live_form(call_id, form_url or config.POINTCARE_URL)
    browser = await describe_live_browser(session, call_id=call_id)
    return {
        "ok": True,
        "call_id": call_id,
        "url": browser.get("url") or session.url,
        "status": session.status,
        "welcome_done": session.welcome_done,
        "error": session.error,
        "history": session.history[-10:],
        "browser": browser,
    }


async def on_call_ended(
    call_id: str,
    call: dict[str, Any] | None = None,
    form_url: str = "",
) -> dict[str, Any]:
    task = _fallback_tasks.pop(call_id, None)
    if task and not task.done():
        task.cancel()
    # Final call_ended payload often carries the first complete transcript.
    if call:
        try:
            await on_caller_transcript(call_id, call, form_url)
        except Exception:
            logger.exception("End-of-call transcript apply failed call_id=%s", call_id)
    _heard.pop(call_id, None)
    _pending.pop(call_id, None)
    _elig_applied.pop(call_id, None)
    _call_snapshots.pop(call_id, None)
    _fallback_locks.pop(call_id, None)
    _transcript_nav_gen.pop(call_id, None)
    try:
        from .call_turn import drop_controller

        await drop_controller(call_id)
    except Exception:
        logger.exception("drop_controller failed call_id=%s", call_id)
    await end_live_form(call_id)
    return {"ok": True, "call_id": call_id, "status": "ended"}
