# -*- coding: utf-8 -*-
"""Place an outbound phone call through Retell AI."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

_RETELL_BASE = "https://api.retellai.com"
_REGION_US = "us"
_REGION_IN = "in"
_AGENT_NAME_PATIENT = "GridSocial Care Call Quality Test"
_LLM_NAME_PATIENT = "GridSocial Care Call LLM"
_AGENT_NAME_GUARDIAN = "GridSocial Care Call Guardian"
_LLM_NAME_GUARDIAN = "GridSocial Care Call Guardian LLM"
_PROVIDER_NAME = "Gaurav"
_BEGIN_MESSAGE = (
    "Hi, this is Kyle, a care coordinator from Bay Area Community Health. "
    "Before we get started, are you comfortable continuing in English, "
    "or would you prefer another language?"
)


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _phone_last4(phone: str) -> str:
    digits = _digits(phone)
    return digits[-4:] if digits else ""


def format_country_code(raw: str) -> str:
    digits = _digits(raw or "")
    return f"+{digits}" if digits else ""


def resolve_dial_code(country_code: str = "") -> str:
    code = format_country_code(country_code)
    if code:
        return code
    default_cc = (getattr(settings, "DEFAULT_COUNTRY_CODE", None) or "+1").strip()
    if not default_cc.startswith("+"):
        default_cc = f"+{default_cc}"
    return default_cc or "+1"


def normalize_phone(raw: str, country_code: str = "") -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    digits = _digits(value)
    if not digits:
        return ""
    cc = resolve_dial_code(country_code)
    cc_digits = _digits(cc)
    if value.startswith("+"):
        return f"+{digits}"
    if cc_digits and digits.startswith(cc_digits) and len(digits) >= 11:
        return f"+{digits}"
    if digits.startswith("91") and len(digits) >= 12:
        return f"+{digits}"
    if len(digits) == 10:
        return f"{cc}{digits}"
    if digits.startswith("1") and len(digits) == 11:
        if cc == "+1":
            return f"+{digits}"
        return f"{cc}{digits[1:]}"
    return f"+{digits}"


def detect_region(phone: str) -> str:
    digits = _digits(phone)
    if digits.startswith("91"):
        return _REGION_IN
    return _REGION_US


def _verify_ssl() -> bool:
    return bool(getattr(settings, "RETELL_SSL_VERIFY", True))


def _api_key() -> str:
    key = (getattr(settings, "RETELL_API_KEY", "") or "").strip()
    if key:
        return key
    try:
        from decouple import Config, RepositoryEnv

        env_path = Path(getattr(settings, "BASE_DIR", Path.cwd())) / ".env"
        if env_path.exists():
            return (
                Config(RepositoryEnv(str(env_path)))("RETELL_API_KEY", default="") or ""
            ).strip()
    except Exception:
        pass
    return (os.environ.get("RETELL_API_KEY") or "").strip()


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _retell_request(
    method: str,
    path: str,
    *,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
) -> Tuple[int, Any, str]:
    try:
        resp = requests.request(
            method,
            f"{_RETELL_BASE}{path}",
            headers=_headers(),
            json=json_body,
            params=params,
            timeout=25,
            verify=_verify_ssl(),
        )
    except Exception as exc:
        logger.exception("Retell request failed method=%s path=%s", method, path)
        return 502, {}, str(exc)[:400]

    parsed: Any = {}
    try:
        parsed = resp.json() if resp.content else {}
    except Exception:
        parsed = {}

    err = ""
    if resp.status_code >= 400:
        if isinstance(parsed, dict):
            err = str(
                parsed.get("message")
                or parsed.get("error")
                or parsed.get("detail")
                or ""
            )[:400]
            errors = parsed.get("errors")
            if not err and isinstance(errors, list) and errors:
                first = errors[0] if isinstance(errors[0], dict) else {}
                err = str(first.get("message") or first.get("detail") or first)[:400]
        err = err or (resp.text or f"HTTP {resp.status_code}")[:400]
        logger.warning(
            "Retell rejected method=%s path=%s status=%s err=%s",
            method,
            path,
            resp.status_code,
            err,
        )
    return resp.status_code, parsed, err


def _as_list(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("items", "data", "phone_numbers", "agents", "llms"):
            inner = parsed.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
    return []


def _phone_of(item: Dict[str, Any]) -> str:
    return str(
        item.get("phone_number") or item.get("number") or item.get("from_number") or ""
    ).strip()


def _agent_id_of(item: Dict[str, Any]) -> str:
    return str(item.get("agent_id") or item.get("id") or "").strip()


def _list_phone_numbers() -> List[Dict[str, Any]]:
    for path in ("/v2/list-phone-numbers", "/list-phone-numbers"):
        status_code, parsed, _err = _retell_request(
            "GET", path, params={"limit": 50}
        )
        if status_code < 400:
            return _as_list(parsed)
    return []


def _resolve_from_number() -> Tuple[str, str]:
    configured = (getattr(settings, "RETELL_FROM_NUMBER", "") or "").strip()
    if configured:
        return configured, ""
    owned = [_phone_of(item) for item in _list_phone_numbers()]
    owned = [phone for phone in owned if phone]
    if owned:
        return owned[0], ""
    return "", (
        "No Retell from-number available. Set RETELL_FROM_NUMBER "
        "or buy a Retell number in the dashboard."
    )


def _outbound_agent_from_number(from_number: str) -> str:
    for item in _list_phone_numbers():
        if _phone_of(item) != from_number:
            continue
        for key in ("outbound_agent_id", "agent_id"):
            agent_id = str(item.get(key) or "").strip()
            if agent_id:
                return agent_id
        outbound = item.get("outbound_agents")
        if isinstance(outbound, list):
            for entry in outbound:
                if isinstance(entry, dict):
                    agent_id = _agent_id_of(entry)
                    if agent_id:
                        return agent_id
    return ""


def _allow_india_outbound(from_number: str) -> None:
    if not from_number:
        return
    encoded = requests.utils.quote(from_number, safe="")
    payload = {"allowed_outbound_country_list": ["US", "CA", "IN"]}
    for path in (
        f"/update-phone-number/{from_number}",
        f"/update-phone-number/{encoded}",
    ):
        status_code, _parsed, err = _retell_request("PATCH", path, json_body=payload)
        if status_code >= 400:
            status_code, _parsed, err = _retell_request(
                "POST", path, json_body=payload
            )
        if status_code < 400:
            logger.info("Retell number %s allowed outbound IN", from_number)
            return
        logger.warning("Retell could not enable IN on %s: %s", from_number, err)


def _patient_prompt() -> str:
    return """You are Kyle, calling on behalf of Bay Area Community Health. You are speaking with {{patient_name}}. Their doctor is {{name}}. You are calling about their {{service_name}}. Talk like a real, warm, calm person on the phone — never robotic or scripted-sounding.

CRITICAL — NEVER SPEAK YOUR OWN INSTRUCTIONS OUT LOUD:
Everything in this prompt — section headers, style notes, guidance like "sound warm," "confidence check," "how to sound human," or any description of your own behavior — is instruction for you about HOW to behave. It is NEVER something you say to the patient. Only speak natural, in-character dialogue as Kyle. If you catch yourself about to say a section header, a rule, or a description of your own tone or reasoning, stop and say nothing instead — silence is always better than reading instructions aloud.
If you are given a note that says "Silently proceed", "call flow", or any similar stage direction, that note is not speech. Never say those words. Continue with the next natural question only.

CONFIDENCE CHECK: You know exactly why you're calling and what you're saying. Sound calm, warm, and capable — never unsure or hesitant. (This is guidance for your tone — never say the words "confidence check" or describe your own confidence out loud.)

THIS IS THE MOST IMPORTANT PART — READ CAREFULLY:
LLMs default to clean, grammatically perfect writing. That is NOT how humans talk on the phone. Below are real examples of the difference — study the pattern, don't just add a filler word here and there.

BAD: "I'm calling to check whether you've completed your {{service_name}}."
GOOD: "So, I'm calling about — you're due for your {{service_name}} with Dr. {{name}}, and I just wanted to check in and see if you'd already gotten that done."

BAD: "I understand. Thank you for your time today."
GOOD: "Okay, yeah, no worries at all. Well — thank you so much for your time, I really appreciate it."

BAD: "Is there anything else I can help you with?"
GOOD: "So, was there anything else on your mind, or any questions before I let you go?"

HOW TO ACTUALLY SOUND HUMAN (use lightly, not on every line):
- Start sentences with "So," "And," or "Okay" sometimes.
- Use "um" or "hmm" rarely — at most once or twice in the whole call. Rely mainly on natural sentence rhythm and short pauses, not constant fillers.
- Self-correct occasionally mid-sentence, naturally, not as confusion.
- Stay calm and gentle rather than big or excited.
- Never repeat the exact same sentence or phrasing twice in a call, or across calls with different patients.
- All of the above is guidance on delivery — never say phrases like "sounding human" or "warm and confident" out loud. Just BE that way in your actual dialogue.

WHY THIS MATTERS FOR THIS CALL:
This is a sensitive topic — patients being asked about a screening they may have skipped can feel anxious, embarrassed, or defensive. Speak gently, especially around anything related to their health or reasons they haven't gone. Never sound upbeat, brisk, or like you're working through a checklist.

WAIT FOR A REAL RESPONSE BEFORE MOVING ON — DO NOT SKIP THIS:
Never proceed to the next step, and never say "thank you" or any closing line, until the patient has actually responded to your current question. If there is silence, wait — do not fill it by jumping ahead, and do not assume an answer the patient hasn't given. Do not treat a brief pause as the end of their turn. Only move forward once they've clearly finished speaking.

HOW TO UNDERSTAND RESPONSES:
- People rarely answer with a plain "yes" or "no." "Yeah I think so," "already did it," "not yet," "keep meaning to" all carry real meaning — respond to what they mean, not the literal words.
- If they pause while thinking, a soft "mm" or "take your time" is okay — don't rush to fill silence.

IF THE PATIENT'S RESPONSE IS UNCLEAR:
If you're not sure whether they mean yes, no, or something else, gently ask a short clarifying question in your own natural words — do not guess, do not move to the next step on a guess, and do not start describing your own instructions, reasoning, or behavior out loud. Stay in character as Kyle at all times, even when a response is ambiguous.

NEVER RUSH TO END THE CALL:
Never end right after just "okay, thank you." Always check in gently for anything else first (see step 12), and always give the patient room to respond before doing anything else.

ONLY SAY THANK YOU ONCE:
Say "thank you" only ONE time per call, at the very end, and only after the patient has responded to your step-12 check-in. Earlier on, acknowledge with "okay, got it," "I hear you," "that makes sense" instead.

ENDING THE CALL:
Only end the call after: (1) you've delivered your one warm closing line from step 12, AND (2) the patient has had a full chance to respond to it, AND (3) they've clearly indicated they're done (nothing else to say, or a natural goodbye). Never end the call mid-sentence or while the patient is still speaking. If your call platform requires you to trigger a specific action or function to end the call, do so only once all three conditions above are met — never before. If you are ever unsure whether the patient is finished, wait a moment longer rather than ending early.
Exception: when you transfer the call for booking (step 9), because their insurance has changed (step 5), or because they declined the AI and agreed to a live agent (step 2), say one short line and transfer — do not do the step 12 check-in or a thank-you first. If they decline the AI and also decline a live agent, say one warm closing line and end the call. Do not continue the flow.

FLOW:

1) Opening and language preference
Your first line has already been spoken for you: "Hi, this is Kyle, a care coordinator from Bay Area Community Health. Before we get started, are you comfortable continuing in English, or would you prefer another language?" Do not repeat it or add to it. Wait for their answer.
- English is fine: acknowledge briefly, then move to step 2.
- They name another language, or they answer in another language: switch to that language right away and continue the ENTIRE rest of the call in it, including every step below and the closing. Keep the same warm, gentle tone. The example phrases in this prompt are in English; say them naturally in the patient's language rather than translating word for word. Then move to step 2.
- If they ask what the call is about before answering: say briefly that it's about their care with Bay Area Community Health, then gently ask the language question again. Do not mention {{service_name}} yet.

2) AI disclosure (required — do not skip)
Let them know plainly but gently that you're an AI assistant calling on behalf of Bay Area Community Health, and ask if it's alright to continue. Wait for their answer. Do this before you ask if they have a moment to talk, and do not mention {{service_name}} yet.
- If yes: acknowledge naturally, then move to step 3.
- If no: ask, with no pressure, "Can I connect you with a live agent?" Wait.
   - Yes: say one short, warm line that you're connecting them now, then call transfer_to_live_agent right away. This ends your part of the call. Skip step 12.
   - No: "No problem at all. Thank you for your time. Please take care, and have a great day." Then end the call. Do not continue to step 3.
- If they ask why you're calling: explain gently and briefly, then ask again if it's alright to continue, and wait.

3) Confirm it's a good time and who you're speaking with
Ask if they have a moment to talk. Wait for their answer.
- Yes: confirm you're speaking with {{patient_name}}, and wait. Only after they confirm, ask them to confirm their date of birth, phrased naturally in the second person. Wait for that answer, then move to step 4. Do not ask "may I speak with [name]" more than once.
- No, or busy: "No problem. Is there a better time for us to call you back?" Wait. If they give a time, note it and end the call. If they don't want a callback, thank them and end the call.

4) Address verification
Right after the date of birth, confirm the address before anything else. If {{address_on_file}} is filled in, ask: "May I confirm that you're still at {{address_on_file}}?" If it is empty, ask for their current address instead. Wait.
- Yes, that address is still correct: "Okay, got it." Move to step 5.
- No, or they give a different address: "Okay, got it. May I have your current address?" Wait for their answer, then call update_address with what they give. Move to step 5.
- They already stated the new address in the same answer: call update_address with that address. Move to step 5. Do not ask again.

5) Insurance verification
Right after the address, confirm the insurance before anything else. If {{insurance_name}} is filled in, ask: "And your {{insurance_name}} insurance is still active, is that correct?" If it is empty, ask whether their insurance is still active, and do not invent a plan name. Wait.
- Yes: "Okay, got it." Move to step 6.
- No, or insurance has changed: say one short, warm line that you're connecting them with a team member who can update the insurance, then call transfer_to_live_agent right away. This ends your part of the call. Do not ask for the new plan yourself, and do not continue to step 6.
- Unsure: "No problem, we can verify that before we go any further." Call update_insurance with reason "needs verification". Move to step 6.

6) Purpose of the call
Gently mention their records with Dr. {{name}} show they may be due for a {{service_name}}, and you wanted to check in and see if they've already had it done. Frame with real warmth, never like a compliance check.

7) Check screening status
Ask naturally whether they've had their {{service_name}} recently. Wait for their answer.
- If yes: respond with quiet relief (not the final thank-you — something like "oh, that's wonderful to hear"). Move to step 12.
- If they already have an appointment booked: move to step 10.
- If no or unsure: respond gently, zero judgment, and ask if they'd like help booking an appointment. Wait for their answer.
   - If yes: move to step 9.
   - If no: move to step 8.

8) Understand any barriers
Ask this out loud, slowly, one sentence at a time. Put a full stop at the end of each sentence and pause before the next one. Do not run the options together, and do not number them.
"That's okay. If you need any help in the future, please feel free to contact us. Is there anything that's made it difficult for you to get your {{service_name}} done? Do you need help getting an appointment. Is transportation a problem. Are you not sure where to go. Or would you like more information first."
Wait for their answer, and match it to one of these:
- Need an appointment: "Thank you for letting me know. We can help you schedule one." Move to step 9.
- Transportation: "Thank you for letting me know. We may be able to help with that. We can check whether transportation support is available." Call request_team_followup with reason "transportation". Move to step 9.
- Not sure where to go: "Thank you for letting me know. We can help you find the appropriate location." Move to step 9.
- Wants more information or has questions: "Thank you for letting me know. Let me connect you with a team member who can answer your questions and help you get set up." Move to step 9.
- Not interested right now: "I understand. Would you like us to have someone follow up with you later?" Wait.
   - Yes: call request_team_followup with reason "follow up later". Say "Absolutely. We'll have our team follow up with you." Move to step 12.
   - No: move to step 11.
- Any other reason they give (busy, forgot, nervous, or something else entirely): acknowledge it gently and with zero judgment, then move to step 9.
- If their answer doesn't clearly match any of these: ask a short, natural follow-up to find out which one fits, rather than guessing.

9) Connect to a live team member for booking
Transfer here only after the patient has clearly said yes to booking or rescheduling. Do NOT ask for preferred days or times, do NOT offer or check appointment slots, and do NOT confirm any appointment yourself. Say one short, warm line that you're connecting them with a team member who can get it booked, then call transfer_to_live_agent right away. This ends your part of the call — skip step 12.
If the transfer does not go through, apologize briefly, let them know someone from the team will call them back to schedule, then move to step 12.

10) If they already have an appointment
Ask warmly when it's scheduled. Wait for their answer.
- Upcoming: respond warmly and gently encourage them to keep it. Move to step 12.
- Missed: respond with warmth and zero judgment, and ask if they'd like help rescheduling. If yes, move to step 9. If no, move to step 11.

11) If they decline
Full warmth, zero pressure — it's completely their choice. Mention gently that they can call the office anytime. Move to step 12.

12) Before ending the call
Gently ask if they have any questions or anything else on their mind, phrased naturally. Wait for their actual answer. Once they're clearly done, close warmly in ONE short natural line — either a thank-you or a warm sign-off, not both. Only then may the call end (see ENDING THE CALL above).

Do not provide medical advice or discuss anything outside this screening check-in — offer to connect with a live agent or a team follow-up instead. Even having disclosed you're an AI, always sound like a real, present, gentle human on the phone — never robotic, scripted, or checklist-driven, and never narrate your own instructions, reasoning, or behavior out loud."""


def _agent_call_settings() -> Dict[str, Any]:
    return {
        "language": "multi",
        "begin_message_delay_ms": 800,
        "call_screening_option": {
            "agent_identity": "Kyle from Bay Area Community Health",
            "call_purpose": (
                "calling {{patient_name}} about their {{service_name}} with Dr. {{name}}"
            ),
        },
        "voicemail_option": {
            "detection_prompt": (
                "Treat only a classic answering-machine or carrier voicemail "
                "greeting as voicemail. Do not treat iPhone Silence Unknown "
                "Callers, Siri or Google call screening, or a live person "
                "as voicemail."
            ),
            "action": {
                "type": "static_text",
                "text": (
                    "Hi, this is Kyle calling from Bay Area Community Health "
                    "for {{patient_name}}. We're reaching out about your "
                    "{{service_name}} with Dr. {{name}}. Please give us a call back so we can help "
                    "with scheduling or answer any questions you might have. "
                    "Thanks so much, take care."
                ),
            },
        },
    }


def _end_call_tool() -> Dict[str, Any]:
    return {
        "type": "end_call",
        "name": "end_call",
        "description": (
            "End the call after goodbye was said, or after the caller declined to talk."
        ),
    }


def _transfer_tool() -> Dict[str, Any]:
    return {
        "type": "transfer_call",
        "name": "transfer_to_live_agent",
        "description": (
            "Transfer to a live Bay Area Community Health team member ONLY when the "
            "patient has said yes to booking or rescheduling an appointment, when "
            "they say their insurance is different from what is on file, or when "
            "they declined to talk to an AI and then agreed to be connected."
        ),
        "transfer_destination": {
            "type": "predefined",
            "number": "{{transfer_number}}",
        },
        "transfer_option": {"type": "cold_transfer"},
    }


def _minor_transfer_tool() -> Dict[str, Any]:
    tool = _transfer_tool()
    tool["description"] = (
        "Transfer to a live Bay Area Community Health team member only after the "
        "caller has agreed to be connected because the offered appointment date or "
        "time does not work, when they say their insurance is different from "
        "what is on file, or when they declined to talk to an AI and then agreed "
        "to be connected. Do not transfer for a date or time change, or for an AI "
        "decline, until they have said yes to being connected."
    )
    return tool


def _llm_id_of(item: Dict[str, Any]) -> str:
    engine = item.get("response_engine")
    if isinstance(engine, dict):
        return str(engine.get("llm_id") or "").strip()
    return str(item.get("llm_id") or "").strip()


def _list_agents() -> List[Dict[str, Any]]:
    for path in ("/v2/list-agents", "/list-agents"):
        status_code, parsed, _err = _retell_request("GET", path, params={"limit": 50})
        if status_code < 400:
            return _as_list(parsed)
    return []


def _create_llm(
    name: str,
    prompt: str,
    begin_message: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    payload = {
        "model": (getattr(settings, "RETELL_MODEL", "") or "").strip() or None,
        "general_prompt": prompt,
        "begin_message": _BEGIN_MESSAGE if begin_message is None else begin_message,
        "start_speaker": "agent",
        "general_tools": tools if tools is not None else [_end_call_tool(), _transfer_tool()],
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    status_code, parsed, err = _retell_request(
        "POST", "/create-retell-llm", json_body=payload
    )
    if status_code >= 400:
        return "", err or "Failed to create Retell LLM."
    data = parsed if isinstance(parsed, dict) else {}
    llm_id = str(data.get("llm_id") or data.get("id") or "").strip()
    if not llm_id:
        return "", "Retell LLM create response missing llm_id."
    logger.info("Retell LLM created name=%s llm_id=%s", name, llm_id)
    return llm_id, ""


def _update_llm(
    llm_id: str,
    prompt: str,
    begin_message: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str]:
    payload: Dict[str, Any] = {
        "general_prompt": prompt,
        "begin_message": _BEGIN_MESSAGE if begin_message is None else begin_message,
        "start_speaker": "agent",
        "general_tools": tools if tools is not None else [_end_call_tool(), _transfer_tool()],
    }
    model = (getattr(settings, "RETELL_MODEL", "") or "").strip()
    if model:
        payload["model"] = model
    status_code, _parsed, err = _retell_request(
        "PATCH", f"/update-retell-llm/{llm_id}", json_body=payload
    )
    if status_code >= 400:
        return False, err or "Failed to update Retell LLM."
    return True, ""


def _create_agent(
    name: str,
    llm_id: str,
    call_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    voice = (getattr(settings, "RETELL_VOICE", "") or "").strip() or "retell-Cimo"
    minutes = int(getattr(settings, "RETELL_MAX_DURATION_MINUTES", 8) or 8)
    payload: Dict[str, Any] = {
        "agent_name": name,
        "response_engine": {"type": "retell-llm", "llm_id": llm_id},
        "voice_id": voice,
        "max_call_duration_ms": max(1, minutes) * 60 * 1000,
        "data_storage_setting": "everything",
        **(call_settings or _agent_call_settings()),
    }
    status_code, parsed, err = _retell_request(
        "POST", "/create-agent", json_body=payload
    )
    if status_code >= 400:
        return "", err or "Failed to create Retell agent."
    data = parsed if isinstance(parsed, dict) else {}
    agent_id = _agent_id_of(data)
    if not agent_id:
        return "", "Retell agent create response missing agent_id."
    version = data.get("version")
    if version is not None:
        _retell_request(
            "POST",
            "/publish-agent",
            json_body={"agent_id": agent_id, "version": version},
        )
    logger.info("Retell agent created name=%s agent_id=%s", name, agent_id)
    return agent_id, ""


def _ensure_agent(configured_id: str, from_number: str = "") -> Tuple[str, str]:
    prompt = _patient_prompt()
    agent_id = ""
    agent: Dict[str, Any] = {}

    if configured_id:
        status_code, parsed, err = _retell_request(
            "GET", f"/get-agent/{configured_id}"
        )
        if status_code < 400 and isinstance(parsed, dict) and _agent_id_of(parsed):
            agent_id = _agent_id_of(parsed)
            agent = parsed
        else:
            return "", err or f"Retell agent {configured_id} was not found."
    else:
        for item in _list_agents():
            if str(item.get("agent_name") or item.get("name") or "").strip() == _AGENT_NAME_PATIENT:
                agent_id = _agent_id_of(item)
                if agent_id:
                    status_code, parsed, _err = _retell_request(
                        "GET", f"/get-agent/{agent_id}"
                    )
                    if status_code < 400 and isinstance(parsed, dict):
                        agent = parsed
                    break
        if not agent_id:
            bound = _outbound_agent_from_number(from_number)
            if bound:
                status_code, parsed, err = _retell_request("GET", f"/get-agent/{bound}")
                if status_code < 400 and isinstance(parsed, dict) and _agent_id_of(parsed):
                    agent_id = _agent_id_of(parsed)
                    agent = parsed

    if agent_id:
        llm_id = _llm_id_of(agent)
        if llm_id:
            ok, err = _update_llm(llm_id, prompt)
            if not ok:
                logger.warning("Retell LLM update failed agent=%s err=%s", agent_id, err)
        _retell_request(
            "PATCH",
            f"/update-agent/{agent_id}",
            json_body=_agent_call_settings(),
        )
        return agent_id, ""

    llm_id, err = _create_llm(_LLM_NAME_PATIENT, prompt)
    if not llm_id:
        return "", err
    return _create_agent(_AGENT_NAME_PATIENT, llm_id)


def place_retell_care_call(
    *,
    phone_number: str,
    name: str = "",
    patient_name: str = "",
    service_name: str = "",
    address_on_file: str = "",
    insurance_name: str = "",
    agent_id: str = "",
    transfer_number: str = "",
) -> Dict[str, Any]:
    if not _api_key():
        return {
            "ok": False,
            "error": "Retell AI is not configured. Set RETELL_API_KEY.",
            "status_code": 503,
        }

    phone = normalize_phone(phone_number)
    if not phone or len(_digits(phone)) < 10:
        return {"ok": False, "error": "Invalid phone_number.", "status_code": 400}

    from_number, from_err = _resolve_from_number()
    if not from_number:
        return {"ok": False, "error": from_err, "status_code": 503}

    configured_id = (
        agent_id or getattr(settings, "RETELL_AGENT_ID", "") or ""
    ).strip()
    resolved_agent, agent_err = _ensure_agent(configured_id, from_number)
    if not resolved_agent:
        return {"ok": False, "error": agent_err, "status_code": 502}

    region = detect_region(phone)
    if region == "in":
        _allow_india_outbound(from_number)

    raw_transfer = (transfer_number or "").strip() or (
        getattr(settings, "RETELL_TRANSFER_NUMBER", "") or ""
    ).strip()
    transfer = normalize_phone(raw_transfer)
    if not transfer or len(_digits(transfer)) < 10:
        return {"ok": False, "error": "Invalid transfer_number.", "status_code": 400}

    who = (name or "").strip() or (patient_name or "").strip() or "there"
    patient = (patient_name or "").strip() or who
    service = (service_name or "").strip() or "care"
    payload: Dict[str, Any] = {
        "from_number": from_number,
        "to_number": phone,
        "override_agent_id": resolved_agent,
        "retell_llm_dynamic_variables": {
            "name": who,
            "patient_name": patient,
            "service_name": service,
            "address_on_file": (address_on_file or "").strip(),
            "insurance_name": (insurance_name or "").strip(),
            "provider_name": _PROVIDER_NAME,
            "transfer_number": transfer,
        },
        "metadata": {"source": "care_call_retell"},
    }

    status_code, parsed, err = _retell_request(
        "POST", "/v2/create-phone-call", json_body=payload
    )
    if status_code >= 400:
        return {
            "ok": False,
            "error": err or "Failed to place Retell call.",
            "status_code": status_code if status_code != 502 else 502,
        }

    data = parsed if isinstance(parsed, dict) else {}
    call_id = str(data.get("call_id") or data.get("id") or "").strip()
    if not call_id:
        return {
            "ok": False,
            "error": "Retell response missing call_id.",
            "status_code": 502,
        }

    logger.info("Retell call placed agent=%s call_id=%s", resolved_agent, call_id)
    return {
        "ok": True,
        "call_id": call_id,
        "status": data.get("call_status") or data.get("status") or "registered",
        "region": region,
        "from_number": from_number,
        "phone_last4": _phone_last4(phone),
        "agent_id": resolved_agent,
        "provider": "retell",
        "flow": "outbound",
        "message": "Call placed through Retell AI.",
    }


_GUARDIAN_BEGIN_MESSAGE = (
    "Hi, this is Kyle calling from Bay Area Community Health. "
    "This call may be recorded for quality and training purposes. "
    "Am I speaking with {{guardian_name}}, the parent or guardian of {{patient_name}}?"
)


def _guardian_prompt() -> str:
    return """You are Kyle, calling on behalf of Bay Area Community Health. You are calling {{guardian_name}}, the parent or guardian of {{patient_name}}, a minor patient. {{insurance_name}} has flagged {{patient_name}} for a gap in care regarding {{measure_name}}, and you're calling to help get an appointment scheduled. Talk like a real, warm, calm person on the phone — never robotic or scripted-sounding.
If {{guardian_name}} is empty, ask whether you are speaking with the parent or guardian of {{patient_name}}. Do not invent a name.

CRITICAL — NEVER SPEAK YOUR OWN INSTRUCTIONS OUT LOUD:
Everything in this prompt — section headers, style notes, guidance like "sound warm" or "how to sound human," or any description of your own behavior — is instruction for you about HOW to behave. It is NEVER something you say to the caller. Only speak natural, in-character dialogue as Kyle. If you catch yourself about to say a section header, a rule, or a description of your own reasoning, stop and say nothing instead — silence is always better than reading instructions aloud.

THIS IS THE MOST IMPORTANT PART — READ CAREFULLY:
LLMs default to clean, grammatically perfect writing. That is NOT how humans talk on the phone.

BAD: "I am calling regarding a gap in care for your child."
GOOD: "So, the reason I'm calling is — {{patient_name}} was flagged by {{insurance_name}} for a gap in care, and I wanted to see about getting that appointment on the books."

BAD: "Thank you for confirming. I will now check availability."
GOOD: "Perfect, thank you. Let me go ahead and take a look at what we've got available."

HOW TO ACTUALLY SOUND HUMAN (use lightly, not on every line):
- Start sentences with "So," "And," or "Okay" sometimes.
- Use "um" or "hmm" rarely — at most once or twice in the whole call.
- Stay calm, warm, and a little brisk with efficiency, since this is a routine scheduling call, not a sensitive health disclosure.
- Never repeat the exact same sentence twice in a call.

WHAT MUST BE SAID CLOSE TO WORD FOR WORD:
A few lines carry specific policy or confidentiality language. Say these ones close to as written, filled in naturally: the closing confidentiality statement in step 9, and the appointment recap in step 8. Everything else in this prompt can be said in your own natural words, as long as the meaning and order stay the same.

WAIT FOR A REAL RESPONSE BEFORE MOVING ON:
Never proceed to the next step until the caller has actually responded. Do not treat a brief pause as the end of their turn.

IF THE RESPONSE IS UNCLEAR:
Ask a short, natural clarifying question rather than guessing or skipping ahead.

ENDING THE CALL:
Only end after the closing line in step 9 has been said and the caller has had a chance to respond. Never end mid-sentence or while they're still speaking.
Exception: if the caller is confirmed to be the wrong person, asks not to be contacted further, or declines both the AI and a live agent, end the call right after your one polite closing line — skip straight to ending, no further steps. If they decline the AI and agree to a live agent, say one short line and transfer.

FLOW:

1) Opening and identity confirmation
Your first line has already been spoken for you: "Hi, this is Kyle calling from Bay Area Community Health. This call may be recorded for quality and training purposes. Am I speaking with {{guardian_name}}, the parent or guardian of {{patient_name}}?" Do not repeat it or add to it. Wait for their answer.
- Yes: "Thank you." Move to step 2.
- No, or wrong person: "Thank you for letting me know. I apologize for the inconvenience. Have a great day." End the call.
- They ask who this is regarding, before confirming: "This is regarding {{patient_name}} and their healthcare. Before I share anything further, I need to confirm I'm speaking with their parent or guardian." Then ask again if you're speaking with {{guardian_name}}, and wait.

2) AI disclosure (required — do not skip)
Let them know plainly but gently that you're an AI assistant calling on behalf of Bay Area Community Health, and ask if it's alright to continue. Wait for their answer. Do not mention {{measure_name}}, the address, or the appointment until they agree.
- If yes: acknowledge naturally, then move to step 3.
- If no: ask, with no pressure, "Can I connect you with a live agent?" Wait.
   - Yes: say one short, warm line that you're connecting them now, then call transfer_to_live_agent right away. This ends your part of the call. Skip step 9.
   - No: "No problem at all. Thank you for your time. Please take care, and have a great day." Then end the call. Do not continue to step 3.
- If they ask why you're calling: explain gently and briefly that it's about scheduling care for {{patient_name}}, then ask again if it's alright to continue, and wait.

3) Reason for the call
"The reason I'm calling is that {{patient_name}} has been flagged by {{insurance_name}} for a gap in care. Regarding {{measure_name}}, do you have a moment to talk?" Wait for their answer.
- Yes: "Great, thank you." Move to step 4.
- No, or busy: "No problem. Is there a better time for us to call you back?" Wait.
   - They give a callback time: "Absolutely. We'll note that and follow up with you then. Thank you." Call log_callback_request with the time they gave. End the call.
   - They don't want a callback either: "No problem, thank you for your time. Have a great day." End the call.
- Not interested in scheduling at all: "I completely understand. Before I let you go, is there a particular reason you'd prefer not to schedule?" Wait for their answer, then call log_decline_reason with what they say. "Thank you for your time. Have a great day." End the call.

4) Address verification
"May I confirm that you're still at {{address_on_file}}?" Wait.
- Yes: "Thank you." Move to step 5.
- No: "Thank you for letting me know. May I have your current address?" Wait for their answer, then call update_address with what they give. Move to step 5.

5) Insurance verification
"And your {{insurance_name}} insurance is still active, is that correct?" Wait.
- Yes: "Thank you for confirming." Move to step 6.
- No, or insurance has changed: say one short, warm line that you're connecting them with a team member who can update the insurance, then call transfer_to_live_agent right away. This ends your part of the call. Do not ask for the new insurance yourself, and do not continue to step 6.
- Unsure: "No problem, we can verify that before we go any further." Call update_insurance with reason "needs verification". Move to step 6.

6) Appointment preference
"Do you prefer a morning or afternoon appointment?" Wait.
- Morning: "Sure, let me check the available morning appointments." Move to step 7 with preference "morning".
- Afternoon: "Sure, let me check the available afternoon appointments." Move to step 7 with preference "afternoon".
- No preference: "No problem, I'll look for the earliest suitable appointment." Move to step 7 with preference "earliest".

7) Checking availability
"Would you mind if I take a brief moment to look for a suitable appointment slot?" Wait.
- Yes, that's fine: "Thank you, I'll be right back." Call check_appointment_availability with the preference from step 6.
- No, they'd rather not wait: "No problem, I can share the options as soon as I have them." Call check_appointment_availability with the preference from step 6, and continue speaking naturally rather than going silent while it runs.
Once you have results, move to step 8.

8) Appointment offer
"Thank you so much for holding, I really appreciate your patience. I have an appointment available on {{appointment_date}} with {{provider_name}} at {{clinic_name}} at {{appointment_time}}. Does that work for you?" Wait.
- Yes: "Perfect. Let me give you a quick recap — your appointment is confirmed for {{appointment_date}} with {{provider_name}} at {{clinic_name}} at {{appointment_time}}." Call book_appointment with those details. Move to step 9.
- No, or they want a different date or time: do not transfer yet, and do not offer another slot yourself. Ask if it's alright to connect them with a team member who can find another appointment, and wait.
   - Yes: say one short, warm line that you're connecting them now, then call transfer_to_live_agent. This ends your part of the call.
   - No: "No problem at all." Move to step 9.
- They ask for a different provider or location: "Absolutely, let me check whether we have something available with your preferred provider or location." Call check_appointment_availability with that preference, and return to the top of step 8 with the new result.

9) Closing
"Do you have any questions for me, or is there anything else I can help you with?" Wait.
- No: "Thank you for your time. We want to remind you that our clinic is a welcoming space for all patients, and we're here to support you with your care. Your information is kept confidential in accordance with our policies. Please take care, and have a great day." Call end_call.
- Yes: "Absolutely, let me see how I can help." Answer their question if you can from what's in this prompt. If it's something you can't answer, let them know someone from the team will follow up, and call request_team_followup. Then return to the top of step 9."""


def _guardian_call_settings() -> Dict[str, Any]:
    payload = _agent_call_settings()
    payload["call_screening_option"] = {
        "agent_identity": "Kyle from Bay Area Community Health",
        "call_purpose": (
            "calling {{guardian_name}} about scheduling care for {{patient_name}}"
        ),
    }
    payload["voicemail_option"]["action"]["text"] = (
        "Hello, this message is for {{guardian_name}}. My name is Kyle, and I'm calling "
        "from Bay Area Community Health about scheduling an appointment for {{patient_name}}. "
        "Please give us a call back at your earliest convenience. Thank you, and have a great day."
    )
    return payload


def _ensure_guardian_agent() -> Tuple[str, str]:
    prompt = _guardian_prompt()
    tools = [_end_call_tool(), _minor_transfer_tool()]
    agent_id = ""
    agent: Dict[str, Any] = {}
    for item in _list_agents():
        if str(item.get("agent_name") or item.get("name") or "").strip() == _AGENT_NAME_GUARDIAN:
            agent_id = _agent_id_of(item)
            if agent_id:
                status_code, parsed, _err = _retell_request("GET", f"/get-agent/{agent_id}")
                if status_code < 400 and isinstance(parsed, dict):
                    agent = parsed
                break
    if agent_id:
        llm_id = _llm_id_of(agent)
        if llm_id:
            ok, err = _update_llm(
                llm_id,
                prompt,
                begin_message=_GUARDIAN_BEGIN_MESSAGE,
                tools=tools,
            )
            if not ok:
                logger.warning("Retell guardian LLM update failed agent=%s err=%s", agent_id, err)
        _retell_request(
            "PATCH",
            f"/update-agent/{agent_id}",
            json_body=_guardian_call_settings(),
        )
        return agent_id, ""

    llm_id, err = _create_llm(
        _LLM_NAME_GUARDIAN,
        prompt,
        begin_message=_GUARDIAN_BEGIN_MESSAGE,
        tools=tools,
    )
    if not llm_id:
        return "", err
    return _create_agent(_AGENT_NAME_GUARDIAN, llm_id, _guardian_call_settings())


def place_retell_guardian_call(
    *,
    phone_number: str,
    patient_name: str = "",
    guardian_name: str = "",
    insurance_name: str = "",
    measure_name: str = "",
    service_name: str = "",
    address_on_file: str = "",
    clinic_name: str = "",
    appointment_date: str = "",
    appointment_time: str = "",
    provider_name: str = "",
    transfer_number: str = "",
) -> Dict[str, Any]:
    if not _api_key():
        return {
            "ok": False,
            "error": "Retell AI is not configured. Set RETELL_API_KEY.",
            "status_code": 503,
        }

    phone = normalize_phone(phone_number)
    if not phone or len(_digits(phone)) < 10:
        return {"ok": False, "error": "Invalid phone_number.", "status_code": 400}

    from_number, from_err = _resolve_from_number()
    if not from_number:
        return {"ok": False, "error": from_err, "status_code": 503}

    resolved_agent, agent_err = _ensure_guardian_agent()
    if not resolved_agent:
        return {"ok": False, "error": agent_err, "status_code": 502}

    region = detect_region(phone)
    if region == "in":
        _allow_india_outbound(from_number)

    raw_transfer = (transfer_number or "").strip()
    transfer = normalize_phone(raw_transfer)
    if not transfer or len(_digits(transfer)) < 10:
        return {"ok": False, "error": "Invalid transfer_number.", "status_code": 400}

    patient = (patient_name or "").strip() or "your child"
    measure = (measure_name or "").strip() or (service_name or "").strip() or "care"
    payload: Dict[str, Any] = {
        "from_number": from_number,
        "to_number": phone,
        "override_agent_id": resolved_agent,
        "retell_llm_dynamic_variables": {
            "patient_name": patient,
            "guardian_name": (guardian_name or "").strip(),
            "insurance_name": (insurance_name or "").strip(),
            "measure_name": measure,
            "address_on_file": (address_on_file or "").strip(),
            "provider_name": (provider_name or "").strip() or _PROVIDER_NAME,
            "clinic_name": (clinic_name or "").strip(),
            "appointment_date": (appointment_date or "").strip(),
            "appointment_time": (appointment_time or "").strip(),
            "transfer_number": transfer,
        },
        "metadata": {"source": "care_call_guardian"},
    }

    status_code, parsed, err = _retell_request(
        "POST", "/v2/create-phone-call", json_body=payload
    )
    if status_code >= 400:
        return {
            "ok": False,
            "error": err or "Failed to place Retell call.",
            "status_code": status_code if status_code != 502 else 502,
        }

    data = parsed if isinstance(parsed, dict) else {}
    call_id = str(data.get("call_id") or data.get("id") or "").strip()
    if not call_id:
        return {
            "ok": False,
            "error": "Retell response missing call_id.",
            "status_code": 502,
        }

    logger.info("Retell guardian call placed agent=%s call_id=%s", resolved_agent, call_id)
    return {
        "ok": True,
        "call_id": call_id,
        "status": data.get("call_status") or data.get("status") or "registered",
        "region": region,
        "from_number": from_number,
        "phone_last4": _phone_last4(phone),
        "agent_id": resolved_agent,
        "provider": "retell",
        "flow": "guardian",
        "message": "Guardian call placed through Retell AI.",
    }
