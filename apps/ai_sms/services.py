# -*- coding: utf-8 -*-
"""SMS conversations with their own full prompt, sent through Twilio."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import uuid
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import requests
from django.conf import settings

from apps.ai_caller.models import Patient
from apps.ai_caller.retell import (
    _AGENT_NAME_PATIENT,
    _PROVIDER_NAME,
    _agent_id_of,
    _as_list,
    _list_agents,
    _llm_id_of,
    _phone_last4,
    _retell_request,
    normalize_phone,
)
from apps.ai_caller.services import _combine_phone, _resolve_live_agent_numbers
from apps.ai_sms.models import SmsConversation
from apps.ai_sms.prompt import (
    assist_ask,
    barriers_ask,
    booked_message,
    closed_reply,
    consent_ask,
    decline_message,
    help_reply,
    live_agent_message,
    minor_closing_ask,
    minor_disclose,
    minor_goodbye,
    minor_preference_ask,
    moment_ask,
    purpose_ask,
    schedule_ask,
    sms_greeting,
    sms_prompt,
)

logger = logging.getLogger(__name__)

_SLOT_1 = "August 28th at 3 PM"
_SLOT_2 = "August 31st at 5 PM"

_CHAT_AGENT_NAME = "GridSocial Care SMS"
_CLINIC_NAME = "Bay Area Community Health"
_TWILIO_API = "https://api.twilio.com/2010-04-01"
_SMS_LIMIT = 1500


def _sms_from_number() -> str:
    """Prefer dedicated care-call SMS number; fall back to warm-transfer Twilio number."""
    return (
        (getattr(settings, "TWILIO_CARECALL_FROM_NUMBER", "") or "").strip()
        or (getattr(settings, "TWILIO_PHONE_NUMBER", "") or "").strip()
    )


def _sms_prompt() -> str:
    return sms_prompt()


def _sms_greeting() -> str:
    return sms_greeting()


def sms_webhook_url() -> str:
    base = (getattr(settings, "BACKEND_URL", "") or "").strip().rstrip("/")
    if not base:
        return ""
    return f"{base}/api/ai-sms/webhook/"


def _twilio_ready() -> Tuple[str, str, str, str]:
    sid = (getattr(settings, "TWILIO_ACCOUNT_SID", "") or "").strip()
    token = (getattr(settings, "TWILIO_AUTH_TOKEN", "") or "").strip()
    from_number = _sms_from_number()
    missing = [
        name
        for name, value in (
            ("TWILIO_ACCOUNT_SID", sid),
            ("TWILIO_AUTH_TOKEN", token),
            ("TWILIO_CARECALL_FROM_NUMBER or TWILIO_PHONE_NUMBER", from_number),
        )
        if not value
    ]
    if missing:
        return "", "", "", "Set " + ", ".join(missing) + " in the shared .env."
    return sid, token, from_number, ""


def _minor_sms_from_number() -> Tuple[str, str]:
    raw = (getattr(settings, "TWILIO_MINOR_SMS_FROM_NUMBER", "") or "").strip()
    number = normalize_phone(raw, "+1")
    digits = "".join(ch for ch in number if ch.isdigit())
    if not number or len(digits) < 10:
        return "", "Set TWILIO_MINOR_SMS_FROM_NUMBER in the shared .env."
    return number, ""


def _first_name(name: str, patient_name: str) -> str:
    full = (patient_name or "").strip() or (name or "").strip()
    return full.split()[0] if full else "there"


def _fill(
    template: str,
    name: str,
    patient_name: str,
    service_name: str,
    transfer_number: str = "",
) -> str:
    return (
        template.replace("{{name}}", name)
        .replace("{{patient_name}}", patient_name)
        .replace("{{patient_first_name}}", _first_name(name, patient_name))
        .replace("{{service_name}}", service_name)
        .replace("{{provider_name}}", _PROVIDER_NAME)
        .replace("{{transfer_number}}", (transfer_number or "").strip())
        .replace(
            "{{office_phone}}",
            _sms_from_number() or "the office",
        )
        .replace("{{slot_1}}", _SLOT_1)
        .replace("{{slot_2}}", _SLOT_2)
    )


def _dynamic_variables(
    name: str,
    patient_name: str,
    service_name: str,
    conversation_so_far: str = "",
    transfer_number: str = "",
) -> Dict[str, str]:
    who = (name or "").strip() or (patient_name or "").strip() or "there"
    patient = (patient_name or "").strip() or who
    service = (service_name or "").strip() or "care"
    return {
        "name": who,
        "patient_name": patient,
        "patient_first_name": _first_name(who, patient),
        "service_name": service,
        "provider_name": _PROVIDER_NAME,
        "conversation_so_far": (conversation_so_far or "").strip() or "No messages yet.",
        "office_phone": _sms_from_number() or "the office",
        "transfer_number": (transfer_number or "").strip(),
        "slot_1": _SLOT_1,
        "slot_2": _SLOT_2,
    }


def _word(text: str) -> str:
    return re.sub(r"[^a-z]", "", (text or "").lower())


def _full_year(year: int) -> int:
    if year >= 100:
        return year
    return 2000 + year if year <= date.today().year % 100 else 1900 + year


def _call_us(sentence: str) -> str:
    return sentence.replace("{{office_phone}}", _sms_from_number() or "the office")


def _remember(conversation, speaker: str, text: str) -> None:
    line = f"{speaker}: {(text or '').strip()}"
    prior = (conversation.transcript or "").rstrip()
    conversation.transcript = f"{prior}\n{line}".strip() if prior else line


def _moment_ask() -> str:
    return moment_ask()


def _consent_ask() -> str:
    return consent_ask()


def _purpose_ask(variables: Dict[str, str]) -> str:
    return purpose_ask(variables["service_name"])


def _yn(text: str) -> str:
    word = _word(text)
    lowered = (text or "").strip().lower()
    if word in {"y", "yes", "yeah", "yep", "sure", "ok", "okay", "alreadydidit"}:
        return "yes"
    if word in {
        "n", "no", "nope", "nah", "notyet", "maybe", "maybelater", "later",
        "notnow", "notrightnow", "idontknow", "idonotknow",
    }:
        return "no"
    if any(
        phrase in lowered
        for phrase in ("maybe later", "not right now", "not now", "i don't know", "i dont know")
    ):
        return "no"
    return ""


def _schedule_ask(service: str) -> str:
    return schedule_ask(service)


def _barriers_ask(service: str) -> str:
    return barriers_ask(service)


def _assist_ask(service: str) -> str:
    return assist_ask(service)


def _reach_number(number: str) -> str:
    cleaned = (number or "").strip()
    if cleaned:
        return cleaned
    return _sms_from_number() or "the office"


def _transfer_from_input(raw: str):
    transfer = normalize_phone((raw or "").strip())
    digits = "".join(ch for ch in transfer if ch.isdigit())
    if not transfer or len(digits) < 10:
        return "", "Invalid transfer_number."
    return transfer, ""


def _booked_message(service: str, number: str) -> str:
    return booked_message(service, _reach_number(number))


def _live_agent_message(number: str) -> str:
    return live_agent_message(_reach_number(number))


def _decline_message(service: str) -> str:
    return decline_message(service, _sms_from_number() or "the office")


def _choice(text: str) -> str:
    match = re.match(r"^\s*([1-5])\b", text or "")
    return match.group(1) if match else ""


def _resume_step(conversation) -> str:
    step = conversation.step or "identity"
    if step != "chat":
        return step
    last = ""
    for line in (conversation.transcript or "").splitlines():
        if line.startswith("Agent:"):
            last = line
    if "Would you like help scheduling" in last:
        return "schedule"
    if "Reply with a number" in last and "Need an appointment" in last:
        return "barriers"
    if "Have you had your" in last:
        return "screening"
    if "Have a great day" in last:
        return "closed"
    return step


def _past_date(text: str) -> bool:
    numbered = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", (text or "").strip())
    if not numbered:
        return False
    month, day, year = (int(part) for part in numbered.groups())
    for candidate in ((month, day), (day, month)):
        try:
            found = date(_full_year(year), candidate[0], candidate[1])
        except ValueError:
            continue
        return found < date.today()
    return False


def _flow_reply(step: str, text: str, variables: Dict[str, str]):
    """Exact texts after the screening question. None leaves the reply to the model."""
    service = variables["service_name"]
    number = variables.get("transfer_number") or ""
    answer = _yn(text)
    number = _choice(text)
    lowered = (text or "").strip().lower()

    if step == "screening":
        if lowered in {"booked", "already booked"} or "already have an appointment" in lowered:
            return (
                f"Great. When is your {service} scheduled? Reply with the date, or MISSED if you missed it.",
                "appointment",
                "ongoing",
            )
        if answer == "yes":
            return "Great, thank you for letting us know.", "ended", "ended"
        if "not sure" in lowered or lowered in {"unsure", "notsure"}:
            return (
                "That's okay. Would you like to speak with a live agent who can help you "
                "check and schedule the screening if needed? Reply YES or NO.",
                "not_sure",
                "ongoing",
            )
        if answer == "no":
            return _schedule_ask(service), "schedule", "ongoing"
        return _purpose_ask(variables), "screening", "ongoing"

    if step == "schedule":
        if answer == "yes":
            return _booked_message(service, number), "ended", "ended"
        if answer == "no":
            return _barriers_ask(service), "barriers", "ongoing"
        return _schedule_ask(service), "schedule", "ongoing"

    if step == "not_sure":
        if answer == "yes":
            return _live_agent_message(number), "ended", "ended"
        if answer == "no":
            return _barriers_ask(service), "barriers", "ongoing"
        return (
            "That's okay. Would you like to speak with a live agent who can help you "
            "check and schedule the screening if needed? Reply YES or NO.",
            "not_sure",
            "ongoing",
        )

    if step == "barriers":
        if number == "1":
            reply = (
                "Thank you for letting me know. We may be able to help with that. "
                "We can help you schedule one.\n\n" + _assist_ask(service)
            )
        elif number == "2":
            reply = (
                "Thank you for letting me know. We may be able to help with that. "
                "We can check whether transportation support is available.\n\n" + _assist_ask(service)
            )
        elif number == "3":
            reply = (
                "Thank you for letting me know. We may be able to help with that. "
                "We can help you find the appropriate location.\n\n" + _assist_ask(service)
            )
        elif number == "4":
            return (
                "Thank you for letting me know. I can connect you with a team member "
                "who can answer your questions.\n\n" + _live_agent_message(number),
                "ended",
                "ended",
            )
        elif number == "5":
            return (
                "I understand. Would you like us to have someone follow up with you later? "
                "Reply YES or NO.",
                "followup",
                "ongoing",
            )
        else:
            reply = (
                "Thank you for letting me know. We may be able to help with that.\n\n"
                + _assist_ask(service)
            )
        return reply, "assist", "ongoing"

    if step == "assist":
        if number == "1":
            return _booked_message(service, number), "ended", "ended"
        if number == "2":
            return _live_agent_message(number), "ended", "ended"
        if number == "3":
            return (
                "Absolutely. We'll have our team follow up with you. Thank you for your time.",
                "ended",
                "ended",
            )
        if number == "4" or answer == "no":
            return _decline_message(service), "ended", "ended"
        return _assist_ask(service), "assist", "ongoing"

    if step == "followup":
        if answer == "yes":
            return (
                "Absolutely. We'll have our team follow up with you. Thank you for your time.",
                "ended",
                "ended",
            )
        if answer == "no":
            return _decline_message(service), "ended", "ended"
        return (
            "I understand. Would you like us to have someone follow up with you later? "
            "Reply YES or NO.",
            "followup",
            "ongoing",
        )

    if step == "appointment":
        if "missed" in lowered or _past_date(text):
            return (
                "Thank you for letting me know. Would you like help rescheduling the "
                "appointment? Reply YES or NO.",
                "reschedule",
                "ongoing",
            )
        if re.search(r"\d", text or ""):
            return (
                "That's great. We'll make a note of that. Please make sure to attend your "
                "appointment, and let us know if you need any assistance.",
                "ended",
                "ended",
            )
        return (
            f"Great. When is your {service} scheduled? Reply with the date, or MISSED if you missed it.",
            "appointment",
            "ongoing",
        )

    if step == "reschedule":
        if answer == "yes":
            return _booked_message(service, number), "ended", "ended"
        if answer == "no":
            return _decline_message(service), "ended", "ended"
        return (
            "Thank you for letting me know. Would you like help rescheduling the "
            "appointment? Reply YES or NO.",
            "reschedule",
            "ongoing",
        )

    if step == "closed":
        return (
            closed_reply(_sms_from_number() or "the office"),
            "ended",
            "ended",
        )

    return None


def _scripted_reply(conversation, text: str, variables: Dict[str, str]) -> Optional[str]:
    """Fixed replies for the name check and date of birth. None lets the model continue."""
    word = _word(text)
    if word in {"stop", "end", "cancel", "unsubscribe"}:
        conversation.step = "ended"
        conversation.status = "ended"
        _remember(conversation, "Patient", text)
        conversation.save(update_fields=["step", "status", "transcript"])
        return ""
    if word == "help":
        reply = help_reply()
        _remember(conversation, "Patient", text)
        _remember(conversation, "Agent", reply)
        conversation.save(update_fields=["transcript"])
        return reply

    variables = dict(variables)
    variables["transfer_number"] = (conversation.transfer_number or "").strip()
    step = conversation.step or "identity"
    if step == "identity":
        _remember(conversation, "Patient", text)
        if word in {"y", "yes", "yeah", "yep", "sure", "ok", "okay"}:
            reply = _consent_ask()
            conversation.step = "consent"
        elif word in {"n", "no", "nope", "nah"}:
            reply = "Sorry about that! Please disregard this message."
            conversation.step = "ended"
            conversation.status = "ended"
        else:
            reply = (
                f"Sorry, I didn't catch that. Am I texting with "
                f"{variables['patient_first_name']}? Reply YES or NO."
            )
        _remember(conversation, "Agent", reply)
        conversation.save(update_fields=["step", "status", "transcript"])
        return reply

    if step == "consent":
        _remember(conversation, "Patient", text)
        if word in {"y", "yes", "yeah", "yep", "sure", "ok", "okay"}:
            reply = _moment_ask()
            conversation.step = "moment"
        elif word in {"n", "no", "nope", "nah"}:
            reply = (
                "Of course. I can connect you with a live team member instead. "
                "Would you like me to do that? Reply YES or NO."
            )
            conversation.step = "consent_no"
        elif any(p in text.lower() for p in ("why", "what is this", "what's this", "who is this")):
            reply = (
                "I'm reaching out to check whether you're up to date with an important "
                "health screening and see if you need any assistance. Is it okay if I "
                "continue? Reply YES or NO."
            )
        else:
            reply = _consent_ask()
        _remember(conversation, "Agent", reply)
        conversation.save(update_fields=["step", "status", "transcript"])
        return reply

    if step == "moment":
        _remember(conversation, "Patient", text)
        if word in {"y", "yes", "yeah", "yep", "sure", "ok", "okay"}:
            reply = _purpose_ask(variables)
            conversation.step = "screening"
        elif word in {"n", "no", "nope", "nah"}:
            reply = (
                "No problem. Is there a better time for us to text you? "
                "Reply with a day and time, or NO."
            )
            conversation.step = "moment_later"
        else:
            reply = _moment_ask()
        _remember(conversation, "Agent", reply)
        conversation.save(update_fields=["step", "status", "transcript"])
        return reply

    if step == "moment_later":
        _remember(conversation, "Patient", text)
        if word in {"n", "no", "nope", "nah"}:
            reply = "No problem, thank you for your time. Have a great day."
        else:
            reply = "Absolutely. We'll follow up with you then. Thank you."
        conversation.step = "ended"
        conversation.status = "ended"
        _remember(conversation, "Agent", reply)
        conversation.save(update_fields=["step", "status", "transcript"])
        return reply

    if step == "consent_no":
        _remember(conversation, "Patient", text)
        if word in {"y", "yes", "yeah", "yep", "sure", "ok", "okay"}:
            reply = _live_agent_message(conversation.transfer_number)
            conversation.step = "ended"
            conversation.status = "ended"
        elif word in {"n", "no", "nope", "nah"}:
            reply = _call_us(
                "No problem. If you need anything, you can reach us at "
                "{{office_phone}}. Have a great day."
            )
            conversation.step = "ended"
            conversation.status = "ended"
        else:
            reply = (
                "Of course. I can connect you with a live team member instead. "
                "Would you like me to do that? Reply YES or NO."
            )
        _remember(conversation, "Agent", reply)
        conversation.save(update_fields=["step", "status", "transcript"])
        return reply

    flowed = _flow_reply(_resume_step(conversation), text, variables)
    if flowed is None:
        return None
    reply, new_step, new_status = flowed
    _remember(conversation, "Patient", text)
    _remember(conversation, "Agent", reply)
    conversation.step = new_step
    conversation.status = new_status
    conversation.save(update_fields=["step", "status", "transcript"])
    return reply


def _minor_variables(conversation) -> Dict[str, str]:
    guardian = (conversation.guardian_name or conversation.name or "").strip()
    patient = (conversation.patient_name or "").strip() or "your child"
    patient_first = patient if patient == "your child" else patient.split()[0]
    return {
        "guardian_name": guardian,
        "guardian_first": guardian.split()[0] if guardian else "",
        "patient_name": patient,
        "patient_first": patient_first,
        "measure": (conversation.service_name or "").strip() or "care",
        "clinic": (conversation.clinic_name or "").strip() or _CLINIC_NAME,
        "provider": (conversation.provider_name or "").strip(),
        "appointment_date": (conversation.appointment_date or "").strip(),
        "appointment_time": (conversation.appointment_time or "").strip(),
        "transfer_number": (conversation.transfer_number or "").strip(),
    }


def _minor_who(variables: Dict[str, str]) -> str:
    patient = variables["patient_first"]
    guardian = variables["guardian_first"]
    if guardian:
        return f"{guardian}, the parent or guardian of {patient}"
    return f"the parent or guardian of {patient}"


def _minor_greeting(variables: Dict[str, str]) -> str:
    return (
        "Hi, this is Kyle from Bay Area Community Health. "
        f"Am I texting with {_minor_who(variables)}? Reply YES or NO. Reply STOP to opt out."
    )


def _minor_disclose() -> str:
    return minor_disclose()


def _minor_reason(variables: Dict[str, str]) -> str:
    patient = variables["patient_first"]
    measure = variables["measure"]
    return (
        f"The reason I'm texting is that {patient} has been flagged for a gap in care. "
        f"Regarding {measure}, do you have a moment to talk? Reply YES or NO."
    )


def _minor_preference_ask() -> str:
    return minor_preference_ask()


def _minor_offer(variables: Dict[str, str], preference: str, reask: bool = False) -> str:
    bits = ["I have an appointment available"]
    if variables["appointment_date"]:
        bits.append(f"on {variables['appointment_date']}")
    if variables["provider"]:
        bits.append(f"with {variables['provider']}")
    if variables["clinic"]:
        bits.append(f"at {variables['clinic']}")
    if variables["appointment_time"]:
        bits.append(f"at {variables['appointment_time']}")
    slot = " ".join(bits)
    if reask:
        lead = ""
    elif preference == "morning":
        lead = "Sure. I looked for a morning appointment. "
    elif preference == "afternoon":
        lead = "Sure. I looked for an afternoon appointment. "
    else:
        lead = "No problem. I looked for the earliest suitable appointment. "
    return f"{lead}{slot}. Does that work for you? Reply YES or NO."


def _minor_recap(variables: Dict[str, str]) -> str:
    bits = ["Your appointment is confirmed"]
    if variables["appointment_date"]:
        bits.append(f"for {variables['appointment_date']}")
    if variables["provider"]:
        bits.append(f"with {variables['provider']}")
    if variables["clinic"]:
        bits.append(f"at {variables['clinic']}")
    if variables["appointment_time"]:
        bits.append(f"at {variables['appointment_time']}")
    return " ".join(bits) + "."


def _minor_closing_ask() -> str:
    return minor_closing_ask()


def _minor_goodbye() -> str:
    return minor_goodbye()


def _minor_preference(text: str) -> str:
    lowered = (text or "").lower()
    if "afternoon" in lowered:
        return "afternoon"
    if "morning" in lowered:
        return "morning"
    if any(
        phrase in lowered
        for phrase in ("either", "no preference", "anytime", "any time", "earliest", "whenever")
    ):
        return "either"
    word = _word(text)
    if word in {"either", "anytime", "earliest", "whenever"}:
        return "either"
    return ""


def _minor_not_interested(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        phrase in lowered
        for phrase in ("not interested", "don't want", "dont want", "no thanks", "not scheduling")
    )


def _minor_next(step: str, text: str, variables: Dict[str, str]):
    """Next guardian text. Always returns reply, step, and status."""
    answer = _yn(text)
    lowered = (text or "").strip().lower()
    who = _minor_who(variables)

    if step == "identity":
        if answer == "yes":
            return _minor_disclose(), "disclose", "ongoing"
        if answer == "no":
            return (
                "Thank you for letting me know. I apologize for the inconvenience. Have a great day.",
                "ended",
                "ended",
            )
        if any(phrase in lowered for phrase in ("why", "what is this", "what's this", "who is this", "regarding")):
            return (
                f"This is regarding {variables['patient_first']} and their healthcare. "
                "Before I share anything further, I need to confirm I'm texting with their "
                f"parent or guardian. Am I texting with {who}? Reply YES or NO.",
                "identity",
                "ongoing",
            )
        return (
            f"Sorry, I didn't catch that. Am I texting with {who}? Reply YES or NO.",
            "identity",
            "ongoing",
        )

    if step == "disclose":
        if answer == "yes":
            return _minor_reason(variables), "reason", "ongoing"
        if answer == "no":
            return (
                "No problem. Can I connect you with a live agent? Reply YES or NO.",
                "disclose_no",
                "ongoing",
            )
        if any(phrase in lowered for phrase in ("why", "what is this", "what's this", "who is this")):
            return (
                f"This is about scheduling care for {variables['patient_first']}. "
                "Is it okay if I continue? Reply YES or NO.",
                "disclose",
                "ongoing",
            )
        return _minor_disclose(), "disclose", "ongoing"

    if step == "disclose_no":
        if answer == "yes":
            return (
                "I'll connect you with a team member. "
                f"Please call {_reach_number(variables['transfer_number'])}.",
                "ended",
                "ended",
            )
        if answer == "no":
            return (
                "No problem at all. Thank you for your time. Please take care, and have a great day.",
                "ended",
                "ended",
            )
        return "Can I connect you with a live agent? Reply YES or NO.", "disclose_no", "ongoing"

    if step == "reason":
        if answer == "yes":
            return _minor_preference_ask(), "preference", "ongoing"
        if _minor_not_interested(text):
            return (
                "I completely understand. Is there a particular reason you'd prefer not to schedule? "
                "Reply with a short reason, or SKIP.",
                "decline_reason",
                "ongoing",
            )
        if answer == "no":
            return (
                "No problem. Is there a better time for us to text you? Reply with a day and time, or NO.",
                "callback",
                "ongoing",
            )
        return _minor_reason(variables), "reason", "ongoing"

    if step == "callback":
        if answer == "no" or _word(text) in {"nope", "nah"}:
            return "No problem, thank you for your time. Have a great day.", "ended", "ended"
        if answer == "yes":
            return "What day and time works best? Reply with a day and time, or NO.", "callback", "ongoing"
        return "Absolutely. We'll follow up with you then. Thank you.", "ended", "ended"

    if step == "decline_reason":
        return "Thank you for your time. Have a great day.", "ended", "ended"

    if step == "preference":
        preference = _minor_preference(text)
        if not preference:
            return _minor_preference_ask(), "preference", "ongoing"
        return _minor_offer(variables, preference), "offer", "ongoing"

    if step == "offer":
        if answer == "yes":
            return _minor_recap(variables) + " " + _minor_closing_ask(), "closing", "ongoing"
        if answer == "no":
            return (
                "Would you like me to connect you with a team member who can find another appointment? "
                "Reply YES or NO.",
                "offer_connect",
                "ongoing",
            )
        if any(phrase in lowered for phrase in ("different provider", "another provider", "different location", "another location", "different clinic")):
            return (
                "Absolutely. Would you like a team member to check that preferred provider or location? "
                "Reply YES or NO.",
                "offer_connect",
                "ongoing",
            )
        return _minor_offer(variables, "either", reask=True), "offer", "ongoing"

    if step == "offer_connect":
        if answer == "yes":
            return (
                "I'll connect you with a team member who can find another appointment. "
                f"Please call {_reach_number(variables['transfer_number'])}.",
                "ended",
                "ended",
            )
        if answer == "no":
            return "No problem at all. " + _minor_closing_ask(), "closing", "ongoing"
        return (
            "Would you like me to connect you with a team member who can find another appointment? "
            "Reply YES or NO.",
            "offer_connect",
            "ongoing",
        )

    if step == "closing":
        if answer == "yes":
            return "Please reply with your question, and someone from the team will follow up.", "question", "ongoing"
        if answer == "no":
            return _minor_goodbye(), "ended", "ended"
        return _minor_closing_ask(), "closing", "ongoing"

    if step == "question":
        return (
            "Someone from the team will follow up with you on that. " + _minor_goodbye(),
            "ended",
            "ended",
        )

    return (
        closed_reply(_sms_from_number() or "the office"),
        "ended",
        "ended",
    )


def _minor_scripted_reply(conversation, text: str) -> str:
    word = _word(text)
    if word in {"stop", "end", "cancel", "unsubscribe"}:
        conversation.step = "ended"
        conversation.status = "ended"
        _remember(conversation, "Guardian", text)
        conversation.save(update_fields=["step", "status", "transcript"])
        return ""
    if word == "help":
        reply = help_reply()
        _remember(conversation, "Guardian", text)
        _remember(conversation, "Agent", reply)
        conversation.save(update_fields=["transcript"])
        return reply

    variables = _minor_variables(conversation)
    prior_step = conversation.step or "identity"
    reply, new_step, new_status = _minor_next(prior_step, text, variables)
    _remember(conversation, "Guardian", text)
    _remember(conversation, "Agent", reply)
    conversation.step = new_step
    conversation.status = new_status
    conversation.save(update_fields=["step", "status", "transcript"])
    return reply


def _voice_llm_id() -> str:
    for item in _list_agents():
        name = str(item.get("agent_name") or item.get("name") or "").strip()
        if name == _AGENT_NAME_PATIENT:
            llm_id = _llm_id_of(item)
            if llm_id:
                return llm_id
    return ""


def _find_chat_agent(configured_id: str) -> Dict[str, Any]:
    if configured_id:
        status_code, parsed, _err = _retell_request("GET", f"/get-chat-agent/{configured_id}")
        if status_code < 400 and isinstance(parsed, dict) and _agent_id_of(parsed):
            return parsed

    agents = []
    for path in ("/list-chat-agents", "/v2/list-chat-agents"):
        status_code, listed, _err = _retell_request("GET", path, params={"limit": 50})
        if status_code < 400:
            agents = _as_list(listed)
            break
    for item in agents:
        if str(item.get("agent_name") or item.get("name") or "").strip() == _CHAT_AGENT_NAME:
            agent_id = _agent_id_of(item)
            if not agent_id:
                continue
            status_code, parsed, _err = _retell_request("GET", f"/get-chat-agent/{agent_id}")
            if status_code < 400 and isinstance(parsed, dict):
                return parsed
            return item
    return {}


def _sms_llm_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "general_prompt": _sms_prompt(),
        "begin_message": None,
        "start_speaker": "agent",
        "default_dynamic_variables": _dynamic_variables("", "", ""),
    }
    model = (getattr(settings, "RETELL_MODEL", "") or "").strip()
    if model:
        payload["model"] = model
    return payload


def _ensure_sms_llm(existing_llm_id: str) -> Tuple[str, str]:
    payload = _sms_llm_payload()
    voice_llm_id = _voice_llm_id()
    if existing_llm_id and existing_llm_id != voice_llm_id:
        status_code, _parsed, err = _retell_request(
            "PATCH",
            f"/update-retell-llm/{existing_llm_id}",
            json_body=payload,
        )
        if status_code >= 400:
            return "", err or "Failed to update the SMS prompt."
        return existing_llm_id, ""

    status_code, created, err = _retell_request(
        "POST",
        "/create-retell-llm",
        json_body=payload,
    )
    if status_code >= 400 or not isinstance(created, dict):
        return "", err or "Failed to create the SMS prompt."
    llm_id = str(created.get("llm_id") or created.get("id") or "").strip()
    if not llm_id:
        return "", "Retell SMS prompt response missing llm_id."
    return llm_id, ""


def _ensure_sms_agent(configured_id: str = "") -> Tuple[str, str]:
    chat_agent = _find_chat_agent(configured_id)
    chat_agent_id = _agent_id_of(chat_agent)
    llm_id, err = _ensure_sms_llm(_llm_id_of(chat_agent))
    if not llm_id:
        return "", err

    engine = {"type": "retell-llm", "llm_id": llm_id}
    if chat_agent_id:
        status_code, _parsed, err = _retell_request(
            "PATCH",
            f"/update-chat-agent/{chat_agent_id}",
            json_body={"response_engine": engine},
        )
        if status_code >= 400:
            return "", err or "Failed to update the SMS agent."
        return chat_agent_id, ""

    status_code, created, err = _retell_request(
        "POST",
        "/create-chat-agent",
        json_body={
            "agent_name": _CHAT_AGENT_NAME,
            "response_engine": engine,
            "language": "en-US",
        },
    )
    if status_code >= 400 or not isinstance(created, dict):
        return "", err or "Failed to create the SMS chat agent."
    chat_agent_id = _agent_id_of(created)
    if not chat_agent_id:
        return "", "Retell chat agent response missing agent_id."
    return chat_agent_id, ""


def _agent_reply(parsed: Any) -> str:
    messages = parsed.get("messages") if isinstance(parsed, dict) else None
    if not isinstance(messages, list):
        return ""
    parts = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").lower()
        if role not in {"agent", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str) and block.strip():
                    parts.append(block.strip())
                elif isinstance(block, dict):
                    text = str(block.get("text") or block.get("content") or "").strip()
                    if text:
                        parts.append(text)
    return parts[-1] if parts else ""


def send_twilio_sms(to_number: str, body: str, from_number: str = "") -> Tuple[str, str]:
    sid, token, default_from, err = _twilio_ready()
    if err:
        return "", err
    from_number = (from_number or default_from).strip()
    text = (body or "").strip()
    if not text:
        return "", "SMS body is empty."
    message_sid = ""
    last_error = ""
    chunks = [text[i : i + _SMS_LIMIT] for i in range(0, len(text), _SMS_LIMIT)]
    for chunk in chunks:
        response = requests.post(
            f"{_TWILIO_API}/Accounts/{sid}/Messages.json",
            auth=(sid, token),
            data={"To": to_number, "From": from_number, "Body": chunk},
            timeout=25,
        )
        try:
            payload = response.json()
        except Exception:
            payload = {}
        if response.status_code >= 400:
            last_error = str(
                payload.get("message") or payload.get("error_message") or response.text
            )[:400]
            logger.warning("Twilio SMS failed status=%s err=%s", response.status_code, last_error)
            return message_sid, last_error or "Twilio rejected the SMS."
        message_sid = str(payload.get("sid") or message_sid)
    return message_sid, ""


def point_twilio_webhook(webhook_url: str, phone_number: str = "") -> None:
    if not webhook_url or "127.0.0.1" in webhook_url or "localhost" in webhook_url:
        return
    sid, token, from_number, err = _twilio_ready()
    if err:
        return
    from_number = (phone_number or from_number).strip()
    listed = requests.get(
        f"{_TWILIO_API}/Accounts/{sid}/IncomingPhoneNumbers.json",
        auth=(sid, token),
        params={"PhoneNumber": from_number},
        timeout=25,
    )
    try:
        numbers = (listed.json() or {}).get("incoming_phone_numbers") or []
    except Exception:
        numbers = []
    if not numbers:
        return
    number_sid = str(numbers[0].get("sid") or "")
    if not number_sid:
        return
    requests.post(
        f"{_TWILIO_API}/Accounts/{sid}/IncomingPhoneNumbers/{number_sid}.json",
        auth=(sid, token),
        data={"SmsUrl": webhook_url, "SmsMethod": "POST"},
        timeout=25,
    )


def _patient_for_sms(patient_id: int, *, user=None) -> Tuple[Optional[Patient], Dict[str, Any]]:
    queryset = Patient.objects.all()
    if user is not None:
        queryset = queryset.filter(user=user)
    patient = queryset.filter(pk=patient_id).first()
    if not patient:
        return None, {
            "ok": False,
            "error": "Patient not found.",
            "status_code": 404,
        }
    if patient.is_blocked:
        return None, {
            "ok": False,
            "error": "Patient is blocked.",
            "status_code": 400,
        }
    return patient, {}


def _transfer_for_patient(patient: Patient) -> Tuple[List[str], str]:
    """Active numbers from ai_caller_liveagentnumber (via caller settings)."""
    numbers = _resolve_live_agent_numbers(patient.user)
    if not numbers:
        return (
            [],
            "No active live agent number configured in caller settings.",
        )
    return numbers, ""


def start_sms_conversation_for_patient(
    patient_id: int,
    *,
    user=None,
    agent_id: str = "",
    webhook_url: str = "",
) -> Dict[str, Any]:
    patient, err = _patient_for_sms(patient_id, user=user)
    if err:
        return err

    dial_number = _combine_phone(patient.country_code, patient.phone_number)
    if not dial_number:
        return {
            "ok": False,
            "error": "Patient phone number is invalid.",
            "status_code": 400,
        }

    numbers, transfer_err = _transfer_for_patient(patient)
    if transfer_err:
        return {"ok": False, "error": transfer_err, "status_code": 400}

    full_name = patient.full_name
    service = (patient.service_name or "").strip() or "care"
    result = start_sms_conversation(
        phone_number=dial_number,
        name=full_name,
        patient_name=full_name,
        service_name=service,
        agent_id=agent_id,
        transfer_number=numbers[0],
        transfer_numbers=numbers,
        webhook_url=webhook_url,
    )
    if result.get("ok"):
        result["patient_id"] = patient.id
        result["transfer_number"] = numbers[0]
        result["transfer_numbers"] = numbers
    return result


def start_minor_sms_conversation_for_patient(
    patient_id: int,
    *,
    user=None,
    guardian_name: str = "",
    appointment_date: str = "",
    appointment_time: str = "",
    webhook_url: str = "",
) -> Dict[str, Any]:
    patient, err = _patient_for_sms(patient_id, user=user)
    if err:
        return err

    dial_number = _combine_phone(patient.country_code, patient.phone_number)
    if not dial_number:
        return {
            "ok": False,
            "error": "Patient phone number is invalid.",
            "status_code": 400,
        }

    numbers, transfer_err = _transfer_for_patient(patient)
    if transfer_err:
        return {"ok": False, "error": transfer_err, "status_code": 400}

    full_name = patient.full_name
    service = (patient.service_name or "").strip() or "care"
    result = start_minor_sms_conversation(
        phone_number=dial_number,
        guardian_name=(guardian_name or "").strip(),
        patient_name=full_name,
        measure_name=service,
        service_name=service,
        clinic_name=_CLINIC_NAME,
        appointment_date=(appointment_date or "").strip(),
        appointment_time=(appointment_time or "").strip(),
        provider_name=(patient.doctor or "").strip(),
        transfer_number=numbers[0],
        transfer_numbers=numbers,
        webhook_url=webhook_url,
    )
    if result.get("ok"):
        result["patient_id"] = patient.id
        result["transfer_number"] = numbers[0]
        result["transfer_numbers"] = numbers
    return result


def start_sms_conversation(
    *,
    phone_number: str,
    name: str = "",
    patient_name: str = "",
    service_name: str = "",
    agent_id: str = "",
    transfer_number: str = "",
    transfer_numbers: Optional[List[str]] = None,
    webhook_url: str = "",
    require_transfer: bool = True,
) -> Dict[str, Any]:
    _sid, _token, from_number, twilio_err = _twilio_ready()
    if twilio_err:
        return {"ok": False, "error": twilio_err, "status_code": 503}

    phone = normalize_phone(phone_number)
    digits = "".join(ch for ch in phone if ch.isdigit())
    if not phone or len(digits) < 10:
        return {"ok": False, "error": "Invalid phone_number.", "status_code": 400}

    transfer = ""
    numbers: List[str] = []
    if transfer_numbers:
        numbers = [str(n).strip() for n in transfer_numbers if str(n).strip()]
    if require_transfer:
        if numbers:
            transfer = numbers[0]
        else:
            transfer, transfer_err = _transfer_from_input(transfer_number)
            if transfer_err:
                return {"ok": False, "error": transfer_err, "status_code": 400}
            numbers = [transfer] if transfer else []
    elif (transfer_number or "").strip():
        transfer, _err = _transfer_from_input(transfer_number)
        if transfer:
            numbers = [transfer]

    configured_id = (agent_id or "").strip()
    chat_agent_id, chat_err = _ensure_sms_agent(configured_id)
    if not chat_agent_id:
        return {"ok": False, "error": chat_err, "status_code": 502}

    variables = _dynamic_variables(
        name, patient_name, service_name, transfer_number=transfer
    )
    status_code, parsed, err = _retell_request(
        "POST",
        "/create-chat",
        json_body={
            "agent_id": chat_agent_id,
            "retell_llm_dynamic_variables": variables,
            "metadata": {"source": "ai_sms"},
        },
    )
    if status_code >= 400 or not isinstance(parsed, dict):
        return {"ok": False, "error": err or "Failed to start Retell chat.", "status_code": 502}

    chat_id = str(parsed.get("chat_id") or "").strip()
    if not chat_id:
        return {"ok": False, "error": "Retell chat response missing chat_id.", "status_code": 502}

    greeting = _fill(
        _sms_greeting(),
        variables["name"],
        variables["patient_name"],
        variables["service_name"],
    )
    message_sid, sms_err = send_twilio_sms(phone, greeting)
    if sms_err:
        return {"ok": False, "error": sms_err, "status_code": 502}

    SmsConversation.objects.filter(to_number=phone, status="ongoing").update(status="ended")
    SmsConversation.objects.create(
        chat_id=chat_id,
        to_number=phone,
        from_number=from_number,
        name=variables["name"],
        patient_name=variables["patient_name"],
        service_name=variables["service_name"],
        transfer_number=transfer,
        transfer_numbers=numbers,
        transfer_number_index=0,
        agent_id=chat_agent_id,
        flow="adult",
        status="ongoing",
        step="identity",
        transcript=f"Agent: {greeting}",
    )
    point_twilio_webhook(webhook_url)
    return {
        "ok": True,
        "chat_id": chat_id,
        "message_sid": message_sid,
        "status": "ongoing",
        "from_number": from_number,
        "phone_last4": _phone_last4(phone),
        "agent_id": chat_agent_id,
        "provider": "twilio",
        "message": "SMS conversation started through Twilio.",
    }


def start_minor_sms_conversation(
    *,
    phone_number: str,
    guardian_name: str = "",
    patient_name: str = "",
    measure_name: str = "",
    service_name: str = "",
    clinic_name: str = "",
    appointment_date: str = "",
    appointment_time: str = "",
    provider_name: str = "",
    transfer_number: str = "",
    transfer_numbers: Optional[List[str]] = None,
    webhook_url: str = "",
) -> Dict[str, Any]:
    _sid, _token, _adult_from, twilio_err = _twilio_ready()
    if twilio_err:
        return {"ok": False, "error": twilio_err, "status_code": 503}
    from_number, minor_from_err = _minor_sms_from_number()
    if minor_from_err:
        return {"ok": False, "error": minor_from_err, "status_code": 503}

    phone = normalize_phone(phone_number)
    digits = "".join(ch for ch in phone if ch.isdigit())
    if not phone or len(digits) < 10:
        return {"ok": False, "error": "Invalid phone_number.", "status_code": 400}

    numbers: List[str] = []
    if transfer_numbers:
        numbers = [str(n).strip() for n in transfer_numbers if str(n).strip()]
        transfer = numbers[0] if numbers else ""
        if not transfer:
            return {"ok": False, "error": "Invalid transfer_number.", "status_code": 400}
    else:
        transfer, transfer_err = _transfer_from_input(transfer_number)
        if transfer_err:
            return {"ok": False, "error": transfer_err, "status_code": 400}
        numbers = [transfer] if transfer else []

    measure = (measure_name or "").strip() or (service_name or "").strip() or "care"
    guardian = (guardian_name or "").strip()
    patient = (patient_name or "").strip()
    conversation = SmsConversation(
        guardian_name=guardian,
        name=guardian,
        patient_name=patient,
        service_name=measure,
        clinic_name=_CLINIC_NAME,
        appointment_date=(appointment_date or "").strip(),
        appointment_time=(appointment_time or "").strip(),
        provider_name=(provider_name or "").strip(),
        transfer_number=transfer,
        transfer_numbers=numbers,
    )
    greeting = _minor_greeting(_minor_variables(conversation))
    message_sid, sms_err = send_twilio_sms(phone, greeting, from_number)
    if sms_err:
        return {"ok": False, "error": sms_err, "status_code": 502}

    chat_id = f"minor-{uuid.uuid4().hex}"
    SmsConversation.objects.filter(to_number=phone, status="ongoing").update(status="ended")
    SmsConversation.objects.create(
        chat_id=chat_id,
        to_number=phone,
        from_number=from_number,
        name=guardian,
        patient_name=patient,
        guardian_name=guardian,
        service_name=measure,
        clinic_name=_CLINIC_NAME,
        appointment_date=(appointment_date or "").strip(),
        appointment_time=(appointment_time or "").strip(),
        provider_name=(provider_name or "").strip(),
        transfer_number=transfer,
        transfer_numbers=numbers,
        transfer_number_index=0,
        flow="minor",
        status="ongoing",
        step="identity",
        transcript=f"Agent: {greeting}",
    )
    point_twilio_webhook(webhook_url, from_number)
    return {
        "ok": True,
        "chat_id": chat_id,
        "message_sid": message_sid,
        "status": "ongoing",
        "from_number": from_number,
        "phone_last4": _phone_last4(phone),
        "provider": "twilio",
        "flow": "minor",
        "message": "Minor SMS conversation started through Twilio.",
    }


def reply_to_inbound_sms(*, from_number: str, body: str, message_sid: str = "") -> str:
    phone = normalize_phone(from_number)
    text = (body or "").strip()
    if not phone or not text:
        return ""

    conversation = (
        SmsConversation.objects.filter(to_number=phone, status="ongoing")
        .order_by("-created_at")
        .first()
    )
    if conversation is None:
        started = start_sms_conversation(
            phone_number=phone,
            name="",
            patient_name="",
            service_name="",
            require_transfer=False,
        )
        if not started.get("ok"):
            logger.warning("Could not start SMS chat for inbound: %s", started.get("error"))
            return ""
        conversation = SmsConversation.objects.filter(chat_id=started["chat_id"]).first()
        if conversation is None:
            return ""

    if message_sid and conversation.last_inbound_sid == message_sid:
        return ""

    if (conversation.flow or "adult") == "minor":
        scripted = _minor_scripted_reply(conversation, text)
    else:
        variables = _dynamic_variables(
            conversation.name,
            conversation.patient_name,
            conversation.service_name,
            conversation.transcript,
            conversation.transfer_number,
        )
        scripted = _scripted_reply(conversation, text, variables)
    if scripted is not None:
        if message_sid:
            conversation.last_inbound_sid = message_sid
            conversation.save(update_fields=["last_inbound_sid"])
        if scripted:
            _sid, sms_err = send_twilio_sms(phone, scripted, conversation.from_number)
            if sms_err:
                logger.warning("Twilio reply failed chat=%s err=%s", conversation.chat_id, sms_err)
            conversation.refresh_from_db()
            from apps.ai_sms.agent_dial import maybe_start_agent_dial_after_reply

            maybe_start_agent_dial_after_reply(conversation, scripted)
        return scripted

    variables = _dynamic_variables(
        conversation.name,
        conversation.patient_name,
        conversation.service_name,
        conversation.transcript,
        conversation.transfer_number,
    )
    _retell_request(
        "PATCH",
        f"/update-chat/{conversation.chat_id}",
        json_body={"override_dynamic_variables": variables},
    )

    status_code, parsed, err = _retell_request(
        "POST",
        "/create-chat-completion",
        json_body={"chat_id": conversation.chat_id, "content": text},
    )
    if status_code >= 400:
        logger.warning("Retell chat completion failed chat=%s err=%s", conversation.chat_id, err)
        return ""

    reply = _fill(
        _agent_reply(parsed),
        variables["name"],
        variables["patient_name"],
        variables["service_name"],
        variables.get("transfer_number") or "",
    )
    _remember(conversation, "Patient", text)
    if reply:
        _remember(conversation, "Agent", reply)
    if message_sid:
        conversation.last_inbound_sid = message_sid
    conversation.save(update_fields=["transcript", "last_inbound_sid"])
    if not reply:
        return ""
    _sid, sms_err = send_twilio_sms(phone, reply, conversation.from_number)
    if sms_err:
        logger.warning("Twilio reply failed chat=%s err=%s", conversation.chat_id, sms_err)
    else:
        conversation.refresh_from_db()
        from apps.ai_sms.agent_dial import maybe_start_agent_dial_after_reply

        maybe_start_agent_dial_after_reply(conversation, reply)
    return reply


def twilio_signature_is_valid(url: str, params: Dict[str, str], signature: str) -> bool:
    _sid, token, _from_number, err = _twilio_ready()
    if err or not signature:
        return False
    signed = url + "".join(key + params[key] for key in sorted(params))
    digest = hmac.new(token.encode("utf-8"), signed.encode("utf-8"), hashlib.sha1).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)
