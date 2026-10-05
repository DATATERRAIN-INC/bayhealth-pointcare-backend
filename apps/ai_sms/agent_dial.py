# -*- coding: utf-8 -*-
"""Dial live agents for SMS transfers, with failover like voice warm-transfer."""

from __future__ import annotations

import logging
from typing import List, Optional
from xml.sax.saxutils import escape

import requests
from django.conf import settings

from apps.ai_sms.models import SmsConversation

logger = logging.getLogger(__name__)

_TWILIO_API = "https://api.twilio.com/2010-04-01"
_AGENT_FAIL_STATUSES = frozenset(
    {"busy", "failed", "no-answer", "canceled", "cancelled"}
)


def _backend_base() -> str:
    return (getattr(settings, "BACKEND_URL", "") or "").rstrip("/")


def _twilio_creds() -> tuple[str, str, str, str]:
    sid = (getattr(settings, "TWILIO_ACCOUNT_SID", "") or "").strip()
    token = (getattr(settings, "TWILIO_AUTH_TOKEN", "") or "").strip()
    from_number = (
        (getattr(settings, "TWILIO_CARECALL_FROM_NUMBER", "") or "").strip()
        or (getattr(settings, "TWILIO_PHONE_NUMBER", "") or "").strip()
    )
    missing = [
        name
        for name, value in (
            ("TWILIO_ACCOUNT_SID", sid),
            ("TWILIO_AUTH_TOKEN", token),
            ("TWILIO from number", from_number),
        )
        if not value
    ]
    if missing:
        return "", "", "", "Set " + ", ".join(missing)
    return sid, token, from_number, ""


def _voice() -> str:
    raw = (getattr(settings, "TWILIO_VOICE", None) or "Joanna").strip()
    if not raw.startswith("Polly."):
        raw = f"Polly.{raw}"
    return raw


def _transfer_list(conversation: SmsConversation) -> List[str]:
    numbers = conversation.transfer_numbers or []
    if isinstance(numbers, list) and numbers:
        return [str(n).strip() for n in numbers if str(n).strip()]
    single = (conversation.transfer_number or "").strip()
    return [single] if single else []


def _is_transfer_reply(reply: str) -> bool:
    text = (reply or "").lower()
    return "please call" in text or "connect you with" in text


def agent_answer_twiml(conversation: SmsConversation) -> str:
    patient = (
        (conversation.patient_name or conversation.name or "").strip() or "a patient"
    )
    service = (conversation.service_name or "").strip() or "care"
    phone = (conversation.to_number or "").strip() or "unknown"
    speak = (
        f"Hello. This is an automated transfer from Bay Area Community Health. "
        f"{patient} needs help with {service}. Their phone number is {phone}. "
        f"Please call them back when you can. Goodbye."
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Say voice=\"{_voice()}\">{escape(speak)}</Say>"
        "<Hangup/></Response>"
    )


def dial_live_agent(conversation: SmsConversation, *, reason: str = "") -> bool:
    """Place outbound call to the current live-agent index. Returns True if dial started."""
    numbers = _transfer_list(conversation)
    if not numbers:
        logger.warning("SMS agent dial skipped chat=%s: no transfer numbers", conversation.chat_id)
        return False

    try:
        idx = int(conversation.transfer_number_index or 0)
    except (TypeError, ValueError):
        idx = 0
    if idx < 0 or idx >= len(numbers):
        return False

    sid, token, from_number, err = _twilio_creds()
    if err:
        logger.warning("SMS agent dial skipped chat=%s: %s", conversation.chat_id, err)
        return False

    base = _backend_base()
    if not base or "127.0.0.1" in base or "localhost" in base:
        logger.warning(
            "SMS agent dial skipped chat=%s: BACKEND_URL must be public for Twilio callbacks",
            conversation.chat_id,
        )
        return False

    destination = numbers[idx]
    answer_url = f"{base}/api/ai-sms/agent-dial/answer/{conversation.chat_id}/"
    status_url = f"{base}/api/ai-sms/agent-dial/status/{conversation.chat_id}/"

    response = requests.post(
        f"{_TWILIO_API}/Accounts/{sid}/Calls.json",
        auth=(sid, token),
        data={
            "To": destination,
            "From": from_number,
            "Url": answer_url,
            "Method": "POST",
            "StatusCallback": status_url,
            "StatusCallbackMethod": "POST",
            "StatusCallbackEvent": "initiated ringing answered completed",
            "Timeout": "25",
        },
        timeout=25,
    )
    try:
        payload = response.json()
    except Exception:
        payload = {}
    if response.status_code >= 400:
        logger.warning(
            "SMS agent dial create failed chat=%s to=%s status=%s body=%s",
            conversation.chat_id,
            destination,
            response.status_code,
            str(payload)[:300],
        )
        return failover_to_next_live_agent(conversation, reason="dial_create_failed")

    call_sid = str(payload.get("sid") or "").strip()
    conversation.transfer_number = destination
    conversation.transfer_number_index = idx
    conversation.dial_call_sid = call_sid
    conversation.transfer_status = "dialing"
    conversation.save(
        update_fields=[
            "transfer_number",
            "transfer_number_index",
            "dial_call_sid",
            "transfer_status",
        ]
    )
    logger.info(
        "SMS agent dial started chat=%s to=%s index=%s sid=%s reason=%s",
        conversation.chat_id,
        destination,
        idx,
        call_sid,
        reason or "",
    )
    return True


def failover_to_next_live_agent(
    conversation: SmsConversation, *, reason: str = ""
) -> bool:
    """If current agent did not answer, dial the next active number."""
    numbers = _transfer_list(conversation)
    if len(numbers) < 2:
        conversation.transfer_status = "agent_unreachable"
        conversation.dial_call_sid = ""
        conversation.save(update_fields=["transfer_status", "dial_call_sid"])
        logger.info(
            "SMS agent failover exhausted chat=%s reason=%s tried=%s",
            conversation.chat_id,
            reason or "unknown",
            len(numbers),
        )
        _notify_patient_unreachable(conversation)
        return False

    try:
        idx = int(conversation.transfer_number_index or 0)
    except (TypeError, ValueError):
        idx = 0
    next_idx = idx + 1
    if next_idx >= len(numbers):
        conversation.transfer_status = "agent_unreachable"
        conversation.dial_call_sid = ""
        conversation.save(update_fields=["transfer_status", "dial_call_sid"])
        logger.info(
            "SMS agent failover exhausted chat=%s reason=%s tried=%s",
            conversation.chat_id,
            reason or "unknown",
            len(numbers),
        )
        _notify_patient_unreachable(conversation)
        return False

    previous = numbers[idx] if idx < len(numbers) else conversation.transfer_number
    nxt = numbers[next_idx]
    conversation.transfer_number_index = next_idx
    conversation.transfer_number = nxt
    conversation.dial_call_sid = ""
    conversation.transfer_status = "failover_dialing"
    conversation.save(
        update_fields=[
            "transfer_number_index",
            "transfer_number",
            "dial_call_sid",
            "transfer_status",
        ]
    )
    logger.info(
        "SMS agent failover chat=%s reason=%s from=%s to=%s index=%s",
        conversation.chat_id,
        reason or "no_answer",
        previous,
        nxt,
        next_idx,
    )
    return dial_live_agent(conversation, reason=reason or "failover")


def maybe_failover_on_agent_status(
    conversation: SmsConversation, *, call_status: str, call_sid: str = ""
) -> bool:
    status = (call_status or "").strip().lower()
    if status not in _AGENT_FAIL_STATUSES:
        if status == "completed" and conversation.transfer_status == "dialing":
            # Completed without prior answered may still be no-answer in some cases;
            # DialCallStatus is the authoritative fail signal.
            pass
        if status == "in-progress" or status == "answered":
            conversation.transfer_status = "connected"
            conversation.save(update_fields=["transfer_status"])
        return False

    dial_sid = (conversation.dial_call_sid or "").strip()
    call_sid = (call_sid or "").strip()
    if not dial_sid:
        return False
    if call_sid and call_sid != dial_sid:
        return False
    if conversation.transfer_status == "connected":
        return False
    return failover_to_next_live_agent(conversation, reason=status)


def _notify_patient_unreachable(conversation: SmsConversation) -> None:
    from apps.ai_sms.services import send_twilio_sms

    phone = (conversation.to_number or "").strip()
    if not phone:
        return
    body = (
        "We're sorry — our team wasn't available just now. "
        "Someone from Bay Area Community Health will follow up with you soon."
    )
    _sid, err = send_twilio_sms(phone, body, conversation.from_number)
    if err:
        logger.warning(
            "Failed to SMS patient after agent unreachable chat=%s err=%s",
            conversation.chat_id,
            err,
        )


def maybe_start_agent_dial_after_reply(
    conversation: SmsConversation, reply: str
) -> None:
    """When SMS tells the patient we're connecting them, dial live agents with failover."""
    if not _is_transfer_reply(reply):
        return
    if conversation.transfer_status in {"dialing", "failover_dialing", "connected"}:
        return
    if not _transfer_list(conversation):
        return
    conversation.transfer_number_index = 0
    if conversation.transfer_numbers:
        conversation.transfer_number = str(conversation.transfer_numbers[0]).strip()
    conversation.save(update_fields=["transfer_number_index", "transfer_number"])
    dial_live_agent(conversation, reason="sms_transfer")


def conversation_by_chat_id(chat_id: str) -> Optional[SmsConversation]:
    return SmsConversation.objects.filter(chat_id=(chat_id or "").strip()).first()
