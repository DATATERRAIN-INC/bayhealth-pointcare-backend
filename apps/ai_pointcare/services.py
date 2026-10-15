"""Place a Medi-Cal renewal call for a phone number that is not already opted out."""
from __future__ import annotations

import logging

from apps.ai_caller.retell import _digits, _phone_last4, normalize_phone
from apps.ai_pointcare.models import CHOICE_CALL_NOW, OptOut, Outreach
from apps.ai_pointcare.retell_client import place_medi_cal_renewal_call

logger = logging.getLogger(__name__)


def _formatted_phone(phone_number: str) -> str:
    formatted = normalize_phone(phone_number)
    if not formatted or len(_digits(formatted)) < 10:
        return ""
    return formatted


def start_medi_cal_call(phone_number, name, agent_number, address=""):
    """Place the Medi-Cal renewal Retell state-machine call."""
    formatted = _formatted_phone(phone_number)
    if not formatted:
        return {
            "ok": False,
            "error": "Invalid phone_number. Provide a 10-digit US number or E.164 value.",
            "status_code": 400,
        }

    display_name = (name or "").strip()
    if OptOut.objects.filter(phone_number=formatted).exists():
        return {
            "ok": False,
            "error": "This phone number has opted out of SMS/calls.",
            "status_code": 400,
            "phone_number": formatted,
            "phone_last4": _phone_last4(formatted),
        }

    outreach = Outreach.objects.create(
        name=display_name,
        phone_number=formatted,
        phone_raw=str(phone_number or "").strip(),
        address=(address or "").strip(),
        agent_number=(agent_number or "").strip(),
        status=Outreach.STATUS_PENDING,
        choice=CHOICE_CALL_NOW,
    )
    result = place_medi_cal_renewal_call(
        phone_number=formatted,
        name=display_name,
        agent_number=agent_number or "",
        address=address or "",
        outreach=outreach,
    )
    if result.get("ok"):
        outreach.status = Outreach.STATUS_CALLING
        outreach.agent_number = result.get("agent_number") or (agent_number or "").strip()
        outreach.address = (address or outreach.address or "").strip()
        outreach.retell_call_id = result.get("call_id") or ""
        outreach.retell_agent_id = result.get("agent_id") or ""
        outreach.call_from_number = result.get("from_number") or ""
        outreach.error = ""
        outreach.save(
            update_fields=[
                "status",
                "agent_number",
                "address",
                "retell_call_id",
                "retell_agent_id",
                "call_from_number",
                "error",
                "updated_at",
            ]
        )
    else:
        outreach.status = Outreach.STATUS_CALL_FAILED
        outreach.error = result.get("error") or "Failed to place call."
        outreach.save(update_fields=["status", "error", "updated_at"])
        logger.error(
            "ai_pointcare call failed outreach_id=%s err=%s",
            outreach.id,
            outreach.error,
        )

    result["outreach_id"] = outreach.id
    result["name"] = display_name
    result["phone_number"] = formatted
    result["phone_last4"] = _phone_last4(formatted)
    return result
