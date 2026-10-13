"""Place Medi-Cal renewal state-machine calls through the shared Retell client."""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from django.conf import settings

from apps.ai_caller.retell import (
    _AGENT_NAME_GUARDIAN,
    _AGENT_NAME_PATIENT,
    _agent_id_of,
    _api_key,
    _create_agent,
    _denoising_mode,
    _digits,
    _list_agents,
    _llm_id_of,
    _phone_last4,
    _resolve_from_number,
    _retell_request,
    normalize_phone,
)
from apps.ai_pointcare.flow_definition import (
    DEFAULT_AGENT_NAME,
    STARTING_STATE,
    begin_message,
    build_llm_payload,
)
from apps.ai_pointcare.models import RenewalCallSession

logger = logging.getLogger(__name__)

_AGENT_NAME = "BACH Medi-Cal Renewal State Flow"
_CARE_AGENT_NAMES = {_AGENT_NAME_PATIENT, _AGENT_NAME_GUARDIAN}
_AGENT_SETTINGS: Dict[str, Any] = {
    "language": "en-US",
    "responsiveness": 1,
    "interruption_sensitivity": 0.7,
    "reminder_trigger_ms": 8000,
    "reminder_max_count": 1,
}


def function_webhook_url() -> str:
    base = (getattr(settings, "BACKEND_URL", "") or "").rstrip("/")
    if not base:
        return ""
    return f"{base}/api/ai-pointcare/retell/function/"


def _usable_phone(raw: str) -> str:
    phone = normalize_phone(raw)
    if not phone or len(_digits(phone)) < 10:
        return ""
    return phone


def _llm_body(patient_name: str, agent_display_name: str, function_url: str) -> Dict[str, Any]:
    payload = build_llm_payload(
        function_url=function_url,
        patient_name=patient_name,
        agent_display_name=agent_display_name,
    )
    model = (getattr(settings, "RETELL_MODEL", "") or "").strip()
    if model:
        payload["model"] = model
    return payload


def _create_llm(patient_name: str, agent_display_name: str, function_url: str) -> Tuple[str, str]:
    status_code, parsed, err = _retell_request(
        "POST",
        "/create-retell-llm",
        json_body=_llm_body(patient_name, agent_display_name, function_url),
    )
    if status_code >= 400:
        return "", err or "Failed to create Retell LLM."
    data = parsed if isinstance(parsed, dict) else {}
    llm_id = str(data.get("llm_id") or data.get("id") or "").strip()
    if not llm_id:
        return "", "Retell LLM create response missing llm_id."
    return llm_id, ""


def _update_llm(
    llm_id: str, patient_name: str, agent_display_name: str, function_url: str
) -> Tuple[bool, str]:
    status_code, _parsed, err = _retell_request(
        "PATCH",
        f"/update-retell-llm/{llm_id}",
        json_body=_llm_body(patient_name, agent_display_name, function_url),
    )
    if status_code >= 400:
        return False, err or "Failed to update Retell LLM."
    return True, ""


def _agent_name(item: Dict[str, Any]) -> str:
    return str(item.get("agent_name") or item.get("name") or "").strip()


def _call_settings(max_minutes: int) -> Dict[str, Any]:
    return {
        **_AGENT_SETTINGS,
        "denoising_mode": _denoising_mode(),
        "max_call_duration_ms": max(1, max_minutes) * 60 * 1000,
    }


def _ensure_agent(
    patient_name: str,
    agent_display_name: str,
    function_url: str,
    max_duration_minutes: int,
) -> Tuple[str, str]:
    agent_id = ""
    agent: Dict[str, Any] = {}

    dedicated = (getattr(settings, "AI_POINTCARE_RETELL_AGENT_ID", "") or "").strip()
    if dedicated:
        status_code, parsed, err = _retell_request("GET", f"/get-agent/{dedicated}")
        if status_code < 400 and isinstance(parsed, dict) and _agent_id_of(parsed):
            if _agent_name(parsed) in _CARE_AGENT_NAMES:
                return (
                    "",
                    "AI_POINTCARE_RETELL_AGENT_ID points at the care-call agent. "
                    "Use a separate Medi-Cal renewal agent.",
                )
            agent_id = _agent_id_of(parsed)
            agent = parsed
        else:
            return "", err or f"Retell agent {dedicated} was not found."

    if not agent_id:
        for item in _list_agents():
            if _agent_name(item) == _AGENT_NAME:
                agent_id = _agent_id_of(item)
                if agent_id:
                    status_code, parsed, _err = _retell_request(
                        "GET", f"/get-agent/{agent_id}"
                    )
                    if status_code < 400 and isinstance(parsed, dict):
                        agent = parsed
                    break

    settings_body = _call_settings(max_duration_minutes)
    if agent_id:
        llm_id = _llm_id_of(agent)
        if llm_id:
            ok, err = _update_llm(
                llm_id, patient_name, agent_display_name, function_url
            )
            if not ok:
                return "", err
        else:
            llm_id, err = _create_llm(patient_name, agent_display_name, function_url)
            if not llm_id:
                return "", err
            _retell_request(
                "PATCH",
                f"/update-agent/{agent_id}",
                json_body={
                    "response_engine": {"type": "retell-llm", "llm_id": llm_id},
                    **settings_body,
                },
            )
            return agent_id, ""
        _retell_request(
            "PATCH",
            f"/update-agent/{agent_id}",
            json_body=settings_body,
        )
        return agent_id, ""

    llm_id, err = _create_llm(patient_name, agent_display_name, function_url)
    if not llm_id:
        return "", err
    return _create_agent(_AGENT_NAME, llm_id, call_settings=settings_body)


def place_medi_cal_renewal_call(
    *,
    phone_number: str,
    name: str,
    agent_number: str,
    address: str = "",
    agent_display_name: str = DEFAULT_AGENT_NAME,
    outreach=None,
) -> Dict[str, Any]:
    if not _api_key():
        return {
            "ok": False,
            "error": "Retell AI is not configured. Set RETELL_API_KEY.",
            "status_code": 503,
        }

    phone = _usable_phone(phone_number)
    if not phone:
        return {"ok": False, "error": "Invalid phone_number.", "status_code": 400}

    transfer_to = _usable_phone(agent_number)
    if not transfer_to:
        return {
            "ok": False,
            "error": "Invalid agent_number. Provide the live agent phone in E.164.",
            "status_code": 400,
        }

    fn_url = function_webhook_url()
    if not fn_url:
        return {
            "ok": False,
            "error": "BACKEND_URL is not configured; required for Retell function webhooks.",
            "status_code": 503,
        }

    who = (name or "").strip() or "there"
    first = who.split()[0] if who.split() else who
    agent = (agent_display_name or "").strip() or DEFAULT_AGENT_NAME
    callback = (
        (getattr(settings, "MEDICAL_RENEWAL_TOLL_FREE_NUMBER", "") or "").strip()
        or (getattr(settings, "TWILIO_PHONE_NUMBER", "") or "").strip()
    )
    addr = (address or "").strip() or "the address we have on file"

    max_minutes = int(getattr(settings, "RETELL_MAX_DURATION_MINUTES", 8) or 8)
    max_minutes = max(max_minutes, 20)

    agent_id, err = _ensure_agent(who, agent, fn_url, max_minutes)
    if not agent_id:
        return {"ok": False, "error": err, "status_code": 502}

    from_number, from_err = _resolve_from_number()
    if not from_number:
        return {"ok": False, "error": from_err, "status_code": 503}

    payload: Dict[str, Any] = {
        "from_number": from_number,
        "to_number": phone,
        "override_agent_id": agent_id,
        "retell_llm_dynamic_variables": {
            "patient_name": who,
            "first_name": first,
            "name": who,
            "agent_name": agent,
            "address": addr,
            "callback_number": callback or "our clinic number",
            "agent_number": transfer_to,
        },
        "metadata": {
            "source": "ai_pointcare_medi_cal_flow",
            "patient_name": who,
            "outreach_id": getattr(outreach, "id", None),
        },
    }
    status_code, parsed, place_err = _retell_request(
        "POST", "/v2/create-phone-call", json_body=payload
    )
    if status_code >= 400:
        return {
            "ok": False,
            "error": place_err or "Failed to place Retell call.",
            "status_code": status_code if status_code < 500 else 502,
        }

    data = parsed if isinstance(parsed, dict) else {}
    call_id = str(data.get("call_id") or data.get("id") or "").strip()
    if not call_id:
        return {
            "ok": False,
            "error": "Retell response missing call_id.",
            "status_code": 502,
        }

    RenewalCallSession.objects.update_or_create(
        retell_call_id=call_id,
        defaults={
            "outreach": outreach,
            "patient_name": who,
            "phone_number": phone,
            "agent_number": transfer_to,
            "address": "" if addr == "the address we have on file" else addr,
            "current_node": STARTING_STATE,
        },
    )

    logger.info(
        "ai_pointcare Medi-Cal flow call placed call_id=%s to=%s transfer=%s",
        call_id,
        _phone_last4(phone),
        _phone_last4(transfer_to),
    )
    return {
        "ok": True,
        "call_id": call_id,
        "status": data.get("call_status") or data.get("status") or "registered",
        "agent_id": agent_id,
        "from_number": from_number,
        "phone_number": phone,
        "phone_last4": _phone_last4(phone),
        "agent_number": transfer_to,
        "starting_state": STARTING_STATE,
        "begin_message": begin_message(who),
        "message": "Medi-Cal renewal state-machine call placed through Retell AI.",
    }
