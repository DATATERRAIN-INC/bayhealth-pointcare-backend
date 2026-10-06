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

from apps.ai_caller.prompt import (
    BEGIN_MESSAGE,
    GUARDIAN_BEGIN_MESSAGE,
    guardian_prompt,
    patient_prompt,
)

logger = logging.getLogger(__name__)

_RETELL_BASE = "https://api.retellai.com"
_REGION_US = "us"
_REGION_IN = "in"
_AGENT_NAME_PATIENT = "GridSocial Care Call Quality Test"
_LLM_NAME_PATIENT = "GridSocial Care Call LLM"
_AGENT_NAME_GUARDIAN = "GridSocial Care Call Guardian"
_LLM_NAME_GUARDIAN = "GridSocial Care Call Guardian LLM"
_PROVIDER_NAME = "Gaurav"


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



def _denoising_mode() -> str:
    """Retell caller-side denoising. Override with RETELL_DENOISING_MODE.

    NOTE: With call screening enabled, do NOT use
    noise-and-background-speech-cancellation — Retell can lock onto the
    screening voice and filter out the real caller. Use noise-cancellation.
    """
    allowed = {
        "no-denoise",
        "noise-cancellation",
        "noise-and-background-speech-cancellation",
    }
    configured = (getattr(settings, "RETELL_DENOISING_MODE", "") or "").strip()
    if configured in allowed:
        return configured
    return "noise-cancellation"


def _interruption_sensitivity() -> float:
    """Lower = less likely to treat noise as an interruption. Default 0.6."""
    raw = getattr(settings, "RETELL_INTERRUPTION_SENSITIVITY", None)
    try:
        value = float(raw if raw is not None and str(raw).strip() != "" else 0.6)
    except (TypeError, ValueError):
        value = 0.6
    return max(0.0, min(1.0, value))


def _responsiveness() -> float:
    """Lower = wait a bit longer before answering (helps noisy lines). Default 0.85."""
    raw = getattr(settings, "RETELL_RESPONSIVENESS", None)
    try:
        value = float(raw if raw is not None and str(raw).strip() != "" else 0.85)
    except (TypeError, ValueError):
        value = 0.85
    return max(0.0, min(1.0, value))


def _agent_call_settings() -> Dict[str, Any]:
    return {
        "language": "multi",
        "begin_message_delay_ms": 1000,
        "denoising_mode": _denoising_mode(),
        "interruption_sensitivity": _interruption_sensitivity(),
        "responsiveness": _responsiveness(),
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
                    "for {{patient_name}}. Just reaching out about your "
                    "{{service_name}} with Dr. {{name}} — nothing urgent. "
                    "Whenever you have a moment, please give us a call back "
                    "and we can help with scheduling or any questions. "
                    "Thanks so much, take care."
                ),
            },
        },
    }


def _tool_webhook_url() -> str:
    base = (getattr(settings, "BACKEND_URL", "") or "").rstrip("/")
    if not base:
        return ""
    return f"{base}/api/ai-call/webhooks/retell-tool/"


def _end_call_tool() -> Dict[str, Any]:
    return {
        "type": "end_call",
        "name": "end_call",
        "description": (
            "End the call after goodbye was said, or after the caller declined to talk."
        ),
    }


def _decline_reason_tool() -> Dict[str, Any]:
    tool: Dict[str, Any] = {
        "type": "custom",
        "name": "log_decline_reason",
        "description": (
            "Required whenever the caller says they are not interested in the call, "
            "screening, or scheduling. Pass their reason in their own words before ending."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "Why the caller is not interested, in their own words. "
                        "Use a short paraphrase only if they refuse to give detail."
                    ),
                }
            },
            "required": ["reason"],
        },
        "speak_during_execution": False,
        "speak_after_execution": False,
    }
    url = _tool_webhook_url()
    if url:
        tool["url"] = url
        tool["method"] = "POST"
    return tool


def _callback_request_tool() -> Dict[str, Any]:
    tool: Dict[str, Any] = {
        "type": "custom",
        "name": "log_callback_request",
        "description": (
            "Required when the caller asks to be called back later and gives a time. "
            "Pass the callback time in their own words (for example: 'tomorrow at 3pm', "
            "'today at 4:30 PM', or 'in 2 hours') before ending the call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "callback_time": {
                    "type": "string",
                    "description": (
                        "When they want to be called back, in their own words. "
                        "Prefer a clear time such as 'tomorrow at 3pm'."
                    ),
                }
            },
            "required": ["callback_time"],
        },
        "speak_during_execution": False,
        "speak_after_execution": False,
    }
    url = _tool_webhook_url()
    if url:
        tool["url"] = url
        tool["method"] = "POST"
    return tool


def _default_care_tools() -> List[Dict[str, Any]]:
    return [
        _end_call_tool(),
        _transfer_tool(),
        _decline_reason_tool(),
        _callback_request_tool(),
    ]


def _default_guardian_tools() -> List[Dict[str, Any]]:
    return [
        _end_call_tool(),
        _minor_transfer_tool(),
        _decline_reason_tool(),
        _callback_request_tool(),
    ]


def _transfer_tool() -> Dict[str, Any]:
    return {
        "type": "transfer_call",
        "name": "transfer_to_live_agent",
        "description": (
            "Transfer to a live Bay Area Community Health team member ONLY when the "
            "patient has said yes to booking or rescheduling an appointment, when "
            "they want a live agent to update address, phone, email, or insurance, "
            "or when they declined to talk to an AI and then agreed to be connected."
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
        "time does not work, when they want a live agent to update address, phone, "
        "email, or insurance, or when they declined to talk to an AI and then agreed "
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
        "begin_message": BEGIN_MESSAGE if begin_message is None else begin_message,
        "start_speaker": "agent",
        "general_tools": tools if tools is not None else _default_care_tools(),
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
        "begin_message": BEGIN_MESSAGE if begin_message is None else begin_message,
        "start_speaker": "agent",
        "general_tools": tools if tools is not None else _default_care_tools(),
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
    prompt = patient_prompt()
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
    email_on_file: str = "",
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
            "phone_last4": _phone_last4(phone),
            "email_on_file": (email_on_file or "").strip(),
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
    prompt = guardian_prompt()
    tools = _default_guardian_tools()
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
                begin_message=GUARDIAN_BEGIN_MESSAGE,
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
        begin_message=GUARDIAN_BEGIN_MESSAGE,
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
    email_on_file: str = "",
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
            "phone_last4": _phone_last4(phone),
            "email_on_file": (email_on_file or "").strip(),
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


def get_retell_call(call_id: str) -> Dict[str, Any]:
    if not call_id:
        return {"ok": False, "error": "call_id is required.", "status_code": 400}

    status_code, parsed, err = _retell_request("GET", f"/v2/get-call/{call_id}")
    if status_code >= 400:
        return {
            "ok": False,
            "error": err or "Failed to fetch Retell call.",
            "status_code": status_code if status_code != 502 else 502,
        }

    data = parsed if isinstance(parsed, dict) else {}
    return {"ok": True, "data": data}
