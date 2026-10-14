from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote

import httpx

from . import config
from .form_filler_script import pointcare_agent_settings

logger = logging.getLogger(__name__)

_RETELL_BASE = "https://api.retellai.com"
# Dedicated agent for the logged-in household-eligibility voice flow.
_AGENT_NAME_ELIGIBILITY = "Pointcare Eligibility"
# Retell cannot change response_engine after versions exist; Custom LLM uses a
# separate agent created with type custom-llm from the start.
_AGENT_NAME_CUSTOM_LLM = "Pointcare Eligibility Collect"
_LLM_NAME_ELIGIBILITY = "Pointcare Eligibility LLM"


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def normalize_phone(raw: str) -> str:
    value = (raw or "").strip()
    digits = _digits(value)
    if not digits:
        return ""
    if value.startswith("+"):
        return f"+{digits}"
    if digits.startswith("91") and len(digits) >= 12:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+91{digits}"
    if digits.startswith("1") and len(digits) == 11:
        return f"+{digits}"
    return f"+{digits}"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.RETELL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request(method: str, path: str, json_body: dict | None = None, params: dict | None = None) -> tuple[int, Any, str]:
    try:
        resp = httpx.request(
            method,
            f"{_RETELL_BASE}{path}",
            headers=_headers(),
            json=json_body,
            params=params,
            timeout=25,
        )
    except Exception as exc:
        return 502, None, str(exc)
    parsed: Any = None
    err = ""
    try:
        parsed = resp.json()
    except Exception:
        parsed = {"raw": (resp.text or "")[:800]}
    if resp.status_code >= 400:
        if isinstance(parsed, dict):
            err = str(parsed.get("message") or parsed.get("error") or parsed)[:800]
        else:
            err = str(parsed)[:800]
    return resp.status_code, parsed, err


def _as_list(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("items", "data", "phone_numbers", "agents", "llms"):
            inner = parsed.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
    return []


def _phone_of(item: dict[str, Any]) -> str:
    return str(item.get("phone_number") or item.get("number") or item.get("from_number") or "").strip()


def _agent_id_of(item: dict[str, Any]) -> str:
    return str(item.get("agent_id") or item.get("id") or "").strip()


def _llm_id_of(item: dict[str, Any]) -> str:
    engine = item.get("response_engine")
    if isinstance(engine, dict):
        return str(engine.get("llm_id") or "").strip()
    return str(item.get("llm_id") or "").strip()


def _end_call_tool() -> dict[str, Any]:
    return {
        "type": "end_call",
        "name": "end_call",
        "description": (
            "End the call after the email sign-in Continue step, after goodbye, "
            "or when the caller declines."
        ),
    }


def _tool_webhook_url() -> str:
    from .form_fields import tool_webhook_url

    return tool_webhook_url(config.FORM_FILLER_WEBHOOK_URL)


def _llm_tools() -> list[dict[str, Any]]:
    from .form_fields import submit_field_tool_spec

    tools: list[dict[str, Any]] = [_end_call_tool()]
    tool_url = _tool_webhook_url()
    if tool_url:
        tools.append(submit_field_tool_spec(tool_url))
    return tools


def _resolve_from_number() -> tuple[str, str]:
    if config.RETELL_FROM_NUMBER:
        return config.RETELL_FROM_NUMBER, ""
    for path in ("/v2/list-phone-numbers", "/list-phone-numbers"):
        status_code, parsed, _err = _request("GET", path, params={"limit": 50})
        if status_code < 400:
            owned = [_phone_of(item) for item in _as_list(parsed) if _phone_of(item)]
            if owned:
                return owned[0], ""
    return "", "No Retell from-number. Set RETELL_FROM_NUMBER or buy a number in Retell."


def _version_of(item: dict[str, Any]) -> int | None:
    version = item.get("version")
    if isinstance(version, bool):
        return None
    if isinstance(version, int):
        return version
    if isinstance(version, str) and version.isdigit():
        return int(version)
    return None


def _published_error(err: str) -> bool:
    return "published" in (err or "").lower()


def _llm_payload() -> dict[str, Any]:
    from .live_fill_prompt import LIVE_FILL_APPENDIX, LIVE_FILL_GREETING

    payload: dict[str, Any] = {
        "general_prompt": LIVE_FILL_APPENDIX,
        "begin_message": LIVE_FILL_GREETING,
        "start_speaker": "agent",
        "general_tools": _llm_tools(),
    }
    if config.RETELL_MODEL:
        payload["model"] = config.RETELL_MODEL
    return payload


def _response_engine() -> dict[str, Any]:
    """Primary Collect Mode uses Custom LLM WebSocket when enabled."""
    if config.USE_CUSTOM_LLM:
        ws_url = config.retell_llm_websocket_url()
        if ws_url:
            return {
                "type": "custom-llm",
                "llm_websocket_url": ws_url,
            }
        logger.warning(
            "USE_CUSTOM_LLM is on but no WebSocket URL; falling back to retell-llm. "
            "Set FORM_FILLER_WEBHOOK_URL or RETELL_LLM_WEBSOCKET_URL."
        )
    return {}


def _attach_response_engine(agent_id: str, version: int | None) -> str:
    engine = _response_engine()
    if not engine:
        return ""
    status_code, _parsed, err = _request(
        "PATCH",
        f"/update-agent/{agent_id}",
        json_body={"response_engine": engine},
        params={"version": version} if version is not None else None,
    )
    if status_code >= 400:
        return err or "Failed to set Custom LLM response engine."
    logger.info(
        "Retell agent %s response_engine=%s",
        agent_id,
        engine.get("type"),
    )
    return ""


def _engine_type_of(agent: dict[str, Any]) -> str:
    engine = agent.get("response_engine")
    if isinstance(engine, dict):
        return str(engine.get("type") or "").strip()
    return ""


def _find_agent_by_name(name: str) -> tuple[str, dict[str, Any]]:
    status_code, parsed, _err = _request("GET", "/list-agents", params={"limit": 50})
    if status_code >= 400:
        status_code, parsed, _err = _request("GET", "/v2/list-agents", params={"limit": 50})
    for item in _as_list(parsed):
        if str(item.get("agent_name") or item.get("name") or "").strip() == name:
            agent_id = _agent_id_of(item)
            if not agent_id:
                continue
            _sc, full, _e = _request("GET", f"/get-agent/{agent_id}")
            if _sc < 400 and isinstance(full, dict):
                return agent_id, full
            return agent_id, item
    return "", {}


def _create_custom_llm_agent(settings_body: dict[str, Any]) -> tuple[str, str, int | None]:
    custom_engine = _response_engine()
    if not custom_engine:
        return "", "Custom LLM WebSocket URL is not configured.", None
    agent_payload = {
        "agent_name": _AGENT_NAME_CUSTOM_LLM,
        "response_engine": custom_engine,
        "voice_id": config.RETELL_VOICE,
        "data_storage_setting": "everything",
        **settings_body,
        "language": "en-US",
    }
    status_code, parsed, err = _request("POST", "/create-agent", json_body=agent_payload)
    if status_code >= 400 or not isinstance(parsed, dict):
        return "", err or "Failed to create Retell Custom LLM agent.", None
    agent_id = _agent_id_of(parsed)
    if not agent_id:
        return "", "Retell agent create response missing agent_id.", None
    version = _version_of(parsed)
    publish_err = _publish_agent(agent_id, version)
    if publish_err:
        return "", publish_err, None
    logger.info(
        "Created Custom LLM agent id=%s version=%s url=%s",
        agent_id,
        version,
        custom_engine.get("llm_websocket_url"),
    )
    return agent_id, "", version


def _create_llm() -> tuple[str, str]:
    payload = {"llm_name": _LLM_NAME_ELIGIBILITY, **_llm_payload()}
    status_code, parsed, err = _request("POST", "/create-retell-llm", json_body=payload)
    if status_code >= 400 or not isinstance(parsed, dict):
        # Older Retell accounts may reject llm_name; retry without it.
        status_code, parsed, err = _request("POST", "/create-retell-llm", json_body=_llm_payload())
    if status_code >= 400 or not isinstance(parsed, dict):
        return "", err or "Failed to create Retell LLM."
    llm_id = str(parsed.get("llm_id") or parsed.get("id") or "").strip()
    if not llm_id:
        return "", "Retell LLM create response missing llm_id."
    logger.info("Retell LLM created name=%s llm_id=%s", _LLM_NAME_ELIGIBILITY, llm_id)
    return llm_id, ""


def _publish_agent(agent_id: str, version: int | None) -> str:
    if version is None:
        return ""
    status_code, _parsed, err = _request(
        "POST",
        f"/publish-agent-version/{agent_id}",
        json_body={"version": version},
    )
    if status_code >= 400:
        status_code, _parsed, fallback_err = _request(
            "POST",
            "/publish-agent",
            json_body={"agent_id": agent_id, "version": version},
        )
        if status_code >= 400:
            return fallback_err or err or "Failed to publish Retell agent version."
    return ""


def _open_draft(agent_id: str, agent: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Published Retell versions cannot be edited. Copy one into a draft."""
    if not agent.get("is_published"):
        return agent, ""
    base = _version_of(agent)
    if base is None:
        return {}, "Retell agent is published and has no version to copy."
    status_code, parsed, err = _request(
        "POST",
        f"/create-agent-version/{agent_id}",
        json_body={"base_version": base},
    )
    if status_code >= 400 or not isinstance(parsed, dict):
        return {}, err or "Failed to create a Retell draft."
    return parsed, ""


def _ensure_agent(configured_id: str) -> tuple[str, str, int | None]:
    settings_body = dict(pointcare_agent_settings())
    # Eligibility flow is English-only; ignore any multi-language script defaults.
    settings_body["language"] = "en-US"
    webhook = config.FORM_FILLER_WEBHOOK_URL
    if webhook:
        settings_body["webhook_url"] = webhook
        settings_body.setdefault(
            "webhook_events",
            ["call_started", "transcript_updated", "call_ended", "call_analyzed"],
        )

    # Prefer Custom LLM agent when enabled (must be created with that engine).
    if config.USE_CUSTOM_LLM and _response_engine():
        preferred_names = (_AGENT_NAME_CUSTOM_LLM, _AGENT_NAME_ELIGIBILITY)
        agent_id = ""
        agent: dict[str, Any] = {}
        if configured_id:
            status_code, parsed, err = _request("GET", f"/get-agent/{configured_id}")
            if status_code < 400 and isinstance(parsed, dict) and _agent_id_of(parsed):
                if _engine_type_of(parsed) == "custom-llm":
                    agent_id = _agent_id_of(parsed)
                    agent = parsed
                else:
                    logger.warning(
                        "Configured agent %s is %s; looking for Custom LLM agent %s",
                        configured_id,
                        _engine_type_of(parsed) or "unknown",
                        _AGENT_NAME_CUSTOM_LLM,
                    )
        if not agent_id:
            for name in preferred_names:
                found_id, found = _find_agent_by_name(name)
                if found_id and _engine_type_of(found) == "custom-llm":
                    agent_id, agent = found_id, found
                    break
        if not agent_id:
            return _create_custom_llm_agent(settings_body)

        agent, draft_err = _open_draft(agent_id, agent)
        if draft_err:
            return "", draft_err, None
        version = _version_of(agent)
        # Keep websocket URL current (allowed as settings-adjacent update when already custom-llm).
        engine_err = _attach_response_engine(agent_id, version)
        if engine_err and "Cannot update response engine" not in engine_err:
            # Retry URL-only style: some accounts reject type change but allow same-type URL patch.
            logger.warning("Custom LLM engine patch note: %s", engine_err[:200])
        status_code, updated, err = _request(
            "PATCH",
            f"/update-agent/{agent_id}",
            json_body=settings_body,
            params={"version": version} if version is not None else None,
        )
        if status_code >= 400:
            return "", err or "Failed to update the Custom LLM agent.", None
        if isinstance(updated, dict):
            version = _version_of(updated) or version
        publish_err = _publish_agent(agent_id, version)
        if publish_err:
            return "", publish_err, None
        return agent_id, "", version

    agent_id = ""
    agent: dict[str, Any] = {}

    if configured_id:
        status_code, parsed, err = _request("GET", f"/get-agent/{configured_id}")
        if status_code >= 400 or not isinstance(parsed, dict) or not _agent_id_of(parsed):
            return "", err or f"Retell agent {configured_id} was not found.", None
        named = str(parsed.get("agent_name") or parsed.get("name") or "").strip()
        if named and named not in (_AGENT_NAME_ELIGIBILITY, _AGENT_NAME_CUSTOM_LLM):
            logger.warning(
                "Ignoring outdated Retell agent %s (%s); using %s instead",
                configured_id,
                named,
                _AGENT_NAME_ELIGIBILITY,
            )
            configured_id = ""
        else:
            agent_id = _agent_id_of(parsed)
            agent = parsed
    if not configured_id and not agent_id:
        agent_id, agent = _find_agent_by_name(_AGENT_NAME_ELIGIBILITY)

    if agent_id:
        agent, draft_err = _open_draft(agent_id, agent)
        if draft_err:
            return "", draft_err, None
        version = _version_of(agent)
        llm_id = _llm_id_of(agent)
        if llm_id:
            status_code, _parsed, err = _request(
                "PATCH",
                f"/update-retell-llm/{llm_id}",
                json_body=_llm_payload(),
                params={"version": version} if version is not None else None,
            )
            if status_code >= 400 and _published_error(err):
                llm_id, create_err = _create_llm()
                if not llm_id:
                    return "", create_err, None
                engine = {"response_engine": {"type": "retell-llm", "llm_id": llm_id}}
                status_code, _parsed, err = _request(
                    "PATCH",
                    f"/update-agent/{agent_id}",
                    json_body=engine,
                    params={"version": version} if version is not None else None,
                )
                if status_code >= 400:
                    return "", err or "Failed to attach a draft Retell LLM.", None
            elif status_code >= 400:
                return "", err or "Failed to update the Retell form-filler agent.", None
        status_code, updated, err = _request(
            "PATCH",
            f"/update-agent/{agent_id}",
            json_body=settings_body,
            params={"version": version} if version is not None else None,
        )
        if status_code >= 400:
            return "", err or "Failed to update the Retell form-filler agent.", None
        if isinstance(updated, dict):
            version = _version_of(updated) or version
        publish_err = _publish_agent(agent_id, version)
        if publish_err:
            return "", publish_err, None
        return agent_id, "", version

    llm_id, create_err = _create_llm()
    if not llm_id:
        return "", create_err, None
    agent_payload = {
        "agent_name": _AGENT_NAME_ELIGIBILITY,
        "response_engine": {"type": "retell-llm", "llm_id": llm_id},
        "voice_id": config.RETELL_VOICE,
        "data_storage_setting": "everything",
        **settings_body,
        "language": "en-US",
    }
    status_code, parsed, err = _request("POST", "/create-agent", json_body=agent_payload)
    if status_code >= 400 or not isinstance(parsed, dict):
        return "", err or "Failed to create Retell agent.", None
    agent_id = _agent_id_of(parsed)
    if not agent_id:
        return "", "Retell agent create response missing agent_id.", None
    version = _version_of(parsed)
    publish_err = _publish_agent(agent_id, version)
    if publish_err:
        return "", publish_err, None
    logger.info(
        "Retell agent created name=%s agent_id=%s version=%s",
        _AGENT_NAME_ELIGIBILITY,
        agent_id,
        version,
    )
    return agent_id, "", version


def place_form_filler_call(
    *,
    phone_number: str,
    name: str = "",
    form_name: str = "",
    agent_id: str = "",
) -> dict[str, Any]:
    if not config.RETELL_API_KEY:
        return {"ok": False, "error": "RETELL_API_KEY is not set in .env", "status_code": 503}

    phone = normalize_phone(phone_number)
    if not phone or len(_digits(phone)) < 10:
        return {"ok": False, "error": "Invalid phone_number.", "status_code": 400}

    from_number, from_err = _resolve_from_number()
    if not from_number:
        return {"ok": False, "error": from_err, "status_code": 503}

    if _digits(phone).startswith("91"):
        encoded = quote(from_number, safe="")
        payload = {"allowed_outbound_country_list": ["US", "CA", "IN"]}
        for path in (f"/update-phone-number/{from_number}", f"/update-phone-number/{encoded}"):
            status_code, _parsed, _err = _request("PATCH", path, json_body=payload)
            if status_code >= 400:
                _request("POST", path, json_body=payload)

    resolved_agent, agent_err, agent_version = _ensure_agent((agent_id or config.FORM_FILLER_AGENT_ID).strip())
    if not resolved_agent:
        return {"ok": False, "error": agent_err, "status_code": 502}

    who = (name or "").strip() or "there"
    form = (form_name or "").strip() or "Pointcare account"
    webhook = config.FORM_FILLER_WEBHOOK_URL
    body: dict[str, Any] = {
        "from_number": from_number,
        "to_number": phone,
        "override_agent_id": resolved_agent,
        "retell_llm_dynamic_variables": {
            "name": who,
            "first_name": who,
            "patient_name": who,
            "form_name": form,
            "phone": phone,
            "partner_link": config.POINTCARE_URL,
            "callback_number": "1-833-234-7833",
            "pointcare_url": config.POINTCARE_URL,
        },
        "metadata": {
            "source": "form_filler_retell",
            "flow": "pointcare_live_fill",
            "form_name": form,
            "pointcare_url": config.POINTCARE_URL,
        },
        "webhook_events": ["call_started", "transcript_updated", "call_ended", "call_analyzed"],
    }
    if agent_version is not None:
        body["override_agent_version"] = agent_version
    if webhook:
        body["webhook_url"] = webhook

    status_code, parsed, err = _request("POST", "/v2/create-phone-call", json_body=body)
    if status_code >= 400 or not isinstance(parsed, dict):
        return {"ok": False, "error": err or "Failed to place Retell call.", "status_code": status_code or 502}

    call_id = str(parsed.get("call_id") or parsed.get("id") or "").strip()
    if not call_id:
        return {"ok": False, "error": "Retell response missing call_id.", "status_code": 502}

    logger.info("Form filler call placed call_id=%s to=%s", call_id, phone[-4:])
    return {
        "ok": True,
        "call_id": call_id,
        "status": parsed.get("call_status") or parsed.get("status") or "registered",
        "from_number": from_number,
        "agent_id": resolved_agent,
        "webhook_url": webhook,
        "tool_webhook_url": _tool_webhook_url(),
        "flow": "pointcare_live_fill",
        "form_name": form,
        "pointcare_url": config.POINTCARE_URL,
        "message": (
            "Call placed. Chrome opens Pointcare onboarding on a fresh sign-in page, "
            "then Kyle asks Google or email, signs in, opens household eligibility, "
            "and submit_field fills each answered question once."
        ),
    }


def fetch_retell_call(call_id: str) -> dict[str, Any]:
    if not call_id:
        return {}
    for path in (f"/v2/get-call/{call_id}", f"/get-call/{call_id}"):
        status_code, parsed, _err = _request("GET", path)
        if status_code < 400 and isinstance(parsed, dict):
            return parsed
    return {}


def nudge_agent_speak(call_id: str, question: str) -> bool:
    """Ask the live Retell agent to speak the next eligibility question.

    Used when the transcript fallback already clicked the answer in Playwright,
    so Retell never received a tool result containing the next question text.
    """
    text = (question or "").strip()
    if not call_id or not text:
        return False
    payload = {
        "call_control": {
            # Keep this minimal — extra meta text makes Retell invent filler instead
            # of asking the real next eligibility question.
            "additional_context": (
                f'Speak exactly this once, then wait silently: "{text}"'
            ),
            "trigger_response": True,
        }
    }
    for path in (f"/v2/update-live-call/{call_id}", f"/update-live-call/{call_id}"):
        status_code, _parsed, err = _request("PATCH", path, json_body=payload)
        if status_code >= 400:
            status_code, _parsed, err = _request("POST", path, json_body=payload)
        if status_code < 400:
            logger.info("Nudged Retell to speak next question call_id=%s", call_id)
            return True
        logger.warning(
            "Retell update-live-call failed call_id=%s path=%s status=%s err=%s",
            call_id,
            path,
            status_code,
            (err or "")[:200],
        )
    return False
