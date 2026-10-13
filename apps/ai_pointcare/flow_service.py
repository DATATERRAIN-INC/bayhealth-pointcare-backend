"""Backend handlers for Retell custom functions in the Medi-Cal flow."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from apps.ai_caller.retell import normalize_phone
from apps.ai_pointcare.models import OptOut, Outreach, RenewalCallSession
from apps.ai_pointcare.renewal_video import send_renewal_video_sms

logger = logging.getLogger(__name__)


def _as_bool(value) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    return None


def get_session_for_call(call_payload: Dict[str, Any]) -> Optional[RenewalCallSession]:
    call_id = str(
        (call_payload or {}).get("call_id")
        or (call_payload or {}).get("id")
        or ""
    ).strip()
    if not call_id:
        return None
    return RenewalCallSession.objects.filter(retell_call_id=call_id).first()


def apply_progress(session: RenewalCallSession, args: Dict[str, Any]) -> Dict[str, Any]:
    args = args or {}
    updates = []

    mapping = {
        "patient_verified": ("patient_verified", _as_bool),
        "address": ("address", str),
        "email": ("email", str),
        "renewal_status": ("renewal_status", str),
        "household_changed": ("household_changed", _as_bool),
        "household_change_details": ("household_change_details", str),
        "income_changed": ("income_changed", _as_bool),
        "income_change_details": ("income_change_details", str),
        "support_status": ("support_status", str),
        "document_ready": ("document_ready", _as_bool),
        "assistance_type": ("assistance_type", str),
        "additional_benefits_interest": ("additional_benefits_interest", _as_bool),
        "appointment_needed": ("appointment_needed", _as_bool),
        "appointment_details": ("appointment_details", str),
        "final_outcome": ("final_outcome", str),
        "callback_details": ("callback_details", str),
        "current_node": ("current_node", str),
    }

    for arg_key, (field, caster) in mapping.items():
        if arg_key not in args or args.get(arg_key) in (None, ""):
            continue
        value = caster(args.get(arg_key))
        if value is None and caster is _as_bool:
            continue
        setattr(session, field, value if caster is not str else str(value).strip())
        updates.append(field)

    if "address" in args and str(args.get("address") or "").strip():
        session.address_confirmed = False
        updates.append("address_confirmed")
    if args.get("address_confirmed") is not None:
        session.address_confirmed = _as_bool(args.get("address_confirmed"))
        updates.append("address_confirmed")
    if args.get("phone_confirmed") is not None:
        session.phone_confirmed = _as_bool(args.get("phone_confirmed"))
        updates.append("phone_confirmed")

    if "phone" in args and str(args.get("phone") or "").strip():
        formatted = normalize_phone(args.get("phone"))
        if formatted:
            session.phone_number = formatted
            session.phone_confirmed = False
            updates.extend(["phone_number", "phone_confirmed"])

    if "preferred_language" in args and str(args.get("preferred_language") or "").strip():
        payload = dict(session.state_payload or {})
        payload["preferred_language"] = str(args.get("preferred_language")).strip()
        session.state_payload = payload
        updates.append("state_payload")

    if "assistance_type" in args:
        assistance = str(args.get("assistance_type") or "").strip()
        if assistance and assistance != "declined":
            session.assistance_requested = True
            updates.append("assistance_requested")
            if assistance in (
                "enrollment_team",
                "phone_help",
                "live_agent",
            ):
                session.transfer_required = True
                updates.append("transfer_required")
        elif assistance == "declined":
            session.assistance_requested = False
            updates.append("assistance_requested")

    if _as_bool(args.get("callback_requested")) is True:
        session.callback_requested = True
        updates.append("callback_requested")
    if _as_bool(args.get("dnc_requested")) is True:
        session.dnc_requested = True
        updates.append("dnc_requested")

    note = str(args.get("note") or "").strip()
    if note:
        payload = dict(session.state_payload or {})
        notes = list(payload.get("notes") or [])
        notes.append(note)
        payload["notes"] = notes[-50:]
        session.state_payload = payload
        updates.append("state_payload")

    if updates:
        session.save(update_fields=list(dict.fromkeys(updates + ["updated_at"])))

    if session.outreach_id and session.final_outcome:
        Outreach.objects.filter(pk=session.outreach_id).update(
            status=Outreach.STATUS_COMPLETED
        )

    return {
        "ok": True,
        "saved": True,
        "current_node": session.current_node,
        "final_outcome": session.final_outcome or "",
        "message": "Progress saved.",
    }


def handle_retell_function(name: str, args: Dict[str, Any], call: Dict[str, Any]) -> Dict[str, Any]:
    session = get_session_for_call(call or {})
    if session is None:
        call_id = str((call or {}).get("call_id") or "").strip()
        if not call_id:
            return {"ok": False, "saved": False, "error": "Unknown call_id."}
        meta = (call or {}).get("metadata") or {}
        dyn = (call or {}).get("retell_llm_dynamic_variables") or {}
        session = RenewalCallSession.objects.create(
            retell_call_id=call_id,
            patient_name=str(dyn.get("patient_name") or meta.get("patient_name") or ""),
            phone_number=str((call or {}).get("to_number") or ""),
            agent_number=str(dyn.get("agent_number") or ""),
            address=str(dyn.get("address") or ""),
        )

    fname = (name or "").strip()
    args = args or {}

    if fname == "save_progress":
        return apply_progress(session, args)

    if fname == "request_callback":
        session.callback_requested = True
        session.callback_details = str(args.get("callback_details") or "").strip()
        if not session.final_outcome:
            session.final_outcome = "FOLLOW_UP_REQUIRED"
        session.save(
            update_fields=[
                "callback_requested",
                "callback_details",
                "final_outcome",
                "updated_at",
            ]
        )
        return {
            "ok": True,
            "saved": True,
            "callback_requested": True,
            "message": "Callback recorded. Confirm briefly, then end the call.",
        }

    if fname == "record_dnc":
        session.dnc_requested = True
        session.final_outcome = "PATIENT_DECLINED"
        session.save(
            update_fields=["dnc_requested", "final_outcome", "updated_at"]
        )
        phone = session.phone_number
        if phone:
            OptOut.objects.get_or_create(
                phone_number=phone,
                defaults={"outreach": session.outreach},
            )
        return {
            "ok": True,
            "saved": True,
            "dnc_requested": True,
            "message": "Do-not-call recorded. Apologize briefly and end the call.",
        }

    if fname == "schedule_appointment":
        preferred = str(args.get("preferred_time") or "").strip()
        notes = str(args.get("notes") or "").strip()
        if not preferred:
            return {
                "ok": False,
                "saved": False,
                "error": "preferred_time is required.",
            }
        session.appointment_needed = True
        session.appointment_scheduled = True
        session.appointment_details = "; ".join(
            part for part in (preferred, notes) if part
        )
        if not session.final_outcome:
            session.final_outcome = "APPOINTMENT_SCHEDULED"
        session.save(
            update_fields=[
                "appointment_needed",
                "appointment_scheduled",
                "appointment_details",
                "final_outcome",
                "updated_at",
            ]
        )
        return {
            "ok": True,
            "saved": True,
            "appointment_scheduled": True,
            "appointment_details": session.appointment_details,
            "message": (
                "Appointment request saved for clinic follow-up. "
                "Confirm the preferred time with the patient."
            ),
        }

    if fname == "send_renewal_video_sms":
        phone = (
            getattr(session, "phone_number", None)
            or str((call or {}).get("to_number") or "")
        ).strip()
        patient_name = (
            getattr(session, "patient_name", None)
            or str(
                ((call or {}).get("retell_llm_dynamic_variables") or {}).get(
                    "patient_name"
                )
                or ""
            )
        ).strip()
        result = send_renewal_video_sms(
            phone_number=phone,
            name=patient_name,
        )
        if not result.get("ok"):
            return {
                "ok": False,
                "saved": False,
                "error": result.get("error") or "Failed to send video SMS.",
            }

        payload = dict(session.state_payload or {})
        payload["renewal_video_sms"] = {
            "sid": result.get("sid") or "",
            "from_number": result.get("from_number") or "",
            "video_url": result.get("video_url") or "",
            "body": result.get("body") or "",
        }
        session.state_payload = payload
        session.assistance_type = "renewal_video"
        session.current_node = "household_income"
        session.save(
            update_fields=[
                "state_payload",
                "assistance_type",
                "current_node",
                "updated_at",
            ]
        )
        return {
            "ok": True,
            "saved": True,
            "video_sent": True,
            "message": (
                "Video SMS sent successfully. Confirm briefly that you texted the video, "
                "then immediately transition_to_household_income and continue the remaining "
                "questions. Do NOT offer Pointcare or transfer again unless asked. "
                "Do NOT say goodbye. Do NOT call end_call."
            ),
        }

    logger.warning("ai_pointcare unknown Retell function name=%s", fname)
    return {"ok": False, "saved": False, "error": f"Unknown function: {fname}"}
