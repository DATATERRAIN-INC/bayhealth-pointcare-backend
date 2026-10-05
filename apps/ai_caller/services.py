from datetime import date, datetime, time, timedelta
import logging

from django.conf import settings
from django.db.models import CharField, Q, Value
from django.db.models.functions import Cast, Concat
from django.utils import timezone

from apps.ai_caller.models import (
    Call,
    CallerSettings,
    Patient,
    PatientSource,
    ScheduledOutreach,
    UploadedFile,
)
from apps.ai_caller.scheduling import (
    cancel_scheduled_for_patient,
    due_scheduled_outreaches,
    mark_outreach_triggered,
    schedule_reminder_for_missed_call,
)
from apps.ai_caller.retell import (
    get_retell_call,
    normalize_phone,
    place_retell_care_call,
    place_retell_guardian_call,
)
from apps.ai_caller.serializers import PatientSerializer, PatientUploadSerializer
from apps.ai_caller.transcript_merge import merge_ai_and_humans
from common.excel import parse_patient_upload
from common.s3 import upload_bytes

dialer_logger = logging.getLogger("ai_caller.dialer")


def get_patient_queryset(*, user=None, search="", source="", is_blocked=None):
    queryset = Patient.objects.all()
    if user is not None:
        queryset = queryset.filter(user=user)
    search = (search or "").strip()
    source = (source or "").strip().lower()

    if search:
        queryset = queryset.annotate(
            _full_name=Concat("first_name", Value(" "), "last_name")
        ).filter(
            Q(first_name__icontains=search)
            | Q(last_name__icontains=search)
            | Q(_full_name__icontains=search)
            | Q(doctor__icontains=search)
            | Q(phone_number__icontains=search)
        )
    if source and source != "all":
        queryset = queryset.filter(source=source)
    if is_blocked is not None:
        queryset = queryset.filter(is_blocked=is_blocked)
    return queryset


def get_call_queryset(
    *,
    user=None,
    search="",
    source="",
    status="",
    patient_id="",
    retell_call_id="",
):
    queryset = Call.objects.select_related("patient").all()
    if user is not None:
        queryset = queryset.filter(user=user)
    search = (search or "").strip()
    source = (source or "").strip().lower()
    status = (status or "").strip().lower()
    patient_id = (patient_id or "").strip()
    retell_call_id = (retell_call_id or "").strip()

    if search:
        queryset = queryset.annotate(
            _patient_full_name=Concat(
                "patient__first_name", Value(" "), "patient__last_name"
            ),
            _dob_text=Cast("patient__dob", CharField()),
        ).filter(
            Q(patient__first_name__icontains=search)
            | Q(patient__last_name__icontains=search)
            | Q(_patient_full_name__icontains=search)
            | Q(patient__doctor__icontains=search)
            | Q(_dob_text__icontains=search)
        )

    if source and source != "all":
        queryset = queryset.filter(patient__source=source)
    if status and status != "all" and status != "queued":
        queryset = queryset.filter(status=status)
    if patient_id:
        queryset = queryset.filter(patient_id=patient_id)
    if retell_call_id:
        queryset = queryset.filter(retell_call_id=retell_call_id)
    return queryset


def get_queued_patients_queryset(*, user=None, search="", source=""):
    """
    Patients still waiting for an outbound call:
    - first-time / eligible auto-dial patients
    - patients with an open ScheduledOutreach (callback/reminder)
    Excludes blocked, completed, in-progress, and attempt-exhausted patients.
    """
    if user is None:
        return Patient.objects.none()

    settings_obj = CallerSettings.load(user)
    trigger_cap = max(1, int(settings_obj.call_trigger_count or 1))

    completed_ids = Call.objects.filter(
        user=user, status=Call.Status.COMPLETED
    ).values_list("patient_id", flat=True)
    in_progress_ids = Call.objects.filter(
        user=user, status=Call.Status.IN_PROGRESS
    ).values_list("patient_id", flat=True)

    from django.db.models import Count

    exhausted_ids = (
        Call.objects.filter(user=user, status=Call.Status.NOT_ATTENDED)
        .values("patient_id")
        .annotate(attempts=Count("id"))
        .filter(attempts__gte=trigger_cap)
        .values_list("patient_id", flat=True)
    )

    scheduled_ids = ScheduledOutreach.objects.filter(
        user=user,
        status=ScheduledOutreach.Status.SCHEDULED,
    ).values_list("patient_id", flat=True)

    # First-time style queue (same exclusions as dialer, no day/limit cut).
    ever_not_attended_ids = Call.objects.filter(
        user=user, status=Call.Status.NOT_ATTENDED
    ).values_list("patient_id", flat=True)

    first_time_ids = (
        Patient.objects.filter(user=user, is_blocked=False)
        .exclude(id__in=completed_ids)
        .exclude(id__in=in_progress_ids)
        .exclude(id__in=exhausted_ids)
        .exclude(id__in=scheduled_ids)
        .exclude(id__in=ever_not_attended_ids)
        .values_list("id", flat=True)
    )

    queued_ids = set(first_time_ids) | set(scheduled_ids)
    queryset = Patient.objects.filter(
        user=user, is_blocked=False, id__in=queued_ids
    ).exclude(id__in=completed_ids).exclude(id__in=in_progress_ids)

    search = (search or "").strip()
    source = (source or "").strip().lower()
    if search:
        queryset = queryset.annotate(
            _full_name=Concat("first_name", Value(" "), "last_name")
        ).filter(
            Q(first_name__icontains=search)
            | Q(last_name__icontains=search)
            | Q(_full_name__icontains=search)
            | Q(doctor__icontains=search)
            | Q(phone_number__icontains=search)
        )
    if source and source != "all":
        queryset = queryset.filter(source=source)
    return queryset.order_by("id")


def serialize_queued_patient_as_call(patient):
    """Shape a waiting patient like a Call list row with status=queued."""
    to_number = f"{patient.country_code or ''}{patient.phone_number or ''}".strip()
    return {
        "id": None,
        "patient": patient.id,
        "patient_name": patient.full_name,
        "retell_call_id": "",
        "flow": "",
        "status": "queued",
        "from_number": "",
        "to_number": to_number,
        "agent_id": "",
        "transfer_number": "",
        "decline_reason": "",
        "started_at": None,
        "ended_at": None,
        "duration_seconds": None,
        "has_transcript": False,
        "message_count": 0,
        "created_at": patient.created_at,
        "updated_at": patient.updated_at,
    }


def _split_full_name(value):
    raw = (value or "").strip()
    if not raw:
        return "", ""
    parts = raw.split(None, 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _normalize_upload_dob(value):
    """Excel may send '2000-10-30 00:00:00'; DateField wants YYYY-MM-DD."""
    text = str(value or "").strip()
    if not text:
        return ""
    if " " in text:
        text = text.split(" ", 1)[0]
    if "T" in text:
        text = text.split("T", 1)[0]
    return text


def create_patient_from_row(row, *, source, upload=None, upload_file_key="", user=None):
    first_name = (row.get("first_name") or "").strip()
    last_name = (row.get("last_name") or "").strip()
    if not first_name and not last_name:
        first_name, last_name = _split_full_name(row.get("name", ""))

    serializer = PatientSerializer(
        data={
            "first_name": first_name,
            "last_name": last_name,
            "address": row.get("address", ""),
            "dob": _normalize_upload_dob(row.get("dob", "")),
            "doctor": row.get("doctor", ""),
            "service_name": row.get("service_name") or row.get("service") or "",
            "country_code": row.get("country_code") or "",
            "phone_number": row.get("phone_number", ""),
        }
    )
    if not serializer.is_valid():
        return None, serializer.errors

    patient = serializer.save(
        source=source,
        upload=upload,
        upload_file_key=upload_file_key or "",
        user=user,
        created_by=user,
        updated_by=user,
    )
    return PatientSerializer(patient).data, None


def _resolve_upload_status(uploaded_count, failed_count, error_message=""):
    if error_message and uploaded_count == 0:
        return UploadedFile.Status.FAILED
    if failed_count and uploaded_count:
        return UploadedFile.Status.PARTIAL
    if failed_count and not uploaded_count:
        return UploadedFile.Status.FAILED
    return UploadedFile.Status.SUCCESS


def upload_patients_from_file(django_file, *, user=None):
    upload_serializer = PatientUploadSerializer(data={"file": django_file})
    if not upload_serializer.is_valid():
        return None, upload_serializer.errors

    upload = upload_serializer.validated_data["file"]
    filename = getattr(upload, "name", "") or ""
    content_type = getattr(upload, "content_type", None)
    file_log = UploadedFile.objects.create(
        user=user,
        created_by=user,
        updated_by=user,
        file_name=filename,
        status=UploadedFile.Status.FAILED,
    )

    try:
        upload.seek(0)
        file_bytes = upload.read()
    except Exception as exc:
        message = f"Could not read upload: {exc}"
        file_log.error_message = message
        file_log.updated_by = user
        file_log.save(update_fields=["error_message", "updated_by"])
        return None, {"file": [message]}

    if not file_bytes:
        message = "Uploaded file is empty."
        file_log.error_message = message
        file_log.updated_by = user
        file_log.save(update_fields=["error_message", "updated_by"])
        return None, {"file": [message]}

    try:
        file_key = upload_bytes(
            file_bytes,
            filename=filename,
            folder="patients",
            content_type=content_type,
        )
    except (ValueError, RuntimeError) as exc:
        file_log.error_message = str(exc)
        file_log.updated_by = user
        file_log.save(update_fields=["error_message", "updated_by"])
        return None, {"file": [str(exc)]}

    file_log.file_key = file_key
    file_log.updated_by = user
    file_log.save(update_fields=["file_key", "updated_by"])

    try:
        rows = parse_patient_upload(file_bytes, filename)
    except Exception as exc:
        message = f"Could not read upload: {exc}"
        file_log.error_message = message
        file_log.status = UploadedFile.Status.FAILED
        file_log.updated_by = user
        file_log.save(update_fields=["error_message", "status", "updated_by"])
        return None, {"file": [message]}

    created = []
    skipped = []

    for row_number, row in rows:
        patient, errors = create_patient_from_row(
            row,
            source=PatientSource.EXCEL,
            upload=file_log,
            upload_file_key=file_key,
            user=user,
        )
        if errors:
            skipped.append({"row": row_number, "errors": errors})
            continue
        created.append(patient)

    file_log.uploaded_count = len(created)
    file_log.failed_count = len(skipped)
    file_log.status = _resolve_upload_status(len(created), len(skipped))
    if skipped and not created:
        file_log.error_message = "All rows failed validation."
    file_log.updated_by = user
    file_log.save(
        update_fields=[
            "uploaded_count",
            "failed_count",
            "status",
            "error_message",
            "updated_by",
        ]
    )

    from apps.notifications.services import notify_patient_upload

    notify_patient_upload(
        uploaded_count=len(created),
        failed_count=len(skipped),
        upload_id=file_log.id,
        file_name=filename,
        user=user,
    )

    return {
        "upload_id": file_log.id,
        "file_name": filename,
        "file_key": file_key,
        "uploaded": len(created),
        "failed": len(skipped),
        "records": created,
        "skipped": skipped,
    }, None


MINOR_AGE_YEARS = 18


def _patient_age(dob):
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _is_minor(dob):
    return _patient_age(dob) < MINOR_AGE_YEARS


def _combine_phone(country_code, phone_number):
    """Build E.164 phone. Missing country_code falls back to DEFAULT_COUNTRY_CODE."""
    return normalize_phone(
        phone_number or "",
        country_code=(country_code or "").strip(),
    )


def _resolve_live_agent_numbers(user):
    """Ordered E.164 list of active live-agent numbers from CallerSettings."""
    if user is None or not getattr(user, "pk", None):
        return []
    settings_obj = CallerSettings.load(user)
    numbers = []
    for agent in settings_obj.active_live_agent_numbers():
        combined = _combine_phone(agent.country_code, agent.phone_number)
        if combined and combined not in numbers:
            numbers.append(combined)
    return numbers


def _resolve_live_agent_number(user):
    """First active live-agent number (compat helper)."""
    numbers = _resolve_live_agent_numbers(user)
    return numbers[0] if numbers else ""


def _map_retell_status(raw_status, *, disconnection_reason="", event=""):
    status = (raw_status or "").strip().lower()
    reason = (disconnection_reason or "").strip().lower()
    event = (event or "").strip().lower()

    not_attended_reasons = {
        "dial_no_answer",
        "dial_busy",
        "dial_failed",
        "dial_invalid",
        "no_answer",
        "busy",
        "failed",
        "voicemail_reached",
        "machine_detected",
    }
    if reason in not_attended_reasons:
        return Call.Status.NOT_ATTENDED
    if status in {"not_connected", "failed", "busy", "no_answer", "voicemail", "error"}:
        return Call.Status.NOT_ATTENDED

    # Answered then hung up (user/agent) still counts as completed.
    if status in {"ended", "completed", "done", "analyzed"}:
        return Call.Status.COMPLETED
    if event in {"call_ended", "call_analyzed"}:
        return Call.Status.COMPLETED

    if status in {"ongoing", "registered", "in_progress", "ringing"}:
        return Call.Status.IN_PROGRESS
    return Call.Status.IN_PROGRESS


def _map_speaker(raw_speaker):
    speaker = (raw_speaker or "").strip().lower()
    if speaker in {"agent", "ai", "assistant", "bot"}:
        return "agent"
    if speaker in {"user", "patient", "human", "customer"}:
        return "patient"
    return speaker or "unknown"


def _normalize_transcript(raw):
    if not raw:
        return []
    if isinstance(raw, str):
        return [{"speaker": "unknown", "text": raw, "at": None}]

    items = []
    if not isinstance(raw, list):
        return [{"speaker": "unknown", "text": str(raw), "at": None}]

    for entry in raw:
        if not isinstance(entry, dict):
            items.append({"speaker": "unknown", "text": str(entry), "at": None})
            continue

        text = entry.get("content") or entry.get("text") or ""
        if not text and isinstance(entry.get("words"), list):
            text = " ".join(
                str(word.get("word") or word.get("text") or "")
                for word in entry["words"]
                if isinstance(word, dict)
            ).strip()

        items.append(
            {
                "speaker": _map_speaker(
                    entry.get("role") or entry.get("speaker") or entry.get("source")
                ),
                "text": text,
                "at": entry.get("created_at")
                or entry.get("timestamp")
                or entry.get("at")
                or entry.get("time")
                or None,
            }
        )
    return items


def _parse_retell_time(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        # Retell often returns epoch ms or seconds
        ts = float(value)
        if ts > 1_000_000_000_000:
            ts = ts / 1000.0
        from datetime import datetime, timezone as dt_timezone

        return datetime.fromtimestamp(ts, tz=dt_timezone.utc)
    if isinstance(value, str):
        from django.utils.dateparse import parse_datetime

        parsed = parse_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _save_call(patient, result, *, dial_number, transfer_number, session_id="", actor=None):
    actor = actor or patient.user
    return Call.objects.create(
        user=patient.user,
        created_by=actor,
        updated_by=actor,
        patient=patient,
        retell_call_id=result["call_id"],
        flow=result.get("flow") or Call.Flow.OUTBOUND,
        status=Call.Status.IN_PROGRESS,
        from_number=result.get("from_number") or "",
        to_number=dial_number,
        agent_id=result.get("agent_id") or "",
        transfer_number=transfer_number,
        warm_transfer_session_id=(session_id or "")[:64],
        started_at=timezone.now(),
    )


def _warm_transfer_enabled():
    return bool(getattr(settings, "WARM_TRANSFER_ENABLED", False))


def _prepare_warm_transfer_bridge(*, patient, dial_number, live_agent_numbers):
    from apps.ai_caller.twilio_bridge import prepare_retell_bridge

    numbers = [n for n in (live_agent_numbers or []) if n]
    primary = numbers[0] if numbers else ""
    return prepare_retell_bridge(
        phone_number=dial_number,
        name=patient.full_name or "there",
        transfer_number=primary,
        transfer_numbers=numbers,
        service_name=(getattr(patient, "service_name", None) or "").strip() or "care",
        extra={
            "patient_id": patient.id,
            "live_agent_number": primary,
            "live_agent_numbers": numbers,
            "inbound_speaker": "patient",
            "dial_speaker": "provider",
        },
    )


def place_outbound_call_for_patient(patient_id, *, user=None):
    queryset = Patient.objects.all()
    if user is not None:
        queryset = queryset.filter(user=user)
    patient = queryset.filter(pk=patient_id).first()
    if not patient:
        return {
            "ok": False,
            "error": "Patient not found.",
            "status_code": 404,
        }

    if patient.is_blocked:
        return {
            "ok": False,
            "error": "Patient is blocked.",
            "status_code": 400,
        }

    if Call.objects.filter(patient=patient, status=Call.Status.COMPLETED).exists():
        return {
            "ok": False,
            "error": "Patient already has a completed call.",
            "status_code": 400,
        }

    if Call.objects.filter(patient=patient, status=Call.Status.IN_PROGRESS).exists():
        return {
            "ok": False,
            "error": "Patient already has a call in progress.",
            "status_code": 400,
        }

    if patient.user_id:
        trigger_cap = max(
            1, int(CallerSettings.load(patient.user).call_trigger_count or 1)
        )
        not_attended_count = Call.objects.filter(
            patient=patient, status=Call.Status.NOT_ATTENDED
        ).count()
        if not_attended_count >= trigger_cap:
            return {
                "ok": False,
                "error": (
                    f"Patient reached max not-attended attempts "
                    f"({trigger_cap})."
                ),
                "status_code": 400,
            }

    dial_number = _combine_phone(patient.country_code, patient.phone_number)
    if not dial_number:
        return {
            "ok": False,
            "error": "Patient phone number is invalid.",
            "status_code": 400,
        }

    live_agent_numbers = _resolve_live_agent_numbers(patient.user)
    if not live_agent_numbers:
        return {
            "ok": False,
            "error": "No active live agent number configured in caller settings.",
            "status_code": 400,
        }
    live_agent_number = live_agent_numbers[0]

    retell_transfer_number = live_agent_number
    session_id = ""
    bridge = None
    if _warm_transfer_enabled():
        bridge = _prepare_warm_transfer_bridge(
            patient=patient,
            dial_number=dial_number,
            live_agent_numbers=live_agent_numbers,
        )
        if not bridge.get("ok"):
            return bridge
        retell_transfer_number = bridge["bridge_number"]
        session_id = bridge.get("session_id") or ""

    service = (getattr(patient, "service_name", None) or "").strip() or "care"

    if _is_minor(patient.dob):
        result = place_retell_guardian_call(
            phone_number=dial_number,
            patient_name=patient.full_name,
            address_on_file=patient.address,
            provider_name=patient.doctor,
            service_name=service,
            measure_name=service,
            transfer_number=retell_transfer_number,
        )
        result["flow"] = result.get("flow") or Call.Flow.GUARDIAN
    else:
        result = place_retell_care_call(
            phone_number=dial_number,
            name=patient.doctor,
            patient_name=patient.full_name,
            address_on_file=patient.address,
            service_name=service,
            transfer_number=retell_transfer_number,
        )
        result["flow"] = result.get("flow") or Call.Flow.OUTBOUND

    if not result.get("ok"):
        if bridge and bridge.get("session"):
            from apps.ai_caller.twilio_bridge import restore_inbound_voice_url

            restore_inbound_voice_url(bridge.get("session") or {})
        return result

    call = _save_call(
        patient,
        result,
        dial_number=dial_number,
        transfer_number=live_agent_number,
        session_id=session_id,
        actor=user or patient.user,
    )
    result["db_call_id"] = call.id
    result["warm_transfer"] = bool(session_id)

    if session_id and result.get("call_id"):
        from apps.ai_caller.twilio_bridge import attach_retell_call

        attach_retell_call(session_id, result["call_id"])

    return result


def _extract_decline_reason(data):
    """Pull decline reason from Retell tool-call payloads or transcript metadata."""
    if not isinstance(data, dict):
        return ""

    candidates = []

    for key in ("tool_calls", "function_calls", "tool_call_outputs"):
        items = data.get(key)
        if isinstance(items, list):
            candidates.extend(items)

    transcript = data.get("transcript_with_tool_calls") or data.get("transcript_object")
    if isinstance(transcript, list):
        for entry in transcript:
            if not isinstance(entry, dict):
                continue
            if entry.get("role") in {"tool_call_invocation", "tool_call_result", "tool"}:
                candidates.append(entry)
            inv = entry.get("tool_call") or entry.get("invocation")
            if isinstance(inv, dict):
                candidates.append(inv)

    collected = data.get("collected_dynamic_variables") or data.get("retell_llm_dynamic_variables")
    if isinstance(collected, dict):
        for key in ("decline_reason", "not_interested_reason", "reason"):
            value = collected.get(key)
            if value:
                return str(value).strip()[:2000]

    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = str(
            item.get("name")
            or item.get("tool_name")
            or item.get("function_name")
            or ""
        ).strip()
        if name and name != "log_decline_reason":
            continue
        args = (
            item.get("arguments")
            or item.get("args")
            or item.get("parameters")
            or item.get("content")
            or {}
        )
        if isinstance(args, str):
            try:
                import json

                args = json.loads(args)
            except Exception:
                if args.strip():
                    return args.strip()[:2000]
                continue
        if isinstance(args, dict):
            reason = (
                args.get("reason")
                or args.get("decline_reason")
                or args.get("not_interested_reason")
                or ""
            )
            if reason:
                return str(reason).strip()[:2000]
    return ""


def save_call_decline_reason(*, retell_call_id="", reason="", call=None):
    """Persist not-interested / decline reason onto the Call row."""
    text = (reason or "").strip()[:2000]
    if not text:
        return None, "Reason is required."

    if call is None:
        call_id = str(retell_call_id or "").strip()
        if not call_id:
            return None, "call_id is missing."
        call = Call.objects.filter(retell_call_id=call_id).first()
        if not call:
            return None, "Call not found."

    call.decline_reason = text
    call.save(update_fields=["decline_reason", "updated_at"])
    dialer_logger.info(
        "DECLINE_REASON call_id=%s retell_call_id=%s reason=%s",
        call.id,
        call.retell_call_id,
        text[:200],
    )
    return call, None


def update_call_from_retell_payload(payload):
    """Update call status/transcript from Retell webhook or get-call payload."""
    if not isinstance(payload, dict):
        return None, "Invalid payload."

    event = str(payload.get("event") or payload.get("name") or "").strip()
    data = payload.get("call") if isinstance(payload.get("call"), dict) else payload
    call_id = str(data.get("call_id") or data.get("id") or "").strip()
    if not call_id:
        return None, "call_id is missing."

    call = Call.objects.filter(retell_call_id=call_id).first()
    if not call:
        return None, "Call not found."

    previous_status = call.status
    raw_status = data.get("call_status") or data.get("status") or ""
    if not raw_status and event.lower() in {"call_ended", "call_analyzed"}:
        raw_status = "ended"

    call.status = _map_retell_status(
        raw_status,
        disconnection_reason=str(
            data.get("disconnection_reason") or data.get("disconnect_reason") or ""
        ),
        event=event,
    )

    started_at = _parse_retell_time(
        data.get("start_timestamp") or data.get("started_at") or data.get("start_time")
    )
    ended_at = _parse_retell_time(
        data.get("end_timestamp") or data.get("ended_at") or data.get("end_time")
    )
    if started_at:
        call.started_at = started_at
    if ended_at:
        call.ended_at = ended_at
    elif call.status in {Call.Status.COMPLETED, Call.Status.NOT_ATTENDED} and not call.ended_at:
        call.ended_at = timezone.now()

    transcript = (
        data.get("transcript_object")
        or data.get("transcript_with_tool_calls")
        or data.get("transcript")
    )
    if transcript is not None:
        # Drop pure tool-call rows; keep spoken turns only.
        if isinstance(transcript, list):
            spoken = []
            for entry in transcript:
                if not isinstance(entry, dict):
                    spoken.append(entry)
                    continue
                role = str(entry.get("role") or "").strip().lower()
                if role in {"tool_call_invocation", "tool_call_result", "tool"}:
                    continue
                spoken.append(entry)
            transcript = spoken
        ai_items = _normalize_transcript(transcript)
        for item in ai_items:
            item["segment"] = "ai"
        # Keep non-empty lines only.
        ai_items = [item for item in ai_items if (item.get("text") or "").strip()]
        if ai_items or not (call.retell_transcript or []):
            call.retell_transcript = ai_items
            call.transcript = merge_ai_and_humans(
                ai_items, call.live_agent_transcript or []
            )

    update_fields = [
        "status",
        "transcript",
        "retell_transcript",
        "started_at",
        "ended_at",
        "updated_at",
    ]
    reason = _extract_decline_reason(data) or _extract_decline_reason(payload)
    if reason and not (call.decline_reason or "").strip():
        call.decline_reason = reason
        update_fields.append("decline_reason")

    recording = data.get("recording_url") or data.get("public_log_url") or ""
    if recording and not (call.recording_url or "").strip():
        call.recording_url = str(recording)[:1024]
        update_fields.append("recording_url")

    call.save(update_fields=update_fields)
    dialer_logger.info(
        "CALL_STATUS_UPDATE call_id=%s retell_call_id=%s status=%s event=%s raw=%s reason=%s",
        call.id,
        call.retell_call_id,
        call.status,
        event or "-",
        raw_status or "-",
        str(data.get("disconnection_reason") or "")[:80],
    )

    if call.status == Call.Status.COMPLETED:
        # Keep a callback requested on this same call (busy → call later).
        # Cancel older reminders/callbacks that this completed outreach replaces.
        cancel_scheduled_for_patient(
            call.patient,
            reason="Patient call completed",
            exclude_source_call=call,
        )
    elif (
        call.status == Call.Status.NOT_ATTENDED
        and previous_status != Call.Status.NOT_ATTENDED
    ):
        schedule_reminder_for_missed_call(call)

    return call, None


def sync_call_transcript(call):
    result = get_retell_call(call.retell_call_id)
    if not result.get("ok"):
        return None, result.get("error") or "Failed to sync transcript."

    updated, error = update_call_from_retell_payload(result.get("data") or {})
    if error:
        return None, error
    return updated, None


def sync_in_progress_calls_from_retell(*, user=None, limit=25):
    """
    Fallback when Retell webhooks are missing/misconfigured:
    pull latest status for recent in_progress rows from Retell get-call.
    """
    qs = Call.objects.filter(status=Call.Status.IN_PROGRESS).order_by("-started_at")
    if user is not None:
        qs = qs.filter(user=user)
    synced = 0
    for call in qs[: max(1, int(limit))]:
        try:
            updated, error = sync_call_transcript(call)
            if updated and not error:
                synced += 1
        except Exception:
            dialer_logger.exception(
                "Failed syncing in_progress call id=%s retell=%s",
                call.id,
                call.retell_call_id,
            )
    return synced


def resolve_timezone(tz_name):
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # Python < 3.9
        from backports.zoneinfo import ZoneInfo

    return ZoneInfo((tz_name or "America/New_York").strip())


def get_caller_settings(user=None):
    user_id = getattr(user, "pk", None)
    if not user_id:
        raise ValueError("Authentication required to load caller settings.")
    return CallerSettings.load(user)


def _as_time(value):
    if isinstance(value, time):
        return value.replace(second=0, microsecond=0)
    if isinstance(value, str):
        raw = value.strip()
        for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M%p"):
            try:
                return datetime.strptime(raw, fmt).time().replace(second=0, microsecond=0)
            except ValueError:
                continue
    raise ValueError(f"Invalid time value: {value!r}")


def build_calling_window_summary(settings_obj=None):
    if settings_obj is None:
        raise ValueError("settings_obj is required")
    start_t = _as_time(settings_obj.start_time)
    end_t = _as_time(settings_obj.end_time)
    start = start_t.strftime("%I:%M %p").lstrip("0")
    end = end_t.strftime("%I:%M %p").lstrip("0")
    tz_name = settings_obj.timezone
    try:
        now = timezone.now().astimezone(resolve_timezone(tz_name))
        abbrev = now.tzname() or tz_name
    except Exception:
        abbrev = tz_name

    start_minutes = start_t.hour * 60 + start_t.minute
    end_minutes = end_t.hour * 60 + end_t.minute
    hours = max(0, (end_minutes - start_minutes) / 60)
    hours_label = f"{hours:g} hour" if hours == 1 else f"{hours:g} hours"
    return (
        f"Calls run {start} – {end} {abbrev} ({hours_label}). "
        "Calls outside this window wait until it opens."
    )


def is_within_calling_window(settings_obj=None, when=None):
    if settings_obj is None:
        raise ValueError("settings_obj is required")
    try:
        tz = resolve_timezone(settings_obj.timezone)
    except Exception:
        return False

    local_now = (when or timezone.now()).astimezone(tz)
    current = local_now.time().replace(second=0, microsecond=0)
    start = _as_time(settings_obj.start_time)
    end = _as_time(settings_obj.end_time)
    return start <= current <= end


def _local_day_bounds(settings_obj):
    tz = resolve_timezone(settings_obj.timezone)
    local_now = timezone.now().astimezone(tz)
    start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    return start_local, end_local


def patients_due_for_outbound(settings_obj=None, limit=None):
    """
    Eligible patients for first-time / non-queue auto dial:
    - not blocked
    - no completed call (never redial)
    - no in-progress call (leave alone)
    - not_attended attempts under call_trigger_count
    - no in-progress/completed attempt today
    - no open ScheduledOutreach (callbacks/reminders are drained separately)
    Retries after not_attended go through the ScheduledOutreach reminder queue.
    """
    settings_obj = settings_obj or None
    if settings_obj is None:
        raise ValueError("settings_obj is required")
    day_start, day_end = _local_day_bounds(settings_obj)
    user = settings_obj.user
    trigger_cap = max(1, int(settings_obj.call_trigger_count or 1))

    completed_ids = Call.objects.filter(
        status=Call.Status.COMPLETED,
        user=user,
    ).values_list("patient_id", flat=True)

    in_progress_ids = Call.objects.filter(
        status=Call.Status.IN_PROGRESS,
        user=user,
    ).values_list("patient_id", flat=True)

    # Block same-day only for in_progress / completed — not_attended may retry via queue.
    active_today_ids = (
        Call.objects.filter(
            user=user,
            started_at__gte=day_start,
            started_at__lt=day_end,
            status__in=[Call.Status.IN_PROGRESS, Call.Status.COMPLETED],
        ).values_list("patient_id", flat=True)
    )

    from django.db.models import Count

    exhausted_ids = (
        Call.objects.filter(user=user, status=Call.Status.NOT_ATTENDED)
        .values("patient_id")
        .annotate(attempts=Count("id"))
        .filter(attempts__gte=trigger_cap)
        .values_list("patient_id", flat=True)
    )

    queued_ids = ScheduledOutreach.objects.filter(
        user=user,
        status=ScheduledOutreach.Status.SCHEDULED,
    ).values_list("patient_id", flat=True)

    # Patients who already have a not-attended attempt wait for the reminder queue.
    ever_not_attended_ids = Call.objects.filter(
        user=user, status=Call.Status.NOT_ATTENDED
    ).values_list("patient_id", flat=True)

    queryset = (
        Patient.objects.filter(user=user, is_blocked=False)
        .exclude(id__in=completed_ids)
        .exclude(id__in=in_progress_ids)
        .exclude(id__in=active_today_ids)
        .exclude(id__in=exhausted_ids)
        .exclude(id__in=queued_ids)
        .exclude(id__in=ever_not_attended_ids)
        .order_by("id")
    )
    limit = limit if limit is not None else settings_obj.max_calls_per_run
    if limit:
        queryset = queryset[: max(1, int(limit))]
    return list(queryset)


def _run_scheduled_outbound_for_settings(settings_obj):
    user_id = settings_obj.user_id
    email = getattr(settings_obj.user, "email", "") or ""

    if not settings_obj.calls_enabled:
        payload = {
            "ok": True,
            "skipped": True,
            "reason": "calls_disabled",
            "placed": 0,
            "failed": 0,
            "user_id": user_id,
        }
        dialer_logger.info(
            "NOT_TRIGGERED user_id=%s email=%s reason=calls_disabled",
            user_id,
            email,
        )
        return payload

    if not is_within_calling_window(settings_obj):
        window_summary = build_calling_window_summary(settings_obj)
        payload = {
            "ok": True,
            "skipped": True,
            "reason": "outside_calling_window",
            "placed": 0,
            "failed": 0,
            "user_id": user_id,
            "window_summary": window_summary,
        }
        dialer_logger.info(
            "NOT_TRIGGERED user_id=%s email=%s reason=outside_calling_window detail=%s",
            user_id,
            email,
            window_summary,
        )
        return payload

    max_per_run = max(1, int(settings_obj.max_calls_per_run or 1))
    due_queue = due_scheduled_outreaches(settings_obj, limit=max_per_run)
    remaining = max(0, max_per_run - len(due_queue))
    patients = patients_due_for_outbound(settings_obj, limit=remaining) if remaining else []

    if not due_queue and not patients:
        total = Patient.objects.filter(user=settings_obj.user).count()
        blocked = Patient.objects.filter(
            user=settings_obj.user, is_blocked=True
        ).count()
        dialer_logger.info(
            "NOT_TRIGGERED user_id=%s email=%s reason=no_eligible_patients "
            "total_patients=%s blocked=%s max_calls_per_run=%s",
            user_id,
            email,
            total,
            blocked,
            settings_obj.max_calls_per_run,
        )
        return {
            "ok": True,
            "skipped": False,
            "placed": 0,
            "failed": 0,
            "attempted": 0,
            "results": [],
            "user_id": user_id,
            "reason": "no_eligible_patients",
        }

    placed = 0
    failed = 0
    results = []

    for row in due_queue:
        result = place_outbound_call_for_patient(row.patient_id, user=settings_obj.user)
        entry = {
            "patient_id": row.patient_id,
            "scheduled_outreach_id": row.id,
            "kind": row.kind,
            "ok": bool(result.get("ok")),
            "error": result.get("error") or "",
            "call_id": result.get("call_id") or "",
        }
        results.append(entry)
        if entry["ok"]:
            placed += 1
            triggered_call = None
            call_pk = result.get("db_call_id") or result.get("call_db_id")
            if call_pk:
                triggered_call = Call.objects.filter(pk=call_pk).first()
            if not triggered_call and entry["call_id"]:
                triggered_call = Call.objects.filter(
                    retell_call_id=entry["call_id"]
                ).first()
            mark_outreach_triggered(row, triggered_call)
            dialer_logger.info(
                "QUEUE_TRIGGERED user_id=%s email=%s patient_id=%s "
                "scheduled_id=%s kind=%s call_id=%s",
                user_id,
                email,
                row.patient_id,
                row.id,
                row.kind,
                entry["call_id"],
            )
        else:
            failed += 1
            # Keep scheduled so the next run can retry; record last error.
            row.error_message = (entry["error"] or "Place failed")[:512]
            row.save(update_fields=["error_message", "updated_at"])
            dialer_logger.info(
                "QUEUE_NOT_TRIGGERED user_id=%s email=%s patient_id=%s "
                "scheduled_id=%s reason=place_failed error=%s",
                user_id,
                email,
                row.patient_id,
                row.id,
                entry["error"],
            )

    for patient in patients:
        result = place_outbound_call_for_patient(patient.id, user=settings_obj.user)
        entry = {
            "patient_id": patient.id,
            "ok": bool(result.get("ok")),
            "error": result.get("error") or "",
            "call_id": result.get("call_id") or "",
        }
        results.append(entry)
        if entry["ok"]:
            placed += 1
            dialer_logger.info(
                "TRIGGERED user_id=%s email=%s patient_id=%s call_id=%s",
                user_id,
                email,
                patient.id,
                entry["call_id"],
            )
        else:
            failed += 1
            dialer_logger.info(
                "NOT_TRIGGERED user_id=%s email=%s patient_id=%s reason=place_failed error=%s",
                user_id,
                email,
                patient.id,
                entry["error"],
            )

    payload = {
        "ok": True,
        "skipped": False,
        "placed": placed,
        "failed": failed,
        "attempted": len(due_queue) + len(patients),
        "results": results,
        "user_id": user_id,
    }
    if due_queue or patients:
        from apps.notifications.services import notify_outbound_batch

        notify_outbound_batch(payload, user=settings_obj.user)
    return payload


def run_scheduled_outbound_calls():
    """
    Celery entrypoint: place outbound calls per user when enabled and in window.
    Reuses place_outbound_call_for_patient (same path as PlaceOutboundCallView).
    """
    runs = []
    queryset = CallerSettings.objects.filter(
        calls_enabled=True, user__isnull=False
    ).select_related("user")
    count = queryset.count()
    if count == 0:
        dialer_logger.info(
            "NOT_TRIGGERED reason=no_users_with_calls_enabled"
        )
    for settings_obj in queryset:
        runs.append(_run_scheduled_outbound_for_settings(settings_obj))
    return {
        "ok": True,
        "runs": runs,
        "users": len(runs),
    }
