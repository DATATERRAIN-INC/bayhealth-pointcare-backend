from datetime import date

from django.db.models import CharField, Q, Value
from django.db.models.functions import Cast, Concat
from django.utils import timezone

from apps.ai_caller.models import Call, Patient, PatientSource, UploadedFile
from apps.ai_caller.retell import (
    get_retell_call,
    normalize_phone,
    place_retell_care_call,
    place_retell_guardian_call,
)
from apps.ai_caller.serializers import PatientSerializer, PatientUploadSerializer
from common.excel import parse_patient_upload
from common.s3 import upload_bytes


def get_patient_queryset(*, search="", source=""):
    queryset = Patient.objects.all()
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
            | Q(live_agent_number__icontains=search)
        )
    if source and source != "all":
        queryset = queryset.filter(source=source)
    return queryset


def get_call_queryset(
    *,
    search="",
    source="",
    status="",
    patient_id="",
    retell_call_id="",
):
    queryset = Call.objects.select_related("patient").all()
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
    if status and status != "all":
        queryset = queryset.filter(status=status)
    if patient_id:
        queryset = queryset.filter(patient_id=patient_id)
    if retell_call_id:
        queryset = queryset.filter(retell_call_id=retell_call_id)
    return queryset


def _split_full_name(value):
    raw = (value or "").strip()
    if not raw:
        return "", ""
    parts = raw.split(None, 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def create_patient_from_row(row, *, source, upload=None, upload_file_key=""):
    first_name = (row.get("first_name") or "").strip()
    last_name = (row.get("last_name") or "").strip()
    if not first_name and not last_name:
        first_name, last_name = _split_full_name(row.get("name", ""))

    serializer = PatientSerializer(
        data={
            "first_name": first_name,
            "last_name": last_name,
            "address": row.get("address", ""),
            "dob": row.get("dob", ""),
            "doctor": row.get("doctor", ""),
            "country_code": row.get("country_code") or "",
            "phone_number": row.get("phone_number", ""),
            "live_agent_country_code": row.get("live_agent_country_code") or "",
            "live_agent_number": row.get("live_agent_number", ""),
        }
    )
    if not serializer.is_valid():
        return None, serializer.errors

    patient = serializer.save(
        source=source,
        upload=upload,
        upload_file_key=upload_file_key or "",
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


def upload_patients_from_file(django_file):
    upload_serializer = PatientUploadSerializer(data={"file": django_file})
    if not upload_serializer.is_valid():
        return None, upload_serializer.errors

    upload = upload_serializer.validated_data["file"]
    filename = getattr(upload, "name", "") or ""
    content_type = getattr(upload, "content_type", None)
    file_log = UploadedFile.objects.create(
        file_name=filename,
        status=UploadedFile.Status.FAILED,
    )

    try:
        upload.seek(0)
        file_bytes = upload.read()
    except Exception as exc:
        message = f"Could not read upload: {exc}"
        file_log.error_message = message
        file_log.save(update_fields=["error_message"])
        return None, {"file": [message]}

    if not file_bytes:
        message = "Uploaded file is empty."
        file_log.error_message = message
        file_log.save(update_fields=["error_message"])
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
        file_log.save(update_fields=["error_message"])
        return None, {"file": [str(exc)]}

    file_log.file_key = file_key
    file_log.save(update_fields=["file_key"])

    try:
        rows = parse_patient_upload(file_bytes, filename)
    except Exception as exc:
        message = f"Could not read upload: {exc}"
        file_log.error_message = message
        file_log.status = UploadedFile.Status.FAILED
        file_log.save(update_fields=["error_message", "status"])
        return None, {"file": [message]}

    created = []
    skipped = []

    for row_number, row in rows:
        patient, errors = create_patient_from_row(
            row,
            source=PatientSource.EXCEL,
            upload=file_log,
            upload_file_key=file_key,
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
    file_log.save(
        update_fields=["uploaded_count", "failed_count", "status", "error_message"]
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


def _map_retell_status(raw_status):
    status = (raw_status or "").strip().lower()
    if status in {"ended", "completed", "done"}:
        return Call.Status.COMPLETED
    if status in {"not_connected", "failed", "busy", "no_answer", "voicemail"}:
        return Call.Status.NOT_ATTENDED
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


def _save_call(patient, result, *, dial_number, transfer_number):
    return Call.objects.create(
        patient=patient,
        retell_call_id=result["call_id"],
        flow=result.get("flow") or Call.Flow.OUTBOUND,
        status=Call.Status.IN_PROGRESS,
        from_number=result.get("from_number") or "",
        to_number=dial_number,
        agent_id=result.get("agent_id") or "",
        transfer_number=transfer_number,
        started_at=timezone.now(),
    )


def place_outbound_call_for_patient(patient_id):
    patient = Patient.objects.filter(pk=patient_id).first()
    if not patient:
        return {
            "ok": False,
            "error": "Patient not found.",
            "status_code": 404,
        }

    dial_number = _combine_phone(patient.country_code, patient.phone_number)
    if not dial_number:
        return {
            "ok": False,
            "error": "Patient phone number is invalid.",
            "status_code": 400,
        }

    transfer_number = _combine_phone(
        patient.live_agent_country_code,
        patient.live_agent_number,
    )
    if not transfer_number:
        return {
            "ok": False,
            "error": "Live agent number is invalid.",
            "status_code": 400,
        }

    if _is_minor(patient.dob):
        result = place_retell_guardian_call(
            phone_number=dial_number,
            patient_name=patient.full_name,
            address_on_file=patient.address,
            provider_name=patient.doctor,
            transfer_number=transfer_number,
        )
        result["flow"] = result.get("flow") or Call.Flow.GUARDIAN
    else:
        result = place_retell_care_call(
            phone_number=dial_number,
            name=patient.doctor,
            patient_name=patient.full_name,
            address_on_file=patient.address,
            transfer_number=transfer_number,
        )
        result["flow"] = result.get("flow") or Call.Flow.OUTBOUND

    if result.get("ok"):
        call = _save_call(
            patient,
            result,
            dial_number=dial_number,
            transfer_number=transfer_number,
        )
        result["db_call_id"] = call.id

    return result


def update_call_from_retell_payload(payload):
    """Update call status/transcript from Retell webhook or get-call payload."""
    if not isinstance(payload, dict):
        return None, "Invalid payload."

    data = payload.get("call") if isinstance(payload.get("call"), dict) else payload
    call_id = str(data.get("call_id") or data.get("id") or "").strip()
    if not call_id:
        return None, "call_id is missing."

    call = Call.objects.filter(retell_call_id=call_id).first()
    if not call:
        return None, "Call not found."

    call.status = _map_retell_status(data.get("call_status") or data.get("status"))

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

    transcript = data.get("transcript_object") or data.get("transcript")
    if transcript is not None:
        call.transcript = _normalize_transcript(transcript)

    call.save(
        update_fields=["status", "transcript", "started_at", "ended_at", "updated_at"]
    )
    return call, None


def sync_call_transcript(call):
    result = get_retell_call(call.retell_call_id)
    if not result.get("ok"):
        return None, result.get("error") or "Failed to sync transcript."

    updated, error = update_call_from_retell_payload(result.get("data") or {})
    if error:
        return None, error
    return updated, None
