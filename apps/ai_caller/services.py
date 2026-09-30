from django.db.models import Q

from apps.ai_caller.models import Patient, PatientSource, UploadedFile
from apps.ai_caller.serializers import PatientSerializer, PatientUploadSerializer
from common.excel import parse_patient_upload
from common.s3 import upload_bytes


def get_patient_queryset(*, search="", source=""):
    queryset = Patient.objects.all()
    search = (search or "").strip()
    source = (source or "").strip().lower()

    if search:
        queryset = queryset.filter(
            Q(name__icontains=search)
            | Q(doctor__icontains=search)
            | Q(phone_number__icontains=search)
            | Q(live_agent_number__icontains=search)
        )
    if source and source != "all":
        queryset = queryset.filter(source=source)
    return queryset


def create_patient_from_row(row, *, source, upload=None, upload_file_key=""):
    serializer = PatientSerializer(
        data={
            "name": row.get("name", ""),
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
