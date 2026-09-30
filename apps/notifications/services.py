"""Reusable notification helpers for any app."""

from apps.notifications.models import Notification, NotificationEvent


def create_notification(*, event_type, title, message="", metadata=None):
    """
    Common entrypoint — call this from patient CRUD, Celery, uploads, etc.
    Returns the created Notification.
    """
    return Notification.objects.create(
        event_type=event_type,
        title=(title or "").strip()[:200],
        message=(message or "").strip(),
        metadata=metadata or {},
    )


def notify_patient_created(patient):
    name = patient.full_name or f"Patient #{patient.id}"
    return create_notification(
        event_type=NotificationEvent.PATIENT_CREATED,
        title="Patient added",
        message=f"{name} was added.",
        metadata={
            "patient_id": patient.id,
            "full_name": name,
            "source": patient.source,
        },
    )


def notify_patient_updated(patient, *, changed_fields=None):
    name = patient.full_name or f"Patient #{patient.id}"
    fields = list(changed_fields or [])
    detail = f" Updated fields: {', '.join(fields)}." if fields else ""
    return create_notification(
        event_type=NotificationEvent.PATIENT_UPDATED,
        title="Patient updated",
        message=f"{name} was updated.{detail}",
        metadata={
            "patient_id": patient.id,
            "full_name": name,
            "changed_fields": fields,
        },
    )


def notify_patient_block_toggle(patient):
    name = patient.full_name or f"Patient #{patient.id}"
    if patient.is_blocked:
        return create_notification(
            event_type=NotificationEvent.PATIENT_BLOCKED,
            title="Patient blocked",
            message=f"{name} was blocked and will not be dialed.",
            metadata={"patient_id": patient.id, "full_name": name, "is_blocked": True},
        )
    return create_notification(
        event_type=NotificationEvent.PATIENT_UNBLOCKED,
        title="Patient unblocked",
        message=f"{name} was unblocked and can be dialed again.",
        metadata={"patient_id": patient.id, "full_name": name, "is_blocked": False},
    )


def notify_patient_upload(*, uploaded_count, failed_count, upload_id=None, file_name=""):
    return create_notification(
        event_type=NotificationEvent.PATIENT_UPLOAD,
        title="Patient upload finished",
        message=(
            f"Uploaded {uploaded_count} patient(s)"
            + (f", {failed_count} failed" if failed_count else "")
            + "."
        ),
        metadata={
            "upload_id": upload_id,
            "file_name": file_name or "",
            "uploaded_count": uploaded_count,
            "failed_count": failed_count,
        },
    )


def notify_outbound_batch(result):
    """Notify after a Celery dialer run (including skipped runs)."""
    if not isinstance(result, dict):
        return None

    if result.get("skipped"):
        reason = result.get("reason") or "skipped"
        title = "Outbound dialer skipped"
        if reason == "calls_disabled":
            message = "Automated calling is disabled."
        elif reason == "outside_calling_window":
            message = "Outside the calling window — no calls placed."
        else:
            message = f"Dialer skipped ({reason})."
        return create_notification(
            event_type=NotificationEvent.OUTBOUND_BATCH,
            title=title,
            message=message,
            metadata=result,
        )

    placed = int(result.get("placed") or 0)
    failed = int(result.get("failed") or 0)
    attempted = int(result.get("attempted") or (placed + failed))
    return create_notification(
        event_type=NotificationEvent.OUTBOUND_BATCH,
        title="Outbound dialer run",
        message=(
            f"Triggered {attempted} patient call(s): "
            f"{placed} placed, {failed} failed."
        ),
        metadata={
            "attempted": attempted,
            "placed": placed,
            "failed": failed,
            "results": result.get("results") or [],
        },
    )
