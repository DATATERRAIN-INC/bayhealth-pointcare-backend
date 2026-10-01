from django.db import models
from django.utils import timezone


class NotificationEvent(models.TextChoices):
    PATIENT_CREATED = "patient_created", "Patient created"
    PATIENT_UPDATED = "patient_updated", "Patient updated"
    PATIENT_BLOCKED = "patient_blocked", "Patient blocked"
    PATIENT_UNBLOCKED = "patient_unblocked", "Patient unblocked"
    PATIENT_UPLOAD = "patient_upload", "Patient upload"
    OUTBOUND_BATCH = "outbound_batch", "Outbound batch"


class Notification(models.Model):
    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="notifications",
        null=True,
        blank=True,
    )
    created_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="notifications_created",
        null=True,
        blank=True,
    )
    updated_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="notifications_updated",
        null=True,
        blank=True,
    )
    event_type = models.CharField(max_length=32, choices=NotificationEvent.choices)
    title = models.CharField(max_length=200)
    message = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return f"{self.event_type}: {self.title}"
