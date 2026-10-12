from django.db import models
from django.utils import timezone
from synccast.models.channel import AbstractSyncCastChannel
from synccast.models.scope import AbstractSyncCastScope


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


class Scope(AbstractSyncCastScope):
    """SyncCast scope. AI caller queue uses name ``ai_calls``."""

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Channel(AbstractSyncCastChannel):
    """SyncCast channel under a scope. Queue channels: in_progress, waiting."""

    scope = models.ForeignKey(
        Scope,
        on_delete=models.CASCADE,
        related_name="channels",
    )

    class Meta:
        ordering = ["scope_id", "name"]
        constraints = [
            models.UniqueConstraint(
                fields=["scope", "name"],
                name="notifications_channel_scope_name_uniq",
            )
        ]

    def __str__(self):
        return f"{self.scope.name}/{self.name}"
