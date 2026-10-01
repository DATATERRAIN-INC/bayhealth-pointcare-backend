from datetime import time

from django.db import models
from django.utils import timezone


class PatientSource(models.TextChoices):
    MANUAL = "manual", "Manual"
    EXCEL = "excel", "Excel"


class UploadedFile(models.Model):
    class Status(models.TextChoices):
        SUCCESS = "success", "Success"
        PARTIAL = "partial", "Partial"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="uploads",
        null=True,
        blank=True,
    )
    created_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="uploads_created",
        null=True,
        blank=True,
    )
    updated_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="uploads_updated",
        null=True,
        blank=True,
    )
    file_name = models.CharField(max_length=255)
    file_key = models.CharField(max_length=512, blank=True, default="")
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.SUCCESS,
    )
    uploaded_count = models.PositiveIntegerField(default=0)
    failed_count = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return f"{self.file_name} ({self.status})"


class Patient(models.Model):
    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="patients",
        null=True,
        blank=True,
    )
    created_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="patients_created",
        null=True,
        blank=True,
    )
    updated_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="patients_updated",
        null=True,
        blank=True,
    )
    first_name = models.CharField(max_length=60)
    last_name = models.CharField(max_length=60, blank=True, default="")
    address = models.CharField(max_length=300)
    dob = models.DateField()
    doctor = models.CharField(max_length=120)
    country_code = models.CharField(max_length=8, blank=True, default="")
    phone_number = models.CharField(max_length=32)
    live_agent_country_code = models.CharField(max_length=8, blank=True, default="")
    live_agent_number = models.CharField(max_length=32)
    is_blocked = models.BooleanField(default=False)
    source = models.CharField(
        max_length=16,
        choices=PatientSource.choices,
        default=PatientSource.MANUAL,
    )
    upload = models.ForeignKey(
        UploadedFile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="patients",
    )
    upload_file_key = models.CharField(max_length=512, blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    @property
    def full_name(self):
        return " ".join(
            part for part in (self.first_name or "", self.last_name or "") if part
        ).strip()

    def __str__(self):
        return f"{self.full_name} ({self.phone_number})"


class Call(models.Model):
    class Status(models.TextChoices):
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETED = "completed", "Completed"
        NOT_ATTENDED = "not_attended", "Not Attended"

    class Flow(models.TextChoices):
        OUTBOUND = "outbound", "Outbound"
        GUARDIAN = "guardian", "Guardian"

    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="calls",
        null=True,
        blank=True,
    )
    created_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="calls_created",
        null=True,
        blank=True,
    )
    updated_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="calls_updated",
        null=True,
        blank=True,
    )
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="calls",
    )
    retell_call_id = models.CharField(max_length=120, unique=True)
    flow = models.CharField(max_length=16, choices=Flow.choices)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.IN_PROGRESS,
    )
    from_number = models.CharField(max_length=32, blank=True, default="")
    to_number = models.CharField(max_length=32, blank=True, default="")
    agent_id = models.CharField(max_length=120, blank=True, default="")
    transfer_number = models.CharField(max_length=32, blank=True, default="")
    transcript = models.JSONField(default=list, blank=True)
    retell_transcript = models.JSONField(default=list, blank=True)
    live_agent_transcript = models.JSONField(default=list, blank=True)
    recording_url = models.CharField(max_length=1024, blank=True, default="")
    warm_transfer_session_id = models.CharField(max_length=64, blank=True, default="")
    started_at = models.DateTimeField(default=timezone.now)
    ended_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-started_at", "-id"]

    def __str__(self):
        return f"Call #{self.id} ({self.status})"

    @property
    def duration_seconds(self):
        if not self.started_at:
            return None
        end = self.ended_at or (timezone.now() if self.status == self.Status.IN_PROGRESS else None)
        if not end:
            return None
        return max(0, int((end - self.started_at).total_seconds()))

    @property
    def has_transcript(self):
        return bool(self.transcript)


class CallerSettings(models.Model):
    """Per-user controller settings for automated AI calling."""

    user = models.OneToOneField(
        "users.User",
        on_delete=models.CASCADE,
        related_name="caller_settings",
    )
    created_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="caller_settings_created",
        null=True,
        blank=True,
    )
    updated_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="caller_settings_updated",
        null=True,
        blank=True,
    )
    calls_enabled = models.BooleanField(default=False)
    recording_enabled = models.BooleanField(default=True)
    start_time = models.TimeField(default=time(9, 0))
    end_time = models.TimeField(default=time(17, 0))
    timezone = models.CharField(max_length=64, default="America/New_York")
    max_calls_per_run = models.PositiveIntegerField(default=5)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Caller settings"
        verbose_name_plural = "Caller settings"

    @classmethod
    def load(cls, user):
        if user is None:
            raise ValueError("user is required for CallerSettings.load()")
        obj, _ = cls.objects.get_or_create(
            user=user,
            defaults={"created_by": user, "updated_by": user},
        )
        return obj

    def __str__(self):
        state = "enabled" if self.calls_enabled else "disabled"
        owner = getattr(self.user, "email", None) or "unassigned"
        return f"Caller settings ({owner}, {state})"
