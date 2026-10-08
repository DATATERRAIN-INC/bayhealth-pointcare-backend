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
    service_name = models.CharField(max_length=120, default="")
    country_code = models.CharField(max_length=8, blank=True, default="")
    phone_number = models.CharField(max_length=32)
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
        QUEUED = "queued", "Queued"
        SCHEDULED = "scheduled", "Scheduled"
        PAUSED = "paused", "Paused"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETED = "completed", "Completed"
        NOT_ATTENDED = "not_attended", "Not Attended"
        CALLBACK = "callback", "Callback"
        CANCEL = "cancel", "Cancel"

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
    # Null until Retell place-call succeeds (queued rows have no Retell id yet).
    retell_call_id = models.CharField(max_length=120, unique=True, null=True, blank=True)
    flow = models.CharField(max_length=16, choices=Flow.choices)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.IN_PROGRESS,
    )
    is_paused = models.BooleanField(
        default=False,
        help_text="When true, Celery will not dial this queued call until resumed.",
    )
    from_number = models.CharField(max_length=32, blank=True, default="")
    to_number = models.CharField(max_length=32, blank=True, default="")
    agent_id = models.CharField(max_length=120, blank=True, default="")
    transfer_number = models.CharField(max_length=32, blank=True, default="")
    decline_reason = models.TextField(blank=True, default="")
    transcript = models.JSONField(default=list, blank=True)
    retell_transcript = models.JSONField(default=list, blank=True)
    live_agent_transcript = models.JSONField(default=list, blank=True)
    # Patient (inbound) warm-transfer recording URL (S3 preferred).
    recording_url = models.CharField(max_length=1024, blank=True, default="")
    # Live-agent / provider warm-transfer recording URL (S3 preferred).
    live_agent_recording_url = models.CharField(max_length=1024, blank=True, default="")
    # Twilio Call SID for the warm-transfer patient leg (CA...).
    warm_transfer_session_id = models.CharField(max_length=64, blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True, default=timezone.now)
    ended_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-started_at", "-id"]

    def __str__(self):
        return f"Call #{self.id} ({self.status})"

    @property
    def duration_seconds(self):
        """Full wall-clock call length: started_at → ended_at.

        For warm transfers, ended_at is extended when the Twilio conference
        ends so this covers Retell AI + live-agent talk (not Retell-only).
        """
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
    text_sms_enabled = models.BooleanField(default=False)
    start_time = models.TimeField(default=time(9, 0))
    end_time = models.TimeField(default=time(17, 0))
    timezone = models.CharField(max_length=64, default="America/New_York")
    max_calls_per_run = models.PositiveIntegerField(
        default=5,
        help_text="Max simultaneous in-progress outbound calls. New dials fill free slots only.",
    )
    call_trigger_count = models.PositiveIntegerField(default=3)
    sms_trigger_after_calls = models.PositiveIntegerField(
        default=3,
        help_text=(
            "After this many consecutive not-attended calls, auto-start SMS "
            "(when text_sms_enabled). Must be <= call_trigger_count."
        ),
    )
    reminder_timeframe_hours = models.PositiveIntegerField(
        default=24,
        help_text="Hours to wait after a missed call before the next reminder call.",
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Caller settings"
        verbose_name_plural = "Caller settings"

    @classmethod
    def load(cls, user):
        user_id = getattr(user, "pk", None)
        if not user_id:
            raise ValueError("A valid authenticated user is required for caller settings.")

        try:
            obj, _ = cls.objects.get_or_create(
                user_id=user_id,
                defaults={
                    "user_id": user_id,
                    "created_by_id": user_id,
                    "updated_by_id": user_id,
                },
            )
            return obj
        except Exception as exc:
            from django.db import IntegrityError

            if not isinstance(exc, IntegrityError):
                raise
            # Concurrent create or leftover bad row — return existing if possible.
            existing = cls.objects.filter(user_id=user_id).first()
            if existing:
                return existing
            raise ValueError(
                "Unable to create caller settings for this account. "
                "Please try again or contact support."
            ) from exc

    def __str__(self):
        state = "enabled" if self.calls_enabled else "disabled"
        owner = getattr(self.user, "email", None) or "unassigned"
        return f"Caller settings ({owner}, {state})"

    def active_live_agent_numbers(self):
        return self.live_agent_numbers.filter(is_active=True).order_by("id")

    def primary_live_agent_number(self):
        """First active live-agent number for outbound transfer."""
        return self.active_live_agent_numbers().first()


class ScheduledOutreach(models.Model):
    """Separate queue for patient callbacks and system reminder calls."""

    class Kind(models.TextChoices):
        CALLBACK_REQUESTED = "callback_requested", "Callback requested"
        REMINDER = "reminder", "Reminder"

    class Status(models.TextChoices):
        SCHEDULED = "scheduled", "Scheduled"
        TRIGGERED = "triggered", "Triggered"
        CANCELLED = "cancelled", "Cancelled"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="scheduled_outreaches",
        null=True,
        blank=True,
    )
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="scheduled_outreaches",
    )
    source_call = models.ForeignKey(
        Call,
        on_delete=models.SET_NULL,
        related_name="scheduled_outreaches_created",
        null=True,
        blank=True,
    )
    queued_call = models.ForeignKey(
        Call,
        on_delete=models.SET_NULL,
        related_name="scheduled_outreaches_queued",
        null=True,
        blank=True,
        help_text="Queued Call row that will be dialed when this outreach is due.",
    )
    triggered_call = models.ForeignKey(
        Call,
        on_delete=models.SET_NULL,
        related_name="scheduled_outreaches_triggered",
        null=True,
        blank=True,
    )
    kind = models.CharField(max_length=32, choices=Kind.choices)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.SCHEDULED,
    )
    scheduled_at = models.DateTimeField()
    raw_time_text = models.CharField(max_length=255, blank=True, default="")
    error_message = models.CharField(max_length=512, blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["scheduled_at", "id"]
        verbose_name = "Scheduled outreach"
        verbose_name_plural = "Scheduled outreaches"
        indexes = [
            models.Index(fields=["user", "status", "scheduled_at"]),
            models.Index(fields=["patient", "status"]),
        ]

    def __str__(self):
        return (
            f"ScheduledOutreach #{self.id} ({self.kind}, {self.status}) "
            f"at {self.scheduled_at}"
        )


class LiveAgentNumber(models.Model):
    """Transfer / live-agent numbers owned by a user's CallerSettings (many per user)."""

    caller_settings = models.ForeignKey(
        CallerSettings,
        on_delete=models.CASCADE,
        related_name="live_agent_numbers",
    )
    country_code = models.CharField(max_length=8, blank=True, default="")
    phone_number = models.CharField(max_length=32)
    label = models.CharField(max_length=64, blank=True, default="")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["id"]
        verbose_name = "Live agent number"
        verbose_name_plural = "Live agent numbers"

    def __str__(self):
        label = (self.label or "").strip()
        number = f"{self.country_code}{self.phone_number}".strip()
        return label or number or f"LiveAgentNumber #{self.id}"
