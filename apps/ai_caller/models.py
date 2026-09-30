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
    first_name = models.CharField(max_length=60)
    last_name = models.CharField(max_length=60, blank=True, default="")
    address = models.CharField(max_length=300)
    dob = models.DateField()
    doctor = models.CharField(max_length=120)
    country_code = models.CharField(max_length=8, blank=True, default="")
    phone_number = models.CharField(max_length=32)
    live_agent_country_code = models.CharField(max_length=8, blank=True, default="")
    live_agent_number = models.CharField(max_length=32)
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
