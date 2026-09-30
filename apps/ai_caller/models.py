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
    name = models.CharField(max_length=120)
    address = models.CharField(max_length=300)
    dob = models.DateField()
    doctor = models.CharField(max_length=120)
    country_code = models.CharField(max_length=8, default="+1")
    phone_number = models.CharField(max_length=32)
    live_agent_country_code = models.CharField(max_length=8, default="+1")
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

    def __str__(self):
        return f"{self.name} ({self.phone_number})"
