from django.db import models


class SmsConversation(models.Model):
    chat_id = models.CharField(max_length=128, unique=True)
    patient = models.ForeignKey(
        "ai_caller.Patient",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="sms_conversations",
    )
    # Use apps.users.User (same as Patient), not django.contrib.auth.User.
    created_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="sms_conversations_created",
        null=True,
        blank=True,
    )
    updated_by = models.ForeignKey(
        "users.User",
        on_delete=models.SET_NULL,
        related_name="sms_conversations_updated",
        null=True,
        blank=True,
    )
    to_number = models.CharField(max_length=32, db_index=True)
    from_number = models.CharField(max_length=32)
    patient_name = models.CharField(max_length=120, blank=True)
    guardian_name = models.CharField(max_length=120, blank=True)
    clinic_name = models.CharField(max_length=200, blank=True)
    appointment_date = models.CharField(max_length=80, blank=True)
    appointment_time = models.CharField(max_length=80, blank=True)
    provider_name = models.CharField(max_length=120, blank=True)
    service_name = models.CharField(max_length=200, blank=True)
    agent_id = models.CharField(max_length=128, blank=True)
    transfer_number = models.CharField(max_length=32, blank=True)
    transfer_numbers = models.JSONField(default=list, blank=True)
    transfer_number_index = models.PositiveIntegerField(default=0)
    dial_call_sid = models.CharField(max_length=64, blank=True, default="")
    transfer_status = models.CharField(max_length=32, blank=True, default="")
    flow = models.CharField(max_length=16, default="adult")
    status = models.CharField(max_length=32, default="ongoing")
    step = models.CharField(max_length=32, default="identity")
    transcript = models.TextField(blank=True)
    last_inbound_sid = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.from_number} -> {self.to_number} ({self.status})"
