from django.db import models

from apps.ai_pointcare.flow_definition import (
    FINAL_OUTCOMES,
    NODE_OPENING,
    RENEWAL_STATUS_VALUES,
)

CHOICE_CALL_NOW = "call_now"


class Outreach(models.Model):
    STATUS_PENDING = "pending"
    STATUS_CALLING = "calling"
    STATUS_CALL_FAILED = "call_failed"
    STATUS_COMPLETED = "completed"
    STATUS_OPTED_OUT = "opted_out"
    STATUS_CHOICES = (
        (STATUS_PENDING, "Pending"),
        (STATUS_CALLING, "Calling"),
        (STATUS_CALL_FAILED, "Call failed"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_OPTED_OUT, "Opted out"),
    )

    name = models.CharField(max_length=120)
    phone_number = models.CharField(max_length=32, db_index=True)
    phone_raw = models.CharField(max_length=64, blank=True, default="")
    address = models.CharField(max_length=255, blank=True, default="")
    agent_number = models.CharField(max_length=32, blank=True, default="")
    status = models.CharField(
        max_length=24, choices=STATUS_CHOICES, default=STATUS_PENDING
    )
    choice = models.CharField(max_length=24, blank=True, default="")
    retell_call_id = models.CharField(max_length=120, blank=True, default="")
    retell_agent_id = models.CharField(max_length=120, blank=True, default="")
    call_from_number = models.CharField(max_length=32, blank=True, default="")
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.name} {self.phone_number} ({self.status})"


class RenewalCallSession(models.Model):
    """Structured state for one Medi-Cal renewal Retell call."""

    RENEWAL_CHOICES = [("", "Unset")] + [(v, v) for v in RENEWAL_STATUS_VALUES]
    OUTCOME_CHOICES = [("", "Unset")] + [(v, v) for v in FINAL_OUTCOMES]

    outreach = models.ForeignKey(
        Outreach,
        on_delete=models.CASCADE,
        related_name="call_sessions",
        null=True,
        blank=True,
    )
    retell_call_id = models.CharField(max_length=120, unique=True, db_index=True)
    patient_name = models.CharField(max_length=120)
    phone_number = models.CharField(max_length=32, db_index=True)
    agent_number = models.CharField(max_length=32, blank=True, default="")
    current_node = models.CharField(max_length=64, default=NODE_OPENING)

    patient_verified = models.BooleanField(default=False)
    address = models.CharField(max_length=255, blank=True, default="")
    address_confirmed = models.BooleanField(null=True, blank=True)
    phone_confirmed = models.BooleanField(null=True, blank=True)
    email = models.CharField(max_length=255, blank=True, default="")

    renewal_status = models.CharField(
        max_length=32, choices=RENEWAL_CHOICES, blank=True, default=""
    )
    household_changed = models.BooleanField(null=True, blank=True)
    household_change_details = models.TextField(blank=True, default="")
    income_changed = models.BooleanField(null=True, blank=True)
    income_change_details = models.TextField(blank=True, default="")
    support_status = models.CharField(max_length=255, blank=True, default="")
    document_ready = models.BooleanField(null=True, blank=True)

    assistance_requested = models.BooleanField(null=True, blank=True)
    assistance_type = models.CharField(max_length=64, blank=True, default="")
    additional_benefits_interest = models.BooleanField(null=True, blank=True)

    appointment_needed = models.BooleanField(null=True, blank=True)
    appointment_scheduled = models.BooleanField(default=False)
    appointment_details = models.TextField(blank=True, default="")

    final_outcome = models.CharField(
        max_length=64, choices=OUTCOME_CHOICES, blank=True, default=""
    )
    callback_requested = models.BooleanField(default=False)
    callback_details = models.TextField(blank=True, default="")
    dnc_requested = models.BooleanField(default=False)
    transfer_required = models.BooleanField(default=False)
    transfer_status = models.CharField(max_length=64, blank=True, default="")

    state_payload = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.patient_name} {self.retell_call_id} [{self.current_node}]"


class OptOut(models.Model):
    phone_number = models.CharField(max_length=32, unique=True)
    outreach = models.ForeignKey(
        Outreach,
        on_delete=models.SET_NULL,
        related_name="opt_outs",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return self.phone_number
