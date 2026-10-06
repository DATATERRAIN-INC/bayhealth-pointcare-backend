from datetime import date

from django.utils import timezone
from rest_framework import serializers

from apps.ai_caller.constants import ALLOWED_UPLOAD_EXTENSIONS
from apps.ai_caller.models import (
    Call,
    CallerSettings,
    LiveAgentNumber,
    Patient,
    ScheduledOutreach,
)
from common.s3 import build_s3_url


def _normalize_country_code(value):
    code = (value or "").strip()
    if not code:
        return ""
    if not code.startswith("+"):
        code = f"+{code.lstrip('+')}"
    return code


class PatientSerializer(serializers.ModelSerializer):
    full_name = serializers.CharField(read_only=True)
    call_status = serializers.SerializerMethodField()
    call_id = serializers.SerializerMethodField()
    retell_call_id = serializers.SerializerMethodField()
    duration_seconds = serializers.SerializerMethodField()

    class Meta:
        model = Patient
        fields = (
            "id",
            "first_name",
            "last_name",
            "full_name",
            "address",
            "dob",
            "doctor",
            "service_name",
            "country_code",
            "phone_number",
            "is_blocked",
            "source",
            "upload",
            "upload_file_key",
            "call_status",
            "call_id",
            "retell_call_id",
            "duration_seconds",
            "created_at",
            "updated_at",
        )
        read_only_fields = (
            "id",
            "full_name",
            "source",
            "upload",
            "upload_file_key",
            "call_status",
            "call_id",
            "retell_call_id",
            "duration_seconds",
            "created_at",
            "updated_at",
        )

    def get_call_id(self, obj):
        value = getattr(obj, "_latest_call_id", None)
        if value is not None:
            return value
        latest = (
            Call.objects.filter(patient=obj)
            .order_by("-started_at", "-id")
            .values_list("id", flat=True)
            .first()
        )
        return latest

    def get_retell_call_id(self, obj):
        value = getattr(obj, "_latest_retell_call_id", None)
        if value is not None:
            return value or None
        latest = (
            Call.objects.filter(patient=obj)
            .order_by("-started_at", "-id")
            .values_list("retell_call_id", flat=True)
            .first()
        )
        return latest or None

    def get_duration_seconds(self, obj):
        started_at = getattr(obj, "_latest_call_started_at", None)
        ended_at = getattr(obj, "_latest_call_ended_at", None)
        status = getattr(obj, "_latest_call_status", None)
        if started_at is None and not hasattr(obj, "_latest_call_id"):
            latest = (
                Call.objects.filter(patient=obj)
                .order_by("-started_at", "-id")
                .only("started_at", "ended_at", "status")
                .first()
            )
            if not latest:
                return None
            started_at = latest.started_at
            ended_at = latest.ended_at
            status = latest.status
        if not started_at:
            return None
        end = ended_at
        if end is None and status == Call.Status.IN_PROGRESS:
            end = timezone.now()
        if end is None:
            return None
        return max(0, int((end - started_at).total_seconds()))

    def get_call_status(self, obj):
        """
        Current dial status for this patient:
        in_progress > paused > queued > latest call status > null if blocked.
        """
        if getattr(obj, "_has_in_progress", None):
            return Call.Status.IN_PROGRESS
        annotated = getattr(obj, "_latest_call_status", None)
        if annotated in {
            Call.Status.IN_PROGRESS,
            Call.Status.PAUSED,
            Call.Status.QUEUED,
        }:
            return annotated
        if annotated:
            # Prefer open dial-queue rows over older completed/not_attended.
            if Call.objects.filter(
                patient=obj, status=Call.Status.PAUSED
            ).exists():
                return Call.Status.PAUSED
            if Call.objects.filter(
                patient=obj, status=Call.Status.QUEUED
            ).exists():
                return Call.Status.QUEUED
            return annotated
        if getattr(obj, "_has_in_progress", None) is False and annotated is None:
            return None if obj.is_blocked else Call.Status.QUEUED

        if Call.objects.filter(patient=obj, status=Call.Status.IN_PROGRESS).exists():
            return Call.Status.IN_PROGRESS
        if Call.objects.filter(patient=obj, status=Call.Status.PAUSED).exists():
            return Call.Status.PAUSED
        if Call.objects.filter(patient=obj, status=Call.Status.QUEUED).exists():
            return Call.Status.QUEUED
        latest = (
            Call.objects.filter(patient=obj)
            .order_by("-started_at", "-id")
            .values_list("status", flat=True)
            .first()
        )
        if latest:
            return latest
        return None if obj.is_blocked else Call.Status.QUEUED

    def to_representation(self, instance):
        data = super().to_representation(instance)
        data["upload_file_key"] = build_s3_url(instance.upload_file_key)
        return data

    def validate_dob(self, value):
        from common.excel import parse_patient_dob

        # DRF DateField may already give a date; still re-check bounds.
        if isinstance(value, str):
            parsed = parse_patient_dob(value)
            if not parsed:
                raise serializers.ValidationError(
                    "Enter a valid date of birth (e.g. YYYY-MM-DD, MM/DD/YYYY)."
                )
            value = parsed
        if value and value > date.today():
            raise serializers.ValidationError("Date of birth can't be in the future.")
        if value and value.year < 1900:
            raise serializers.ValidationError("Date of birth year must be 1900 or later.")
        return value

    def to_internal_value(self, data):
        if isinstance(data, dict) and "dob" in data and data.get("dob") not in (None, ""):
            from common.excel import parse_patient_dob

            parsed = parse_patient_dob(data.get("dob"))
            if parsed:
                data = {**data, "dob": parsed.isoformat()}
        return super().to_internal_value(data)

    def validate_country_code(self, value):
        return _normalize_country_code(value)

    def validate_service_name(self, value):
        name = (value or "").strip()
        if not name:
            raise serializers.ValidationError("Service name is required.")
        return name


class PatientUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

    def validate_file(self, value):
        filename = getattr(value, "name", "") or ""
        extension = f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ""
        if extension not in ALLOWED_UPLOAD_EXTENSIONS:
            raise serializers.ValidationError("Unsupported file type. Use .xlsx or .csv.")
        return value


class PlaceOutboundCallSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)


class CallSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source="patient.full_name", read_only=True)
    # Same value set on patient create as service_name (why we are calling).
    reason = serializers.CharField(source="patient.service_name", read_only=True)
    duration_seconds = serializers.IntegerField(read_only=True)
    has_transcript = serializers.BooleanField(read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = Call
        fields = (
            "id",
            "patient",
            "patient_name",
            "reason",
            "retell_call_id",
            "flow",
            "status",
            "is_paused",
            "from_number",
            "to_number",
            "agent_id",
            "transfer_number",
            "decline_reason",
            "started_at",
            "ended_at",
            "duration_seconds",
            "has_transcript",
            "message_count",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields

    def get_message_count(self, obj):
        return len(obj.transcript or [])


class CallPauseSerializer(serializers.Serializer):
    paused = serializers.BooleanField(required=True)


class ScheduledOutreachSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source="patient.full_name", read_only=True)
    patient_phone = serializers.SerializerMethodField()
    kind_label = serializers.CharField(source="get_kind_display", read_only=True)
    status_label = serializers.CharField(source="get_status_display", read_only=True)

    class Meta:
        model = ScheduledOutreach
        fields = (
            "id",
            "patient",
            "patient_name",
            "patient_phone",
            "source_call",
            "triggered_call",
            "kind",
            "kind_label",
            "status",
            "status_label",
            "scheduled_at",
            "raw_time_text",
            "error_message",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields

    def get_patient_phone(self, obj):
        patient = obj.patient
        if not patient:
            return ""
        return f"{patient.country_code or ''}{patient.phone_number or ''}".strip()


class LiveAgentNumberSerializer(serializers.ModelSerializer):
    class Meta:
        model = LiveAgentNumber
        fields = (
            "id",
            "country_code",
            "phone_number",
            "label",
            "is_active",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "created_at", "updated_at")

    def validate_country_code(self, value):
        return _normalize_country_code(value)

    def validate_phone_number(self, value):
        number = (value or "").strip()
        if not number:
            raise serializers.ValidationError("Phone number is required.")
        return number


class CallerSettingsSerializer(serializers.ModelSerializer):
    window_summary = serializers.SerializerMethodField()
    live_agent_numbers = LiveAgentNumberSerializer(many=True, required=False)

    class Meta:
        model = CallerSettings
        fields = (
            "calls_enabled",
            "recording_enabled",
            "text_sms_enabled",
            "start_time",
            "end_time",
            "timezone",
            "max_calls_per_run",
            "call_trigger_count",
            "reminder_timeframe_hours",
            "live_agent_numbers",
            "window_summary",
            "updated_at",
        )
        read_only_fields = ("window_summary", "updated_at")

    def get_window_summary(self, obj):
        from apps.ai_caller.services import build_calling_window_summary

        return build_calling_window_summary(obj)

    def validate_timezone(self, value):
        tz_name = (value or "").strip()
        if not tz_name:
            raise serializers.ValidationError("Timezone is required.")
        try:
            from apps.ai_caller.services import resolve_timezone

            resolve_timezone(tz_name)
        except Exception:
            raise serializers.ValidationError("Invalid timezone.")
        return tz_name

    def validate(self, attrs):
        start = attrs.get("start_time", getattr(self.instance, "start_time", None))
        end = attrs.get("end_time", getattr(self.instance, "end_time", None))
        if start and end and start >= end:
            raise serializers.ValidationError(
                {"end_time": "End time must be after start time."}
            )
        trigger = attrs.get(
            "call_trigger_count",
            getattr(self.instance, "call_trigger_count", None),
        )
        if trigger is not None and int(trigger) < 1:
            raise serializers.ValidationError(
                {"call_trigger_count": "Must be at least 1."}
            )
        timeframe = attrs.get(
            "reminder_timeframe_hours",
            getattr(self.instance, "reminder_timeframe_hours", None),
        )
        if timeframe is not None:
            hours = int(timeframe)
            if hours < 1:
                raise serializers.ValidationError(
                    {"reminder_timeframe_hours": "Must be at least 1 hour."}
                )
            if hours > 24 * 30:
                raise serializers.ValidationError(
                    {
                        "reminder_timeframe_hours": (
                            "Must be 720 hours (30 days) or less."
                        )
                    }
                )
        return attrs

    def update(self, instance, validated_data):
        numbers_data = validated_data.pop("live_agent_numbers", None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()

        if numbers_data is not None:
            # Full replace of the list when provided in PATCH/PUT.
            instance.live_agent_numbers.all().delete()
            LiveAgentNumber.objects.bulk_create(
                [
                    LiveAgentNumber(
                        caller_settings=instance,
                        country_code=item.get("country_code") or "",
                        phone_number=item.get("phone_number") or "",
                        label=item.get("label") or "",
                        is_active=item.get("is_active", True),
                    )
                    for item in numbers_data
                ]
            )
        return instance
