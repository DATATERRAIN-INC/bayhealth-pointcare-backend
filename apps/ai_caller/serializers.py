from datetime import date

from rest_framework import serializers

from apps.ai_caller.constants import ALLOWED_UPLOAD_EXTENSIONS
from apps.ai_caller.models import Call, CallerSettings, Patient
from common.s3 import build_s3_url


class PatientSerializer(serializers.ModelSerializer):
    full_name = serializers.CharField(read_only=True)

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
            "country_code",
            "phone_number",
            "live_agent_country_code",
            "live_agent_number",
            "is_blocked",
            "source",
            "upload_file_key",
            "created_at",
            "updated_at",
        )
        read_only_fields = (
            "id",
            "full_name",
            "source",
            "upload_file_key",
            "created_at",
            "updated_at",
        )

    def to_representation(self, instance):
        data = super().to_representation(instance)
        data["upload_file_key"] = build_s3_url(instance.upload_file_key)
        return data

    def validate_dob(self, value):
        if value and value > date.today():
            raise serializers.ValidationError("Date of birth can't be in the future.")
        return value

    def validate_country_code(self, value):
        return self._normalize_country_code(value)

    def validate_live_agent_country_code(self, value):
        return self._normalize_country_code(value)

    def _normalize_country_code(self, value):
        code = (value or "").strip()
        if not code:
            return ""
        if not code.startswith("+"):
            code = f"+{code.lstrip('+')}"
        return code


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
    duration_seconds = serializers.IntegerField(read_only=True)
    has_transcript = serializers.BooleanField(read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = Call
        fields = (
            "id",
            "patient",
            "patient_name",
            "retell_call_id",
            "flow",
            "status",
            "from_number",
            "to_number",
            "agent_id",
            "transfer_number",
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


class CallerSettingsSerializer(serializers.ModelSerializer):
    window_summary = serializers.SerializerMethodField()

    class Meta:
        model = CallerSettings
        fields = (
            "calls_enabled",
            "recording_enabled",
            "start_time",
            "end_time",
            "timezone",
            "max_calls_per_run",
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
        return attrs
