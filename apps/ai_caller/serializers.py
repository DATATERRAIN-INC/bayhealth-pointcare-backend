from datetime import date

from django.conf import settings
from rest_framework import serializers

from apps.ai_caller.constants import ALLOWED_UPLOAD_EXTENSIONS
from apps.ai_caller.models import Patient
from common.s3 import build_s3_url


class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = (
            "id",
            "name",
            "address",
            "dob",
            "doctor",
            "country_code",
            "phone_number",
            "live_agent_country_code",
            "live_agent_number",
            "source",
            "upload_file_key",
            "created_at",
            "updated_at",
        )
        read_only_fields = (
            "id",
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
            return settings.DEFAULT_COUNTRY_CODE
        if not code.startswith("+"):
            code = f"+{code.lstrip('+')}"
        return code

    def to_internal_value(self, data):
        if isinstance(data, dict):
            data = {**data}
            if not (data.get("country_code") or "").strip():
                data["country_code"] = settings.DEFAULT_COUNTRY_CODE
            if not (data.get("live_agent_country_code") or "").strip():
                data["live_agent_country_code"] = settings.DEFAULT_COUNTRY_CODE
        return super().to_internal_value(data)


class PatientUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

    def validate_file(self, value):
        filename = getattr(value, "name", "") or ""
        extension = f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ""
        if extension not in ALLOWED_UPLOAD_EXTENSIONS:
            raise serializers.ValidationError("Unsupported file type. Use .xlsx or .csv.")
        return value


class PlaceRetellCareCallSerializer(serializers.Serializer):
    phone_number = serializers.CharField(max_length=32)
    name = serializers.CharField(max_length=120, required=False, allow_blank=True, default="")
    patient_name = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    service_name = serializers.CharField(
        max_length=200, required=False, allow_blank=True, default=""
    )
    address_on_file = serializers.CharField(
        max_length=300, required=False, allow_blank=True, default=""
    )
    insurance_name = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    agent_id = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    transfer_number = serializers.CharField(
        max_length=32, required=False, allow_blank=True, default=""
    )


class PlaceRetellGuardianCallSerializer(serializers.Serializer):
    phone_number = serializers.CharField(max_length=32)
    patient_name = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    guardian_name = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    insurance_name = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    measure_name = serializers.CharField(
        max_length=200, required=False, allow_blank=True, default=""
    )
    service_name = serializers.CharField(
        max_length=200, required=False, allow_blank=True, default=""
    )
    address_on_file = serializers.CharField(
        max_length=300, required=False, allow_blank=True, default=""
    )
    clinic_name = serializers.CharField(
        max_length=200, required=False, allow_blank=True, default=""
    )
    appointment_date = serializers.CharField(
        max_length=80, required=False, allow_blank=True, default=""
    )
    appointment_time = serializers.CharField(
        max_length=80, required=False, allow_blank=True, default=""
    )
    provider_name = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    transfer_number = serializers.CharField(max_length=32)
