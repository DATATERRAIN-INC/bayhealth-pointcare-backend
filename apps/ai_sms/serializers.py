from rest_framework import serializers

from apps.ai_sms.models import SmsConversation


class PlaceSmsConversationSerializer(serializers.Serializer):
    """Start adult SMS from an existing ai_caller Patient row."""

    id = serializers.IntegerField(min_value=1)
    agent_id = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )


class PlaceMinorSmsConversationSerializer(serializers.Serializer):
    """Start guardian/minor SMS from an existing ai_caller Patient row."""

    id = serializers.IntegerField(min_value=1)


class SmsConversationSerializer(serializers.ModelSerializer):
    """Full conversation row (detail / internal use)."""

    patient_id = serializers.IntegerField(read_only=True, allow_null=True)
    created_by = serializers.IntegerField(source="created_by_id", read_only=True, allow_null=True)
    updated_by = serializers.IntegerField(source="updated_by_id", read_only=True, allow_null=True)

    class Meta:
        model = SmsConversation
        fields = [
            "id",
            "chat_id",
            "patient_id",
            "created_by",
            "updated_by",
            "to_number",
            "from_number",
            "patient_name",
            "guardian_name",
            "clinic_name",
            "provider_name",
            "service_name",
            "flow",
            "status",
            "step",
            "transfer_number",
            "transfer_status",
            "transcript",
            "created_at",
            "updated_at",
        ]


class SmsConversationListSerializer(serializers.ModelSerializer):
    """Lean list payload for the conversations index API."""

    patient_id = serializers.IntegerField(read_only=True, allow_null=True)

    class Meta:
        model = SmsConversation
        fields = [
            "id",
            "chat_id",
            "patient_id",
            "patient_name",
            "status",
            "created_at",
        ]
