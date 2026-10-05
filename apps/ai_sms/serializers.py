from rest_framework import serializers


class PlaceSmsConversationSerializer(serializers.Serializer):
    """Start adult SMS from an existing ai_caller Patient row."""

    id = serializers.IntegerField(min_value=1)
    agent_id = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )


class PlaceMinorSmsConversationSerializer(serializers.Serializer):
    """Start guardian/minor SMS from an existing ai_caller Patient row."""

    id = serializers.IntegerField(min_value=1)
    guardian_name = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    appointment_date = serializers.CharField(
        max_length=80, required=False, allow_blank=True, default=""
    )
    appointment_time = serializers.CharField(
        max_length=80, required=False, allow_blank=True, default=""
    )
