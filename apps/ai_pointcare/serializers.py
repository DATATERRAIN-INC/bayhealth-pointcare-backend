from rest_framework import serializers


class PlaceMediCalCallSerializer(serializers.Serializer):
    phone_number = serializers.CharField(max_length=32)
    name = serializers.CharField(max_length=120)
    agent_number = serializers.CharField(max_length=32)
    address = serializers.CharField(
        max_length=255, required=False, allow_blank=True, default=""
    )
