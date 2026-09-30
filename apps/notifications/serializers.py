from rest_framework import serializers

from apps.notifications.models import Notification


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = (
            "id",
            "event_type",
            "title",
            "message",
            "metadata",
            "is_read",
            "created_at",
        )
        read_only_fields = (
            "id",
            "event_type",
            "title",
            "message",
            "metadata",
            "created_at",
        )
