from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from apps.notifications.mqtt import publish_app_notification, publish_call_change


def _channel(name):
    return SimpleNamespace(name=name, scope=SimpleNamespace(name="scope"))


class PublishCallChangeTests(SimpleTestCase):
    def test_uses_channels_named_like_the_status(self):
        call = SimpleNamespace(pk=1, user_id=2, status="in_progress")
        rows = [_channel("queued"), _channel("in_progress")]
        with patch("apps.notifications.mqtt.Channel.objects") as objects:
            objects.filter.return_value.select_related.return_value.filter.return_value = rows
            with patch("apps.notifications.mqtt.send") as send:
                publish_call_change(call, previous_status="queued")
        names = objects.filter.return_value.select_related.return_value.filter.call_args.kwargs[
            "name__in"
        ]
        self.assertEqual(set(names), {"queued", "in_progress"})
        self.assertEqual(send.call_count, 2)


class PublishAppNotificationTests(SimpleTestCase):
    def test_sends_the_notification_list_shape(self):
        notification = SimpleNamespace(id=4)
        item = {
            "id": 4,
            "event_type": "patient_upload",
            "title": "Patient upload finished",
            "message": "Uploaded 10 patient(s).",
            "metadata": {},
            "is_read": False,
            "created_at": "2026-10-12T10:00:00Z",
        }
        with patch("apps.notifications.models.Notification.objects.filter") as rows:
            rows.return_value.count.return_value = 6
            rows.return_value.aggregate.return_value = {"read": 4, "unread": 2}
            with patch(
                "apps.notifications.serializers.NotificationSerializer"
            ) as serializer:
                serializer.return_value.data = item
                with patch("apps.notifications.mqtt.Channel.objects") as objects:
                    objects.filter.return_value.select_related.return_value = [
                        _channel("notifications")
                    ]
                    with patch("apps.notifications.mqtt.send") as send:
                        publish_app_notification(5, notification)
        body = send.call_args.args[2]
        self.assertEqual(body["count"], 6)
        self.assertEqual(body["page"], 1)
        self.assertEqual(body["page_size"], 10)
        self.assertEqual(body["read"], 4)
        self.assertEqual(body["unread"], 2)
        self.assertEqual(body["results"], [item])
        self.assertNotIn("mqtt", body)
