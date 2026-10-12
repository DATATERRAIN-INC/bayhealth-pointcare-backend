"""SyncCast MQTT.

Topics are built from Scope and Channel rows. Nothing here names a scope or channel.
"""

import logging

from django.conf import settings
from synccast import synccast
from synccast.core.topic import SyncCastTopicBuilder

from apps.notifications.models import Channel, Scope

logger = logging.getLogger(__name__)

synccast.set_credentials(
    app_id=settings.MQTT_APP_ID,
    app_secret=settings.MQTT_APP_SECRET,
)


def build_topic(user_id, scope, channel):
    return (
        SyncCastTopicBuilder(synccast.app_id, scope.name)
        .for_user(user_id)
        .channel(channel.name)
        .build()
    )


def topics_for_user(user):
    """One topic for each active channel."""
    topics = []
    scopes = Scope.objects.filter(is_active=True).prefetch_related("channels")
    for scope in scopes:
        for channel in scope.channels.all():
            if channel.is_active:
                topics.append(build_topic(user.id, scope, channel))
    return topics


def send(user_id, channel, data):
    """Publish on one stored channel."""
    if not getattr(settings, "MQTT_THROTTLE", True):
        return
    synccast.notify.send_notification(
        user_id=str(user_id),
        data=data,
        scope=channel.scope.name,
        topic=build_topic(user_id, channel.scope, channel),
        sender_name="AI Caller",
        sender_role="system",
        platform="web",
    )


def publish(user_id, data, *, channel_names=None):
    """Send to active channels.

    Pass channel_names to limit the send to channels with those names.
    With no names, every active channel is used.
    """
    if not user_id:
        return
    channels = Channel.objects.filter(is_active=True).select_related("scope")
    if channel_names is not None:
        names = [name for name in channel_names if name]
        channels = channels.filter(name__in=names)
    for channel in channels:
        try:
            send(user_id, channel, data)
        except Exception:
            logger.exception(
                "MQTT send failed user=%s channel=%s",
                user_id,
                channel.name,
            )


def publish_app_notification(user_id, notification=None):
    """Publish the same body as GET /api/notifications/."""
    if not user_id:
        return
    from math import ceil

    from django.db.models import Count, Q

    from apps.notifications.models import Notification
    from apps.notifications.serializers import NotificationSerializer

    rows = Notification.objects.filter(user_id=user_id)
    total = rows.count()
    page_size = 10
    counts = rows.aggregate(
        read=Count("id", filter=Q(is_read=True)),
        unread=Count("id", filter=Q(is_read=False)),
    )
    results = [NotificationSerializer(notification).data] if notification is not None else []
    data = {
        "count": total,
        "total_pages": max(1, ceil(total / page_size)) if total else 1,
        "page": 1,
        "page_size": page_size,
        "read": counts["read"],
        "unread": counts["unread"],
        "results": results,
    }
    publish(user_id, data)


def publish_call_change(call, previous_status=None):
    """Tell channels named like the old or new call status."""
    user_id = getattr(call, "user_id", None)
    if not user_id:
        return
    publish(
        user_id,
        {
            "event": "call_queue_changed",
            "call_id": getattr(call, "pk", None),
            "status": getattr(call, "status", None),
            "previous_status": previous_status,
        },
        channel_names=[previous_status, getattr(call, "status", None)],
    )
