from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.notifications.models import Channel, Notification, Scope
from apps.notifications.mqtt import publish_app_notification, topics_for_user
from apps.notifications.serializers import NotificationSerializer
from common.pagination import CommonPagination
from common.responses import error_response, message_response


class MqttMessageAPIView(APIView):
    """List every MQTT topic this user should subscribe to.

    Path: ``notifications/get-topics/``.
    """

    def get(self, request):
        topics = topics_for_user(request.user)
        message = "Notifications retrieved." if topics else "No Messages found."
        return Response(
            {
                "status": "success",
                "result": {
                    "message": message,
                    "data": topics,
                },
            }
        )


class ScopeAPIView(APIView):
    """Create a SyncCast scope. Same role as Grid Social insert-scope."""

    def post(self, request):
        name = str(request.data.get("name") or "").strip()
        if not name:
            return error_response("Name is required.")
        scope, created = Scope.objects.get_or_create(
            name=name,
            defaults={
                "description": str(request.data.get("description") or "").strip(),
                "is_active": True,
            },
        )
        return Response(
            {
                "message": "Scope created successfully."
                if created
                else "Scope already exists.",
                "data": {
                    "id": scope.id,
                    "name": scope.name,
                    "description": scope.description,
                },
            },
            status=201 if created else 200,
        )


class ChannelAPIView(APIView):
    """Create a SyncCast channel under a scope. Same role as Grid Social insert-channel."""

    def post(self, request):
        name = str(request.data.get("name") or "").strip()
        scope_id = request.data.get("scope")
        if not name or not scope_id:
            return error_response("Name and scope are required.")
        scope = Scope.objects.filter(pk=scope_id).first()
        if scope is None:
            return error_response("Scope not found.", 404)
        channel, created = Channel.objects.get_or_create(
            scope=scope,
            name=name,
            defaults={
                "description": str(request.data.get("description") or "").strip(),
                "is_active": True,
            },
        )
        return Response(
            {
                "message": "Channel created successfully."
                if created
                else "Channel already exists.",
                "data": {
                    "id": channel.id,
                    "name": channel.name,
                    "scope": scope.id,
                    "description": channel.description,
                },
            },
            status=201 if created else 200,
        )


class NotificationViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = NotificationSerializer
    pagination_class = CommonPagination

    def get_queryset(self):
        queryset = Notification.objects.filter(user=self.request.user)
        event_type = (self.request.query_params.get("event_type") or "").strip()
        is_read = self.request.query_params.get("is_read")
        if event_type:
            queryset = queryset.filter(event_type=event_type)
        if is_read is not None and str(is_read).strip() != "":
            queryset = queryset.filter(
                is_read=str(is_read).strip().lower() in ("1", "true", "yes")
            )
        return queryset

    @action(detail=False, methods=["get"], url_path="summary")
    def summary(self, request):
        # Frontend polls this often; use it to refresh call status/transcript
        # when Retell webhooks cannot reach localhost.
        try:
            from apps.ai_caller.services import sync_in_progress_calls_from_retell

            sync_in_progress_calls_from_retell(user=request.user, limit=10)
        except Exception:
            pass
        queryset = Notification.objects.filter(user=request.user)
        return Response(
            {
                "all": queryset.count(),
                "unread": queryset.filter(is_read=False).count(),
                "read": queryset.filter(is_read=True).count(),
            }
        )

    @action(detail=True, methods=["post"], url_path="mark-read")
    def mark_read(self, request, pk=None):
        notification = self.get_object()
        if not notification.is_read:
            notification.is_read = True
            notification.updated_by = request.user
            notification.save(update_fields=["is_read", "updated_by"])
            publish_app_notification(notification.user_id, notification)
        return message_response("Notification marked as read.")

    @action(detail=False, methods=["post"], url_path="mark-all-read")
    def mark_all_read(self, request):
        updated = Notification.objects.filter(
            user=request.user, is_read=False
        ).update(is_read=True, updated_by=request.user)
        publish_app_notification(request.user.pk)
        return message_response(f"Marked {updated} notification(s) as read.")
