from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.notifications.models import Notification
from apps.notifications.serializers import NotificationSerializer
from common.pagination import CommonPagination
from common.responses import message_response


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
        return message_response("Notification marked as read.")

    @action(detail=False, methods=["post"], url_path="mark-all-read")
    def mark_all_read(self, request):
        updated = Notification.objects.filter(
            user=request.user, is_read=False
        ).update(is_read=True, updated_by=request.user)
        return message_response(f"Marked {updated} notification(s) as read.")
