from django.contrib import admin

from apps.notifications.models import Notification


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ("id", "event_type", "title", "is_read", "created_at")
    list_filter = ("event_type", "is_read")
    search_fields = ("title", "message")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)
