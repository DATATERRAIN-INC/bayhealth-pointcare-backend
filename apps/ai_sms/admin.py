from django.contrib import admin

from apps.ai_sms.models import SmsConversation


@admin.register(SmsConversation)
class SmsConversationAdmin(admin.ModelAdmin):
    list_display = (
        "to_number",
        "from_number",
        "flow",
        "step",
        "status",
        "service_name",
        "chat_id",
        "created_at",
    )
    list_filter = ("flow", "status", "step")
    search_fields = ("to_number", "chat_id", "name", "patient_name", "guardian_name")
