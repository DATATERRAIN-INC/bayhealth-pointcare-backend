from django.contrib import admin

from apps.ai_sms.models import SmsConversation


@admin.register(SmsConversation)
class SmsConversationAdmin(admin.ModelAdmin):
    list_display = (
        "chat_id",
        "patient",
        "to_number",
        "from_number",
        "provider_name",
        "flow",
        "step",
        "status",
        "service_name",
        "created_at",
    )
    list_filter = ("flow", "status", "step")
    search_fields = (
        "to_number",
        "chat_id",
        "patient_name",
        "guardian_name",
        "provider_name",
    )
    raw_id_fields = ("patient",)
