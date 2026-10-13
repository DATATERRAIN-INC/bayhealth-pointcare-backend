from django.contrib import admin

from apps.ai_pointcare.models import OptOut, Outreach, RenewalCallSession


@admin.register(Outreach)
class OutreachAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "phone_number",
        "status",
        "choice",
        "retell_call_id",
        "created_at",
    )
    list_filter = ("status", "choice")
    search_fields = ("name", "phone_number", "retell_call_id")
    ordering = ("-created_at",)
    readonly_fields = ("created_at", "updated_at")


@admin.register(RenewalCallSession)
class RenewalCallSessionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "patient_name",
        "phone_number",
        "current_node",
        "final_outcome",
        "retell_call_id",
        "created_at",
    )
    list_filter = ("current_node", "final_outcome", "dnc_requested")
    search_fields = ("patient_name", "phone_number", "retell_call_id")
    ordering = ("-created_at",)
    readonly_fields = ("created_at", "updated_at")


@admin.register(OptOut)
class OptOutAdmin(admin.ModelAdmin):
    list_display = ("id", "phone_number", "outreach", "created_at")
    search_fields = ("phone_number",)
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)
