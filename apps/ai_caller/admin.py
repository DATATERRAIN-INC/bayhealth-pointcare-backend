from django.contrib import admin

from apps.ai_caller.models import Call, Patient, UploadedFile


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "file_name",
        "status",
        "uploaded_count",
        "failed_count",
        "created_at",
    )
    list_filter = ("status",)
    search_fields = ("file_name", "file_key", "error_message")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "first_name",
        "last_name",
        "phone_number",
        "live_agent_country_code",
        "live_agent_number",
        "doctor",
        "source",
        "upload",
        "dob",
        "created_at",
    )
    list_filter = ("source", "doctor")
    search_fields = (
        "first_name",
        "last_name",
        "doctor",
        "phone_number",
        "live_agent_number",
        "address",
    )
    ordering = ("-created_at",)
    raw_id_fields = ("upload",)


@admin.register(Call)
class CallAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "patient",
        "retell_call_id",
        "flow",
        "status",
        "to_number",
        "started_at",
        "ended_at",
        "created_at",
    )
    list_filter = ("status", "flow")
    search_fields = (
        "retell_call_id",
        "to_number",
        "patient__first_name",
        "patient__last_name",
    )
    ordering = ("-started_at",)
    raw_id_fields = ("patient",)
    readonly_fields = ("created_at", "updated_at")
