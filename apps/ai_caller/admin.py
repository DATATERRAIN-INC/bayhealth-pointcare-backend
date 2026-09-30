from django.contrib import admin

from apps.ai_caller.models import Patient, UploadedFile


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
        "name",
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
    search_fields = ("name", "doctor", "phone_number", "live_agent_number", "address")
    ordering = ("-created_at",)
    raw_id_fields = ("upload",)
