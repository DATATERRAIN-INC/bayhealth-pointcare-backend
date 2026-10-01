from django.contrib import admin

from apps.ai_caller.models import Call, CallerSettings, Patient, UploadedFile


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "file_name",
        "status",
        "uploaded_count",
        "failed_count",
        "created_at",
    )
    list_filter = ("status",)
    search_fields = ("file_name", "file_key", "error_message", "user__email")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)
    raw_id_fields = ("user", "created_by", "updated_by")


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "first_name",
        "last_name",
        "phone_number",
        "is_blocked",
        "live_agent_country_code",
        "live_agent_number",
        "doctor",
        "source",
        "upload",
        "dob",
        "created_at",
    )
    list_filter = ("source", "doctor", "is_blocked")
    search_fields = (
        "first_name",
        "last_name",
        "doctor",
        "phone_number",
        "live_agent_number",
        "address",
        "user__email",
    )
    ordering = ("-created_at",)
    raw_id_fields = ("upload", "user", "created_by", "updated_by")


@admin.register(Call)
class CallAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
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
        "user__email",
    )
    ordering = ("-started_at",)
    raw_id_fields = ("patient", "user", "created_by", "updated_by")
    readonly_fields = ("created_at", "updated_at")


@admin.register(CallerSettings)
class CallerSettingsAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "calls_enabled",
        "recording_enabled",
        "start_time",
        "end_time",
        "timezone",
        "max_calls_per_run",
        "updated_at",
    )
    readonly_fields = ("updated_at",)
    raw_id_fields = ("user", "created_by", "updated_by")
