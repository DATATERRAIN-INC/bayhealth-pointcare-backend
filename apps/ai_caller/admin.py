from django.contrib import admin

from apps.ai_caller.models import (
    Call,
    CallerSettings,
    LiveAgentNumber,
    Patient,
    ScheduledOutreach,
    UploadedFile,
)


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
        "doctor",
        "service_name",
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
        "service_name",
        "phone_number",
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
        "decline_reason",
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


class LiveAgentNumberInline(admin.TabularInline):
    model = LiveAgentNumber
    extra = 1
    fields = ("country_code", "phone_number", "label", "is_active")


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
        "call_trigger_count",
        "sms_trigger_after_calls",
        "reminder_timeframe_hours",
        "text_sms_enabled",
        "updated_at",
    )
    readonly_fields = ("updated_at",)
    raw_id_fields = ("user", "created_by", "updated_by")
    inlines = [LiveAgentNumberInline]


@admin.register(ScheduledOutreach)
class ScheduledOutreachAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "patient",
        "kind",
        "status",
        "scheduled_at",
        "raw_time_text",
        "source_call",
        "triggered_call",
        "created_at",
    )
    list_filter = ("kind", "status")
    search_fields = (
        "patient__first_name",
        "patient__last_name",
        "patient__phone_number",
        "raw_time_text",
        "user__email",
    )
    ordering = ("scheduled_at", "id")
    raw_id_fields = ("user", "patient", "source_call", "triggered_call")
    readonly_fields = ("created_at", "updated_at")


@admin.register(LiveAgentNumber)
class LiveAgentNumberAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "caller_settings",
        "country_code",
        "phone_number",
        "label",
        "is_active",
        "created_at",
    )
    list_filter = ("is_active",)
    search_fields = ("phone_number", "label", "caller_settings__user__email")
    raw_id_fields = ("caller_settings",)
    readonly_fields = ("created_at", "updated_at")
