from django.contrib import admin

from apps.users.models import EmailTemplate, PasswordResetRequest, User


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = (
        "email",
        "first_name",
        "last_name",
        "cognito_id",
        "cognito_username",
        "is_active",
        "is_confirmed",
        "created_at",
    )
    search_fields = ("email", "first_name", "last_name", "cognito_id", "cognito_username", "phone_number")
    readonly_fields = ("cognito_id", "cognito_username", "created_at", "updated_at")
    list_filter = ("is_active", "is_confirmed")


@admin.register(EmailTemplate)
class EmailTemplateAdmin(admin.ModelAdmin):
    list_display = ("name", "subject", "updated_at")
    search_fields = ("name", "subject")


@admin.register(PasswordResetRequest)
class PasswordResetRequestAdmin(admin.ModelAdmin):
    list_display = ("user", "token", "expires_at", "is_used", "created_at")
    search_fields = ("user__email", "token")
    readonly_fields = ("reset_code", "token", "created_at", "used_at")
