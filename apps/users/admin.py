from django.contrib import admin

from apps.users.models import User


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
