from django.apps import AppConfig


class AiCallerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.ai_caller"
    label = "ai_caller"
    verbose_name = "AI Caller"

    def ready(self):
        from apps.ai_caller import signals  # noqa: F401
