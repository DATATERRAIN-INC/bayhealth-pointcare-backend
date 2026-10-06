import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

SECRET_KEY = os.getenv("SECRET_KEY", "django-insecure-dev-key")
DEBUG = True
ALLOWED_HOSTS = ["*"]
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:4200",
    "https://bach.gapincare.socialroots-dev.net",
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "apps.ai_caller.apps.AiCallerConfig",
    "apps.ai_sms.apps.AiSmsConfig",
    "apps.users.apps.UsersConfig",
    "apps.notifications.apps.NotificationsConfig",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ["DB_NAME"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "HOST": os.getenv("DB_HOST", "localhost"),
        "PORT": os.getenv("DB_PORT", "5432"),
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "apps.users.authentication.CognitoBearerAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "EXCEPTION_HANDLER": "common.exceptions.custom_exception_handler",
    "UNAUTHENTICATED_USER": None,
}

DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "+1")
RETELL_API_KEY = os.getenv("RETELL_API_KEY", "")
RETELL_FROM_NUMBER = os.getenv("RETELL_FROM_NUMBER", "")
RETELL_AGENT_ID = os.getenv("RETELL_AGENT_ID", "")
RETELL_MODEL = os.getenv("RETELL_MODEL", "")
RETELL_VOICE = os.getenv("RETELL_VOICE", "")
RETELL_MAX_DURATION_MINUTES = int(os.getenv("RETELL_MAX_DURATION_MINUTES") or 8)
RETELL_SSL_VERIFY = os.getenv("RETELL_SSL_VERIFY", "true").lower() in ("1", "true", "yes")
RETELL_TRANSFER_NUMBER = os.getenv("RETELL_TRANSFER_NUMBER", "")
# With call screening on, prefer noise-cancellation (not background-speech mode).
# Options: noise-cancellation | noise-and-background-speech-cancellation | no-denoise
RETELL_DENOISING_MODE = os.getenv("RETELL_DENOISING_MODE", "noise-cancellation")
RETELL_INTERRUPTION_SENSITIVITY = os.getenv("RETELL_INTERRUPTION_SENSITIVITY", "0.6")
RETELL_RESPONSIVENESS = os.getenv("RETELL_RESPONSIVENESS", "0.85")
# Comma-separated Retell locales (do not use legacy "multi").
RETELL_LANGUAGES = os.getenv(
    "RETELL_LANGUAGES",
    "en-US,es-ES,hi-IN,zh-CN,vi-VN",
)
AWS_S3_BASE_URL = os.getenv("AWS_S3_BASE_URL", "").rstrip("/")
AWS_MEDIA_FOLDER = os.getenv("AWS_MEDIA_FOLDER", "bayhealth")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_STORAGE_BUCKET_NAME = os.environ.get("AWS_STORAGE_BUCKET_NAME", "")
AWS_S3_REGION_NAME = os.environ.get("AWS_S3_REGION_NAME", "us-east-1")
AWS_REGION = os.environ.get("COGNITO_AWS_REGION", "AWS_REGION") or AWS_S3_REGION_NAME
COGNITO_USER_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID")
COGNITO_APP_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "COGNITO_APP_CLIENT_ID")
COGNITO_APP_CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET", "COGNITO_APP_CLIENT_SECRET")
AWS_S3_BASE_URL = os.environ.get("AWS_S3_BASE_URL", "").rstrip("/")
AWS_MEDIA_FOLDER = os.environ.get("AWS_MEDIA_FOLDER", "bayhealth")

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
DEFAULT_FROM_EMAIL = os.environ.get("DEFAULT_FROM_EMAIL", "")
EMAIL_TITLE_CARD_NAME = os.environ.get("EMAIL_TITLE_CARD_NAME", "Gap In Care")
EMAIL_RESTRICTION = os.environ.get("EMAIL_RESTRICTION", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
PASSWORD_RESET_URL = os.environ.get("PASSWORD_RESET_URL", "")

# Celery — automated AI caller
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = os.getenv("CELERY_TIMEZONE", "UTC")
CELERY_BEAT_SCHEDULE = {
    "ai-caller-process-outbound-calls": {
        "task": "ai_caller.process_outbound_calls",
        "schedule": timedelta(
            seconds=float(os.getenv("CELERY_OUTBOUND_INTERVAL_SECONDS", "60"))
        ),
    },
    "ai-caller-sync-in-progress-calls": {
        "task": "ai_caller.sync_in_progress_calls",
        "schedule": timedelta(
            seconds=float(os.getenv("CELERY_SYNC_CALLS_INTERVAL_SECONDS", "45"))
        ),
    },
}

# Twilio warm-transfer (AI + live-agent H2H recording / merge)
BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "") or os.getenv(
    "TOLLFREE_TWILIO_PHONE_NUMBER", ""
)
TWILIO_CARECALL_FROM_NUMBER = os.getenv(
    "TWILIO_CARECALL_FROM_NUMBER", ""
) or TWILIO_PHONE_NUMBER
TWILIO_MINOR_SMS_FROM_NUMBER = os.getenv("TWILIO_MINOR_SMS_FROM_NUMBER", "")
TWILIO_VOICE = os.getenv("TWILIO_VOICE", "Joanna")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CALL_TRANSCRIPTION_MODEL = os.getenv("CALL_TRANSCRIPTION_MODEL", "")
WARM_TRANSFER_ENABLED = os.getenv("WARM_TRANSFER_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Dialer / outbound call decision logs (separate folder)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "dialer": {
            "format": "[{asctime}] {levelname} {message}",
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "dialer_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "dialer.log"),
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 5,
            "formatter": "dialer",
        },
    },
    "loggers": {
        "ai_caller.dialer": {
            "handlers": ["dialer_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
