import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=True)

SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "django-insecure-tku-8yj7t*vhnv@e@x15(9-l^c!$mfent!6v6vz#(sak#=pw)9",
)

DEBUG = os.environ.get("DEBUG", "true").strip().lower() in ("1", "true", "yes")

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("ALLOWED_HOSTS", "").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "apps.ai_caller.apps.AiCallerConfig",
    "apps.users.apps.UsersConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

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

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ["DB_NAME"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    "EXCEPTION_HANDLER": "common.exceptions.custom_exception_handler",
    "UNAUTHENTICATED_USER": None,
}

DEFAULT_COUNTRY_CODE = os.environ.get("DEFAULT_COUNTRY_CODE", "+1")
RETELL_API_KEY = os.environ.get("RETELL_API_KEY", "")
RETELL_FROM_NUMBER = os.environ.get("RETELL_FROM_NUMBER", "")
RETELL_AGENT_ID = os.environ.get("RETELL_AGENT_ID", "")
RETELL_MODEL = os.environ.get("RETELL_MODEL", "")
RETELL_VOICE = os.environ.get("RETELL_VOICE", "")
RETELL_MAX_DURATION_MINUTES = int(os.environ.get("RETELL_MAX_DURATION_MINUTES") or 8)
RETELL_SSL_VERIFY = os.environ.get("RETELL_SSL_VERIFY", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
RETELL_TRANSFER_NUMBER = os.environ.get("RETELL_TRANSFER_NUMBER", "")


def _env(*names):
    for name in names:
        value = os.environ.get(name, "")
        if isinstance(value, str):
            value = value.strip().strip("'\"").strip()
        if value:
            return value
    return ""


AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_STORAGE_BUCKET_NAME = os.environ.get("AWS_STORAGE_BUCKET_NAME", "")
AWS_S3_REGION_NAME = os.environ.get("AWS_S3_REGION_NAME", "us-east-1")
AWS_REGION = _env("COGNITO_AWS_REGION", "AWS_REGION") or AWS_S3_REGION_NAME
COGNITO_USER_POOL_ID = _env("COGNITO_USER_POOL_ID")
COGNITO_APP_CLIENT_ID = _env("COGNITO_CLIENT_ID", "COGNITO_APP_CLIENT_ID")
COGNITO_APP_CLIENT_SECRET = _env("COGNITO_CLIENT_SECRET", "COGNITO_APP_CLIENT_SECRET")
AWS_S3_BASE_URL = os.environ.get("AWS_S3_BASE_URL", "").rstrip("/")
AWS_MEDIA_FOLDER = os.environ.get("AWS_MEDIA_FOLDER", "bayhealth")
