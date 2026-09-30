import os
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
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
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

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME", "")
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME", "us-east-1")
AWS_S3_BASE_URL = os.getenv("AWS_S3_BASE_URL", "").rstrip("/")
AWS_MEDIA_FOLDER = os.getenv("AWS_MEDIA_FOLDER", "bayhealth")
