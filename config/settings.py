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
    "apps.users.apps.UsersConfig",
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
