import logging
import mimetypes
import secrets
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from django.conf import settings

logger = logging.getLogger(__name__)


def _s3_client():
    kwargs = {"region_name": settings.AWS_S3_REGION_NAME}
    access_key = (settings.AWS_ACCESS_KEY_ID or "").strip()
    secret_key = (settings.AWS_SECRET_ACCESS_KEY or "").strip()
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def build_object_key(folder, filename):
    safe_name = Path(filename).name.replace(" ", "_")
    token = secrets.token_hex(4)
    prefix = (settings.AWS_MEDIA_FOLDER or "uploads").strip("/")
    return f"{prefix}/{folder.strip('/')}/{token}_{safe_name}"


def upload_fileobj(file_obj, object_key, content_type=None):
    bucket = (settings.AWS_STORAGE_BUCKET_NAME or "").strip()
    if not bucket:
        raise ValueError("AWS_STORAGE_BUCKET_NAME is not configured.")

    content_type = content_type or mimetypes.guess_type(object_key)[0] or "application/octet-stream"
    extra_args = {"ContentType": content_type}

    try:
        file_obj.seek(0)
    except Exception:
        pass

    try:
        _s3_client().upload_fileobj(
            file_obj,
            bucket,
            object_key,
            ExtraArgs=extra_args,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.exception("S3 upload failed key=%s", object_key)
        raise RuntimeError(f"Failed to upload file to S3: {exc}") from exc

    return object_key


def upload_bytes(file_bytes, filename, folder, content_type=None):
    from io import BytesIO

    object_key = build_object_key(folder, filename)
    return upload_fileobj(BytesIO(file_bytes), object_key, content_type=content_type)


def upload_django_file(django_file, folder):
    object_key = build_object_key(folder, django_file.name)
    content_type = getattr(django_file, "content_type", None)
    return upload_fileobj(django_file, object_key, content_type=content_type)


def build_s3_url(object_key):
    """Build a full HTTPS URL using AWS_S3_BASE_URL from .env."""
    import os

    key = (object_key or "").strip()
    if not key:
        return ""
    if key.startswith("http://") or key.startswith("https://"):
        return key

    base_url = (
        getattr(settings, "AWS_S3_BASE_URL", "")
        or os.environ.get("AWS_S3_BASE_URL", "")
        or ""
    ).rstrip("/")
    if not base_url:
        return key

    return f"{base_url}/{key.lstrip('/')}"
