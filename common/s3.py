import logging
import mimetypes
import secrets
from io import BytesIO
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from django.conf import settings

logger = logging.getLogger(__name__)


def _s3_client():
    kwargs = {"region_name": settings.AWS_S3_REGION_NAME}
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    return boto3.client("s3", **kwargs)


def build_object_key(folder, filename):
    name = Path(filename).name.replace(" ", "_")
    prefix = (settings.AWS_MEDIA_FOLDER or "uploads").strip("/")
    return f"{prefix}/{folder.strip('/')}/{secrets.token_hex(4)}_{name}"


def upload_bytes(file_bytes, filename, folder, content_type=None):
    bucket = (settings.AWS_STORAGE_BUCKET_NAME or "").strip()
    if not bucket:
        raise ValueError("AWS_STORAGE_BUCKET_NAME is not configured.")

    object_key = build_object_key(folder, filename)
    content_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    try:
        _s3_client().upload_fileobj(
            BytesIO(file_bytes),
            bucket,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )
    except (BotoCoreError, ClientError) as exc:
        logger.exception("S3 upload failed key=%s", object_key)
        raise RuntimeError(f"Failed to upload file to S3: {exc}") from exc

    return object_key


def build_s3_url(object_key):
    key = (object_key or "").strip()
    if not key:
        return ""
    if key.startswith(("http://", "https://")):
        return key

    base_url = (settings.AWS_S3_BASE_URL or "").rstrip("/")
    return f"{base_url}/{key.lstrip('/')}" if base_url else key
