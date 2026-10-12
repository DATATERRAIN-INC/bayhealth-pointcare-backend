from django.db import IntegrityError
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)

    if response is None:
        if isinstance(exc, IntegrityError):
            return Response(
                {
                    "message": (
                        "Unable to save data due to a database conflict. "
                        "Please try again or contact support."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        if isinstance(exc, ValueError):
            raw = str(exc).strip() or "Invalid request."
            lowered = raw.lower()
            # Hide internal Django FK / model-instance errors from clients.
            if "must be" in lowered and "instance" in lowered:
                message = "Unable to load data for this account. Sign in again or contact support."
            elif "expected a number" in lowered:
                message = "Invalid request."
            else:
                message = raw
            return Response({"message": message}, status=status.HTTP_400_BAD_REQUEST)
        return None

    data = response.data
    if isinstance(data, dict) and data.get("detail"):
        message = str(data["detail"])
    elif isinstance(data, dict):
        parts = []
        for field, errors in data.items():
            err = errors[0] if isinstance(errors, (list, tuple)) and errors else errors
            parts.append(f"{field}: {err}")
        message = "; ".join(parts) if parts else "Invalid request."
    elif isinstance(data, list) and data:
        message = str(data[0])
    else:
        message = str(data)

    # Never leak raw Cognito / AWS validation text to clients.
    lowered = message.lower()
    if (
        "validation error detected" in lowered
        or "failed to satisfy constraint" in lowered
        or "member must satisfy regular expression" in lowered
        or "accesstoken" in lowered.replace(" ", "")
    ):
        message = "Invalid or expired access token."

    if "null value in column" in lowered and "user_id" in lowered:
        message = (
            "Unable to load caller settings for this account. "
            "Sign in again or contact support."
        )
    elif "duplicate key" in lowered or "unique constraint" in lowered:
        message = (
            "Unable to save data due to a database conflict. "
            "Please try again or contact support."
        )

    response.data = {"message": message}
    return response
