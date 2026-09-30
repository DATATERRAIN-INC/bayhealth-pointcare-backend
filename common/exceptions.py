from rest_framework.views import exception_handler


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is None:
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

    response.data = {"message": message}
    return response
