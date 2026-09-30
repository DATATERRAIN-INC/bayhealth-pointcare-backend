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

    response.data = {"message": message}
    return response
