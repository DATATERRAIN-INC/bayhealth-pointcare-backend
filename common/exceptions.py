from rest_framework.views import exception_handler


def _first_error(value):
    if isinstance(value, (list, tuple)):
        return _first_error(value[0]) if value else "Invalid request."
    if isinstance(value, dict):
        for nested in value.values():
            return _first_error(nested)
        return "Invalid request."
    return str(value)


def _format_message(data):
    if data is None:
        return "Something went wrong."

    if isinstance(data, list):
        return _first_error(data)

    if not isinstance(data, dict):
        return str(data)

    if "message" in data and data["message"]:
        return str(data["message"])

    if "detail" in data and data["detail"]:
        return str(data["detail"])

    parts = []
    for field, errors in data.items():
        if field == "errors":
            continue
        label = field.replace("_", " ")
        parts.append(f"{label}: {_first_error(errors)}")

    return "; ".join(parts) if parts else "Invalid request."


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is None:
        return None

    response.data = {"message": _format_message(response.data)}
    return response
