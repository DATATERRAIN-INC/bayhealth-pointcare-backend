from rest_framework.response import Response


def message_response(message, code=200):
    return Response({"message": message}, status=code)


def error_response(message, code=400):
    return Response({"message": message}, status=code)
