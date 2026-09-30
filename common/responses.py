from rest_framework import status
from rest_framework.response import Response


def message_response(message, code=status.HTTP_200_OK):
    return Response({"message": message}, status=code)


def error_response(message, code=status.HTTP_400_BAD_REQUEST):
    return Response({"message": message}, status=code)
