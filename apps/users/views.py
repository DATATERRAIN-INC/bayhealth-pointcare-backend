from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.users.cognito import CognitoError
from apps.users.serializers import (
    ForgotPasswordSerializer,
    LoginSerializer,
    LogoutSerializer,
    RegisterSerializer,
    ResetPasswordSerializer,
    VerifyResetCodeSerializer,
)
from apps.users.services import (
    forgot_password_user,
    login_user,
    logout_user,
    register_user,
    reset_password_user,
    validate_password_token,
    verify_reset_code_user,
)


class UsersAPIView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    @staticmethod
    def _invalid_body():
        return Response(
            {"error": "Request body must be a JSON object."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    @staticmethod
    def _cognito_error(exc):
        return Response(
            {"error": exc.message, "code": exc.code},
            status=exc.status,
        )

    def _payload(self, request):
        if not isinstance(request.data, dict):
            return None, self._invalid_body()
        return request.data, None


def _bearer_token(request):
    header = request.headers.get("Authorization", "")
    prefix = "bearer "
    if header.lower().startswith(prefix):
        return header[len(prefix) :].strip()
    return ""


class RegisterView(UsersAPIView):
    def post(self, request):
        data, error_response = self._payload(request)
        if error_response is not None:
            return error_response
        serializer = RegisterSerializer(data=data)
        serializer.is_valid(raise_exception=False)
        try:
            body = register_user(serializer.initial_data)
        except CognitoError as exc:
            return self._cognito_error(exc)
        return Response(body, status=status.HTTP_201_CREATED)


class LoginView(UsersAPIView):
    def post(self, request):
        data, error_response = self._payload(request)
        if error_response is not None:
            return error_response
        serializer = LoginSerializer(data=data)
        serializer.is_valid(raise_exception=False)
        try:
            body = login_user(serializer.initial_data)
        except CognitoError as exc:
            return self._cognito_error(exc)
        return Response(body, status=status.HTTP_200_OK)


class LogoutView(UsersAPIView):
    def post(self, request):
        data, error_response = self._payload(request)
        if error_response is not None:
            return error_response
        serializer = LogoutSerializer(data=data)
        serializer.is_valid(raise_exception=False)
        try:
            body = logout_user(
                access_token=_bearer_token(request),
                refresh_token=serializer.initial_data.get("refresh_token"),
            )
        except CognitoError as exc:
            return self._cognito_error(exc)
        return Response(body, status=status.HTTP_200_OK)


class ForgotPasswordView(UsersAPIView):
    def post(self, request):
        data, error_response = self._payload(request)
        if error_response is not None:
            return error_response
        serializer = ForgotPasswordSerializer(data=data)
        serializer.is_valid(raise_exception=False)
        try:
            body = forgot_password_user(serializer.initial_data)
        except CognitoError as exc:
            return self._cognito_error(exc)
        return Response(body, status=status.HTTP_200_OK)


class VerifyResetCodeView(UsersAPIView):
    def post(self, request):
        data, error_response = self._payload(request)
        if error_response is not None:
            return error_response
        serializer = VerifyResetCodeSerializer(data=data)
        serializer.is_valid(raise_exception=False)
        try:
            body = verify_reset_code_user(serializer.initial_data)
        except CognitoError as exc:
            return self._cognito_error(exc)
        return Response(body, status=status.HTTP_200_OK)


class ValidatePasswordTokenView(UsersAPIView):
    def get(self, request):
        token = (request.query_params.get("token") or "").strip()
        if not token:
            return Response({"valid": False}, status=status.HTTP_400_BAD_REQUEST)
        return Response(
            {"valid": validate_password_token(token)},
            status=status.HTTP_200_OK,
        )


class ResetPasswordView(UsersAPIView):
    def post(self, request):
        data, error_response = self._payload(request)
        if error_response is not None:
            return error_response
        serializer = ResetPasswordSerializer(data=data)
        serializer.is_valid(raise_exception=False)
        try:
            body = reset_password_user(serializer.initial_data)
        except CognitoError as exc:
            return self._cognito_error(exc)
        return Response(body, status=status.HTTP_200_OK)
