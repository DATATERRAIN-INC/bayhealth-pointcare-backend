from rest_framework import serializers


class RegisterSerializer(serializers.Serializer):
    email = serializers.CharField(required=False, allow_blank=True)
    password = serializers.CharField(required=False, allow_blank=True, write_only=True)
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    phone_number = serializers.CharField(required=False, allow_blank=True)


class LoginSerializer(serializers.Serializer):
    email = serializers.CharField(required=False, allow_blank=True)
    password = serializers.CharField(required=False, allow_blank=True, write_only=True)


class LogoutSerializer(serializers.Serializer):
    refresh_token = serializers.CharField(required=False, allow_blank=True)


class ForgotPasswordSerializer(serializers.Serializer):
    email = serializers.CharField(required=False, allow_blank=True)


class VerifyResetCodeSerializer(serializers.Serializer):
    email = serializers.CharField(required=False, allow_blank=True)
    reset_code = serializers.CharField(required=False, allow_blank=True)


class ResetPasswordSerializer(serializers.Serializer):
    token = serializers.CharField(required=False, allow_blank=True)
    new_password = serializers.CharField(required=False, allow_blank=True, write_only=True)
    confirm_password = serializers.CharField(required=False, allow_blank=True, write_only=True)
