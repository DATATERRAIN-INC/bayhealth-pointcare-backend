import re

from rest_framework import authentication, exceptions

from apps.users.cognito import CognitoError, get_user
from apps.users.models import User

# Cognito AccessToken constraint: [A-Za-z0-9-_=.]+
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9\-_=.]+$")
_AUTH_FAIL_CODES = {
    "NotAuthorizedException",
    "InvalidParameterException",
    "UserNotFoundException",
    "ExpiredTokenException",
}


def _auth_failed_message(exc=None):
    if exc is None:
        return "Invalid or expired access token."
    if getattr(exc, "code", "") in _AUTH_FAIL_CODES:
        return "Invalid or expired access token."
    message = (getattr(exc, "message", "") or str(exc)).strip()
    if "accesstoken" in message.lower() or "access token" in message.lower():
        return "Invalid or expired access token."
    return message or "Invalid or expired access token."


class CognitoBearerAuthentication(authentication.BaseAuthentication):
    """Authenticate requests with Cognito access token: Authorization: Bearer <token>."""

    keyword = "Bearer"

    def authenticate(self, request):
        header = authentication.get_authorization_header(request).decode("utf-8")
        if not header:
            return None

        parts = header.split(None, 1)
        if not parts or parts[0].lower() != self.keyword.lower():
            return None
        if len(parts) != 2 or not parts[1].strip():
            raise exceptions.AuthenticationFailed(
                "Authorization header must be: Bearer <access_token>."
            )

        token = parts[1].strip()
        if not _TOKEN_PATTERN.fullmatch(token) or token.count(".") < 2:
            raise exceptions.AuthenticationFailed("Invalid or expired access token.")

        try:
            cognito_user = get_user(access_token=token)
        except CognitoError as exc:
            raise exceptions.AuthenticationFailed(_auth_failed_message(exc)) from exc

        attributes = {
            item["Name"]: item["Value"]
            for item in cognito_user.get("UserAttributes", [])
        }
        cognito_id = (attributes.get("sub") or "").strip()
        username = (cognito_user.get("Username") or "").strip()
        email = (attributes.get("email") or "").strip().lower()

        user = None
        if cognito_id:
            user = User.objects.filter(cognito_id=cognito_id).first()
        if user is None and username:
            user = User.objects.filter(cognito_username=username).first()
        if user is None and email:
            user = User.objects.filter(email=email).first()

        if user is None:
            raise exceptions.AuthenticationFailed("User not found.")
        if not user.is_active:
            raise exceptions.AuthenticationFailed("This account is inactive.")

        return (user, token)

    def authenticate_header(self, request):
        return self.keyword
