from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import IntegrityError

from apps.users.cognito import (
    CognitoError,
    admin_confirm,
    confirm_forgot_password,
    decode_id_token,
    forgot_password,
    global_sign_out,
    login as cognito_login,
    lookup_cognito_user,
    revoke_refresh_token,
    sign_up,
)
from apps.users.models import User


def _clean_text(value, field_name, *, required=False, max_length=150):
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise CognitoError(f"{field_name} must be a string.", "InvalidParameterException", 400)
    value = value.strip()
    if required and not value:
        raise CognitoError(f"{field_name} is required.", "InvalidParameterException", 400)
    if len(value) > max_length:
        raise CognitoError(
            f"{field_name} must be {max_length} characters or fewer.",
            "InvalidParameterException",
            400,
        )
    return value


def _clean_email(value):
    email = _clean_text(value, "email", required=True, max_length=254).lower()
    try:
        validate_email(email)
    except ValidationError as exc:
        raise CognitoError("Enter a valid email address.", "InvalidParameterException", 400) from exc
    return email


def _clean_phone(value):
    phone = _clean_text(value, "phone_number", required=False, max_length=20)
    if phone and not phone.startswith("+"):
        raise CognitoError(
            "phone_number must be in E.164 format, for example +15551234567.",
            "InvalidParameterException",
            400,
        )
    return phone


def user_payload(user):
    return {
        "id": user.id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
    }


def register_user(data):
    email = _clean_email(data.get("email"))
    password = data.get("password")
    if not isinstance(password, str) or not password:
        raise CognitoError("password is required.", "InvalidParameterException", 400)
    if len(password) < 8:
        raise CognitoError(
            "password must be at least 8 characters.",
            "InvalidPasswordException",
            400,
        )
    first_name = _clean_text(data.get("first_name"), "first_name", required=True)
    last_name = _clean_text(data.get("last_name"), "last_name", required=True)
    phone_number = _clean_phone(data.get("phone_number"))

    if User.objects.filter(email=email).exists():
        raise CognitoError(
            "A user with this email already exists.",
            "UsernameExistsException",
            409,
        )

    confirmed = False
    cognito_username = ""
    try:
        response = sign_up(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            phone_number=phone_number,
        )
        cognito_id = response["UserSub"]
        cognito_username = response["Username"]
        confirmed = bool(response.get("UserConfirmed"))
    except CognitoError as exc:
        if exc.code not in ("UsernameExistsException", "AliasExistsException"):
            raise
        existing = lookup_cognito_user(email)
        if not existing:
            raise
        cognito_id, confirmed, cognito_username = existing
        if User.objects.filter(cognito_id=cognito_id).exists():
            raise

    try:
        user = User.objects.create(
            cognito_id=cognito_id,
            cognito_username=cognito_username or None,
            email=email,
            first_name=first_name,
            last_name=last_name,
            phone_number=phone_number,
            is_confirmed=confirmed,
        )
    except IntegrityError as exc:
        raise CognitoError(
            "A user with this email already exists.",
            "UsernameExistsException",
            409,
        ) from exc

    if user.cognito_username:
        admin_confirm(user.cognito_username)
        if not user.is_confirmed:
            user.is_confirmed = True
            user.save(update_fields=["is_confirmed", "updated_at"])

    return {
        "message": "User registered successfully.",
        "user": user_payload(user),
    }


def login_user(data):
    email = _clean_email(data.get("email"))
    password = data.get("password")
    if not isinstance(password, str) or not password:
        raise CognitoError("password is required.", "InvalidParameterException", 400)

    local_user = User.objects.filter(email=email).first()
    if local_user is not None and not local_user.is_active:
        raise CognitoError("This account is inactive.", "UserDisabled", 403)

    login_name = email
    if local_user is not None and local_user.cognito_username:
        login_name = local_user.cognito_username

    try:
        tokens = cognito_login(username=login_name, password=password)
    except CognitoError as exc:
        if exc.code != "UserNotConfirmedException":
            raise
        admin_confirm(login_name)
        if local_user is not None and not local_user.is_confirmed:
            local_user.is_confirmed = True
            local_user.save(update_fields=["is_confirmed", "updated_at"])
        tokens = cognito_login(username=login_name, password=password)
    claims = decode_id_token(tokens["IdToken"])
    cognito_id = claims.get("sub")
    if not cognito_id:
        raise CognitoError(
            "Cognito ID token did not include a user id.",
            "InvalidIdToken",
            502,
        )

    token_email = (claims.get("email") or email).strip().lower()
    user = User.objects.filter(cognito_id=cognito_id).first()
    if user is None:
        user = User.objects.filter(email=token_email).first()
        if user is None:
            user = User.objects.create(
                cognito_id=cognito_id,
                cognito_username=claims.get("cognito:username") or None,
                email=token_email,
                first_name=claims.get("given_name") or "",
                last_name=claims.get("family_name") or "",
                phone_number=claims.get("phone_number") or "",
                is_confirmed=True,
            )
        elif user.cognito_id != cognito_id:
            raise CognitoError(
                "This email is already linked to a different Cognito user.",
                "AliasExistsException",
                409,
            )
    if not user.is_active:
        raise CognitoError("This account is inactive.", "UserDisabled", 403)

    updates = []
    if not user.is_confirmed:
        user.is_confirmed = True
        updates.append("is_confirmed")
    if updates:
        updates.append("updated_at")
        user.save(update_fields=updates)

    return {
        "access_token": tokens.get("AccessToken"),
        "refresh_token": tokens.get("RefreshToken"),
    }


def _cognito_username(email):
    user = User.objects.filter(email=email).first()
    if user is not None and user.cognito_username:
        return user.cognito_username
    return email


def logout_user(*, access_token, refresh_token):
    if not access_token:
        raise CognitoError("Access token is required.", "NotAuthorizedException", 401)
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise CognitoError("refresh_token is required.", "InvalidParameterException", 400)

    revoke_error = None
    try:
        revoke_refresh_token(refresh_token=refresh_token.strip())
    except CognitoError as exc:
        if exc.code not in ("NotAuthorizedException", "InvalidParameterException"):
            raise
        revoke_error = exc

    try:
        global_sign_out(access_token=access_token)
    except CognitoError as exc:
        if exc.code != "NotAuthorizedException" or revoke_error is not None:
            if revoke_error is not None:
                raise revoke_error
            raise

    return {"message": "Logged out successfully."}


def forgot_password_user(data):
    email = _clean_email(data.get("email"))
    forgot_password(username=_cognito_username(email))
    return {"message": "Password reset code sent."}


def reset_password_user(data):
    email = _clean_email(data.get("email"))
    code = _clean_text(data.get("code"), "code", required=True, max_length=32)
    password = data.get("new_password")
    if not isinstance(password, str) or not password:
        raise CognitoError("new_password is required.", "InvalidParameterException", 400)
    if len(password) < 8:
        raise CognitoError(
            "new_password must be at least 8 characters.",
            "InvalidPasswordException",
            400,
        )
    confirm_forgot_password(
        username=_cognito_username(email),
        code=code,
        password=password,
    )
    return {"message": "Password reset successfully."}
