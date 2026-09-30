from django.contrib.auth.hashers import check_password
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import IntegrityError
from django.utils.timezone import now as dj_now

from apps.users.cognito import (
    CognitoError,
    admin_confirm,
    admin_set_user_password,
    decode_id_token,
    global_sign_out,
    login as cognito_login,
    lookup_cognito_user,
    revoke_refresh_token,
    sign_up,
)
from apps.users.constants import (
    FORGOT_PASSWORD_ATTEMPT_WINDOW_SECONDS,
    FORGOT_PASSWORD_MAX_ATTEMPTS,
    PASSWORD_RESET_TEMPLATE,
    VERIFY_RESET_ATTEMPT_WINDOW_SECONDS,
    VERIFY_RESET_MAX_ATTEMPTS,
)
from apps.users.models import PasswordResetRequest, User
from common.email import EmailUtils


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


def _display_name(user):
    if user.first_name and user.last_name:
        return f"{user.first_name} {user.last_name}"
    return user.first_name or user.last_name or ""


def forgot_password_user(data):
    email = _clean_email(data.get("email"))
    try:
        user = User.objects.get(email=email, is_active=True)
    except User.DoesNotExist:
        raise CognitoError("User not found.", "UserNotFoundException", 404)

    cache_key = f"reset_attempts_{user.id}"
    reset_attempts = cache.get(cache_key, 0)
    if reset_attempts >= FORGOT_PASSWORD_MAX_ATTEMPTS:
        raise CognitoError(
            "Too many reset attempts. Please try again later.",
            "LimitExceededException",
            429,
        )

    PasswordResetRequest.objects.filter(user=user).delete()
    reset_request, reset_code = PasswordResetRequest.create_reset_request(user)
    cache.set(cache_key, reset_attempts + 1, timeout=FORGOT_PASSWORD_ATTEMPT_WINDOW_SECONDS)

    context_data = {
        "user_name": _display_name(user),
        "reset_code": reset_code,
    }
    try:
        EmailUtils.send_email_via_sendgrid_template(
            to_emails=[user.email],
            template_name=PASSWORD_RESET_TEMPLATE,
            context_data=context_data,
        )
    except (ValueError, RuntimeError) as exc:
        reset_request.delete()
        raise CognitoError(str(exc), "EmailSendFailed", 502) from exc

    return {"message": "Password reset code sent to your email."}


def verify_reset_code_user(data):
    email = _clean_email(data.get("email"))
    reset_code = _clean_text(data.get("reset_code"), "reset_code", required=True, max_length=32)

    rate_limit_key = f"verify_reset_attempts_{email}"
    attempts = cache.get(rate_limit_key, 0)
    if attempts >= VERIFY_RESET_MAX_ATTEMPTS:
        raise CognitoError(
            "Too many verification attempts. Please try again later.",
            "LimitExceededException",
            429,
        )

    used_otp = PasswordResetRequest.objects.filter(user__email=email, is_used=True).first()
    if used_otp and check_password(reset_code, used_otp.reset_code):
        cache.set(rate_limit_key, attempts + 1, timeout=VERIFY_RESET_ATTEMPT_WINDOW_SECONDS)
        raise CognitoError(
            "This reset code has already been used. Please request a new code.",
            "CodeMismatchException",
            400,
        )

    try:
        reset_request = PasswordResetRequest.objects.get(
            user__email=email,
            is_used=False,
        )
    except PasswordResetRequest.DoesNotExist:
        PasswordResetRequest.objects.filter(
            expires_at__isnull=False,
            expires_at__lt=dj_now(),
        ).delete()
        raise CognitoError("No active password reset request found.", "UserNotFoundException", 404)

    if reset_request.expires_at and reset_request.expires_at < dj_now():
        cache.set(rate_limit_key, attempts + 1, timeout=VERIFY_RESET_ATTEMPT_WINDOW_SECONDS)
        raise CognitoError("Reset code has expired.", "ExpiredCodeException", 400)

    if not reset_request.verify_reset_code(reset_code):
        cache.set(rate_limit_key, attempts + 1, timeout=VERIFY_RESET_ATTEMPT_WINDOW_SECONDS)
        raise CognitoError("Invalid reset code.", "CodeMismatchException", 400)

    reset_request.is_used = True
    reset_request.used_at = dj_now()
    reset_request.save(update_fields=["is_used", "used_at"])

    return {
        "email": reset_request.user.email,
        "token": reset_request.token,
    }


def validate_password_token(token):
    if not token or not isinstance(token, str):
        return False
    token = token.strip()
    if not token:
        return False
    try:
        reset_request = PasswordResetRequest.objects.get(token=token)
    except PasswordResetRequest.DoesNotExist:
        return False
    if reset_request.expires_at and reset_request.expires_at < dj_now():
        return False
    return True


def reset_password_user(data):
    token = data.get("token")
    if not isinstance(token, str) or not token.strip():
        raise CognitoError("token is required.", "InvalidParameterException", 400)
    token = token.strip()

    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")
    if not isinstance(new_password, str) or not new_password:
        raise CognitoError("new_password is required.", "InvalidParameterException", 400)
    if not isinstance(confirm_password, str) or not confirm_password:
        raise CognitoError("confirm_password is required.", "InvalidParameterException", 400)
    if new_password != confirm_password:
        raise CognitoError(
            "New password and confirm password do not match.",
            "InvalidParameterException",
            400,
        )
    if len(new_password) < 8:
        raise CognitoError(
            "new_password must be at least 8 characters.",
            "InvalidPasswordException",
            400,
        )

    try:
        reset_request = PasswordResetRequest.objects.get(token=token)
    except PasswordResetRequest.DoesNotExist:
        raise CognitoError(
            "Invalid or expired reset token. Please request a new password reset.",
            "UserNotFoundException",
            404,
        )

    if reset_request.expires_at and reset_request.expires_at < dj_now():
        raise CognitoError(
            "Reset token has expired. Please request a new password reset.",
            "ExpiredCodeException",
            400,
        )

    user = reset_request.user
    if user.password_changed_at and reset_request.created_at:
        if user.password_changed_at > reset_request.created_at:
            raise CognitoError(
                "This reset token has already been used. Please request a new password reset.",
                "CodeMismatchException",
                400,
            )

    cognito_names = []
    if user.cognito_username:
        cognito_names.append(user.cognito_username)
    if user.email not in cognito_names:
        cognito_names.append(user.email)

    last_error = None
    for cognito_name in cognito_names:
        try:
            admin_set_user_password(username=cognito_name, password=new_password)
            last_error = None
            break
        except CognitoError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error

    user.password_changed_at = dj_now()
    if not user.is_confirmed:
        user.is_confirmed = True
    user.save(update_fields=["password_changed_at", "is_confirmed", "updated_at"])
    reset_request.delete()

    return {"message": "Password reset successfully."}
