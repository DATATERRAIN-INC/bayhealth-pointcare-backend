import base64
import hashlib
import hmac
import json
import uuid

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from django.conf import settings


class CognitoError(Exception):
    def __init__(self, message, code="CognitoError", status=400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


_ERROR_STATUS = {
    "UsernameExistsException": 409,
    "AliasExistsException": 409,
    "InvalidPasswordException": 400,
    "InvalidParameterException": 400,
    "CodeMismatchException": 400,
    "ExpiredCodeException": 400,
    "UserNotConfirmedException": 403,
    "NotAuthorizedException": 401,
    "UserNotFoundException": 401,
    "TooManyRequestsException": 429,
    "LimitExceededException": 429,
    "UserLambdaValidationException": 400,
}


def ensure_config():
    missing = [
        name
        for name, value in (
            ("AWS_REGION", settings.AWS_REGION),
            ("COGNITO_USER_POOL_ID", settings.COGNITO_USER_POOL_ID),
            ("COGNITO_APP_CLIENT_ID", settings.COGNITO_APP_CLIENT_ID),
        )
        if not value
    ]
    if missing:
        raise CognitoError(
            "Missing Cognito configuration in .env: " + ", ".join(missing) + ".",
            "MissingConfiguration",
            500,
        )


def secret_hash(username):
    client_secret = settings.COGNITO_APP_CLIENT_SECRET
    if not client_secret:
        return None
    message = (username + settings.COGNITO_APP_CLIENT_ID).encode("utf-8")
    digest = hmac.new(client_secret.encode("utf-8"), message, hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def decode_id_token(id_token):
    try:
        payload = id_token.split(".")[1]
        padding = "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload + padding))
    except (IndexError, ValueError, json.JSONDecodeError) as exc:
        raise CognitoError(
            "Cognito returned an ID token that could not be read.",
            "InvalidIdToken",
            502,
        ) from exc


def _aws_credentials():
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        return {
            "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
        }
    return {}


def _public_client():
    credentials = _aws_credentials()
    if credentials:
        return boto3.client("cognito-idp", region_name=settings.AWS_REGION, **credentials)
    return boto3.client(
        "cognito-idp",
        region_name=settings.AWS_REGION,
        config=Config(signature_version=UNSIGNED),
    )


def _admin_client():
    return boto3.client("cognito-idp", region_name=settings.AWS_REGION, **_aws_credentials())


def raise_for_client_error(exc, *, login=False):
    error = exc.response.get("Error", {})
    code = error.get("Code", "CognitoError")
    message = error.get("Message", "Cognito request failed.")
    if login and code in ("NotAuthorizedException", "UserNotFoundException"):
        raise CognitoError("Incorrect email or password.", code, 401) from exc
    if code == "UserNotConfirmedException":
        raise CognitoError("User is not confirmed in Cognito.", code, 403) from exc
    raise CognitoError(message, code, _ERROR_STATUS.get(code, 400)) from exc


def _call(func, *, login=False):
    try:
        return func()
    except ClientError as exc:
        raise_for_client_error(exc, login=login)
    except NoCredentialsError as exc:
        raise CognitoError(
            "AWS credentials are required for this Cognito admin call. "
            "Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to .env.",
            "NoCredentials",
            500,
        ) from exc
    except BotoCoreError as exc:
        raise CognitoError("Could not reach Cognito.", "BotoCoreError", 502) from exc


def sign_up(*, email, password, first_name, last_name, phone_number):
    ensure_config()
    # This pool uses email as an alias, so Username cannot be an email address.
    username = uuid.uuid4().hex
    attributes = [{"Name": "email", "Value": email}]
    if first_name:
        attributes.append({"Name": "given_name", "Value": first_name})
    if last_name:
        attributes.append({"Name": "family_name", "Value": last_name})
    if phone_number:
        attributes.append({"Name": "phone_number", "Value": phone_number})

    params = {
        "ClientId": settings.COGNITO_APP_CLIENT_ID,
        "Username": username,
        "Password": password,
        "UserAttributes": attributes,
    }
    hashed = secret_hash(username)
    if hashed:
        params["SecretHash"] = hashed

    result = _call(lambda: _public_client().sign_up(**params))
    result["Username"] = username
    return result


def login(*, username, password):
    ensure_config()
    auth_parameters = {
        "USERNAME": username,
        "PASSWORD": password,
    }
    hashed = secret_hash(username)
    if hashed:
        auth_parameters["SECRET_HASH"] = hashed

    response = _call(
        lambda: _public_client().initiate_auth(
            ClientId=settings.COGNITO_APP_CLIENT_ID,
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters=auth_parameters,
        ),
        login=True,
    )
    result = response.get("AuthenticationResult")
    if not result:
        raise CognitoError(
            "Cognito did not return tokens. Check that the app client allows USER_PASSWORD_AUTH.",
            "AuthenticationIncomplete",
            400,
        )
    return result


def admin_confirm(username):
    """Confirm the Cognito user and mark the email verified."""
    ensure_config()
    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        raise CognitoError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to confirm users.",
            "NoCredentials",
            500,
        )

    client = _admin_client()

    def confirm():
        try:
            client.admin_confirm_sign_up(
                UserPoolId=settings.COGNITO_USER_POOL_ID,
                Username=username,
            )
        except ClientError as exc:
            message = exc.response.get("Error", {}).get("Message", "")
            if "CONFIRMED" not in message.upper():
                raise

    def mark_email_verified():
        client.admin_update_user_attributes(
            UserPoolId=settings.COGNITO_USER_POOL_ID,
            Username=username,
            UserAttributes=[{"Name": "email_verified", "Value": "true"}],
        )

    _call(confirm)
    _call(mark_email_verified)


def forgot_password(*, username):
    ensure_config()
    params = {
        "ClientId": settings.COGNITO_APP_CLIENT_ID,
        "Username": username,
    }
    hashed = secret_hash(username)
    if hashed:
        params["SecretHash"] = hashed
    return _call(lambda: _public_client().forgot_password(**params))


def confirm_forgot_password(*, username, code, password):
    ensure_config()
    params = {
        "ClientId": settings.COGNITO_APP_CLIENT_ID,
        "Username": username,
        "ConfirmationCode": code,
        "Password": password,
    }
    hashed = secret_hash(username)
    if hashed:
        params["SecretHash"] = hashed
    return _call(lambda: _public_client().confirm_forgot_password(**params))


def global_sign_out(*, access_token):
    ensure_config()
    return _call(lambda: _public_client().global_sign_out(AccessToken=access_token))


def revoke_refresh_token(*, refresh_token):
    ensure_config()
    params = {
        "Token": refresh_token,
        "ClientId": settings.COGNITO_APP_CLIENT_ID,
    }
    if settings.COGNITO_APP_CLIENT_SECRET:
        params["ClientSecret"] = settings.COGNITO_APP_CLIENT_SECRET
    return _call(lambda: _public_client().revoke_token(**params))


def admin_set_user_password(*, username, password):
    """Set a permanent password via Cognito admin API (requires AWS credentials)."""
    ensure_config()
    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        raise CognitoError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to reset passwords.",
            "NoCredentials",
            500,
        )

    def set_password():
        _admin_client().admin_set_user_password(
            UserPoolId=settings.COGNITO_USER_POOL_ID,
            Username=username,
            Password=password,
            Permanent=True,
        )

    _call(set_password)


def lookup_cognito_user(username):
    """Return (sub, is_confirmed, username) using AdminGetUser, or None without credentials."""
    ensure_config()
    try:
        response = _admin_client().admin_get_user(
            UserPoolId=settings.COGNITO_USER_POOL_ID,
            Username=username,
        )
    except (NoCredentialsError, BotoCoreError):
        return None
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in (
            "AccessDeniedException",
            "UnrecognizedClientException",
            "InvalidClientTokenId",
            "InvalidSignatureException",
            "NotAuthorizedException",
            "UserNotFoundException",
        ):
            return None
        raise_for_client_error(exc)

    attributes = {
        item["Name"]: item["Value"] for item in response.get("UserAttributes", [])
    }
    sub = attributes.get("sub")
    if not sub:
        return None
    return sub, response.get("UserStatus") == "CONFIRMED", response.get("Username") or ""
