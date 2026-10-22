"""Cognito bearer auth: verified tokens are cached briefly, never past expiry."""

import base64
import json
import time
from unittest.mock import patch

from django.core.cache import cache
from django.test import TestCase, override_settings
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.test import APIRequestFactory

from apps.users import authentication
from apps.users.cognito import CognitoError
from apps.users.models import User

LOCMEM = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}


def _token(exp_in_seconds):
    payload = json.dumps({"exp": int(time.time()) + exp_in_seconds}).encode()
    body = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"eyJhbGciOiJSUzI1NiJ9.{body}.signature"


COGNITO_USER = {
    "Username": "user-1",
    "UserAttributes": [{"Name": "sub", "Value": "sub-1"}, {"Name": "email", "Value": "a@b.com"}],
}


@override_settings(CACHES=LOCMEM)
class CognitoBearerAuthenticationTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create(cognito_id="sub-1", email="a@b.com")
        self.auth = authentication.CognitoBearerAuthentication()

    def _authenticate(self, token):
        request = APIRequestFactory().get("/", HTTP_AUTHORIZATION=f"Bearer {token}")
        return self.auth.authenticate(request)

    def test_second_request_skips_cognito(self):
        token = _token(3600)
        with patch.object(authentication, "get_user", return_value=COGNITO_USER) as get_user:
            self.assertEqual(self._authenticate(token)[0], self.user)
            self.assertEqual(self._authenticate(token)[0], self.user)
        get_user.assert_called_once()

    def test_expired_token_is_never_cached(self):
        token = _token(-5)
        with patch.object(authentication, "get_user", return_value=COGNITO_USER) as get_user:
            self._authenticate(token)
            self._authenticate(token)
        self.assertEqual(get_user.call_count, 2)

    def test_logout_forgets_token(self):
        token = _token(3600)
        with patch.object(authentication, "get_user", return_value=COGNITO_USER) as get_user:
            self._authenticate(token)
            authentication.forget_verified_token(token)
            self._authenticate(token)
        self.assertEqual(get_user.call_count, 2)

    def test_rejected_token_is_not_cached(self):
        token = _token(3600)
        error = CognitoError("bad", "NotAuthorizedException", 401)
        with patch.object(authentication, "get_user", side_effect=error) as get_user:
            for _ in range(2):
                with self.assertRaises(AuthenticationFailed):
                    self._authenticate(token)
        self.assertEqual(get_user.call_count, 2)

    def test_inactive_user_rejected_even_when_cached(self):
        token = _token(3600)
        with patch.object(authentication, "get_user", return_value=COGNITO_USER):
            self._authenticate(token)
            self.user.is_active = False
            self.user.save()
            with self.assertRaises(AuthenticationFailed):
                self._authenticate(token)


class MqttBackgroundSendTests(TestCase):
    def test_publish_does_not_wait_for_synccast(self):
        import threading
        from types import SimpleNamespace

        from apps.notifications import mqtt

        started, release = threading.Event(), threading.Event()

        def slow_send(*args, **kwargs):
            started.set()
            release.wait(5)

        rows = [SimpleNamespace(name="completed", scope=SimpleNamespace(name="scope"))]
        t = time.perf_counter()
        # Includes running the on-commit hook: a synchronous send would take 5s.
        with patch("apps.notifications.mqtt.Channel.objects") as objects, patch.object(
            mqtt, "send", side_effect=slow_send
        ), self.captureOnCommitCallbacks(execute=True):
            objects.filter.return_value.select_related.return_value.filter.return_value = rows
            mqtt.publish(1, {"event": "x"}, channel_names=["completed"])
        elapsed = time.perf_counter() - t
        self.assertLess(elapsed, 0.5)
        self.assertTrue(started.wait(2))
        release.set()
