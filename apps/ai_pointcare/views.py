# -*- coding: utf-8 -*-
import hashlib
import hmac
import json
import logging
import re
import time

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.ai_caller.retell import _api_key
from apps.ai_pointcare.flow_service import handle_retell_function
from apps.ai_pointcare.serializers import PlaceMediCalCallSerializer
from apps.ai_pointcare.services import start_medi_cal_call

logger = logging.getLogger(__name__)

_RETELL_SIGNATURE_RE = re.compile(r"v=(\d+),d=([0-9a-fA-F]+)")
_RETELL_SIGNATURE_WINDOW_MS = 5 * 60 * 1000


def _verify_retell_signature(raw_body: bytes, signature: str):
    """Return (ok, reason). reason is empty when the signature is valid."""
    api_key = _api_key()
    if not api_key:
        return False, "missing_api_key"
    if not signature:
        return False, "missing_signature"
    match = _RETELL_SIGNATURE_RE.fullmatch(signature.strip())
    if not match:
        match = re.fullmatch(
            r"v\s*=\s*(\d+)\s*,\s*d\s*=\s*([0-9a-fA-F]+)",
            signature.strip(),
            flags=re.I,
        )
    if not match:
        return False, "bad_format"
    timestamp, digest = match.group(1), match.group(2).lower()
    try:
        posted_ms = int(timestamp)
    except ValueError:
        return False, "bad_timestamp"
    age_ms = abs(int(time.time() * 1000) - posted_ms)
    if age_ms > _RETELL_SIGNATURE_WINDOW_MS:
        return False, f"stale_timestamp age_ms={age_ms}"
    try:
        body_text = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        return False, "body_not_utf8"

    expected = hmac.new(
        api_key.encode("utf-8"),
        (body_text + timestamp).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if hmac.compare_digest(expected, digest):
        return True, ""

    legacy = hmac.new(
        api_key.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    if hmac.compare_digest(legacy, digest):
        return True, ""
    return False, "digest_mismatch"


def _failure(message, code=400, errors=None):
    payload = {"status": "failed", "message": message}
    if errors is not None:
        payload["errors"] = errors
    return Response(payload, status=code)


class PlaceMediCalCallView(APIView):
    """Place a Retell Medi-Cal renewal state-machine call."""

    def post(self, request):
        serializer = PlaceMediCalCallSerializer(data=request.data)
        if not serializer.is_valid():
            return _failure(
                "Invalid parameters.",
                code=400,
                errors=serializer.errors,
            )
        result = start_medi_cal_call(
            phone_number=serializer.validated_data["phone_number"],
            name=serializer.validated_data["name"],
            agent_number=serializer.validated_data["agent_number"],
            address=serializer.validated_data.get("address") or "",
        )
        if not result.get("ok"):
            code = int(result.get("status_code") or 502)
            return _failure(
                result.get("error") or "Failed to place Medi-Cal call.",
                code=code,
            )
        return Response(
            {
                "status": "success",
                "result": {
                    "outreach_id": result.get("outreach_id"),
                    "call_id": result.get("call_id") or "",
                    "status": result.get("status") or "registered",
                    "agent_id": result.get("agent_id") or "",
                    "from_number": result.get("from_number") or None,
                    "phone_number": result.get("phone_number"),
                    "phone_last4": result.get("phone_last4") or "",
                    "agent_number": result.get("agent_number") or "",
                    "starting_state": result.get("starting_state") or "",
                    "name": result.get("name"),
                    "message": result.get("message")
                    or "Medi-Cal renewal call placed through Retell AI.",
                },
            }
        )


@method_decorator(csrf_exempt, name="dispatch")
class RetellFunctionWebhookView(APIView):
    """Retell custom-function webhook for Medi-Cal flow tools."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        raw = request.body or b""
        signature = request.headers.get("X-Retell-Signature", "")
        if signature:
            valid, reason = _verify_retell_signature(raw, signature)
            if not valid:
                logger.warning(
                    "ai_pointcare retell function signature rejected reason=%s "
                    "body_len=%s — still running the tool so the call does not stall",
                    reason,
                    len(raw),
                )
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return Response({"message": "Invalid JSON"}, status=400)

        name = str(
            payload.get("name")
            or payload.get("function_name")
            or payload.get("tool_name")
            or ""
        ).strip()
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        if not args:
            args = {
                k: v
                for k, v in payload.items()
                if k
                not in (
                    "name",
                    "function_name",
                    "tool_name",
                    "call",
                    "args",
                    "arguments",
                )
            }
        if not args and isinstance(payload.get("arguments"), dict):
            args = payload.get("arguments") or {}
        call = payload.get("call") if isinstance(payload.get("call"), dict) else {}
        if not call and isinstance(payload.get("call_id"), str):
            call = {"call_id": payload.get("call_id")}

        logger.info(
            "ai_pointcare retell function name=%s call_id=%s",
            name,
            (call or {}).get("call_id") or "",
        )
        result = handle_retell_function(name, args, call)
        return Response(result, status=200)
