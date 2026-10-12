# -*- coding: utf-8 -*-
from typing import Optional

from django.http import HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.ai_sms.models import SmsConversation
from apps.ai_sms.serializers import (
    PlaceMinorSmsConversationSerializer,
    PlaceSmsConversationSerializer,
    SmsConversationListSerializer,
)
from common.pagination import CommonPagination
from apps.ai_sms.services import (
    reply_to_inbound_sms,
    sms_webhook_url,
    start_minor_sms_conversation_for_patient,
    start_sms_conversation_for_patient,
    twilio_signature_is_valid,
)
from apps.ai_sms.agent_dial import (
    agent_answer_twiml,
    conversation_by_chat_id,
    handle_agent_dial_recording,
    handle_bridge_status,
    maybe_failover_on_agent_status,
)
from apps.users.authentication import CognitoBearerAuthentication
from common.responses import error_response

_EMPTY_TWIML = "<Response></Response>"


def _request_actor(request):
    user = getattr(request, "user", None)
    if user is not None and getattr(user, "is_authenticated", False):
        return user
    return None


def _sms_transcript_items(raw: str) -> list:
    """Convert stored Agent/Patient text into voice-style transcript objects."""
    speaker_map = {
        "agent": "agent",
        "patient": "patient",
        "guardian": "patient",
    }
    items = []
    for line in (raw or "").splitlines():
        text = line.strip()
        if not text:
            continue
        speaker = "unknown"
        body = text
        if ":" in text:
            label, rest = text.split(":", 1)
            mapped = speaker_map.get(label.strip().lower())
            if mapped:
                speaker = mapped
                body = rest.strip()
        if not body:
            continue
        items.append(
            {
                "at": None,
                "text": body,
                "segment": "ai" if speaker == "agent" else "human",
                "speaker": speaker,
                "name": (
                    "AI agent"
                    if speaker == "agent"
                    else "Patient"
                    if speaker == "patient"
                    else "Unknown"
                ),
            }
        )
    return items


def _live_call_transcript_items(raw) -> list:
    """Normalize stored live-call JSON turns (patient / live_agent)."""
    items = []
    for row in raw or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        speaker = str(row.get("speaker") or "").strip().lower()
        if speaker in {"provider"}:
            speaker = "live_agent"
        if speaker not in {"patient", "live_agent"}:
            continue
        items.append(
            {
                "at": row.get("at"),
                "text": text,
                "segment": "human",
                "speaker": speaker,
                "name": "Patient" if speaker == "patient" else "Live agent",
            }
        )
    return items


def _recording_url_for_conversation(conversation, request) -> Optional[str]:
    """Full bridge mix URL, or None when recording is disabled."""
    from apps.ai_caller.models import CallerSettings
    from apps.users.models import User

    user = getattr(request, "user", None)
    if not isinstance(user, User) or not getattr(user, "is_authenticated", False):
        user = getattr(conversation, "created_by", None)
    enabled = True
    if isinstance(user, User) and getattr(user, "pk", None):
        try:
            enabled = bool(CallerSettings.load(user).recording_enabled)
        except Exception:
            enabled = True
    if not enabled:
        return None
    return (conversation.recording_url or "").strip() or None


def _conversation_transcript_payload(conversation, request=None):
    recording_url = (
        _recording_url_for_conversation(conversation, request)
        if request is not None
        else ((conversation.recording_url or "").strip() or None)
    )
    # One timeline: SMS (agent/patient) then live-call (patient/live_agent).
    # Speakers already distinguish AI agent vs Patient vs Live agent.
    transcript = _sms_transcript_items(conversation.transcript or "")
    transcript.extend(
        _live_call_transcript_items(conversation.live_call_transcript or [])
    )
    return {
        "chat_id": conversation.chat_id,
        "transcript": transcript,
        "recording_url": recording_url,
    }


class SmsConversationListView(APIView):
    """GET SMS conversations created by (or owned via patient of) the auth user."""

    authentication_classes = [CognitoBearerAuthentication]
    permission_classes = [IsAuthenticated]
    renderer_classes = [JSONRenderer]

    def get(self, request):
        from apps.users.models import User

        user = request.user
        if not isinstance(user, User) or not getattr(user, "pk", None):
            return error_response(
                "Authentication required. Sign in again.",
                status.HTTP_401_UNAUTHORIZED,
            )

        # Filter by ids so we never hit auth.User vs users.User FK mismatches.
        qs = (
            SmsConversation.objects.filter(
                Q(created_by_id=user.pk) | Q(patient__user_id=user.pk)
            )
            .select_related("patient", "created_by", "updated_by")
            .distinct()
            .order_by("-created_at", "-id")
        )

        # ?chat_id=... → SMS + live-call transcript / recordings for that chat.
        chat_id = (request.query_params.get("chat_id") or "").strip()
        if chat_id:
            conversation = qs.filter(chat_id=chat_id).first()
            if conversation is None:
                return error_response(
                    "SMS conversation not found.",
                    status.HTTP_404_NOT_FOUND,
                )
            return Response(
                _conversation_transcript_payload(conversation, request),
                status=status.HTTP_200_OK,
            )

        status_filter = (request.query_params.get("status") or "").strip()
        if status_filter:
            qs = qs.filter(status=status_filter)
        flow_filter = (request.query_params.get("flow") or "").strip()
        if flow_filter:
            qs = qs.filter(flow=flow_filter)

        try:
            paginator = CommonPagination()
            page = paginator.paginate_queryset(qs, request, view=self)
            data = SmsConversationListSerializer(page, many=True).data
            return paginator.get_paginated_response(data)
        except ValueError:
            return error_response(
                "Unable to load SMS conversations for this account.",
                status.HTTP_400_BAD_REQUEST,
            )
        except Exception:
            return error_response(
                "Unable to load SMS conversations. Please try again.",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@method_decorator(csrf_exempt, name="dispatch")
class SmsConversationDetailView(APIView):
    """GET transcript by path or ?chat_id= query param (voice-compatible shape)."""

    authentication_classes = [CognitoBearerAuthentication]
    permission_classes = [AllowAny]
    renderer_classes = [JSONRenderer]

    def get(self, request, lookup: str = ""):
        chat_id = (request.query_params.get("chat_id") or "").strip()
        lookup = (lookup or "").strip()
        conversation = None
        if chat_id:
            conversation = conversation_by_chat_id(chat_id)
        elif lookup.isdigit():
            conversation = SmsConversation.objects.filter(pk=int(lookup)).first()
        elif lookup:
            conversation = conversation_by_chat_id(lookup)
        if conversation is None:
            return error_response("SMS conversation not found.", status.HTTP_404_NOT_FOUND)
        return Response(
            _conversation_transcript_payload(conversation, request),
            status=status.HTTP_200_OK,
        )


@method_decorator(csrf_exempt, name="dispatch")
class StartSmsConversationView(APIView):
    """Start an adult SMS conversation for a patient id from ai_caller_patient."""

    authentication_classes = [CognitoBearerAuthentication]
    permission_classes = [AllowAny]
    renderer_classes = [JSONRenderer]

    def post(self, request):
        serializer = PlaceSmsConversationSerializer(data=request.data)
        if not serializer.is_valid():
            return error_response(serializer.errors, status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        result = start_sms_conversation_for_patient(
            data["id"],
            user=_request_actor(request),
            agent_id=data.get("agent_id") or "",
            webhook_url=sms_webhook_url()
            or request.build_absolute_uri("/api/ai-sms/webhook/"),
        )
        if not result.get("ok"):
            code = int(result.get("status_code") or status.HTTP_502_BAD_GATEWAY)
            return error_response(
                result.get("error") or "Failed to start SMS.",
                code,
            )

        return Response(
            {
                "patient_id": result.get("patient_id"),
                "patient_name": result.get("patient_name") or "",
                "chat_id": result["chat_id"],
                "message_sid": result.get("message_sid") or "",
                "status": result.get("status") or "ongoing",
                "type": "SMS",
                "started_at": result.get("started_at"),
                "created_by": result.get("created_by"),
                "updated_by": result.get("updated_by"),
                "from_number": result.get("from_number") or None,
                "phone_last4": result.get("phone_last4") or "",
                "agent_id": result.get("agent_id") or None,
                "transfer_number": result.get("transfer_number") or "",
                "provider": "twilio",
                "webhook": sms_webhook_url()
                or request.build_absolute_uri("/api/ai-sms/webhook/"),
                "message": result.get("message") or "SMS conversation started.",
            },
            status=status.HTTP_200_OK,
        )


@method_decorator(csrf_exempt, name="dispatch")
class StartMinorSmsConversationView(APIView):
    """Start a guardian SMS for a patient id from ai_caller_patient."""

    authentication_classes = [CognitoBearerAuthentication]
    permission_classes = [AllowAny]
    renderer_classes = [JSONRenderer]

    def post(self, request):
        serializer = PlaceMinorSmsConversationSerializer(data=request.data)
        if not serializer.is_valid():
            return error_response(serializer.errors, status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        result = start_minor_sms_conversation_for_patient(
            data["id"],
            user=_request_actor(request),
            webhook_url=sms_webhook_url()
            or request.build_absolute_uri("/api/ai-sms/webhook/"),
        )
        if not result.get("ok"):
            code = int(result.get("status_code") or status.HTTP_502_BAD_GATEWAY)
            return error_response(
                result.get("error") or "Failed to start SMS.",
                code,
            )

        return Response(
            {
                "patient_id": result.get("patient_id"),
                "patient_name": result.get("patient_name") or "",
                "created_by": result.get("created_by"),
                "updated_by": result.get("updated_by"),
                "chat_id": result["chat_id"],
                "message_sid": result.get("message_sid") or "",
                "status": result.get("status") or "ongoing",
                "type": "SMS",
                "started_at": result.get("started_at"),
                "from_number": result.get("from_number") or None,
                "phone_last4": result.get("phone_last4") or "",
                "transfer_number": result.get("transfer_number") or "",
                "provider": "twilio",
                "flow": "minor",
                "webhook": sms_webhook_url()
                or request.build_absolute_uri("/api/ai-sms/webhook/"),
                "message": result.get("message") or "Minor SMS conversation started.",
            },
            status=status.HTTP_200_OK,
        )


@method_decorator(csrf_exempt, name="dispatch")
class SmsWebhookView(APIView):
    """Twilio calls this when the patient replies by text."""

    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request):
        params = {key: request.POST.get(key, "") for key in request.POST}
        signature = request.META.get("HTTP_X_TWILIO_SIGNATURE", "")
        signed_url = sms_webhook_url() or request.build_absolute_uri()
        if not twilio_signature_is_valid(signed_url, params, signature):
            return HttpResponse(status=403)

        reply_to_inbound_sms(
            from_number=params.get("From", ""),
            body=params.get("Body", ""),
            message_sid=params.get("MessageSid", ""),
        )
        return HttpResponse(_EMPTY_TWIML, content_type="text/xml")


@method_decorator(csrf_exempt, name="dispatch")
class SmsAgentDialAnswerView(APIView):
    """TwiML when a live agent answers an SMS-transfer dial — then bridge to patient."""

    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, chat_id: str):
        conversation = conversation_by_chat_id(chat_id)
        if conversation is None:
            return HttpResponse(_EMPTY_TWIML, content_type="text/xml")
        conversation.transfer_status = "bridging"
        conversation.save(update_fields=["transfer_status"])
        return HttpResponse(
            agent_answer_twiml(conversation),
            content_type="text/xml",
        )


@method_decorator(csrf_exempt, name="dispatch")
class SmsAgentDialBridgeStatusView(APIView):
    """Twilio Dial action: patient bridge result after agent answered."""

    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, chat_id: str):
        conversation = conversation_by_chat_id(chat_id)
        if conversation is None:
            return HttpResponse(_EMPTY_TWIML, content_type="text/xml")
        dial_status = (
            request.POST.get("DialCallStatus")
            or request.POST.get("CallStatus")
            or ""
        )
        handle_bridge_status(conversation, dial_call_status=dial_status)
        return HttpResponse(_EMPTY_TWIML, content_type="text/xml")


@method_decorator(csrf_exempt, name="dispatch")
class SmsAgentDialStatusView(APIView):
    """Twilio status callback for SMS live-agent dial / failover."""

    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, chat_id: str):
        conversation = conversation_by_chat_id(chat_id)
        if conversation is None:
            return HttpResponse(status=204)
        call_status = (
            request.POST.get("CallStatus")
            or request.POST.get("DialCallStatus")
            or ""
        )
        call_sid = request.POST.get("CallSid") or ""
        maybe_failover_on_agent_status(
            conversation,
            call_status=call_status,
            call_sid=call_sid,
        )
        return HttpResponse(status=204)


@method_decorator(csrf_exempt, name="dispatch")
class SmsAgentDialRecordingView(APIView):
    """Twilio recording callback after SMS live-agent ↔ patient Dial ends."""

    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, chat_id: str):
        conversation = conversation_by_chat_id(chat_id)
        if conversation is None:
            return HttpResponse(status=204)
        recording_url = (request.POST.get("RecordingUrl") or "").strip()
        recording_sid = (request.POST.get("RecordingSid") or "").strip()
        if recording_url:
            handle_agent_dial_recording(
                conversation,
                recording_url=recording_url,
                recording_sid=recording_sid,
            )
        return HttpResponse(status=204)
