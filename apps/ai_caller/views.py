from django.http import HttpResponse
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.ai_caller.models import Call, PatientSource, ScheduledOutreach
from apps.ai_caller.scheduling import (
    get_scheduled_outreach_queryset,
    schedule_callback_request,
)
from apps.ai_caller.serializers import (
    CallSerializer,
    CallerSettingsSerializer,
    PatientSerializer,
    PlaceOutboundCallSerializer,
    ScheduledOutreachSerializer,
)
from apps.ai_caller.services import (
    get_call_queryset,
    get_caller_settings,
    get_patient_queryset,
    place_outbound_call_for_patient,
    save_call_decline_reason,
    sync_call_transcript,
    sync_in_progress_calls_from_retell,
    update_call_from_retell_payload,
    upload_patients_from_file,
)
from common.excel import build_patient_template_bytes
from common.pagination import CommonPagination
from common.responses import error_response, message_response


class PatientViewSet(viewsets.ModelViewSet):
    serializer_class = PatientSerializer
    pagination_class = CommonPagination

    def get_queryset(self):
        blocked = self.request.query_params.get("is_blocked")
        is_blocked = None
        if blocked is not None and str(blocked).strip() != "":
            is_blocked = str(blocked).strip().lower() in ("1", "true", "yes")
        return get_patient_queryset(
            user=self.request.user,
            search=self.request.query_params.get("search", ""),
            source=self.request.query_params.get("source", ""),
            is_blocked=is_blocked,
        )

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        patient = serializer.save(
            source=PatientSource.MANUAL,
            user=request.user,
            created_by=request.user,
            updated_by=request.user,
        )
        from apps.notifications.services import notify_patient_created

        notify_patient_created(patient)
        return message_response("Patient saved successfully.", 201)

    def update(self, request, *args, **kwargs):
        patient = self.get_object()
        was_blocked = patient.is_blocked
        serializer = self.get_serializer(
            patient, data=request.data, partial=kwargs.pop("partial", False)
        )
        serializer.is_valid(raise_exception=True)
        changed_fields = sorted(serializer.validated_data.keys())
        patient = serializer.save(updated_by=request.user)

        from apps.notifications.services import (
            notify_patient_block_toggle,
            notify_patient_updated,
        )

        if "is_blocked" in changed_fields and patient.is_blocked != was_blocked:
            notify_patient_block_toggle(patient)
        elif changed_fields:
            notify_patient_updated(patient, changed_fields=changed_fields)
        return message_response("Patient updated successfully.")

    def destroy(self, request, *args, **kwargs):
        self.get_object().delete()
        return message_response("Patient deleted successfully.")

    @action(detail=False, methods=["post"], url_path="upload")
    def upload(self, request):
        result, errors = upload_patients_from_file(
            request.FILES.get("file"), user=request.user
        )
        if errors:
            field = next(iter(errors))
            value = errors[field]
            detail = value[0] if isinstance(value, (list, tuple)) and value else value
            return error_response(str(detail))

        uploaded = int((result or {}).get("uploaded") or 0)
        failed = int((result or {}).get("failed") or 0)
        skipped = (result or {}).get("skipped") or []
        payload = {
            "message": (
                "Patients uploaded successfully."
                if uploaded and not failed
                else (
                    "Upload finished with some row errors."
                    if uploaded
                    else "Upload finished but no patients were inserted."
                )
            ),
            "upload_id": (result or {}).get("upload_id"),
            "uploaded": uploaded,
            "failed": failed,
            "skipped": skipped,
        }
        # File accepted but every row failed validation (common on server).
        if uploaded == 0:
            return Response(payload, status=400)
        return Response(payload, status=201)

    @action(detail=False, methods=["get"], url_path="template")
    def template(self, request):
        response = HttpResponse(
            build_patient_template_bytes(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        response["Content-Disposition"] = 'attachment; filename="patient_upload_template.xlsx"'
        return response


class CallViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = CallSerializer
    pagination_class = CommonPagination

    def get_queryset(self):
        return get_call_queryset(
            user=self.request.user,
            search=self.request.query_params.get("search", ""),
            source=self.request.query_params.get("source", ""),
            status=self.request.query_params.get("status", ""),
            patient_id=self.request.query_params.get("patient_id", ""),
            retell_call_id=self.request.query_params.get("retell_call_id", ""),
        )

    def list(self, request, *args, **kwargs):
        # Keep statuses fresh even if Retell webhooks were missed.
        sync_in_progress_calls_from_retell(user=request.user, limit=25)
        retell_call_id = (request.query_params.get("retell_call_id") or "").strip()
        if retell_call_id:
            call = self.get_queryset().first()
            if not call:
                return error_response("Call not found.", 404)
            return Response({"transcript": call.transcript or []})
        return super().list(request, *args, **kwargs)

    @action(detail=False, methods=["get"], url_path="summary")
    def summary(self, request):
        sync_in_progress_calls_from_retell(user=request.user, limit=25)
        queryset = Call.objects.filter(user=request.user)
        return Response(
            {
                "all": queryset.count(),
                "completed": queryset.filter(status=Call.Status.COMPLETED).count(),
                "in_progress": queryset.filter(status=Call.Status.IN_PROGRESS).count(),
                "not_attended": queryset.filter(status=Call.Status.NOT_ATTENDED).count(),
            }
        )

    @action(detail=True, methods=["post"], url_path="sync-transcript")
    def sync_transcript(self, request, pk=None):
        call, error = sync_call_transcript(self.get_object())
        if error:
            return error_response(error)
        return message_response("Transcript synced successfully.")


class PlaceOutboundCallView(APIView):
    def post(self, request):
        serializer = PlaceOutboundCallSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        result = place_outbound_call_for_patient(
            serializer.validated_data["id"], user=request.user
        )
        if not result.get("ok"):
            return error_response(
                result.get("error") or "Failed to place call.",
                int(result.get("status_code") or 502),
            )

        if result.get("flow") == "guardian":
            return message_response("Guardian call placed successfully.")
        return message_response("Call placed successfully.")


class ScheduledOutreachViewSet(viewsets.ReadOnlyModelViewSet):
    """Separate list for patient callbacks and reminder queue rows."""

    serializer_class = ScheduledOutreachSerializer
    pagination_class = CommonPagination

    def get_queryset(self):
        return get_scheduled_outreach_queryset(
            user=self.request.user,
            search=self.request.query_params.get("search", ""),
            kind=self.request.query_params.get("kind", ""),
            status=self.request.query_params.get("status", ""),
        )

    @action(detail=False, methods=["get"], url_path="summary")
    def summary(self, request):
        queryset = ScheduledOutreach.objects.filter(user=request.user)
        return Response(
            {
                "all": queryset.count(),
                "scheduled": queryset.filter(
                    status=ScheduledOutreach.Status.SCHEDULED
                ).count(),
                "callback_requested": queryset.filter(
                    kind=ScheduledOutreach.Kind.CALLBACK_REQUESTED,
                    status=ScheduledOutreach.Status.SCHEDULED,
                ).count(),
                "reminder": queryset.filter(
                    kind=ScheduledOutreach.Kind.REMINDER,
                    status=ScheduledOutreach.Status.SCHEDULED,
                ).count(),
                "triggered": queryset.filter(
                    status=ScheduledOutreach.Status.TRIGGERED
                ).count(),
                "cancelled": queryset.filter(
                    status=ScheduledOutreach.Status.CANCELLED
                ).count(),
                "failed": queryset.filter(
                    status=ScheduledOutreach.Status.FAILED
                ).count(),
            }
        )

    @action(detail=True, methods=["post"], url_path="cancel")
    def cancel(self, request, pk=None):
        row = self.get_object()
        if row.status != ScheduledOutreach.Status.SCHEDULED:
            return error_response("Only scheduled items can be cancelled.", 400)
        row.status = ScheduledOutreach.Status.CANCELLED
        row.error_message = "Cancelled by user"
        row.save(update_fields=["status", "error_message", "updated_at"])
        return message_response("Scheduled outreach cancelled.")


class CallerSettingsView(APIView):
    """GET/PATCH controller settings for the automated AI caller."""

    def get(self, request):
        try:
            settings_obj = get_caller_settings(request.user)
        except ValueError as exc:
            return error_response(str(exc), 400)
        return Response(CallerSettingsSerializer(settings_obj).data)

    def patch(self, request):
        try:
            settings_obj = get_caller_settings(request.user)
        except ValueError as exc:
            return error_response(str(exc), 400)
        serializer = CallerSettingsSerializer(
            settings_obj, data=request.data, partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save(updated_by=request.user)
        return Response(CallerSettingsSerializer(settings_obj).data)

    def put(self, request):
        return self.patch(request)


class RetellWebhookView(APIView):
    """Receive Retell call events and store status/transcript. Public (no auth)."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        data = request.data if isinstance(request.data, dict) else {}
        call, error = update_call_from_retell_payload(data)
        if error:
            return error_response(error, 404 if error == "Call not found." else 400)

        # On end/analyze events, refresh from Retell get-call so status/transcript stick.
        event = str(data.get("event") or data.get("name") or "").strip().lower()
        if call and event in {"call_ended", "call_analyzed"}:
            sync_call_transcript(call)
        return message_response("Call updated successfully.")


class RetellToolWebhookView(APIView):
    """Receive Retell custom function calls (decline / callback). Public."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        data = request.data if isinstance(request.data, dict) else {}
        name = str(
            data.get("name")
            or data.get("tool_name")
            or data.get("function_name")
            or ""
        ).strip()
        call_id = str(
            data.get("call_id")
            or data.get("retell_call_id")
            or (data.get("call") or {}).get("call_id")
            or ""
        ).strip()
        args = data.get("args") or data.get("arguments") or data.get("parameters") or {}
        if isinstance(args, str):
            try:
                import json

                args = json.loads(args)
            except Exception:
                args = {"reason": args}

        if name == "log_callback_request":
            callback_time = ""
            if isinstance(args, dict):
                callback_time = str(
                    args.get("callback_time")
                    or args.get("time")
                    or args.get("when")
                    or args.get("reason")
                    or ""
                ).strip()
            if not callback_time:
                callback_time = str(
                    data.get("callback_time") or data.get("time") or ""
                ).strip()
            call = Call.objects.filter(retell_call_id=call_id).first()
            if not call:
                return error_response("Call not found.", 404)
            _, error = schedule_callback_request(
                call=call, raw_time_text=callback_time
            )
            if error:
                return error_response(error, 400)
            return message_response("Callback request saved.")

        if name and name != "log_decline_reason":
            return message_response("Tool ignored.")

        reason = ""
        if isinstance(args, dict):
            reason = str(args.get("reason") or args.get("decline_reason") or "").strip()
        if not reason:
            reason = str(data.get("reason") or "").strip()

        _, error = save_call_decline_reason(retell_call_id=call_id, reason=reason)
        if error:
            return error_response(error, 404 if error == "Call not found." else 400)
        return message_response("Decline reason saved.")
