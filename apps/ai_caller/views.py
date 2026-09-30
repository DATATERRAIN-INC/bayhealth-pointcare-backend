from django.http import HttpResponse
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.ai_caller.models import Call, PatientSource
from apps.ai_caller.serializers import (
    CallSerializer,
    CallerSettingsSerializer,
    PatientSerializer,
    PlaceOutboundCallSerializer,
)
from apps.ai_caller.services import (
    get_call_queryset,
    get_caller_settings,
    get_patient_queryset,
    place_outbound_call_for_patient,
    sync_call_transcript,
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
            search=self.request.query_params.get("search", ""),
            source=self.request.query_params.get("source", ""),
            is_blocked=is_blocked,
        )

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        patient = serializer.save(source=PatientSource.MANUAL)
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
        patient = serializer.save()

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
        _, errors = upload_patients_from_file(request.FILES.get("file"))
        if errors:
            field = next(iter(errors))
            value = errors[field]
            detail = value[0] if isinstance(value, (list, tuple)) and value else value
            return error_response(str(detail))
        return message_response("Patients uploaded successfully.", 201)

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
            search=self.request.query_params.get("search", ""),
            source=self.request.query_params.get("source", ""),
            status=self.request.query_params.get("status", ""),
            patient_id=self.request.query_params.get("patient_id", ""),
            retell_call_id=self.request.query_params.get("retell_call_id", ""),
        )

    def list(self, request, *args, **kwargs):
        retell_call_id = (request.query_params.get("retell_call_id") or "").strip()
        if retell_call_id:
            call = self.get_queryset().first()
            if not call:
                return error_response("Call not found.", 404)
            return Response({"transcript": call.transcript or []})
        return super().list(request, *args, **kwargs)

    @action(detail=False, methods=["get"], url_path="summary")
    def summary(self, request):
        queryset = Call.objects.all()
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

        result = place_outbound_call_for_patient(serializer.validated_data["id"])
        if not result.get("ok"):
            return error_response(
                result.get("error") or "Failed to place call.",
                int(result.get("status_code") or 502),
            )

        if result.get("flow") == "guardian":
            return message_response("Guardian call placed successfully.")
        return message_response("Call placed successfully.")


class CallerSettingsView(APIView):
    """GET/PATCH controller settings for the automated AI caller."""

    def get(self, request):
        settings_obj = get_caller_settings()
        return Response(CallerSettingsSerializer(settings_obj).data)

    def patch(self, request):
        settings_obj = get_caller_settings()
        serializer = CallerSettingsSerializer(
            settings_obj, data=request.data, partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(CallerSettingsSerializer(settings_obj).data)

    def put(self, request):
        return self.patch(request)


class RetellWebhookView(APIView):
    """Receive Retell call events and store status/transcript. Public (no auth)."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        _, error = update_call_from_retell_payload(request.data)
        if error:
            return error_response(error, 404 if error == "Call not found." else 400)
        return message_response("Call updated successfully.")
