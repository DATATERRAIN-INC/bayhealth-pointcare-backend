from django.http import HttpResponse
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.ai_caller.models import Call, PatientSource
from apps.ai_caller.serializers import (
    CallSerializer,
    PatientSerializer,
    PlaceOutboundCallSerializer,
)
from apps.ai_caller.services import (
    get_call_queryset,
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
    permission_classes = [AllowAny]
    authentication_classes = []
    pagination_class = CommonPagination

    def get_queryset(self):
        return get_patient_queryset(
            search=self.request.query_params.get("search", ""),
            source=self.request.query_params.get("source", ""),
        )

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(source=PatientSource.MANUAL)
        return message_response("Patient saved successfully.", 201)

    def update(self, request, *args, **kwargs):
        patient = self.get_object()
        serializer = self.get_serializer(
            patient, data=request.data, partial=kwargs.pop("partial", False)
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
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
    permission_classes = [AllowAny]
    authentication_classes = []
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
    permission_classes = [AllowAny]
    authentication_classes = []

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


class RetellWebhookView(APIView):
    """Receive Retell call events and store status/transcript."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        _, error = update_call_from_retell_payload(request.data)
        if error:
            return error_response(error, 404 if error == "Call not found." else 400)
        return message_response("Call updated successfully.")
