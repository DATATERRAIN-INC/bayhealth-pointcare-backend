from django.http import HttpResponse
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView

from apps.ai_caller.models import PatientSource
from apps.ai_caller.retell import place_retell_care_call, place_retell_guardian_call
from apps.ai_caller.serializers import (
    PatientSerializer,
    PlaceRetellCareCallSerializer,
    PlaceRetellGuardianCallSerializer,
)
from apps.ai_caller.services import get_patient_queryset, upload_patients_from_file
from common.excel import build_patient_template_bytes
from common.responses import error_response, message_response


class PatientViewSet(viewsets.ModelViewSet):
    """CRUD for patient records, plus Excel upload and template download."""

    serializer_class = PatientSerializer
    permission_classes = [AllowAny]
    authentication_classes = []

    def get_queryset(self):
        return get_patient_queryset(
            search=self.request.query_params.get("search", ""),
            source=self.request.query_params.get("source", ""),
        )

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(source=PatientSource.MANUAL)
        return message_response(
            "Patient saved successfully.",
            code=status.HTTP_201_CREATED,
        )

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop("partial", False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return message_response("Patient updated successfully.")

    def destroy(self, request, *args, **kwargs):
        self.get_object().delete()
        return message_response("Patient deleted successfully.")

    @action(detail=False, methods=["post"], url_path="upload")
    def upload(self, request):
        result, errors = upload_patients_from_file(request.FILES.get("file"))
        if errors:
            first_field = next(iter(errors))
            first_error = errors[first_field]
            if isinstance(first_error, (list, tuple)) and first_error:
                detail = first_error[0]
            else:
                detail = first_error
            return error_response(str(detail))
        return message_response(
            "Patients uploaded successfully.",
            code=status.HTTP_201_CREATED,
        )

    @action(detail=False, methods=["get"], url_path="template")
    def template(self, request):
        content = build_patient_template_bytes()
        response = HttpResponse(
            content,
            content_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        )
        response["Content-Disposition"] = (
            'attachment; filename="patient_upload_template.xlsx"'
        )
        return response


class PlaceRetellCareCallView(APIView):
    """Place an outbound phone call through Retell AI."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = PlaceRetellCareCallSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        result = place_retell_care_call(
            phone_number=serializer.validated_data["phone_number"],
            name=serializer.validated_data.get("name") or "",
            patient_name=serializer.validated_data.get("patient_name") or "",
            service_name=serializer.validated_data.get("service_name") or "",
            address_on_file=serializer.validated_data.get("address_on_file") or "",
            insurance_name=serializer.validated_data.get("insurance_name") or "",
            agent_id=serializer.validated_data.get("agent_id") or "",
            transfer_number=serializer.validated_data.get("transfer_number") or "",
        )
        if not result.get("ok"):
            code = int(result.get("status_code") or status.HTTP_502_BAD_GATEWAY)
            return error_response(
                result.get("error") or "Failed to place call.",
                code=code,
            )

        return message_response("Call placed successfully.")


class PlaceRetellGuardianCallView(APIView):
    """Place an outbound call to a parent or guardian of a minor patient."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = PlaceRetellGuardianCallSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        result = place_retell_guardian_call(
            phone_number=data["phone_number"],
            patient_name=data.get("patient_name") or "",
            guardian_name=data.get("guardian_name") or "",
            insurance_name=data.get("insurance_name") or "",
            measure_name=data.get("measure_name") or "",
            service_name=data.get("service_name") or "",
            address_on_file=data.get("address_on_file") or "",
            clinic_name=data.get("clinic_name") or "",
            appointment_date=data.get("appointment_date") or "",
            appointment_time=data.get("appointment_time") or "",
            provider_name=data.get("provider_name") or "",
            transfer_number=data["transfer_number"],
        )
        if not result.get("ok"):
            code = int(result.get("status_code") or status.HTTP_502_BAD_GATEWAY)
            return error_response(
                result.get("error") or "Failed to place call.",
                code=code,
            )

        return message_response("Guardian call placed successfully.")
