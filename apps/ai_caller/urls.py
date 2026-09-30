from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.ai_caller.views import (
    CallViewSet,
    PatientViewSet,
    PlaceOutboundCallView,
    RetellWebhookView,
)

router = DefaultRouter()
router.register("patients", PatientViewSet, basename="patients")
router.register("calls", CallViewSet, basename="calls")

urlpatterns = [
    path(
        "outbound/",
        PlaceOutboundCallView.as_view(),
        name="care-call-retell-outbound",
    ),
    path(
        "webhooks/retell/",
        RetellWebhookView.as_view(),
        name="retell-webhook",
    ),
    path("", include(router.urls)),
]
