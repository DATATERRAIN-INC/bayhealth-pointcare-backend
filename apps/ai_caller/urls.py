from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.ai_caller.views import (
    PatientViewSet,
    PlaceRetellCareCallView,
    PlaceRetellGuardianCallView,
)

router = DefaultRouter()
router.register("patients", PatientViewSet, basename="patients")

urlpatterns = [
    path(
        "outbound/",
        PlaceRetellCareCallView.as_view(),
        name="care-call-retell-outbound",
    ),
    path(
        "minor/outbound/",
        PlaceRetellGuardianCallView.as_view(),
        name="care-call-minor-outbound",
    ),
    path("", include(router.urls)),
]
