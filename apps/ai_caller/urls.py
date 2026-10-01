from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.ai_caller.views import (
    CallViewSet,
    CallerSettingsView,
    PatientViewSet,
    PlaceOutboundCallView,
    RetellToolWebhookView,
    RetellWebhookView,
)
from apps.ai_caller import twiml_views

router = DefaultRouter()
router.register("patients", PatientViewSet, basename="patients")
router.register("calls", CallViewSet, basename="calls")

urlpatterns = [
    path(
        "settings/",
        CallerSettingsView.as_view(),
        name="ai-caller-settings",
    ),
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
    path(
        "webhooks/retell-tool/",
        RetellToolWebhookView.as_view(),
        name="retell-tool-webhook",
    ),
    path(
        "twilio/warm-transfer/answer/",
        twiml_views.warm_transfer_answer,
        name="twilio-warm-transfer-answer",
    ),
    path(
        "twilio/warm-transfer/inbound/",
        twiml_views.warm_transfer_inbound,
        name="twilio-warm-transfer-inbound",
    ),
    path(
        "twilio/warm-transfer/gather/",
        twiml_views.warm_transfer_gather,
        name="twilio-warm-transfer-gather",
    ),
    path(
        "twilio/warm-transfer/join/",
        twiml_views.warm_transfer_join,
        name="twilio-warm-transfer-join",
    ),
    path(
        "twilio/warm-transfer/speak-now/",
        twiml_views.warm_transfer_speak_now,
        name="twilio-warm-transfer-speak-now",
    ),
    path(
        "twilio/warm-transfer/conference-status/",
        twiml_views.warm_transfer_conference_status,
        name="twilio-warm-transfer-conference-status",
    ),
    path(
        "twilio/warm-transfer/dial-status/",
        twiml_views.warm_transfer_dial_status,
        name="twilio-warm-transfer-dial-status",
    ),
    path(
        "twilio/warm-transfer/recording/",
        twiml_views.warm_transfer_recording,
        name="twilio-warm-transfer-recording",
    ),
    path(
        "twilio/warm-transfer/status/",
        twiml_views.warm_transfer_status,
        name="twilio-warm-transfer-status",
    ),
    path("", include(router.urls)),
]
