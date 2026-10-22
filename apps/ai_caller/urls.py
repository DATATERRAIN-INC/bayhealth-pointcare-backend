from django.urls import include, path
from rest_framework.routers import DefaultRouter, Route

from apps.ai_caller.views import (
    CallViewSet,
    CallerSettingsView,
    PatientViewSet,
    PlaceOutboundCallView,
    RetellToolWebhookView,
    RetellWebhookView,
    ScheduledOutreachViewSet,
)
from apps.ai_caller import twiml_views


class CallBulkUpdateRouter(DefaultRouter):
    """List route also accepts PUT/PATCH (ids come from the request body)."""

    routes = [
        Route(
            url=r"^{prefix}{trailing_slash}$",
            mapping={
                "get": "list",
                "post": "create",
                "put": "update",
                "patch": "partial_update",
            },
            name="{basename}-list",
            detail=False,
            initkwargs={"suffix": "List"},
        ),
        *DefaultRouter.routes[1:],
    ]


router = DefaultRouter()
router.register("patients", PatientViewSet, basename="patients")
router.register(
    "scheduled-outreach",
    ScheduledOutreachViewSet,
    basename="scheduled-outreach",
)

call_router = CallBulkUpdateRouter()
call_router.register("calls", CallViewSet, basename="calls")

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
        "twilio/warm-transfer/queue-tone/",
        twiml_views.warm_transfer_queue_tone,
        name="twilio-warm-transfer-queue-tone",
    ),
    path(
        "twilio/warm-transfer/queue-tone-audio/",
        twiml_views.warm_transfer_queue_tone_audio,
        name="twilio-warm-transfer-queue-tone-audio",
    ),
    path(
        "twilio/warm-transfer/wait/",
        twiml_views.warm_transfer_wait,
        name="twilio-warm-transfer-wait",
    ),
    path(
        "twilio/warm-transfer/next-agent/",
        twiml_views.warm_transfer_next_agent,
        name="twilio-warm-transfer-next-agent",
    ),
    path(
        "twilio/warm-transfer/hold/",
        twiml_views.warm_transfer_hold,
        name="twilio-warm-transfer-hold",
    ),
    path(
        "twilio/warm-transfer/connecting/",
        twiml_views.warm_transfer_connecting,
        name="twilio-warm-transfer-connecting",
    ),
    path(
        "twilio/warm-transfer/ask-callback/",
        twiml_views.warm_transfer_ask_callback,
        name="twilio-warm-transfer-ask-callback",
    ),
    path(
        "twilio/warm-transfer/long-wait-offer/",
        twiml_views.warm_transfer_long_wait_offer,
        name="twilio-warm-transfer-long-wait-offer",
    ),
    path(
        "twilio/warm-transfer/long-wait-choice/",
        twiml_views.warm_transfer_long_wait_choice,
        name="twilio-warm-transfer-long-wait-choice",
    ),
    path(
        "twilio/warm-transfer/callback-gather/",
        twiml_views.warm_transfer_callback_gather,
        name="twilio-warm-transfer-callback-gather",
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
    path("", include(call_router.urls)),
    path("", include(router.urls)),
]
