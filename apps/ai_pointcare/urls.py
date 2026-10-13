from django.urls import path

from apps.ai_pointcare.views import PlaceMediCalCallView, RetellFunctionWebhookView


def _pair(route, view, name=None):
    named = {"name": name} if name else {}
    return [
        path(f"{route}/", view, **named),
        path(route, view),
    ]


urlpatterns = [
    *_pair("call", PlaceMediCalCallView.as_view(), "ai-pointcare-medi-cal-call"),
    *_pair(
        "retell/function",
        RetellFunctionWebhookView.as_view(),
        "ai-pointcare-retell-function",
    ),
]
