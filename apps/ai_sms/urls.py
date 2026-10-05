from django.urls import path

from apps.ai_sms.views import (
    SmsAgentDialAnswerView,
    SmsAgentDialStatusView,
    SmsWebhookView,
    StartMinorSmsConversationView,
    StartSmsConversationView,
)

urlpatterns = [
    path(
        "outbound/",
        StartSmsConversationView.as_view(),
        name="ai-sms-outbound",
    ),
    path(
        "minor/outbound/",
        StartMinorSmsConversationView.as_view(),
        name="ai-sms-minor-outbound",
    ),
    path(
        "webhook/",
        SmsWebhookView.as_view(),
        name="ai-sms-webhook",
    ),
    path(
        "agent-dial/answer/<str:chat_id>/",
        SmsAgentDialAnswerView.as_view(),
        name="ai-sms-agent-dial-answer",
    ),
    path(
        "agent-dial/status/<str:chat_id>/",
        SmsAgentDialStatusView.as_view(),
        name="ai-sms-agent-dial-status",
    ),
]
