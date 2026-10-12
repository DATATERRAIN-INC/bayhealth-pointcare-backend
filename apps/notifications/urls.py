from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.notifications.views import (
    ChannelAPIView,
    MqttMessageAPIView,
    NotificationViewSet,
    ScopeAPIView,
)

router = DefaultRouter()
router.register("", NotificationViewSet, basename="notifications")

urlpatterns = [
    path("get-topics/", MqttMessageAPIView.as_view(), name="get_topics"),
    path("get-mqtt-messages/", MqttMessageAPIView.as_view(), name="get_mqtt_message"),
    path("insert-scope/", ScopeAPIView.as_view(), name="insert_scope"),
    path("insert-channel/", ChannelAPIView.as_view(), name="insert_channel"),
    path("", include(router.urls)),
]
