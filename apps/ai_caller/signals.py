"""Publish MQTT when a call status changes. QuerySet.update() does not fire these."""

from django.db.models.signals import post_delete, post_save, pre_save
from django.dispatch import receiver

from apps.ai_caller.models import Call
from apps.notifications.mqtt import publish_call_change


@receiver(pre_save, sender=Call)
def remember_call_status(sender, instance, **kwargs):
    update_fields = kwargs.get("update_fields")
    if update_fields is not None and "status" not in update_fields:
        instance._previous_status = instance.status
        return
    if not instance.pk:
        instance._previous_status = None
        return
    instance._previous_status = (
        Call.objects.filter(pk=instance.pk).values_list("status", flat=True).first()
    )


@receiver(post_save, sender=Call)
def publish_call_status_on_save(sender, instance, created, **kwargs):
    previous = getattr(instance, "_previous_status", None)
    if not created and previous == instance.status:
        return
    publish_call_change(instance, previous)


@receiver(post_delete, sender=Call)
def publish_call_status_on_delete(sender, instance, **kwargs):
    publish_call_change(instance, instance.status)
