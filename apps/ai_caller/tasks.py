from celery import shared_task

from apps.ai_caller.services import (
    run_scheduled_outbound_calls,
    sync_in_progress_calls_from_retell,
)


@shared_task(name="ai_caller.process_outbound_calls")
def process_outbound_calls():
    """Periodic dialer: respects calls_enabled + calling window settings."""
    return run_scheduled_outbound_calls()


@shared_task(name="ai_caller.sync_in_progress_calls")
def sync_in_progress_calls():
    """Pull status/transcript from Retell when webhooks are missed (incl. callback)."""
    synced = sync_in_progress_calls_from_retell(limit=50)
    return {"ok": True, "synced": synced}


@shared_task(name="ai_caller.offer_callback_after_long_wait")
def offer_callback_after_long_wait(session_id, wait_token):
    """Patient still on hold after the max wait → offer a callback."""
    from apps.ai_caller.twilio_bridge import offer_callback_to_long_waiter

    return {"offered": offer_callback_to_long_waiter(session_id, wait_token)}


@shared_task(name="ai_caller.sync_in_progress_calls_for_user")
def sync_in_progress_calls_for_user(user_id, limit=10):
    """Non-blocking per-user sync triggered from list/summary (debounced)."""
    from django.contrib.auth import get_user_model

    user = get_user_model().objects.filter(pk=user_id).first()
    if not user:
        return {"ok": False, "synced": 0}
    synced = sync_in_progress_calls_from_retell(user=user, limit=limit)
    return {"ok": True, "synced": synced}
