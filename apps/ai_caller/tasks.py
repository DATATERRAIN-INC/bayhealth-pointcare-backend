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
    """Pull ended-call status/transcript from Retell when webhooks are missed."""
    synced = sync_in_progress_calls_from_retell(limit=50)
    return {"ok": True, "synced": synced}
