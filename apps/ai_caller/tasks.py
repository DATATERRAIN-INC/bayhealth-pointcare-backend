from celery import shared_task

from apps.ai_caller.services import run_scheduled_outbound_calls


@shared_task(name="ai_caller.process_outbound_calls")
def process_outbound_calls():
    """Periodic dialer: respects calls_enabled + calling window settings."""
    return run_scheduled_outbound_calls()
