"""Retell agent settings for Pointcare Collect Mode (self-contained).

Previously loaded from form_filler/retell_services.py. Prompt/greeting for the
Custom LLM path live in live_fill_prompt.py; this module only supplies the
agent call settings body used by retell_call._ensure_agent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def pointcare_agent_settings() -> dict[str, Any]:
    """Default Retell agent settings for outbound Pointcare eligibility calls."""
    return {
        "language": "en-US",
        "begin_message_delay_ms": 800,
        "call_screening_option": {
            "agent_identity": "Kyle",
            "call_purpose": (
                "calling from Bay Area Community Health to assist with "
                "Pointcare registration"
            ),
        },
        "voicemail_option": {
            "detection_prompt": (
                "Treat only a classic answering-machine or carrier voicemail "
                "greeting as voicemail. Do not treat iPhone Silence Unknown "
                "Callers, Siri or Google call screening, or a live person "
                "as voicemail."
            ),
            "action": {
                "type": "static_text",
                "text": (
                    "Hi, this is Kyle, an AI agent calling from Bay Area "
                    "Community Health to assist with Pointcare registration. "
                    "Please call {{callback_number}} when you have a moment."
                ),
            },
        },
        "webhook_events": [
            "call_started",
            "transcript_updated",
            "call_ended",
            "call_analyzed",
        ],
    }


# Back-compat aliases (unused by Custom LLM path; kept for older call sites).
@lru_cache(maxsize=1)
def pointcare_prompt() -> str:
    from .live_fill_prompt import LIVE_FILL_APPENDIX

    return LIVE_FILL_APPENDIX.strip()


@lru_cache(maxsize=1)
def pointcare_greeting() -> str:
    from .live_fill_prompt import LIVE_FILL_GREETING

    return LIVE_FILL_GREETING


def reload_pointcare_script() -> None:
    pointcare_prompt.cache_clear()
    pointcare_greeting.cache_clear()
    pointcare_agent_settings.cache_clear()
