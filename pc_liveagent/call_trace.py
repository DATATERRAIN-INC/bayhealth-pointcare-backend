"""Bounded per-call structured event log (no emails/passwords/OTPs/PHI)."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger(__name__)

_MAX_EVENTS = 200
_events: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=_MAX_EVENTS))


def trace_event(call_id: str, event: str, **fields: Any) -> None:
    """Append a structured event and mirror a compact line to the logger."""
    if not call_id:
        return
    safe: dict[str, Any] = {"ts": time.time(), "event": event, "call_id": call_id}
    for key, value in fields.items():
        if key in ("email", "password", "otp", "code", "secret"):
            continue
        if isinstance(value, str) and len(value) > 160:
            value = value[:157] + "..."
        safe[key] = value
    _events[call_id].append(safe)
    parts = [f"{k}={v!r}" for k, v in safe.items() if k not in ("ts", "event", "call_id")]
    logger.info("state_event event=%s call_id=%s %s", event, call_id, " ".join(parts))


def get_call_trace(call_id: str) -> list[dict[str, Any]]:
    return list(_events.get(call_id) or ())


def clear_call_trace(call_id: str) -> None:
    _events.pop(call_id, None)
