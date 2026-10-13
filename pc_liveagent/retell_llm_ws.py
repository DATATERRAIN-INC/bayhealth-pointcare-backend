"""Retell Custom LLM WebSocket — primary Collect Mode conversation path.

Protocol: https://docs.retellai.com/api-references/llm-websocket
Retell connects to: {llm_websocket_url}/{call_id}

agent_interrupt is the only mechanism used for the delayed post-SSO first
eligibility question. Normal response events handle response_required turns.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from .call_turn import (
    SSO_BUSY,
    begin_speech,
    deliver_first_eligibility_question,
    get_controller,
    latest_user_utterance,
    process_user_turn,
    recover_pending_outbound,
    sync_controller_to_eligibility,
)

logger = logging.getLogger(__name__)

# Active Custom LLM sockets — used to push Stage D after late Google SSO.
_active_ws: dict[str, WebSocket] = {}


def _send(ws: WebSocket, payload: dict[str, Any]) -> Any:
    return ws.send_text(json.dumps(payload))


def websocket_connected(call_id: str) -> bool:
    return call_id in _active_ws and _active_ws.get(call_id) is not None


async def speak_now(
    call_id: str,
    content: str,
    *,
    interrupt_id: int | None = None,
) -> bool:
    """Proactively speak via agent_interrupt (Retell Custom LLM protocol).

    Returns True only if the WebSocket send succeeded — not that the caller heard it.
    """
    text = (content or "").strip()
    ws = _active_ws.get(call_id)
    if not text:
        logger.warning("agent_interrupt skipped call_id=%s reason=empty", call_id)
        return False
    if ws is None:
        logger.warning(
            "agent_interrupt skipped call_id=%s reason=disconnected speech=%r",
            call_id,
            text[:80],
        )
        return False
    iid = int(interrupt_id) if interrupt_id is not None else 1
    payload = {
        "response_type": "agent_interrupt",
        "interrupt_id": iid,
        "content": text,
        "content_complete": True,
        "no_interruption_allowed": True,
    }
    try:
        await _send(ws, payload)
        logger.info(
            "agent_interrupt sent call_id=%s interrupt_id=%s speech=%r",
            call_id,
            iid,
            text[:80],
        )
        return True
    except Exception:
        logger.exception(
            "agent_interrupt failed call_id=%s interrupt_id=%s", call_id, iid
        )
        return False


async def handle_retell_llm_websocket(websocket: WebSocket, call_id: str) -> None:
    await websocket.accept()
    _active_ws[call_id] = websocket
    ctrl = await get_controller(call_id)
    ctrl.active_websocket_status = "connected"
    logger.info(
        "socket_connected call_id=%s phase=%s elig_op=%s pending_outbound=%s",
        call_id,
        ctrl.phase,
        ctrl.elig_op_in_progress,
        bool(ctrl.pending_outbound_speech),
    )
    latest_response_id = 0

    try:
        # Config + begin message (response_id 0).
        await _send(
            websocket,
            {
                "response_type": "config",
                "config": {
                    "auto_reconnect": True,
                    "call_details": True,
                    "transcript_with_tool_calls": True,
                },
            },
        )
        greeting = await begin_speech(call_id)
        await _send(
            websocket,
            {
                "response_type": "response",
                "response_id": 0,
                "content": greeting,
                "content_complete": True,
            },
        )
        # Reconnect after SSO: if marital is ready but never interrupt-delivered, do it once.
        ctrl0 = await get_controller(call_id)
        if (
            ctrl0.phase == "eligibility"
            and ctrl0.eligibility_screen_verified
            and not ctrl0.eligibility_question_sent
            and not ctrl0.sso_in_progress
        ):
            await deliver_first_eligibility_question(call_id, speech=greeting)
        # Documented recovery: verified next question queued while socket was down.
        await recover_pending_outbound(call_id)

        while True:
            raw = await websocket.receive_text()
            try:
                request = json.loads(raw)
            except Exception:
                logger.warning("Custom LLM WS non-JSON call_id=%s", call_id)
                continue
            if not isinstance(request, dict):
                continue

            interaction = str(request.get("interaction_type") or "").strip()

            if interaction == "ping_pong":
                await _send(
                    websocket,
                    {
                        "response_type": "ping_pong",
                        "timestamp": request.get("timestamp"),
                    },
                )
                continue

            if interaction == "call_details":
                logger.info("Custom LLM WS call_details call_id=%s", call_id)
                continue

            if interaction == "update_only":
                # Quiet promote only; speech goes through agent_interrupt when ready.
                c = await get_controller(call_id)
                if not c.sso_in_progress and not c.eligibility_question_sent:
                    speech = await sync_controller_to_eligibility(call_id)
                    if speech:
                        await deliver_first_eligibility_question(
                            call_id, speech=speech
                        )
                continue

            if interaction in ("response_required", "reminder_required"):
                response_id = int(request.get("response_id") or 0)
                latest_response_id = response_id
                ctrl_r = await get_controller(call_id)
                ctrl_r.active_response_id = response_id
                transcript = request.get("transcript") or request.get("transcript_object")
                if not isinstance(transcript, list):
                    transcript = request.get("transcript_with_tool_calls")
                utterance = latest_user_utterance(
                    transcript if isinstance(transcript, list) else []
                )

                if interaction == "reminder_required" and not utterance:
                    ctrl = await get_controller(call_id)
                    if ctrl.sso_in_progress:
                        speech = SSO_BUSY
                    elif (
                        ctrl.phase == "eligibility"
                        and ctrl.eligibility_screen_verified
                        and not ctrl.eligibility_question_sent
                    ):
                        await deliver_first_eligibility_question(call_id)
                        speech = ""
                    else:
                        promoted = await sync_controller_to_eligibility(call_id)
                        if (
                            promoted
                            and not ctrl.eligibility_question_sent
                            and ctrl.eligibility_screen_verified
                        ):
                            await deliver_first_eligibility_question(
                                call_id, speech=promoted
                            )
                            speech = ""
                        else:
                            speech = promoted or ctrl.last_speech or "Are you still there?"
                    await _send(
                        websocket,
                        {
                            "response_type": "response",
                            "response_id": response_id,
                            "content": speech,
                            "content_complete": True,
                        },
                    )
                    continue

                # Track newest response_id; discard work if superseded mid-flight.
                my_id = response_id
                result = await process_user_turn(
                    call_id,
                    utterance,
                    response_id=response_id,
                )
                if my_id != latest_response_id:
                    logger.info(
                        "Discarding stale response call_id=%s response_id=%s latest=%s",
                        call_id,
                        my_id,
                        latest_response_id,
                    )
                    continue

                payload: dict[str, Any] = {
                    "response_type": "response",
                    "response_id": response_id,
                    "content": result.speech,
                    "content_complete": True,
                }
                if result.end_call:
                    payload["end_call"] = True
                await _send(websocket, payload)
                if result.detail == "sso_ack":
                    logger.info(
                        "acknowledgment_sent call_id=%s response_id=%s via=response",
                        call_id,
                        response_id,
                    )
                logger.info(
                    "Custom LLM WS replied call_id=%s response_id=%s phase=%s "
                    "detail=%s end=%s speech=%r",
                    call_id,
                    response_id,
                    result.phase,
                    result.detail,
                    result.end_call,
                    (result.speech or "")[:100],
                )
                continue

            logger.debug(
                "Custom LLM WS ignored interaction=%s call_id=%s",
                interaction,
                call_id,
            )

    except WebSocketDisconnect:
        logger.info("Custom LLM WS disconnected call_id=%s", call_id)
    except Exception:
        logger.exception("Custom LLM WS error call_id=%s", call_id)
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if _active_ws.get(call_id) is websocket:
            _active_ws.pop(call_id, None)
        try:
            ctrl_f = await get_controller(call_id)
            ctrl_f.active_websocket_status = "disconnected"
        except Exception:
            pass
        logger.info("socket_disconnected call_id=%s", call_id)
