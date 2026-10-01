"""Twilio TwiML webhooks for recorded two-human warm transfer."""

from __future__ import annotations

import logging

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from apps.ai_caller.twilio_bridge import (
    answer_twiml,
    connect_twiml,
    declined_twiml,
    dial_status_twiml,
    handle_conference_status,
    inbound_connect_twiml,
    is_affirmative,
    is_negative,
    load_pending_session,
    load_session,
    maybe_failover_on_agent_status,
    participant_join_twiml,
    persist_humans_transcript,
    save_session,
    speak_now_twiml,
    start_coordinator_leg,
    start_leg_recording,
)

logger = logging.getLogger(__name__)


def _xml(body: str) -> HttpResponse:
    return HttpResponse(body, content_type="text/xml")


def _session(request):
    session_id = (
        request.GET.get("session_id") or request.POST.get("session_id") or ""
    ).strip()
    session = load_session(session_id) if session_id else {}
    if not session:
        session = load_pending_session()
    call_sid = (request.POST.get("CallSid") or "").strip()
    if session and call_sid and not session.get("call_sid"):
        session["call_sid"] = call_sid
        save_session(session)
    return session


@csrf_exempt
@require_POST
def warm_transfer_answer(request):
    session = _session(request)
    if not session:
        return _xml(
            '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'
        )
    return _xml(answer_twiml(session))


@csrf_exempt
@require_POST
def warm_transfer_inbound(request):
    """Retell transferred person A onto our Twilio number. Dial and record B."""
    session = _session(request)
    if not session:
        logger.warning("Warm-transfer inbound with no session")
        return _xml(
            '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'
        )
    session["status"] = "inbound_from_retell"
    save_session(session)
    if session.get("call_sid"):
        inbound_speaker = str(session.get("inbound_speaker") or "patient")
        start_leg_recording(session, str(session.get("call_sid") or ""), inbound_speaker)
    start_coordinator_leg(session)
    return _xml(inbound_connect_twiml(session))


@csrf_exempt
@require_POST
def warm_transfer_gather(request):
    session = _session(request)
    if not session:
        return _xml(
            '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'
        )
    speech = (request.POST.get("SpeechResult") or "").strip()
    digits = (request.POST.get("Digits") or "").strip()
    if is_negative(speech, digits):
        session["status"] = "declined"
        save_session(session)
        return _xml(declined_twiml())
    if is_affirmative(speech, digits):
        session["status"] = "connecting"
        save_session(session)
        start_coordinator_leg(session)
        return _xml(connect_twiml(session))
    retries = int(session.get("gather_retries") or 0)
    if retries < 1:
        session["gather_retries"] = retries + 1
        save_session(session)
        return _xml(answer_twiml(session))
    session["status"] = "no_reply"
    save_session(session)
    return _xml(declined_twiml())


@csrf_exempt
@require_POST
def warm_transfer_join(request):
    session = _session(request)
    if not session:
        return _xml(
            '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'
        )
    if request.POST.get("CallSid"):
        session["dial_call_sid"] = request.POST.get("CallSid")
        save_session(session)
        start_leg_recording(
            session,
            str(session.get("dial_call_sid") or ""),
            str(session.get("dial_speaker") or "provider"),
        )
    return _xml(participant_join_twiml(session))


@csrf_exempt
@require_POST
def warm_transfer_speak_now(request):
    return _xml(speak_now_twiml())


@csrf_exempt
@require_POST
def warm_transfer_conference_status(request):
    session = _session(request)
    if session:
        payload = {k: (request.POST.get(k) or "") for k in request.POST.keys()}
        handle_conference_status(session, payload)
    return HttpResponse("ok")


@csrf_exempt
@require_POST
def warm_transfer_dial_status(request):
    status = (
        request.POST.get("DialCallStatus")
        or request.POST.get("DialBridged")
        or request.POST.get("CallStatus")
        or ""
    )
    session = _session(request)
    if session:
        session["dial_status"] = status
        dial_sid = (request.POST.get("DialCallSid") or "").strip()
        if dial_sid:
            session["dial_call_sid"] = dial_sid
        save_session(session)
        if maybe_failover_on_agent_status(
            session, call_status=status, call_sid=dial_sid
        ):
            # Keep patient on the line; next agent is being dialed.
            return _xml(
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<Response></Response>"
            )
    return _xml(dial_status_twiml(status))


@csrf_exempt
@require_POST
def warm_transfer_recording(request):
    session = _session(request)
    recording_url = (request.POST.get("RecordingUrl") or "").strip()
    recording_sid = (request.POST.get("RecordingSid") or "").strip()
    leg = (request.GET.get("leg") or request.POST.get("leg") or "").strip()
    if session and recording_url:
        persist_humans_transcript(
            session,
            recording_url=recording_url,
            recording_sid=recording_sid,
            leg=leg,
        )
    return HttpResponse("ok")


@csrf_exempt
@require_POST
def warm_transfer_status(request):
    session = _session(request)
    call_status = (request.POST.get("CallStatus") or "").strip()
    call_sid = (request.POST.get("CallSid") or "").strip()
    if session:
        session["call_status"] = call_status
        if call_sid and not session.get("call_sid"):
            session["call_sid"] = call_sid
        save_session(session)
        maybe_failover_on_agent_status(
            session, call_status=call_status, call_sid=call_sid
        )
    return HttpResponse("ok")
