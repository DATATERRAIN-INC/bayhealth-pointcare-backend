"""Patient callbacks and system reminder scheduling."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Optional, Tuple

from django.db.models import Q
from django.utils import timezone

from apps.ai_caller.models import Call, CallerSettings, ScheduledOutreach

dialer_logger = logging.getLogger("ai_caller.dialer")


def _resolve_timezone(tz_name: str):
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # Python < 3.9
        from backports.zoneinfo import ZoneInfo

    return ZoneInfo((tz_name or "America/New_York").strip())


def cancel_scheduled_for_patient(
    patient,
    *,
    kinds=None,
    reason: str = "",
    exclude_source_call=None,
) -> int:
    """Cancel open scheduled rows for a patient. Returns count cancelled."""
    qs = ScheduledOutreach.objects.filter(
        patient=patient,
        status=ScheduledOutreach.Status.SCHEDULED,
    )
    if kinds is not None:
        qs = qs.filter(kind__in=list(kinds))
    if exclude_source_call is not None:
        qs = qs.exclude(source_call=exclude_source_call)

    queued_call_ids = list(
        qs.exclude(queued_call_id__isnull=True).values_list("queued_call_id", flat=True)
    )
    count = qs.count()
    if count:
        qs.update(
            status=ScheduledOutreach.Status.CANCELLED,
            error_message=(reason or "")[:512],
            updated_at=timezone.now(),
        )
    # Drop unused queued dials that belonged to cancelled outreaches.
    if queued_call_ids:
        Call.objects.filter(
            id__in=queued_call_ids,
            status__in=[
                Call.Status.QUEUED,
                Call.Status.SCHEDULED,
                Call.Status.PAUSED,
            ],
        ).delete()
    return count


def create_queued_call_for_patient(
    patient, *, user=None, actor=None, status: str = ""
) -> Call:
    """
    Create a dial-queue Call.
    Use status=scheduled for callback/reminder outreaches; queued for ready-now.
    """
    actor = actor or user or getattr(patient, "user", None)
    to_number = f"{patient.country_code or ''}{patient.phone_number or ''}".strip()
    call_status = (status or "").strip() or Call.Status.QUEUED
    if call_status not in {
        Call.Status.QUEUED,
        Call.Status.SCHEDULED,
    }:
        call_status = Call.Status.QUEUED
    return Call.objects.create(
        user=user or getattr(patient, "user", None),
        created_by=actor,
        updated_by=actor,
        patient=patient,
        retell_call_id=None,
        flow=Call.Flow.OUTBOUND,
        status=call_status,
        is_paused=False,
        to_number=to_number,
        started_at=None,
    )


_WORD_HOURS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _normalize_callback_time_text(raw: str) -> str:
    """Collapse spoken filler so clock patterns can match."""
    text = re.sub(r"\s+", " ", (raw or "").strip().lower())
    # Prefer the day+time clause if the tool passed a full sentence.
    m = re.search(
        r"\b((?:today|tomorrow)\b.*|\bin\s+\d+\s*(?:hours?|hrs?|minutes?|mins?)\b.*|"
        r"\d{1,2}(?::\d{2})?\s*[ap]\.?\s*m\.?\b.*|"
        r"\d{4}-\d{2}-\d{2}[ t]\d{1,2}:\d{2}(?::\d{2})?)",
        text,
    )
    if m:
        text = m.group(1)
    # Word hours first so "one pm" can collapse like "1 pm"
    for word, num in _WORD_HOURS.items():
        text = re.sub(rf"\b{word}\b", str(num), text)
    # "1 p.m." / "1 PM" → "1pm"
    text = re.sub(r"\b([0-9]{1,2})\s*([ap])\.?\s*m\.?\b", r"\1\2m", text)
    text = re.sub(r"\b([ap])\.?\s*m\.?\b", r"\1m", text)
    # Drop filler around the clock: after/around/about/before/by/like/uh
    text = re.sub(
        r"\b(after|around|about|before|by|like|uh|um|approx(?:imately)?)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip(" ,.?!")
    return text
def parse_callback_datetime(
    raw_text: str,
    *,
    timezone_name: str = "America/New_York",
    now: Optional[datetime] = None,
) -> Tuple[Optional[datetime], str]:
    """
    Parse a patient-spoken callback time into an aware UTC datetime.
    Returns (datetime_or_None, error_message).
    """
    text = re.sub(r"\s+", " ", (raw_text or "").strip())
    if not text:
        return None, "Callback time is required."

    try:
        tz = _resolve_timezone(timezone_name)
    except Exception:
        tz = _resolve_timezone("America/New_York")

    local_now = (now or timezone.now()).astimezone(tz)
    lowered = _normalize_callback_time_text(text)

    # Relative: in N hours / minutes
    rel = re.match(
        r"^(?:in\s+)?(\d+)\s*(hours?|hrs?|minutes?|mins?)$",
        lowered,
    )
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2)
        delta = (
            timedelta(hours=amount)
            if unit.startswith("h")
            else timedelta(minutes=amount)
        )
        when = local_now + delta
        return when.astimezone(dt_timezone.utc), ""

    # tomorrow / today + optional clock
    day_offset = 0
    work = lowered
    if work.startswith("tomorrow"):
        day_offset = 1
        work = work[len("tomorrow") :].strip(" ,")
    elif work.startswith("today"):
        day_offset = 0
        work = work[len("today") :].strip(" ,")

    # Optional leading "at" left after day word
    work = re.sub(r"^at\s+", "", work).strip()

    # Bare "tomorrow" / "today" (no clock) → 10:00 local that day.
    if day_offset or work in {"", "morning", "afternoon", "evening"}:
        if not work or work in {"morning", "afternoon", "evening"}:
            hour = {"morning": 10, "afternoon": 14, "evening": 18}.get(work, 10)
            target_date = (local_now + timedelta(days=day_offset)).date()
            when = datetime(
                target_date.year,
                target_date.month,
                target_date.day,
                hour,
                0,
                tzinfo=tz,
            )
            if day_offset == 0 and when <= local_now:
                when = when + timedelta(days=1)
            return when.astimezone(dt_timezone.utc), ""

    clock = re.match(
        r"^(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?$",
        work,
        flags=re.I,
    )
    if not clock and not day_offset:
        # bare clock without today/tomorrow
        clock = re.match(
            r"^(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?$",
            lowered,
            flags=re.I,
        )

    if clock:
        hour = int(clock.group(1))
        minute = int(clock.group(2) or 0)
        meridiem = (clock.group(3) or "").replace(".", "").lower()
        if meridiem:
            if hour < 1 or hour > 12:
                return None, f"Could not understand callback time: {text}"
            hour = hour % 12
            if meridiem == "pm":
                hour += 12
        elif hour > 23 or minute > 59:
            return None, f"Could not understand callback time: {text}"

        target_date = (local_now + timedelta(days=day_offset)).date()
        when = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            hour,
            minute,
            tzinfo=tz,
        )
        # If "3pm" with no day and already past, push to tomorrow.
        if day_offset == 0 and when <= local_now:
            when = when + timedelta(days=1)
        return when.astimezone(dt_timezone.utc), ""

    # ISO-ish: 2026-10-06T15:00 or 2026-10-06 15:00
    iso = re.match(
        r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?$",
        text,
    )
    if iso:
        when = datetime(
            int(iso.group(1)),
            int(iso.group(2)),
            int(iso.group(3)),
            int(iso.group(4)),
            int(iso.group(5)),
            int(iso.group(6) or 0),
            tzinfo=tz,
        )
        return when.astimezone(dt_timezone.utc), ""

    return None, f"Could not understand callback time: {text}"


_CALLBACK_INTENT = re.compile(
    r"\b("
    r"call\s+(me\s+)?back"
    r"|call\s+me\b"
    r"|reach\s+(me\s+)?back"
    r"|callback"
    r"|call\s+later"
    r")\b",
    re.I,
)


def extract_callback_time_from_transcript(transcript) -> str:
    """
    Best-effort: find a patient turn that asks for a callback and return
    a time phrase (or the utterance) for parse_callback_datetime.
    """
    if not isinstance(transcript, list):
        return ""
    for entry in transcript:
        if not isinstance(entry, dict):
            continue
        speaker = str(entry.get("speaker") or entry.get("role") or "").strip().lower()
        if speaker not in {"patient", "user"}:
            continue
        text = str(entry.get("text") or "").strip()
        if not text or not _CALLBACK_INTENT.search(text):
            continue
        # Prefer an explicit day/time clause inside the utterance.
        lowered = text.lower()
        m = re.search(
            r"\b("
            r"tomorrow\b.*|today\b.*|"
            r"in\s+\d+\s*(?:hours?|hrs?|minutes?|mins?)\b.*"
            r")",
            lowered,
        )
        if m:
            return m.group(1).strip(" ,.?!'")
        return text
    return ""


def maybe_schedule_callback_from_transcript(call: Call) -> Optional[ScheduledOutreach]:
    """
    Fallback when the Retell agent confirms a callback verbally but never
    invokes log_callback_request.
    """
    if not call or not call.patient_id:
        return None
    if call.status == Call.Status.CALLBACK:
        return None
    if ScheduledOutreach.objects.filter(
        source_call=call,
        kind=ScheduledOutreach.Kind.CALLBACK_REQUESTED,
        status=ScheduledOutreach.Status.SCHEDULED,
    ).exists():
        return None

    raw = extract_callback_time_from_transcript(call.transcript or [])
    if not raw:
        raw = extract_callback_time_from_transcript(call.retell_transcript or [])
    if not raw:
        return None

    row, err = schedule_callback_request(call=call, raw_time_text=raw)
    if err:
        dialer_logger.warning(
            "CALLBACK_TRANSCRIPT_FALLBACK_FAILED call_id=%s raw=%s err=%s",
            call.id,
            raw[:120],
            err,
        )
        return None
    dialer_logger.info(
        "CALLBACK_TRANSCRIPT_FALLBACK call_id=%s outreach_id=%s raw=%s",
        call.id,
        getattr(row, "id", None),
        raw[:120],
    )
    return row


def schedule_callback_request(
    *,
    call: Call,
    raw_time_text: str,
) -> Tuple[Optional[ScheduledOutreach], Optional[str]]:
    """Create a patient-requested callback row from a Retell tool call."""
    patient = call.patient
    if not patient:
        return None, "Call has no patient."

    user = call.user or getattr(patient, "user", None)
    settings_obj = CallerSettings.load(user) if user else None
    tz_name = settings_obj.timezone if settings_obj else "America/New_York"

    scheduled_at, err = parse_callback_datetime(
        raw_time_text, timezone_name=tz_name
    )
    if err or not scheduled_at:
        return None, err or "Invalid callback time."

    # Patient callback wins over system reminders.
    cancel_scheduled_for_patient(
        patient,
        kinds=[
            ScheduledOutreach.Kind.CALLBACK_REQUESTED,
            ScheduledOutreach.Kind.REMINDER,
        ],
        reason="Replaced by new callback request",
    )

    queued_call = create_queued_call_for_patient(
        patient, user=user, actor=user, status=Call.Status.SCHEDULED
    )
    row = ScheduledOutreach.objects.create(
        user=user,
        patient=patient,
        source_call=call,
        queued_call=queued_call,
        kind=ScheduledOutreach.Kind.CALLBACK_REQUESTED,
        status=ScheduledOutreach.Status.SCHEDULED,
        scheduled_at=scheduled_at,
        raw_time_text=(raw_time_text or "")[:255],
    )
    call.status = Call.Status.CALLBACK
    call.save(update_fields=["status", "updated_at"])
    dialer_logger.info(
        "CALLBACK_SCHEDULED id=%s patient_id=%s queued_call_id=%s scheduled_at=%s raw=%s",
        row.id,
        patient.id,
        queued_call.id,
        scheduled_at.isoformat(),
        (raw_time_text or "")[:120],
    )
    return row, None


def schedule_reminder_for_missed_call(call: Call) -> Optional[ScheduledOutreach]:
    """
    After a not-attended call, queue a reminder using settings.reminder_timeframe_hours.
    Skips if patient already completed, is blocked, exhausted attempts, or has a
    pending patient callback.
    """
    if not call or call.status != Call.Status.NOT_ATTENDED:
        return None

    patient = call.patient
    if not patient or patient.is_blocked:
        return None

    if Call.objects.filter(
        patient=patient, status=Call.Status.COMPLETED
    ).exists():
        cancel_scheduled_for_patient(
            patient, reason="Patient already completed"
        )
        return None

    user = call.user or getattr(patient, "user", None)
    if not user:
        return None

    settings_obj = CallerSettings.load(user)
    trigger_cap = max(1, int(settings_obj.call_trigger_count or 1))
    not_attended_count = Call.objects.filter(
        patient=patient, status=Call.Status.NOT_ATTENDED
    ).count()
    if not_attended_count >= trigger_cap:
        cancel_scheduled_for_patient(
            patient,
            kinds=[ScheduledOutreach.Kind.REMINDER],
            reason="Max not-attended attempts reached",
        )
        return None

    # Do not override an explicit patient callback.
    if ScheduledOutreach.objects.filter(
        patient=patient,
        status=ScheduledOutreach.Status.SCHEDULED,
        kind=ScheduledOutreach.Kind.CALLBACK_REQUESTED,
    ).exists():
        return None

    hours = max(1, int(getattr(settings_obj, "reminder_timeframe_hours", 24) or 24))
    base = call.ended_at or call.started_at or timezone.now()
    scheduled_at = base + timedelta(hours=hours)

    cancel_scheduled_for_patient(
        patient,
        kinds=[ScheduledOutreach.Kind.REMINDER],
        reason="Replaced by newer reminder",
    )

    queued_call = create_queued_call_for_patient(
        patient, user=user, actor=user, status=Call.Status.SCHEDULED
    )
    row = ScheduledOutreach.objects.create(
        user=user,
        patient=patient,
        source_call=call,
        queued_call=queued_call,
        kind=ScheduledOutreach.Kind.REMINDER,
        status=ScheduledOutreach.Status.SCHEDULED,
        scheduled_at=scheduled_at,
        raw_time_text=f"Auto reminder after {hours}h",
    )
    dialer_logger.info(
        "REMINDER_SCHEDULED id=%s patient_id=%s queued_call_id=%s scheduled_at=%s hours=%s",
        row.id,
        patient.id,
        queued_call.id,
        scheduled_at.isoformat(),
        hours,
    )
    return row


def due_scheduled_outreaches(settings_obj, *, limit: int):
    """Open scheduled rows that are due now for this user (skip paused dials)."""
    now = timezone.now()
    completed_patient_ids = Call.objects.filter(
        user=settings_obj.user,
        status=Call.Status.COMPLETED,
    ).values_list("patient_id", flat=True)
    in_progress_patient_ids = Call.objects.filter(
        user=settings_obj.user,
        status=Call.Status.IN_PROGRESS,
    ).values_list("patient_id", flat=True)

    qs = (
        ScheduledOutreach.objects.filter(
            user=settings_obj.user,
            status=ScheduledOutreach.Status.SCHEDULED,
            scheduled_at__lte=now,
            patient__is_blocked=False,
        )
        .exclude(patient_id__in=in_progress_patient_ids)
        # Reminders skip patients who already completed; patient-requested
        # callbacks must still dial even if an earlier call was completed.
        .exclude(
            kind=ScheduledOutreach.Kind.REMINDER,
            patient_id__in=completed_patient_ids,
        )
        # Paused future dials stay scheduled but are not placed.
        .exclude(queued_call__status=Call.Status.PAUSED)
        .exclude(queued_call__is_paused=True)
        .filter(
            Q(queued_call__isnull=True)
            | Q(
                queued_call__status__in=[
                    Call.Status.SCHEDULED,
                    Call.Status.QUEUED,
                ]
            )
        )
        .select_related("patient", "queued_call")
        .order_by("scheduled_at", "id")
    )
    return list(qs[: max(1, int(limit))])


def mark_outreach_triggered(row: ScheduledOutreach, call: Optional[Call] = None):
    row.status = ScheduledOutreach.Status.TRIGGERED
    row.triggered_call = call
    row.error_message = ""
    row.save(
        update_fields=[
            "status",
            "triggered_call",
            "error_message",
            "updated_at",
        ]
    )


def mark_outreach_failed(row: ScheduledOutreach, error: str):
    row.status = ScheduledOutreach.Status.FAILED
    row.error_message = (error or "")[:512]
    row.save(update_fields=["status", "error_message", "updated_at"])


def get_scheduled_outreach_queryset(
    *,
    user=None,
    search: str = "",
    kind: str = "",
    status: str = "",
):
    qs = ScheduledOutreach.objects.select_related("patient", "source_call", "triggered_call")
    if user is not None:
        qs = qs.filter(user=user)

    search = (search or "").strip()
    if search:
        qs = qs.filter(
            Q(patient__first_name__icontains=search)
            | Q(patient__last_name__icontains=search)
            | Q(patient__phone_number__icontains=search)
            | Q(raw_time_text__icontains=search)
        )

    kind = (kind or "").strip().lower()
    if kind:
        qs = qs.filter(kind=kind)

    status = (status or "").strip().lower()
    if status:
        qs = qs.filter(status=status)

    return qs.order_by("scheduled_at", "id")
