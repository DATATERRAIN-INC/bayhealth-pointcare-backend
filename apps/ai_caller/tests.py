"""Call status + warm-transfer transcript regression tests. External APIs are mocked."""

import json
from datetime import time
from unittest import skipUnless
from unittest.mock import patch

from django.db import connection

from django.test import TestCase, override_settings

from apps.ai_caller import services, twilio_bridge
from apps.ai_caller.models import Call, Patient

S = Call.Status

def _clear_caches():
    from django.core.cache import caches

    for alias in ("default", "warm_transfer", "warm_transfer_file"):
        caches[alias].clear()


LOCAL_CACHES = {
    "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
    "warm_transfer": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "wt-tests",
    },
    "warm_transfer_file": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "wt-file-tests",
    },
}

SPOKEN = [
    {"role": "agent", "content": "Hi, this is Kyle from Bay Area Community Health."},
    {"role": "user", "content": "Hi, yes this is Ana."},
]
AGENT_ONLY = [{"role": "agent", "content": "Hi, this is Kyle from Bay Area Community Health."}]


def _retell(event, *, status="ended", reason="", transcript=None, analysis=None):
    call = {"call_id": "call_test", "call_status": status, "disconnection_reason": reason}
    if transcript is not None:
        call["transcript_object"] = transcript
    if analysis is not None:
        call["call_analysis"] = analysis
    return {"event": event, "call": call}


class MapRetellStatusTests(TestCase):
    def check(self, expected, status="ended", **kwargs):
        self.assertEqual(services._map_retell_status(status, **kwargs), expected, kwargs)

    def test_voicemail(self):
        self.check(S.VOICEMAIL_REACHED, disconnection_reason="voicemail_reached")
        self.check(S.VOICEMAIL_REACHED, disconnection_reason="agent_hangup", in_voicemail=True)

    def test_never_reached_a_person(self):
        for reason in (
            "dial_no_answer",
            "invalid_destination",
            "ivr_reached",
            "user_declined",
            "error_user_not_joined",
            "concurrency_limit_reached",
            "telephony_provider_unavailable",
        ):
            self.check(S.NOT_ATTENDED, disconnection_reason=reason)
        self.check(S.NOT_ATTENDED, status="not_connected")

    def test_answered_call_completes(self):
        self.check(S.COMPLETED, disconnection_reason="user_hangup", patient_spoke=True)
        self.check(
            S.COMPLETED,
            disconnection_reason="error_llm_websocket_lost_connection",
            patient_spoke=True,
        )
        # No transcript to judge from keeps the old behaviour.
        self.check(S.COMPLETED, disconnection_reason="user_hangup", patient_spoke=None)

    def test_pickup_with_no_patient_speech_is_not_attended(self):
        self.check(S.NOT_ATTENDED, disconnection_reason="user_hangup", patient_spoke=False)
        self.check(S.NOT_ATTENDED, disconnection_reason="inactivity", patient_spoke=False)

    def test_transfer_is_never_downgraded(self):
        self.check(
            S.COMPLETED,
            disconnection_reason="call_transfer",
            patient_spoke=False,
            in_voicemail=True,
        )

    def test_live_states(self):
        self.check(S.ONGOING, status="ongoing")
        self.check(S.IN_PROGRESS, status="registered")


@override_settings(CACHES=LOCAL_CACHES)
class UpdateCallFromRetellTests(TestCase):
    def setUp(self):
        _clear_caches()
        patient = Patient.objects.create(
            first_name="Ana", address="1 Main St", phone_number="+14155550123"
        )
        self.call = Call.objects.create(
            patient=patient,
            retell_call_id="call_test",
            flow=Call.Flow.OUTBOUND,
            status=S.ONGOING,
        )
        patches = {
            "callback": patch.object(services, "maybe_schedule_callback_from_transcript"),
            "cancel": patch.object(services, "cancel_scheduled_for_patient"),
            "reminder": patch.object(services, "schedule_reminder_for_missed_call"),
            "sms": patch.object(services, "maybe_trigger_sms_after_missed_call"),
            "stay": patch.object(
                twilio_bridge, "transfer_call_should_stay_open", return_value=False
            ),
            "mqtt": patch("apps.ai_caller.signals.publish_call_change"),
        }
        self.m = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def update(self, payload):
        call, error = services.update_call_from_retell_payload(payload)
        self.assertIsNone(error)
        self.call.refresh_from_db()
        return call

    def test_late_call_started_does_not_reopen_finished_call(self):
        self.update(_retell("call_ended", reason="user_hangup", transcript=SPOKEN))
        self.assertEqual(self.call.status, S.COMPLETED)
        self.update(_retell("call_started", status="ongoing"))
        self.assertEqual(self.call.status, S.COMPLETED)

    def test_post_call_analysis_marks_voicemail(self):
        self.update(_retell("call_ended", reason="agent_hangup", transcript=SPOKEN))
        self.assertEqual(self.call.status, S.COMPLETED)
        self.update(
            _retell(
                "call_analyzed",
                reason="agent_hangup",
                transcript=SPOKEN,
                analysis={"in_voicemail": True},
            )
        )
        self.assertEqual(self.call.status, S.VOICEMAIL_REACHED)
        self.m["reminder"].assert_called_once()
        self.m["sms"].assert_called_once()

    def test_voicemail_greeting_is_not_treated_as_callback_request(self):
        greeting = AGENT_ONLY + [
            {"role": "user", "content": "Leave a message and I'll call back."}
        ]
        self.update(_retell("call_ended", reason="voicemail_reached", transcript=greeting))
        self.assertEqual(self.call.status, S.VOICEMAIL_REACHED)
        self.m["callback"].assert_not_called()

    def test_silent_pickup_is_not_attended(self):
        self.update(_retell("call_ended", reason="user_hangup", transcript=AGENT_ONLY))
        self.assertEqual(self.call.status, S.NOT_ATTENDED)
        self.m["callback"].assert_not_called()
        self.m["reminder"].assert_called_once()

    def test_live_call_does_not_run_callback_inference(self):
        self.update(_retell("call_started", status="ongoing", transcript=SPOKEN))
        self.assertEqual(self.call.status, S.ONGOING)
        self.m["callback"].assert_not_called()

    def test_completed_call_runs_callback_inference(self):
        self.update(_retell("call_ended", reason="user_hangup", transcript=SPOKEN))
        self.m["callback"].assert_called_once()
        self.assertEqual(
            [i["speaker"] for i in self.call.retell_transcript], ["agent", "patient"]
        )

    def test_stores_retell_call_history_details(self):
        self.update(_retell("call_started", status="ongoing"))
        self.assertEqual(self.call.retell_call_status, "ongoing")
        self.assertEqual(self.call.disconnection_reason, "")

        self.update(
            _retell(
                "call_analyzed",
                reason="user_hangup",
                transcript=SPOKEN,
                analysis={"user_sentiment": "Neutral", "call_successful": False},
            )
        )
        self.assertEqual(self.call.retell_call_status, "ended")
        self.assertEqual(self.call.disconnection_reason, "user_hangup")
        self.assertEqual(self.call.user_sentiment, "Neutral")
        self.assertIs(self.call.call_successful, False)

        from apps.ai_caller.serializers import CallSerializer

        data = CallSerializer(self.call).data
        self.assertEqual(data["user_sentiment"], "Neutral")
        self.assertEqual(data["session_outcome"], "Unsuccessful")
        self.assertEqual(data["disconnection_reason"], "user_hangup")

    def test_voicemail_verdict_undoes_callback_read_from_greeting(self):
        from apps.ai_caller.models import ScheduledOutreach

        queued = Call.objects.create(
            patient=self.call.patient, flow=Call.Flow.OUTBOUND, status=S.SCHEDULED
        )
        outreach = ScheduledOutreach.objects.create(
            patient=self.call.patient,
            source_call=self.call,
            queued_call=queued,
            kind=ScheduledOutreach.Kind.CALLBACK_REQUESTED,
            status=ScheduledOutreach.Status.SCHEDULED,
            scheduled_at=self.call.created_at,
        )
        self.call.status = S.CALLBACK
        self.call.save()
        self.update(
            _retell(
                "call_analyzed",
                reason="agent_hangup",
                transcript=SPOKEN,
                analysis={"in_voicemail": True},
            )
        )
        self.assertEqual(self.call.status, S.VOICEMAIL_REACHED)
        outreach.refresh_from_db()
        self.assertEqual(outreach.status, ScheduledOutreach.Status.CANCELLED)
        self.assertFalse(Call.objects.filter(pk=queued.pk).exists())
        self.m["reminder"].assert_called_once()

    def test_stale_poll_does_not_move_retell_status_backwards(self):
        self.update(_retell("call_ended", reason="user_hangup", transcript=SPOKEN))
        self.update(_retell("call_started", status="ongoing"))
        self.assertEqual(self.call.retell_call_status, "ended")

    def test_sync_fetches_analysis_that_arrived_after_call_ended(self):
        # Like call #485: call_ended stored (no analysis yet), webhook never came.
        ended = _retell("call_ended", reason="agent_hangup", transcript=SPOKEN)
        self.update(ended)
        self.assertEqual((self.call.status, self.call.user_sentiment), (S.COMPLETED, ""))

        analyzed = _retell(
            "call_analyzed",
            reason="agent_hangup",
            transcript=SPOKEN,
            analysis={"user_sentiment": "Neutral", "call_successful": False},
        )["call"]
        with patch.object(
            services, "get_retell_call", return_value={"ok": True, "data": analyzed}
        ) as get_call:
            services.sync_in_progress_calls_from_retell()
            self.call.refresh_from_db()
            self.assertEqual(self.call.user_sentiment, "Neutral")
            self.assertIs(self.call.call_successful, False)
            # Analysis stored → no more polling for this call.
            get_call.reset_mock()
            services.sync_in_progress_calls_from_retell()
            get_call.assert_not_called()

    def test_retell_log_file_is_never_stored_as_recording(self):
        payload = _retell("call_ended", status="not_connected", reason="dial_no_answer", transcript=[])
        payload["call"]["public_log_url"] = "https://cdn.example/abc/public.log"
        self.update(payload)
        self.assertEqual(self.call.recording_url, "")

    def test_old_log_url_is_cleared_and_hidden_from_api(self):
        from apps.ai_caller.serializers import CallSerializer

        self.call.recording_url = "https://cdn.example/abc/public.log"
        self.call.save()
        self.assertIsNone(CallSerializer(self.call).data["recording_url"])
        self.update(_retell("call_ended", status="not_connected", reason="dial_no_answer", transcript=[]))
        self.assertEqual(self.call.recording_url, "")

    def test_retell_ai_recording_wins_recording_url(self):
        # A Twilio leg filled recording_url first (older transfer); Retell's AI audio replaces it.
        self.call.recording_url = "https://s3.example/warm-transfer/patient.wav"
        self.call.save()
        payload = _retell("call_ended", reason="call_transfer", transcript=SPOKEN)
        payload["call"]["recording_url"] = "https://cdn.example/abc/recording.wav"
        self.update(payload)
        self.assertEqual(self.call.recording_url, "https://cdn.example/abc/recording.wav")

    def test_late_mid_call_snapshot_does_not_shrink_final_transcript(self):
        final = SPOKEN + [{"role": "agent", "content": "Thanks, goodbye."}]
        self.update(_retell("call_ended", reason="user_hangup", transcript=final))
        self.assertEqual(len(self.call.retell_transcript), 3)
        self.update(_retell("", status="ongoing", transcript=SPOKEN[:1]))
        self.assertEqual(len(self.call.retell_transcript), 3)
        self.assertEqual(self.call.retell_transcript[-1]["text"], "Thanks, goodbye.")

    def test_transfer_keeps_waiting_status_while_patient_on_line(self):
        self.call.status = S.WAITING
        self.call.save()
        self.m["stay"].return_value = True
        self.update(_retell("call_ended", reason="call_transfer", transcript=SPOKEN))
        self.assertEqual(self.call.status, S.WAITING)
        self.assertIsNone(self.call.ended_at)


@override_settings(CACHES=LOCAL_CACHES)
class ConferenceEventTests(TestCase):
    def setUp(self):
        self.session = twilio_bridge.save_session(
            {
                "session_id": "s-join",
                "retell_call_id": "call_wt",
                "call_sid": "CApatient",
                "dial_call_sid": "CAagent",
                "queue_state": "active",
                "agent_claimed": True,
            }
        )

    def test_participant_join_starts_recording_and_announces(self):
        with patch.object(twilio_bridge.threading, "Thread") as thread, patch.object(
            twilio_bridge, "announce_you_can_speak_now"
        ) as speak:
            twilio_bridge.handle_conference_status(
                self.session,
                {
                    "StatusCallbackEvent": "participant-join",
                    "ConferenceSid": "CF1",
                    "CallSid": "CApatient",
                    "ParticipantCount": "2",
                },
            )
        self.assertEqual(thread.call_args.kwargs["args"], ("s-join", "CApatient", "patient"))
        speak.assert_called_once()

    def test_conference_end_releases(self):
        with patch.object(twilio_bridge, "extend_call_ended_at") as extend, patch.object(
            twilio_bridge, "release_agent_slot"
        ) as release:
            twilio_bridge.handle_conference_status(
                self.session, {"StatusCallbackEvent": "conference-end", "ConferenceSid": "CF1"}
            )
        extend.assert_called_once()
        release.assert_called_once()


@override_settings(CACHES=LOCAL_CACHES)
class TransferTranscriptPersistTests(TestCase):
    def _persist(self, session):
        with patch.object(twilio_bridge, "save_humans_transcript_for_call") as save:
            save.return_value = None
            twilio_bridge._persist_to_db(session)
        return save

    def test_dual_channel_provider_uses_patient_recording_clock(self):
        save = self._persist(
            {
                "retell_call_id": "call_wt",
                "patient_transcript": "hello",
                "provider_transcript": "hi there",
                "dual_channel_split": True,
                "patient_recording_started_at": 1000.0,
                "provider_recording_started_at": 1030.0,
            }
        )
        kwargs = save.call_args.kwargs
        self.assertEqual(kwargs["patient_offset"], kwargs["provider_offset"])

    def test_mono_legs_wait_for_mix_then_save_when_mix_fails(self):
        session = {
            "retell_call_id": "call_wt",
            "patient_transcript": "hello",
            "provider_transcript": "hi there",
        }
        self.assertIsNone(self._persist(session).call_args)
        save = self._persist({**session, "mix_failed": True})
        self.assertEqual(save.call_args.kwargs["provider_text"], "hi there")


@override_settings(CACHES=LOCAL_CACHES)
class TransferRecordingTests(TestCase):
    def setUp(self):
        patient = Patient.objects.create(first_name="Ana", address="x", phone_number="+14155550123")
        self.call = Call.objects.create(
            patient=patient,
            retell_call_id="call_rec",
            flow=Call.Flow.OUTBOUND,
            status=S.COMPLETED,
            recording_url="https://cdn.example/abc/recording.wav",
        )
        p = patch("apps.ai_caller.signals.publish_call_change")
        p.start()
        self.addCleanup(p.stop)

    def test_live_agent_player_gets_both_voices_and_ai_audio_is_kept(self):
        session = {
            "retell_call_id": "call_rec",
            "call_sid": "CApat",
            "patient_recording_dual": True,
            "patient_recording_s3_url": "https://s3.example/patient_dual.wav",
            "provider_recording_s3_url": "https://s3.example/provider_right.wav",
            "mix_recording_s3_url": "https://s3.example/mix.wav",
        }
        self.assertEqual(twilio_bridge._human_recording_url(session), "https://s3.example/patient_dual.wav")
        twilio_bridge._sync_call_twilio_ids(
            session,
            recording_url="https://s3.example/patient_dual.wav",
            live_agent_recording_url="https://s3.example/provider_right.wav",
        )
        self.call.refresh_from_db()
        self.assertEqual(self.call.recording_url, "https://cdn.example/abc/recording.wav")
        self.assertEqual(self.call.live_agent_recording_url, "https://s3.example/patient_dual.wav")

    def test_mix_then_agent_leg_fallback_order(self):
        self.assertEqual(
            twilio_bridge._human_recording_url(
                {"mix_recording_s3_url": "m.wav", "provider_recording_s3_url": "p.wav"}
            ),
            "m.wav",
        )
        self.assertEqual(
            twilio_bridge._human_recording_url({"provider_recording_s3_url": "p.wav"}), "p.wav"
        )

    def test_humans_transcript_save_keeps_ai_recording(self):
        from apps.ai_caller.transcript_merge import save_humans_transcript_for_call

        save_humans_transcript_for_call(
            retell_call_id="call_rec",
            patient_text="I need help booking.",
            provider_text="Sure, I can help.",
            recording_url="https://s3.example/patient.wav",
            live_agent_recording_url="https://s3.example/patient_dual.wav",
        )
        self.call.refresh_from_db()
        self.assertEqual(self.call.recording_url, "https://cdn.example/abc/recording.wav")
        self.assertEqual(self.call.live_agent_recording_url, "https://s3.example/patient_dual.wav")
        self.assertEqual(
            {i["speaker"] for i in self.call.live_agent_transcript}, {"patient", "live_agent"}
        )


@override_settings(CACHES=LOCAL_CACHES)
class ReleaseAgentSlotTests(TestCase):
    def test_callback_offer_still_sets_final_status(self):
        session = twilio_bridge.save_session(
            {"session_id": "s-cb", "retell_call_id": "call_cb", "callback_offer_started": True}
        )
        with patch.object(twilio_bridge, "set_transfer_call_status") as set_status, patch.object(
            twilio_bridge, "restore_inbound_voice_url"
        ):
            twilio_bridge.release_agent_slot(session)
        set_status.assert_called_once()
        self.assertEqual(set_status.call_args.args[1], "completed")


@override_settings(CACHES=LOCAL_CACHES)
class DemoReviewTransferTests(TestCase):
    """B3, B5: transfer session matching and the one-agent failure path."""

    def setUp(self):
        _clear_caches()

    def _session(self, sid, *, phone="+14155550123", age_seconds=0, **extra):
        import time as _time

        return twilio_bridge.save_session(
            {
                "session_id": sid,
                "retell_call_id": f"call_{sid}",
                "phone_number": phone,
                "created_at": _time.time() - age_seconds,
                **extra,
            }
        )

    def test_new_transfer_binds_to_newest_call_for_same_phone(self):
        self._session("old", age_seconds=20 * 60)  # rehearsal call 20 min ago
        self._session("new")
        chosen = twilio_bridge.resolve_inbound_session(call_sid="CA1", from_number="+14155550123")
        self.assertEqual(chosen["session_id"], "new")

    def test_sessions_older_than_an_hour_are_ignored(self):
        self._session("yesterday", age_seconds=24 * 3600)
        self.assertEqual(twilio_bridge.resolve_inbound_session(call_sid="CA2"), {})

    def test_single_agent_dial_failure_offers_callback(self):
        session = self._session("solo", transfer_numbers=["+14155550001"], transfer_number="+14155550001")
        with patch.object(twilio_bridge, "offer_agent_unreachable_callback", return_value=True) as offer:
            self.assertTrue(twilio_bridge.failover_to_next_live_agent(session, reason="dial_create_failed"))
        offer.assert_called_once()

    def test_unconfigured_dial_does_not_leave_patient_ringing(self):
        session = self._session("noclient", transfer_numbers=["+14155550001"], transfer_number="+14155550001")
        with patch.object(twilio_bridge, "_twilio_client", return_value=None), patch.object(
            twilio_bridge, "offer_agent_unreachable_callback", return_value=True
        ) as offer:
            twilio_bridge.start_coordinator_leg(session)
        offer.assert_called_once()


@override_settings(CACHES=LOCAL_CACHES)
class DemoReviewCallSyncTests(TestCase):
    """B4, B8: callback calls keep syncing; the callback check runs once."""

    def setUp(self):
        _clear_caches()
        patient = Patient.objects.create(first_name="Ana", address="x", phone_number="+14155550123")
        self.call = Call.objects.create(
            patient=patient, retell_call_id="call_cb", flow=Call.Flow.OUTBOUND, status=S.ONGOING
        )
        for p in (
            patch("apps.ai_caller.signals.publish_call_change"),
            patch.object(services, "cancel_scheduled_for_patient"),
            patch.object(twilio_bridge, "transfer_call_should_stay_open", return_value=False),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_callback_requested_mid_call_keeps_syncing_until_ended(self):
        from django.utils import timezone

        Call.objects.filter(pk=self.call.pk).update(
            status=S.CALLBACK, retell_call_status="ongoing", transcript=[{"text": "hi"}],
            started_at=timezone.now(),
        )
        ended = _retell("call_ended", reason="user_hangup", transcript=SPOKEN)["call"]
        ended["call_id"] = "call_cb"
        ended["recording_url"] = "https://cdn.example/cb/recording.wav"
        with patch.object(services, "get_retell_call", return_value={"ok": True, "data": ended}) as get_call:
            services.sync_in_progress_calls_from_retell()
        get_call.assert_called_once()
        self.call.refresh_from_db()
        self.assertEqual(self.call.status, S.CALLBACK)
        self.assertEqual(self.call.retell_call_status, "ended")
        self.assertEqual(self.call.recording_url, "https://cdn.example/cb/recording.wav")
        self.assertEqual(len(self.call.retell_transcript), 2)

    def test_callback_check_runs_once_per_completed_call(self):
        payload = _retell("call_ended", reason="user_hangup", transcript=SPOKEN)
        payload["call"]["call_id"] = "call_cb"
        with patch.object(services, "maybe_schedule_callback_from_transcript") as check:
            services.update_call_from_retell_payload(payload)
            services.update_call_from_retell_payload({**payload, "event": "call_analyzed"})
            services.update_call_from_retell_payload(payload)
        check.assert_called_once()


@override_settings(CACHES=LOCAL_CACHES)
class DemoReviewPatientApiTests(TestCase):
    """B6, B7: patients list shows the real latest call; transcript returns recordings."""

    def setUp(self):
        from django.utils import timezone
        from rest_framework.test import APIClient

        from apps.ai_caller.models import CallerSettings
        from apps.users.models import User

        self.user = User.objects.create(cognito_id="sub-api", email="api@example.com")
        CallerSettings.load(self.user)
        self.patient = Patient.objects.create(
            user=self.user, first_name="Ana", address="x", phone_number="4155550123", country_code="+1"
        )
        p = patch("apps.ai_caller.signals.publish_call_change")
        p.start()
        self.addCleanup(p.stop)
        now = timezone.now()
        self.done = Call.objects.create(
            user=self.user, patient=self.patient, flow="outbound", status=S.NOT_ATTENDED,
            retell_call_id="call_real", started_at=now, ended_at=now,
            recording_url="https://cdn.example/real/recording.wav",
        )
        # Pending reminder row: no started_at (created after the real call).
        Call.objects.create(
            user=self.user, patient=self.patient, flow="outbound", status=S.SCHEDULED, started_at=None
        )
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def _results(self, url):
        body = self.client.get(url).json()
        return body.get("results") or body.get("data", {}).get("results") or []

    def test_patients_list_shows_real_latest_call_not_queue_row(self):
        rows = self._results("/api/ai-call/patients/")
        self.assertEqual(rows[0]["retell_call_id"], "call_real")
        self.assertEqual(rows[0]["call_status"], S.NOT_ATTENDED)

    def test_outcome_filter_includes_patient_with_pending_reminder(self):
        rows = self._results("/api/ai-call/patients/?call_status=not_attended")
        self.assertEqual([r["id"] for r in rows], [self.patient.id])

    def test_transcript_response_includes_recordings(self):
        with patch("apps.ai_caller.views.CallViewSet._enqueue_retell_sync"), patch(
            "apps.ai_caller.serializers._recording_enabled_for", return_value=True
        ):
            body = self.client.get("/api/ai-call/calls/?retell_call_id=call_real").json()
        self.assertEqual(body["recording_url"], "https://cdn.example/real/recording.wav")
        self.assertIn("transcript", body)


@skipUnless(connection.vendor == "postgresql", "calls list uses Postgres jsonb_array_length")
class CallSentimentFilterTests(TestCase):
    def setUp(self):
        from rest_framework.test import APIClient

        from apps.ai_caller.models import CallerSettings
        from apps.users.models import User

        self.user = User.objects.create(cognito_id="sub-sent", email="sent@example.com")
        CallerSettings.load(self.user)
        patient = Patient.objects.create(user=self.user, first_name="Ana", address="x", phone_number="4155550123")
        p = patch("apps.ai_caller.signals.publish_call_change")
        p.start()
        self.addCleanup(p.stop)
        for i, sentiment in enumerate(["Positive", "Negative", "Negative", "Neutral", ""]):
            Call.objects.create(
                user=self.user, patient=patient, flow="outbound", status=S.COMPLETED,
                retell_call_id=f"call_s{i}", user_sentiment=sentiment,
            )
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def _sentiments(self, query):
        with patch("apps.ai_caller.views.CallViewSet._enqueue_retell_sync"):
            body = self.client.get(f"/api/ai-call/calls/?{query}").json()
        return sorted(row["user_sentiment"] for row in body["results"])

    def test_filter_by_one_sentiment(self):
        self.assertEqual(self._sentiments("user_sentiment=Negative"), ["Negative", "Negative"])
        self.assertEqual(self._sentiments("user_sentiment=positive"), ["Positive"])

    def test_filter_by_several_and_pending(self):
        self.assertEqual(self._sentiments("user_sentiment=Neutral,pending"), ["", "Neutral"])

    def test_no_filter_returns_all_and_unknown_values_are_ignored(self):
        self.assertEqual(len(self._sentiments("")), 5)
        self.assertEqual(len(self._sentiments("user_sentiment=happy")), 5)


class LiveAgentTranscriptCleanupTests(TestCase):
    """System announcements and Whisper silence inventions never reach the transcript."""

    def test_announcements_and_silence_words_are_removed(self):
        from apps.ai_caller.transcript_merge import clean_human_items

        items = [
            {"speaker": "patient", "text": "You", "segment": "human"},
            {"speaker": "live_agent", "text": "You can speak now.", "segment": "human"},
            {"speaker": "patient", "text": "Hello, yes,", "segment": "human"},
            {
                "speaker": "live_agent",
                "text": "Hi, this is a Bay Area Community Health care coordinator. I am connecting "
                "Sneha about Dental Exam. I will introduce you both, then leave. Hi Sneha, how can I help?",
                "segment": "human",
            },
            {"speaker": "patient", "text": "Thank you.", "segment": "human"},
        ]
        self.assertEqual(
            [(i["speaker"], i["text"]) for i in clean_human_items(items)],
            [
                ("patient", "Hello, yes,"),
                ("live_agent", "Hi Sneha, how can I help?"),
                ("patient", "Thank you."),
            ],
        )

    def test_whisper_no_speech_segments_are_dropped(self):
        from apps.ai_caller.transcript_merge import _parse_verbose_segments

        segments = _parse_verbose_segments(
            {
                "segments": [
                    {"start": 0, "end": 2, "text": " You", "no_speech_prob": 0.9, "avg_logprob": -1.4},
                    {"start": 2, "end": 4, "text": " Hello, yes.", "no_speech_prob": 0.02, "avg_logprob": -0.2},
                ]
            }
        )
        self.assertEqual([s["text"] for s in segments], ["Hello, yes."])

    def test_saved_live_agent_transcript_is_clean(self):
        from apps.ai_caller.transcript_merge import save_humans_transcript_for_call

        patient = Patient.objects.create(first_name="Sneha", address="x", phone_number="+14155550123")
        with patch("apps.ai_caller.signals.publish_call_change"):
            call = Call.objects.create(patient=patient, retell_call_id="call_clean", flow="outbound", status=S.COMPLETED)
            save_humans_transcript_for_call(
                retell_call_id="call_clean",
                mix_items=[
                    {"speaker": "live_agent", "text": "You can speak now.", "segment": "human", "at": 1},
                    {"speaker": "patient", "text": "Hello, yes,", "segment": "human", "at": 2},
                    {"speaker": "live_agent", "text": "Hi, I can book that for you.", "segment": "human", "at": 3},
                    {"speaker": "patient", "text": "You", "segment": "human", "at": 4},
                ],
            )
        call.refresh_from_db()
        self.assertEqual(
            [i["text"] for i in call.live_agent_transcript],
            ["Hello, yes,", "Hi, I can book that for you."],
        )

    def test_doctor_name_is_not_doubled(self):
        from apps.ai_caller.retell import doctor_display_name

        self.assertEqual(doctor_display_name("Dr John"), "John")
        self.assertEqual(doctor_display_name("Dr. Daniel Adams"), "Daniel Adams")
        self.assertEqual(doctor_display_name("Drake Smith"), "Drake Smith")


class TransferTranscriptAccuracyTests(TestCase):
    """Echo between the two speaker channels, prompt echoes, and STT retries."""

    def test_echo_of_agent_on_patient_channel_is_removed(self):
        from apps.ai_caller.transcript_merge import interleave_speaker_segments

        agent = [
            {"start": 6.5, "end": 11.0, "text": "Hello, this is Maria. I can help you book your dental exam today.", "confidence": -0.2},
            {"start": 17.0, "end": 19.5, "text": "Sure, does Tuesday at 3 p.m. work for you?", "confidence": -0.2},
            {"start": 27.0, "end": 31.0, "text": "Great, you are all set for Tuesday at 3 p.m. Have a good day.", "confidence": -0.2},
        ]
        patient = [
            # Speakerphone echo of the agent, quieter → lower confidence.
            {"start": 6.7, "end": 11.1, "text": "Hello, this is Maria. I can help you book your dental exam today.", "confidence": -0.9},
            {"start": 12.5, "end": 15.0, "text": "Hi Maria, yes, I need an appointment.", "confidence": -0.2},
            {"start": 21.5, "end": 25.0, "text": "Tuesday at 3 p.m. works for me.", "confidence": -0.2},
            # Echo tail glued to the patient's own words in one segment.
            {"start": 30.5, "end": 34.0, "text": "Have a good day. Thank you, bye.", "confidence": -0.5},
        ]
        items = interleave_speaker_segments(patient_segments=patient, provider_segments=agent)
        self.assertEqual(
            [(i["speaker"], i["text"]) for i in items],
            [
                ("live_agent", "Hello, this is Maria. I can help you book your dental exam today."),
                ("patient", "Hi Maria, yes, I need an appointment."),
                ("live_agent", "Sure, does Tuesday at 3 p.m. work for you?"),
                ("patient", "Tuesday at 3 p.m. works for me."),
                ("live_agent", "Great, you are all set for Tuesday at 3 p.m. Have a good day."),
                ("patient", "Thank you, bye."),
            ],
        )

    def test_reply_after_the_agent_finishes_is_kept(self):
        from apps.ai_caller.transcript_merge import interleave_speaker_segments

        agent = [{"start": 0.0, "end": 3.0, "text": "Great, you are all set. Have a good day."}]
        patient = [{"start": 4.5, "end": 6.0, "text": "You too, have a good day."}]
        items = interleave_speaker_segments(patient_segments=patient, provider_segments=agent)
        self.assertEqual(items[-1], {**items[-1], "speaker": "patient", "text": "You too, have a good day."})

    def test_prompt_words_heard_in_silence_are_dropped(self):
        from apps.ai_caller.transcript_merge import _drop_prompt_echo_segments

        prompt = "A phone call between Sneha, a patient, and a Bay Area Community Health care coordinator."
        kept = _drop_prompt_echo_segments(
            [{"text": "Have a good day."}, {"text": "Bay Area Community Health"}], prompt
        )
        self.assertEqual([s["text"] for s in kept], ["Have a good day."])

    def test_speech_to_text_retries_temporary_failures(self):
        from apps.ai_caller import transcript_merge as tm

        busy = type("R", (), {"status_code": 429, "text": ""})()
        ok = type("R", (), {"status_code": 200, "text": '{"text": "Hello", "segments": [{"start": 0, "end": 1, "text": "Hello"}]}'})()
        with override_settings(OPENAI_API_KEY="k"), patch.object(tm.time, "sleep"), patch.object(
            tm.requests, "post", side_effect=[tm.requests.ConnectionError("down"), busy, ok]
        ) as post:
            result = tm.transcribe_recording_detailed(b"audio", "patient.wav")
        self.assertEqual(result["text"], "Hello")
        self.assertEqual(post.call_count, 3)


class AgentBriefingTests(TestCase):
    def test_agent_hears_who_is_waiting_before_joining(self):
        twiml = twilio_bridge.participant_join_twiml(
            {"session_id": "s1", "name": "Sneha Manoharan", "service_name": "Dental Exam"}
        )
        say = twiml.index("Bay Area Community Health transfer. Connecting you with Sneha Manoharan about Dental Exam.")
        self.assertLess(say, twiml.index("<Conference"))
        self.assertIn('endConferenceOnExit="true"', twiml)  # agent leaving ends the call

    def test_briefing_without_name_or_service(self):
        self.assertEqual(
            twilio_bridge.agent_briefing_text({"name": "there", "service_name": "care"}),
            "Bay Area Community Health transfer. Connecting you with a patient.",
        )

    def test_name_is_xml_safe(self):
        twiml = twilio_bridge.participant_join_twiml({"session_id": "s1", "name": "Ana <&> Lee"})
        self.assertIn("Ana &lt;&amp;&gt; Lee", twiml)

    def test_briefing_never_shows_as_agent_speech(self):
        from apps.ai_caller.transcript_merge import strip_system_speech

        self.assertEqual(
            strip_system_speech(
                "Bay Area Community Health transfer. Connecting you with Sneha Manoharan about Dental Exam. "
                "Hi Sneha, this is Maria."
            ),
            "Hi Sneha, this is Maria.",
        )


AGENT_1, AGENT_2, AGENT_3 = "+14155550001", "+14155550002", "+14155550003"


@override_settings(CACHES=LOCAL_CACHES)
class MultiAgentAssignmentTests(TestCase):
    """Each free live agent takes their own patient; extra patients wait on hold."""

    def setUp(self):
        _clear_caches()
        self.mocks = {}
        for name in (
            "start_coordinator_leg",
            "announce_connecting",
            "announce_next_agent",
            "announce_hold",
            "notify_agent_waiting_tone",
            "set_transfer_call_status",
            "_hangup_call",
            "restore_inbound_voice_url",
        ):
            p = patch.object(twilio_bridge, name)
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)

    def _patient(self, sid, agents=(AGENT_1, AGENT_2, AGENT_3)):
        return twilio_bridge.save_session(
            {
                "session_id": sid,
                "user_id": 7,
                "call_sid": f"CA{sid}",
                "retell_call_id": f"call_{sid}",
                "transfer_numbers": list(agents),
                "transfer_number": agents[0],
            }
        )

    def _agent_of(self, sid):
        return twilio_bridge.load_session(sid).get("assigned_agent")

    def test_each_patient_gets_their_own_free_agent_and_d_waits(self):
        for sid in "ABCD":
            self._patient(sid)
        results = [twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session(sid)) for sid in "ABCD"]
        self.assertEqual(results, ["active", "active", "active", "waiting"])
        self.assertEqual([self._agent_of(s) for s in "ABC"], [AGENT_1, AGENT_2, AGENT_3])
        self.assertEqual(twilio_bridge.load_session("D")["queue_state"], "waiting")

    def test_second_patient_goes_straight_to_agent_2(self):
        self._patient("A")
        self._patient("B")
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("A"))
        self.assertEqual(twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("B")), "active")
        self.assertEqual(self._agent_of("B"), AGENT_2)

    def test_busy_agents_hear_a_beep_for_every_new_waiter(self):
        for sid in "ABCDE":
            self._patient(sid)
        for sid in "ABCDE":
            twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session(sid))
        beeped = [c.args[0]["session_id"] for c in self.mocks["notify_agent_waiting_tone"].call_args_list]
        # D and E each start waiting → the three busy agents (with A, B, C) are beeped each time.
        self.assertEqual(sorted(beeped), sorted(["A", "B", "C", "A", "B", "C"]))

    def test_waiting_patient_gets_the_agent_who_frees_up(self):
        for sid in "ABCD":
            self._patient(sid)
        for sid in "ABCD":
            twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session(sid))
        twilio_bridge.release_agent_slot(twilio_bridge.load_session("B"))  # agent 2 finishes
        dialed = self.mocks["start_coordinator_leg"].call_args.args[0]
        self.assertEqual((dialed["session_id"], dialed["assigned_agent"]), ("D", AGENT_2))
        self.assertEqual(self._agent_of("A"), AGENT_1)  # agent 1 still with A

    def test_freed_agent_is_reused_not_agent_1_again(self):
        # Reported issue: when agent 1 finishes, the waiting patient must not
        # be sent back to agent 1 while agent 2 is free. With per-agent
        # assignment, B never waits in the first place.
        self._patient("A", agents=(AGENT_1, AGENT_2))
        self._patient("B", agents=(AGENT_1, AGENT_2))
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("A"))
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("B"))
        self.assertEqual((self._agent_of("A"), self._agent_of("B")), (AGENT_1, AGENT_2))
        twilio_bridge.release_agent_slot(twilio_bridge.load_session("A"))
        self.mocks["start_coordinator_leg"].assert_not_called()  # nobody waiting
        self._patient("C", agents=(AGENT_1, AGENT_2))
        self.assertEqual(twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("C")), "active")
        self.assertEqual(self._agent_of("C"), AGENT_1)  # agent 1 is free again

    def test_no_answer_moves_to_next_free_agent(self):
        self._patient("A")
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("A"))
        twilio_bridge.failover_to_next_live_agent(twilio_bridge.load_session("A"), reason="no-answer")
        self.assertEqual(self._agent_of("A"), AGENT_2)
        self._patient("B")
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("B"))
        self.assertEqual(self._agent_of("B"), AGENT_1)  # agent 1 was released

    def test_no_answer_with_other_agents_busy_waits_at_front_of_line(self):
        for sid in "AB":
            self._patient(sid, agents=(AGENT_1, AGENT_2))
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("A"))  # agent 1
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("B"))  # agent 2
        self._patient("C", agents=(AGENT_1, AGENT_2))
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("C"))  # waits
        twilio_bridge.failover_to_next_live_agent(twilio_bridge.load_session("B"), reason="no-answer")
        # B waits (agent 1 busy) at the front; agent 2 is offered to C instead of sitting idle.
        self.assertEqual(twilio_bridge.load_session("B")["queue_state"], "waiting")
        self.mocks["announce_hold"].assert_called_once()
        self.assertEqual(self._agent_of("C"), AGENT_2)
        twilio_bridge.release_agent_slot(twilio_bridge.load_session("A"))  # agent 1 frees
        dialed = self.mocks["start_coordinator_leg"].call_args.args[0]
        self.assertEqual((dialed["session_id"], dialed["assigned_agent"]), ("B", AGENT_1))

    def test_waiting_patient_is_not_rerung_on_agent_who_missed_them(self):
        for sid in "AB":
            self._patient(sid, agents=(AGENT_1, AGENT_2))
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("A"))  # agent 1
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("B"))  # agent 2
        twilio_bridge.failover_to_next_live_agent(twilio_bridge.load_session("B"), reason="no-answer")
        # Agent 2 is free but missed B → B keeps waiting for agent 1.
        self.assertEqual(twilio_bridge.load_session("B")["queue_state"], "waiting")
        self.assertFalse(self._agent_of("B"))

    def test_every_agent_tried_offers_callback(self):
        self._patient("A", agents=(AGENT_1, AGENT_2))
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("A"))
        with patch.object(twilio_bridge, "offer_agent_unreachable_callback", return_value=True) as offer:
            twilio_bridge.failover_to_next_live_agent(twilio_bridge.load_session("A"), reason="no-answer")
            twilio_bridge.failover_to_next_live_agent(twilio_bridge.load_session("A"), reason="no-answer")
        offer.assert_called_once()

    def test_leftover_busy_marker_clears_itself(self):
        self._patient("old")
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("old"))
        old = twilio_bridge.load_session("old")
        old["conference_ended"] = True  # call ended without a clean release
        twilio_bridge.save_session(old)
        self._patient("new")
        twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session("new"))
        self.assertEqual(self._agent_of("new"), AGENT_1)

    def test_two_patients_at_the_same_instant_never_share_an_agent(self):
        import threading

        for sid in ("P1", "P2"):
            self._patient(sid, agents=(AGENT_1,))
        results = {}

        def admit(sid):
            results[sid] = twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session(sid))

        threads = [threading.Thread(target=admit, args=(sid,)) for sid in ("P1", "P2")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(results.values()), ["active", "waiting"])


REDIS_TEST_CACHES = {
    **LOCAL_CACHES,
    "warm_transfer": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/15",
        "KEY_PREFIX": "wt-test",
    },
}


def _redis_available():
    try:
        import redis

        return redis.Redis.from_url("redis://127.0.0.1:6379/15").ping()
    except Exception:
        return False


@skipUnless(_redis_available(), "local Redis not running")
@override_settings(CACHES=REDIS_TEST_CACHES)
class MultiAgentAssignmentRedisTests(MultiAgentAssignmentTests):
    """Same scenarios against real Redis (production's atomic SET NX)."""


@override_settings(CACHES=LOCAL_CACHES, BACKEND_URL="https://example.test")
class LongWaitCallbackTests(TestCase):
    """All agents busy for 3 minutes → offer a callback at the Settings time."""

    def setUp(self):
        from apps.ai_caller.models import CallerSettings
        from apps.users.models import User

        _clear_caches()
        self.user = User.objects.create(cognito_id="sub-wait", email="wait@example.com")
        cs = CallerSettings.load(self.user)
        cs.timezone, cs.reminder_timeframe_hours = "America/New_York", 1
        cs.start_time, cs.end_time = time(9, 0), time(17, 0)
        cs.save()
        self.settings_obj = cs
        patient = Patient.objects.create(user=self.user, first_name="Ana", address="x", phone_number="4155550123")
        p = patch("apps.ai_caller.signals.publish_call_change")
        p.start()
        self.addCleanup(p.stop)
        self.call = Call.objects.create(
            user=self.user, patient=patient, retell_call_id="call_W", flow="outbound", status=S.WAITING
        )
        self.mocks = {}
        for name in ("start_coordinator_leg", "announce_connecting", "announce_hold",
                     "notify_agent_waiting_tone", "restore_inbound_voice_url", "_schedule_long_wait_offer"):
            p = patch.object(twilio_bridge, name)
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)
        # Agent 1 busy with A; W waits.
        for sid in ("A", "W"):
            twilio_bridge.save_session({
                "session_id": sid, "user_id": self.user.pk, "call_sid": f"CA{sid}",
                "retell_call_id": f"call_{sid}", "transfer_numbers": [AGENT_1], "transfer_number": AGENT_1,
            })
            twilio_bridge.admit_patient_to_agent(twilio_bridge.load_session(sid))

    def _w(self):
        return twilio_bridge.load_session("W")

    def test_waiting_schedules_the_three_minute_check(self):
        self.mocks["_schedule_long_wait_offer"].assert_called_once()
        self.assertEqual(self._w()["queue_state"], "waiting")

    def test_offer_steps_patient_out_without_losing_their_place(self):
        client = patch.object(twilio_bridge, "_twilio_client").start()
        self.addCleanup(patch.stopall)
        self.assertTrue(twilio_bridge.offer_callback_to_long_waiter("W", self._w()["wait_token"]))
        url = client.return_value.calls.return_value.update.call_args.kwargs["url"]
        self.assertIn("/warm-transfer/long-wait-offer/", url)
        # Conference ends as they step out: not a hang-up, and no agent is given to them meanwhile.
        twilio_bridge.handle_conference_status(self._w(), {"StatusCallbackEvent": "conference-end"})
        twilio_bridge.release_agent_slot(twilio_bridge.load_session("A"))  # agent 1 frees
        self.mocks["start_coordinator_leg"].assert_not_called()
        self.assertEqual(self._w()["queue_state"], "waiting")

    def test_stale_check_does_nothing(self):
        with patch.object(twilio_bridge, "_twilio_client") as client:
            self.assertFalse(twilio_bridge.offer_callback_to_long_waiter("W", "old-token"))
        client.assert_not_called()

    def test_yes_books_callback_at_settings_time_and_leaves_line(self):
        from apps.ai_caller.models import ScheduledOutreach

        twiml = twilio_bridge.handle_long_wait_choice(self._w(), speech="yes please call me back", digits="", choice="")
        self.assertIn("We will call you back", twiml)
        self.assertIn("<Hangup/>", twiml)
        self.call.refresh_from_db()
        self.assertEqual(self.call.status, S.CALLBACK)
        row = ScheduledOutreach.objects.get(source_call=self.call)
        from apps.ai_caller.services import next_callback_time

        self.assertLess(abs((row.scheduled_at - next_callback_time(self.settings_obj)).total_seconds()), 120)
        self.assertEqual(self._w()["queue_state"], "ended")

    def test_no_keeps_holding_in_same_place(self):
        twiml = twilio_bridge.handle_long_wait_choice(self._w(), speech="", digits="2", choice="")
        self.assertIn("Please continue to hold", twiml)
        self.assertIn("<Conference", twiml)
        self.assertEqual(self._w()["queue_state"], "waiting")
        self.call.refresh_from_db()
        self.assertEqual(self.call.status, S.WAITING)

    def test_answer_wording(self):
        cases = {
            ("", "1"): True, ("", "2"): False, ("yes", ""): True, ("sure, call me back later", ""): True,
            ("no, please keep me on hold", ""): False, ("I'll wait", ""): False, ("", ""): False,
        }
        for (speech, digits), expected in cases.items():
            self.assertEqual(twilio_bridge.wants_long_wait_callback(speech, digits), expected, (speech, digits))

    def test_callback_time_follows_settings_window(self):
        from datetime import datetime as dt
        from zoneinfo import ZoneInfo

        from apps.ai_caller.services import next_callback_time

        ny = ZoneInfo("America/New_York")
        at = lambda h, m=0: dt(2026, 10, 21, h, m, tzinfo=ny)
        self.assertEqual(next_callback_time(self.settings_obj, at(10)).hour, 11)        # +1h, inside window
        nxt = next_callback_time(self.settings_obj, at(16, 30))                          # 17:30 → tomorrow 9:00
        self.assertEqual((nxt.day, nxt.hour, nxt.minute), (22, 9, 0))
        early = next_callback_time(self.settings_obj, at(6))                             # 7:00 → today 9:00
        self.assertEqual((early.day, early.hour), (21, 9))



@override_settings(CACHES=LOCAL_CACHES, BACKEND_URL="https://example.test")
class HoldAndHangupTests(TestCase):
    def test_waiting_patient_hears_hold_message_that_repeats(self):
        twiml = twilio_bridge.conference_wait_twiml({"session_id": "w", "queue_state": "waiting"})
        self.assertIn(twilio_bridge.HOLD_MESSAGE, twiml)
        # Finite ring (not loop="0") so Twilio asks again and the message repeats.
        self.assertIn('<Play loop="8">', twiml)
        self.assertNotIn('loop="0"', twiml)

    def test_patient_whose_agent_is_ringing_hears_ringback_only(self):
        twiml = twilio_bridge.conference_wait_twiml({"session_id": "a", "queue_state": "active"})
        self.assertNotIn(twilio_bridge.HOLD_MESSAGE, twiml)
        self.assertIn("<Play", twiml)

    def test_wait_url_uses_the_patients_session(self):
        twilio_bridge.save_session({"session_id": "w2", "queue_state": "waiting"})
        r = self.client.post("/api/ai-call/twilio/warm-transfer/wait/?session_id=w2")
        self.assertIn(twilio_bridge.HOLD_MESSAGE, r.content.decode())

    def test_agent_hanging_up_ends_call_with_goodbye_for_patient(self):
        patient_room = twilio_bridge.join_conference_twiml({"session_id": "p"}, is_host=True)
        agent_room = twilio_bridge.join_conference_twiml({"session_id": "p"}, is_host=False)
        self.assertIn('endConferenceOnExit="true"', agent_room)
        self.assertNotIn("Goodbye", agent_room)
        # After the conference ends the patient hears a goodbye, then the call ends.
        self.assertLess(patient_room.index("</Dial>"), patient_room.index(twilio_bridge.CALL_ENDED_MESSAGE))

    def test_system_messages_never_show_in_transcript(self):
        from apps.ai_caller.transcript_merge import strip_system_speech

        self.assertEqual(strip_system_speech(twilio_bridge.HOLD_MESSAGE), "")
        self.assertEqual(strip_system_speech(twilio_bridge.CALL_ENDED_MESSAGE), "")


class DialerLinesTests(TestCase):
    """Patients on hold for a live agent use one of the "max calls at once" lines."""

    def setUp(self):
        from apps.ai_caller.models import CallerSettings
        from apps.users.models import User

        self.user = User.objects.create(cognito_id="sub-lines", email="lines@example.com")
        cs = CallerSettings.load(self.user)
        cs.calls_enabled, cs.max_calls_per_run = True, 2
        cs.start_time, cs.end_time, cs.timezone = time(0, 0), time(23, 59), "UTC"
        cs.save()
        self.settings_obj = cs
        p = patch("apps.ai_caller.signals.publish_call_change")
        p.start()
        self.addCleanup(p.stop)

    def _call(self, status, name):
        patient = Patient.objects.create(user=self.user, first_name=name, address="x", phone_number="4155550100")
        return Call.objects.create(user=self.user, patient=patient, flow="outbound", status=status)

    def _run(self):
        with patch.object(services, "due_scheduled_outreaches", return_value=[]) as due, patch.object(
            services, "queued_calls_due_for_outbound", return_value=[]
        ) as queued:
            services._run_scheduled_outbound_for_settings(self.settings_obj)
        return due, queued

    def test_no_new_calls_while_lines_are_taken_by_a_call_and_a_waiting_patient(self):
        self._call(S.ONGOING, "Talking")
        self._call(S.WAITING, "OnHold")
        due, queued = self._run()
        due.assert_not_called()      # 2 of 2 lines busy → dialer stops here
        queued.assert_not_called()

    def test_line_frees_when_the_waiting_patient_finishes(self):
        self._call(S.ONGOING, "Talking")
        self._call(S.COMPLETED, "Done")
        due, _queued = self._run()
        self.assertEqual(due.call_args.kwargs["limit"], 1)  # one free line → one new call

    def test_patient_on_hold_is_not_dialed_again(self):
        waiting = self._call(S.WAITING, "OnHold")
        result = services.place_outbound_call_for_patient(waiting.patient.id, user=self.user)
        self.assertFalse(result.get("ok"))
        self.assertIn("already has a call in progress", result.get("error", ""))


@override_settings(CACHES=LOCAL_CACHES, BACKEND_URL="https://example.test")
class TransferReviewFixesTests(TestCase):
    """Fixes from the independent review of the multi-agent / long-wait changes."""

    def setUp(self):
        _clear_caches()
        self.mocks = {}
        for name in ("start_coordinator_leg", "announce_connecting", "announce_next_agent",
                     "announce_hold", "notify_agent_waiting_tone", "set_transfer_call_status",
                     "_hangup_call", "restore_inbound_voice_url", "_schedule_long_wait_offer"):
            p = patch.object(twilio_bridge, name)
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)

    def _patient(self, sid, agents=(AGENT_1, AGENT_2)):
        return twilio_bridge.save_session({
            "session_id": sid, "user_id": 7, "call_sid": f"CA{sid}", "retell_call_id": f"call_{sid}",
            "transfer_numbers": list(agents), "transfer_number": agents[0],
        })

    def _load(self, sid):
        return twilio_bridge.load_session(sid)

    # 1
    def test_late_no_answer_after_patient_left_changes_nothing(self):
        for sid in "AB":
            self._patient(sid)
        twilio_bridge.admit_patient_to_agent(self._load("A"))
        twilio_bridge.admit_patient_to_agent(self._load("B"))
        a = self._load("A")
        a["dial_call_sid"] = "CAagentA"
        twilio_bridge.save_session(a)
        twilio_bridge.release_agent_slot(self._load("A"))          # patient A hangs up
        self.mocks["_hangup_call"].assert_any_call("CAagentA")    # ringing agent leg stopped
        self.mocks["set_transfer_call_status"].reset_mock()
        self.mocks["start_coordinator_leg"].reset_mock()
        twilio_bridge.maybe_failover_on_agent_status(self._load("A"), call_status="no-answer", call_sid="CAagentA")
        self.mocks["set_transfer_call_status"].assert_not_called()  # stays completed
        self.mocks["start_coordinator_leg"].assert_not_called()     # nobody rung for nobody

    # 4
    def test_agent_hanging_up_before_joining_tries_next_agent(self):
        self._patient("A")
        twilio_bridge.admit_patient_to_agent(self._load("A"))
        a = self._load("A")
        a["dial_call_sid"] = "CAagent1"
        twilio_bridge.save_session(a)
        self.client.post("/api/ai-call/twilio/warm-transfer/status/?session_id=A",
                         {"CallSid": "CAagent1", "CallStatus": "completed"})
        self.assertEqual(self._load("A")["assigned_agent"], AGENT_2)

    # 6
    def test_keep_holding_puts_patient_back_in_line_if_missing(self):
        for sid in "AB":
            self._patient(sid, agents=(AGENT_1,))
        twilio_bridge.admit_patient_to_agent(self._load("A"))
        twilio_bridge.admit_patient_to_agent(self._load("B"))     # waits
        twilio_bridge._session_cache().set(twilio_bridge._queue_key("user:7"), [])  # lost by a race
        twilio_bridge.handle_long_wait_choice(self._load("B"), speech="", digits="2", choice="")
        self.assertIn("B", twilio_bridge._session_cache().get(twilio_bridge._queue_key("user:7")))

    # 9
    def test_old_conference_ending_after_rejoin_is_ignored(self):
        for sid in "AB":
            self._patient(sid, agents=(AGENT_1,))
        twilio_bridge.admit_patient_to_agent(self._load("A"))
        twilio_bridge.admit_patient_to_agent(self._load("B"))
        b = self._load("B")
        b["conference_sid"] = "CFold"
        twilio_bridge.save_session(b)
        with patch.object(twilio_bridge, "_twilio_client"):
            twilio_bridge.offer_callback_to_long_waiter("B", self._load("B")["wait_token"])
        twilio_bridge.handle_long_wait_choice(self._load("B"), speech="", digits="2", choice="")
        # The old room's "end" arrives late, after B rejoined a new room.
        twilio_bridge.handle_conference_status(self._load("B"), {"StatusCallbackEvent": "conference-end", "ConferenceSid": "CFold"})
        self.assertEqual(self._load("B")["queue_state"], "waiting")

    # 8
    def test_hold_music_explicitly_comes_back_for_the_next_message(self):
        twiml = twilio_bridge.conference_wait_twiml({"session_id": "w", "queue_state": "waiting"})
        self.assertIn('<Redirect method="POST">', twiml)
        self.assertIn("/warm-transfer/wait/", twiml)

    # 10
    def test_queue_scope_is_fixed_at_creation(self):
        session = {"session_id": "x", "transfer_number": AGENT_1, "queue_scope": "agent:14155550001"}
        session["transfer_number"] = AGENT_2   # changes on failover
        self.assertEqual(twilio_bridge._queue_scope(session), "agent:14155550001")


class TranscriptReviewFixesTests(TestCase):
    # 2
    def test_silence_never_stores_raw_json(self):
        from apps.ai_caller import transcript_merge as tm

        body = '{"text": " Thank you.", "segments": [{"start": 0, "end": 2, "text": " Thank you.", "no_speech_prob": 0.95, "avg_logprob": -1.5}]}'
        ok = type("R", (), {"status_code": 200, "text": body})()
        with override_settings(OPENAI_API_KEY="k"), patch.object(tm.requests, "post", return_value=ok):
            result = tm.transcribe_recording_detailed(b"audio", "patient.wav")
        self.assertEqual(result, {"text": "", "segments": []})

    # 3
    def test_numbers_names_and_services_are_kept(self):
        from apps.ai_caller.transcript_merge import _drop_prompt_echo_segments, strip_system_speech

        for answer in ("555-1234", "3.", "01/02/1980"):
            self.assertEqual(strip_system_speech(answer), answer)
        prompt = "A phone call between John Smith, a patient, and a Bay Area Community Health care coordinator about a mammogram."
        kept = _drop_prompt_echo_segments(
            [{"text": "John Smith."}, {"text": "About a mammogram."}, {"text": "Bay Area Community Health"}], prompt
        )
        self.assertEqual([s["text"] for s in kept], ["John Smith.", "About a mammogram."])


class WaitingLineAgeTests(TestCase):
    # 7
    def test_old_leftover_waiting_row_does_not_hold_a_line(self):
        from datetime import timedelta

        from django.utils import timezone

        patient = Patient.objects.create(first_name="Ana", address="x", phone_number="4155550100")
        with patch("apps.ai_caller.signals.publish_call_change"):
            fresh = Call.objects.create(patient=patient, flow="outbound", status=S.WAITING)
            stale = Call.objects.create(patient=patient, flow="outbound", status=S.WAITING)
        Call.objects.filter(pk=stale.pk).update(updated_at=timezone.now() - timedelta(hours=2))
        busy = set(Call.objects.filter(Call.line_busy_q()).values_list("pk", flat=True))
        self.assertIn(fresh.pk, busy)
        self.assertNotIn(stale.pk, busy)


def _wav_seconds(seconds=6.0, rate=8000):
    import io as _io
    import wave as _wave

    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate))
    return buf.getvalue()


class TwoModelTranscriptionTests(TestCase):
    """whisper-1 gives the timing; gpt-4o-transcribe re-writes each sentence's words."""

    SEGMENTS = [
        {"start": 0.5, "end": 2.0, "text": "Hi, this is Mariah.", "confidence": -0.3},
        {"start": 2.4, "end": 4.0, "text": "Does Tuesday at tree work?", "confidence": -0.4},
    ]

    def _resp(self, text, status=200):
        return type("R", (), {"status_code": status, "text": json.dumps({"text": text})})()

    def test_better_words_same_timing(self):
        from apps.ai_caller import transcript_merge as tm

        answers = {0.3: "Hi, this is Maria.", 2.2: "Does Tuesday at three work?"}

        def post(key, filename, clip, data):
            import io as _io
            import wave as _wave

            with _wave.open(_io.BytesIO(clip)) as w:
                frames = w.getnframes()
            # Identify the sentence by clip length (first: 0.3–2.2s, second: 2.2–4.2s).
            return self._resp(answers[0.3] if frames < 8000 * 1.95 else answers[2.2])

        with override_settings(CALL_TRANSCRIPTION_REFINE_MODEL="gpt-4o-transcribe"), patch.object(
            tm, "_post_transcription", side_effect=post
        ):
            out = tm.refine_segment_text(_wav_seconds(), list(self.SEGMENTS), key="k", prompt="p")
        self.assertEqual([s["text"] for s in out], ["Hi, this is Maria.", "Does Tuesday at three work?"])
        self.assertEqual([(s["start"], s["end"]) for s in out], [(0.5, 2.0), (2.4, 4.0)])
        self.assertEqual(out[0]["whisper_text"], "Hi, this is Mariah.")

    def test_clips_never_overlap_the_next_sentence(self):
        from apps.ai_caller import transcript_merge as tm

        clips = []
        with override_settings(CALL_TRANSCRIPTION_REFINE_MODEL="gpt-4o-transcribe"), patch.object(
            tm, "_wav_clip", side_effect=lambda a, s, e: clips.append((round(s, 2), round(e, 2))) or b"x"
        ), patch.object(tm, "_refine_one", return_value=""):
            tm.refine_segment_text(b"wav", list(self.SEGMENTS), key="k", prompt="p")
        self.assertEqual(clips, [(0.3, 2.2), (2.2, 4.2)])  # 2.0+0.2 meets 2.4-0.2, no overlap

    def test_failure_or_invented_text_keeps_whisper(self):
        from apps.ai_caller import transcript_merge as tm

        responses = [self._resp("", status=500), self._resp("word " * 40)]
        with override_settings(CALL_TRANSCRIPTION_REFINE_MODEL="gpt-4o-transcribe"), patch.object(
            tm, "_post_transcription", side_effect=responses
        ):
            out = tm.refine_segment_text(_wav_seconds(), list(self.SEGMENTS), key="k", prompt="p")
        self.assertEqual([s["text"] for s in out], [s["text"] for s in self.SEGMENTS])

    def test_can_be_switched_off(self):
        from apps.ai_caller import transcript_merge as tm

        with override_settings(CALL_TRANSCRIPTION_REFINE_MODEL=""), patch.object(tm, "_post_transcription") as post:
            out = tm.refine_segment_text(_wav_seconds(), list(self.SEGMENTS), key="k", prompt="p")
        post.assert_not_called()
        self.assertEqual(out, self.SEGMENTS)

    def test_invented_sentence_dropped_when_better_model_hears_nothing(self):
        from apps.ai_caller import transcript_merge as tm

        segs = [
            {"start": 0.5, "end": 2.0, "text": "Yes.", "confidence": -0.3},
            {"start": 2.4, "end": 4.0, "text": "I'd like to make a phone call about a Dental Exam.", "confidence": -0.2},
            {"start": 4.4, "end": 5.5, "text": "Okay, see you then.", "confidence": -0.2},
        ]
        # "Yes." too short to judge -> kept; long invented sentence -> dropped;
        # prompt markup is cleaned out of a real answer.
        responses = [self._resp(""), self._resp("context: ###"), self._resp("context: Okay, see you then.")]
        with override_settings(CALL_TRANSCRIPTION_REFINE_MODEL="gpt-4o-transcribe"), patch.object(
            tm, "_post_transcription", side_effect=responses
        ), patch.object(tm, "_REFINE_WORKERS", 1):
            out = tm.refine_segment_text(_wav_seconds(), segs, key="k", prompt="p")
        self.assertEqual([s["text"] for s in out], ["Yes.", "Okay, see you then."])
