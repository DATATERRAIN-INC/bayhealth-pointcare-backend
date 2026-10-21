"""Call status + warm-transfer transcript regression tests. External APIs are mocked."""

from unittest.mock import patch

from django.test import TestCase, override_settings

from apps.ai_caller import services, twilio_bridge
from apps.ai_caller.models import Call, Patient

S = Call.Status

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
