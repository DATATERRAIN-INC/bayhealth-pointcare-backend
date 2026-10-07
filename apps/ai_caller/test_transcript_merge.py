"""Prove warm-transfer human transcripts stay speaker-separated.

Run:
  ./venv/bin/python -m unittest apps.ai_caller.test_transcript_merge -v
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from apps.ai_caller.transcript_merge import (
    assert_speakers_separated,
    humans_items_from_texts,
    interleave_speaker_segments,
    labeled_leg_items,
    merge_ai_and_humans,
    save_humans_transcript_for_call,
    transcribe_recording_detailed,
)


# Exact style of the broken mixed wall the user reported.
_MIXED_WALL = (
    "Hello Vijay. Actually, I want to book an appointment. Can you help me with that? "
    "Yes, we can book an appointment for you. When would you like to get the appointment? "
    "Tomorrow 10 am and another slot day after tomorrow 11 am is fine for me. "
    "It's fine to work with you? I think no. Tomorrow I mean I'm available only after 4 pm. "
    "Okay, that time is fine for me. Can we book an appointment to that particular time and date? "
    "Yes, we can. Okay, thank you Prakash. Thank you Vijay."
)

_PATIENT_TEXT = (
    "Hello Vijay. Actually, I want to book an appointment. Can you help me with that? "
    "Tomorrow 10 am and another slot day after tomorrow 11 am is fine for me. "
    "I think no. Tomorrow I mean I'm available only after 4 pm. "
    "Can we book an appointment to that particular time and date? "
    "Okay, thank you Prakash."
)

_PROVIDER_TEXT = (
    "Yes, we can book an appointment for you. When would you like to get the appointment? "
    "It's fine to work with you? Okay, that time is fine for me. "
    "Yes, we can. Thank you Vijay."
)


class SpeakerSeparationTests(unittest.TestCase):
    def test_labeled_legs_never_unknown(self):
        items = labeled_leg_items(
            patient_text=_PATIENT_TEXT,
            provider_text=_PROVIDER_TEXT,
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["speaker"], "patient")
        self.assertEqual(items[1]["speaker"], "live_agent")
        self.assertTrue(assert_speakers_separated(items))
        # Must not be one fused unknown blob.
        self.assertNotEqual(items[0]["text"], _MIXED_WALL)

    def test_plain_legs_preferred_over_mix_wall(self):
        items = humans_items_from_texts(
            patient_text=_PATIENT_TEXT,
            provider_text=_PROVIDER_TEXT,
            mix_text=_MIXED_WALL,
        )
        speakers = [i["speaker"] for i in items]
        self.assertIn("patient", speakers)
        self.assertIn("live_agent", speakers)
        self.assertNotIn("unknown", speakers)
        self.assertTrue(assert_speakers_separated(items))

    def test_timestamp_interleave_alternates_speakers(self):
        patient_segs = [
            {"start": 0.0, "end": 3.0, "text": "I want to book an appointment."},
            {"start": 8.0, "end": 12.0, "text": "Tomorrow after 4 pm."},
            {"start": 16.0, "end": 18.0, "text": "Thank you Prakash."},
        ]
        provider_segs = [
            {"start": 3.5, "end": 7.0, "text": "When would you like the appointment?"},
            {"start": 12.5, "end": 15.0, "text": "Okay, that time works."},
            {"start": 18.5, "end": 20.0, "text": "Thank you Vijay."},
        ]
        items = interleave_speaker_segments(
            patient_segments=patient_segs,
            provider_segments=provider_segs,
            patient_offset=1000.0,
            provider_offset=1000.0,
        )
        speakers = [i["speaker"] for i in items]
        self.assertEqual(
            speakers,
            [
                "patient",
                "live_agent",
                "patient",
                "live_agent",
                "patient",
                "live_agent",
            ],
        )
        self.assertTrue(assert_speakers_separated(items))
        # Provider joined later — offsets must keep order.
        late = interleave_speaker_segments(
            patient_segments=patient_segs,
            provider_segments=provider_segs,
            patient_offset=1000.0,
            provider_offset=1005.0,
        )
        self.assertEqual(late[0]["speaker"], "patient")
        self.assertGreater(late[1]["at"], late[0]["at"])

    def test_mix_only_is_last_resort_unknown(self):
        items = humans_items_from_texts(mix_text=_MIXED_WALL)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["speaker"], "unknown")
        self.assertFalse(assert_speakers_separated(items))

    def test_diarized_mix_items_used_when_no_legs(self):
        mix_items = [
            {"speaker": "patient", "text": "Book me please", "segment": "human", "at": 0.1},
            {"speaker": "live_agent", "text": "Sure", "segment": "human", "at": 1.2},
        ]
        items = humans_items_from_texts(mix_items=mix_items, mix_text=_MIXED_WALL)
        self.assertEqual(items, mix_items)
        self.assertTrue(assert_speakers_separated(items))

    def test_merge_keeps_ai_then_human(self):
        humans = labeled_leg_items(
            patient_text="I need help",
            provider_text="I can help",
        )
        merged = merge_ai_and_humans(
            [{"speaker": "agent", "text": "Transferring you now", "at": 1}],
            humans,
        )
        self.assertEqual(merged[0]["segment"], "ai")
        self.assertEqual(merged[1]["segment"], "human")
        self.assertEqual(merged[2]["speaker"], "live_agent")


class TranscribeDetailedTests(unittest.TestCase):
    @patch("apps.ai_caller.transcript_merge._openai_key", return_value="sk-test")
    @patch("apps.ai_caller.transcript_merge.requests.post")
    def test_whisper_verbose_json_segments(self, mock_post, _key):
        payload = {
            "text": "Hello. Sure thing.",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": " Hello."},
                {"start": 1.0, "end": 2.5, "text": " Sure thing."},
            ],
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.text = json.dumps(payload)
        mock_post.return_value = resp

        result = transcribe_recording_detailed(b"fake-audio", filename="patient.wav")
        self.assertEqual(result["text"], "Hello. Sure thing.")
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0]["text"], "Hello.")
        self.assertEqual(result["segments"][1]["start"], 1.0)


class PersistToCallTests(unittest.TestCase):
    @patch("apps.ai_caller.models.Call.objects")
    def test_save_writes_labeled_live_agent_transcript(self, mock_objects):
        call = MagicMock()
        call.retell_transcript = [
            {"speaker": "agent", "text": "Connecting you now", "at": 1, "segment": "ai"}
        ]
        call.warm_transfer_session_id = ""
        call.recording_url = ""
        mock_objects.filter.return_value.first.return_value = call

        saved = save_humans_transcript_for_call(
            retell_call_id="call_test_123",
            patient_text=_PATIENT_TEXT,
            provider_text=_PROVIDER_TEXT,
            patient_segments=[
                {"start": 0.0, "end": 2.0, "text": "I want to book an appointment."},
                {"start": 6.0, "end": 8.0, "text": "After 4 pm works."},
            ],
            provider_segments=[
                {"start": 2.5, "end": 5.0, "text": "When works for you?"},
                {"start": 8.5, "end": 10.0, "text": "Booked."},
            ],
            patient_offset=100.0,
            provider_offset=100.0,
            recording_url="https://s3.example.com/warm-transfer-recordings/patient.wav",
            twilio_call_sid="CAfdf3f0de1aa3fad3010bca21bf9d2c27",
        )
        self.assertIs(saved, call)
        humans = call.live_agent_transcript
        self.assertTrue(assert_speakers_separated(humans))
        self.assertGreaterEqual(len(humans), 4)
        self.assertEqual(
            [h["speaker"] for h in humans],
            ["patient", "live_agent", "patient", "live_agent"],
        )
        # Merged transcript must include AI then humans.
        self.assertEqual(call.transcript[0]["segment"], "ai")
        self.assertIn(call.transcript[1]["speaker"], {"patient", "live_agent"})
        self.assertEqual(
            call.warm_transfer_session_id, "CAfdf3f0de1aa3fad3010bca21bf9d2c27"
        )
        call.save.assert_called_once()


class UserRegressionTests(unittest.TestCase):
    """The exact failure mode from production: one unlabeled wall of text."""

    def test_user_sample_is_speaker_split_with_per_leg_audio(self):
        items = humans_items_from_texts(
            patient_text=_PATIENT_TEXT,
            provider_text=_PROVIDER_TEXT,
        )
        # Can attribute booking intent to patient and confirmation to agent.
        patient_text = " ".join(
            i["text"] for i in items if i["speaker"] == "patient"
        )
        agent_text = " ".join(
            i["text"] for i in items if i["speaker"] == "live_agent"
        )
        self.assertIn("book an appointment", patient_text.lower())
        self.assertIn("after 4 pm", patient_text.lower())
        self.assertIn("yes, we can", agent_text.lower())
        self.assertNotIn("yes, we can book", patient_text.lower())
        self.assertTrue(assert_speakers_separated(items))


if __name__ == "__main__":
    unittest.main()
