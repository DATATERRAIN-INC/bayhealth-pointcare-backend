"""Dry-run verification for warm-transfer speaker-split transcripts.

Runs without placing a live call:
  1) Logic tests (recording retry, mix-skip, stereo split, merge)
  2) Live STT/diarize against the latest Twilio recording in DB (if any)

Run:
  ./venv/bin/python -m unittest apps.ai_caller.test_warm_transfer_dry -v
"""

from __future__ import annotations

import io
import json
import os
import struct
import unittest
import wave
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from django.conf import settings

from apps.ai_caller.transcript_merge import (
    assert_speakers_separated,
    humans_items_from_texts,
    save_humans_transcript_for_call,
    transcribe_mix_with_diarization,
)
from apps.ai_caller.twilio_bridge import (
    _has_speaker_split,
    _should_skip_mix,
    patch_session,
    load_session,
    split_stereo_wav,
    start_leg_recording,
)


def _stereo_wav_bytes(seconds: float = 0.2, framerate: int = 8000) -> bytes:
    n = int(framerate * seconds)
    left = b"".join(struct.pack("<h", 1200) for _ in range(n))
    right = b"".join(struct.pack("<h", -1200) for _ in range(n))
    interleaved = b"".join(
        left[i : i + 2] + right[i : i + 2] for i in range(0, len(left), 2)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(interleaved)
    return buf.getvalue()


class MixSkipLogicTests(unittest.TestCase):
    def test_do_not_skip_mix_when_only_provider_started(self):
        # Regression: 21220 patient fail + provider started used to skip mix forever.
        self.assertFalse(
            _should_skip_mix(
                {
                    "provider_recording_started": True,
                    "provider_transcript": "hello from agent only",
                }
            )
        )

    def test_skip_mix_when_both_legs_have_text(self):
        self.assertTrue(
            _should_skip_mix(
                {
                    "patient_transcript": "hi",
                    "provider_transcript": "hello",
                }
            )
        )
        self.assertTrue(_has_speaker_split({"dual_channel_split": True}))


class StereoSplitTests(unittest.TestCase):
    def test_split_stereo_returns_two_mono_wavs(self):
        raw = _stereo_wav_bytes()
        left, right = split_stereo_wav(raw)
        self.assertTrue(left)
        self.assertTrue(right)
        self.assertNotEqual(left, right)
        with wave.open(io.BytesIO(left), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
        with wave.open(io.BytesIO(right), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)


class RecordingRetryTests(unittest.TestCase):
    @patch("apps.ai_caller.twilio_bridge._twilio_client")
    @patch("apps.ai_caller.twilio_bridge.save_session", side_effect=lambda s: s)
    @patch("apps.ai_caller.twilio_bridge.patch_session")
    @patch("apps.ai_caller.twilio_bridge.load_session")
    def test_patient_dual_21220_falls_back_to_mono(
        self, mock_load, mock_patch, _save, mock_client
    ):
        mock_load.return_value = {"session_id": "dry1"}
        client = MagicMock()
        mock_client.return_value = client

        class E21220(Exception):
            pass

        dual_err = E21220(
            "Unable to create record: Requested resource is not eligible "
            "for recording ... 21220"
        )
        mono_rec = MagicMock()
        mono_rec.sid = "REmono123"
        client.calls.return_value.recordings.create.side_effect = [
            dual_err,
            mono_rec,
        ]

        start_leg_recording(
            {"session_id": "dry1"},
            call_sid="CAdrypatient",
            leg="patient",
        )
        self.assertEqual(client.calls.return_value.recordings.create.call_count, 2)
        # Final patch should store mono recording sid.
        patched = False
        for call in mock_patch.call_args_list:
            args, kwargs = call
            updates = args[1] if len(args) > 1 else {}
            if updates.get("patient_twilio_recording_sid") == "REmono123":
                patched = True
                self.assertFalse(updates.get("patient_recording_dual"))
        self.assertTrue(patched)


class SessionPatchRaceTests(unittest.TestCase):
    def test_parallel_leg_patches_keep_both_transcripts(self):
        sid = "dryrace999"
        patch_session(sid, {"session_id": sid, "patient_transcript": "patient only"})
        patch_session(sid, {"provider_transcript": "agent only"})
        final = load_session(sid)
        self.assertEqual(final.get("patient_transcript"), "patient only")
        self.assertEqual(final.get("provider_transcript"), "agent only")


class MergePersistTests(unittest.TestCase):
    @patch("apps.ai_caller.models.Call.objects")
    def test_both_legs_never_one_patient_wall(self, mock_objects):
        call = MagicMock()
        call.retell_transcript = [
            {"speaker": "agent", "text": "Connecting you", "segment": "ai", "at": 1}
        ]
        call.live_agent_transcript = []
        call.warm_transfer_session_id = ""
        call.recording_url = ""
        call.live_agent_recording_url = ""
        mock_objects.filter.return_value.first.return_value = call

        save_humans_transcript_for_call(
            retell_call_id="call_dry_merge",
            patient_text="I want Friday afternoon.",
            provider_text="Four o'clock is available.",
            patient_segments=[
                {"start": 0.0, "end": 2.0, "text": "I want Friday afternoon."}
            ],
            provider_segments=[
                {"start": 2.5, "end": 4.0, "text": "Four o'clock is available."}
            ],
            twilio_call_sid="CAdry",
        )
        humans = call.live_agent_transcript
        speakers = {h["speaker"] for h in humans}
        self.assertIn("patient", speakers)
        self.assertIn("live_agent", speakers)
        self.assertTrue(assert_speakers_separated(humans))
        # Must not be a single fused patient wall.
        self.assertFalse(len(humans) == 1 and humans[0]["speaker"] == "patient")


@unittest.skipUnless(
    bool((getattr(settings, "OPENAI_API_KEY", "") or "").strip())
    and bool((getattr(settings, "TWILIO_ACCOUNT_SID", "") or "").strip()),
    "Needs OPENAI_API_KEY + Twilio credentials",
)
class LiveRecordingDiarizeDryTest(unittest.TestCase):
    """Download latest warm-transfer recording and prove diarize returns 2 speakers."""

    def test_latest_call_recording_diarizes_to_two_speakers(self):
        from twilio.rest import Client

        from apps.ai_caller.models import Call
        from apps.ai_caller.twilio_bridge import _download_twilio_recording

        call = (
            Call.objects.exclude(warm_transfer_session_id="")
            .exclude(warm_transfer_session_id__isnull=True)
            .order_by("-id")
            .first()
        )
        self.assertIsNotNone(call, "No warm-transfer Call row found")
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        recs = [
            r
            for r in client.recordings.list(
                call_sid=call.warm_transfer_session_id, limit=20
            )
            if str(r.status).lower() == "completed" and int(r.duration or 0) >= 5
        ]
        self.assertTrue(recs, "No completed Twilio recording on latest warm-transfer call")
        recs.sort(key=lambda r: int(r.duration or 0), reverse=True)
        rec = recs[0]
        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{settings.TWILIO_ACCOUNT_SID}/Recordings/{rec.sid}"
        )
        audio = _download_twilio_recording(url)
        self.assertGreater(len(audio or b""), 1000, "Recording download empty")

        items = transcribe_mix_with_diarization(audio, filename="dry_mix.wav")
        self.assertGreaterEqual(len(items), 2, f"Diarize returned too few turns: {items!r}")
        speakers = {str(i.get("speaker") or "") for i in items}
        self.assertIn("patient", speakers)
        self.assertIn("live_agent", speakers)
        self.assertTrue(assert_speakers_separated(items))
        # No unlabeled wall.
        self.assertNotIn("unknown", speakers)
        wall = humans_items_from_texts(mix_text=" ".join(i["text"] for i in items))
        self.assertFalse(assert_speakers_separated(wall))  # wall alone fails
        self.assertTrue(assert_speakers_separated(items))  # diarized passes


if __name__ == "__main__":
    unittest.main(verbosity=2)
