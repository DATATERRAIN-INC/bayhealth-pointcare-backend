"""Dry-run checks for SMS → live-agent recording + merged transcript.

Run:
  ./venv/bin/python -m unittest apps.ai_sms.test_live_call_dry -v
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from apps.ai_sms.agent_dial import (
    _persist_agent_dial_recording,
    agent_answer_twiml,
    handle_agent_dial_recording,
)
from apps.ai_sms.models import SmsConversation
from apps.ai_sms.views import (
    _conversation_transcript_payload,
    _live_call_transcript_items,
    _sms_transcript_items,
)


def _sample_conversation(**extra) -> SmsConversation:
    defaults = dict(
        chat_id="chat_dry_sms_live",
        to_number="+15551234567",
        from_number="+15557654321",
        patient_name="Vijith Vijayan",
        service_name="dental",
        transcript=(
            "Agent: Hi, this is Bay Area Community Health. Am I texting with Vijith?\n"
            "Patient: YES\n"
            "Agent: I'll connect you with a team member now — you'll receive a call shortly."
        ),
        recording_url="",
        live_call_transcript=[],
    )
    defaults.update(extra)
    return SmsConversation(**defaults)


class SmsTranscriptMergeDryTests(unittest.TestCase):
    def test_sms_items_label_agent_and_patient(self):
        items = _sms_transcript_items(
            "Agent: hello there\nPatient: YES\nGuardian: ok"
        )
        speakers = [i["speaker"] for i in items]
        self.assertEqual(speakers, ["agent", "patient", "patient"])
        self.assertEqual(items[0]["name"], "AI agent")
        self.assertEqual(items[1]["name"], "Patient")
        self.assertEqual(items[1]["segment"], "human")

    def test_live_call_items_patient_and_live_agent_only(self):
        items = _live_call_transcript_items(
            [
                {"speaker": "patient", "text": "I need an appointment"},
                {"speaker": "live_agent", "text": "I can help"},
                {"speaker": "agent", "text": "should be dropped"},
                {"speaker": "patient", "text": "  "},
            ]
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["speaker"], "patient")
        self.assertEqual(items[0]["name"], "Patient")
        self.assertEqual(items[1]["speaker"], "live_agent")
        self.assertEqual(items[1]["name"], "Live agent")

    def test_payload_merges_sms_then_live_call_no_live_call_key(self):
        conversation = _sample_conversation(
            recording_url="https://s3.example.com/sms_live_call.wav",
            live_call_transcript=[
                {
                    "speaker": "patient",
                    "text": "Hi, I need help scheduling.",
                    "at": 0.5,
                },
                {
                    "speaker": "live_agent",
                    "text": "Sure, what day works?",
                    "at": 4.2,
                },
            ],
        )
        payload = _conversation_transcript_payload(conversation, request=None)
        self.assertIn("chat_id", payload)
        self.assertIn("transcript", payload)
        self.assertIn("recording_url", payload)
        self.assertNotIn("live_call_transcript", payload)

        speakers = [row["speaker"] for row in payload["transcript"]]
        # SMS first (agent, patient, agent), then live call (patient, live_agent)
        self.assertEqual(
            speakers,
            ["agent", "patient", "agent", "patient", "live_agent"],
        )
        self.assertEqual(
            payload["recording_url"],
            "https://s3.example.com/sms_live_call.wav",
        )
        names = {row["speaker"]: row["name"] for row in payload["transcript"]}
        self.assertEqual(names["agent"], "AI agent")
        self.assertEqual(names["patient"], "Patient")
        self.assertEqual(names["live_agent"], "Live agent")


class AgentDialRecordingTwimlDryTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {"BACKEND_URL": "https://example.tunnel/backend"},
        clear=False,
    )
    @patch("apps.ai_sms.agent_dial._backend_base", return_value="https://example.tunnel")
    @patch(
        "apps.ai_sms.agent_dial._twilio_creds",
        return_value=("ACxxx", "token", "+15550001111", ""),
    )
    def test_answer_twiml_enables_dual_record_and_callback(self, _creds, _base):
        conversation = _sample_conversation()
        xml = agent_answer_twiml(conversation)
        self.assertIn('record="record-from-answer-dual"', xml)
        self.assertIn(
            "/api/ai-sms/agent-dial/recording/chat_dry_sms_live/",
            xml,
        )
        self.assertIn(conversation.to_number, xml)


class PersistAgentDialRecordingDryTests(unittest.TestCase):
    @patch("apps.ai_sms.agent_dial.conversation_by_chat_id")
    @patch("apps.ai_caller.twilio_bridge.split_stereo_wav")
    @patch("apps.ai_caller.twilio_bridge._download_twilio_recording")
    @patch("apps.ai_caller.twilio_bridge._upload_recording_audio")
    @patch("apps.ai_caller.transcript_merge.transcribe_recording_detailed")
    def test_persist_saves_one_mix_url_and_split_transcript(
        self,
        mock_stt,
        mock_upload,
        mock_download,
        mock_split,
        mock_get,
    ):
        conversation = MagicMock()
        conversation.chat_id = "chat_dry_sms_live"
        conversation.recording_url = ""
        conversation.live_call_transcript = []
        mock_get.return_value = conversation
        mock_download.return_value = b"FAKE_WAV_BYTES_DUAL"
        mock_split.return_value = (b"AGENT_MONO", b"PATIENT_MONO")
        mock_upload.return_value = "https://s3.example.com/mix.wav"

        def _stt(audio, filename="x.wav"):
            if b"PATIENT" in audio:
                return {
                    "text": "I need an appointment",
                    "segments": [{"start": 0.5, "end": 2.0, "text": "I need an appointment"}],
                }
            return {
                "text": "I can help",
                "segments": [{"start": 2.5, "end": 4.0, "text": "I can help"}],
            }

        mock_stt.side_effect = _stt

        _persist_agent_dial_recording(
            "chat_dry_sms_live",
            "https://api.twilio.com/Recordings/RExxx",
            "RExxx",
        )

        mock_upload.assert_called_once()
        self.assertEqual(mock_upload.call_args.kwargs.get("leg"), "sms_live_call")
        self.assertEqual(conversation.recording_url, "https://s3.example.com/mix.wav")
        self.assertTrue(conversation.live_call_transcript)
        speakers = {row["speaker"] for row in conversation.live_call_transcript}
        self.assertEqual(speakers, {"patient", "live_agent"})
        conversation.save.assert_called_once()
        saved_fields = conversation.save.call_args.kwargs["update_fields"]
        self.assertIn("recording_url", saved_fields)
        self.assertIn("live_call_transcript", saved_fields)
        self.assertNotIn("live_agent_recording_url", saved_fields)

    @patch("apps.ai_sms.agent_dial.threading.Thread")
    def test_handle_starts_background_thread(self, mock_thread):
        conversation = _sample_conversation()
        handle_agent_dial_recording(
            conversation,
            recording_url="https://api.twilio.com/Recordings/RExxx",
            recording_sid="RExxx",
        )
        mock_thread.assert_called_once()
        self.assertTrue(mock_thread.call_args.kwargs.get("daemon"))
        mock_thread.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
