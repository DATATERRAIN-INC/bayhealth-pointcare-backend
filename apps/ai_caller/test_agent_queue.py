"""One live-agent conversation at a time; other patients stay waiting calls.

Run:
  ./venv/bin/python -m unittest apps.ai_caller.test_agent_queue -v
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from apps.ai_caller.twilio_bridge import (
    admit_patient_to_agent,
    release_agent_slot,
    resolve_inbound_session,
    save_session,
)


class _FakeCache:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value, timeout=None):
        self.data[key] = value

    def add(self, key, value, timeout=None):
        if key in self.data:
            return False
        self.data[key] = value
        return True

    def delete(self, key):
        self.data.pop(key, None)


class AgentQueueTests(unittest.TestCase):
    def setUp(self):
        self.cache = _FakeCache()
        self.cache_patch = patch(
            "apps.ai_caller.twilio_bridge._session_cache", return_value=self.cache
        )
        self.cache_patch.start()
        self.status_patch = patch(
            "apps.ai_caller.twilio_bridge.set_transfer_call_status"
        )
        self.status = self.status_patch.start()
        self.dial_patch = patch("apps.ai_caller.twilio_bridge.start_coordinator_leg")
        self.dial = self.dial_patch.start()
        self.announce_patch = patch("apps.ai_caller.twilio_bridge.announce_connecting")
        self.announce_patch.start()

    def tearDown(self):
        patch.stopall()

    def _session(self, sid, phone):
        return save_session(
            {
                "session_id": sid,
                "phone_number": phone,
                "user_id": 7,
                "transfer_number": "+15550001111",
                "retell_call_id": f"call_{sid}",
            }
        )

    def test_only_first_patient_is_connected(self):
        a = self._session("a", "+15551110001")
        b = self._session("b", "+15551110002")
        c = self._session("c", "+15551110003")
        self.assertEqual(admit_patient_to_agent(a), "active")
        self.assertEqual(admit_patient_to_agent(b), "waiting")
        self.assertEqual(admit_patient_to_agent(c), "waiting")
        self.dial.assert_not_called()

    def test_finishing_active_call_promotes_next_waiting_patient(self):
        a = self._session("a", "+15551110001")
        b = self._session("b", "+15551110002")
        admit_patient_to_agent(a)
        admit_patient_to_agent(b)
        from apps.ai_caller.twilio_bridge import load_session

        a = load_session("a")
        b = load_session("b")
        a["call_sid"] = "CAa"
        b["call_sid"] = "CAb"
        save_session(a)
        save_session(b)
        release_agent_slot(a)
        self.dial.assert_called_once()
        promoted = self.dial.call_args.args[0]
        self.assertEqual(promoted["session_id"], "b")
        self.assertEqual(promoted["queue_state"], "active")

    def test_inbound_matches_patient_phone_not_the_latest_session(self):
        self._session("a", "+15551110001")
        self._session("b", "+15551110002")
        # Voice URL was overwritten to session A, but the caller is patient B.
        chosen = resolve_inbound_session(
            call_sid="CAb",
            from_number="+15551110002",
            to_number="+18005550199",
        )
        self.assertEqual(chosen["session_id"], "b")
        self.assertEqual(chosen["call_sid"], "CAb")

    def test_inbound_without_caller_id_assigns_separate_sessions(self):
        self._session("a", "+15551110001")
        self._session("b", "+15551110002")
        self._session("c", "+15551110003")
        first = resolve_inbound_session(call_sid="CA1", from_number="+18005550199")
        second = resolve_inbound_session(call_sid="CA2", from_number="+18005550199")
        third = resolve_inbound_session(call_sid="CA3", from_number="+18005550199")
        self.assertEqual(
            [first["session_id"], second["session_id"], third["session_id"]],
            ["a", "b", "c"],
        )
