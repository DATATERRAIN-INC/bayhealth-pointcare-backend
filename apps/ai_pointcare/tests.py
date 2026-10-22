from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient

from apps.ai_pointcare.flow_service import handle_retell_function
from apps.ai_pointcare.models import OptOut, Outreach, RenewalCallSession


class _AuthUser:
    is_authenticated = True
    is_active = True
    pk = 1


class MediCalCallApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.client.force_authenticate(user=_AuthUser())

    def test_requires_agent_number(self):
        response = self.client.post(
            "/api/ai-pointcare/call/",
            {"phone_number": "4085551234", "name": "Jane Doe"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["status"], "failed")

    @patch("apps.ai_pointcare.services.place_medi_cal_renewal_call")
    def test_places_call(self, place_call):
        place_call.return_value = {
            "ok": True,
            "call_id": "call_123",
            "status": "registered",
            "agent_id": "agent_123",
            "from_number": "+14085550000",
            "agent_number": "+14085559999",
            "starting_state": "opening_identity",
            "message": "Medi-Cal renewal state-machine call placed through Retell AI.",
        }
        response = self.client.post(
            "/api/ai-pointcare/call/",
            {
                "phone_number": "4085551234",
                "name": "Jane Doe",
                "agent_number": "4085559999",
                "address": "123 Main St, San Jose, CA",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        result = response.data["result"]
        self.assertEqual(result["call_id"], "call_123")
        self.assertEqual(result["phone_number"], "+14085551234")
        self.assertEqual(result["name"], "Jane Doe")
        outreach = Outreach.objects.get(pk=result["outreach_id"])
        self.assertEqual(outreach.status, Outreach.STATUS_CALLING)
        self.assertEqual(outreach.address, "123 Main St, San Jose, CA")
        place_call.assert_called_once()

    def test_opted_out_number_is_rejected(self):
        OptOut.objects.create(phone_number="+14085551234")
        response = self.client.post(
            "/api/ai-pointcare/call/",
            {
                "phone_number": "4085551234",
                "name": "Jane Doe",
                "agent_number": "4085559999",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("opted out", response.data["message"])
        self.assertFalse(Outreach.objects.exists())


class RetellFunctionTests(TestCase):
    def test_record_dnc_opts_out(self):
        session = RenewalCallSession.objects.create(
            retell_call_id="call_dnc",
            patient_name="Jane Doe",
            phone_number="+14085551234",
        )
        result = handle_retell_function(
            "record_dnc",
            {},
            {"call_id": session.retell_call_id},
        )
        self.assertTrue(result["ok"])
        self.assertTrue(OptOut.objects.filter(phone_number="+14085551234").exists())
        session.refresh_from_db()
        self.assertEqual(session.final_outcome, "PATIENT_DECLINED")

    def test_unknown_function(self):
        result = handle_retell_function("not_a_tool", {}, {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "Unknown call_id.")


class ContactVerificationRemovedTests(TestCase):
    """The renewal call goes purpose -> renewal status; no address/phone/email check."""

    def test_flow_skips_contact_information(self):
        from apps.ai_pointcare.flow_definition import NODE_PURPOSE, NODE_RENEWAL, build_states

        states = build_states("https://example.test/fn/")
        names = {s["name"] for s in states}
        self.assertNotIn("contact_information", names)
        for state in states:
            for edge in state.get("edges") or []:
                self.assertIn(edge["destination_state_name"], names)
        purpose = next(s for s in states if s["name"] == NODE_PURPOSE)
        self.assertIn(NODE_RENEWAL, [e["destination_state_name"] for e in purpose["edges"]])
        self.assertNotIn("transition_to_contact_information", purpose["state_prompt"])

    def test_language_step_waits_for_patient_answer(self):
        from apps.ai_pointcare.flow_definition import NODE_LANGUAGE, build_states

        lang = next(s for s in build_states("https://example.test/fn/") if s["name"] == NODE_LANGUAGE)
        self.assertIn("NOT an answer to the language question", lang["state_prompt"])
        self.assertIn("NEVER call transition_to_purpose in the same turn", lang["state_prompt"])
        self.assertEqual(lang["edges"][0]["parameters"]["required"], ["preferred_language"])

    def test_step_names_never_spoken(self):
        from apps.ai_pointcare.flow_definition import build_llm_payload, build_states, build_tools

        payload = build_llm_payload(function_url="https://example.test/fn/", patient_name="Sam")
        self.assertIn("NEVER say a function or step name", payload["general_prompt"])
        for state in build_states("https://example.test/fn/"):
            self.assertIn("silent function calls", state["state_prompt"])
        # Anything spoken while a tool runs is fixed text, never written by the model.
        for tool in build_tools("https://example.test/fn/").values():
            if tool.get("speak_during_execution"):
                self.assertEqual(tool.get("execution_message_type"), "static_text", tool["name"])


class RenewalVideoResendTests(TestCase):
    """Patient says the text never came: one real resend, then no more texts."""

    def test_one_resend_then_stop(self):
        from apps.ai_pointcare.flow_definition import build_llm_payload

        session = RenewalCallSession.objects.create(
            retell_call_id="call_video", patient_name="Jane Doe", phone_number="+14085551234"
        )
        call = {"call_id": "call_video"}
        sent = {"ok": True, "sid": "SM1", "from_number": "+1800", "video_url": "u", "body": "b"}
        with patch("apps.ai_pointcare.flow_service.send_renewal_video_sms", return_value=sent) as sms:
            first = handle_retell_function("send_renewal_video_sms", {}, call)
            second = handle_retell_function("send_renewal_video_sms", {}, call)
            third = handle_retell_function("send_renewal_video_sms", {}, call)
        self.assertEqual(sms.call_count, 2)
        self.assertTrue(first["video_sent"])
        self.assertIn("silently call transition_to_household_income", first["message"])
        self.assertTrue(second["video_sent"])
        self.assertIn("continue the question you were on", second["message"])
        self.assertFalse(third["video_sent"])
        session.refresh_from_db()
        self.assertEqual(session.state_payload["renewal_video_sms_count"], 2)
        # The tool exists in every step, so a resend later in the call is possible.
        payload = build_llm_payload(function_url="https://example.test/fn/", patient_name="Jane")
        self.assertIn("send_renewal_video_sms", [t["name"] for t in payload["general_tools"]])
        for state in payload["states"]:
            self.assertNotIn("send_renewal_video_sms", [t["name"] for t in state.get("tools") or []])

    def test_help_step_asks_before_transfer(self):
        from apps.ai_pointcare.flow_definition import NODE_ASSISTANCE, build_states

        step = next(s for s in build_states("https://example.test/fn/") if s["name"] == NODE_ASSISTANCE)
        self.assertIn('Do NOT say "I\'ll connect you"', step["state_prompt"])
        self.assertIn("NOT an answer", step["state_prompt"])

    def test_no_command_style_text_for_the_ai_to_copy(self):
        """The AI copied 'transition_to_X with a=b' from its instructions into speech."""
        import re

        from apps.ai_pointcare.flow_definition import build_llm_payload

        payload = build_llm_payload(function_url="https://example.test/fn/", patient_name="Jane")
        texts = [payload["general_prompt"]]
        for state in payload["states"]:
            texts.append(state["state_prompt"])
            texts.extend(edge["description"] for edge in state.get("edges") or [])
        for text in texts:
            self.assertIsNone(re.search(r"\b[a-z_]+=", text), text[:120])
            self.assertIsNone(re.search(r"transition_to_\w+ with ", text), text[:120])
