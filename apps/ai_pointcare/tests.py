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
