"""SMS flow regression tests. Twilio / OpenAI / Retell are always mocked."""

from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from apps.ai_sms import agent_dial, services
from apps.ai_sms.models import SmsConversation
from apps.ai_sms.prompt import booked_message, closed_reply, live_agent_message

ADULT_LINE = "+15550000001"
MINOR_LINE = "+15550000002"
PATIENT = "+14155550123"
AGENTS = ["+14155550001", "+14155550002"]

TWILIO = dict(
    BACKEND_URL="https://example.test",
    TWILIO_ACCOUNT_SID="ACtest",
    TWILIO_AUTH_TOKEN="token",
    TWILIO_PHONE_NUMBER=ADULT_LINE,
    TWILIO_CARECALL_FROM_NUMBER=ADULT_LINE,
    TWILIO_MINOR_SMS_FROM_NUMBER=MINOR_LINE,
    OPENAI_API_KEY="",
)


def _conversation(**extra) -> SmsConversation:
    fields = dict(
        chat_id="chat_test",
        to_number=PATIENT,
        from_number=ADULT_LINE,
        patient_name="Ana Diaz",
        service_name="mammogram",
        transfer_number=AGENTS[0],
        transfer_numbers=list(AGENTS),
        flow="adult",
        status=SmsConversation.Status.REPLIED,
        step="identity",
        transcript="Agent: Hi",
    )
    fields.update(extra)
    return SmsConversation.objects.create(**fields)


def _twilio_call_response(sid="CA1"):
    response = MagicMock(status_code=201)
    response.json.return_value = {"sid": sid}
    return response


class IntentTests(TestCase):
    def test_negated_requests_are_not_intents(self):
        for text in (
            "No, I don't need an appointment",
            "I already have an appointment",
            "I already scheduled it",
            "no need to schedule",
        ):
            self.assertFalse(services._wants_schedule(text), text)
        for text in (
            "Please don't call me",
            "do not call me",
            "Is this about my human papillomavirus test?",
        ):
            self.assertFalse(services._wants_live_agent(text), text)

    def test_negation_stops_at_clause_boundary(self):
        self.assertTrue(services._wants_live_agent("No rush, call me tomorrow"))
        self.assertTrue(services._wants_live_agent("I don't know, call me"))
        self.assertTrue(services._wants_schedule("Not yet, please schedule me"))
        self.assertFalse(services._wants_schedule("I already have an appointment scheduled"))

    def test_plain_requests_are_still_intents(self):
        for text in (
            "I need to schedule",
            "Can you schedule it for me?",
            "I want to reschedule",
            "book an appointment please",
        ):
            self.assertTrue(services._wants_schedule(text), text)
        for text in ("Can you call me?", "I want a real person", "talk to a human"):
            self.assertTrue(services._wants_live_agent(text), text)


class TransferReplyTests(TestCase):
    def test_only_call_coming_replies_start_a_dial(self):
        self.assertTrue(agent_dial._is_transfer_reply(booked_message("mammogram")))
        self.assertTrue(agent_dial._is_transfer_reply(live_agent_message()))
        self.assertTrue(agent_dial._is_transfer_reply(services._minor_connect_message("question")))

    def test_definite_connect_wording_starts_a_dial(self):
        self.assertTrue(agent_dial._is_transfer_reply("I'll connect you with a team member now."))
        self.assertTrue(agent_dial._is_transfer_reply("I\u2019m connecting you now."))
        self.assertFalse(agent_dial._is_transfer_reply("I'll connect you if you like? Reply YES or NO."))

    def test_offers_and_office_number_do_not_start_a_dial(self):
        self.assertFalse(agent_dial._is_transfer_reply(services._minor_offer_connect_for_query()))
        self.assertFalse(agent_dial._is_transfer_reply(closed_reply("+15550000001")))
        self.assertFalse(
            agent_dial._is_transfer_reply(
                "Of course. I can connect you with a live team member instead. "
                "Would you like me to do that? Reply YES or NO."
            )
        )


@override_settings(**TWILIO)
class AgentFailoverTests(TestCase):
    def test_ringing_then_no_answer_dials_next_agent(self):
        convo = _conversation(dial_call_sid="CA1", transfer_status="dialing")
        agent_dial.maybe_failover_on_agent_status(convo, call_status="ringing", call_sid="CA1")
        convo.refresh_from_db()
        self.assertEqual(convo.transfer_status, "dialing")

        with patch.object(agent_dial.requests, "post", return_value=_twilio_call_response("CA2")) as post:
            moved = agent_dial.maybe_failover_on_agent_status(
                convo, call_status="no-answer", call_sid="CA1"
            )
        self.assertTrue(moved)
        self.assertEqual(post.call_args.kwargs["data"]["To"], AGENTS[1])
        convo.refresh_from_db()
        self.assertEqual(convo.transfer_number_index, 1)
        self.assertEqual(convo.dial_call_sid, "CA2")

    def test_answered_agent_is_not_failed_over(self):
        convo = _conversation(dial_call_sid="CA1", transfer_status="dialing")
        agent_dial.maybe_failover_on_agent_status(convo, call_status="in-progress", call_sid="CA1")
        convo.refresh_from_db()
        self.assertEqual(convo.transfer_status, "bridging")
        with patch.object(agent_dial.requests, "post") as post:
            self.assertFalse(
                agent_dial.maybe_failover_on_agent_status(
                    convo, call_status="completed", call_sid="CA1"
                )
            )
        post.assert_not_called()

    def test_read_timeout_waits_for_callback_instead_of_ringing_two_agents(self):
        convo = _conversation()
        with patch.object(
            agent_dial.requests, "post", side_effect=agent_dial.requests.ReadTimeout("slow")
        ) as post:
            self.assertTrue(agent_dial.dial_live_agent(convo))
        self.assertEqual(post.call_count, 1)
        convo.refresh_from_db()
        self.assertEqual((convo.transfer_number_index, convo.dial_call_sid), (0, ""))
        # Twilio's status callback reveals the CallSid; no-answer then fails over.
        agent_dial.maybe_failover_on_agent_status(convo, call_status="ringing", call_sid="CA9")
        convo.refresh_from_db()
        self.assertEqual(convo.dial_call_sid, "CA9")
        with patch.object(agent_dial.requests, "post", return_value=_twilio_call_response("CA10")):
            self.assertTrue(
                agent_dial.maybe_failover_on_agent_status(convo, call_status="no-answer", call_sid="CA9")
            )
        convo.refresh_from_db()
        self.assertEqual(convo.transfer_number_index, 1)

    def test_network_error_fails_over_instead_of_crashing(self):
        convo = _conversation()
        with patch.object(
            agent_dial.requests,
            "post",
            side_effect=[agent_dial.requests.ConnectionError("down"), _twilio_call_response("CA2")],
        ):
            self.assertTrue(agent_dial.dial_live_agent(convo))
        convo.refresh_from_db()
        self.assertEqual(convo.transfer_number_index, 1)


@override_settings(**TWILIO)
class InboundReplyTests(TestCase):
    def setUp(self):
        self.sent = []
        patches = [
            patch.object(services, "send_twilio_sms", side_effect=self._send),
            patch.object(services, "_openai_conversational_reply", return_value=""),
            patch.object(services, "_retell_request", return_value=(500, None, "down")),
            patch.object(agent_dial, "dial_live_agent", return_value=True),
            patch.object(services, "start_sms_conversation"),
        ]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.dial = self.mocks[3]
        self.start_new = self.mocks[4]

    def _send(self, to, body, from_number=""):
        self.sent.append((to, body, from_number))
        return "SM1", ""

    def _inbound(self, text, sid="", to=ADULT_LINE):
        return services.reply_to_inbound_sms(
            from_number=PATIENT, body=text, message_sid=sid, to_number=to
        )

    def test_consent_no_offers_without_dialing(self):
        _conversation(step="consent")
        reply = self._inbound("no")
        self.assertIn("Would you like me to do that?", reply)
        self.dial.assert_not_called()
        self.assertEqual(SmsConversation.objects.get().step, "consent_no")

        self._inbound("yes")
        self.dial.assert_called_once()
        self.assertEqual(SmsConversation.objects.get().status, SmsConversation.Status.LIVE_AGENT)

    def test_negated_appointment_does_not_transfer(self):
        _conversation(step="screening")
        reply = self._inbound("I already have an appointment")
        self.assertIn("When is your mammogram scheduled?", reply)
        self.dial.assert_not_called()
        self.assertEqual(SmsConversation.objects.get().step, "appointment")

    def test_menu_numbers_follow_the_scripted_flow(self):
        convo = _conversation(step="barriers")
        self.assertIn("Reply 1, 2, 3, or 4.", self._inbound("2"))
        convo.refresh_from_db()
        self.assertEqual(convo.step, "assist")

        reply = self._inbound("4")
        self.assertIn("That's completely okay", reply)
        convo.refresh_from_db()
        self.assertEqual((convo.step, convo.status), ("ended", SmsConversation.Status.COMPLETED))
        self.dial.assert_not_called()

    def test_appointment_date_ends_conversation(self):
        convo = _conversation(step="appointment")
        self.assertIn("We'll make a note of that", self._inbound("12/25/2099"))
        convo.refresh_from_db()
        self.assertEqual(convo.step, "ended")

    def test_fallback_offer_waits_for_yes(self):
        convo = _conversation(step="screening")
        reply = self._inbound("what is this about exactly")
        self.assertIn("Reply YES or NO", reply)
        self.dial.assert_not_called()
        convo.refresh_from_db()
        self.assertEqual(convo.step, "screening")
        self.assertNotEqual(convo.status, SmsConversation.Status.LIVE_AGENT)

        self._inbound("yes")
        self.dial.assert_called_once()

    def test_yes_after_model_recovers_answers_the_model_not_the_old_offer(self):
        convo = _conversation(step="screening")
        self._inbound("what is this about exactly")  # model down → connect offer
        self.mocks[1].return_value = "It's about your mammogram. Have you had it? Reply YES or NO."
        self._inbound("why do you ask")              # model back, asks something else
        self.mocks[1].return_value = ""
        self._inbound("yes")
        self.dial.assert_not_called()
        convo.refresh_from_db()
        self.assertEqual(convo.step, "ended")        # screening YES = already done

    def test_no_duplicate_greeting_for_chat_started_by_this_text(self):
        def start(**kwargs):
            SmsConversation.objects.create(
                chat_id="new", to_number=PATIENT, from_number=ADULT_LINE,
                step="identity", transcript="Agent: Hi, am I texting with you?",
            )
            return {"ok": True, "chat_id": "new"}

        self.start_new.side_effect = start
        self.assertEqual(self._inbound("hello there", to=""), "")
        self.assertEqual(self.sent, [])

    def test_clause_negation_keeps_real_requests(self):
        _conversation(step="followup")
        reply = self._inbound("No rush, call me tomorrow")
        self.assertIn("receive a call shortly", reply)
        self.dial.assert_called_once()

    def test_numbers_in_free_text_go_to_the_model(self):
        convo = _conversation(step="appointment")
        self._inbound("I have 2 kids so I can't go")
        convo.refresh_from_db()
        self.assertEqual(convo.step, "appointment")
        convo.step = "barriers"
        convo.save()
        self._inbound("2 weeks ago I went")
        convo.refresh_from_db()
        self.assertEqual(convo.step, "barriers")

    def test_fallback_at_identity_never_connects(self):
        convo = _conversation(step="identity", flow="minor", chat_id="minor-1", from_number=MINOR_LINE)
        reply = self._inbound("can someone call me about my kid", to=MINOR_LINE)
        self.assertIn("Are you the parent or guardian", reply)
        self.dial.assert_not_called()
        convo.refresh_from_db()
        self.assertEqual(convo.step, "identity")

    def test_message_after_close_gets_office_number_once(self):
        _conversation(step="ended", status=SmsConversation.Status.LIVE_AGENT)
        self.assertIn("please call us at", self._inbound("ok thanks", sid="SM10"))
        self.assertEqual(self._inbound("hello?", sid="SM11"), "")
        self.assertEqual(self._inbound("STOP", sid="SM12"), "")
        self.start_new.assert_not_called()
        self.dial.assert_not_called()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(SmsConversation.objects.count(), 1)

    def test_duplicate_message_sid_is_handled_once(self):
        _conversation(step="identity")
        self.assertTrue(self._inbound("yes", sid="SM20"))
        self.assertEqual(self._inbound("yes", sid="SM20"), "")
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(SmsConversation.objects.get().step, "consent")

    def test_reply_goes_to_conversation_on_texted_line(self):
        adult = _conversation(chat_id="adult", step="identity")
        minor = _conversation(
            chat_id="minor-2", flow="minor", from_number=MINOR_LINE, step="identity"
        )
        self._inbound("yes", to=MINOR_LINE)
        adult.refresh_from_db()
        minor.refresh_from_db()
        self.assertEqual(minor.step, "disclose")
        self.assertEqual(adult.step, "identity")
        self.assertEqual(self.sent[-1][2], MINOR_LINE)

    def test_unknown_number_still_starts_new_conversation(self):
        self.start_new.return_value = {"ok": False, "error": "x"}
        self._inbound("hi")
        self.start_new.assert_called_once()


@override_settings(**TWILIO)
class ConversationLifecycleTests(TestCase):
    def test_starting_on_one_line_keeps_other_line_open(self):
        adult = _conversation(chat_id="adult")
        minor = _conversation(chat_id="minor-3", flow="minor", from_number=MINOR_LINE)
        services._close_open_sms_for_number(PATIENT, MINOR_LINE)
        adult.refresh_from_db()
        minor.refresh_from_db()
        self.assertEqual(adult.status, SmsConversation.Status.REPLIED)
        self.assertEqual(minor.status, SmsConversation.Status.COMPLETED)

    def test_send_sms_network_error_returns_error(self):
        with patch.object(
            services.requests, "post", side_effect=services.requests.Timeout("slow")
        ):
            sid, err = services.send_twilio_sms(PATIENT, "hello")
        self.assertEqual(sid, "")
        self.assertTrue(err)
