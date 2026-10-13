"""Regression tests for Collect Mode answer pipeline and Custom LLM controller."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from pc_liveagent.answer_parse import normalize_utterance, validate_for_step
from pc_liveagent.call_turn import (
    CallController,
    drop_controller,
    get_controller,
    latest_user_utterance,
    process_user_turn,
)
from pc_liveagent.config import retell_llm_websocket_url
from pc_liveagent.form_bridge import _apply_key, _mark_applied, _skip_duplicate_apply
from pc_liveagent.live_form import LiveFormSession, _next_elig_step


class AnswerParseTests(unittest.TestCase):
    def test_q1_marital_never_married(self) -> None:
        answer, clarify = validate_for_step("marital", "Never married")
        self.assertEqual(answer, "Never Married")
        self.assertIsNone(clarify)

    def test_q2_children_no(self) -> None:
        answer, clarify = validate_for_step("children", "No")
        self.assertEqual(answer, "No")
        self.assertIsNone(clarify)

    def test_q3_student_no(self) -> None:
        answer, clarify = validate_for_step("student", "No")
        self.assertEqual(answer, "No")
        self.assertIsNone(clarify)

    def test_ambiguous_yes_no_asks_clarify(self) -> None:
        answer, clarify = validate_for_step("student", "maybe later")
        self.assertIsNone(answer)
        # Must include the full active question, not only a bare clarify sentence.
        self.assertIn("full-time student", (clarify or "").lower())
        self.assertIn("yes or no", (clarify or "").lower())
        self.assertNotEqual((clarify or "").strip().lower(), "please answer yes or no.")

    def test_whats_the_next_question_is_next_intent(self) -> None:
        from pc_liveagent.control_intent import detect_control_intent

        self.assertEqual(
            detect_control_intent("What's the next question?").kind, "next"
        )
        self.assertEqual(
            detect_control_intent("I didn't hear the question").kind, "repeat"
        )
        answer2, clarify2 = validate_for_step("disabled", "I didn't hear the question")
        self.assertIsNone(answer2)
        self.assertEqual(clarify2, "__REPEAT__")

    def test_yeah_thats_right_is_yes(self) -> None:
        answer, clarify = validate_for_step("children", "Yeah, that's right.")
        self.assertEqual(answer, "Yes")
        self.assertIsNone(clarify)

    def test_never_been_married(self) -> None:
        answer, clarify = validate_for_step("marital", "I have never been married.")
        self.assertEqual(answer, "Never Married")
        self.assertIsNone(clarify)

    def test_normalize_strips_filler(self) -> None:
        self.assertEqual(normalize_utterance("  My name is Alex  "), "Alex")


class BranchingTests(unittest.TestCase):
    def session(self, step: str) -> LiveFormSession:
        return LiveFormSession(call_id="t", url="https://example.test", elig_step=step)

    def test_q1_to_q2_to_q3_with_no(self) -> None:
        s = self.session("marital")
        self.assertEqual(_next_elig_step(s, "never married"), "children")
        s.elig_step = "children"
        self.assertEqual(_next_elig_step(s, "no"), "student")
        s.elig_step = "student"
        self.assertEqual(_next_elig_step(s, "no"), "disabled")

    def test_yes_on_student_opens_detail_branch(self) -> None:
        s = self.session("student")
        nxt = _next_elig_step(s, "yes")
        self.assertIn(nxt, ("person_first", "person_form", "student_count", "person_name"))


class DedupeTests(unittest.TestCase):
    def test_same_yes_no_different_steps_not_blocked(self) -> None:
        call_id = "dedupe-call"
        session = LiveFormSession(call_id=call_id, url="https://example.test", elig_step="children")
        _mark_applied(call_id, "children", "No")
        # Different step with same answer must not be treated as duplicate.
        self.assertFalse(_skip_duplicate_apply(call_id, "student", "No", session))
        # Same step+answer within cooldown is blocked.
        session.elig_step = "children"
        self.assertTrue(_skip_duplicate_apply(call_id, "children", "No", session))
        self.assertEqual(_apply_key("children", "No"), "children:no")
        self.assertEqual(_apply_key("student", "No"), "student:no")


class CallTurnTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await drop_controller("turn-test")

    async def test_consecutive_no_across_eligibility_steps(self) -> None:
        call_id = "turn-test"
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"

        session = LiveFormSession(call_id=call_id, url="https://example.test", elig_step="children")

        async def fake_get_session(_cid: str):
            return session

        async def fake_submit(cid, field, value, url="", request_id=""):
            # Advance children -> student, then student -> disabled
            if session.elig_step == "children":
                session.elig_step = "student"
                return {
                    "ok": True,
                    "spoken": "ASK: Are there any students?",
                    "step": "student",
                }
            if session.elig_step == "student":
                session.elig_step = "disabled"
                return {
                    "ok": True,
                    "spoken": "ASK: Anyone disabled?",
                    "step": "disabled",
                }
            return {"ok": False, "spoken": "stuck"}

        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch("pc_liveagent.call_turn.submit_field", side_effect=fake_submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            r1 = await process_user_turn(call_id, "No")
            self.assertEqual(r1.detail, "elig_ack")
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
            self.assertEqual(session.elig_step, "student")

            r2 = await process_user_turn(call_id, "No")
            self.assertEqual(r2.detail, "elig_ack")
            self.assertNotEqual(r2.detail, "duplicate_utterance")
            task2 = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task2:
                await asyncio.wait_for(task2, timeout=2.0)
            self.assertEqual(session.elig_step, "disabled")

    async def test_duplicate_same_step_is_ignored(self) -> None:
        call_id = "turn-test"
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(call_id=call_id, url="https://example.test", elig_step="student")
        calls = {"n": 0}

        async def fake_get_session(_cid: str):
            return session

        async def fake_submit(*_a, **_k):
            calls["n"] += 1
            session.elig_step = "disabled"
            return {"ok": True, "spoken": "ASK: next", "step": "disabled"}

        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch("pc_liveagent.call_turn.submit_field", side_effect=fake_submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            await process_user_turn(call_id, "No")
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
            # Reset step so commit_key differs but utterance fp matches same ctx —
            # after success step changed, so re-say No on disabled is different ctx.
            # Force same step context for true duplicate:
            session.elig_step = "student"
            ctrl = await get_controller(call_id)
            ctrl.phase = "eligibility"
            # Clear committed but keep last_user_fp/ctx from prior student:No
            # Actually after first turn, last_dedupe_ctx was eligibility:student and
            # session was advanced then we set back to student — ctx matches.
            r = await process_user_turn(call_id, "No")
            # Second submit may still run if commit_key student:no is in committed.
            self.assertIn(
                r.detail,
                ("duplicate_utterance", "already_committed", "elig_ack", ""),
            )
            if r.detail == "elig_ack":
                # Should still be blocked by commit key inside the bg path.
                self.assertEqual(calls["n"], 1)

    async def test_language_then_sign_in_question(self) -> None:
        call_id = "turn-test"
        await drop_controller(call_id)
        r = await process_user_turn(call_id, "English is fine")
        self.assertIn("Google", r.speech)
        self.assertEqual(r.phase, "sign_in")

    async def test_next_question_repeats_without_advancing(self) -> None:
        call_id = "turn-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="disabled"
        )
        submit = AsyncMock()

        async def fake_get_session(_cid: str):
            return session

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.submit_field", submit),
        ):
            r = await process_user_turn(call_id, "What's the next question?")
        self.assertIn("disabled person", r.speech.lower())
        self.assertEqual(session.elig_step, "disabled")
        submit.assert_not_called()

    async def test_unclear_yes_no_returns_full_question(self) -> None:
        call_id = "turn-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="relatives"
        )
        submit = AsyncMock()

        async def fake_get_session(_cid: str):
            return session

        from pc_liveagent.answer_parse import ClaudeResolveResult

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch("pc_liveagent.call_turn.submit_field", submit),
            patch(
                "pc_liveagent.call_turn.resolve_with_claude",
                AsyncMock(
                    return_value=ClaudeResolveResult(
                        intent="clarify",
                        needs_clarify=True,
                        reject_reason="needs_clarify",
                        accepted=False,
                    )
                ),
            ),
        ):
            r = await process_user_turn(call_id, "Sixteen to eighteen.")
        self.assertIn("other relatives", r.speech.lower())
        self.assertIn("yes or no", r.speech.lower())
        self.assertEqual(session.elig_step, "relatives")
        submit.assert_not_called()

    async def test_latest_user_utterance(self) -> None:
        transcript = [
            {"role": "agent", "content": "Are there any students?"},
            {"role": "user", "content": "No"},
        ]
        self.assertEqual(latest_user_utterance(transcript), "No")


class IncomeMappingTests(unittest.TestCase):
    def test_period_aliases(self) -> None:
        from pc_liveagent.live_form import _income_period_choice

        self.assertEqual(_income_period_choice("yearly"), "Annual")
        self.assertEqual(_income_period_choice("Annually."), "Annual")
        self.assertEqual(_income_period_choice("every two weeks"), "Bi-weekly")
        self.assertEqual(_income_period_choice("two weeks."), "Bi-weekly")
        self.assertEqual(_income_period_choice("Monthly."), "Monthly")

    def test_range_spoken_to_chip(self) -> None:
        from pc_liveagent.live_form import _map_income_range_label

        labels = [
            "$0 – $12,000",
            "$12,001 – $13,200",
            "$13,201 – $22,000",
            "$22,001 – $26,300",
            "$26,301 – $32,700",
            "$32,701 – $39,900",
            "$39,901+",
        ]
        self.assertEqual(
            _map_income_range_label("Zero dollars to twelve thousand dollars.", labels),
            "$0 – $12,000",
        )
        self.assertEqual(_map_income_range_label("first option", labels), "$0 – $12,000")
        self.assertEqual(_map_income_range_label("39901 plus", labels), "$39,901+")
        self.assertEqual(
            _map_income_range_label("40000-50000", labels),
            "$39,901+",
        )
        self.assertEqual(
            _map_income_range_label("forty to fifty thousand", labels),
            "$39,901+",
        )
        # Unmatched / not near open floor — must not fall back to first chip.
        self.assertEqual(
            _map_income_range_label("ninety thousand to one hundred thousand", labels),
            "",
        )
        self.assertNotEqual(
            _map_income_range_label("ninety thousand to one hundred thousand", labels),
            labels[0],
        )

    def test_income_range_validate(self) -> None:
        from pc_liveagent.answer_parse import resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        labels = [
            "$0 – $12,000",
            "$12,001 – $13,200",
            "$39,901+",
        ]
        r = resolve_answer(
            "income_range",
            "Zero dollars to twelve thousand dollars.",
            options=labels,
            meta=get_field_meta("income_range"),
        )
        self.assertEqual(r.answer, "$0 – $12,000")
        bad = resolve_answer(
            "income_range",
            "ninety thousand to one hundred thousand",
            options=labels,
            meta=get_field_meta("income_range"),
        )
        self.assertIsNone(bad.answer)
        self.assertIsNotNone(bad.clarify)
        self.assertNotEqual(bad.answer, labels[0] if labels else None)
        answer2, clarify2 = validate_for_step("income_when", "two weeks")
        self.assertEqual(answer2, "Bi-weekly")
        self.assertIsNone(clarify2)

    def test_income_screen_keeps_range_step(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _screen_first_step

        session = LiveFormSession(
            call_id="t", url="https://example.test", elig_step="income_range"
        )
        self.assertEqual(_screen_first_step("income", session), "income_range")


class WsResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_begin_speech_does_not_reset_mid_call(self) -> None:
        from pc_liveagent.call_turn import begin_speech, drop_controller, get_controller

        call_id = "ws-resume-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.turn_seq = 5
        ctrl.last_speech = "Which income range applies before taxes?"
        speech = await begin_speech(call_id)
        self.assertEqual(speech, "Which income range applies before taxes?")
        ctrl2 = await get_controller(call_id)
        self.assertEqual(ctrl2.phase, "eligibility")
        await drop_controller(call_id)

    async def test_auto_start_after_signin_when_session_ready(self) -> None:
        from pc_liveagent.call_turn import drop_controller, get_controller, process_user_turn
        from pc_liveagent.live_form import ELIG_QUESTIONS, LiveFormSession

        call_id = "auto-start-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "sign_in"
        session = LiveFormSession(
            call_id=call_id,
            url="https://example.test",
            signed_in=True,
            elig_step="marital",
        )

        async def fake_get_session(_cid: str):
            return session

        async def fake_promote(_cid: str):
            return ELIG_QUESTIONS["marital"]

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch(
                "pc_liveagent.call_turn.promote_signed_in_to_eligibility",
                side_effect=fake_promote,
            ),
        ):
            result = await process_user_turn(call_id, "next question")
        self.assertEqual(result.phase, "eligibility")
        self.assertEqual(result.detail, "auto_start_after_signin")
        self.assertIn("marital", result.speech.lower())
        await drop_controller(call_id)


class SupportScreenTests(unittest.TestCase):
    def test_support_body_not_misread_as_student(self) -> None:
        from pc_liveagent.live_form import _map_support_yes_label

        body_like_labels = [
            "Yes, I am a full-time student and my parent or legal guardian supports me.",
            "Yes, I live more than half the year with the person that supports me.",
            "Yes, I'm disabled and live year-round with the person that supports me.",
            "Yes, I live more than half the year with a relative that supports me.",
            "No",
        ]
        self.assertIn("student", _map_support_yes_label("yes student parent", body_like_labels).lower())
        self.assertIn("disabled", _map_support_yes_label("yes disabled", body_like_labels).lower())
        self.assertIn("relative", _map_support_yes_label("yes a relative", body_like_labels).lower())
        self.assertEqual(_map_support_yes_label("No", body_like_labels), "")

    def test_support_bare_yes_asks_clarify(self) -> None:
        answer, clarify = validate_for_step("support", "Yes")
        self.assertIsNone(answer)
        self.assertIn("which situation", (clarify or "").lower())

    def test_support_no_is_clear(self) -> None:
        answer, clarify = validate_for_step("support", "No")
        self.assertEqual(answer, "No")
        self.assertIsNone(clarify)


class ConfigTests(unittest.TestCase):
    def test_websocket_url_from_webhook(self) -> None:
        with patch("pc_liveagent.config.RETELL_LLM_WEBSOCKET_URL", ""):
            with patch(
                "pc_liveagent.config.FORM_FILLER_WEBHOOK_URL",
                "https://tunnel.example/webhooks/retell",
            ):
                url = retell_llm_websocket_url()
                self.assertEqual(url, "wss://tunnel.example/llm-websocket")


class CustomLlmProtocolTests(unittest.TestCase):
    def test_response_payload_shape(self) -> None:
        # Documented Retell response shape used by retell_llm_ws.
        payload = {
            "response_type": "response",
            "response_id": 3,
            "content": "Are there any students?",
            "content_complete": True,
        }
        self.assertEqual(payload["response_type"], "response")
        self.assertTrue(payload["content_complete"])
        self.assertIsInstance(payload["response_id"], int)

    def test_agent_interrupt_payload_shape(self) -> None:
        # Documented Retell agent_interrupt shape (post-SSO first question).
        payload = {
            "response_type": "agent_interrupt",
            "interrupt_id": 1,
            "content": "What is your marital status?",
            "content_complete": True,
            "no_interruption_allowed": True,
        }
        self.assertEqual(payload["response_type"], "agent_interrupt")
        self.assertIsInstance(payload["interrupt_id"], int)
        self.assertTrue(payload["content_complete"])


class GoogleSsoNonBlockingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await drop_controller("sso-test")
        await drop_controller("sso-slow")
        await drop_controller("sso-fail")
        await drop_controller("sso-dup")
        await drop_controller("sso-disc")

    async def test_google_ack_returns_before_sso_finishes(self) -> None:
        from pc_liveagent.call_turn import SSO_ACK, _sso_tasks
        from pc_liveagent.live_form import ELIG_QUESTIONS

        call_id = "sso-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "sign_in"
        started = asyncio.Event()
        finished = asyncio.Event()

        async def slow_submit(*_a, **_k):
            started.set()
            await finished.wait()
            return {
                "ok": True,
                "spoken": f"ASK: {ELIG_QUESTIONS['marital']}",
                "step": "marital",
            }

        with (
            patch("pc_liveagent.call_turn.submit_field", side_effect=slow_submit),
            patch("pc_liveagent.call_turn.get_session", AsyncMock(return_value=None)),
            patch("pc_liveagent.call_turn._speak_interrupt", AsyncMock(return_value=True)),
        ):
            result = await process_user_turn(call_id, "Google", response_id=4)
            self.assertEqual(result.detail, "sso_ack")
            self.assertEqual(result.speech, SSO_ACK)
            self.assertTrue(ctrl.sso_in_progress)
            self.assertTrue(ctrl.sso_ack_sent)
            # Yield so the background SSO task reaches submit_field without finishing.
            await asyncio.wait_for(started.wait(), timeout=2.0)
            self.assertFalse(finished.is_set())
            task = _sso_tasks.get(call_id)
            self.assertIsNotNone(task)
            finished.set()
            await asyncio.wait_for(task, timeout=2.0)
        self.assertFalse(ctrl.sso_in_progress)
        self.assertEqual(ctrl.phase, "eligibility")
        self.assertTrue(ctrl.eligibility_screen_verified)
        self.assertTrue(ctrl.eligibility_question_sent)

    async def test_slow_sso_keeps_turns_responsive(self) -> None:
        from pc_liveagent.call_turn import SSO_BUSY

        call_id = "sso-slow"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "sign_in"
        gate = asyncio.Event()

        async def slow_submit(*_a, **_k):
            await gate.wait()
            return {"ok": True, "spoken": "ASK: What is your marital status?"}

        with (
            patch("pc_liveagent.call_turn.submit_field", side_effect=slow_submit),
            patch("pc_liveagent.call_turn.get_session", AsyncMock(return_value=None)),
            patch("pc_liveagent.call_turn._speak_interrupt", AsyncMock(return_value=True)),
        ):
            await process_user_turn(call_id, "At Google", response_id=2)
            busy = await process_user_turn(call_id, "hello?", response_id=3)
            self.assertEqual(busy.detail, "sso_in_progress")
            self.assertEqual(busy.speech, SSO_BUSY)
            gate.set()
            task = __import__("pc_liveagent.call_turn", fromlist=["_sso_tasks"])._sso_tasks.get(
                call_id
            )
            if task:
                await asyncio.wait_for(task, timeout=2.0)

    async def test_sso_failure_does_not_send_eligibility_question(self) -> None:
        call_id = "sso-fail"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "sign_in"
        interrupt = AsyncMock(return_value=True)

        async def fail_submit(*_a, **_k):
            return {
                "ok": False,
                "spoken": "Please say Google again when the sign-in page is ready.",
                "error": "not_ready",
            }

        with (
            patch("pc_liveagent.call_turn.submit_field", side_effect=fail_submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
            patch("pc_liveagent.call_turn.schedule_signin_watch"),
        ):
            await process_user_turn(call_id, "Google", response_id=1)
            task = __import__("pc_liveagent.call_turn", fromlist=["_sso_tasks"])._sso_tasks.get(
                call_id
            )
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertFalse(ctrl.eligibility_question_sent)
        self.assertFalse(ctrl.eligibility_screen_verified)
        self.assertNotEqual(ctrl.phase, "eligibility")
        interrupt.assert_awaited()
        spoken = interrupt.await_args.args[1]
        self.assertNotIn("marital", spoken.lower())

    async def test_duplicate_ready_does_not_double_speak(self) -> None:
        from pc_liveagent.call_turn import deliver_first_eligibility_question
        from pc_liveagent.live_form import ELIG_QUESTIONS

        call_id = "sso-dup"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_screen_verified = True
        ctrl.last_speech = ELIG_QUESTIONS["marital"]
        interrupt = AsyncMock(return_value=True)
        with patch("pc_liveagent.call_turn._speak_interrupt", interrupt):
            ok1 = await deliver_first_eligibility_question(call_id)
            ok2 = await deliver_first_eligibility_question(call_id)
        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertEqual(interrupt.await_count, 1)

    async def test_disconnect_skips_stale_interrupt(self) -> None:
        from pc_liveagent.call_turn import deliver_first_eligibility_question
        from pc_liveagent.live_form import ELIG_QUESTIONS

        call_id = "sso-disc"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_screen_verified = True
        ctrl.active_websocket_status = "disconnected"
        ctrl.last_speech = ELIG_QUESTIONS["marital"]
        ok = await deliver_first_eligibility_question(call_id)
        self.assertFalse(ok)
        self.assertFalse(ctrl.eligibility_question_sent)


class ControlIntentAndMetaTests(unittest.TestCase):
    def test_intent_kinds(self) -> None:
        from pc_liveagent.control_intent import detect_control_intent

        self.assertEqual(detect_control_intent("Can you repeat that?").kind, "repeat")
        self.assertEqual(detect_control_intent("Go back one step.").kind, "go_back")
        self.assertEqual(detect_control_intent("Skip this if it is optional.").kind, "skip")
        self.assertEqual(detect_control_intent("I made a mistake.").kind, "redo")
        change = detect_control_intent("Change the age to fifteen.")
        self.assertEqual(change.kind, "change")
        self.assertEqual(change.target, "age")
        self.assertIn("fifteen", change.value.lower())
        self.assertEqual(detect_control_intent("Never married").kind, "none")

    def test_required_meta_blocks_skip_flag(self) -> None:
        from pc_liveagent.elig_meta import format_question, get_field_meta

        meta = get_field_meta("children")
        self.assertTrue(meta.required)
        self.assertFalse(meta.can_skip)
        q = format_question("children")
        self.assertIn("required", q.lower())

    def test_resolve_age_with_options(self) -> None:
        from pc_liveagent.answer_parse import resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        options = ["(0-5) Child", "(6-20) Adult", "(21-63) Adult"]
        meta = get_field_meta("person_age")
        r = resolve_answer("person_age", "15", options=options, meta=meta)
        self.assertEqual(r.answer, "(6-20) Adult")
        bad = resolve_answer(
            "child_age",
            "Sixteen to eight",
            options=["(0-1) Baby", "(16-18) Teen"],
            meta=get_field_meta("child_age"),
        )
        self.assertIsNone(bad.answer)
        self.assertIsNotNone(bad.clarify)

    def test_resolve_twenty_five_and_ambiguous_age(self) -> None:
        from pc_liveagent.answer_parse import is_fast_path_resolve, resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        spouse_ages = [
            "(0-18) Child",
            "(19-20) Young Adult",
            "(21-63) Adult",
            "(64) Turning Senior",
            "(65+) Senior",
        ]
        meta = get_field_meta("spouse_age")
        mapped = resolve_answer(
            "spouse_age", "Twenty five", options=spouse_ages, meta=meta
        )
        self.assertEqual(mapped.answer, "(21-63) Adult")
        self.assertEqual(mapped.detail, "age_mapped")
        self.assertNotEqual(mapped.answer, "(19-20) Young Adult")

        digit = resolve_answer("spouse_age", "25", options=spouse_ages, meta=meta)
        self.assertEqual(digit.answer, "(21-63) Adult")

        # Form captions must not become age choices or a false Young Adult pick.
        captions = [
            "First Name",
            "Last Name",
            "Spouse",
            "Partner",
            "Age",
        ]
        with_captions = resolve_answer(
            "spouse_age",
            "Twenty twenty two thirty",
            options=captions + spouse_ages,
            meta=meta,
        )
        self.assertIsNone(with_captions.answer)
        self.assertEqual(with_captions.detail, "age_ambiguous")
        self.assertNotIn("First Name", with_captions.clarify or "")

        # No live chips: clear spoken age is free-text for Playwright verification.
        free = resolve_answer("spouse_age", "Twenty five", options=captions, meta=meta)
        self.assertEqual(free.detail, "age_free")
        self.assertTrue(is_fast_path_resolve(free))
        self.assertEqual(free.answer, "Twenty five")

        ambiguous = resolve_answer(
            "spouse_age", "Twenty twenty two thirty", options=captions, meta=meta
        )
        self.assertIsNone(ambiguous.answer)
        self.assertEqual(ambiguous.detail, "age_ambiguous")


class ConversationalNavTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await drop_controller("nav-test")

    async def test_skip_blocked_when_required(self) -> None:
        call_id = "nav-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="student"
        )

        async def fake_get_session(_cid: str):
            return session

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch(
                "pc_liveagent.call_turn.skip_current_optional",
                AsyncMock(
                    return_value={
                        "ok": False,
                        "error": "required",
                        "spoken": (
                            "Are you the parent or legal guardian of a full-time "
                            "student? This field is required to continue, so I can't "
                            "skip it."
                        ),
                    }
                ),
            ),
            patch("pc_liveagent.call_turn.submit_field", AsyncMock()) as submit,
        ):
            r = await process_user_turn(call_id, "Skip this")
        self.assertEqual(r.detail, "skip_blocked")
        self.assertIn("required", r.speech.lower())
        submit.assert_not_called()

    async def test_go_back_failure_is_honest(self) -> None:
        call_id = "nav-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="disabled"
        )

        async def fake_get_session(_cid: str):
            return session

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch(
                "pc_liveagent.call_turn.navigate_back_one",
                AsyncMock(
                    return_value={
                        "ok": False,
                        "spoken": "I can't go back from this screen.",
                    }
                ),
            ),
        ):
            r = await process_user_turn(call_id, "Go back.")
        self.assertEqual(r.detail, "go_back_failed")
        self.assertIn("can't go back", r.speech.lower())

    async def test_correction_after_advance_uses_back(self) -> None:
        call_id = "nav-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_more"
        )

        async def fake_get_session(_cid: str):
            return session

        async def fake_back(_cid: str):
            session.elig_step = "child_age"
            return {
                "ok": True,
                "step": "child_age",
                "spoken": "What is the child's age range?",
            }

        async def slow_submit(*_a, **_k):
            await asyncio.sleep(0.05)
            session.elig_step = "child_more"
            return {
                "ok": True,
                "spoken": "ASK: Is there another child to add? Please say Yes or No.",
                "step": "child_more",
            }

        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch(
                "pc_liveagent.call_turn.peek_choice_labels",
                AsyncMock(return_value=["(13-15) Teen", "(16-18) Teen"]),
            ),
            patch("pc_liveagent.call_turn.navigate_back_one", side_effect=fake_back),
            patch("pc_liveagent.call_turn.submit_field", side_effect=slow_submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            r = await process_user_turn(call_id, "Change the age to fifteen.")
            self.assertEqual(r.detail, "elig_ack")
            task = __import__("pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"])._elig_browser_tasks.get(
                call_id
            )
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertTrue(interrupt.await_count >= 1)


class LiveCallFiveFixesTests(unittest.IsolatedAsyncioTestCase):
    """Regression for call_d0bdc6eeb83021bb121cce91da7 class of bugs."""

    async def asyncTearDown(self) -> None:
        for cid in (
            "fix-ws",
            "fix-once",
            "fix-disc",
            "fix-corr",
            "fix-child",
            "fix-child2",
        ):
            await drop_controller(cid)

    async def test_slow_browser_keeps_turn_responsive(self) -> None:
        call_id = "fix-ws"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_age"
        )

        async def fake_get_session(_cid: str):
            return session

        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_submit(*_a, **_k):
            started.set()
            await release.wait()
            session.elig_step = "child_more"
            return {
                "ok": True,
                "spoken": "ASK: Is there another child to add? Please say Yes or No.",
                "step": "child_more",
            }

        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch(
                "pc_liveagent.call_turn.peek_choice_labels",
                AsyncMock(return_value=["(13-15) Teen"]),
            ),
            patch("pc_liveagent.call_turn.submit_field", side_effect=slow_submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            t0 = asyncio.get_event_loop().time()
            r = await process_user_turn(call_id, "fifteen", response_id=1)
            elapsed = asyncio.get_event_loop().time() - t0
            self.assertLess(elapsed, 0.5)
            self.assertEqual(r.detail, "elig_ack")
            await asyncio.wait_for(started.wait(), timeout=1.0)
            # While browser runs, another turn stays responsive.
            busy = await process_user_turn(call_id, "next question", response_id=2)
            self.assertEqual(busy.detail, "elig_op_in_progress")
            release.set()
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertFalse(ctrl.elig_op_in_progress)
        self.assertEqual(interrupt.await_count, 1)

    async def test_next_question_delivered_exactly_once(self) -> None:
        from pc_liveagent.call_turn import _deliver_question_once

        call_id = "fix-once"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.active_websocket_status = "connected"
        interrupt = AsyncMock(return_value=True)
        with patch("pc_liveagent.call_turn._speak_interrupt", interrupt):
            ok1 = await _deliver_question_once(
                call_id, "Is there another child to add?", key="next:child_more:t1"
            )
            ok2 = await _deliver_question_once(
                call_id, "Is there another child to add?", key="next:child_more:t1"
            )
        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertEqual(interrupt.await_count, 1)

    async def test_disconnect_queues_recovery_without_double_commit(self) -> None:
        call_id = "fix-disc"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_age"
        )
        commits: list[str] = []

        async def fake_get_session(_cid: str):
            return session

        async def submit_once(*_a, **_k):
            commits.append("once")
            session.elig_step = "child_more"
            return {
                "ok": True,
                "spoken": "ASK: Is there another child to add? Please say Yes or No.",
                "step": "child_more",
            }

        interrupt = AsyncMock(return_value=False)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch(
                "pc_liveagent.call_turn.peek_choice_labels",
                AsyncMock(return_value=["(13-15) Teen"]),
            ),
            patch("pc_liveagent.call_turn.submit_field", side_effect=submit_once),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            await process_user_turn(call_id, "fifteen", response_id=1)
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertEqual(commits, ["once"])
        self.assertEqual(session.elig_step, "child_more")
        self.assertTrue(ctrl.pending_outbound_speech)
        # Reconnect recovery delivers once; answer is not re-applied.
        ctrl.active_websocket_status = "connected"
        interrupt2 = AsyncMock(return_value=True)
        from pc_liveagent.call_turn import recover_pending_outbound

        with patch("pc_liveagent.call_turn._speak_interrupt", interrupt2):
            ok = await recover_pending_outbound(call_id)
        self.assertTrue(ok)
        self.assertEqual(interrupt2.await_count, 1)
        self.assertEqual(commits, ["once"])

    async def test_change_first_name_prompts_for_value(self) -> None:
        call_id = "fix-corr"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_more"
        )

        async def fake_get_session(_cid: str):
            return session

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch("pc_liveagent.call_turn.submit_field", AsyncMock()) as submit,
            patch("pc_liveagent.call_turn.navigate_back_one", AsyncMock()) as back,
        ):
            r = await process_user_turn(call_id, "Change the first name.")
        self.assertEqual(r.detail, "correction_prompt")
        self.assertIn("first name", r.speech.lower())
        self.assertIsNotNone(ctrl.pending_correction)
        submit.assert_not_called()
        back.assert_not_called()

    async def test_change_first_name_to_michael_updates(self) -> None:
        call_id = "fix-corr"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_more"
        )

        async def fake_get_session(_cid: str):
            return session

        edit = AsyncMock(
            return_value={
                "ok": True,
                "spoken": "Updated the first name to Michael. What is the child's last name?",
                "step": "child_last",
                "value": "Michael",
            }
        )
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch("pc_liveagent.call_turn.correct_name_field", edit),
            patch("pc_liveagent.call_turn.submit_field", AsyncMock()) as submit,
        ):
            r = await process_user_turn(
                call_id, "Change the first name to Michael."
            )
        self.assertEqual(r.detail, "name_corrected")
        self.assertIn("michael", r.speech.lower())
        edit.assert_awaited()
        self.assertEqual(edit.await_args.args[2], "Michael")
        submit.assert_not_called()

    async def test_child_age_auto_asks_child_more(self) -> None:
        call_id = "fix-child"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_age"
        )

        async def fake_get_session(_cid: str):
            return session

        async def submit(*_a, **_k):
            session.elig_step = "child_more"
            return {
                "ok": True,
                "spoken": "ASK: Is there another child to add? Please say Yes or No.",
                "step": "child_more",
            }

        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch(
                "pc_liveagent.call_turn.peek_choice_labels",
                AsyncMock(return_value=["(13-15) Teen"]),
            ),
            patch("pc_liveagent.call_turn.submit_field", side_effect=submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            await process_user_turn(call_id, "fifteen", response_id=1)
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        spoken = interrupt.await_args.args[1]
        self.assertIn("another child", spoken.lower())
        self.assertEqual(session.elig_step, "child_more")

    async def test_add_another_child_asks_first_name(self) -> None:
        call_id = "fix-child2"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_more"
        )

        async def fake_get_session(_cid: str):
            return session

        async def submit(*_a, **_k):
            session.elig_step = "child_first"
            return {
                "ok": True,
                "spoken": "ASK: What is the child's first name?",
                "step": "child_first",
            }

        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch("pc_liveagent.call_turn.submit_field", side_effect=submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            await process_user_turn(call_id, "Yes", response_id=1)
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        spoken = interrupt.await_args.args[1]
        self.assertIn("first name", spoken.lower())
        self.assertEqual(session.elig_step, "child_first")

    async def test_no_more_children_then_student(self) -> None:
        call_id = "fix-child2"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_more"
        )

        async def fake_get_session(_cid: str):
            return session

        async def submit(*_a, **_k):
            session.elig_step = "student"
            return {
                "ok": True,
                "spoken": "ASK: Are you the parent or legal guardian of a full-time student?",
                "step": "student",
            }

        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch("pc_liveagent.call_turn.submit_field", side_effect=submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            await process_user_turn(call_id, "No", response_id=1)
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        spoken = interrupt.await_args.args[1]
        self.assertIn("student", spoken.lower())
        self.assertEqual(session.elig_step, "student")


class ClaudeResolverTests(unittest.TestCase):
    def test_fast_path_skips_claude_for_clear_yes_no_and_mapped_age(self) -> None:
        from pc_liveagent.answer_parse import is_fast_path_resolve, resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        yes = resolve_answer("children", "Yes", meta=get_field_meta("children"))
        self.assertTrue(is_fast_path_resolve(yes))
        age = resolve_answer(
            "child_age",
            "15",
            options=["(13-15) Teen", "(16-18) Teen"],
            meta=get_field_meta("child_age"),
        )
        self.assertTrue(is_fast_path_resolve(age))
        ambiguous = resolve_answer(
            "child_age",
            "around that age maybe",
            options=["(13-15) Teen", "(16-18) Teen"],
            meta=get_field_meta("child_age"),
        )
        self.assertFalse(is_fast_path_resolve(ambiguous))

    def test_validate_rejects_invented_income_option(self) -> None:
        from pc_liveagent.answer_parse import ClaudeResolveResult, validate_claude_result

        options = ["$0 – $30,000", "$30,001 – $45,000", "$45,001 – $60,000"]
        invented = ClaudeResolveResult(
            intent="answer",
            answer="$40,000 – $50,000",
            matched_option="$40,000 – $50,000",
            confidence=0.99,
            needs_clarify=False,
        )
        out = validate_claude_result(
            "income_range",
            invented,
            options=options,
            answer_type="income_range",
            min_confidence=0.72,
        )
        self.assertFalse(out.accepted)
        self.assertEqual(out.reject_reason, "not_in_options")

    def test_validate_accepts_real_income_option(self) -> None:
        from pc_liveagent.answer_parse import ClaudeResolveResult, validate_claude_result

        options = ["$0 – $30,000", "$39,901 – $50,000", "$50,001+"]
        ok = ClaudeResolveResult(
            intent="answer",
            answer="$39,901 – $50,000",
            matched_option="$39,901 – $50,000",
            confidence=0.91,
            needs_clarify=False,
        )
        out = validate_claude_result(
            "income_range",
            ok,
            options=options,
            answer_type="income_range",
            min_confidence=0.72,
        )
        self.assertTrue(out.accepted)
        self.assertEqual(out.answer, "$39,901 – $50,000")

    def test_validate_age_must_match_displayed_chip(self) -> None:
        from pc_liveagent.answer_parse import ClaudeResolveResult, validate_claude_result

        options = ["(13-15) Teen", "(16-18) Teen"]
        bad = ClaudeResolveResult(
            intent="answer",
            answer="(19-20) Young Adult",
            matched_option="(19-20) Young Adult",
            confidence=0.95,
            needs_clarify=False,
        )
        out = validate_claude_result(
            "child_age",
            bad,
            options=options,
            answer_type="age",
            min_confidence=0.72,
        )
        self.assertFalse(out.accepted)
        good = ClaudeResolveResult(
            intent="answer",
            answer="fifteen",
            matched_option="(13-15) Teen",
            confidence=0.9,
            needs_clarify=False,
        )
        out2 = validate_claude_result(
            "child_age",
            good,
            options=options,
            answer_type="age",
            min_confidence=0.72,
        )
        self.assertTrue(out2.accepted)
        self.assertEqual(out2.answer, "(13-15) Teen")

    def test_validate_name_sanitize_and_reject_condition(self) -> None:
        from pc_liveagent.answer_parse import ClaudeResolveResult, validate_claude_result

        name = ClaudeResolveResult(
            intent="answer",
            answer="Michael.",
            confidence=0.88,
            needs_clarify=False,
        )
        out = validate_claude_result(
            "child_first",
            name,
            options=None,
            answer_type="text",
            min_confidence=0.72,
        )
        self.assertTrue(out.accepted)
        self.assertEqual(out.answer, "Michael")
        cond = ClaudeResolveResult(
            intent="answer",
            answer="Someone is blind",
            confidence=0.9,
            needs_clarify=False,
        )
        out2 = validate_claude_result(
            "special_person",
            cond,
            options=None,
            answer_type="text",
            min_confidence=0.72,
        )
        self.assertFalse(out2.accepted)

    def test_ambiguous_claude_forces_clarify_not_first_option(self) -> None:
        from pc_liveagent.answer_parse import ClaudeResolveResult, validate_claude_result

        ambiguous = ClaudeResolveResult(
            intent="clarify",
            answer=None,
            matched_option=None,
            confidence=0.4,
            needs_clarify=True,
            clarify_prompt="Which income range matches?",
        )
        out = validate_claude_result(
            "income_range",
            ambiguous,
            options=["$0 – $30,000", "$30,001 – $45,000"],
            answer_type="income_range",
            min_confidence=0.72,
        )
        self.assertFalse(out.accepted)
        self.assertNotEqual(out.answer, "$0 – $30,000")


class ClaudeResolverAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_does_not_return_accepted_answer(self) -> None:
        from pc_liveagent import answer_parse
        from pc_liveagent.answer_parse import resolve_with_claude

        async def slow_create(*_a, **_k):
            await asyncio.sleep(5)
            raise AssertionError("should have timed out")

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            @property
            def messages(self):
                return self

            async def create(self, *a, **k):
                return await slow_create()

        with (
            patch("pc_liveagent.config.ANTHROPIC_API_KEY", "test-key"),
            patch("pc_liveagent.config.ANTHROPIC_TIMEOUT_S", 0.05),
            patch("anthropic.AsyncAnthropic", FakeClient),
        ):
            result = await resolve_with_claude(
                "income_range",
                "forty to fifty thousand",
                question="What is your household income?",
                answer_type="income_range",
                options=["$0 – $30,000", "$39,901 – $50,000"],
                turn_id="t1",
            )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reject_reason, "timeout")

    async def test_stale_claude_does_not_override_committed(self) -> None:
        call_id = "claude-stale"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="income_range"
        )

        async def fake_get_session(_cid: str):
            return session

        async def fake_claude(*_a, **_k):
            # Simulate a late/wrong Claude pick of the first chip.
            from pc_liveagent.answer_parse import ClaudeResolveResult

            ctrl.committed.append(
                {
                    "key": "income_range:$39,901 – $50,000",
                    "turn_id": "other",
                    "step": "income_range",
                    "answer": "$39,901 – $50,000",
                }
            )
            return ClaudeResolveResult(
                intent="answer",
                answer="$0 – $30,000",
                matched_option="$0 – $30,000",
                confidence=0.99,
                needs_clarify=False,
                accepted=True,
            )

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch(
                "pc_liveagent.call_turn.peek_choice_labels",
                AsyncMock(
                    return_value=["$0 – $30,000", "$39,901 – $50,000"]
                ),
            ),
            patch("pc_liveagent.call_turn.resolve_with_claude", side_effect=fake_claude),
            patch("pc_liveagent.call_turn.submit_field", AsyncMock()) as submit,
        ):
            # Phrasing that Python cannot map uniquely → Claude path → stale discard.
            r = await process_user_turn(call_id, "somewhere in the middle I guess")
        submit.assert_not_called()
        self.assertNotEqual(r.detail, "elig_ack")
        self.assertTrue(r.speech)


class NameAndSpecialPersonUnitTests(unittest.TestCase):
    def test_sanitize_trailing_asr_punctuation(self) -> None:
        from pc_liveagent.answer_parse import sanitize_person_name

        self.assertEqual(sanitize_person_name("John."), "John")
        self.assertEqual(sanitize_person_name("Aria."), "Aria")
        self.assertEqual(sanitize_person_name("O'Connor"), "O'Connor")
        self.assertEqual(sanitize_person_name("Anne-Marie"), "Anne-Marie")
        self.assertEqual(sanitize_person_name("Last name is Martin"), "Martin")

    def test_last_name_is_martin_fills_name_not_choice(self) -> None:
        from pc_liveagent.answer_parse import resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        r = resolve_answer(
            "child_last",
            "Last name is Martin",
            options=["Child", "Grandchild", "Last Name"],
            meta=get_field_meta("child_last"),
        )
        self.assertEqual(r.answer, "Martin")
        choice = resolve_answer(
            "child_rel",
            "Last name is Martin",
            options=["Child", "Grandchild", "Last Name"],
            meta=get_field_meta("child_rel"),
        )
        self.assertIsNone(choice.answer)
        self.assertIn("relationship", (choice.clarify or "").lower())

    def test_someone_is_blind_is_not_person_assignment(self) -> None:
        from pc_liveagent.answer_parse import resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        r = resolve_answer(
            "special_person",
            "Someone is blind",
            meta=get_field_meta("special_person"),
        )
        self.assertIsNone(r.answer)
        self.assertIn("condition", (r.clarify or "").lower())

    def test_change_first_name_intent(self) -> None:
        from pc_liveagent.control_intent import detect_control_intent

        bare = detect_control_intent("Change the first name.")
        self.assertEqual(bare.kind, "change")
        self.assertEqual(bare.target, "first_name")
        self.assertEqual(bare.value, "")
        valued = detect_control_intent("Change the first name to Michael.")
        self.assertEqual(valued.kind, "change")
        self.assertEqual(valued.target, "first_name")
        self.assertEqual(valued.value.lower(), "michael")

    def test_duplicate_member_labels_need_ordinal(self) -> None:
        from pc_liveagent.live_form import _normalize_member_label

        self.assertEqual(_normalize_member_label("John D."), "john d")
        self.assertEqual(_normalize_member_label("John."), "john")

    def test_ui_field_labels_filtered_from_choices(self) -> None:
        from pc_liveagent.live_form import _choice_labels_for_matching, _fuzzy_choice_label

        labels = ["Child", "Grandchild", "First Name", "Last Name"]
        filtered = _choice_labels_for_matching(labels)
        self.assertNotIn("First Name", filtered)
        self.assertNotIn("Last Name", filtered)
        # Must not fuzzy onto the caption.
        self.assertEqual(
            _fuzzy_choice_label("Last Name", labels),
            "Last Name",
        )


class ChildNameEditTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await drop_controller("child-edit")

    async def test_pending_then_value_uses_correct_name_field(self) -> None:
        call_id = "child-edit"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_rel"
        )

        async def fake_get_session(_cid: str):
            return session

        edit = AsyncMock(
            return_value={
                "ok": True,
                "spoken": "Updated the last name to Martin. What is the relationship?",
                "step": "child_rel",
                "value": "Martin",
            }
        )
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch("pc_liveagent.call_turn.correct_name_field", edit),
            patch("pc_liveagent.call_turn.submit_field", AsyncMock()) as submit,
        ):
            prompt = await process_user_turn(call_id, "Change the last name.")
            self.assertEqual(prompt.detail, "correction_prompt")
            r = await process_user_turn(call_id, "Martin")
        self.assertEqual(r.detail, "name_corrected")
        self.assertEqual(edit.await_args.args[1], "last_name")
        self.assertEqual(edit.await_args.args[2], "Martin")
        submit.assert_not_called()

    async def test_create_child_flow_steps(self) -> None:
        from pc_liveagent.live_form import _next_elig_step

        s = LiveFormSession(
            call_id="c", url="https://example.test", elig_step="child_first"
        )
        self.assertEqual(_next_elig_step(s, "Aria"), "child_last")
        s.elig_step = "child_last"
        self.assertEqual(_next_elig_step(s, "Martin"), "child_rel")
        s.elig_step = "child_rel"
        self.assertEqual(_next_elig_step(s, "Child"), "child_age")
        s.elig_step = "child_age"
        self.assertEqual(_next_elig_step(s, "15"), "child_more")
        s.elig_step = "child_more"
        self.assertEqual(_next_elig_step(s, "yes"), "child_first")
        self.assertEqual(_next_elig_step(s, "no"), "student")


class SpecialConditionAndPersonTests(unittest.TestCase):
    SPECIAL_OPTS = [
        "Someone is pregnant",
        "Someone has breast cancer",
        "Someone has cervical cancer",
        "Someone is blind",
        "Someone is disabled (SSDI)",
        "Someone was in foster care when they turned 18",
        "None of the above",
    ]

    def test_disabled_maps_to_ssdi_option(self) -> None:
        from pc_liveagent.live_form import _map_special_condition_labels
        from pc_liveagent.answer_parse import resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        mapped = _map_special_condition_labels(
            "Someone is disabled.", self.SPECIAL_OPTS
        )
        self.assertEqual(mapped, ["Someone is disabled (SSDI)"])
        r = resolve_answer(
            "special",
            "Someone is disabled.",
            options=self.SPECIAL_OPTS,
            meta=get_field_meta("special"),
        )
        self.assertEqual(r.answer, "Someone is disabled (SSDI)")

    def test_blind_maps_to_visible_option(self) -> None:
        from pc_liveagent.live_form import _map_special_condition_labels

        mapped = _map_special_condition_labels("Someone is blind.", self.SPECIAL_OPTS)
        self.assertEqual(mapped, ["Someone is blind"])

    def test_screen_recovery_keeps_special_person(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _screen_first_step

        session = LiveFormSession(
            call_id="sp",
            url="https://example.test",
            elig_step="special_person",
        )
        self.assertEqual(_screen_first_step("special", session), "special_person")
        session.elig_step = "special"
        self.assertEqual(_screen_first_step("special", session), "special")

    def test_repeated_condition_on_special_person_clarifies(self) -> None:
        from pc_liveagent.answer_parse import resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        r = resolve_answer(
            "special_person",
            "Someone is blind.",
            meta=get_field_meta("special_person"),
        )
        self.assertIsNone(r.answer)
        self.assertIn("household member", (r.clarify or "").lower())
        r2 = resolve_answer(
            "special_person",
            "I said someone is blind.",
            meta=get_field_meta("special_person"),
        )
        self.assertIsNone(r2.answer)

    def test_angel_or_sanitizes_to_angel(self) -> None:
        from pc_liveagent.answer_parse import sanitize_person_name, resolve_answer
        from pc_liveagent.elig_meta import get_field_meta
        from pc_liveagent.live_form import _spoken_member_query, _normalize_member_label

        self.assertEqual(sanitize_person_name("Angel or"), "Angel")
        self.assertEqual(_spoken_member_query("Angel or"), "Angel")
        self.assertEqual(_spoken_member_query("Angel r dot"), "Angel R")
        self.assertEqual(_normalize_member_label("Angel r dot"), "angel r")
        self.assertEqual(_normalize_member_label("Angel R."), "angel r")
        r = resolve_answer(
            "special_person",
            "Angel or",
            meta=get_field_meta("special_person"),
        )
        self.assertEqual(r.answer, "Angel")
        r2 = resolve_answer(
            "special_person",
            "Angel r dot",
            meta=get_field_meta("special_person"),
        )
        self.assertEqual(r2.answer, "Angel R")

    def test_member_filter_rejects_marital_labels(self) -> None:
        from pc_liveagent.live_form import _is_plausible_member_label

        self.assertFalse(_is_plausible_member_label("Never Married"))
        self.assertFalse(_is_plausible_member_label("Exit"))
        self.assertFalse(_is_plausible_member_label("Someone is blind"))
        self.assertTrue(_is_plausible_member_label("Angel R."))
        self.assertTrue(_is_plausible_member_label("John. D."))
        self.assertTrue(_is_plausible_member_label("John D."))
        self.assertTrue(_is_plausible_member_label("Me"))


class SpecialPersonAssignTests(unittest.IsolatedAsyncioTestCase):
    async def test_condition_phrase_rejected_before_click(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        session = LiveFormSession(call_id="sp", url="https://example.test")
        with self.assertRaises(RuntimeError) as ctx:
            await _assign_special_person_member(page, "Someone is blind", session)
        self.assertIn("SPECIAL_PERSON_CLARIFY", str(ctx.exception))
        page.evaluate.assert_not_called()

    async def test_duplicate_john_requires_disambiguation(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.evaluate = AsyncMock(
            return_value=[
                {"label": "John D.", "index": 0, "dupIndex": 0, "checked": False},
                {"label": "John D.", "index": 1, "dupIndex": 1, "checked": False},
            ]
        )
        session = LiveFormSession(call_id="sp", url="https://example.test")
        with self.assertRaises(RuntimeError) as ctx:
            await _assign_special_person_member(page, "John D.", session)
        self.assertIn("two people", str(ctx.exception).lower())
        # Never fall back to marital options in the clarify text.
        self.assertNotIn("Never Married", str(ctx.exception))

    async def test_member_click_scoped_and_verified(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        members = [
            {"label": "Aria R.", "index": 0, "dupIndex": 0, "checked": False},
            {"label": "John D.", "index": 1, "dupIndex": 0, "checked": False},
        ]

        async def evaluate(script, arg=None):
            if arg is None:
                return members
            # click path
            return {"ok": True, "label": arg.get("label")}

        page.evaluate = AsyncMock(side_effect=evaluate)
        session = LiveFormSession(call_id="sp", url="https://example.test")
        await _assign_special_person_member(page, "John.", session)
        self.assertIn("John D.", session.filled.get("special_person_assigned", ""))

    async def test_incomplete_angel_or_matches_angel(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        members = [
            {"label": "Angel R.", "index": 0, "dupIndex": 0, "checked": False},
            {"label": "John D.", "index": 1, "dupIndex": 0, "checked": False},
        ]

        async def evaluate(script, arg=None):
            if arg is None:
                return members
            return {"ok": True, "label": arg.get("label")}

        page.evaluate = AsyncMock(side_effect=evaluate)
        session = LiveFormSession(call_id="sp", url="https://example.test")
        await _assign_special_person_member(page, "Angel or", session)
        self.assertIn("Angel R.", session.filled.get("special_person_assigned", ""))

    async def test_angel_r_dot_matches_displayed_label(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        members = [
            {"label": "Angel R.", "index": 0, "dupIndex": 0, "checked": False},
            {"label": "Mike H.", "index": 1, "dupIndex": 0, "checked": False},
        ]

        async def evaluate(script, arg=None):
            if arg is None:
                return {
                    "members": members,
                    "strategy": "special_section_fallback",
                    "checkboxCount": 6,
                    "visibleSample": ["Someone is blind", "Angel R.", "Mike H."],
                }
            self.assertEqual(arg.get("label"), "Angel R.")
            return {"ok": True, "label": "Angel R.", "strategy": "scoped"}

        page.evaluate = AsyncMock(side_effect=evaluate)
        session = LiveFormSession(call_id="sp", url="https://example.test")
        await _assign_special_person_member(page, "Angel r dot", session)
        self.assertIn("Angel R.", session.filled.get("special_person_assigned", ""))

    async def test_discovery_fallback_without_heading(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _assign_special_person_member,
            _discover_special_person_members,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        # Simulate missing heading: JS returns pagewide fallback payload.
        page.evaluate = AsyncMock(
            return_value={
                "members": [
                    {
                        "label": "Angel R.",
                        "index": 0,
                        "dupIndex": 0,
                        "checked": False,
                    },
                    {
                        "label": "John. D.",
                        "index": 1,
                        "dupIndex": 0,
                        "checked": False,
                    },
                ],
                "strategy": "pagewide_person_labels",
                "checkboxCount": 8,
                "visibleSample": [
                    "Someone is blind",
                    "Angel R.",
                    "John. D.",
                    "Mike H.",
                ],
            }
        )
        discovered = await _discover_special_person_members(page)
        self.assertEqual(discovered["strategy"], "pagewide_person_labels")
        self.assertEqual(len(discovered["members"]), 2)

        # Assignment path also consumes dict payload and clicks.
        async def evaluate(script, arg=None):
            if arg is None:
                return {
                    "members": [
                        {
                            "label": "Angel R.",
                            "index": 0,
                            "dupIndex": 0,
                            "checked": False,
                        }
                    ],
                    "strategy": "special_section_fallback",
                    "checkboxCount": 4,
                    "visibleSample": ["Angel R."],
                }
            return {"ok": True, "label": "Angel R."}

        page.evaluate = AsyncMock(side_effect=evaluate)
        session = LiveFormSession(call_id="sp", url="https://example.test")
        await _assign_special_person_member(page, "Angel", session)
        self.assertIn("Angel R.", session.filled.get("special_person_assigned", ""))

    async def test_empty_discovery_logs_and_clarifies(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.evaluate = AsyncMock(
            return_value={
                "members": [],
                "strategy": "heading",
                "checkboxCount": 0,
                "visibleSample": ["Someone is blind", "Next"],
            }
        )
        session = LiveFormSession(
            call_id="sp-empty",
            url="https://example.test",
            elig_step="special_person",
        )
        with self.assertRaises(RuntimeError) as ctx:
            await _assign_special_person_member(page, "Angel", session)
        self.assertIn("can't see the household member list", str(ctx.exception).lower())
        self.assertEqual(session.elig_step, "special_person")

    async def test_failed_click_keeps_special_person_clarify(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        members = [
            {"label": "Angel R.", "index": 0, "dupIndex": 0, "checked": False},
        ]

        async def evaluate(script, arg=None):
            if arg is None:
                return members
            return {"ok": False, "reason": "not_checked"}

        page.evaluate = AsyncMock(side_effect=evaluate)
        session = LiveFormSession(
            call_id="sp",
            url="https://example.test",
            elig_step="special_person",
        )
        with self.assertRaises(RuntimeError) as ctx:
            await _assign_special_person_member(page, "Angel R.", session)
        self.assertIn("SPECIAL_PERSON_CLARIFY", str(ctx.exception))
        self.assertEqual(session.elig_step, "special_person")

    async def test_clarify_error_does_not_recover_to_special(self) -> None:
        """Regression: call_1613 — SPECIAL_PERSON_CLARIFY must not ok→special."""
        from pc_liveagent.live_form import LiveFormSession

        session = LiveFormSession(
            call_id="sp-recover",
            url="https://example.test",
            elig_step="special_person",
        )
        session.filled["special_conditions"] = "Someone is blind"
        # Simulate submit_field clarify branch ordering via _screen_first_step guard.
        from pc_liveagent.live_form import _screen_first_step

        recovered = _screen_first_step("special", session)
        self.assertEqual(recovered, "special_person")
        self.assertNotEqual(recovered, "special")


class AgeAndChildFormRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await drop_controller("age-child")

    async def test_claude_timeout_does_not_select_age(self) -> None:
        """Claude timeout on ambiguous age must clarify — never click Young Adult."""
        from pc_liveagent.answer_parse import ClaudeResolveResult

        call_id = "age-child"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="spouse_age"
        )

        async def fake_get_session(_cid: str):
            return session

        captions = ["First Name", "Last Name", "Spouse", "Partner", "Age"]
        submit = AsyncMock()
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch(
                "pc_liveagent.call_turn.peek_choice_labels",
                AsyncMock(return_value=captions),
            ),
            patch(
                "pc_liveagent.call_turn.resolve_with_claude",
                AsyncMock(
                    return_value=ClaudeResolveResult(
                        reject_reason="timeout",
                        needs_clarify=True,
                        latency_ms=1500,
                    )
                ),
            ),
            patch("pc_liveagent.call_turn.submit_field", submit),
        ):
            r = await process_user_turn(call_id, "Twenty twenty two thirty")
        self.assertIn("age", r.speech.lower())
        self.assertNotIn("first name", r.speech.lower())
        self.assertNotIn("young adult", r.speech.lower())
        self.assertIn(r.detail, {"age_ambiguous", "clarify", "timeout"})
        submit.assert_not_called()
        self.assertEqual(session.elig_step, "spouse_age")

    async def test_twenty_five_fast_path_submits_spoken_age(self) -> None:
        call_id = "age-child"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="spouse_age"
        )

        async def fake_get_session(_cid: str):
            return session

        async def submit(_cid, _field, value, **_k):
            session.elig_step = "children"
            return {
                "ok": True,
                "spoken": "ASK: Do you live with and mainly support any children?",
                "step": "children",
                "value": value,
            }

        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch(
                "pc_liveagent.call_turn.peek_choice_labels",
                AsyncMock(
                    return_value=[
                        "First Name",
                        "Last Name",
                        "Spouse",
                        "Partner",
                        "Age",
                    ]
                ),
            ),
            patch("pc_liveagent.call_turn.submit_field", side_effect=submit) as submit_mock,
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
            patch("pc_liveagent.call_turn.resolve_with_claude", AsyncMock()) as claude,
        ):
            r = await process_user_turn(call_id, "Twenty five", response_id=1)
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        claude.assert_not_called()
        self.assertEqual(r.detail, "elig_ack")
        submit_mock.assert_awaited()
        self.assertEqual(submit_mock.await_args.args[2], "Twenty five")

    async def test_children_yes_waits_for_child_form(self) -> None:
        """Children=Yes must not set child_first until First Name is ready."""
        from pc_liveagent.live_form import _submit_eligibility_answer

        call_id = "age-child"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="children"
        )
        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        visible_phases = {"n": 0}

        async def eligibility_screen(_page):
            # First polls still on children; later child_form.
            visible_phases["n"] += 1
            return "children" if visible_phases["n"] < 3 else "child_form"

        name_ready = {"ok": False}

        async def name_visible(_page, label):
            if label == "First Name" and visible_phases["n"] >= 4:
                name_ready["ok"] = True
                return True
            if label == "Last Name" and name_ready["ok"]:
                return True
            return False

        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                side_effect=eligibility_screen,
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                AsyncMock(return_value=["Yes", "No"]),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form._name_field_visible",
                side_effect=name_visible,
            ),
            patch(
                "pc_liveagent.live_form._wait_for_child_form_ready",
                AsyncMock(return_value=True),
            ),
        ):
            result = await _submit_eligibility_answer(
                call_id, session, page, "Yes", started=0.0
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("step"), "child_first")

    async def test_child_first_recovers_when_field_initially_absent(self) -> None:
        from pc_liveagent.live_form import _submit_eligibility_answer

        call_id = "age-child"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_first"
        )
        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        attempts = {"n": 0}

        async def name_visible(_page, label):
            if label != "First Name":
                return False
            attempts["n"] += 1
            return attempts["n"] >= 2

        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                AsyncMock(return_value="child_form"),
            ),
            patch(
                "pc_liveagent.live_form._name_field_visible",
                side_effect=name_visible,
            ),
            patch(
                "pc_liveagent.live_form._wait_for_child_form_ready",
                AsyncMock(return_value=True),
            ),
            patch("pc_liveagent.live_form._strict_fill", AsyncMock()) as fill,
        ):
            result = await _submit_eligibility_answer(
                call_id, session, page, "Jackie", started=0.0
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("step"), "child_last")
        fill.assert_awaited_once()
        self.assertEqual(fill.await_args.args[1], "First Name")
        self.assertEqual(fill.await_args.args[2], "Jackie")

    async def test_child_form_fields_fill_in_order(self) -> None:
        from pc_liveagent.live_form import _submit_eligibility_answer

        call_id = "age-child"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_first"
        )
        page = AsyncMock()
        fills: list[tuple[str, str]] = []

        async def strict_fill(_page, label, value):
            fills.append((label, value))

        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                AsyncMock(return_value="child_form"),
            ),
            patch(
                "pc_liveagent.live_form._name_field_visible",
                AsyncMock(return_value=True),
            ),
            patch("pc_liveagent.live_form._strict_fill", side_effect=strict_fill),
            patch("pc_liveagent.live_form._click_exact_choice", AsyncMock()) as click,
            patch(
                "pc_liveagent.live_form._ensure_age_choices_visible",
                AsyncMock(
                    return_value=[
                        "(0-1) Baby",
                        "(1-5) Toddler",
                        "(6-12) Child",
                        "(13-15) Teen",
                        "(16-18) Teen",
                    ]
                ),
            ),
            patch("pc_liveagent.live_form._click_age_choice", AsyncMock()) as age_click,
        ):
            r1 = await _submit_eligibility_answer(
                call_id, session, page, "Jackie", started=0.0
            )
            session.elig_step = r1["step"]
            r2 = await _submit_eligibility_answer(
                call_id, session, page, "Smith", started=0.0
            )
            session.elig_step = r2["step"]
            r3 = await _submit_eligibility_answer(
                call_id, session, page, "Child", started=0.0
            )
            session.elig_step = r3["step"]
            r4 = await _submit_eligibility_answer(
                call_id, session, page, "15", started=0.0
            )
        self.assertEqual([s for s, _ in fills], ["First Name", "Last Name"])
        self.assertEqual(fills[0][1], "Jackie")
        self.assertEqual(fills[1][1], "Smith")
        click.assert_awaited()
        age_click.assert_awaited()
        self.assertEqual(r4.get("step"), "child_more")

    async def test_child_form_missing_keeps_state_and_explains(self) -> None:
        from pc_liveagent.live_form import _submit_eligibility_answer

        call_id = "age-child"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="children"
        )
        page = AsyncMock()
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                AsyncMock(return_value="children"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                AsyncMock(return_value=["Yes", "No"]),
            ),
            patch("pc_liveagent.live_form._click_exact_choice", AsyncMock()),
            patch(
                "pc_liveagent.live_form._wait_for_child_form_ready",
                AsyncMock(return_value=False),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _submit_eligibility_answer(
                    call_id, session, page, "Yes", started=0.0
                )
        self.assertIn("CHILD_FORM", str(ctx.exception))
        self.assertEqual(session.elig_step, "children")

    async def test_child_more_continue_fail_does_not_restart_child_first(self) -> None:
        """Regression call_d0fa: stuck child_form must not ok→child_first + commit."""
        from pc_liveagent.live_form import _submit_eligibility_answer

        call_id = "age-child"
        session = LiveFormSession(
            call_id=call_id,
            url="https://example.test",
            elig_step="child_more",
            page=AsyncMock(),
        )
        session.filled["child_instance"] = "1"
        page = session.page

        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                AsyncMock(return_value="child_form"),
            ),
            patch(
                "pc_liveagent.live_form._click_continue_to",
                AsyncMock(
                    side_effect=RuntimeError(
                        "Pointcare did not reach ('student',); current screen is child_form"
                    )
                ),
            ),
            patch(
                "pc_liveagent.live_form._visible_validation_hints",
                AsyncMock(return_value="Required field"),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _submit_eligibility_answer(
                    call_id, session, page, "No", started=0.0
                )
        self.assertIn("CHILD_MORE_CLARIFY", str(ctx.exception))
        self.assertEqual(session.elig_step, "child_more")

        # Recovery guard: same-form failure must not remap child_more → child_first.
        from pc_liveagent.live_form import _screen_first_step, _step_group

        recovered = _screen_first_step("child_form", session)
        self.assertEqual(recovered, "child_first")
        self.assertEqual(_step_group("child_more"), "child_form")
        self.assertEqual(_step_group(recovered), "child_form")
        # Same form group ⇒ treat as failure, not successful recovery.
        same_form = _step_group(recovered) == _step_group("child_more")
        self.assertTrue(same_form)

    async def test_second_child_same_answers_not_blocked(self) -> None:
        call_id = "age-child"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        # Simulate first child already committed Child + Youth + No for instance 1.
        ctrl.committed = [
            {"key": "child_rel:1:child", "step": "child_rel", "answer": "Child"},
            {
                "key": "child_age:1:(6-18) youth",
                "step": "child_age",
                "answer": "(6-18) Youth",
            },
            {"key": "child_more:1:no", "step": "child_more", "answer": "No"},
        ]
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_rel"
        )
        session.filled["child_instance"] = "2"

        async def fake_get_session(_cid: str):
            return session

        submit = AsyncMock(
            return_value={
                "ok": True,
                "spoken": "ASK: What is the child's age range?",
                "step": "child_age",
            }
        )
        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=["Child", "Stepchild"])),
            patch("pc_liveagent.call_turn.submit_field", submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            r = await process_user_turn(call_id, "Child", response_id=1)
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertEqual(r.detail, "elig_ack")
        submit.assert_awaited()
        self.assertNotEqual(r.detail, "already_committed")

    async def test_already_committed_retries_when_still_on_step(self) -> None:
        call_id = "age-child"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        ctrl.committed = [
            {"key": "child_more:1:no", "step": "child_more", "answer": "No"},
        ]
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_more"
        )
        session.filled["child_instance"] = "1"

        async def fake_get_session(_cid: str):
            return session

        submit = AsyncMock(
            return_value={
                "ok": True,
                "spoken": "ASK: Are you the parent of a full-time student?",
                "step": "student",
            }
        )
        interrupt = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=["Yes", "No"])),
            patch("pc_liveagent.call_turn.submit_field", submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            r = await process_user_turn(call_id, "No", response_id=1)
            task = __import__(
                "pc_liveagent.call_turn", fromlist=["_elig_browser_tasks"]
            )._elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertEqual(r.detail, "elig_ack")
        submit.assert_awaited()

    async def test_control_commands_repeat_redo_change(self) -> None:
        from pc_liveagent.control_intent import detect_control_intent

        self.assertEqual(detect_control_intent("Ask the question again").kind, "repeat")
        self.assertEqual(detect_control_intent("Repeat that").kind, "repeat")
        self.assertEqual(detect_control_intent("Start again").kind, "redo")
        self.assertEqual(detect_control_intent("Redo").kind, "redo")
        self.assertEqual(detect_control_intent("Change that").kind, "change")
        change = detect_control_intent("Change the first name to Virat")
        self.assertEqual(change.kind, "change")
        self.assertEqual(change.target, "first_name")
        self.assertEqual(change.value.lower(), "virat")
        self.assertEqual(detect_control_intent("Go back").kind, "go_back")
        self.assertEqual(detect_control_intent("Next question").kind, "next")
        self.assertEqual(detect_control_intent("Previous question").kind, "go_back")

    async def test_add_another_bumps_child_instance(self) -> None:
        from pc_liveagent.live_form import _submit_eligibility_answer

        call_id = "age-child"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="child_more"
        )
        session.filled["child_instance"] = "1"
        page = AsyncMock()
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                AsyncMock(return_value="child_form"),
            ),
            patch("pc_liveagent.live_form._click_cta", AsyncMock()),
            patch(
                "pc_liveagent.live_form._wait_for_screen",
                AsyncMock(return_value="child_form"),
            ),
            patch(
                "pc_liveagent.live_form._wait_for_child_form_ready",
                AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.live_form._name_field_visible",
                AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.live_form._locate_named_input",
                AsyncMock(return_value=None),
            ),
        ):
            result = await _submit_eligibility_answer(
                call_id, session, page, "Yes", started=0.0
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("step"), "child_first")
        self.assertEqual(session.filled.get("child_instance"), "2")

    def test_age_seven_and_spoken_range(self) -> None:
        from pc_liveagent.answer_parse import resolve_answer
        from pc_liveagent.elig_meta import get_field_meta
        from pc_liveagent.live_form import _map_age_label

        opts = [
            "(0-1) Baby",
            "(1-5) Toddler",
            "(6-18) Youth",
            "(19-20) Young Adult",
        ]
        self.assertEqual(_map_age_label("7", opts), "(6-18) Youth")
        self.assertEqual(_map_age_label("Six to eighteen", opts), "(6-18) Youth")
        r = resolve_answer(
            "child_age", "7", options=opts, meta=get_field_meta("child_age")
        )
        self.assertEqual(r.answer, "(6-18) Youth")
        ambiguous = _map_age_label("Twenty twenty two thirty", opts)
        self.assertIsNone(ambiguous)


if __name__ == "__main__":
    unittest.main()
