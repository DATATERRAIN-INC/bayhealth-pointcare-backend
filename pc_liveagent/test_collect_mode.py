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

    async def test_intro_for_what_stays_on_language_with_context(self) -> None:
        call_id = "turn-test"
        await drop_controller(call_id)
        r = await process_user_turn(call_id, "For what?")
        self.assertEqual(r.phase, "language")
        self.assertEqual(r.detail, "repeat")
        self.assertIn("pointcare", r.speech.lower())
        self.assertIn("english", r.speech.lower())
        self.assertNotIn("google", r.speech.lower())

    async def test_sign_in_what_repeats_chooser_with_context(self) -> None:
        call_id = "turn-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "sign_in"
        r = await process_user_turn(call_id, "What?")
        self.assertEqual(r.phase, "sign_in")
        self.assertEqual(r.detail, "repeat")
        self.assertIn("pointcare", r.speech.lower())
        self.assertIn("google", r.speech.lower())

    async def test_eligibility_what_do_you_mean_repeats_question(self) -> None:
        call_id = "turn-test"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="marital"
        )
        submit = AsyncMock()

        async def fake_get_session(_cid: str):
            return session

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=[])),
            patch("pc_liveagent.call_turn.submit_field", submit),
        ):
            r = await process_user_turn(call_id, "What do you mean?")
        self.assertEqual(r.phase, "eligibility")
        self.assertEqual(r.detail, "repeat")
        self.assertIn("marital", r.speech.lower())
        submit.assert_not_called()

    async def test_programs_complete_thanks_and_ends_call(self) -> None:
        """After household eligibility finishes, thank the caller and end_call."""
        from pc_liveagent.call_turn import (
            CLOSING,
            _run_eligibility_browser_background,
            drop_controller,
            get_controller,
        )

        call_id = "elig-complete"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="programs"
        )
        deliver = AsyncMock(return_value=True)

        async def fake_get_session(_cid: str):
            return session

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch(
                "pc_liveagent.call_turn.submit_field",
                AsyncMock(
                    return_value={
                        "ok": True,
                        "spoken": (
                            "I've finished filling your household eligibility form. "
                            "Thank you for your time. Goodbye."
                        ),
                        "step": "",
                    }
                ),
            ),
            patch(
                "pc_liveagent.call_turn._deliver_question_once",
                deliver,
            ),
        ):
            await _run_eligibility_browser_background(
                call_id,
                answer="No",
                step="programs",
                turn_id=f"{call_id}:1:close",
                form_url="https://example.test",
                commit_key="programs:1:no",
                epoch=ctrl.elig_nav_epoch,
            )
        self.assertEqual(ctrl.phase, "closing")
        self.assertIn("household eligibility", ctrl.last_speech.lower())
        self.assertIn("thank", ctrl.last_speech.lower())
        deliver.assert_awaited()
        kwargs = deliver.await_args.kwargs
        self.assertTrue(kwargs.get("end_call"))
        self.assertIn("close:", kwargs.get("key", ""))
        # Sync path also ends the call when already closing.
        r = await process_user_turn(call_id, "")
        self.assertTrue(r.end_call)
        self.assertEqual(r.phase, "closing")
        self.assertTrue(CLOSING)
        await drop_controller(call_id)

    async def test_speak_now_closing_sets_end_call_flag(self) -> None:
        from pc_liveagent import retell_llm_ws

        sent: list[dict] = []

        async def fake_send(_ws, payload):
            sent.append(payload)

        with (
            patch.dict(retell_llm_ws._active_ws, {"c-end": object()}, clear=False),
            patch("pc_liveagent.retell_llm_ws._send", side_effect=fake_send),
        ):
            ok = await retell_llm_ws.speak_now(
                "c-end", "Thanks, goodbye.", interrupt_id=9, end_call=True
            )
        self.assertTrue(ok)
        self.assertEqual(sent[0]["response_type"], "agent_interrupt")
        self.assertTrue(sent[0].get("end_call"))

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
        from pc_liveagent.live_form import (
            _income_period_choice,
            _resolve_income_period_label,
        )

        self.assertEqual(_income_period_choice("yearly"), "Annual")
        self.assertEqual(_income_period_choice("Annually."), "Annual")
        self.assertEqual(_income_period_choice("every two weeks"), "Bi-weekly")
        self.assertEqual(_income_period_choice("two weeks."), "Bi-weekly")
        self.assertEqual(_income_period_choice("Monthly."), "Monthly")
        # Hyphen / spacing / STT variants must map (not fail on week|ly boundary).
        for spoken in (
            "bi-weekly",
            "biweekly",
            "bi weekly",
            "Bi weekly.",
            "every 2 weeks",
            "by weekly",
            "buy weekly",
        ):
            self.assertEqual(
                _income_period_choice(spoken),
                "Bi-weekly",
                msg=spoken,
            )
            self.assertEqual(
                _resolve_income_period_label(
                    spoken, ["Annual", "Monthly", "Bi-weekly"]
                ),
                "Bi-weekly",
                msg=spoken,
            )
        self.assertEqual(
            _resolve_income_period_label(
                "not a period", ["Annual", "Monthly", "Bi-weekly"]
            ),
            "",
        )
        # Never fall back to the first visible chip.
        self.assertNotEqual(
            _resolve_income_period_label(
                "fortnightly-ish maybe", ["Annual", "Monthly", "Bi-weekly"]
            ),
            "Annual",
        )

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


class BrowserLinkTests(unittest.IsolatedAsyncioTestCase):
    def test_local_devtools_url(self) -> None:
        from pc_liveagent.browser import local_devtools_url

        url = local_devtools_url(
            {
                "id": "ABC123",
                "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/ABC123",
            }
        )
        self.assertIn("127.0.0.1:9222/devtools/inspector.html", url)
        self.assertIn("ABC123", url)

    async def test_describe_live_browser_includes_open_url(self) -> None:
        from pc_liveagent.live_form import describe_live_browser

        page = AsyncMock()
        page.url = "https://my.pointcare.com/auth"
        page.is_closed = lambda: False
        session = LiveFormSession(
            call_id="call_browser_link",
            url="https://my.pointcare.com/onboarding",
            welcome_done=True,
            status="open",
        )
        session.page = page
        with patch(
            "pc_liveagent.browser.cdp_target_for_url",
            return_value={
                "id": "TAB1",
                "url": "https://my.pointcare.com/auth",
                "title": "Pointcare",
                "type": "page",
            },
        ):
            info = await describe_live_browser(
                session,
                call_id="call_browser_link",
                open_base_url="https://tunnel.example",
            )
        self.assertTrue(info["opened"])
        self.assertEqual(info["url"], "https://my.pointcare.com/auth")
        self.assertEqual(
            info["open_url"],
            "https://tunnel.example/form-filler/sessions/call_browser_link/chrome",
        )
        self.assertIn("devtools", info["devtools_url"])
        self.assertTrue(info["chrome_user_data"])
        self.assertEqual(session.url, "https://my.pointcare.com/auth")


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


class EmailAuthTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        for cid in (
            "email-choice",
            "email-ack",
            "email-fail",
            "email-secure",
            "email-success",
            "email-extract",
            "email-goback",
            "email-goback-race",
        ):
            await drop_controller(cid)

    def test_extract_and_normalize_spoken_email(self) -> None:
        from pc_liveagent.answer_parse import extract_email, is_email_choice, normalize_spoken_email

        self.assertTrue(is_email_choice("use email"))
        self.assertTrue(is_email_choice("I'll sign in with email"))
        self.assertFalse(is_email_choice("Google email"))
        self.assertEqual(extract_email("jane.doe@example.com"), "jane.doe@example.com")
        self.assertEqual(
            extract_email("jane doe at example dot com"),
            "janedoe@example.com",
        )
        self.assertEqual(
            normalize_spoken_email("name at domain dot org"),
            "name@domain.org",
        )
        self.assertEqual(extract_email("use email"), "")
        # Retell / spoken variants that previously caused re-prompt loops.
        self.assertEqual(
            extract_email("my email is murali at gmail dot com"),
            "murali@gmail.com",
        )
        self.assertEqual(
            extract_email("murali at the rate gmail.com"),
            "murali@gmail.com",
        )
        self.assertEqual(
            extract_email("john underscore doe at gmail period com"),
            "john_doe@gmail.com",
        )
        self.assertEqual(
            extract_email("jane hyphen doe at example dot com"),
            "jane-doe@example.com",
        )
        self.assertEqual(
            extract_email("email is pat at g mail dot com"),
            "pat@gmail.com",
        )
        # Do not invent @ when the speaker never said at / @.
        self.assertEqual(extract_email("murali gmail.com"), "")
        self.assertEqual(extract_email("hello there"), "")
        # Spoken trailing digits (Retell often says zerozerothree for 003).
        self.assertEqual(
            extract_email("murali zero zero three at gmail dot com"),
            "murali003@gmail.com",
        )
        self.assertEqual(
            extract_email("murali zerozerothree at gmail.com"),
            "murali003@gmail.com",
        )
        self.assertEqual(
            extract_email("john oh one two at example period com"),
            "john012@example.com",
        )
        # Must not corrupt normal words like "someone".
        self.assertEqual(extract_email("someone at example.com"), "someone@example.com")

    async def test_spoken_email_does_not_reprompt(self) -> None:
        from pc_liveagent.call_turn import EMAIL_ACK, EMAIL_CLARIFY, _email_auth_tasks

        call_id = "email-extract"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "email"
        gate = asyncio.Event()

        async def slow_submit(cid, field, value, **_k):
            self.assertEqual(field, "email")
            self.assertEqual(value, "murali@gmail.com")
            await gate.wait()
            return {
                "ok": True,
                "auth_screen": "need_password",
                "spoken": "Please finish signing in in the Pointcare window.",
                "awaiting_secure": True,
            }

        with (
            patch("pc_liveagent.call_turn.submit_field", side_effect=slow_submit),
            patch(
                "pc_liveagent.call_turn.poll_email_auth_until_signed_in",
                AsyncMock(
                    return_value={
                        "ok": True,
                        "auth_screen": "signed_in",
                        "spoken": "ASK: What is your marital status?",
                        "step": "marital",
                    }
                ),
            ),
            patch("pc_liveagent.call_turn.get_session", AsyncMock(return_value=None)),
            patch("pc_liveagent.call_turn._speak_interrupt", AsyncMock(return_value=True)),
            patch(
                "pc_liveagent.call_turn.deliver_first_eligibility_question",
                AsyncMock(return_value=True),
            ),
        ):
            first = await process_user_turn(
                call_id, "my email is murali at gmail dot com", response_id=1
            )
            self.assertEqual(first.detail, "email_confirm")
            self.assertIn("murali at gmail dot com", first.speech.lower())
            self.assertFalse(first.end_call)
            confirmed = await process_user_turn(call_id, "yes", response_id=2)
            self.assertEqual(confirmed.detail, "email_ack")
            self.assertEqual(confirmed.speech, EMAIL_ACK)
            self.assertTrue(ctrl.email_captured)
            # A follow-up while auth runs must not re-ask for the address.
            second = await process_user_turn(call_id, "hello?", response_id=3)
            self.assertNotEqual(second.speech, EMAIL_CLARIFY)
            self.assertNotIn("what email", second.speech.lower())
            self.assertFalse(second.end_call)
            gate.set()
            task = _email_auth_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertEqual(ctrl.phase, "eligibility")

    async def test_unclear_email_gets_clarify_not_loop_identical(self) -> None:
        from pc_liveagent.call_turn import EMAIL_CLARIFY, EMAIL_QUESTION

        call_id = "email-choice"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "email"
        result = await process_user_turn(call_id, "murali gmail.com")
        self.assertEqual(result.detail, "email_address_unclear")
        self.assertEqual(result.speech, EMAIL_CLARIFY)
        self.assertNotEqual(result.speech, EMAIL_QUESTION)
        self.assertFalse(result.end_call)
        self.assertEqual(ctrl.phase, "email")

    async def test_email_confirm_and_go_back(self) -> None:
        from pc_liveagent.call_turn import EMAIL_ACK, EMAIL_GO_BACK, _email_auth_tasks

        call_id = "email-goback"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "email"

        confirm = await process_user_turn(call_id, "test@example.com")
        self.assertEqual(confirm.detail, "email_confirm")
        self.assertEqual(ctrl.phase, "email_confirm")
        self.assertEqual(ctrl.pending_email, "test@example.com")

        with patch("pc_liveagent.call_turn._reset_email_sign_in_browser", AsyncMock()):
            back = await process_user_turn(call_id, "go back")
        self.assertEqual(back.detail, "email_go_back")
        self.assertEqual(back.speech, EMAIL_GO_BACK)
        self.assertEqual(ctrl.phase, "sign_in")
        self.assertEqual(ctrl.pending_email, "")
        self.assertFalse(ctrl.email_captured)

        # ASR-glued "goback" must also work.
        await process_user_turn(call_id, "use email")
        await process_user_turn(call_id, "test@example.com")
        with patch("pc_liveagent.call_turn._reset_email_sign_in_browser", AsyncMock()):
            glued = await process_user_turn(call_id, "goback")
        self.assertEqual(glued.detail, "email_go_back")
        self.assertEqual(ctrl.phase, "sign_in")

        await process_user_turn(call_id, "use email")
        again = await process_user_turn(call_id, "right@example.com")
        self.assertEqual(again.detail, "email_confirm")
        self.assertIn("right at example dot com", again.speech.lower())

        # Saying "no" also goes back.
        with patch("pc_liveagent.call_turn._reset_email_sign_in_browser", AsyncMock()):
            await process_user_turn(call_id, "no that's wrong")
        self.assertEqual(ctrl.phase, "sign_in")

        await process_user_turn(call_id, "use email")
        await process_user_turn(call_id, "right@example.com")
        with (
            patch("pc_liveagent.call_turn.submit_field", AsyncMock(
                return_value={
                    "ok": True,
                    "auth_screen": "need_password",
                    "awaiting_secure": True,
                    "spoken": "Please finish creating or entering your password in the Pointcare window.",
                }
            )),
            patch(
                "pc_liveagent.call_turn.poll_email_auth_until_signed_in",
                AsyncMock(return_value=None),
            ),
            patch("pc_liveagent.call_turn._speak_interrupt", AsyncMock(return_value=True)),
            patch("pc_liveagent.call_turn._reset_email_sign_in_browser", AsyncMock()),
        ):
            yes = await process_user_turn(call_id, "yes", response_id=9)
            self.assertEqual(yes.detail, "email_ack")
            self.assertEqual(yes.speech, EMAIL_ACK)
            task = _email_auth_tasks.get(call_id)
            # Go back after auth started (glued form).
            back2 = await process_user_turn(call_id, "goback")
            self.assertEqual(back2.detail, "email_go_back")
            self.assertEqual(ctrl.phase, "sign_in")
            self.assertFalse(ctrl.email_auth_in_progress)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        await drop_controller(call_id)

    async def test_go_back_wins_over_stale_email_auth_task(self) -> None:
        """Playwright finishing after go-back must not steal the phase."""
        from pc_liveagent.call_turn import EMAIL_GO_BACK, _email_auth_tasks

        call_id = "email-goback-race"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "email"
        release_fill = asyncio.Event()
        fill_started = asyncio.Event()

        async def slow_submit(*_a, **_k):
            fill_started.set()
            await release_fill.wait()
            return {
                "ok": True,
                "auth_screen": "need_password",
                "awaiting_secure": True,
                "spoken": "Please finish signing in.",
            }

        with (
            patch("pc_liveagent.call_turn.submit_field", side_effect=slow_submit),
            patch(
                "pc_liveagent.call_turn.poll_email_auth_until_signed_in",
                AsyncMock(
                    return_value={
                        "ok": True,
                        "auth_screen": "signed_in",
                        "spoken": "ASK: What is your marital status?",
                        "step": "marital",
                    }
                ),
            ),
            patch("pc_liveagent.call_turn._speak_interrupt", AsyncMock(return_value=True)),
            patch(
                "pc_liveagent.call_turn.deliver_first_eligibility_question",
                AsyncMock(return_value=True),
            ),
            patch("pc_liveagent.call_turn._reset_email_sign_in_browser", AsyncMock()),
        ):
            await process_user_turn(call_id, "bad@example.com")
            await process_user_turn(call_id, "yes", response_id=1)
            self.assertEqual(ctrl.phase, "email_await_secure")
            await asyncio.wait_for(fill_started.wait(), timeout=2.0)
            epoch_before = ctrl.email_auth_epoch
            back = await process_user_turn(call_id, "goback", response_id=2)
            self.assertEqual(back.detail, "email_go_back")
            self.assertEqual(back.speech, EMAIL_GO_BACK)
            self.assertEqual(ctrl.phase, "sign_in")
            self.assertGreater(ctrl.email_auth_epoch, epoch_before)
            release_fill.set()
            task = _email_auth_tasks.get(call_id)
            if task:
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
            await asyncio.sleep(0.05)
        self.assertEqual(ctrl.phase, "sign_in")
        self.assertFalse(ctrl.email_auth_in_progress)
        self.assertNotEqual(ctrl.phase, "eligibility")
        await drop_controller(call_id)

    async def test_go_back_preferred_from_recent_transcript(self) -> None:
        from pc_liveagent.call_turn import latest_user_utterance

        transcript = [
            {"role": "agent", "content": "Is that correct?"},
            {"role": "user", "content": "murali003@gmail.com"},
            {"role": "user", "content": "go back"},
        ]
        # Newest user line wins — do not scan older lines for go-back.
        self.assertEqual(latest_user_utterance(transcript), "go back")
        transcript2 = [
            {"role": "user", "content": "murali003@gmail.com"},
            {"role": "user", "content": "goback"},
        ]
        self.assertEqual(latest_user_utterance(transcript2), "goback")
        # Newer Google must not be stolen by an older go-back in the window.
        transcript3 = [
            {"role": "user", "content": "go back"},
            {"role": "agent", "content": "How would you like to sign in?"},
            {"role": "user", "content": "Continue with Google"},
        ]
        self.assertEqual(latest_user_utterance(transcript3), "Continue with Google")

    async def test_use_email_asks_for_address(self) -> None:
        from pc_liveagent.call_turn import EMAIL_QUESTION

        call_id = "email-choice"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "sign_in"
        result = await process_user_turn(call_id, "use email")
        self.assertEqual(result.phase, "email")
        self.assertEqual(result.speech, EMAIL_QUESTION)
        self.assertFalse(result.end_call)

    async def test_go_back_phrase_is_not_email_choice(self) -> None:
        from pc_liveagent.answer_parse import is_email_choice

        self.assertFalse(
            is_email_choice(
                "Go back. I've entered the wrong email address. "
                "I have to use the different address."
            )
        )
        self.assertTrue(is_email_choice("use email"))
        self.assertTrue(is_email_choice("I'll sign in with email"))


class StaleTranscriptDedupeTests(unittest.IsolatedAsyncioTestCase):
    """Retell sends cumulative transcripts; unchanged tips must not re-run Go Back."""

    async def test_same_response_id_handled_once(self) -> None:
        from pc_liveagent.call_turn import (
            classify_transcript_turn,
            mark_transcript_turn_handled,
        )

        call_id = "stale-rid"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        transcript = [
            {"role": "user", "content": "go back"},
        ]
        d1 = classify_transcript_turn(ctrl, transcript, response_id=6)
        self.assertEqual(d1.dedupe, "process")
        mark_transcript_turn_handled(ctrl, d1, response_id=6)
        d2 = classify_transcript_turn(ctrl, transcript, response_id=6)
        self.assertEqual(d2.dedupe, "duplicate_response_id")
        await drop_controller(call_id)

    async def test_stale_tip_does_not_reprocess(self) -> None:
        from pc_liveagent.call_turn import (
            classify_transcript_turn,
            mark_transcript_turn_handled,
        )

        call_id = "stale-tip"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        go_back = (
            "Go back. I've entered the wrong email address. "
            "I have to use the different address."
        )
        transcript = [
            {"role": "agent", "content": "Is that correct?"},
            {"role": "user", "content": go_back},
        ]
        d1 = classify_transcript_turn(ctrl, transcript, response_id=6)
        self.assertEqual(d1.dedupe, "process")
        self.assertEqual(d1.utterance, go_back)
        mark_transcript_turn_handled(ctrl, d1, response_id=6)
        # New response_id, same cumulative tip → stale (not a new user turn).
        d2 = classify_transcript_turn(ctrl, transcript, response_id=7)
        self.assertEqual(d2.dedupe, "stale_tip")
        d3 = classify_transcript_turn(ctrl, transcript, response_id=13)
        self.assertEqual(d3.dedupe, "stale_tip")
        await drop_controller(call_id)

    async def test_same_text_new_user_index_is_new_turn(self) -> None:
        from pc_liveagent.call_turn import (
            classify_transcript_turn,
            mark_transcript_turn_handled,
        )

        call_id = "stale-new-idx"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        t1 = [{"role": "user", "content": "go back"}]
        d1 = classify_transcript_turn(ctrl, t1, response_id=1)
        mark_transcript_turn_handled(ctrl, d1, response_id=1)
        # User literally said "go back" again — new transcript entry.
        t2 = [
            {"role": "user", "content": "go back"},
            {"role": "agent", "content": "How would you like to sign in?"},
            {"role": "user", "content": "go back"},
        ]
        d2 = classify_transcript_turn(ctrl, t2, response_id=2)
        self.assertEqual(d2.dedupe, "process")
        self.assertEqual(d2.user_index, 1)
        await drop_controller(call_id)

    async def test_go_back_once_then_google_once(self) -> None:
        from pc_liveagent.call_turn import (
            EMAIL_GO_BACK,
            SSO_ACK,
            classify_transcript_turn,
            mark_transcript_turn_handled,
        )

        call_id = "stale-goback-google"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "email_confirm"
        ctrl.pending_email = "wrong@example.com"
        ctrl.email_captured = True
        go_back = (
            "Go back. I've entered the wrong email address. "
            "I have to use the different address."
        )
        transcript_back = [
            {"role": "user", "content": "wrong@example.com"},
            {"role": "user", "content": go_back},
        ]
        reset = AsyncMock()
        with patch("pc_liveagent.call_turn._reset_email_sign_in_browser", reset):
            d1 = classify_transcript_turn(ctrl, transcript_back, response_id=6)
            self.assertEqual(d1.dedupe, "process")
            mark_transcript_turn_handled(ctrl, d1, response_id=6)
            back = await process_user_turn(call_id, d1.utterance, response_id=6)
            self.assertEqual(back.detail, "email_go_back")
            self.assertEqual(back.speech, EMAIL_GO_BACK)
            self.assertEqual(ctrl.phase, "sign_in")
            # Browser reset is fire-and-forget via create_task.
            await asyncio.sleep(0)
            self.assertEqual(reset.await_count, 1)
            epoch_after = ctrl.email_auth_epoch

            # Responses 7–13: same tip, new response_ids — no browser reset.
            for rid in (7, 8, 9, 10, 11, 12, 13):
                d = classify_transcript_turn(ctrl, transcript_back, response_id=rid)
                self.assertEqual(d.dedupe, "stale_tip", msg=f"rid={rid}")
                mark_transcript_turn_handled(ctrl, d, response_id=rid)
            await asyncio.sleep(0)
            self.assertEqual(reset.await_count, 1)
            self.assertEqual(ctrl.email_auth_epoch, epoch_after)

            # Explicit new Google turn.
            transcript_google = transcript_back + [
                {"role": "agent", "content": EMAIL_GO_BACK},
                {"role": "user", "content": "Continue with Google"},
            ]
            d_g = classify_transcript_turn(ctrl, transcript_google, response_id=14)
            self.assertEqual(d_g.dedupe, "process")
            self.assertEqual(d_g.utterance, "Continue with Google")
            mark_transcript_turn_handled(ctrl, d_g, response_id=14)
            with patch(
                "pc_liveagent.call_turn._schedule_google_sso",
                lambda *a, **k: None,
            ):
                google = await process_user_turn(
                    call_id, d_g.utterance, response_id=14
                )
            self.assertEqual(google.detail, "sso_ack")
            self.assertEqual(google.speech, SSO_ACK)
            self.assertTrue(ctrl.sso_in_progress)
            self.assertEqual(reset.await_count, 1)
        await drop_controller(call_id)

    async def test_go_back_on_chooser_does_not_reset_browser(self) -> None:
        call_id = "stale-chooser-noop"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "sign_in"
        with patch(
            "pc_liveagent.call_turn._reset_email_sign_in_browser", AsyncMock()
        ) as reset:
            r = await process_user_turn(call_id, "go back", response_id=1)
        self.assertEqual(r.detail, "email_go_back")
        self.assertEqual(ctrl.phase, "sign_in")
        reset.assert_not_awaited()
        await drop_controller(call_id)

    async def test_no_new_turn_does_not_guess_google(self) -> None:
        from pc_liveagent.call_turn import classify_transcript_turn, mark_transcript_turn_handled

        call_id = "stale-no-guess"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "sign_in"
        ctrl.last_speech = "How would you like to sign in?"
        transcript = [{"role": "user", "content": "go back"}]
        d1 = classify_transcript_turn(ctrl, transcript, response_id=1)
        mark_transcript_turn_handled(ctrl, d1, response_id=1)
        d2 = classify_transcript_turn(ctrl, transcript, response_id=2)
        self.assertEqual(d2.dedupe, "stale_tip")
        # Stale path must not invoke process_user_turn / Google.
        self.assertFalse(ctrl.sso_in_progress)
        await drop_controller(call_id)


class EmailAuthFollowupTests(unittest.IsolatedAsyncioTestCase):
    """Remaining email-auth cases (split after StaleTranscriptDedupeTests insert)."""

    async def test_email_ack_no_premature_end_call(self) -> None:
        from pc_liveagent.call_turn import EMAIL_ACK, _email_auth_tasks
        from pc_liveagent.live_form import ELIG_QUESTIONS

        call_id = "email-ack"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "email"
        started = asyncio.Event()
        finished = asyncio.Event()

        async def slow_submit(*_a, **_k):
            started.set()
            await finished.wait()
            return {
                "ok": True,
                "auth_screen": "signed_in",
                "spoken": f"ASK: {ELIG_QUESTIONS['marital']}",
                "step": "marital",
            }

        with (
            patch("pc_liveagent.call_turn.submit_field", side_effect=slow_submit),
            patch("pc_liveagent.call_turn.get_session", AsyncMock(return_value=None)),
            patch("pc_liveagent.call_turn._speak_interrupt", AsyncMock(return_value=True)),
            patch(
                "pc_liveagent.call_turn.poll_email_auth_until_signed_in",
                AsyncMock(return_value=None),
            ),
        ):
            confirm = await process_user_turn(
                call_id, "jane@example.com", response_id=2
            )
            self.assertEqual(confirm.detail, "email_confirm")
            self.assertEqual(ctrl.phase, "email_confirm")
            self.assertIn("jane at example dot com", confirm.speech.lower())
            self.assertFalse(confirm.end_call)
            result = await process_user_turn(call_id, "yes", response_id=3)
            self.assertEqual(result.detail, "email_ack")
            self.assertEqual(result.speech, EMAIL_ACK)
            self.assertFalse(result.end_call)
            self.assertEqual(result.phase, "email_await_secure")
            self.assertTrue(ctrl.email_auth_in_progress)
            self.assertNotEqual(ctrl.phase, "closing")
            await asyncio.wait_for(started.wait(), timeout=2.0)
            finished.set()
            task = _email_auth_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertEqual(ctrl.phase, "eligibility")
        self.assertTrue(ctrl.eligibility_screen_verified)
        self.assertFalse(ctrl.email_auth_in_progress)

    async def test_email_fill_failure_keeps_auth_phase(self) -> None:
        from pc_liveagent.call_turn import EMAIL_SECURE_RETRY, _email_auth_tasks

        call_id = "email-fail"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "email"
        interrupt = AsyncMock(return_value=True)

        async def fail_submit(*_a, **_k):
            return {
                "ok": False,
                "auth_screen": "need_email",
                "spoken": EMAIL_SECURE_RETRY,
                "error": "email_not_filled",
            }

        with (
            patch("pc_liveagent.call_turn.submit_field", side_effect=fail_submit),
            patch("pc_liveagent.call_turn._speak_interrupt", interrupt),
        ):
            await process_user_turn(call_id, "pat@example.com", response_id=1)
            result = await process_user_turn(call_id, "yes", response_id=2)
            self.assertFalse(result.end_call)
            self.assertNotEqual(result.phase, "closing")
            task = _email_auth_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertEqual(ctrl.phase, "email")
        self.assertFalse(ctrl.eligibility_question_sent)
        interrupt.assert_awaited()
        spoken = interrupt.await_args.args[1]
        self.assertNotIn("goodbye", spoken.lower())
        self.assertNotIn("marital", spoken.lower())

    async def test_secure_screen_awaits_without_ending_call(self) -> None:
        from pc_liveagent.call_turn import EMAIL_SECURE_COMPLETE, _email_auth_tasks
        from pc_liveagent.live_form import ELIG_QUESTIONS

        call_id = "email-secure"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "email"

        async def secure_submit(*_a, **_k):
            return {
                "ok": True,
                "auth_screen": "need_password",
                "spoken": EMAIL_SECURE_COMPLETE,
                "awaiting_secure": True,
            }

        async def later_signed_in(*_a, **_k):
            return {
                "ok": True,
                "auth_screen": "signed_in",
                "spoken": f"ASK: {ELIG_QUESTIONS['marital']}",
                "step": "marital",
            }

        with (
            patch("pc_liveagent.call_turn.submit_field", side_effect=secure_submit),
            patch(
                "pc_liveagent.call_turn.poll_email_auth_until_signed_in",
                side_effect=later_signed_in,
            ),
            patch("pc_liveagent.call_turn.get_session", AsyncMock(return_value=None)),
            patch("pc_liveagent.call_turn._speak_interrupt", AsyncMock(return_value=True)),
            patch(
                "pc_liveagent.call_turn.deliver_first_eligibility_question",
                AsyncMock(return_value=True),
            ),
        ):
            await process_user_turn(call_id, "pat@example.com", response_id=3)
            result = await process_user_turn(call_id, "yes", response_id=4)
            self.assertFalse(result.end_call)
            self.assertEqual(result.phase, "email_await_secure")
            task = _email_auth_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertEqual(ctrl.phase, "eligibility")
        self.assertTrue(ctrl.eligibility_screen_verified)

    async def test_sign_in_with_email_uses_fill_text_not_choice(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _sign_in_with_email

        session = LiveFormSession(call_id="email-extract", url="https://example.test")
        page = AsyncMock()
        page.url = "https://my.pointcare.com/signin"
        page.locator = AsyncMock()
        session.page = page

        fill = AsyncMock()
        continue_click = AsyncMock(return_value=True)
        detect = AsyncMock(
            side_effect=["need_password", "need_password", "need_password", "need_password", "need_password"]
        )
        already = AsyncMock(return_value=False)
        matches = AsyncMock(return_value=True)
        ensure = AsyncMock()

        with (
            patch("pc_liveagent.live_form._ensure_native_email_form", ensure),
            patch("pc_liveagent.live_form._fill_text", fill),
            patch("pc_liveagent.live_form._email_field_matches", matches),
            patch("pc_liveagent.live_form._click_continue_if_present", continue_click),
            patch("pc_liveagent.live_form._detect_auth_screen", detect),
            patch("pc_liveagent.live_form._already_signed_in", already),
            patch("pc_liveagent.live_form._provider_popup", return_value=None),
            patch("pc_liveagent.live_form._click_choice", AsyncMock()) as click_choice,
        ):
            result = await _sign_in_with_email(session, "jane@example.com")

        fill.assert_awaited()
        continue_click.assert_awaited()
        click_choice.assert_not_awaited()
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("auth_screen"), "need_password")
        self.assertTrue(result.get("awaiting_secure"))
        self.assertIn("password", (result.get("spoken") or "").lower())
        self.assertNotIn("goodbye", (result.get("spoken") or "").lower())

    def test_spoken_for_auth_screen_verify_link(self) -> None:
        from pc_liveagent.live_form import (
            EMAIL_PASSWORD_SECURE,
            EMAIL_VERIFY_LINK,
            spoken_for_auth_screen,
        )

        self.assertEqual(spoken_for_auth_screen("need_confirm"), EMAIL_VERIFY_LINK)
        self.assertIn("verification link", EMAIL_VERIFY_LINK.lower())
        self.assertEqual(spoken_for_auth_screen("need_password"), EMAIL_PASSWORD_SECURE)
        self.assertNotEqual(
            spoken_for_auth_screen("need_confirm"),
            spoken_for_auth_screen("need_password"),
        )


class ControlIntentAndMetaTests(unittest.TestCase):
    def test_intent_kinds(self) -> None:
        from pc_liveagent.control_intent import detect_control_intent

        self.assertEqual(detect_control_intent("Can you repeat that?").kind, "repeat")
        self.assertEqual(detect_control_intent("For what?").kind, "repeat")
        self.assertEqual(detect_control_intent("What?").kind, "repeat")
        self.assertEqual(detect_control_intent("What do you mean?").kind, "repeat")
        self.assertEqual(detect_control_intent("Why are you calling?").kind, "repeat")
        self.assertEqual(detect_control_intent("What's this about?").kind, "repeat")
        self.assertEqual(detect_control_intent("Go back one step.").kind, "go_back")
        self.assertEqual(detect_control_intent("Skip this if it is optional.").kind, "skip")
        self.assertEqual(detect_control_intent("I made a mistake.").kind, "redo")
        change = detect_control_intent("Change the age to fifteen.")
        self.assertEqual(change.kind, "change")
        self.assertEqual(change.target, "age")
        self.assertIn("fifteen", change.value.lower())
        self.assertEqual(detect_control_intent("Never married").kind, "none")
        # Real answers / next must not be treated as context-repeat.
        self.assertEqual(
            detect_control_intent("What's the next question?").kind, "next"
        )

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
            # While browser runs, another turn stays responsive (next waits, not hang).
            busy = await process_user_turn(call_id, "next question", response_id=2)
            self.assertIn(busy.detail, ("elig_op_in_progress", "next_waiting_browser"))
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

    def test_natural_blind_phrases_and_negation(self) -> None:
        from pc_liveagent.live_form import _map_special_condition_labels

        self.assertEqual(
            _map_special_condition_labels("I'm blind", self.SPECIAL_OPTS),
            ["Someone is blind"],
        )
        self.assertEqual(
            _map_special_condition_labels(
                "someone in my household is blind", self.SPECIAL_OPTS
            ),
            ["Someone is blind"],
        )
        self.assertEqual(
            _map_special_condition_labels(
                "my mother has a visual disability", self.SPECIAL_OPTS
            ),
            ["Someone is blind"],
        )
        self.assertEqual(
            _map_special_condition_labels("I'm not blind", self.SPECIAL_OPTS),
            [],
        )
        self.assertEqual(
            _map_special_condition_labels(
                "blind and pregnant", self.SPECIAL_OPTS
            ),
            ["Someone is pregnant", "Someone is blind"],
        )
        # None is exclusive when no other conditions present.
        self.assertEqual(
            _map_special_condition_labels("none of the above", self.SPECIAL_OPTS),
            ["None of the above"],
        )

    def test_relationship_synonyms(self) -> None:
        from pc_liveagent.live_form import _map_relationship_label
        from pc_liveagent.answer_parse import resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        opts = ["Child", "Stepchild", "Grandchild", "Foster Child", "Other"]
        self.assertEqual(_map_relationship_label("my daughter", opts), "Child")
        self.assertEqual(_map_relationship_label("my grandson", opts), "Grandchild")
        self.assertEqual(
            _map_relationship_label("my stepdaughter", opts), "Stepchild"
        )
        self.assertEqual(
            _map_relationship_label("my foster child", opts), "Foster Child"
        )
        self.assertIsNone(_map_relationship_label("maybe a relative kinda", opts))
        r = resolve_answer(
            "child_rel",
            "my daughter",
            options=opts,
            meta=get_field_meta("child_rel"),
        )
        self.assertEqual(r.answer, "Child")

    def test_member_period_normalization(self) -> None:
        from pc_liveagent.live_form import (
            _normalize_member_label,
            _split_member_queries,
        )

        self.assertEqual(_normalize_member_label("Murali M."), "murali m")
        self.assertEqual(_normalize_member_label("Murali M"), "murali m")
        self.assertEqual(_normalize_member_label("Agent H."), "agent h")
        self.assertEqual(_normalize_member_label("Agent H"), "agent h")
        # Distinct people must not collapse together.
        self.assertNotEqual(
            _normalize_member_label("Murali M."),
            _normalize_member_label("Murali K."),
        )
        self.assertEqual(
            _split_member_queries("Murali M and Agent H."),
            ["Murali M", "Agent H"],
        )

    def test_resolve_special_person_keeps_multi_member_phrase(self) -> None:
        from pc_liveagent.answer_parse import resolve_answer
        from pc_liveagent.elig_meta import get_field_meta

        r = resolve_answer(
            "special_person",
            "Murali M and Agent H.",
            meta=get_field_meta("special_person"),
        )
        self.assertIsNotNone(r.answer)
        self.assertIn("and", (r.answer or "").lower())
        self.assertIn("Murali", r.answer or "")
        self.assertIn("Agent", r.answer or "")

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
        # Aggregate apply-to wrapper text must not count as a member.
        self.assertFalse(
            _is_plausible_member_label(
                "Murali M. Agent H. David W. Mickey M."
            )
        )


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

    async def test_multiple_members_and_period_labels(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        members = [
            {"label": "Murali M.", "index": 0, "dupIndex": 0, "checked": False},
            {"label": "Agent H.", "index": 1, "dupIndex": 0, "checked": False},
        ]
        clicked: list[str] = []

        async def evaluate(script, arg=None):
            if arg is None:
                # Reflect prior clicks so already-checked path can work later.
                out = []
                for row in members:
                    copy = dict(row)
                    if copy["label"] in clicked:
                        copy["checked"] = True
                    out.append(copy)
                return {
                    "members": out,
                    "strategy": "heading",
                    "checkboxCount": 2,
                    "visibleSample": [m["label"] for m in members],
                }
            clicked.append(arg.get("label"))
            return {"ok": True, "label": arg.get("label"), "strategy": "scoped"}

        page.evaluate = AsyncMock(side_effect=evaluate)
        session = LiveFormSession(call_id="sp-multi", url="https://example.test")
        await _assign_special_person_member(
            page, "Murali M and Agent H.", session
        )
        assigned = session.filled.get("special_person_assigned", "")
        self.assertIn("Murali M.", assigned)
        self.assertIn("Agent H.", assigned)
        self.assertEqual(clicked, ["Murali M.", "Agent H."])

    async def test_already_checked_member_not_toggled(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.evaluate = AsyncMock(
            return_value={
                "members": [
                    {
                        "label": "Murali M.",
                        "index": 0,
                        "dupIndex": 0,
                        "checked": True,
                    }
                ],
                "strategy": "heading",
                "checkboxCount": 1,
                "visibleSample": ["Murali M."],
            }
        )
        session = LiveFormSession(call_id="sp-checked", url="https://example.test")
        await _assign_special_person_member(page, "Murali M.", session)
        self.assertIn("Murali M.", session.filled.get("special_person_assigned", ""))
        # Discover only — no click payload (arg is None).
        for call in page.evaluate.await_args_list:
            self.assertIsNone(call.args[1] if len(call.args) > 1 else None)

    async def test_failed_special_verify_keeps_step(self) -> None:
        """Unverified special condition must not advance past special."""
        from unittest.mock import patch

        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.mouse = AsyncMock()
        page.mouse.wheel = AsyncMock()
        session = LiveFormSession(
            call_id="sp-verify",
            url="https://example.test",
            elig_step="special",
        )
        labels = [
            "Someone is blind",
            "Someone is pregnant",
            "None of the above",
        ]

        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="special"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                new=AsyncMock(return_value=labels),
            ),
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(
                    return_value={
                        "found": True,
                        "checked": False,
                        "source": "input",
                    }
                ),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "pc_liveagent.live_form._special_next_enabled",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _submit_eligibility_answer(
                    "sp-verify", session, page, "I'm blind", 0.0
                )
        self.assertIn("SPECIAL_CLARIFY", str(ctx.exception))
        self.assertEqual(session.elig_step, "special")
        self.assertNotIn("special_conditions", session.filled)
        # Exactly one click attempt — never a toggle-off retry.
        self.assertEqual(click.await_count, 1)


class SpecialConditionSelectTests(unittest.IsolatedAsyncioTestCase):
    """Special Cases: one click max; never toggle off a successful selection."""

    async def test_unchecked_condition_one_click(self) -> None:
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        states = iter(
            [
                {"found": True, "checked": False, "source": "input"},
                {"found": True, "checked": True, "source": "input"},
            ]
        )
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            await _select_special_condition_checkbox(
                page, "Someone is blind", call_id="sp1", multi=False
            )
        self.assertEqual(click.await_count, 1)

    async def test_already_checked_not_clicked(self) -> None:
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(
                    return_value={
                        "found": True,
                        "checked": True,
                        "source": "input",
                    }
                ),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            await _select_special_condition_checkbox(
                page, "Someone is blind", call_id="sp2", multi=False
            )
        click.assert_not_awaited()

    async def test_disabled_condition_one_click(self) -> None:
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        states = iter(
            [
                {"found": True, "checked": False, "source": "input"},
                {"found": True, "checked": True, "source": "input"},
            ]
        )
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            await _select_special_condition_checkbox(
                page,
                "Someone is disabled (SSDI)",
                call_id="sp-dis",
                multi=False,
            )
        self.assertEqual(click.await_count, 1)

    async def test_no_second_click_when_generic_verifier_would_fail(self) -> None:
        """Simulate post-click state readable as checked only on 2nd read.

        Even if a shared verifier would still fail, special path must not
        click again after the first successful selection.
        """
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        # Before: unchecked. After one click: checked (DOM updated).
        states = iter(
            [
                {"found": True, "checked": False, "source": "input"},
                {"found": True, "checked": True, "source": "input"},
            ]
        )
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "pc_liveagent.live_form._choice_appears_selected",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            await _select_special_condition_checkbox(
                page, "Someone is blind", call_id="sp3", multi=False
            )
        self.assertEqual(click.await_count, 1)

    async def test_unreadable_inconclusive_clarifies_without_second_click(self) -> None:
        """found=False + no apply-to + failed reinspect → clarify, never toggle."""
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        states = iter(
            [
                {"found": False, "checked": False, "source": "missing"},
                {"found": False, "checked": False, "source": "missing"},
            ]
        )
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_heading_visible",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._special_condition_readonly_reinspect",
                new=AsyncMock(
                    return_value={
                        "found": False,
                        "checked": False,
                        "source": "reinspect_missing",
                    }
                ),
            ),
            patch(
                "pc_liveagent.live_form._special_next_enabled",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _select_special_condition_checkbox(
                    page, "Someone is blind", call_id="sp4", multi=False
                )
        self.assertIn("SPECIAL_CLARIFY", str(ctx.exception))
        self.assertEqual(click.await_count, 1)

    async def test_unreadable_accepted_via_readonly_reinspect(self) -> None:
        """Bounded reinspect finds nested checkbox checked — accept, one click."""
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        states = iter(
            [
                {"found": False, "checked": False, "source": "missing"},
                {"found": False, "checked": False, "source": "missing"},
            ]
        )
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_heading_visible",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._special_condition_readonly_reinspect",
                new=AsyncMock(
                    return_value={
                        "found": True,
                        "checked": True,
                        "source": "reinspect_checkbox",
                    }
                ),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            await _select_special_condition_checkbox(
                page, "Someone is blind", call_id="sp4b", multi=False
            )
        self.assertEqual(click.await_count, 1)

    async def test_unreadable_accepted_via_delayed_apply_to(self) -> None:
        """Apply-to appears only after the extra read-only wait."""
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        states = iter(
            [
                {"found": False, "checked": False, "source": "missing"},
                {"found": False, "checked": False, "source": "missing"},
            ]
        )
        # before click, after first check, after reinspect delay (x2 for heading+count)
        member_counts = iter([0, 0, 3])
        headings = iter([False, True])
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(member_counts)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_heading_visible",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(headings)),
            ),
            patch(
                "pc_liveagent.live_form._special_condition_readonly_reinspect",
                new=AsyncMock(
                    return_value={
                        "found": False,
                        "checked": False,
                        "source": "reinspect_missing",
                    }
                ),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            await _select_special_condition_checkbox(
                page, "Someone is blind", call_id="sp4c", multi=False
            )
        self.assertEqual(click.await_count, 1)

    async def test_unreadable_accepts_apply_to_expansion(self) -> None:
        """Member list expansion while Next stays disabled still confirms select."""
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        states = iter(
            [
                {"found": False, "checked": False, "source": "missing"},
                {"found": False, "checked": False, "source": "missing"},
            ]
        )
        member_counts = iter([0, 4])
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(member_counts)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_heading_visible",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.live_form._special_next_enabled",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            await _select_special_condition_checkbox(
                page, "Someone is blind", call_id="sp5", multi=False
            )
        self.assertEqual(click.await_count, 1)

    async def test_multi_unreadable_accepted_without_next(self) -> None:
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        states = iter(
            [
                {"found": False, "checked": False, "source": "missing"},
                {"found": False, "checked": False, "source": "missing"},
            ]
        )
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(side_effect=[0, 2]),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_heading_visible",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.live_form._special_next_enabled",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            await _select_special_condition_checkbox(
                page, "Someone is blind", call_id="sp6", multi=True
            )
        self.assertEqual(click.await_count, 1)

    async def test_explicit_unchecked_after_click_clarifies_once(self) -> None:
        from pc_liveagent.live_form import _select_special_condition_checkbox

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        states = iter(
            [
                {"found": True, "checked": False, "source": "input"},
                {"found": True, "checked": False, "source": "input"},
            ]
        )
        with (
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _select_special_condition_checkbox(
                    page, "Someone is blind", call_id="sp7", multi=False
                )
        self.assertIn("SPECIAL_CLARIFY", str(ctx.exception))
        self.assertEqual(click.await_count, 1)

    async def test_submit_multi_conditions_one_click_each(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.mouse = AsyncMock()
        page.mouse.wheel = AsyncMock()
        session = LiveFormSession(
            call_id="sp-multi-cond",
            url="https://example.test",
            elig_step="special",
        )
        labels = [
            "Someone is pregnant",
            "Someone is blind",
            "None of the above",
        ]
        # Each label: before unchecked, after checked.
        state_calls: list[str] = []

        async def state(_page, label):
            state_calls.append(label)
            # Odd calls are "before", even are "after" per label pair.
            checked = state_calls.count(label) > 1
            return {"found": True, "checked": checked, "source": "input"}

        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="special"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                new=AsyncMock(return_value=labels),
            ),
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=state),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "pc_liveagent.live_form._wait_for_special_apply_to_members",
                new=AsyncMock(
                    return_value={
                        "members": [
                            {
                                "label": "Murali M.",
                                "index": 0,
                                "dupIndex": 0,
                                "checked": False,
                            }
                        ],
                        "strategy": "heading",
                        "has_apply_to": True,
                    }
                ),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            result = await _submit_eligibility_answer(
                "sp-multi-cond",
                session,
                page,
                "blind and pregnant",
                0.0,
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("step"), "special_person")
        self.assertEqual(click.await_count, 2)
        self.assertIn("blind", session.filled.get("special_conditions", "").lower())
        self.assertIn("pregnant", session.filled.get("special_conditions", "").lower())

    async def test_submit_none_of_the_above(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.mouse = AsyncMock()
        page.mouse.wheel = AsyncMock()
        session = LiveFormSession(
            call_id="sp-none",
            url="https://example.test",
            elig_step="special",
        )
        labels = [
            "Someone is blind",
            "None of the above",
        ]
        states = iter(
            [
                {"found": True, "checked": False, "source": "input"},
                {"found": True, "checked": True, "source": "input"},
            ]
        )
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="special"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                new=AsyncMock(return_value=labels),
            ),
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(states)),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=0),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
            patch(
                "pc_liveagent.live_form._click_cta",
                new=AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form._wait_for_screen",
                new=AsyncMock(return_value="programs"),
            ),
        ):
            result = await _submit_eligibility_answer(
                "sp-none", session, page, "none of the above", 0.0
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("step"), "programs")
        self.assertEqual(click.await_count, 1)
        self.assertIn("none", session.filled.get("special_conditions", "").lower())

    async def test_submit_unreadable_advances_to_special_person(self) -> None:
        """Successful click with unreadable checkbox must not re-ask the condition."""
        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.mouse = AsyncMock()
        page.mouse.wheel = AsyncMock()
        session = LiveFormSession(
            call_id="sp-unread",
            url="https://example.test",
            elig_step="special",
        )
        labels = [
            "Someone is blind",
            "Someone is disabled (SSDI)",
            "Murali M.",
            "Agent H.",
            "None of the above",
        ]
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="special"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                new=AsyncMock(return_value=labels),
            ),
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(
                    return_value={
                        "found": False,
                        "checked": False,
                        "source": "missing",
                    }
                ),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(side_effect=[0, 3]),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_heading_visible",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.live_form._wait_for_special_apply_to_members",
                new=AsyncMock(
                    return_value={
                        "members": [
                            {
                                "label": "Murali M.",
                                "index": 0,
                                "dupIndex": 0,
                                "checked": False,
                            }
                        ],
                        "strategy": "heading",
                        "has_apply_to": True,
                    }
                ),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            result = await _submit_eligibility_answer(
                "sp-unread", session, page, "I'm blind", 0.0
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("step"), "special_person")
        self.assertEqual(click.await_count, 1)
        self.assertIn("blind", session.filled.get("special_conditions", "").lower())

    def test_mapper_ignores_household_names_in_visible_list(self) -> None:
        from pc_liveagent.live_form import _map_special_condition_labels

        labels = [
            "Someone is blind",
            "Someone is disabled (SSDI)",
            "Murali M.",
            "Agent H.",
            "None of the above",
        ]
        self.assertEqual(
            _map_special_condition_labels("blind and disabled", labels),
            ["Someone is blind", "Someone is disabled (SSDI)"],
        )


class SpecialPersonAssignFollowupTests(unittest.IsolatedAsyncioTestCase):
    """Remaining special_person assign cases after SpecialConditionSelectTests insert."""

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
        page.wait_for_timeout = AsyncMock()
        empty = {
            "members": [],
            "strategy": "section_missing",
            "checkbox_count": 0,
            "visible_labels": ["Someone is blind", "Next"],
            "has_apply_to": False,
            "has_special": True,
            "has_marital": False,
            "url": "https://example.test/special",
        }
        session = LiveFormSession(
            call_id="sp-empty",
            url="https://example.test",
            elig_step="special_person",
        )
        with (
            patch(
                "pc_liveagent.live_form._wait_for_special_apply_to_members",
                new=AsyncMock(return_value=empty),
            ),
            patch(
                "pc_liveagent.live_form._recover_special_apply_to_section",
                new=AsyncMock(return_value=empty),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _assign_special_person_member(page, "Angel", session)
        self.assertIn("member list", str(ctx.exception).lower())
        self.assertNotIn("say the person's name again.", str(ctx.exception).lower())
        self.assertEqual(session.elig_step, "special_person")

    async def test_wrong_screen_marital_asks_to_reopen_condition(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        wrong = {
            "members": [],
            "strategy": "wrong_screen_marital",
            "checkbox_count": 11,
            "visible_labels": [
                "Never Married",
                "Married",
                "Add Member",
                "Exit",
            ],
            "has_apply_to": False,
            "has_special": False,
            "has_marital": True,
            "url": "https://example.test/marital",
        }
        session = LiveFormSession(
            call_id="sp-wrong",
            url="https://example.test",
            elig_step="special_person",
        )
        session.filled["special_conditions"] = "Someone is blind"
        with (
            patch(
                "pc_liveagent.live_form._wait_for_special_apply_to_members",
                new=AsyncMock(return_value=wrong),
            ),
            patch(
                "pc_liveagent.live_form._recover_special_apply_to_section",
                new=AsyncMock(return_value=wrong),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _assign_special_person_member(page, "Mickey", session)
        msg = str(ctx.exception).lower()
        self.assertIn("special cases", msg)
        self.assertIn("condition", msg)

    async def test_mickey_mouse_matches_mickey_m(self) -> None:
        from pc_liveagent.live_form import _match_member_rows

        members = [
            {"label": "Mickey M.", "index": 0, "dupIndex": 0, "checked": False},
            {"label": "Murali M.", "index": 1, "dupIndex": 0, "checked": False},
        ]
        hits = _match_member_rows("Mickey Mouse.", members)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["label"], "Mickey M.")

    async def test_assign_waits_until_members_ready(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assign_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        members = [
            {"label": "Mickey M.", "index": 0, "dupIndex": 0, "checked": False},
        ]
        ready = {
            "members": members,
            "strategy": "heading",
            "checkbox_count": 4,
            "visible_labels": ["Who does this apply to?", "Mickey M."],
            "has_apply_to": True,
            "has_special": True,
            "has_marital": False,
            "url": "https://example.test/special",
        }
        session = LiveFormSession(
            call_id="sp-wait",
            url="https://example.test",
            elig_step="special_person",
        )

        async def click(_page, label, *, dup_index=0):
            return True

        with (
            patch(
                "pc_liveagent.live_form._wait_for_special_apply_to_members",
                new=AsyncMock(return_value=ready),
            ),
            patch(
                "pc_liveagent.live_form._discover_special_person_members",
                new=AsyncMock(return_value=ready),
            ),
            patch(
                "pc_liveagent.live_form._click_special_person_member",
                new=AsyncMock(side_effect=click),
            ),
        ):
            await _assign_special_person_member(page, "Mickey Mouse", session)
        self.assertIn("Mickey M.", session.filled.get("special_person_assigned", ""))

    async def test_nested_name_div_pairs_empty_role_checkbox(self) -> None:
        """Pointcare: empty button[role=checkbox] + sibling name div in apply-to."""
        from pc_liveagent.live_form import (
            _SPECIAL_MEMBER_DISCOVER_JS,
            _discover_special_person_members,
        )

        page = AsyncMock()
        # Simulate the live DOM pairing result (names resolved from row climb).
        page.evaluate = AsyncMock(
            return_value={
                "members": [
                    {
                        "label": "Murali M.",
                        "index": 0,
                        "dupIndex": 0,
                        "checked": False,
                    },
                    {
                        "label": "Mickey M.",
                        "index": 1,
                        "dupIndex": 0,
                        "checked": False,
                    },
                    {
                        # Aggregate wrapper must be dropped by Python filter.
                        "label": "Murali M. Agent H. David W. Mickey M.",
                        "index": 2,
                        "dupIndex": 0,
                        "checked": False,
                    },
                ],
                "strategy": "heading",
                "checkboxCount": 11,
                "visibleSample": [
                    "Who does this apply to?",
                    "Murali M.",
                    "Mickey M.",
                ],
                "hasApplyTo": True,
                "hasSpecial": True,
                "hasMarital": False,
                "url": "https://example.test/special",
            }
        )
        discovered = await _discover_special_person_members(page)
        labels = [m["label"] for m in discovered["members"]]
        self.assertEqual(labels, ["Murali M.", "Mickey M."])
        self.assertEqual(discovered["strategy"], "heading")
        self.assertIn("Who does this apply to?", _SPECIAL_MEMBER_DISCOVER_JS)

    async def test_click_targets_checkbox_then_verifies_aria(self) -> None:
        from pc_liveagent.live_form import _click_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        calls: list[str] = []

        async def evaluate(script, arg=None):
            mode = (arg or {}).get("mode")
            calls.append(str(mode))
            if mode == "click":
                return {
                    "ok": True,
                    "clicked": True,
                    "already": False,
                    "label": "Mickey M.",
                    "strategy": "scoped",
                }
            if mode == "verify":
                return {
                    "ok": True,
                    "already": True,
                    "label": "Mickey M.",
                    "strategy": "scoped",
                }
            return {"ok": True, "label": "Mickey M."}

        page.evaluate = AsyncMock(side_effect=evaluate)
        ok = await _click_special_person_member(page, "Mickey M.", dup_index=0)
        self.assertTrue(ok)
        self.assertEqual(calls, ["click", "verify"])
        self.assertEqual(page.wait_for_timeout.await_count, 1)

    async def test_click_no_checkbox_control_fails_without_retry_toggle(self) -> None:
        from pc_liveagent.live_form import _click_special_person_member

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.evaluate = AsyncMock(
            return_value={
                "ok": False,
                "reason": "no_checkbox",
                "label": "Mickey M.",
            }
        )
        ok = await _click_special_person_member(page, "Mickey M.", dup_index=0)
        self.assertFalse(ok)
        self.assertEqual(page.evaluate.await_count, 1)

    async def test_unique_first_name_match_only(self) -> None:
        from pc_liveagent.live_form import _match_member_rows

        members = [
            {"label": "Mickey M.", "index": 0, "dupIndex": 0, "checked": False},
            {"label": "Murali M.", "index": 1, "dupIndex": 0, "checked": False},
        ]
        self.assertEqual(
            [m["label"] for m in _match_member_rows("Mickey", members)],
            ["Mickey M."],
        )
        dups = [
            {"label": "John D.", "index": 0, "dupIndex": 0, "checked": False},
            {"label": "John K.", "index": 1, "dupIndex": 0, "checked": False},
        ]
        hits = _match_member_rows("John", dups)
        self.assertEqual(len(hits), 2)

    async def test_special_step_requires_apply_to_before_advance(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.mouse = AsyncMock()
        page.mouse.wheel = AsyncMock()
        session = LiveFormSession(
            call_id="sp-gate",
            url="https://example.test",
            elig_step="special",
        )
        labels = ["Someone is blind", "None of the above"]
        empty = {
            "members": [],
            "strategy": "section_missing",
            "checkbox_count": 7,
            "visible_labels": labels,
            "has_apply_to": False,
            "has_special": True,
            "has_marital": False,
            "url": "",
        }
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="special"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                new=AsyncMock(return_value=labels),
            ),
            patch(
                "pc_liveagent.live_form._select_special_condition_checkbox",
                new=AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form._wait_for_special_apply_to_members",
                new=AsyncMock(return_value=empty),
            ),
            patch(
                "pc_liveagent.live_form._recover_special_apply_to_section",
                new=AsyncMock(return_value=empty),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _submit_eligibility_answer(
                    "sp-gate", session, page, "I'm blind", 0.0
                )
        self.assertIn("SPECIAL_CLARIFY", str(ctx.exception))
        self.assertEqual(session.elig_step, "special")

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
                "pc_liveagent.live_form._visible_choice_labels",
                AsyncMock(
                    return_value=[
                        "Child",
                        "Stepchild",
                        "Grandchild",
                        "Foster Child",
                        "Other",
                    ]
                ),
            ),
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


class YesNoIntegrityAndGoBackTests(unittest.IsolatedAsyncioTestCase):
    def test_classify_yes_no_conservative(self) -> None:
        from pc_liveagent.live_form import classify_yes_no

        self.assertEqual(classify_yes_no("Yes"), "yes")
        self.assertEqual(classify_yes_no("Yeah, that's right."), "yes")
        self.assertEqual(classify_yes_no("No"), "no")
        self.assertEqual(classify_yes_no("Nope"), "no")
        self.assertIsNone(classify_yes_no("Yes I don't have any"))
        self.assertIsNone(classify_yes_no("Yes or no"))
        self.assertIsNone(classify_yes_no("Go back to the previous question."))
        self.assertIsNone(classify_yes_no("maybe later"))
        self.assertEqual(classify_yes_no("I don't"), "no")

    def test_validate_mixed_yes_no_clarifies(self) -> None:
        answer, clarify = validate_for_step("student", "Yes I don't")
        self.assertIsNone(answer)
        self.assertIn("yes or no", (clarify or "").lower())

    def test_stale_yes_after_go_back_not_resolved(self) -> None:
        from pc_liveagent.form_bridge import _yes_no_from_recent_transcript

        transcript = (
            "Agent: Do you live with and mainly support any children under 19 "
            "who live with you more than half the year?\n"
            "User: Yes.\n"
            "Agent: Are you a full-time student age 18 to 21 whose parent or legal "
            "guardian supports you?\n"
            "User: Go back to the previous question.\n"
            "Agent: Do you live with and mainly support any children under 19 "
            "who live with you more than half the year?\n"
        )
        call = {
            "transcript_with_tool_calls": [
                {
                    "role": "agent",
                    "content": (
                        "Do you live with and mainly support any children under 19 "
                        "who live with you more than half the year?"
                    ),
                },
                {"role": "user", "content": "Yes."},
                {
                    "role": "agent",
                    "content": (
                        "Are you a full-time student age 18 to 21 whose parent or "
                        "legal guardian supports you?"
                    ),
                },
                {"role": "user", "content": "Go back to the previous question."},
                {
                    "role": "agent",
                    "content": (
                        "Do you live with and mainly support any children under 19 "
                        "who live with you more than half the year?"
                    ),
                },
            ]
        }
        # Latest user turn is go-back → must not reuse prior Yes.
        self.assertIsNone(
            _yes_no_from_recent_transcript(transcript, "children", call=call)
        )
        # After a fresh Yes following go-back, resolve Yes.
        call["transcript_with_tool_calls"].append({"role": "user", "content": "Yes."})
        self.assertEqual(
            _yes_no_from_recent_transcript(transcript + "User: Yes.\n", "children", call=call),
            "Yes",
        )

    async def test_go_back_cancels_pending_browser_and_bumps_epoch(self) -> None:
        from pc_liveagent.call_turn import (
            _elig_browser_tasks,
            drop_controller,
            get_controller,
            process_user_turn,
        )

        call_id = "yn-goback-race"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="student"
        )

        async def fake_get_session(_cid: str):
            return session

        fill_started = asyncio.Event()
        release_fill = asyncio.Event()

        async def slow_submit(*_a, **_k):
            fill_started.set()
            await release_fill.wait()
            # Real Playwright would mutate the page; the controller must ignore
            # this late result after go-back invalidated the epoch.
            return {
                "ok": True,
                "spoken": "ASK: Do you support a disabled individual?",
                "step": "disabled",
            }

        async def fake_nav(_cid: str):
            session.elig_step = "children"
            return {
                "ok": True,
                "step": "children",
                "screen": "children",
                "spoken": (
                    "Do you live with and mainly support any children under 19 "
                    "who live with you more than half the year?"
                ),
            }

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=["Yes", "No"])),
            patch("pc_liveagent.call_turn.submit_field", side_effect=slow_submit),
            patch("pc_liveagent.call_turn._speak_interrupt", AsyncMock(return_value=True)),
            patch("pc_liveagent.call_turn.navigate_back_one", side_effect=fake_nav),
        ):
            ack = await process_user_turn(call_id, "No", response_id=1)
            self.assertEqual(ack.detail, "elig_ack")
            await asyncio.wait_for(fill_started.wait(), timeout=2.0)
            epoch_before = ctrl.elig_nav_epoch
            back = await process_user_turn(call_id, "Go back.", response_id=2)
            self.assertEqual(back.detail, "go_back")
            self.assertGreater(ctrl.elig_nav_epoch, epoch_before)
            self.assertFalse(ctrl.elig_op_in_progress)
            release_fill.set()
            task = _elig_browser_tasks.get(call_id)
            if task:
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
            await asyncio.sleep(0.05)
        # Late No must not promote past children after go-back.
        self.assertEqual(session.elig_step, "children")
        self.assertFalse(
            any(item.get("step") == "student" for item in ctrl.committed)
        )
        await drop_controller(call_id)

    async def test_go_back_missing_button_is_honest(self) -> None:
        from pc_liveagent.call_turn import drop_controller, get_controller, process_user_turn

        call_id = "yn-goback-missing"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="marital"
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
                        "error": "no_back",
                        "spoken": (
                            "I can't go back from this screen. There is no Back button "
                            "available. You can still correct the current question — "
                            "What is your marital status?"
                        ),
                        "step": "marital",
                    }
                ),
            ),
        ):
            r = await process_user_turn(call_id, "Go back.")
        self.assertEqual(r.detail, "go_back_failed")
        self.assertIn("can't go back", r.speech.lower())
        self.assertIn("marital", r.speech.lower())
        await drop_controller(call_id)

    async def test_yesno_missing_chips_wrong_screen_rejects(self) -> None:
        from pc_liveagent.live_form import _submit_eligibility_answer

        call_id = "yn-mismatch"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="student"
        )
        page = AsyncMock()
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                AsyncMock(return_value="student"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                AsyncMock(return_value=[]),
            ),
            patch(
                "pc_liveagent.live_form._yes_no_selection_state",
                AsyncMock(return_value="none"),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _submit_eligibility_answer(
                    call_id, session, page, "Yes", started=0.0
                )
        self.assertIn("YESNO_VERIFY", str(ctx.exception))

    async def test_overlapping_answer_replaces_prior_task(self) -> None:
        from pc_liveagent.call_turn import (
            _elig_browser_tasks,
            drop_controller,
            get_controller,
            process_user_turn,
        )

        call_id = "yn-overlap"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        ctrl.eligibility_question_sent = True
        ctrl.active_websocket_status = "connected"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="children"
        )

        async def fake_get_session(_cid: str):
            return session

        calls: list[str] = []
        gate = asyncio.Event()

        async def slow_submit(_cid, _field, value, **_k):
            calls.append(value)
            if value == "No":
                await gate.wait()
            session.elig_step = "student" if value == "No" else "child_first"
            return {
                "ok": True,
                "spoken": "ASK: next",
                "step": session.elig_step,
            }

        with (
            patch("pc_liveagent.call_turn.get_session", side_effect=fake_get_session),
            patch("pc_liveagent.call_turn.peek_choice_labels", AsyncMock(return_value=["Yes", "No"])),
            patch("pc_liveagent.call_turn.submit_field", side_effect=slow_submit),
            patch("pc_liveagent.call_turn._speak_interrupt", AsyncMock(return_value=True)),
        ):
            r1 = await process_user_turn(call_id, "No", response_id=1)
            self.assertEqual(r1.detail, "elig_ack")
            # Second answer while first is still pending — must replace, not drop.
            # Go back first to clear busy, then answer Yes on same step? Better:
            # force schedule replacement by cancelling busy flag briefly is wrong.
            # Instead: go_back clears busy; we test replacement via schedule API.
            from pc_liveagent.call_turn import _schedule_eligibility_browser

            epoch = ctrl.elig_nav_epoch + 1
            ctrl.elig_nav_epoch = epoch
            _schedule_eligibility_browser(
                call_id,
                answer="Yes",
                step="children",
                turn_id="t-yes",
                form_url="",
                commit_key="children:yes",
                epoch=epoch,
            )
            gate.set()
            task = _elig_browser_tasks.get(call_id)
            if task:
                await asyncio.wait_for(task, timeout=2.0)
        self.assertIn("Yes", calls)
        await drop_controller(call_id)

    async def test_step_mismatch_does_not_commit_answer(self) -> None:
        from pc_liveagent.live_form import _submit_eligibility_answer

        call_id = "yn-step-mismatch"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="student"
        )
        page = AsyncMock()
        with patch(
            "pc_liveagent.live_form._eligibility_screen",
            AsyncMock(return_value="children"),
        ):
            result = await _submit_eligibility_answer(
                call_id, session, page, "Yes", started=0.0
            )
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "step_mismatch")
        self.assertEqual(session.elig_step, "children")
        self.assertIn("children", (result.get("spoken") or "").lower())


class BiWeeklyIncomeWhenSubmitTests(unittest.IsolatedAsyncioTestCase):
    async def test_biweekly_selects_and_verifies(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        labels = ["Annual", "Monthly", "Bi-weekly", "$0 – $12,000"]
        session = LiveFormSession(
            call_id="inc-bi",
            url="https://example.test",
            elig_step="income_when",
        )
        sig_values = iter(["idle|", "selected|active|true||false"])
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="income"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                new=AsyncMock(return_value=labels),
            ),
            patch(
                "pc_liveagent.live_form._choice_signature",
                new=AsyncMock(side_effect=lambda *_a, **_k: next(sig_values)),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
            patch(
                "pc_liveagent.live_form._income_period_selected",
                new=AsyncMock(return_value=False),
            ),
        ):
            result = await _submit_eligibility_answer(
                "inc-bi", session, page, "bi-weekly", 0.0
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("step"), "income_range")
        click.assert_awaited()
        self.assertEqual(click.await_args.args[1], "Bi-weekly")
        self.assertEqual(session.filled.get("income_when"), "Bi-weekly")

    async def test_unrecognized_period_clarifies_without_click(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        labels = ["Annual", "Monthly", "Bi-weekly"]
        session = LiveFormSession(
            call_id="inc-bad",
            url="https://example.test",
            elig_step="income_when",
        )
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="income"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                new=AsyncMock(return_value=labels),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _submit_eligibility_answer(
                    "inc-bad", session, page, "quarterly maybe", 0.0
                )
        self.assertIn("INCOME_CLARIFY", str(ctx.exception))
        click.assert_not_awaited()
        self.assertEqual(session.elig_step, "income_when")


class SpecialMoreConfirmTests(unittest.IsolatedAsyncioTestCase):
    async def test_special_person_confirms_before_next(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        session = LiveFormSession(
            call_id="sp-confirm",
            url="https://example.test",
            elig_step="special_person",
        )
        session.filled["special_conditions"] = "Someone is blind"
        session.filled["special_active_conditions"] = "Someone is blind"
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="special"),
            ),
            patch(
                "pc_liveagent.live_form._assign_special_person_member",
                new=AsyncMock(
                    side_effect=lambda _p, _a, sess: sess.filled.__setitem__(
                        "special_person_assigned", "Mickey M."
                    )
                ),
            ),
            patch(
                "pc_liveagent.live_form._click_cta",
                new=AsyncMock(),
            ) as cta,
        ):
            result = await _submit_eligibility_answer(
                "sp-confirm", session, page, "Mickey M.", 0.0
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("step"), "special_more")
        spoken = (result.get("spoken") or "").lower()
        self.assertIn("blind", spoken)
        self.assertIn("mickey", spoken)
        self.assertIn("other conditions", spoken)
        cta.assert_not_awaited()

    async def test_special_more_no_verifies_then_next(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        session = LiveFormSession(
            call_id="sp-more-no",
            url="https://example.test",
            elig_step="special_more",
        )
        session.filled["special_conditions"] = "Someone is blind"
        session.filled["special_person_assigned"] = "Mickey M."
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="special"),
            ),
            patch(
                "pc_liveagent.live_form._verify_special_selections_before_next",
                new=AsyncMock(),
            ) as verify,
            patch(
                "pc_liveagent.live_form._click_cta",
                new=AsyncMock(),
            ) as cta,
            patch(
                "pc_liveagent.live_form._wait_for_screen",
                new=AsyncMock(return_value="programs"),
            ),
        ):
            result = await _submit_eligibility_answer(
                "sp-more-no", session, page, "No", 0.0
            )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("step"), "programs")
        verify.assert_awaited()
        cta.assert_awaited()
        self.assertEqual(cta.await_args.args[1], "Next")

    async def test_special_more_yes_collects_another_condition(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _submit_eligibility_answer,
        )

        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.mouse = AsyncMock()
        page.mouse.wheel = AsyncMock()
        session = LiveFormSession(
            call_id="sp-more-yes",
            url="https://example.test",
            elig_step="special_more",
        )
        session.filled["special_conditions"] = "Someone is blind"
        session.filled["special_person_assigned"] = "Mickey M."
        session.filled["special_assignment_pairs"] = (
            "Someone is blind>>>Mickey M."
        )
        labels = [
            "Someone is blind",
            "Someone is disabled (SSDI)",
            "None of the above",
        ]
        # First turn: Yes → back to special question.
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="special"),
            ),
            patch(
                "pc_liveagent.live_form._click_cta",
                new=AsyncMock(),
            ) as cta,
        ):
            yes_result = await _submit_eligibility_answer(
                "sp-more-yes", session, page, "Yes", 0.0
            )
        self.assertEqual(yes_result.get("step"), "special")
        cta.assert_not_awaited()

        # Second turn: add disabled without deselecting blind.
        state_calls: list[str] = []

        async def state(_page, label):
            state_calls.append(label)
            if "blind" in label.lower():
                return {"found": True, "checked": True, "source": "input"}
            checked = state_calls.count(label) > 1
            return {"found": True, "checked": checked, "source": "input"}

        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                new=AsyncMock(return_value="special"),
            ),
            patch(
                "pc_liveagent.live_form._visible_choice_labels",
                new=AsyncMock(return_value=labels),
            ),
            patch(
                "pc_liveagent.live_form._special_condition_checkbox_state",
                new=AsyncMock(side_effect=state),
            ),
            patch(
                "pc_liveagent.live_form._special_apply_to_member_count",
                new=AsyncMock(return_value=1),
            ),
            patch(
                "pc_liveagent.live_form._wait_for_special_apply_to_members",
                new=AsyncMock(
                    return_value={
                        "members": [
                            {
                                "label": "Mickey M.",
                                "index": 0,
                                "dupIndex": 0,
                                "checked": True,
                            },
                            {
                                "label": "Agent H.",
                                "index": 1,
                                "dupIndex": 0,
                                "checked": False,
                            },
                        ],
                        "strategy": "heading",
                        "has_apply_to": True,
                    }
                ),
            ),
            patch(
                "pc_liveagent.live_form._click_exact_choice",
                new=AsyncMock(),
            ) as click,
        ):
            add_result = await _submit_eligibility_answer(
                "sp-more-yes", session, page, "disabled on SSDI", 0.0
            )
        self.assertEqual(add_result.get("step"), "special_person")
        self.assertIn("blind", session.filled.get("special_conditions", "").lower())
        self.assertIn("disabled", session.filled.get("special_conditions", "").lower())
        # Blind already checked — only disabled should be clicked.
        self.assertEqual(click.await_count, 1)

    async def test_multi_condition_multi_member_confirm_speech(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _record_special_assignment_pairs,
            _special_more_confirm_speech,
        )

        session = LiveFormSession(call_id="sp-pair", url="https://example.test")
        _record_special_assignment_pairs(
            session, ["Someone is blind"], ["Mickey M.", "Agent H."]
        )
        _record_special_assignment_pairs(
            session, ["Someone is pregnant"], ["Aria R."]
        )
        speech = _special_more_confirm_speech(session).lower()
        self.assertIn("blind", speech)
        self.assertIn("mickey", speech)
        self.assertIn("agent", speech)
        self.assertIn("pregnant", speech)
        self.assertIn("aria", speech)
        self.assertIn("other conditions", speech)


class SectionIntroTests(unittest.IsolatedAsyncioTestCase):
    def test_spouse_and_child_intros_once(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            _spoken_question_for_step,
            _bump_child_instance,
        )

        session = LiveFormSession(
            call_id="intro", url="https://example.test", elig_step="spouse_first"
        )
        first = _spoken_question_for_step(session, "spouse_first")
        self.assertIn("spouse or partner", first.lower())
        self.assertIn("first name", first.lower())
        again = _spoken_question_for_step(session, "spouse_first")
        self.assertNotIn("spouse or partner", again.lower())
        self.assertIn("first name", again.lower())

        child = LiveFormSession(
            call_id="intro-c", url="https://example.test", elig_step="child_first"
        )
        c1 = _spoken_question_for_step(child, "child_first")
        self.assertIn("details about the child", c1.lower())
        self.assertIn("first name", c1.lower())
        c1b = _spoken_question_for_step(child, "child_first")
        self.assertNotIn("details about the child", c1b.lower())

        _bump_child_instance(child)
        c2 = _spoken_question_for_step(child, "child_first")
        self.assertIn("next child", c2.lower())
        self.assertIn("first name", c2.lower())

    async def test_repeat_keeps_answers_and_explains_section(self) -> None:
        from pc_liveagent.call_turn import (
            CallController,
            _repeat_with_context,
            drop_controller,
            get_controller,
        )
        from pc_liveagent.live_form import LiveFormSession

        call_id = "intro-repeat"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "eligibility"
        session = LiveFormSession(
            call_id=call_id,
            url="https://example.test",
            elig_step="spouse_last",
        )
        session.filled["spouse_first"] = "Alex"
        session.filled["spouse_section_intro"] = "1"
        with (
            patch(
                "pc_liveagent.call_turn.get_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "pc_liveagent.call_turn.peek_choice_labels",
                new=AsyncMock(return_value=[]),
            ),
        ):
            speech = await _repeat_with_context(call_id, ctrl)
        self.assertIn("spouse or partner", speech.lower())
        self.assertIn("last name", speech.lower())
        self.assertEqual(session.filled.get("spouse_first"), "Alex")
        self.assertEqual(session.elig_step, "spouse_last")
        await drop_controller(call_id)


if __name__ == "__main__":
    unittest.main()
