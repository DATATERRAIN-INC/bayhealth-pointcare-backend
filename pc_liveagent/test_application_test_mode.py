"""Application-only test mode. Mocks only — no Chrome, calls, or submissions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pc_liveagent.live_form import LiveFormSession


DASHBOARD = (
    "Next steps Household members Coverage unknown Recheck Eligibility "
    "Add Member Quick Apply"
)
PROGRAMS = (
    "Great News! MediKids / Florida Healthy Kids (Medicaid) may be eligible "
    "Medicare not eligible at this time ACA Marketplace not eligible at this time "
    "Start Applications"
)
APPLICATION = "Application Eligibility Are all household members U.S. citizens?"


class _Body:
    def __init__(self, page: "FakePage") -> None:
        self.page = page

    async def inner_text(self, timeout: int | None = None) -> str:
        return self.page.body


class FakePage:
    def __init__(self, body: str, url: str = "https://my.pointcare.com/home") -> None:
        self.body = body
        self.url = url
        self.clicked: list[str] = []

    def locator(self, _selector: str) -> _Body:
        return _Body(self)

    async def evaluate(self, _script: str, arg: str | None = None) -> bool:
        self.clicked.append(arg or "")
        return True

    async def wait_for_timeout(self, _ms: int) -> None:
        return None

    async def goto(self, url: str, **_kwargs) -> None:
        self.url = url


class ApplicationTestModeFlagTests(unittest.TestCase):
    def test_disabled_by_default_and_blocked_in_production(self) -> None:
        from pc_liveagent.config import application_test_mode_from_env

        self.assertFalse(application_test_mode_from_env({}))
        self.assertFalse(
            application_test_mode_from_env(
                {"POINTCARE_APPLICATION_TEST_MODE": "false"}
            )
        )
        self.assertTrue(
            application_test_mode_from_env(
                {"POINTCARE_APPLICATION_TEST_MODE": "true"}
            )
        )
        self.assertFalse(
            application_test_mode_from_env(
                {
                    "POINTCARE_APPLICATION_TEST_MODE": "true",
                    "APP_ENV": "production",
                }
            )
        )
        self.assertTrue(
            application_test_mode_from_env(
                {
                    "POINTCARE_APPLICATION_TEST_MODE": "true",
                    "APP_ENV": "production",
                    "POINTCARE_APPLICATION_TEST_MODE_ALLOW_PROD": "true",
                }
            )
        )


class ChromeProfileRelaunchTests(unittest.TestCase):
    def test_reopen_uses_configured_profile_and_localhost_port(self) -> None:
        from pc_liveagent import browser as browser_mod

        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "PointcareAgentChrome"
            with (
                patch.object(browser_mod, "_chrome_executable", return_value="chrome.exe"),
                patch.object(browser_mod.subprocess, "Popen") as popen,
            ):
                browser_mod._launch_chrome(profile, "Profile 2", "https://my.pointcare.com/onboarding")
            args = popen.call_args.args[0]
        joined = " ".join(args)
        self.assertIn("--remote-debugging-address=127.0.0.1", joined)
        self.assertIn(f"--user-data-dir={profile}", joined)
        self.assertIn("--profile-directory=Profile 2", joined)
        self.assertIn("--start-maximized", joined)
        self.assertNotIn("User Data", joined)


class SavedHouseholdNavigationTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_asks_before_clicking_quick_apply(self) -> None:
        from pc_liveagent.app_stage import prepare_application_test

        page = FakePage(
            "Next steps Continue Quick Apply New Quick Apply Recheck Eligibility "
            "Add Member Household",
            url="https://my.pointcare.com/",
        )
        session = LiveFormSession(call_id="dash", url=page.url)
        spoken = await prepare_application_test(session, page)
        self.assertEqual(page.clicked, [])
        self.assertEqual(session.elig_step, "proceed_application")
        self.assertEqual(session.filled.get("awaiting_quick_apply"), "1")
        self.assertEqual(session.filled.get("application_started"), "0")
        self.assertIn("eligibility check is finished", spoken.lower())
        self.assertIn("can i start the application", spoken.lower())
        self.assertNotIn("APPLICATION_TEST", spoken)

    async def test_continue_quick_apply_preferred_over_new(self) -> None:
        from pc_liveagent.app_stage import click_quick_apply

        page = FakePage(
            "Next steps Continue Quick Apply New Quick Apply Recheck Eligibility",
            url="https://my.pointcare.com/",
        )

        async def _eval(script: str, arg: str | None = None):
            # Exact-label path tries Continue first via click_exact_visible_text.
            if arg == "Continue Quick Apply":
                page.clicked.append(arg)
                return True
            if arg in ("Quick Apply", "New Quick Apply"):
                return False
            page.clicked.append("dom-fallback")
            return "continue quick apply"

        page.evaluate = _eval  # type: ignore[method-assign]
        label = await click_quick_apply(page)
        self.assertEqual(label, "Continue Quick Apply")
        self.assertEqual(page.clicked, ["Continue Quick Apply"])

    async def test_already_open_program_summary_does_not_click_quick_apply(self) -> None:
        from pc_liveagent.app_stage import prepare_application_test

        page = FakePage(PROGRAMS)

        async def _store(session, _page) -> None:
            session.filled["programs_permission_speech"] = (
                "Your household eligibility check is complete. "
                "Would you like to continue and start the application?"
            )

        session = LiveFormSession(call_id="news", url=page.url)
        with patch("pc_liveagent.app_stage.store_program_summary", new=_store):
            await prepare_application_test(session, page)
        self.assertEqual(page.clicked, [])
        self.assertEqual(session.elig_step, "programs")

    async def test_continue_to_applications_great_news_is_programs_surface(self) -> None:
        from pc_liveagent.app_stage import (
            classify_saved_household_page,
            prepare_application_test,
        )

        page = FakePage(
            "Great News! We found 2 eligible programs. "
            "Mickey may be eligible for MediKids / Florida Healthy Kids. "
            "Not eligible at this time. Partner may be eligible for ACA Marketplace. "
            "Applying to 2 programs for 7 members. Continue to Applications "
            "Return to Dashboard",
            url="https://my.pointcare.com/programs",
        )
        self.assertEqual(await classify_saved_household_page(page), "programs")

        async def _store(session, _page) -> None:
            session.filled["programs_permission_speech"] = (
                "Your household eligibility check is complete. "
                "Would you like to continue and start the application?"
            )

        session = LiveFormSession(call_id="continue-cta", url=page.url)
        with patch("pc_liveagent.app_stage.store_program_summary", new=_store):
            spoken = await prepare_application_test(session, page)
        self.assertEqual(page.clicked, [])
        self.assertEqual(session.elig_step, "programs")
        self.assertIn("continue and start the application", spoken.lower())

    async def test_in_progress_application_restarts_at_question_one(self) -> None:
        from pc_liveagent.app_stage import prepare_application_test

        page = FakePage(APPLICATION, url="https://my.pointcare.com/application/step-1")
        session = LiveFormSession(call_id="restart-q1", url=page.url)
        with patch(
            "pc_liveagent.app_stage.restart_application_to_first_question",
            new=AsyncMock(return_value="app_citizenship"),
        ) as restart:
            spoken = await prepare_application_test(session, page)
        restart.assert_awaited_once()
        self.assertEqual(session.app_step, "app_citizenship")
        self.assertEqual(session.filled.get("application_started"), "1")
        self.assertIn("first question", spoken.lower())
        self.assertIn("u.s. citizen", spoken.lower())
        self.assertEqual(session.elig_step, "")

    async def test_missing_screen_is_a_diagnostic_not_a_fake_session(self) -> None:
        from pc_liveagent.app_stage import prepare_application_test

        page = FakePage("Welcome to Example. Nothing to apply.", url="https://example.test/empty")
        session = LiveFormSession(call_id="missing", url=page.url)
        with self.assertRaises(RuntimeError) as ctx:
            await prepare_application_test(session, page)
        self.assertIn("APPLICATION_TEST", str(ctx.exception))
        self.assertEqual(session.elig_step, "")
        self.assertEqual(session.app_step, "")
        self.assertEqual(page.clicked, [])

    def test_program_names_keep_medikids(self) -> None:
        from pc_liveagent.app_stage import build_programs_permission_speech

        speech = build_programs_permission_speech(
            {
                "programs": [
                    {
                        "name": "MediKids / Florida Healthy Kids (Medicaid)",
                        "may_be_eligible": True,
                        "detail": "Mickey may be eligible",
                    },
                    {
                        "name": "Medicare",
                        "may_be_eligible": False,
                        "detail": "Not eligible at this time",
                    },
                ]
            }
        )
        low = speech.lower()
        self.assertIn("medikids", low)
        self.assertIn("would you like to continue and start the application", low)
        # Do not announce not-eligible programs.
        self.assertNotIn("medicare", low)
        self.assertNotIn("not eligible", low)
        self.assertNotIn("approved", low)


class PermissionDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_yes_clicks_start_applications_once(self) -> None:
        from pc_liveagent.app_stage import handle_programs_decision

        session = LiveFormSession(call_id="yes", url="https://example.test", elig_step="programs")
        session.filled["program_summary_json"] = '{"programs":[{"name":"MediKids"}]}'
        page = FakePage(PROGRAMS)
        start = AsyncMock(return_value="Start Applications")
        with (
            patch("pc_liveagent.app_stage.click_start_applications", start),
            patch(
                "pc_liveagent.app_stage.wait_for_application_started",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.app_stage.detect_application_screen",
                new=AsyncMock(return_value="app_citizenship"),
            ),
        ):
            first = await handle_programs_decision("yes", session, page, "Yes")
            second = await handle_programs_decision("yes", session, page, "Yes")
        start.assert_awaited_once()
        self.assertEqual(first.get("phase"), "application")
        self.assertEqual(second.get("step"), "app_citizenship")
        self.assertEqual(session.filled.get("program_summary_json"), '{"programs":[{"name":"MediKids"}]}')

    async def test_no_leaves_eligibility_and_never_starts(self) -> None:
        from pc_liveagent.app_stage import handle_programs_decision

        session = LiveFormSession(call_id="no", url="https://example.test", elig_step="programs")
        session.filled["program_summary_json"] = '{"programs":[{"name":"MediKids"}]}'
        page = FakePage(PROGRAMS)
        start = AsyncMock(side_effect=AssertionError("must not start"))
        with patch("pc_liveagent.app_stage.click_start_applications", start):
            result = await handle_programs_decision("no", session, page, "No")
        start.assert_not_awaited()
        self.assertTrue(result.get("end_call"))
        self.assertEqual(session.filled.get("application_started"), "0")
        self.assertEqual(
            session.filled.get("program_summary_json"),
            '{"programs":[{"name":"MediKids"}]}',
        )
        self.assertIn("have not started", str(result.get("spoken") or "").lower())

    async def test_ambiguous_does_not_click(self) -> None:
        from pc_liveagent.app_stage import handle_programs_decision

        session = LiveFormSession(call_id="maybe", url="https://example.test", elig_step="programs")
        page = FakePage(PROGRAMS)
        start = AsyncMock(side_effect=AssertionError("must not start"))
        with patch("pc_liveagent.app_stage.click_start_applications", start):
            result = await handle_programs_decision("maybe", session, page, "maybe later")
        start.assert_not_awaited()
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("step"), "programs")


class StartupPathTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        from pc_liveagent.live_form import end_live_form

        for call_id in (
            "test-on",
            "test-off",
            "test-logged-out",
            "test-reopen",
            "test-wins",
        ):
            await end_live_form(call_id)

    async def test_enabled_authenticated_skips_logout_and_stage_d(self) -> None:
        from pc_liveagent.live_form import start_live_form

        page = FakePage(DASHBOARD, url="https://my.pointcare.com/home")
        logout = AsyncMock()
        welcome = AsyncMock(return_value=["Get Started"])
        prepare = AsyncMock(return_value="ASK: Your household eligibility check is complete.")
        begin_elig = AsyncMock(side_effect=AssertionError("stage d must not open"))
        reset = AsyncMock(side_effect=AssertionError("must not reset eligibility"))

        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", True),
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", False),
            patch("pc_liveagent.live_form.open_page", new=AsyncMock(return_value=page)),
            patch("pc_liveagent.live_form._logout_for_fresh_signin", logout),
            patch("pc_liveagent.live_form._page_wait", new=AsyncMock()),
            patch(
                "pc_liveagent.live_form._application_test_authenticated",
                new=AsyncMock(return_value=True),
            ),
            patch("pc_liveagent.app_stage.prepare_application_test", prepare),
            patch("pc_liveagent.live_form.advance_welcome_screens", welcome),
            patch("pc_liveagent.live_form._begin_eligibility", begin_elig),
            patch("pc_liveagent.live_form._reset_eligibility_to_first_question", reset),
        ):
            session = await start_live_form("test-on", "https://my.pointcare.com/onboarding")
        logout.assert_not_awaited()
        welcome.assert_not_awaited()
        prepare.assert_awaited()
        begin_elig.assert_not_awaited()
        reset.assert_not_awaited()
        self.assertEqual(session.elig_step, "")
        self.assertTrue(session.welcome_done)
        self.assertIn("application_test_opening", session.filled)
        self.assertEqual(session.history[-1]["action"], "resume_saved_household")

    async def test_google_poll_accepts_dashboard_without_false_failure(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _sign_in_with_provider

        page = FakePage(DASHBOARD, url="https://my.pointcare.com/")
        page.context = type("Ctx", (), {"pages": []})()

        class _PopupCtx:
            async def __aenter__(self):
                raise RuntimeError("no popup")

            async def __aexit__(self, *_args):
                return False

        page.expect_popup = lambda timeout=0: _PopupCtx()  # type: ignore[attr-defined]
        session = LiveFormSession(call_id="google-ready", url=page.url, page=page)
        begin = AsyncMock(return_value="ASK: Would you like to continue and start the application?")
        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", True),
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", False),
            patch(
                "pc_liveagent.live_form._already_signed_in",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._application_test_authenticated",
                new=AsyncMock(return_value=True),
            ),
            patch("pc_liveagent.live_form._begin_eligibility", begin),
        ):
            spoken = await _sign_in_with_provider(
                session, "Continue with Google", ""
            )
        begin.assert_awaited()
        self.assertIn("continue and start the application", spoken.lower())
        self.assertNotIn("has not finished", spoken.lower())

    async def test_test_mode_wins_when_after_signin_is_also_on(self) -> None:
        from pc_liveagent.live_form import start_live_form

        page = FakePage(DASHBOARD, url="https://my.pointcare.com/home")
        logout = AsyncMock()
        prepare = AsyncMock(return_value="ASK: Would you like to continue and start the application?")
        after = AsyncMock(side_effect=AssertionError("after-signin route must not run"))

        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", True),
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", True),
            patch("pc_liveagent.live_form.open_page", new=AsyncMock(return_value=page)),
            patch("pc_liveagent.live_form._logout_for_fresh_signin", logout),
            patch("pc_liveagent.live_form._page_wait", new=AsyncMock()),
            patch(
                "pc_liveagent.live_form._application_test_authenticated",
                new=AsyncMock(return_value=True),
            ),
            patch("pc_liveagent.app_stage.prepare_application_test", prepare),
            patch("pc_liveagent.live_form._open_saved_application", after),
            patch("pc_liveagent.live_form.advance_welcome_screens", new=AsyncMock()),
        ):
            session = await start_live_form(
                "test-wins", "https://my.pointcare.com/onboarding"
            )
        logout.assert_not_awaited()
        prepare.assert_awaited()
        after.assert_not_awaited()
        self.assertIn("application_test_opening", session.filled)

    async def test_enabled_logged_out_uses_signin_without_logout(self) -> None:
        from pc_liveagent.live_form import start_live_form

        page = FakePage("Sign in with Google", url="https://my.pointcare.com/auth")
        logout = AsyncMock()
        welcome = AsyncMock(return_value=["Get Started"])
        prepare = AsyncMock(side_effect=AssertionError("eligibility must not start"))

        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", True),
            patch("pc_liveagent.live_form.open_page", new=AsyncMock(return_value=page)),
            patch("pc_liveagent.live_form._logout_for_fresh_signin", logout),
            patch("pc_liveagent.live_form._page_wait", new=AsyncMock()),
            patch(
                "pc_liveagent.live_form._application_test_authenticated",
                new=AsyncMock(return_value=False),
            ),
            patch("pc_liveagent.live_form._already_signed_in", new=AsyncMock(return_value=False)),
            patch("pc_liveagent.app_stage.prepare_application_test", prepare),
            patch("pc_liveagent.live_form.advance_welcome_screens", welcome),
            patch("pc_liveagent.live_form.settle_on_signin_page", new=AsyncMock()),
        ):
            session = await start_live_form(
                "test-logged-out", "https://my.pointcare.com/onboarding"
            )
        logout.assert_not_awaited()
        welcome.assert_awaited()
        prepare.assert_not_awaited()
        self.assertTrue(session.welcome_done)
        self.assertNotIn("application_test_opening", session.filled)

    async def test_disabled_still_logs_out_and_runs_welcome(self) -> None:
        from pc_liveagent.live_form import start_live_form

        page = FakePage("Get Started", url="https://my.pointcare.com/auth")
        logout = AsyncMock()

        async def welcome(page, *, call_id=""):
            page.url = "https://my.pointcare.com/auth"
            return ["Get Started"]

        prepare = AsyncMock(side_effect=AssertionError("test mode must stay off"))

        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", False),
            patch("pc_liveagent.live_form.open_page", new=AsyncMock(return_value=page)),
            patch("pc_liveagent.live_form._logout_for_fresh_signin", logout),
            patch("pc_liveagent.live_form._clear_pointcare_session", new=AsyncMock()),
            patch("pc_liveagent.live_form._page_wait", new=AsyncMock()),
            patch("pc_liveagent.live_form._already_signed_in", new=AsyncMock(return_value=False)),
            patch("pc_liveagent.app_stage.prepare_application_test", prepare),
            patch(
                "pc_liveagent.live_form.advance_welcome_screens",
                new=AsyncMock(side_effect=welcome),
            ) as welcome_mock,
            patch("pc_liveagent.live_form.settle_on_signin_page", new=AsyncMock()),
        ):
            session = await start_live_form("test-off", "https://my.pointcare.com/onboarding")
        logout.assert_not_awaited()
        welcome_mock.assert_awaited()
        prepare.assert_not_awaited()
        self.assertNotIn("application_test", [item.get("field_id") for item in session.history])

    async def test_closed_chrome_reopens_through_existing_open_page(self) -> None:
        from pc_liveagent.live_form import start_live_form

        page = FakePage(DASHBOARD)
        opened = AsyncMock(return_value=page)
        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", True),
            patch("pc_liveagent.live_form.open_page", opened),
            patch("pc_liveagent.live_form._logout_for_fresh_signin", new=AsyncMock()),
            patch("pc_liveagent.live_form._page_wait", new=AsyncMock()),
            patch(
                "pc_liveagent.live_form._application_test_authenticated",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.app_stage.prepare_application_test",
                new=AsyncMock(return_value="ASK: ready"),
            ),
        ):
            await start_live_form("test-reopen", "https://my.pointcare.com/onboarding?code=test")
        opened.assert_awaited()
        self.assertIn("pointcare.com", opened.await_args.args[0])

    async def test_begin_eligibility_skips_stage_d_only_when_enabled(self) -> None:
        from pc_liveagent.live_form import _begin_eligibility

        page = FakePage(DASHBOARD)
        session = LiveFormSession(call_id="elig", url=page.url)
        opened = AsyncMock(return_value=True)
        reset = AsyncMock(return_value=True)
        prepare = AsyncMock(return_value="ASK: program summary")
        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", True),
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", False),
            patch("pc_liveagent.live_form._open_household_eligibility", opened),
            patch("pc_liveagent.live_form._reset_eligibility_to_first_question", reset),
            patch("pc_liveagent.app_stage.prepare_application_test", prepare),
        ):
            spoken = await _begin_eligibility(session, page)
        self.assertEqual(spoken, "ASK: program summary")
        opened.assert_not_awaited()
        reset.assert_not_awaited()

        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", False),
            patch("pc_liveagent.live_form._open_household_eligibility", opened),
            patch("pc_liveagent.live_form._reset_eligibility_to_first_question", reset),
            patch("pc_liveagent.app_stage.prepare_application_test", prepare),
        ):
            spoken_off = await _begin_eligibility(session, page)
        opened.assert_awaited()
        reset.assert_awaited()
        self.assertIn("ASK:", spoken_off)
        self.assertEqual(session.elig_step, "marital")


class OpeningSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        from pc_liveagent.call_turn import drop_controller
        from pc_liveagent.live_form import end_live_form

        await drop_controller("speech-on")
        await drop_controller("speech-off")
        await drop_controller("turn-yes")
        await end_live_form("speech-on")
        await end_live_form("speech-off")
        await end_live_form("turn-yes")

    async def test_opening_skips_greeting_only_in_test_mode(self) -> None:
        from pc_liveagent.call_turn import begin_speech
        from pc_liveagent.live_fill_prompt import LIVE_FILL_GREETING
        from pc_liveagent.live_form import _sessions, _sessions_lock

        opening = (
            "Your household eligibility check is complete. "
            "Pointcare indicates that your household may be eligible for the programs shown. "
            "Would you like to continue and start the application?"
        )
        on = LiveFormSession(call_id="speech-on", url="https://example.test", elig_step="programs")
        on.filled["application_test_opening"] = opening
        off = LiveFormSession(call_id="speech-off", url="https://example.test", elig_step="programs")
        off.filled["application_test_opening"] = opening
        async with _sessions_lock:
            _sessions["speech-on"] = on
            _sessions["speech-off"] = off
        with patch("pc_liveagent.config.APPLICATION_TEST_MODE", True):
            heard = await begin_speech("speech-on")
        self.assertEqual(heard, opening)
        self.assertNotEqual(heard, LIVE_FILL_GREETING)
        with patch("pc_liveagent.config.APPLICATION_TEST_MODE", False):
            heard_off = await begin_speech("speech-off")
        self.assertEqual(heard_off, LIVE_FILL_GREETING)

    async def test_yes_from_language_phase_starts_application_not_signin(self) -> None:
        from pc_liveagent.call_turn import drop_controller, get_controller, process_user_turn

        call_id = "turn-yes"
        await drop_controller(call_id)
        ctrl = await get_controller(call_id)
        ctrl.phase = "language"
        ctrl.last_speech = "greeting"
        session = LiveFormSession(
            call_id=call_id, url="https://example.test", elig_step="programs"
        )
        session.filled["application_test_opening"] = (
            "Your household eligibility check is complete. "
            "Would you like to continue and start the application?"
        )
        submit = AsyncMock(
            return_value={
                "ok": True,
                "spoken": "ASK: Are all household members U.S. citizens?",
                "step": "app_citizenship",
                "phase": "application",
            }
        )
        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", True),
            patch("pc_liveagent.call_turn.get_session", new=AsyncMock(return_value=session)),
            patch("pc_liveagent.call_turn.submit_field", submit),
            patch("pc_liveagent.call_turn._SLOW_ELIG_STEPS", frozenset()),
            patch(
                "pc_liveagent.call_turn.peek_choice_labels",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await process_user_turn(call_id, "Yes")
        self.assertNotEqual(result.phase, "sign_in")
        self.assertNotEqual(result.detail, "language_clarify")
        submit.assert_awaited()


if __name__ == "__main__":
    unittest.main()
