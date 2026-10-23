"""Post-sign-in application flow. Mocks only — no Chrome, calls, or submissions."""

from __future__ import annotations

import subprocess
import sys
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
    "Medicare not eligible at this time Start Applications"
)
APPLICATION = "Application Eligibility Are all household members U.S. citizens?"
PROCEED = (
    "Your household eligibility check is finished. "
    "Can I start the application now? Please say Yes to start, or No to end the call."
)


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


class ApplicationAfterSigninFlagTests(unittest.TestCase):
    def test_flag_defaults_off_and_is_separate_from_skip_intro_mode(self) -> None:
        from pc_liveagent.config import (
            application_after_signin_from_env,
            application_test_mode_from_env,
        )

        self.assertFalse(application_after_signin_from_env({}))
        self.assertFalse(
            application_after_signin_from_env(
                {"POINTCARE_APPLICATION_TEST_MODE": "true"}
            )
        )
        self.assertTrue(
            application_after_signin_from_env(
                {"POINTCARE_APPLICATION_AFTER_SIGNIN": "true"}
            )
        )
        self.assertFalse(
            application_test_mode_from_env(
                {"POINTCARE_APPLICATION_AFTER_SIGNIN": "true"}
            )
        )
        self.assertFalse(
            application_after_signin_from_env(
                {
                    "POINTCARE_APPLICATION_AFTER_SIGNIN": "true",
                    "APP_ENV": "production",
                }
            )
        )

    def test_project_env_uses_full_call_flow_flags(self) -> None:
        """Normal full call: TEST_MODE and AFTER_SIGNIN both off."""
        root = Path(__file__).resolve().parents[1]
        text = (root / ".env").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^POINTCARE_APPLICATION_TEST_MODE=false\s*$")
        self.assertRegex(text, r"(?m)^POINTCARE_APPLICATION_AFTER_SIGNIN=false\s*$")
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pc_liveagent.config import "
                "APPLICATION_AFTER_SIGNIN, APPLICATION_TEST_MODE; "
                "print(int(APPLICATION_AFTER_SIGNIN), int(APPLICATION_TEST_MODE))",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "0 0")


class LandingPageTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_asks_before_clicking_quick_apply(self) -> None:
        from pc_liveagent.app_stage import begin_application_after_signin

        page = FakePage(DASHBOARD)
        session = LiveFormSession(call_id="dash", url=page.url)
        spoken = await begin_application_after_signin(session, page)
        self.assertEqual(page.clicked, [])
        self.assertEqual(session.elig_step, "proceed_application")
        self.assertEqual(session.filled.get("awaiting_quick_apply"), "1")
        self.assertIn("eligibility check is finished", spoken.lower())
        self.assertIn("can i start the application", spoken.lower())
        self.assertNotEqual(session.elig_step, "marital")
        self.assertEqual(session.filled.get("application_started"), "0")

    async def test_already_open_program_page_does_not_click_quick_apply(self) -> None:
        from pc_liveagent.app_stage import begin_application_after_signin

        page = FakePage(PROGRAMS)

        async def _store(session, _page) -> None:
            session.filled["program_summary_json"] = '{"programs":[]}'

        session = LiveFormSession(call_id="news", url=page.url)
        with patch("pc_liveagent.app_stage.store_program_summary", new=_store):
            spoken = await begin_application_after_signin(session, page)
        self.assertEqual(page.clicked, [])
        self.assertEqual(session.elig_step, "proceed_application")
        self.assertIn(PROCEED, spoken)

    async def test_already_open_application_waits_for_yes_before_resuming(self) -> None:
        from pc_liveagent.app_stage import begin_application_after_signin

        page = FakePage(APPLICATION, url="https://my.pointcare.com/application/step-1")
        session = LiveFormSession(call_id="resume", url=page.url)
        spoken = await begin_application_after_signin(session, page)
        self.assertEqual(page.clicked, [])
        self.assertEqual(session.elig_step, "proceed_application")
        self.assertEqual(session.filled.get("resume_app_step"), "app_citizenship")
        self.assertEqual(session.app_step, "")
        self.assertIn(PROCEED, spoken)

    async def test_unexpected_page_reports_the_failure(self) -> None:
        from pc_liveagent.app_stage import begin_application_after_signin

        page = FakePage("Totally unrelated page", url="https://example.test/nope")
        session = LiveFormSession(call_id="bad", url=page.url)
        with self.assertRaises(RuntimeError) as ctx:
            await begin_application_after_signin(session, page)
        self.assertIn("APPLICATION_AFTER_SIGNIN", str(ctx.exception))
        self.assertEqual(session.elig_step, "")
        self.assertNotEqual(session.elig_step, "marital")


class GateDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_yes_from_programs_starts_application_once(self) -> None:
        from pc_liveagent.app_stage import handle_application_gate

        session = LiveFormSession(
            call_id="yes1", url="https://example.test", elig_step="proceed_application"
        )
        session.filled["application_gate_speech"] = PROCEED
        session.filled["awaiting_quick_apply"] = "1"
        page = FakePage(PROGRAMS)
        with (
            patch(
                "pc_liveagent.app_stage.click_quick_apply",
                new=AsyncMock(return_value=""),
            ),
            patch(
                "pc_liveagent.app_stage.classify_saved_household_page",
                new=AsyncMock(return_value="programs"),
            ),
            patch(
                "pc_liveagent.app_stage.click_start_applications",
                new=AsyncMock(return_value="Start Applications"),
            ) as start,
            patch(
                "pc_liveagent.app_stage.wait_for_application_started",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.app_stage.detect_application_screen",
                new=AsyncMock(return_value="app_citizenship"),
            ),
            patch("pc_liveagent.app_stage.store_program_summary", new=AsyncMock()),
        ):
            result = await handle_application_gate("yes1", session, page, "Yes")
        start.assert_awaited_once()
        self.assertEqual(result.get("phase"), "application")
        self.assertEqual(result.get("step"), "app_citizenship")
        self.assertIn("started the application", str(result.get("spoken") or "").lower())
        self.assertEqual(session.filled.get("application_started"), "1")

    async def test_yes_from_dashboard_clicks_continue_quick_apply_then_starts(
        self,
    ) -> None:
        from pc_liveagent.app_stage import handle_application_gate

        session = LiveFormSession(
            call_id="yes-dash",
            url="https://example.test",
            elig_step="proceed_application",
        )
        session.filled["application_gate_speech"] = PROCEED
        session.filled["awaiting_quick_apply"] = "1"
        page = FakePage(DASHBOARD)
        with (
            patch(
                "pc_liveagent.app_stage.click_quick_apply",
                new=AsyncMock(return_value="Continue Quick Apply"),
            ) as qa,
            patch(
                "pc_liveagent.app_stage.classify_saved_household_page",
                new=AsyncMock(side_effect=["dashboard", "programs"]),
            ),
            patch(
                "pc_liveagent.app_stage._wait_surface_after_quick_apply",
                new=AsyncMock(return_value="programs"),
            ),
            patch(
                "pc_liveagent.app_stage.click_start_applications",
                new=AsyncMock(return_value="Start Applications"),
            ) as start,
            patch(
                "pc_liveagent.app_stage.wait_for_application_started",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.app_stage.detect_application_screen",
                new=AsyncMock(return_value="app_citizenship"),
            ),
            patch("pc_liveagent.app_stage.store_program_summary", new=AsyncMock()),
        ):
            result = await handle_application_gate("yes-dash", session, page, "Yes")
        qa.assert_awaited_once()
        start.assert_awaited_once()
        self.assertEqual(result.get("step"), "app_citizenship")
        self.assertEqual(session.filled.get("quick_apply_clicked"), "Continue Quick Apply")

    async def test_confirmation_yes_clicks_start_applications_once(self) -> None:
        from pc_liveagent.app_stage import handle_application_gate

        session = LiveFormSession(
            call_id="yes2",
            url="https://example.test",
            elig_step="confirm_start_application",
        )
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
            first = await handle_application_gate("yes2", session, page, "Yes")
            session.elig_step = "confirm_start_application"
            second = await handle_application_gate("yes2", session, page, "Yes")
        start.assert_awaited_once()
        self.assertEqual(first.get("phase"), "application")
        self.assertEqual(first.get("step"), "app_citizenship")
        self.assertIn("started the application", str(first.get("spoken") or "").lower())
        self.assertEqual(second.get("step"), "app_citizenship")

    async def test_no_ends_without_starting_or_stage_d(self) -> None:
        from pc_liveagent.app_stage import handle_application_gate

        session = LiveFormSession(
            call_id="no", url="https://example.test", elig_step="proceed_application"
        )
        session.filled["program_summary_json"] = '{"programs":[{"name":"MediKids"}]}'
        page = FakePage(PROGRAMS)
        start = AsyncMock(side_effect=AssertionError("must not start"))
        with patch("pc_liveagent.app_stage.click_start_applications", start):
            result = await handle_application_gate("no", session, page, "No")
        start.assert_not_awaited()
        self.assertTrue(result.get("end_call"))
        self.assertEqual(result.get("step"), "")
        self.assertEqual(session.elig_step, "")
        self.assertIn("will not start the application", str(result.get("spoken") or "").lower())
        self.assertEqual(
            session.filled.get("program_summary_json"),
            '{"programs":[{"name":"MediKids"}]}',
        )

    async def test_yes_on_in_progress_application_does_not_click_start(self) -> None:
        from pc_liveagent.app_stage import handle_application_gate

        session = LiveFormSession(
            call_id="resume-yes",
            url="https://example.test",
            elig_step="proceed_application",
        )
        session.filled["resume_app_step"] = "app_citizenship"
        session.filled["application_started"] = "1"
        page = FakePage(APPLICATION)
        start = AsyncMock(side_effect=AssertionError("must not start again"))
        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.app_stage.click_start_applications", start),
            patch(
                "pc_liveagent.app_stage._prepare_resume_speech",
                new=AsyncMock(
                    return_value=(
                        "app_citizenship",
                        "Are all household members listed on this application "
                        "U.S. citizens? Please say Yes or No.",
                    )
                ),
            ),
        ):
            result = await handle_application_gate("resume-yes", session, page, "Yes")
        start.assert_not_awaited()
        self.assertEqual(result.get("phase"), "application")
        self.assertEqual(result.get("step"), "app_citizenship")
        self.assertIn("already open", str(result.get("spoken") or "").lower())


class SignInRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        from pc_liveagent.call_turn import drop_controller
        from pc_liveagent.live_form import end_live_form

        await drop_controller("greet")
        for call_id in ("saved", "logged-out", "promote"):
            await end_live_form(call_id)

    async def test_begin_eligibility_does_not_open_stage_d_when_enabled(self) -> None:
        from pc_liveagent.live_form import _begin_eligibility, _open_saved_application

        page = FakePage(PROGRAMS)
        session = LiveFormSession(call_id="elig", url=page.url)
        opened = AsyncMock(side_effect=AssertionError("stage d must not open"))
        reset = AsyncMock(side_effect=AssertionError("must not reset eligibility"))
        begin = AsyncMock(return_value="ASK: " + PROCEED)
        with (
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", True),
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.live_form._open_household_eligibility", opened),
            patch("pc_liveagent.live_form._reset_eligibility_to_first_question", reset),
            patch("pc_liveagent.app_stage.begin_application_after_signin", begin),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _begin_eligibility(session, page)
            spoken = await _open_saved_application(session, page)
        self.assertIn("refused to start Stage D", str(ctx.exception))
        opened.assert_not_awaited()
        reset.assert_not_awaited()
        begin.assert_awaited()
        self.assertIn("Can I start the application", spoken)

        opened2 = AsyncMock(return_value=True)
        reset = AsyncMock(return_value=True)
        with (
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", False),
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.live_form._open_household_eligibility", opened2),
            patch("pc_liveagent.live_form._reset_eligibility_to_first_question", reset),
        ):
            production = await _begin_eligibility(session, page)
        opened2.assert_awaited()
        self.assertEqual(session.elig_step, "marital")
        self.assertIn("ASK:", production)

    async def test_saved_account_keeps_intro_and_does_not_log_out(self) -> None:
        from pc_liveagent.call_turn import begin_speech
        from pc_liveagent.live_fill_prompt import LIVE_FILL_GREETING
        from pc_liveagent.live_form import start_live_form

        page = FakePage(DASHBOARD)
        logout = AsyncMock()
        welcome = AsyncMock(return_value=["Get Started"])
        with (
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", True),
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.live_form.open_page", new=AsyncMock(return_value=page)),
            patch("pc_liveagent.live_form._logout_for_fresh_signin", logout),
            patch("pc_liveagent.live_form._page_wait", new=AsyncMock()),
            patch(
                "pc_liveagent.live_form._application_test_authenticated",
                new=AsyncMock(return_value=True),
            ),
            patch("pc_liveagent.live_form.advance_welcome_screens", welcome),
        ):
            session = await start_live_form("saved", "https://my.pointcare.com/onboarding")
            greeting = await begin_speech("greet")
        logout.assert_not_awaited()
        welcome.assert_not_awaited()
        self.assertEqual(session.elig_step, "")
        self.assertFalse(session.signed_in)
        self.assertTrue(session.welcome_done)
        self.assertEqual(greeting, LIVE_FILL_GREETING)

    async def test_logged_out_still_reaches_sign_in(self) -> None:
        from pc_liveagent.live_form import start_live_form

        page = FakePage("Sign in", url="https://my.pointcare.com/auth")
        logout = AsyncMock()
        welcome = AsyncMock(return_value=["Get Started"])
        with (
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", True),
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.live_form.open_page", new=AsyncMock(return_value=page)),
            patch("pc_liveagent.live_form._logout_for_fresh_signin", logout),
            patch("pc_liveagent.live_form._page_wait", new=AsyncMock()),
            patch(
                "pc_liveagent.live_form._application_test_authenticated",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._already_signed_in",
                new=AsyncMock(return_value=False),
            ),
            patch("pc_liveagent.live_form.advance_welcome_screens", welcome),
            patch("pc_liveagent.live_form.settle_on_signin_page", new=AsyncMock()),
        ):
            session = await start_live_form(
                "logged-out", "https://my.pointcare.com/onboarding"
            )
        logout.assert_not_awaited()
        welcome.assert_awaited()
        self.assertEqual(session.elig_step, "")
        self.assertTrue(session.welcome_done)

    async def test_onboarding_copy_still_clicks_through_to_sign_in(self) -> None:
        from pc_liveagent.live_form import _application_test_authenticated, start_live_form

        page = FakePage(
            "Get health insurance coverage. Get Started",
            url="https://my.pointcare.com/onboarding?code=abc",
        )
        welcome_calls: list[str] = []

        async def _welcome(target, call_id: str = "") -> list[str]:
            welcome_calls.append(target.url)
            target.url = "https://my.pointcare.com/auth"
            return ["Get Started", "Continue", "Create My Profile"]

        settle = AsyncMock()
        with (
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", True),
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.live_form.open_page", new=AsyncMock(return_value=page)),
            patch("pc_liveagent.live_form._logout_for_fresh_signin", new=AsyncMock()),
            patch("pc_liveagent.live_form._page_wait", new=AsyncMock()),
            patch("pc_liveagent.live_form._already_signed_in", new=AsyncMock(return_value=False)),
            patch("pc_liveagent.live_form.advance_welcome_screens", new=_welcome),
            patch("pc_liveagent.live_form.settle_on_signin_page", settle),
        ):
            self.assertFalse(await _application_test_authenticated(page))
            session = await start_live_form(
                "onboarding", "https://my.pointcare.com/onboarding?code=abc"
            )
        self.assertEqual(welcome_calls, ["https://my.pointcare.com/onboarding?code=abc"])
        settle.assert_awaited()
        self.assertIn("/auth", session.url)
        self.assertTrue(session.welcome_done)
        self.assertEqual(session.elig_step, "")

    async def test_sign_in_failure_does_not_start_stage_d(self) -> None:
        from pc_liveagent.live_form import _sign_in_with_provider

        page = FakePage("Sign in", url="https://my.pointcare.com/auth")
        page.context = type("Ctx", (), {"pages": []})()

        class _PopupCtx:
            async def __aenter__(self):
                raise RuntimeError("no popup")

            async def __aexit__(self, *_args):
                return False

        page.expect_popup = lambda timeout=0: _PopupCtx()  # type: ignore[attr-defined]
        session = LiveFormSession(call_id="fail", url=page.url, page=page)
        opened = AsyncMock(side_effect=AssertionError("stage d must not open"))
        with (
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", True),
            patch("pc_liveagent.live_form._open_household_eligibility", opened),
            patch(
                "pc_liveagent.live_form._already_signed_in",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._application_test_authenticated",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._visible_button_text",
                new=AsyncMock(return_value="Sign in"),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _sign_in_with_provider(session, "Continue with Google", "")
        self.assertIn("was not on the page", str(ctx.exception))
        opened.assert_not_awaited()
        self.assertEqual(session.elig_step, "")

    async def test_promote_waits_for_sign_in_choice(self) -> None:
        from pc_liveagent.live_form import promote_signed_in_to_eligibility

        session = LiveFormSession(call_id="promote", url="https://my.pointcare.com/home")
        session.page = FakePage(DASHBOARD)  # type: ignore[assignment]
        begin = AsyncMock(side_effect=AssertionError("must not open eligibility yet"))
        with (
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", True),
            patch(
                "pc_liveagent.live_form.get_session",
                new=AsyncMock(return_value=session),
            ),
            patch("pc_liveagent.live_form._begin_eligibility", begin),
        ):
            speech = await promote_signed_in_to_eligibility("promote")
        self.assertIsNone(speech)
        begin.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
