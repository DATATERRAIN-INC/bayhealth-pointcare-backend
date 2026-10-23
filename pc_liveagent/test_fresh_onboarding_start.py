"""Full-call browser start: welcome page 1 first, then 4 clicks, then sign-in."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class FreshOnboardingStartTests(unittest.IsolatedAsyncioTestCase):
    async def test_logout_clears_then_opens_welcome_not_ui_logout(self) -> None:
        from pc_liveagent.live_form import _logout_for_fresh_signin

        page = AsyncMock()
        page.url = "about:blank"
        page.wait_for_timeout = AsyncMock()

        async def fake_goto(url, **_kwargs):
            page.url = url
            return None

        page.goto = AsyncMock(side_effect=fake_goto)

        with (
            patch(
                "pc_liveagent.live_form._clear_pointcare_session",
                new=AsyncMock(),
            ) as clear,
            patch(
                "pc_liveagent.live_form._already_signed_in",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._on_saved_household_surface",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._confirm_welcome_page_one",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.live_form._ui_logout_saved_session",
                new=AsyncMock(side_effect=AssertionError("must not UI-logout on welcome")),
            ),
        ):
            await _logout_for_fresh_signin(
                page, "https://my.pointcare.com/onboarding?code=x"
            )
        self.assertEqual(clear.await_count, 2)
        self.assertEqual(page.goto.await_count, 1)
        self.assertIn("/onboarding", page.url)

    async def test_saved_dashboard_triggers_ui_logout_then_welcome(self) -> None:
        from pc_liveagent.live_form import _logout_for_fresh_signin

        page = AsyncMock()
        page.url = "about:blank"
        page.wait_for_timeout = AsyncMock()
        # First confirm checks fail (dashboard); after UI logout, welcome succeeds.
        surfaces = iter([True, True, False, False])
        welcomes = iter([False, False, True])

        async def fake_goto(url, **_kwargs):
            page.url = url
            return None

        page.goto = AsyncMock(side_effect=fake_goto)

        with (
            patch(
                "pc_liveagent.live_form._clear_pointcare_session",
                new=AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form._on_saved_household_surface",
                new=AsyncMock(side_effect=lambda _p: next(surfaces, False)),
            ),
            patch(
                "pc_liveagent.live_form._confirm_welcome_page_one",
                new=AsyncMock(side_effect=lambda _p: next(welcomes, True)),
            ),
            patch(
                "pc_liveagent.live_form._already_signed_in",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._ui_logout_saved_session",
                new=AsyncMock(),
            ) as ui_logout,
        ):
            await _logout_for_fresh_signin(
                page, "https://my.pointcare.com/onboarding?code=x"
            )
        self.assertGreaterEqual(ui_logout.await_count, 1)

    async def test_already_signed_in_detects_quick_apply_dashboard(self) -> None:
        from pc_liveagent.live_form import _already_signed_in

        class _Body:
            async def inner_text(self, timeout=2500):
                return (
                    "Next steps Continue Quick Apply New Quick Apply "
                    "Household Coverage unknown Murali M."
                )

        page = AsyncMock()
        page.url = "https://my.pointcare.com/home"
        page.locator = lambda _s: _Body()

        with patch(
            "pc_liveagent.live_form._visible_button_text",
            new=AsyncMock(return_value="Recheck Eligibility | Add Member"),
        ):
            self.assertTrue(await _already_signed_in(page))

    async def test_onboarding_url_with_dashboard_content_is_saved_surface(self) -> None:
        from pc_liveagent.live_form import _on_saved_household_surface

        class _Body:
            async def inner_text(self, timeout=2500):
                return (
                    "Next steps Continue the application Continue Quick Apply "
                    "New Quick Apply Household Coverage unknown"
                )

        page = AsyncMock()
        page.url = "https://my.pointcare.com/onboarding?code=x"
        page.locator = lambda _s: _Body()

        with patch(
            "pc_liveagent.live_form._visible_button_text",
            new=AsyncMock(
                return_value="Continue Quick Apply | New Quick Apply | Recheck Eligibility"
            ),
        ):
            self.assertTrue(await _on_saved_household_surface(page))

    def test_dashboard_text_helper(self) -> None:
        from pc_liveagent.live_form import _text_looks_like_saved_dashboard

        self.assertTrue(
            _text_looks_like_saved_dashboard(
                "next steps household coverage unknown",
                "Continue Quick Apply",
            )
        )
        self.assertFalse(
            _text_looks_like_saved_dashboard(
                "get covered with health insurance for your household",
                "Get Started",
            )
        )

    async def test_full_flow_opens_blank_then_fresh_onboarding(self) -> None:
        from pc_liveagent import live_form as lf

        call_id = "fresh-start"
        async with lf._sessions_lock:
            lf._sessions.pop(call_id, None)

        page = AsyncMock()
        page.url = "about:blank"
        page.on = MagicMock()
        opens: list[str] = []

        async def fake_open(url):
            opens.append(url or "")
            if url:
                page.url = url
            return page

        async def fake_goto(url, **_kwargs):
            page.url = url
            return None

        page.goto = AsyncMock(side_effect=fake_goto)

        async def welcome_side_effect(_page, *, call_id=""):
            page.url = "https://my.pointcare.com/auth"
            return ["Get Started", "Continue", "Continue", "Create My Profile"]

        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", False),
            patch(
                "pc_liveagent.live_form.open_page",
                new=AsyncMock(side_effect=fake_open),
            ),
            patch(
                "pc_liveagent.live_form._logout_for_fresh_signin",
                new=AsyncMock(),
            ) as fresh,
            patch(
                "pc_liveagent.live_form._clear_pointcare_session",
                new=AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form._already_signed_in",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form._on_saved_household_surface",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "pc_liveagent.live_form.advance_welcome_screens",
                new=AsyncMock(side_effect=welcome_side_effect),
            ),
            patch(
                "pc_liveagent.live_form.settle_on_signin_page",
                new=AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form._page_wait",
                new=AsyncMock(),
            ),
        ):
            session = await lf.start_live_form(
                call_id, "https://my.pointcare.com/onboarding?code=x"
            )
        self.assertEqual(opens[0], "")
        fresh.assert_not_awaited()
        page.goto.assert_awaited_once()
        self.assertIn("/onboarding", page.goto.await_args.args[0])
        self.assertTrue(session.welcome_done)
        async with lf._sessions_lock:
            lf._sessions.pop(call_id, None)

    async def test_get_started_landing_is_not_signed_in(self) -> None:
        from pc_liveagent.live_form import _already_signed_in

        class _Body:
            async def inner_text(self, timeout=2500):
                return (
                    "Your family may qualify for free or low-cost health coverage. "
                    "Get Started Find out who qualifies in 5 min"
                )

        page = AsyncMock()
        page.url = "https://my.pointcare.com/onboarding?code=x"
        page.locator = lambda _s: _Body()

        with (
            patch(
                "pc_liveagent.live_form._visible_button_text",
                new=AsyncMock(return_value="en | Get Started"),
            ),
            patch(
                "pc_liveagent.live_form._on_saved_household_surface",
                new=AsyncMock(return_value=False),
            ),
        ):
            self.assertFalse(await _already_signed_in(page))

    async def test_mid_call_reattach_skips_logout_and_welcome(self) -> None:
        from pc_liveagent import live_form as lf

        call_id = "reattach-no-welcome"
        async with lf._sessions_lock:
            lf._sessions.pop(call_id, None)

        session = lf.LiveFormSession(
            call_id=call_id,
            url="https://my.pointcare.com/auth",
            welcome_done=True,
            status="open",
        )
        session.ready.set()
        session.page = None
        async with lf._sessions_lock:
            lf._sessions[call_id] = session
            lf._active_call_id = call_id

        page = AsyncMock()
        page.url = "https://my.pointcare.com/onboarding?code=x"
        page.is_closed = lambda: False
        page.goto = AsyncMock()

        async def fake_goto(url, **_kwargs):
            page.url = url
            return None

        page.goto = AsyncMock(side_effect=fake_goto)

        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", False),
            patch(
                "pc_liveagent.live_form.open_page",
                new=AsyncMock(return_value=page),
            ) as open_page,
            patch(
                "pc_liveagent.live_form._logout_for_fresh_signin",
                new=AsyncMock(),
            ) as logout,
            patch(
                "pc_liveagent.live_form.advance_welcome_screens",
                new=AsyncMock(),
            ) as welcome,
            patch(
                "pc_liveagent.live_form._welcome_get_started_visible",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.live_form.settle_on_signin_page",
                new=AsyncMock(),
            ),
        ):
            out = await lf.start_live_form(
                call_id, "https://my.pointcare.com/onboarding?code=x"
            )

        logout.assert_not_awaited()
        welcome.assert_not_awaited()
        open_page.assert_awaited()
        page.goto.assert_awaited()
        self.assertIn("/auth", page.url)
        self.assertTrue(out.welcome_done)
        async with lf._sessions_lock:
            lf._sessions.pop(call_id, None)

    async def test_clear_pointcare_keeps_google_cookies(self) -> None:
        from pc_liveagent.live_form import _clear_pointcare_session

        stored: list[dict] = []
        context = AsyncMock()
        context.cookies = AsyncMock(
            return_value=[
                {
                    "name": "SID",
                    "value": "google",
                    "domain": ".google.com",
                    "path": "/",
                    "expires": 2000000000,
                    "httpOnly": True,
                    "secure": True,
                    "sameSite": "Lax",
                },
                {
                    "name": "session",
                    "value": "pointcare",
                    "domain": ".pointcare.com",
                    "path": "/",
                    "secure": True,
                },
            ]
        )

        async def add_cookies(cookies):
            stored.extend(cookies)

        context.add_cookies = AsyncMock(side_effect=add_cookies)
        context.clear_cookies = AsyncMock()
        page = AsyncMock()
        page.url = "https://my.pointcare.com/"
        page.context = context

        await _clear_pointcare_session(page)
        context.clear_cookies.assert_awaited()
        self.assertEqual([item["name"] for item in stored], ["SID"])
        self.assertEqual(stored[0]["domain"], ".google.com")

    async def test_promote_skips_welcome_get_started(self) -> None:
        from pc_liveagent import live_form as lf

        call_id = "promote-welcome"
        async with lf._sessions_lock:
            lf._sessions.pop(call_id, None)

        page = AsyncMock()
        page.url = "https://my.pointcare.com/onboarding?code=x"
        page.is_closed = lambda: False
        session = lf.LiveFormSession(
            call_id=call_id,
            url=page.url,
            welcome_done=True,
            status="open",
            page=page,
        )
        session.ready.set()
        async with lf._sessions_lock:
            lf._sessions[call_id] = session

        with (
            patch(
                "pc_liveagent.live_form._welcome_get_started_visible",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "pc_liveagent.live_form._begin_eligibility",
                new=AsyncMock(),
            ) as begin,
        ):
            speech = await lf.promote_signed_in_to_eligibility(call_id)

        self.assertIsNone(speech)
        begin.assert_not_awaited()
        async with lf._sessions_lock:
            lf._sessions.pop(call_id, None)


if __name__ == "__main__":
    unittest.main()
