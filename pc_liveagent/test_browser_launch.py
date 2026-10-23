"""Tests for Chrome executable discovery and form-filler session routes."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from pc_liveagent.main import app


class ChromeExecutableTests(unittest.TestCase):
    def test_linux_discovers_google_chrome_stable(self) -> None:
        from pc_liveagent import browser as browser_mod

        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "google-chrome-stable"
            fake.write_text("#!/bin/sh\n", encoding="utf-8")
            fake.chmod(0o755)
            with (
                patch.object(browser_mod.sys, "platform", "linux"),
                patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("CHROME_EXECUTABLE", None)
                with (
                    patch.object(browser_mod, "CHROME_CHANNEL", "chrome"),
                    patch.object(
                        browser_mod.shutil,
                        "which",
                        side_effect=lambda name: str(fake)
                        if name == "google-chrome-stable"
                        else None,
                    ),
                    patch.object(
                        browser_mod,
                        "_is_executable_file",
                        side_effect=lambda p: Path(p).resolve() == fake.resolve(),
                    ),
                ):
                    resolved = browser_mod._chrome_executable()
            self.assertEqual(Path(resolved).resolve(), fake.resolve())

    def test_windows_discovers_program_files_chrome(self) -> None:
        from pc_liveagent import browser as browser_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "Google" / "Chrome" / "Application" / "chrome.exe"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"MZ")
            with (
                patch.object(browser_mod.sys, "platform", "win32"),
                patch.dict(
                    os.environ,
                    {
                        "PROGRAMFILES": str(root),
                        "PROGRAMFILES(X86)": str(root / "x86"),
                        "LOCALAPPDATA": str(root / "local"),
                    },
                    clear=False,
                ),
            ):
                os.environ.pop("CHROME_EXECUTABLE", None)
                with (
                    patch.object(browser_mod, "CHROME_CHANNEL", "chrome"),
                    patch.object(
                        browser_mod,
                        "_is_executable_file",
                        side_effect=lambda p: Path(p) == exe,
                    ),
                ):
                    resolved = browser_mod._chrome_executable()
            self.assertEqual(Path(resolved), exe)

    def test_explicit_chrome_executable_env(self) -> None:
        from pc_liveagent import browser as browser_mod

        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "chrome"
            fake.write_text("x", encoding="utf-8")
            fake.chmod(0o755)
            with (
                patch.dict(os.environ, {"CHROME_EXECUTABLE": str(fake)}, clear=False),
                patch.object(
                    browser_mod, "_is_executable_file", side_effect=lambda p: Path(p) == fake
                ),
            ):
                resolved = browser_mod._chrome_executable()
            self.assertEqual(Path(resolved), fake)

    def test_missing_chrome_raises_clear_error(self) -> None:
        from pc_liveagent import browser as browser_mod

        with (
            patch.object(browser_mod.sys, "platform", "linux"),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("CHROME_EXECUTABLE", None)
            with (
                patch.object(browser_mod, "CHROME_CHANNEL", "chrome"),
                patch.object(browser_mod.shutil, "which", return_value=None),
                patch.object(browser_mod, "_is_executable_file", return_value=False),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    browser_mod._chrome_executable()
        msg = str(ctx.exception).lower()
        self.assertIn("chrome was not found", msg)
        self.assertIn("chrome_executable", msg)

    def test_bad_chrome_executable_env_raises(self) -> None:
        from pc_liveagent import browser as browser_mod

        with (
            patch.dict(
                os.environ,
                {"CHROME_EXECUTABLE": "/no/such/chrome-binary"},
                clear=False,
            ),
            patch.object(browser_mod, "_is_executable_file", return_value=False),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                browser_mod._chrome_executable()
        self.assertIn("CHROME_EXECUTABLE", str(ctx.exception))


class FormFillerRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_openapi_registers_session_routes(self) -> None:
        resp = self.client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        paths = resp.json()["paths"]
        self.assertIn("/form-filler/sessions/{call_id}", paths)
        self.assertIn("/form-filler/sessions/{call_id}/chrome", paths)
        self.assertIn("/form-filler/outbound/", paths)
        self.assertIn("/form-filler/welcome/", paths)

    def test_missing_session_json_404(self) -> None:
        resp = self.client.get("/form-filler/sessions/call_does_not_exist")
        self.assertEqual(resp.status_code, 404)
        body = resp.json()
        detail = body.get("detail") or body
        if isinstance(detail, dict):
            self.assertEqual(detail.get("error"), "session_not_found")
            self.assertEqual(detail.get("service"), "pointcare-liveagent")
        else:
            self.assertIn("session", str(detail).lower())

    def test_missing_session_chrome_html_404_is_fastapi(self) -> None:
        resp = self.client.get("/form-filler/sessions/call_does_not_exist/chrome")
        self.assertEqual(resp.status_code, 404)
        text = resp.text.lower()
        self.assertIn("pointcare live agent", text)
        self.assertIn("session not found", text)
        # Distinctive marker vs Django's generic 404 template.
        self.assertIn("pointcare-liveagent", text)

    def test_chrome_url_uses_public_base(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, describe_live_browser

        async def _run():
            session = LiveFormSession(
                call_id="call_url_gen",
                url="https://my.pointcare.com/onboarding",
                welcome_done=True,
                status="open",
            )
            page = AsyncMock()
            page.url = "https://my.pointcare.com/auth"
            page.is_closed = lambda: False
            session.page = page
            with patch(
                "pc_liveagent.browser.cdp_target_for_url",
                return_value={"id": "T1", "url": page.url, "type": "page"},
            ):
                return await describe_live_browser(
                    session,
                    call_id="call_url_gen",
                    open_base_url="https://bach.gapincare-backend.socialroots-dev.net",
                )

        import asyncio

        info = asyncio.run(_run())
        self.assertEqual(
            info["open_url"],
            "https://bach.gapincare-backend.socialroots-dev.net"
            "/form-filler/sessions/call_url_gen/chrome",
        )
        self.assertTrue(info["opened"])

    def test_active_session_chrome_page(self) -> None:
        from pc_liveagent.live_form import LiveFormSession
        import pc_liveagent.live_form as live_form

        call_id = "call_active_chrome"
        session = LiveFormSession(
            call_id=call_id, url="https://my.pointcare.com/auth", welcome_done=True
        )
        page = AsyncMock()
        page.url = "https://my.pointcare.com/auth"
        page.is_closed = lambda: False
        page.bring_to_front = AsyncMock()
        session.page = page
        session.ready.set()

        async def fake_get(cid: str):
            return session if cid == call_id else None

        with (
            patch.object(live_form, "get_session", side_effect=fake_get),
            patch(
                "pc_liveagent.main.get_session",
                side_effect=fake_get,
            ),
            patch(
                "pc_liveagent.main.describe_live_browser",
                new=AsyncMock(
                    return_value={
                        "opened": True,
                        "url": "https://my.pointcare.com/auth",
                        "open_url": f"http://test/form-filler/sessions/{call_id}/chrome",
                        "devtools_url": "http://127.0.0.1:9222/devtools",
                        "chrome_profile": "Default",
                        "cdp_port": 9222,
                    }
                ),
            ),
        ):
            resp = self.client.get(f"/form-filler/sessions/{call_id}/chrome")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("pointcare-liveagent", resp.text.lower())
        self.assertIn(call_id, resp.text)
        self.assertNotIn("User data:", resp.text)  # no filesystem path leak


class BrowserStartupFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_outbound_reports_opened_false_on_chrome_missing(self) -> None:
        from pc_liveagent.main import form_filler_outbound, FormFillerCallIn
        from fastapi import BackgroundTasks, Request

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/form-filler/outbound/",
            "headers": [],
        }
        request = Request(scope)
        payload = FormFillerCallIn(phone_number="+15551234567", name="Test")
        bg = BackgroundTasks()

        with (
            patch(
                "pc_liveagent.main.place_form_filler_call",
                return_value={"ok": True, "call_id": "call_fail_chrome"},
            ),
            patch(
                "pc_liveagent.main.on_call_started",
                new=AsyncMock(
                    side_effect=RuntimeError(
                        "Google Chrome was not found. Install Chrome or set CHROME_EXECUTABLE."
                    )
                ),
            ),
            patch("pc_liveagent.main.config.PUBLIC_BASE_URL", "https://example.test"),
        ):
            result = await form_filler_outbound(payload, bg, request)

        self.assertEqual(result.get("call_id"), "call_fail_chrome")
        browser = result.get("browser") or {}
        self.assertFalse(browser.get("opened"))
        self.assertIn("Chrome was not found", browser.get("error") or "")
        self.assertEqual(
            result.get("chrome_url"),
            "https://example.test/form-filler/sessions/call_fail_chrome/chrome",
        )


class SessionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import pc_liveagent.live_form as live_form

        self.live_form = live_form
        async with live_form._sessions_lock:
            live_form._sessions.clear()
            live_form._active_call_id = ""

    async def asyncTearDown(self) -> None:
        async with self.live_form._sessions_lock:
            self.live_form._sessions.clear()
            self.live_form._active_call_id = ""

    async def test_end_live_form_idempotent_and_keeps_other_session(self) -> None:
        from pc_liveagent.live_form import (
            LiveFormSession,
            end_live_form,
            session_ids_with_pages,
        )

        a = LiveFormSession(call_id="call_a", url="https://example.test/a", status="open")
        b = LiveFormSession(call_id="call_b", url="https://example.test/b", status="open")
        page_a = AsyncMock()
        page_a.is_closed = lambda: False
        page_b = AsyncMock()
        page_b.is_closed = lambda: False
        a.page = page_a
        b.page = page_b
        async with self.live_form._sessions_lock:
            self.live_form._sessions["call_a"] = a
            self.live_form._sessions["call_b"] = b
            self.live_form._active_call_id = "call_b"

        await end_live_form("call_a")
        await end_live_form("call_a")  # idempotent
        self.assertIsNone(a.page)
        self.assertEqual(a.status, "ended")
        self.assertIs(b.page, page_b)
        self.assertEqual(session_ids_with_pages(), ["call_b"])

    async def test_disconnect_refuses_while_session_holds_page(self) -> None:
        import pc_liveagent.browser as browser_mod
        from pc_liveagent.live_form import LiveFormSession

        session = LiveFormSession(
            call_id="call_hold", url="https://example.test", status="open"
        )
        page = AsyncMock()
        page.is_closed = lambda: False
        session.page = page
        async with self.live_form._sessions_lock:
            self.live_form._sessions["call_hold"] = session

        fake_pw = AsyncMock()
        browser_mod._playwright = fake_pw
        browser_mod._browser = object()
        try:
            await browser_mod._disconnect_playwright(force=False, reason="test")
            # Should have skipped stop because session still owns a page.
            fake_pw.stop.assert_not_awaited()
            self.assertIs(browser_mod._playwright, fake_pw)

            await self.live_form.detach_all_pages_for_browser_restart(reason="test")
            await browser_mod._disconnect_playwright(force=True, reason="test")
            fake_pw.stop.assert_awaited()
            self.assertIsNone(browser_mod._playwright)
            self.assertIsNone(browser_mod._browser)
        finally:
            browser_mod._playwright = None
            browser_mod._browser = None

    async def test_welcome_closed_handler_marks_error_and_clears_page(self) -> None:
        from pc_liveagent.live_form import start_live_form

        page = AsyncMock()
        page.url = "https://my.pointcare.com/onboarding"
        page.is_closed = lambda: False

        async def boom(*_a, **_k):
            raise RuntimeError(
                "Page.wait_for_timeout: unable to perform operation ... the handler is closed"
            )

        with (
            patch("pc_liveagent.config.APPLICATION_TEST_MODE", False),
            patch("pc_liveagent.config.APPLICATION_AFTER_SIGNIN", False),
            patch("pc_liveagent.live_form.open_page", new=AsyncMock(return_value=page)),
            patch(
                "pc_liveagent.live_form._logout_for_fresh_signin",
                new=AsyncMock(),
            ),
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
                new=AsyncMock(side_effect=boom),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await start_live_form("call_welcome_fail", "https://my.pointcare.com/onboarding")
        self.assertIn("handler is closed", str(ctx.exception).lower())
        session = self.live_form._sessions.get("call_welcome_fail")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.status, "error")
        self.assertIsNone(session.page)
        self.assertIn("welcome:", session.error)

    async def test_concurrent_starts_serialize_shared_chrome(self) -> None:
        from pc_liveagent.live_form import start_live_form

        order: list[str] = []
        pages = {
            "call_1": AsyncMock(url="https://my.pointcare.com/auth", is_closed=lambda: False),
            "call_2": AsyncMock(url="https://my.pointcare.com/auth", is_closed=lambda: False),
        }

        async def fake_open(url=""):
            # Identify by active call being registered ends up in session; use stack.
            return AsyncMock(url="https://my.pointcare.com/auth", is_closed=lambda: False)

        started = 0

        async def fake_open_tracked(url=""):
            nonlocal started
            started += 1
            cur = started
            order.append(f"open:{cur}")
            await asyncio.sleep(0.05)
            page = AsyncMock()
            page.url = "https://my.pointcare.com/auth"
            page.is_closed = lambda: False
            return page

        async def fake_welcome(page, *, call_id=""):
            order.append(f"welcome:{call_id}")
            await asyncio.sleep(0.05)
            return ["Continue"]

        with (
            patch(
                "pc_liveagent.live_form.open_page",
                new=AsyncMock(side_effect=fake_open_tracked),
            ),
            patch(
                "pc_liveagent.live_form._logout_for_fresh_signin",
                new=AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form._clear_pointcare_session",
                new=AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form._page_wait",
                new=AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form.advance_welcome_screens",
                new=AsyncMock(side_effect=fake_welcome),
            ),
            patch(
                "pc_liveagent.live_form.settle_on_signin_page",
                new=AsyncMock(),
            ),
            patch(
                "pc_liveagent.live_form._already_signed_in",
                new=AsyncMock(return_value=False),
            ),
        ):
            t1 = asyncio.create_task(start_live_form("call_1", "https://example.test/1"))
            await asyncio.sleep(0.01)
            t2 = asyncio.create_task(start_live_form("call_2", "https://example.test/2"))
            s1, s2 = await asyncio.gather(t1, t2)

        # Second start must not interleave welcome before first finishes open+welcome.
        self.assertEqual(
            order,
            ["open:1", "welcome:call_1", "open:2", "welcome:call_2"],
        )
        self.assertEqual(s2.status, "open")
        # First session was superseded when second registered (same object as s1).
        self.assertEqual(s1.status, "superseded")
        self.assertEqual(self.live_form._sessions["call_1"].status, "superseded")
        self.assertIsNone(self.live_form._sessions["call_1"].page)

    async def test_failed_session_not_reported_opened(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, describe_live_browser

        session = LiveFormSession(
            call_id="call_err",
            url="https://my.pointcare.com/onboarding",
            status="error",
            welcome_done=False,
            error="welcome: BROWSER_CLOSED",
        )
        page = AsyncMock()
        page.url = "https://my.pointcare.com/onboarding"
        page.is_closed = lambda: False
        session.page = page
        with patch(
            "pc_liveagent.browser.cdp_target_for_url",
            return_value={},
        ):
            info = await describe_live_browser(session, call_id="call_err")
        self.assertFalse(info["opened"])
        self.assertIn("BROWSER_CLOSED", info["error"])


class ClosedChromeRecoveryTests(unittest.IsolatedAsyncioTestCase):
    """After the user closes agent Chrome, reopen maximized and recover the tab."""

    async def asyncTearDown(self) -> None:
        import pc_liveagent.browser as browser_mod

        browser_mod._browser = None
        browser_mod._playwright = None

    async def test_page_from_browser_reuses_context_never_new_context(self) -> None:
        import pc_liveagent.browser as browser_mod

        page = AsyncMock()
        page.is_closed = lambda: False
        context = AsyncMock()
        context.pages = [page]
        context.new_page = AsyncMock()
        browser = AsyncMock()
        browser.contexts = [context]
        browser.new_context = AsyncMock(
            side_effect=RuntimeError("new_context must not be called over CDP")
        )

        got = await browser_mod._page_from_browser(browser)
        self.assertIs(got, page)
        browser.new_context.assert_not_called()
        context.new_page.assert_not_called()

    async def test_open_page_relaunches_after_target_closed(self) -> None:
        import pc_liveagent.browser as browser_mod

        good = AsyncMock()
        good.url = "https://my.pointcare.com/onboarding"
        attempts = {"n": 0}

        async def once(url=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError(
                    "Browser.new_context: Target page, context or browser has been closed"
                )
            return good

        with (
            patch.object(browser_mod, "_open_page_once", side_effect=once),
            patch.object(browser_mod, "_restart_agent_chrome", new=AsyncMock()) as restart,
            patch.object(browser_mod, "_runtime_user_data_dir", return_value=Path("x")),
            patch.object(browser_mod, "_runtime_profile_directory", return_value="Default"),
        ):
            page = await browser_mod.open_page("https://my.pointcare.com/onboarding")

        self.assertIs(page, good)
        self.assertEqual(attempts["n"], 2)
        restart.assert_awaited_once()
        self.assertEqual(
            restart.await_args.kwargs.get("reason"), "open_page_closed_browser"
        )

    async def test_ensure_chrome_restarts_when_cdp_has_no_pages(self) -> None:
        import pc_liveagent.browser as browser_mod

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Profile_2"
            folder.mkdir()
            with (
                patch.object(browser_mod, "_runtime_user_data_dir", return_value=folder),
                patch.object(browser_mod, "_runtime_profile_directory", return_value="Default"),
                patch.object(browser_mod, "_cdp_up", return_value=True),
                patch.object(browser_mod, "_cdp_belongs_to", return_value=True),
                patch.object(browser_mod, "_cdp_has_page_targets", return_value=False),
                patch.object(browser_mod, "_restart_agent_chrome", new=AsyncMock()) as restart,
            ):
                await browser_mod._ensure_chrome_process(
                    "https://my.pointcare.com/onboarding"
                )

        restart.assert_awaited_once()
        self.assertEqual(
            restart.await_args.kwargs.get("reason"), "cdp_up_but_no_page_targets"
        )


class AgentProfileReuseTests(unittest.TestCase):
    def test_profile_directory_keeps_source_name(self) -> None:
        from pc_liveagent import browser as browser_mod

        with patch.object(browser_mod, "CHROME_PROFILE_DIRECTORY", "Profile 2"):
            self.assertEqual(
                browser_mod._runtime_profile_directory(Path("C:/agent")),
                "Profile 2",
            )

    def test_existing_agent_profile_is_not_replaced(self) -> None:
        from pc_liveagent import browser as browser_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = root / "PointcareAgentChrome" / "Profile_2"
            source = root / "Profile 2"
            source.mkdir()
            (source / "Preferences").write_text("{}", encoding="utf-8")
            dest = agent / "Profile 2"
            dest.mkdir(parents=True)
            (dest / "Preferences").write_text("keep-me", encoding="utf-8")
            with (
                patch.object(browser_mod, "CHROME_AGENT_DATA", agent),
                patch.object(browser_mod, "CHROME_SOURCE_PROFILE_DIR", source),
                patch.object(browser_mod, "CHROME_PROFILE_DIRECTORY", "Profile 2"),
                patch.object(browser_mod, "CHROME_PROFILE_DIR", root),
                patch.object(
                    browser_mod.shutil,
                    "rmtree",
                    side_effect=AssertionError("must not delete the agent profile"),
                ),
                patch.object(
                    browser_mod.subprocess,
                    "run",
                    side_effect=AssertionError("must not copy over an existing profile"),
                ),
            ):
                browser_mod._seed_agent_profile(agent)
            self.assertEqual(
                (dest / "Preferences").read_text(encoding="utf-8"),
                "keep-me",
            )

    def test_locked_copy_does_not_delete_or_replace_key(self) -> None:
        from pc_liveagent import browser as browser_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = root / "PointcareAgentChrome" / "Profile_2"
            source = root / "Profile 2"
            source.mkdir()
            (source / "Preferences").write_text("{}", encoding="utf-8")
            with (
                patch.object(browser_mod, "CHROME_AGENT_DATA", agent),
                patch.object(browser_mod, "CHROME_SOURCE_PROFILE_DIR", source),
                patch.object(browser_mod, "CHROME_PROFILE_DIRECTORY", "Profile 2"),
                patch.object(browser_mod, "CHROME_PROFILE_DIR", root),
                patch.object(browser_mod.sys, "platform", "win32"),
                patch.object(
                    browser_mod.shutil,
                    "rmtree",
                    side_effect=AssertionError("must not delete a locked copy"),
                ),
                patch.object(
                    browser_mod.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(args=[], returncode=8),
                ),
                patch.object(
                    browser_mod,
                    "_sync_os_crypt",
                    side_effect=AssertionError("must not replace the login key"),
                ),
            ):
                browser_mod._seed_agent_profile(agent)
            self.assertTrue((agent / ".seeded_from").is_file())


if __name__ == "__main__":
    unittest.main()
