"""Tests for Chrome executable discovery and form-filler session routes."""

from __future__ import annotations

import os
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


if __name__ == "__main__":
    unittest.main()
