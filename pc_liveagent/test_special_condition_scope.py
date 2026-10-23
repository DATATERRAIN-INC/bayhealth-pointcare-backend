"""Special Cases member clicks stay inside the active condition card."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from pc_liveagent.live_form import LiveFormSession


class SpecialConditionScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cervical_member_does_not_change_blind_selection(self) -> None:
        from pc_liveagent.live_form import _assign_special_person_member

        session = LiveFormSession(call_id="scope", url="https://example.test")
        session.filled["special_conditions"] = (
            "Someone is blind, Someone has cervical cancer"
        )
        session.filled["special_active_conditions"] = "Someone has cervical cancer"
        session.filled["special_person_assigned"] = "Pillow W."
        seen: list[str] = []

        async def _wait(_page, *, timeout_ms=8000, call_id="", condition=""):
            self.assertIn("cervical", condition.lower())
            self.assertNotIn("blind", condition.lower())
            return {
                "members": [
                    {"label": "Mickey M.", "checked": False, "dupIndex": 0},
                ],
                "strategy": "condition_scope",
            }

        async def _click(_page, label, *, dup_index=0, condition=""):
            seen.append(condition)
            self.assertEqual(label, "Mickey M.")
            self.assertIn("cervical", condition.lower())
            return True

        page = AsyncMock()
        with (
            patch("pc_liveagent.live_form._wait_for_special_apply_to_members", new=_wait),
            patch("pc_liveagent.live_form._click_special_person_member", new=_click),
        ):
            await _assign_special_person_member(page, "Mickey", session)
        self.assertEqual(seen, ["Someone has cervical cancer"])
        assigned = session.filled["special_person_assigned"]
        self.assertIn("Pillow W.", assigned)
        self.assertIn("Mickey M.", assigned)

    async def test_condition_name_asks_for_a_member_and_does_not_click(self) -> None:
        from pc_liveagent.live_form import _assign_special_person_member

        session = LiveFormSession(call_id="cond", url="https://example.test")
        session.elig_step = "special_person"
        session.filled["special_active_conditions"] = "Someone has cervical cancer"
        click = AsyncMock(side_effect=AssertionError("must not click"))
        page = AsyncMock()
        with patch("pc_liveagent.live_form._click_special_person_member", click):
            with self.assertRaises(RuntimeError) as ctx:
                await _assign_special_person_member(page, "cervical cancer", session)
        self.assertIn("household member", str(ctx.exception).lower())
        click.assert_not_awaited()

    async def test_missing_condition_list_clarifies_without_another_card(self) -> None:
        from pc_liveagent.live_form import _assign_special_person_member

        session = LiveFormSession(call_id="missing", url="https://example.test")
        session.filled["special_active_conditions"] = "Someone has cervical cancer"

        async def _wait(_page, *, timeout_ms=8000, call_id="", condition=""):
            self.assertIn("cervical", condition.lower())
            return {"members": [], "strategy": "condition_scope_missing"}

        click = AsyncMock(side_effect=AssertionError("must not click"))
        with (
            patch("pc_liveagent.live_form._wait_for_special_apply_to_members", new=_wait),
            patch("pc_liveagent.live_form._click_special_person_member", click),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _assign_special_person_member(
                    AsyncMock(), "Mickey", session
                )
        self.assertIn("cervical", str(ctx.exception).lower())
        click.assert_not_awaited()

    async def test_ambiguous_names_do_not_click(self) -> None:
        from pc_liveagent.live_form import _assign_special_person_member

        session = LiveFormSession(call_id="ambig", url="https://example.test")
        session.filled["special_active_conditions"] = "Someone has cervical cancer"

        async def _wait(_page, *, timeout_ms=8000, call_id="", condition=""):
            return {
                "members": [
                    {"label": "Mickey M.", "checked": False, "dupIndex": 0},
                    {"label": "Mickey W.", "checked": False, "dupIndex": 0},
                ],
                "strategy": "condition_scope",
            }

        click = AsyncMock(side_effect=AssertionError("must not click"))
        with (
            patch("pc_liveagent.live_form._wait_for_special_apply_to_members", new=_wait),
            patch("pc_liveagent.live_form._click_special_person_member", click),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _assign_special_person_member(AsyncMock(), "Mickey", session)
        self.assertIn("several people", str(ctx.exception).lower())
        click.assert_not_awaited()

    async def test_failed_verification_does_not_record_the_member(self) -> None:
        from pc_liveagent.live_form import _assign_special_person_member

        session = LiveFormSession(call_id="verify", url="https://example.test")
        session.filled["special_active_conditions"] = "Someone has cervical cancer"

        async def _wait(_page, *, timeout_ms=8000, call_id="", condition=""):
            return {
                "members": [{"label": "Mickey M.", "checked": False, "dupIndex": 0}],
                "strategy": "condition_scope",
            }

        async def _click(_page, label, *, dup_index=0, condition=""):
            self.assertIn("cervical", condition.lower())
            return False

        async def _discover(_page, condition=""):
            self.assertIn("cervical", condition.lower())
            return {"members": [], "strategy": "condition_scope"}

        with (
            patch("pc_liveagent.live_form._wait_for_special_apply_to_members", new=_wait),
            patch("pc_liveagent.live_form._click_special_person_member", new=_click),
            patch("pc_liveagent.live_form._discover_special_person_members", new=_discover),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _assign_special_person_member(AsyncMock(), "Mickey", session)
        self.assertIn("couldn't select", str(ctx.exception).lower())
        self.assertNotIn("Mickey", session.filled.get("special_person_assigned") or "")

    def test_discover_script_scopes_before_any_pagewide_fallback(self) -> None:
        from pc_liveagent.live_form import _SPECIAL_MEMBER_DISCOVER_JS, _SPECIAL_MEMBER_CLICK_JS

        scoped = _SPECIAL_MEMBER_DISCOVER_JS.find("if (conditionLabel)")
        pagewide = _SPECIAL_MEMBER_DISCOVER_JS.find("Strategy 1:")
        self.assertGreater(scoped, 0)
        self.assertGreater(pagewide, scoped)
        self.assertIn("condition_scope_missing", _SPECIAL_MEMBER_DISCOVER_JS)
        self.assertIn("condition_scope_missing", _SPECIAL_MEMBER_CLICK_JS)
        self.assertNotIn("return open || document.body", _SPECIAL_MEMBER_CLICK_JS)
        self.assertIn("if (condition) return findConditionCard(condition);", _SPECIAL_MEMBER_CLICK_JS)


class SpecialConditionDomTests(unittest.IsolatedAsyncioTestCase):
    """Run the real browser scripts against two expanded condition cards."""

    async def asyncSetUp(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            channel="chrome", headless=True
        )
        self._page = await self._browser.new_page()

    async def asyncTearDown(self) -> None:
        await self._browser.close()
        await self._playwright.stop()

    async def test_cervical_click_leaves_the_blind_checkbox_checked(self) -> None:
        from pc_liveagent.live_form import (
            _SPECIAL_MEMBER_CLICK_JS,
            _SPECIAL_MEMBER_DISCOVER_JS,
            _click_special_person_member,
        )

        await self._page.set_content(
            """
            <div id="blind">
              <h3>Someone is blind</h3>
              <p>Who does this apply to?</p>
              <label><input id="blind-box" type="checkbox" checked> Mickey M.</label>
            </div>
            <div id="cervical">
              <h3>Someone has cervical cancer</h3>
              <p>Who does this apply to?</p>
              <label><input id="cervical-box" type="checkbox"> Mickey M.</label>
            </div>
            """
        )
        discovered = await self._page.evaluate(
            _SPECIAL_MEMBER_DISCOVER_JS, "Someone has cervical cancer"
        )
        self.assertEqual(discovered["strategy"], "condition_scope")
        self.assertEqual(
            [row["label"] for row in discovered["members"]],
            ["Mickey M."],
        )
        self.assertFalse(discovered["members"][0]["checked"])
        clicked = await _click_special_person_member(
            self._page,
            "Mickey M.",
            condition="Someone has cervical cancer",
        )
        self.assertTrue(clicked)
        blind_checked, cervical_checked = await self._page.evaluate(
            """() => [
              document.querySelector('#blind-box').checked,
              document.querySelector('#cervical-box').checked,
            ]"""
        )
        self.assertTrue(blind_checked)
        self.assertTrue(cervical_checked)
        missed = await self._page.evaluate(
            _SPECIAL_MEMBER_CLICK_JS,
            {
                "label": "Mickey M.",
                "dupIndex": 0,
                "mode": "click",
                "condition": "not a listed condition",
            },
        )
        self.assertFalse(missed["ok"])
        self.assertEqual(missed["reason"], "condition_scope_missing")

    async def test_shared_parent_still_scopes_second_condition(self) -> None:
        """Pointcare may nest both cards under one wrapper — still find disabled."""
        from pc_liveagent.live_form import _SPECIAL_MEMBER_DISCOVER_JS

        await self._page.set_content(
            """
            <section id="special-wrap">
              <div id="blind">
                <h3>Someone is blind</h3>
                <p>Who does this apply to?</p>
                <label><input type="checkbox" checked> Mickey M.</label>
              </div>
              <div id="disabled">
                <h3>Someone is disabled (SSDI)</h3>
                <p>Who does this apply to?</p>
                <label><input type="checkbox"> Agent H.</label>
              </div>
            </section>
            """
        )
        discovered = await self._page.evaluate(
            _SPECIAL_MEMBER_DISCOVER_JS, "Someone is disabled (SSDI)"
        )
        self.assertEqual(discovered["strategy"], "condition_scope")
        labels = [row["label"] for row in discovered["members"]]
        self.assertIn("Agent H.", labels)
        # Must not return only the blind card's members as the disabled list.
        self.assertNotEqual(labels, ["Mickey M."])


if __name__ == "__main__":
    unittest.main()
