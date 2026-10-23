"""HTML-fixture checks for relationship and age verification.

These load real DOM in headless Chrome. They do not open Pointcare.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from playwright.async_api import async_playwright


_REL_BUTTONS = ("Child", "Stepchild", "Grandchild", "Foster Child", "Other")


def _person_form_html(*, selected: str = "", hint: str = "", age_selected: bool = False) -> str:
    chips = []
    for label in _REL_BUTTONS:
        cls = "active" if label == selected else ""
        chips.append(
            f'<button type="button" class="{cls}" '
            f'onclick="this.classList.add(\'active\'); '
            f"var h=document.getElementById('rel-hint'); if(h) h.remove();\""
            f">{label}</button>"
        )
    age_cls = "active" if age_selected else ""
    hint_html = f'<p id="rel-hint">{hint}</p>' if hint else ""
    return f"""
    <section style="padding:12px">
      <p>Relationship to me</p>
      <div>{''.join(chips)}{hint_html}</div>
      <p>Age</p>
      <button type="button" id="age" class="{age_cls}">(21-63) Adult</button>
    </section>
    """


def _spouse_form_html(*, selected: str = "") -> str:
    buttons = []
    for label in ("Spouse", "Partner"):
        cls = "active" if label == selected else ""
        buttons.append(
            f'<button type="button" class="{cls}" '
            f"onclick=\"this.classList.add('active')\">{label}</button>"
        )
    return f"""
    <section style="padding:12px">
      <p>Are they your spouse or your partner?</p>
      <div>{''.join(buttons)}</div>
      <button type="button" id="age" class="active">(21-63) Adult</button>
    </section>
    """


class RelationshipDomTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            channel="chrome", headless=True
        )
        self._page = await self._browser.new_page()

    async def asyncTearDown(self) -> None:
        await self._browser.close()
        await self._playwright.stop()

    async def test_plain_text_please_select_is_a_validation_hint(self) -> None:
        from pc_liveagent.live_form import _visible_validation_hints

        await self._page.set_content(
            _person_form_html(hint="Please select an option")
        )
        hints = (await _visible_validation_hints(self._page)).lower()
        self.assertIn("please select an option", hints)
        self.assertNotIn("role=alert", _person_form_html(hint="Please select an option"))

    async def test_unmarked_click_does_not_count_as_selected(self) -> None:
        from pc_liveagent.live_form import _ensure_relationship_choice_selected

        await self._page.set_content(
            """
            <section style="padding:12px">
              <p>Relationship to me</p>
              <button type="button" id="gc">Grandchild</button>
              <p>Please select an option</p>
            </section>
            """
        )
        await self._page.evaluate("() => { window.__clicks = 0; }")
        await self._page.eval_on_selector(
            "#gc",
            "el => el.addEventListener('click', () => { window.__clicks += 1; })",
        )
        with self.assertRaises(RuntimeError) as ctx:
            await _ensure_relationship_choice_selected(self._page, "Grandchild")
        self.assertIn("couldn't select", str(ctx.exception).lower())
        clicks = await self._page.evaluate("() => window.__clicks")
        self.assertEqual(clicks, 1)

    async def test_spouse_click_without_aria_succeeds_when_prompt_clears(self) -> None:
        from pc_liveagent.live_form import _ensure_relationship_choice_selected

        await self._page.set_content(
            """
            <section style="padding:12px">
              <p>Are they your spouse or your partner?</p>
              <button type="button" id="sp"
                onclick="var h=document.getElementById('rel-hint'); if(h) h.remove();">Spouse</button>
              <p id="rel-hint">Please select an option</p>
            </section>
            """
        )
        await self._page.evaluate("() => { window.__clicks = 0; }")
        await self._page.eval_on_selector(
            "#sp",
            "el => el.addEventListener('click', () => { window.__clicks += 1; })",
        )
        await _ensure_relationship_choice_selected(self._page, "Spouse")
        clicks = await self._page.evaluate("() => window.__clicks")
        self.assertEqual(clicks, 1)
        hint = await self._page.evaluate(
            "() => document.getElementById('rel-hint')"
        )
        self.assertIsNone(hint)

    async def test_new_person_form_clears_both_name_inputs(self) -> None:
        from pc_liveagent.live_form import _clear_new_person_name_fields

        await self._page.set_content(
            """
            <label>First Name <input id="first" value="Pat" /></label>
            <label>Last Name <input id="last" value="Lee" /></label>
            """
        )
        await _clear_new_person_name_fields(self._page)
        first, last = await self._page.evaluate(
            "() => [document.querySelector('#first').value, document.querySelector('#last').value]"
        )
        self.assertEqual(first, "")
        self.assertEqual(last, "")

    async def test_active_class_without_aria_is_not_clicked_again(self) -> None:
        from pc_liveagent.live_form import _ensure_relationship_choice_selected

        await self._page.set_content(_person_form_html(selected="Stepchild"))
        await self._page.evaluate("() => { window.__clicks = 0; }")
        await self._page.evaluate(
            """() => {
              document.querySelectorAll('button').forEach((el) => {
                el.addEventListener('click', () => { window.__clicks += 1; });
              });
            }"""
        )
        await _ensure_relationship_choice_selected(self._page, "Stepchild")
        clicks = await self._page.evaluate("() => window.__clicks")
        self.assertEqual(clicks, 0)

    async def test_each_relationship_chip_selects_once(self) -> None:
        from pc_liveagent.live_form import _ensure_relationship_choice_selected

        for label in _REL_BUTTONS:
            await self._page.set_content(_person_form_html())
            await _ensure_relationship_choice_selected(self._page, label)
            selected = await self._page.evaluate(
                """(label) => {
                  const btn = [...document.querySelectorAll('button')]
                    .find((el) => (el.innerText || '').trim() === label);
                  return !!(btn && btn.classList.contains('active'));
                }""",
                label,
            )
            self.assertTrue(selected, label)

    async def test_spouse_and_partner_select_once(self) -> None:
        from pc_liveagent.live_form import _ensure_relationship_choice_selected

        for label in ("Spouse", "Partner"):
            await self._page.set_content(_spouse_form_html())
            await _ensure_relationship_choice_selected(self._page, label)
            selected = await self._page.evaluate(
                """(label) => {
                  const btn = [...document.querySelectorAll('button')]
                    .find((el) => (el.innerText || '').trim() === label);
                  return btn.classList.contains('active');
                }""",
                label,
            )
            self.assertTrue(selected, label)

    async def test_confirm_yes_blocks_when_page_still_requires_relationship(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _submit_eligibility_answer

        await self._page.set_content(
            _person_form_html(hint="Please select an option", age_selected=True)
        )
        session = LiveFormSession(
            call_id="dom-confirm-block",
            url="https://example.test",
            elig_step="person_confirm",
            after_person="disabled",
        )
        session.filled.update(
            {
                "person_first": "Sam",
                "person_last": "Lee",
                "person_rel": "Foster Child",
                "person_age": "(21-63) Adult",
            }
        )
        cont = AsyncMock()
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                AsyncMock(return_value="person_form"),
            ),
            patch("pc_liveagent.live_form._mark"),
            patch("pc_liveagent.live_form._click_continue_to", cont),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await _submit_eligibility_answer(
                    "dom-confirm-block", session, self._page, "Yes", 0.0
                )
        self.assertIn("relationship", str(ctx.exception).lower())
        cont.assert_not_awaited()
        self.assertEqual(session.elig_step, "person_confirm")
        clicks = await self._page.evaluate(
            """() => [...document.querySelectorAll('button')]
              .filter((el) => el.classList.contains('active'))
              .map((el) => (el.innerText || '').trim())"""
        )
        self.assertNotIn("Foster Child", clicks)

    async def test_confirm_yes_continues_when_chip_is_visually_selected(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _submit_eligibility_answer

        await self._page.set_content(
            _person_form_html(selected="Grandchild", age_selected=True)
        )
        session = LiveFormSession(
            call_id="dom-confirm-ok",
            url="https://example.test",
            elig_step="person_confirm",
            after_person="disabled",
        )
        session.filled.update(
            {
                "person_first": "Sam",
                "person_last": "Lee",
                "person_rel": "Grandchild",
                "person_age": "(21-63) Adult",
            }
        )
        cont = AsyncMock()
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                AsyncMock(return_value="person_form"),
            ),
            patch("pc_liveagent.live_form._mark"),
            patch("pc_liveagent.live_form._click_continue_to", cont),
        ):
            result = await _submit_eligibility_answer(
                "dom-confirm-ok", session, self._page, "Yes", 0.0
            )
        cont.assert_awaited()
        self.assertTrue(result.get("ok"))
        self.assertEqual(session.elig_step, "disabled")

    async def test_age_chip_is_not_clicked_when_state_is_uncertain(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assert_member_form_ready_before_continue

        await self._page.set_content(_person_form_html(selected="Child", age_selected=False))
        await self._page.evaluate("() => { window.__ageClicks = 0; }")
        await self._page.eval_on_selector(
            "#age",
            "el => el.addEventListener('click', () => { window.__ageClicks += 1; })",
        )
        session = LiveFormSession(
            call_id="dom-age",
            url="https://example.test",
            elig_step="person_confirm",
        )
        session.filled.update(
            {
                "person_rel": "Child",
                "person_age": "(21-63) Adult",
            }
        )
        await _assert_member_form_ready_before_continue(session, self._page, "person")
        clicks = await self._page.evaluate("() => window.__ageClicks")
        self.assertEqual(clicks, 0)

    async def test_selected_age_without_aria_is_not_clicked(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assert_member_form_ready_before_continue

        await self._page.set_content(
            _spouse_form_html(selected="Partner")
        )
        await self._page.evaluate("() => { window.__ageClicks = 0; }")
        await self._page.eval_on_selector(
            "#age",
            "el => el.addEventListener('click', () => { window.__ageClicks += 1; })",
        )
        session = LiveFormSession(
            call_id="dom-age-ok",
            url="https://example.test",
            elig_step="spouse_confirm",
        )
        session.filled.update(
            {
                "spouse_role": "Partner",
                "spouse_age": "(21-63) Adult",
            }
        )
        await _assert_member_form_ready_before_continue(session, self._page, "spouse")
        clicks = await self._page.evaluate("() => window.__ageClicks")
        self.assertEqual(clicks, 0)

    async def test_confirm_continues_when_chip_has_no_marker_and_no_prompt(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _submit_eligibility_answer

        await self._page.set_content(
            """
            <section style="padding:12px">
              <p>Are they your spouse or your partner?</p>
              <button type="button">Spouse</button>
              <button type="button" id="age" class="active">(21-63) Adult</button>
            </section>
            """
        )
        session = LiveFormSession(
            call_id="dom-confirm-quiet",
            url="https://example.test",
            elig_step="spouse_confirm",
        )
        session.filled.update(
            {"spouse_role": "Spouse", "spouse_age": "(21-63) Adult"}
        )
        cont = AsyncMock()
        with (
            patch(
                "pc_liveagent.live_form._eligibility_screen",
                AsyncMock(return_value="spouse_form"),
            ),
            patch("pc_liveagent.live_form._mark"),
            patch("pc_liveagent.live_form._click_continue_to", cont),
        ):
            result = await _submit_eligibility_answer(
                "dom-confirm-quiet", session, self._page, "Yes", 0.0
            )
        cont.assert_awaited()
        self.assertTrue(result.get("ok"))

    async def test_saved_age_without_marker_confirms_without_click(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assert_member_form_ready_before_continue

        await self._page.set_content(
            """
            <section style="padding:12px">
              <p>Are they your spouse or your partner?</p>
              <button type="button" class="active">Spouse</button>
              <button type="button" id="age">(21-63) Adult</button>
            </section>
            """
        )
        await self._page.evaluate("() => { window.__ageClicks = 0; }")
        await self._page.eval_on_selector(
            "#age",
            "el => el.addEventListener('click', () => { window.__ageClicks += 1; })",
        )
        session = LiveFormSession(
            call_id="dom-age-quiet",
            url="https://example.test",
            elig_step="spouse_confirm",
        )
        session.filled.update(
            {"spouse_role": "Spouse", "spouse_age": "(21-63) Adult"}
        )
        await _assert_member_form_ready_before_continue(session, self._page, "spouse")
        clicks = await self._page.evaluate("() => window.__ageClicks")
        self.assertEqual(clicks, 0)

    async def test_required_age_text_blocks_continue_without_click(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assert_member_form_ready_before_continue

        await self._page.set_content(
            _person_form_html(selected="Child", hint="Please select an option")
        )
        await self._page.evaluate("() => { window.__ageClicks = 0; }")
        await self._page.eval_on_selector(
            "#age",
            "el => el.addEventListener('click', () => { window.__ageClicks += 1; })",
        )
        session = LiveFormSession(
            call_id="dom-age-required",
            url="https://example.test",
            elig_step="person_confirm",
        )
        session.filled.update(
            {"person_rel": "Child", "person_age": "(21-63) Adult"}
        )
        with self.assertRaises(RuntimeError) as ctx:
            await _assert_member_form_ready_before_continue(session, self._page, "person")
        self.assertTrue(str(ctx.exception))
        clicks = await self._page.evaluate("() => window.__ageClicks")
        self.assertEqual(clicks, 0)

    async def test_missing_saved_age_blocks_continue(self) -> None:
        from pc_liveagent.live_form import LiveFormSession, _assert_member_form_ready_before_continue

        await self._page.set_content(_person_form_html(selected="Child"))
        session = LiveFormSession(
            call_id="dom-age-missing",
            url="https://example.test",
            elig_step="person_confirm",
        )
        session.filled["person_rel"] = "Child"
        with self.assertRaises(RuntimeError) as ctx:
            await _assert_member_form_ready_before_continue(session, self._page, "person")
        self.assertIn("age", str(ctx.exception).lower())

    async def test_new_person_form_clears_delayed_autofill_only_once(self) -> None:
        import asyncio

        from pc_liveagent.live_form import LiveFormSession, _clear_new_person_name_fields

        await self._page.set_content(
            """
            <div style="display:none">
              <label>First Name <input id="hidden-first" value="Hidden" /></label>
              <label>Last Name <input id="hidden-last" value="Person" /></label>
            </div>
            <section style="padding:12px">
              <label>First Name <input id="first" value="Pat" /></label>
              <label>Last Name <input id="last" value="Lee" /></label>
            </section>
            """
        )
        session = LiveFormSession(
            call_id="dom-names",
            url="https://example.test",
            elig_step="spouse_first",
        )

        async def late_fill() -> None:
            await asyncio.sleep(0.25)
            await self._page.evaluate(
                "() => { const el = document.querySelector('#first'); if (el) el.value = 'Old'; }"
            )

        task = asyncio.create_task(late_fill())
        await _clear_new_person_name_fields(self._page, session, kind="spouse")
        await task
        visible = await self._page.evaluate(
            "() => [document.querySelector('#first').value, document.querySelector('#last').value]"
        )
        hidden = await self._page.evaluate(
            "() => [document.querySelector('#hidden-first').value, document.querySelector('#hidden-last').value]"
        )
        self.assertEqual(visible, ["", ""])
        self.assertEqual(hidden, ["Hidden", "Person"])
        await self._page.evaluate(
            "() => { document.querySelector('#first').value = 'Pat'; document.querySelector('#last').value = 'Lee'; }"
        )
        await _clear_new_person_name_fields(self._page, session, kind="spouse")
        again = await self._page.evaluate(
            "() => [document.querySelector('#first').value, document.querySelector('#last').value]"
        )
        self.assertEqual(again, ["Pat", "Lee"])
        session.filled["spouse_first"] = "Alex"
        session.filled.pop("name_fields_reset_for", None)
        await _clear_new_person_name_fields(self._page, session, kind="spouse")
        kept = await self._page.evaluate("() => document.querySelector('#first').value")
        self.assertEqual(kept, "Pat")


if __name__ == "__main__":
    unittest.main()
