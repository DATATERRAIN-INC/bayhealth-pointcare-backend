import unittest

from pc_liveagent.live_form import (
    ELIG_QUESTIONS,
    LiveFormSession,
    _age_choice,
    _fuzzy_choice_label,
    _map_age_label,
    _next_elig_step,
    _screen_first_step,
    _step_group,
    _yes_no_acceptable_screens,
    _yes_no_transition,
)


class EligibilityStateTests(unittest.TestCase):
    def session(self, step: str) -> LiveFormSession:
        return LiveFormSession(call_id="test", url="https://example.test", elig_step=step)

    def test_marital_branching(self) -> None:
        self.assertEqual(_next_elig_step(self.session("marital"), "married"), "spouse_first")
        self.assertEqual(
            _next_elig_step(self.session("marital"), "never married"), "children"
        )

    def test_consecutive_no_answers_advance_different_steps(self) -> None:
        self.assertEqual(_next_elig_step(self.session("children"), "no"), "student")
        self.assertEqual(_next_elig_step(self.session("student"), "no"), "disabled")

    def test_age_answers_map_to_visible_pointcare_choices(self) -> None:
        self.assertEqual(_age_choice("17"), "(0-18) Child")
        self.assertEqual(_age_choice("20 years old"), "(19-20) Young Adult")
        self.assertEqual(_age_choice("42"), "(21-63) Adult")
        self.assertEqual(_age_choice("64"), "(64) Turning Senior")
        self.assertEqual(_age_choice("70"), "(65+) Senior")

    def test_page_groups_own_grouped_form_questions(self) -> None:
        self.assertEqual(_step_group("spouse_last"), "spouse_form")
        self.assertEqual(_step_group("child_rel"), "child_form")
        self.assertEqual(_step_group("person_age"), "person_form")
        self.assertEqual(_step_group("income_range"), "income")

    def test_browser_screen_can_recover_server_step(self) -> None:
        session = self.session("marital")
        self.assertEqual(_screen_first_step("spouse_form", session), "spouse_first")
        self.assertEqual(_screen_first_step("children", session), "children")


class RelativesIncomeTransitionTests(unittest.TestCase):
    def test_relatives_no_to_support(self) -> None:
        self.assertEqual(_yes_no_transition("relatives", False, "support"), "support")
        self.assertIn("support", _yes_no_acceptable_screens("relatives", False))

    def test_relatives_no_skips_to_income(self) -> None:
        self.assertEqual(
            _yes_no_transition("relatives", False, "income"), "income_when"
        )
        self.assertIn("income", _yes_no_acceptable_screens("relatives", False))
        self.assertIn("period", ELIG_QUESTIONS["income_when"].lower())

    def test_relatives_unexpected_screen_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            _yes_no_transition("relatives", False, "programs")


class ChildAgeMappingTests(unittest.TestCase):
    CHILD_AGES = [
        "(0-1) Baby",
        "(1-5) Toddler",
        "(6-12) Child",
        "(13-15) Teen",
        "(16-18) Teen",
    ]
    ADULT_STYLE = [
        "(0-5) Child",
        "(6-20) Adult",
        "(21-63) Adult",
        "(65+) Senior",
    ]

    def test_sixteen_to_eight_does_not_map_to_baby(self) -> None:
        mapped = _map_age_label("Sixteen to eight.", self.CHILD_AGES)
        self.assertIsNone(mapped)
        fuzzy = _fuzzy_choice_label("Sixteen to eight.", self.CHILD_AGES)
        self.assertNotEqual(fuzzy, "(0-1) Baby")
        self.assertEqual(fuzzy, "Sixteen to eight.")

    def test_ambiguous_age_returns_none(self) -> None:
        self.assertIsNone(_map_age_label("maybe around there", self.CHILD_AGES))

    def test_sixteen_to_eighteen_maps_to_teen_band(self) -> None:
        self.assertEqual(
            _map_age_label("Sixteen to eighteen.", self.CHILD_AGES),
            "(16-18) Teen",
        )
        self.assertEqual(
            _map_age_label("16 to 18", self.CHILD_AGES),
            "(16-18) Teen",
        )

    def test_fifteen_maps_to_containing_band(self) -> None:
        self.assertEqual(_map_age_label("15", self.CHILD_AGES), "(13-15) Teen")
        self.assertEqual(
            _map_age_label("I'm fifteen years old", self.ADULT_STYLE),
            "(6-20) Adult",
        )

    def test_six_to_twenty_maps_to_displayed_range(self) -> None:
        self.assertEqual(
            _map_age_label("Six to twenty", self.ADULT_STYLE),
            "(6-20) Adult",
        )

    def test_income_mapper_not_applied_to_age_chips(self) -> None:
        # Regression: income $ tolerance previously scored Baby as a match.
        resolved = _fuzzy_choice_label(
            "Sixteen to eight.",
            self.CHILD_AGES + ["First Name", "Last Name", "Grandchild"],
        )
        self.assertNotIn("Baby", resolved)

    def test_age_boundary_and_spoken_forms(self) -> None:
        self.assertEqual(_map_age_label("13", self.CHILD_AGES), "(13-15) Teen")
        self.assertEqual(_map_age_label("15 years old", self.CHILD_AGES), "(13-15) Teen")
        self.assertEqual(
            _map_age_label("six through twenty", self.ADULT_STYLE),
            "(6-20) Adult",
        )

    def test_overlapping_age_keywords_clarify(self) -> None:
        # Two Teen chips — bare "teen" is ambiguous.
        self.assertIsNone(_map_age_label("teen", self.CHILD_AGES))

    def test_twenty_five_maps_to_adult_not_young_adult(self) -> None:
        spouse_ages = [
            "(0-18) Child",
            "(19-20) Young Adult",
            "(21-63) Adult",
            "(64) Turning Senior",
            "(65+) Senior",
        ]
        from pc_liveagent.live_form import _extract_age_numbers

        self.assertEqual(_extract_age_numbers("Twenty five"), [25])
        self.assertEqual(_extract_age_numbers("twenty-five"), [25])
        self.assertEqual(_extract_age_numbers("25 years old"), [25])
        self.assertEqual(_map_age_label("Twenty five", spouse_ages), "(21-63) Adult")
        self.assertEqual(_map_age_label("25", spouse_ages), "(21-63) Adult")
        # Age 25 must never select Young Adult unless that chip contains 25.
        self.assertNotEqual(
            _map_age_label("Twenty five", spouse_ages), "(19-20) Young Adult"
        )
        self.assertIsNone(
            _map_age_label("25", ["(19-20) Young Adult", "(0-18) Child"])
        )

    def test_ambiguous_age_speech_does_not_click(self) -> None:
        spouse_ages = [
            "(0-18) Child",
            "(19-20) Young Adult",
            "(21-63) Adult",
            "(64) Turning Senior",
            "(65+) Senior",
        ]
        self.assertIsNone(
            _map_age_label("Twenty twenty two thirty", spouse_ages)
        )
        self.assertIsNone(_map_age_label("Twenty two thirty", spouse_ages))

    def test_form_captions_are_not_age_choices(self) -> None:
        from pc_liveagent.live_form import _age_option_labels, _looks_like_age_label

        scraped = [
            "First Name",
            "Last Name",
            "Living with me",
            "Spouse",
            "Partner",
            "Age",
            "Child",
            "(19-20) Young Adult",
            "(21-63) Adult",
        ]
        opts = _age_option_labels(scraped)
        self.assertEqual(opts, ["(19-20) Young Adult", "(21-63) Adult"])
        for caption in ("First Name", "Last Name", "Spouse", "Partner", "Age", "Child"):
            self.assertFalse(_looks_like_age_label(caption))
        self.assertIsNone(_map_age_label("Twenty five", scraped[:6]))


class IncomeRangeMappingTests(unittest.TestCase):
    LABELS = [
        "$0 – $30,000",
        "$30,001 – $45,000",
        "$45,001 – $60,000",
        "$60,001+",
    ]
    TIGHT = [
        "$0 – $12,000",
        "$12,001 – $39,900",
        "$39,901 – $50,000",
        "$50,001+",
    ]

    def test_spoken_forms_map_to_containing_band(self) -> None:
        from pc_liveagent.live_form import _map_income_range_label, _parse_spoken_income_interval

        self.assertEqual(_parse_spoken_income_interval("40000-50000"), (40000, 50000))
        self.assertEqual(
            _parse_spoken_income_interval("40,000 to 50,000"), (40000, 50000)
        )
        self.assertEqual(
            _parse_spoken_income_interval("forty thousand to fifty thousand"),
            (40000, 50000),
        )
        self.assertEqual(
            _parse_spoken_income_interval("forty to fifty thousand"),
            (40000, 50000),
        )
        self.assertEqual(_parse_spoken_income_interval("40k to 50k"), (40000, 50000))
        self.assertEqual(
            _parse_spoken_income_interval("between 40 and 50 thousand"),
            (40000, 50000),
        )
        self.assertEqual(
            _map_income_range_label("40000-50000", self.TIGHT),
            "$39,901 – $50,000",
        )
        self.assertEqual(
            _map_income_range_label("40k to 50k", self.TIGHT),
            "$39,901 – $50,000",
        )

    def test_fully_contained_range(self) -> None:
        from pc_liveagent.live_form import _map_income_range_label

        self.assertEqual(
            _map_income_range_label("10000 to 20000", self.LABELS),
            "$0 – $30,000",
        )

    def test_ambiguous_overlap_does_not_pick(self) -> None:
        from pc_liveagent.live_form import _map_income_range_label

        # Spans two chips equally — must clarify, not pick $0–$30k.
        self.assertEqual(_map_income_range_label("40000-50000", self.LABELS), "")

    def test_no_match_never_selects_first_chip(self) -> None:
        from pc_liveagent.live_form import _map_income_range_label

        closed = [
            "$0 – $30,000",
            "$30,001 – $45,000",
            "$45,001 – $60,000",
        ]
        mapped = _map_income_range_label("200000 to 250000", closed)
        self.assertEqual(mapped, "")
        self.assertNotEqual(mapped, closed[0])
        # Far above an open-ended floor must not silently map to $+ chip.
        self.assertEqual(
            _map_income_range_label("200000 to 250000", self.LABELS),
            "",
        )


if __name__ == "__main__":
    unittest.main()
