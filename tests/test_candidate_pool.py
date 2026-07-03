import unittest

import pandas as pd

from src.candidate_pool import (
    build_candidate_pool,
    get_candidate_display_label,
    get_valid_candidates,
    normalize_candidate_key,
)


class CandidatePoolTest(unittest.TestCase):
    def test_july_excludes_hard_seasonal_candidates(self):
        candidates = set(get_valid_candidates("2026-07"))

        self.assertIn("bounty_hunter", candidates)
        self.assertIn("trader", candidates)
        self.assertIn("collector", candidates)
        self.assertIn("moonshiner", candidates)
        self.assertIn("naturalist", candidates)
        self.assertIn("all_role", candidates)
        self.assertIn("free_roam", candidates)
        self.assertIn("races", candidates)
        self.assertIn("blood_money", candidates)
        self.assertIn("telegram_missions", candidates)
        self.assertIn("call_to_arms", candidates)
        self.assertIn("featured_series", candidates)
        self.assertIn("other_non_role", candidates)

        self.assertNotIn("halloween", candidates)
        self.assertNotIn("halloween_call_to_arms", candidates)
        self.assertNotIn("holiday", candidates)
        self.assertNotIn("holiday_call_to_arms", candidates)
        self.assertNotIn("holiday_rewards", candidates)

    def test_october_includes_halloween_candidates_only(self):
        candidates = set(get_valid_candidates("2026-10"))

        self.assertIn("halloween", candidates)
        self.assertIn("halloween_call_to_arms", candidates)
        self.assertNotIn("holiday", candidates)
        self.assertNotIn("holiday_call_to_arms", candidates)

    def test_december_includes_holiday_candidates_only(self):
        candidates = set(get_valid_candidates("2026-12"))

        self.assertIn("holiday", candidates)
        self.assertIn("holiday_call_to_arms", candidates)
        self.assertNotIn("halloween", candidates)
        self.assertNotIn("halloween_call_to_arms", candidates)
        self.assertNotIn("holiday_rewards", candidates)

    def test_holiday_rewards_alias_is_not_duplicate_candidate(self):
        self.assertEqual(
            normalize_candidate_key("holiday_rewards"),
            "holiday_call_to_arms",
        )

        pool = build_candidate_pool("2026-12")
        self.assertIn("holiday_call_to_arms", set(pool["canonical_candidate"]))
        self.assertNotIn("holiday_rewards", set(pool["canonical_candidate"]))
        self.assertFalse(pool["canonical_candidate"].duplicated().any())

    def test_telegram_alias_uses_telegram_missions_canonical(self):
        self.assertEqual(normalize_candidate_key("telegram"), "telegram_missions")
        self.assertEqual(
            normalize_candidate_key("telegram_missions"),
            "telegram_missions",
        )
        self.assertEqual(
            get_candidate_display_label("telegram"),
            "Telegram Missions",
        )

        candidates = set(get_valid_candidates("2026-07"))
        self.assertIn("telegram_missions", candidates)
        self.assertNotIn("telegram", candidates)

    def test_thanksgiving_is_not_candidate(self):
        candidates = set(get_valid_candidates("2026-11"))
        pool_candidates = set(build_candidate_pool("2026-11")["canonical_candidate"])

        self.assertNotIn("thanksgiving", candidates)
        self.assertNotIn("thanksgiving", pool_candidates)

    def test_valentines_and_easter_are_not_candidates(self):
        candidates = set(get_valid_candidates("2026-02"))
        pool_candidates = set(build_candidate_pool("2026-02")["canonical_candidate"])

        for candidate in ["valentines", "easter"]:
            self.assertNotIn(candidate, candidates)
            self.assertNotIn(candidate, pool_candidates)

    def test_all_role_is_always_present(self):
        for target_month in ["2026-07", "2026-10", "2026-12"]:
            with self.subTest(target_month=target_month):
                candidates = set(get_valid_candidates(target_month))
                self.assertIn("all_role", candidates)

    def test_none_and_invalid_target_month_do_not_exclude_seasonal(self):
        for target_month in [None, "bad-value"]:
            with self.subTest(target_month=target_month):
                candidates = set(get_valid_candidates(target_month))
                self.assertIn("halloween", candidates)
                self.assertIn("halloween_call_to_arms", candidates)
                self.assertIn("holiday", candidates)
                self.assertIn("holiday_call_to_arms", candidates)
                self.assertIn("all_role", candidates)

    def test_build_candidate_pool_dataframe_output(self):
        pool = build_candidate_pool("2026-07")
        required_columns = {
            "candidate",
            "canonical_candidate",
            "candidate_group",
            "display_label",
            "is_seasonal_candidate",
            "valid_for_target_month",
            "excluded_reason",
            "eligible_months",
        }

        self.assertIsInstance(pool, pd.DataFrame)
        self.assertTrue(required_columns.issubset(pool.columns))

        halloween = pool[pool["candidate"].eq("halloween")].iloc[0]
        self.assertEqual(halloween["candidate_group"], "seasonal")
        self.assertFalse(bool(halloween["valid_for_target_month"]))
        self.assertEqual(
            halloween["excluded_reason"],
            "out_of_season_target_month",
        )


if __name__ == "__main__":
    unittest.main()
