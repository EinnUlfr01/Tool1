import unittest

from src.candidate_scorer import (
    PRIOR_STRENGTH_OPTIONS,
    CandidateScoringConfig,
    score_candidates,
)

import pandas as pd


ALL_ROLE_COMPONENTS = "bounty_hunter|trader|collector|moonshiner|naturalist"


def row(
    month_label: str,
    candidate: str,
    primary_category: str = "role",
    is_mixed: bool = False,
    mixed_components: str = "",
    is_all_role: bool = False,
    all_role_components: str = "",
) -> dict:
    effective_candidate = "all_role" if is_all_role else candidate
    effective_category = "all_role" if is_all_role else primary_category
    return {
        "month_label": month_label,
        "primary_category": effective_category,
        "primary_sub_category": effective_candidate,
        "effective_primary_category": effective_category,
        "effective_primary_sub_category": effective_candidate,
        "effective_candidate": effective_candidate,
        "is_mixed": is_mixed,
        "mixed_components": mixed_components,
        "is_all_role": is_all_role,
        "all_role_components": all_role_components,
        "is_seasonal": False,
        "seasonal_type": "",
    }


def scored_by_candidate(rows: list[dict], target_month: str | None) -> pd.DataFrame:
    scored = score_candidates(pd.DataFrame(rows), target_month)
    return scored.set_index("candidate")


class CandidateScorerTest(unittest.TestCase):
    def test_normalized_probability_sums_to_one(self):
        scored = score_candidates(
            pd.DataFrame(
                [
                    row("2026-01", "bounty_hunter"),
                    row("2026-02", "call_to_arms", primary_category="non_role"),
                ]
            ),
            "2026-07",
        )

        self.assertAlmostEqual(float(scored["final_probability"].sum()), 1.0, places=6)
        self.assertAlmostEqual(
            float(scored["final_probability_percent"].sum()),
            100.0,
            places=4,
        )

    def test_july_excludes_seasonal_from_scoring(self):
        scored = score_candidates(pd.DataFrame([row("2026-01", "bounty_hunter")]), "2026-07")
        candidates = set(scored["candidate"])

        self.assertNotIn("halloween", candidates)
        self.assertNotIn("halloween_call_to_arms", candidates)
        self.assertNotIn("holiday", candidates)
        self.assertNotIn("holiday_call_to_arms", candidates)

    def test_october_and_december_seasonal_boost(self):
        october = scored_by_candidate([], "2026-10")
        december = scored_by_candidate([], "2026-12")

        self.assertIn("halloween", october.index)
        self.assertEqual(october.loc["halloween", "seasonal_factor"], 1.2)
        self.assertEqual(
            october.loc["halloween_call_to_arms", "seasonal_factor"],
            1.2,
        )
        self.assertIn("holiday", december.index)
        self.assertEqual(december.loc["holiday", "seasonal_factor"], 1.2)
        self.assertEqual(
            december.loc["holiday_call_to_arms", "seasonal_factor"],
            1.2,
        )

    def test_all_role_candidate_penalty(self):
        recent = scored_by_candidate(
            [
                row(
                    "2026-06",
                    "all_role",
                    primary_category="all_role",
                    is_all_role=True,
                    all_role_components=ALL_ROLE_COMPONENTS,
                )
            ],
            "2026-07",
        )
        mid_gap = scored_by_candidate(
            [
                row(
                    "2026-02",
                    "all_role",
                    primary_category="all_role",
                    is_all_role=True,
                    all_role_components=ALL_ROLE_COMPONENTS,
                )
            ],
            "2026-12",
        )
        old_gap = scored_by_candidate(
            [
                row(
                    "2025-12",
                    "all_role",
                    primary_category="all_role",
                    is_all_role=True,
                    all_role_components=ALL_ROLE_COMPONENTS,
                )
            ],
            "2026-12",
        )
        never = scored_by_candidate([], "2026-12")

        self.assertEqual(recent.loc["all_role", "all_role_candidate_factor"], 0.1)
        self.assertEqual(mid_gap.loc["all_role", "all_role_candidate_factor"], 0.5)
        self.assertEqual(old_gap.loc["all_role", "all_role_candidate_factor"], 1.0)
        self.assertEqual(never.loc["all_role", "all_role_candidate_factor"], 1.0)

    def test_all_role_inclusion_does_not_reset_role_direct(self):
        scored = scored_by_candidate(
            [
                row("2025-01", "bounty_hunter"),
                row(
                    "2026-06",
                    "all_role",
                    primary_category="all_role",
                    is_all_role=True,
                    all_role_components=ALL_ROLE_COMPONENTS,
                ),
            ],
            "2026-07",
        )
        bounty_hunter = scored.loc["bounty_hunter"]

        self.assertEqual(bounty_hunter["months_since_direct"], 18)
        self.assertEqual(bounty_hunter["months_since_all_role"], 1)
        self.assertEqual(bounty_hunter["direct_recency_factor"], 1.0)
        self.assertAlmostEqual(
            bounty_hunter["all_role_inclusion_recency_factor"],
            0.8875,
            places=4,
        )

    def test_mixed_does_not_reset_role_direct(self):
        scored = scored_by_candidate(
            [
                row("2025-01", "bounty_hunter"),
                row(
                    "2026-06",
                    "collector",
                    primary_category="both",
                    is_mixed=True,
                    mixed_components="bounty_hunter|free_roam",
                ),
            ],
            "2026-07",
        )
        bounty_hunter = scored.loc["bounty_hunter"]

        self.assertEqual(bounty_hunter["months_since_direct"], 18)
        self.assertEqual(bounty_hunter["months_since_mixed"], 1)
        self.assertAlmostEqual(
            bounty_hunter["mixed_recency_factor"],
            0.775,
            places=4,
        )

    def test_direct_recent_penalty_is_stronger_than_all_role_soft_penalty(self):
        scored = scored_by_candidate(
            [
                row("2026-06", "trader"),
                row(
                    "2026-06",
                    "all_role",
                    primary_category="all_role",
                    is_all_role=True,
                    all_role_components=ALL_ROLE_COMPONENTS,
                ),
            ],
            "2026-07",
        )

        self.assertLess(
            scored.loc["trader", "direct_recency_factor"],
            scored.loc["collector", "all_role_inclusion_recency_factor"],
        )

    def test_combined_penalty_has_floor(self):
        scored = scored_by_candidate(
            [
                row("2026-06", "bounty_hunter"),
                row(
                    "2026-06",
                    "collector",
                    primary_category="both",
                    is_mixed=True,
                    mixed_components="bounty_hunter|free_roam",
                ),
                row(
                    "2026-06",
                    "all_role",
                    primary_category="all_role",
                    is_all_role=True,
                    all_role_components=ALL_ROLE_COMPONENTS,
                ),
            ],
            "2026-07",
        )
        bounty_hunter = scored.loc["bounty_hunter"]

        self.assertLess(bounty_hunter["combined_recency_factor_before_clamp"], 0.45)
        self.assertEqual(bounty_hunter["combined_recency_factor"], 0.45)
        self.assertEqual(bounty_hunter["recency_factor"], 0.45)

    def test_non_role_overdue_is_lighter_than_role_overdue(self):
        scored = scored_by_candidate(
            [
                row("2025-11", "bounty_hunter"),
                row("2025-11", "call_to_arms", primary_category="non_role"),
            ],
            "2026-07",
        )

        self.assertEqual(scored.loc["bounty_hunter", "overdue_factor"], 1.2)
        self.assertEqual(scored.loc["call_to_arms", "overdue_factor"], 1.05)

    def test_non_role_context_factor(self):
        role_streak = scored_by_candidate(
            [
                row("2026-04", "bounty_hunter"),
                row("2026-05", "collector"),
                row("2026-06", "trader"),
            ],
            "2026-07",
        )
        previous_all_role = scored_by_candidate(
            [
                row("2026-05", "collector"),
                row(
                    "2026-06",
                    "all_role",
                    primary_category="all_role",
                    is_all_role=True,
                    all_role_components=ALL_ROLE_COMPONENTS,
                ),
            ],
            "2026-07",
        )

        self.assertEqual(
            role_streak.loc["call_to_arms", "non_role_context_factor"],
            1.2,
        )
        self.assertAlmostEqual(
            previous_all_role.loc["call_to_arms", "non_role_context_factor"],
            1.155,
            places=4,
        )

    def test_prior_strength_default_and_range(self):
        self.assertEqual(CandidateScoringConfig().direct_weight, 1.0)
        self.assertEqual(PRIOR_STRENGTH_OPTIONS, [3, 4, 5])

        low = score_candidates(pd.DataFrame([]), "2026-07", prior_strength=2)
        default = score_candidates(pd.DataFrame([]), "2026-07")
        high = score_candidates(pd.DataFrame([]), "2026-07", prior_strength=8)

        self.assertTrue(low["prior_strength"].eq(3.0).all())
        self.assertTrue(default["prior_strength"].eq(3.0).all())
        self.assertTrue(high["prior_strength"].eq(5.0).all())


if __name__ == "__main__":
    unittest.main()
