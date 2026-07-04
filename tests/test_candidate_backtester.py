import unittest
from unittest.mock import patch

import pandas as pd

from src.candidate_backtester import (
    extract_actual_candidates,
    get_actual_candidates,
    run_candidate_walk_forward_backtest,
    summarize_candidate_backtest_results,
    summarize_target_month,
)
from src.candidate_parameter_optimizer import optimize_candidate_scoring_parameters


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
        "weight": 1.0,
        "component_weight_rule": "equal_split",
    }


class CandidateBacktesterTest(unittest.TestCase):
    def test_actual_candidates_include_mixed_components(self):
        actual_rows = pd.DataFrame(
            [
                row(
                    "2026-07",
                    "bounty_hunter",
                    primary_category="both",
                    is_mixed=True,
                    mixed_components="bounty_hunter|telegram",
                )
            ]
        )

        self.assertEqual(
            get_actual_candidates(actual_rows),
            ["bounty_hunter", "telegram_missions"],
        )

    def test_actual_candidates_for_all_role_is_all_role_only(self):
        actual_rows = pd.DataFrame(
            [
                row(
                    "2026-07",
                    "all_role",
                    primary_category="all_role",
                    is_all_role=True,
                    all_role_components=ALL_ROLE_COMPONENTS,
                )
            ]
        )

        candidates = get_actual_candidates(actual_rows)

        self.assertEqual(candidates, ["all_role"])

    def test_thanksgiving_does_not_create_candidate(self):
        actual = extract_actual_candidates(
            pd.Series(
                {
                    **row("2026-11", "trader"),
                    "is_seasonal": True,
                    "seasonal_type": "thanksgiving",
                }
            )
        )

        self.assertEqual(actual, ["trader"])
        self.assertNotIn("thanksgiving", actual)

    def test_legacy_all_roles_row_extracts_all_role(self):
        actual = extract_actual_candidates(
            pd.Series(
                {
                    "month_label": "2026-07",
                    "primary_category": "all_roles",
                    "primary_sub_category": "all_roles",
                    "is_all_role": "TRUE",
                    "all_role_components": ALL_ROLE_COMPONENTS,
                }
            )
        )

        self.assertEqual(actual, ["all_role"])

    def test_candidate_walk_forward_returns_metrics(self):
        rows = [
            row(f"2024-{month:02d}", "bounty_hunter")
            for month in range(1, 13)
        ] + [
            row(f"2025-{month:02d}", "collector")
            for month in range(1, 13)
        ] + [
            row("2026-01", "trader"),
            row("2026-02", "telegram", primary_category="non_role"),
        ]

        result = run_candidate_walk_forward_backtest(
            pd.DataFrame(rows),
            min_train_months=24,
            prior_strength=3,
        )

        self.assertEqual(len(result), 2)
        self.assertTrue(result["actual_candidates"].str.len().gt(0).all())
        self.assertTrue(result["best_actual_candidate_rank"].ge(1).all())
        self.assertTrue(result["actual_candidate_probability_percent"].ge(0).all())

    def test_candidate_backtest_does_not_leak_target_or_future_rows(self):
        rows = [
            row("2024-01", "bounty_hunter"),
            row("2024-02", "collector"),
            row("2024-03", "trader"),
        ]
        seen_training_months = []

        def fake_score_candidates(train_df, target_month, prior_strength=3, config=None):
            seen_training_months.append((target_month, train_df["month_label"].tolist()))
            return pd.DataFrame(
                [
                    {
                        "candidate": "trader",
                        "display_label": "Trader",
                        "rank": 1,
                        "final_probability": 0.5,
                        "final_probability_percent": 50.0,
                    },
                    {
                        "candidate": "collector",
                        "display_label": "Collector",
                        "rank": 2,
                        "final_probability": 0.3,
                        "final_probability_percent": 30.0,
                    },
                ]
            )

        with patch("src.candidate_backtester.score_candidates", fake_score_candidates):
            run_candidate_walk_forward_backtest(
                pd.DataFrame(rows),
                min_train_months=1,
                prior_strength=3,
            )

        self.assertEqual(seen_training_months[0], ("2024-02", ["2024-01"]))
        self.assertEqual(seen_training_months[1], ("2024-03", ["2024-01", "2024-02"]))

    def test_candidate_backtest_rank_and_hits_are_correct(self):
        ranking = pd.DataFrame(
            [
                {
                    "candidate": "bounty_hunter",
                    "display_label": "Bounty Hunter",
                    "rank": 1,
                    "final_probability": 0.40,
                    "final_probability_percent": 40.0,
                },
                {
                    "candidate": "collector",
                    "display_label": "Collector",
                    "rank": 2,
                    "final_probability": 0.30,
                    "final_probability_percent": 30.0,
                },
                {
                    "candidate": "call_to_arms",
                    "display_label": "Call To Arms",
                    "rank": 3,
                    "final_probability": 0.20,
                    "final_probability_percent": 20.0,
                },
                {
                    "candidate": "trader",
                    "display_label": "Trader",
                    "rank": 4,
                    "final_probability": 0.10,
                    "final_probability_percent": 10.0,
                },
            ]
        )

        summary = summarize_target_month(
            "2026-07",
            "mixed",
            ranking,
            ["call_to_arms", "trader"],
            3,
        )

        self.assertEqual(summary["candidate_top1_hit"], 0)
        self.assertEqual(summary["candidate_top3_hit"], 1)
        self.assertEqual(summary["candidate_top5_hit"], 1)
        self.assertEqual(summary["best_actual_candidate_rank"], 3)
        self.assertEqual(summary["average_actual_candidate_rank"], 3.5)
        self.assertEqual(summary["median_actual_candidate_rank"], 3.5)

    def test_all_role_hit_uses_all_role_not_role_components(self):
        ranking = pd.DataFrame(
            [
                {
                    "candidate": "bounty_hunter",
                    "display_label": "Bounty Hunter",
                    "rank": 1,
                    "final_probability": 0.40,
                    "final_probability_percent": 40.0,
                },
                {
                    "candidate": "all_role",
                    "display_label": "All Roles",
                    "rank": 6,
                    "final_probability": 0.05,
                    "final_probability_percent": 5.0,
                },
            ]
        )

        summary = summarize_target_month(
            "2026-07",
            "all_role",
            ranking,
            ["all_role"],
            3,
        )

        self.assertEqual(summary["candidate_top1_hit"], 0)
        self.assertEqual(summary["best_actual_candidate_rank"], 6)

    def test_backtest_summary_uses_hit_rank_miss_rate_and_capped_rank(self):
        results = pd.DataFrame(
            [
                {
                    "candidate_top1_hit": 1,
                    "candidate_top3_hit": 1,
                    "candidate_top5_hit": 1,
                    "average_actual_candidate_rank": 2,
                    "median_actual_candidate_rank": 2,
                    "average_hit_actual_candidate_rank": 2,
                    "median_hit_actual_candidate_rank": 2,
                    "miss_rate": 0,
                    "average_capped_actual_candidate_rank": 2,
                    "average_actual_candidate_probability_percent": 20,
                    "top_candidate_probability_percent": 30,
                    "top_candidate": "bounty_hunter",
                    "target_type": "role",
                    "prior_strength": 3,
                },
                {
                    "candidate_top1_hit": 0,
                    "candidate_top3_hit": 0,
                    "candidate_top5_hit": 0,
                    "average_actual_candidate_rank": 0,
                    "median_actual_candidate_rank": 0,
                    "average_hit_actual_candidate_rank": 0,
                    "median_hit_actual_candidate_rank": 0,
                    "miss_rate": 1,
                    "average_capped_actual_candidate_rank": 6,
                    "average_actual_candidate_probability_percent": 0,
                    "top_candidate_probability_percent": 25,
                    "top_candidate": "call_to_arms",
                    "target_type": "non_role",
                    "prior_strength": 3,
                },
            ]
        )

        summary = summarize_candidate_backtest_results(results)

        self.assertEqual(summary["average_hit_actual_candidate_rank"], 2)
        self.assertEqual(summary["median_hit_actual_candidate_rank"], 2)
        self.assertEqual(summary["miss_rate"], 0.5)
        self.assertEqual(summary["average_capped_actual_candidate_rank"], 4)
        self.assertEqual(summary["percentage_top1_role"], 0.5)
        self.assertEqual(summary["percentage_top1_non_role"], 0.5)

    def test_candidate_optimizer_uses_debug_prior_range(self):
        rows = [
            row(f"2024-{month:02d}", "bounty_hunter")
            for month in range(1, 13)
        ] + [
            row(f"2025-{month:02d}", "collector")
            for month in range(1, 13)
        ] + [
            row("2026-01", "trader"),
            row("2026-02", "telegram", primary_category="non_role"),
        ]

        result = optimize_candidate_scoring_parameters(
            pd.DataFrame(rows),
            min_train_months=24,
        )

        self.assertTrue(result.enough_data)
        self.assertEqual(set(result.comparison_table["prior_strength"]), {3, 4, 5})
        self.assertIn(result.best_prior_strength, {3, 4, 5})

    def test_candidate_calibration_compatibility_wrapper_uses_debug_prior_range(self):
        rows = [
            row(f"2024-{month:02d}", "bounty_hunter")
            for month in range(1, 13)
        ] + [
            row(f"2025-{month:02d}", "collector")
            for month in range(1, 13)
        ] + [
            row("2026-01", "trader"),
            row("2026-02", "telegram", primary_category="non_role"),
        ]

        result = optimize_candidate_scoring_parameters(pd.DataFrame(rows), min_train_months=24)

        self.assertNotIn(2, set(result.comparison_table["prior_strength"]))
        self.assertNotIn(7, set(result.comparison_table["prior_strength"]))


if __name__ == "__main__":
    unittest.main()
