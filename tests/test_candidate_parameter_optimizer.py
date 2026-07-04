import unittest

import pandas as pd

from src.candidate_parameter_optimizer import (
    CANDIDATE_PRIOR_STRENGTH_CANDIDATES,
    optimize_candidate_scoring_parameters,
)
from src.parameter_optimizer import PRIOR_STRENGTH_CANDIDATES


def row(month_label: str, candidate: str) -> dict:
    return {
        "month_label": month_label,
        "primary_category": "role",
        "primary_sub_category": candidate,
        "effective_primary_category": "role",
        "effective_primary_sub_category": candidate,
        "effective_candidate": candidate,
        "is_mixed": False,
        "mixed_components": "",
        "is_all_role": False,
        "all_role_components": "",
        "is_seasonal": False,
        "seasonal_type": "",
        "weight": 1.0,
        "component_weight_rule": "equal_split",
    }


class CandidateParameterOptimizerTest(unittest.TestCase):
    def test_candidate_prior_range_is_separate_from_legacy_range(self):
        self.assertEqual(CANDIDATE_PRIOR_STRENGTH_CANDIDATES, [3, 4, 5])
        self.assertEqual(PRIOR_STRENGTH_CANDIDATES, [2, 3, 4, 5, 6, 7, 8])

    def test_optimizer_does_not_write_or_mutate_external_config(self):
        rows = [
            row(f"2024-{month:02d}", "bounty_hunter")
            for month in range(1, 13)
        ] + [
            row(f"2025-{month:02d}", "collector")
            for month in range(1, 13)
        ] + [
            row("2026-01", "trader"),
            row("2026-02", "moonshiner"),
        ]

        result = optimize_candidate_scoring_parameters(
            pd.DataFrame(rows),
            min_train_months=24,
        )

        self.assertTrue(result.enough_data)
        self.assertIn(result.best_prior_strength, {3, 4, 5})
        self.assertEqual(CANDIDATE_PRIOR_STRENGTH_CANDIDATES, [3, 4, 5])
        self.assertEqual(PRIOR_STRENGTH_CANDIDATES, [2, 3, 4, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
