import json
import tempfile
import unittest
from pathlib import Path

from src.adjusted_predictor import calculate_adjusted_primary_category_prediction
from src.conditional_predictor import (
    DEFAULT_COOLDOWN_CONFIG,
    calculate_conditional_sub_category_prediction_with_config,
)
from src.data_loader import load_raw_benefits
from src.global_predictor import calculate_final_global_component_prediction
from src.optimization_cache import (
    get_cached_best_params,
    load_optimization_cache,
    save_optimization_cache,
)
from src.parameter_optimizer import OptimizationResult, empty_optimizer_table
from src.probability import get_sub_category_display_label
from src.seasonal_rules import (
    get_component_eligible_months,
    is_component_prediction_eligible,
)


HARD_SEASONAL_COMPONENTS = {
    "halloween",
    "halloween_call_to_arms",
    "holiday",
    "holiday_call_to_arms",
    "holiday_rewards",
}


class PredictionLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_raw_benefits()
        cls.primary_prediction = calculate_adjusted_primary_category_prediction(cls.df)
        cls.conditional = calculate_conditional_sub_category_prediction_with_config(
            cls.df,
            cooldown_config=DEFAULT_COOLDOWN_CONFIG,
            parent_primary_prediction_df=cls.primary_prediction.table,
            predicted_next_month=cls.primary_prediction.predicted_next_month,
        )
        cls.final = calculate_final_global_component_prediction(cls.conditional)

    def build_final_for_target(self, predicted_next_month: str):
        conditional = calculate_conditional_sub_category_prediction_with_config(
            self.df,
            cooldown_config=DEFAULT_COOLDOWN_CONFIG,
            parent_primary_prediction_df=self.primary_prediction.table,
            predicted_next_month=predicted_next_month,
        )
        return calculate_final_global_component_prediction(conditional)

    def test_primary_prediction_has_only_three_categories_after_all_role(self):
        table = self.primary_prediction.table
        self.assertEqual(set(table["primary_category"]), {"role", "non_role", "both"})
        self.assertAlmostEqual(
            float(table["adjusted_prediction_percent"].sum()),
            100.0,
            places=2,
        )

        multipliers = table.set_index("primary_category")["domain_multiplier"].to_dict()
        self.assertEqual(multipliers["role"], 0.45)
        self.assertEqual(multipliers["non_role"], 1.45)
        self.assertEqual(multipliers["both"], 1.15)

    def test_target_month_7_excludes_hard_seasonal_components(self):
        final_components = set(self.final["component"])
        self.assertFalse(HARD_SEASONAL_COMPONENTS & final_components)
        self.assertAlmostEqual(
            float(self.final["global_probability_percent"].sum()),
            100.0,
            places=2,
        )

    def test_target_month_10_allows_halloween_only(self):
        final = self.build_final_for_target("2026-10")
        final_components = set(final["component"])
        self.assertIn("halloween_call_to_arms", final_components)
        self.assertNotIn("holiday_call_to_arms", final_components)
        self.assertNotIn("holiday_rewards", final_components)
        self.assertAlmostEqual(
            float(final["global_probability_percent"].sum()),
            100.0,
            places=2,
        )

    def test_target_month_12_allows_holiday_only(self):
        final = self.build_final_for_target("2026-12")
        final_components = set(final["component"])
        self.assertIn("holiday_call_to_arms", final_components)
        self.assertIn("holiday_rewards", final_components)
        self.assertNotIn("halloween_call_to_arms", final_components)
        self.assertAlmostEqual(
            float(final["global_probability_percent"].sum()),
            100.0,
            places=2,
        )

    def test_seasonal_rules_direct_eligibility(self):
        self.assertEqual(get_component_eligible_months("halloween"), {10})
        self.assertEqual(get_component_eligible_months("holiday_rewards"), {12})
        self.assertFalse(is_component_prediction_eligible("holiday_call_to_arms", "2026-07"))
        self.assertTrue(is_component_prediction_eligible("halloween_call_to_arms", "2026-10"))
        self.assertTrue(is_component_prediction_eligible("holiday_rewards", "2026-12"))
        self.assertTrue(is_component_prediction_eligible("blood_money", "2026-07"))

    def test_neutral_multipliers_do_not_render_as_applied_rules(self):
        rendered_rules = "|".join(self.final["applied_rules"].astype(str).to_list())
        self.assertNotIn("in_season=1.00", rendered_rules)
        self.assertNotIn("out_of_season=1.00", rendered_rules)

    def test_duplicate_components_are_merged_in_final_global(self):
        duplicated = self.final["component"].duplicated()
        self.assertFalse(bool(duplicated.any()))
        self.assertIn("call_to_arms", set(self.final["component"]))

    def test_other_non_role_is_non_role_and_labeled(self):
        row = self.final[self.final["component"] == "other_non_role"].iloc[0]
        self.assertEqual(row["component_type"], "non_role")
        self.assertEqual(get_sub_category_display_label("other_non_role"), "Other Non-role")

    def test_non_seasonal_components_are_not_excluded(self):
        final_components = set(self.final["component"])
        for component in [
            "bounty_hunter",
            "trader",
            "collector",
            "naturalist",
            "moonshiner",
            "blood_money",
            "telegram",
            "call_to_arms",
            "free_roam",
            "races",
            "other_non_role",
        ]:
            self.assertIn(component, final_components)

    def test_final_global_total_is_one_hundred(self):
        self.assertAlmostEqual(
            float(self.final["global_probability_percent"].sum()),
            100.0,
            places=2,
        )


class OptimizationCacheTest(unittest.TestCase):
    def test_cache_hit_and_stale_detection(self):
        fake_result = OptimizationResult(
            enough_data=True,
            comparison_table=empty_optimizer_table(),
            best_prior_strength=5,
            best_cooldown_config=DEFAULT_COOLDOWN_CONFIG,
            best_final_backtest_score=0.42,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "optimization_result.json"
            save_optimization_cache(fake_result, cache_path=cache_path)

            hit = load_optimization_cache(cache_path=cache_path)
            self.assertTrue(hit.is_valid)
            prior_strength, cooldown_config = get_cached_best_params(hit.cache)
            self.assertEqual(prior_strength, 5)
            self.assertEqual(cooldown_config, DEFAULT_COOLDOWN_CONFIG)

            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["csv_hash"] = "stale"
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            stale = load_optimization_cache(cache_path=cache_path)
            self.assertEqual(stale.status, "stale")


if __name__ == "__main__":
    unittest.main()
