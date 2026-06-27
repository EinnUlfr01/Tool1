import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.adjusted_predictor import calculate_adjusted_primary_category_prediction
from src.conditional_predictor import (
    DEFAULT_COOLDOWN_CONFIG,
    calculate_conditional_sub_category_prediction_with_config,
    get_non_role_recency_multiplier,
    get_role_last_seen_months,
)
from src.data_loader import (
    ROLE_COMPONENTS,
    expand_benefit_components,
    load_raw_benefits,
)
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
from src.ui_helpers import (
    build_all_benefits_table,
    build_simple_final_component_table,
    build_simple_primary_prediction_table,
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
            "featured_series",
            "other_non_role",
        ]:
            self.assertIn(component, final_components)

    def test_final_global_total_is_one_hundred(self):
        self.assertAlmostEqual(
            float(self.final["global_probability_percent"].sum()),
            100.0,
            places=2,
        )

    def test_role_recency_order_for_target_month_2026_07(self):
        role_rows = self.final[
            self.final["component"].isin(ROLE_COMPONENTS)
        ].sort_values("global_probability_percent", ascending=False)
        self.assertEqual(
            role_rows["component"].to_list(),
            [
                "collector",
                "bounty_hunter",
                "moonshiner",
                "trader",
                "naturalist",
            ],
        )

    def test_all_role_rows_are_ignored_for_role_recency(self):
        expanded = expand_benefit_components(self.df)
        expanded["month_period"] = (
            pd.to_datetime(
                expanded["month_label"],
                format="%Y-%m",
                errors="coerce",
            ).dt.to_period("M")
        )
        last_seen = get_role_last_seen_months(expanded, ignore_all_role_rows=True)
        self.assertEqual(str(last_seen["collector"]), "2025-09")
        self.assertNotEqual(str(last_seen["collector"]), "2026-06")
        self.assertEqual(str(last_seen["trader"]), "2026-04")

    def test_role_recency_multipliers_are_ordered_by_last_seen(self):
        role_rows = self.conditional[
            self.conditional["primary_category"].eq("role")
            & self.conditional["primary_sub_category"].isin(ROLE_COMPONENTS)
        ].set_index("primary_sub_category")
        multipliers = role_rows["cooldown_multiplier"]
        self.assertLess(multipliers["naturalist"], multipliers["trader"])
        self.assertLess(multipliers["trader"], multipliers["moonshiner"])
        self.assertLess(multipliers["moonshiner"], multipliers["bounty_hunter"])
        self.assertLess(multipliers["bounty_hunter"], multipliers["collector"])

        self.assertEqual(role_rows.loc["naturalist", "last_seen_month"], "2026-05")
        self.assertEqual(role_rows.loc["trader", "last_seen_month"], "2026-04")
        self.assertEqual(role_rows.loc["moonshiner", "last_seen_month"], "2026-03")
        self.assertEqual(role_rows.loc["bounty_hunter", "last_seen_month"], "2026-01")
        self.assertEqual(role_rows.loc["collector", "last_seen_month"], "2025-09")

    def test_non_role_recency_exists(self):
        non_role_rows = self.conditional[
            self.conditional["component_type"].eq("non_role")
        ]
        self.assertGreater(len(non_role_rows), 0)
        self.assertIn("last_seen_month", non_role_rows.columns)
        self.assertIn("months_since_last_seen", non_role_rows.columns)
        self.assertIn("cooldown_multiplier", non_role_rows.columns)

        traced_components = {
            "blood_money",
            "telegram",
            "call_to_arms",
            "free_roam",
            "races",
            "featured_series",
            "other_non_role",
        }
        traced_rows = non_role_rows[
            non_role_rows["primary_sub_category"].isin(traced_components)
        ]
        self.assertEqual(set(traced_rows["primary_sub_category"]), traced_components)
        self.assertFalse(traced_rows["last_seen_month"].eq("").any())
        self.assertFalse(traced_rows["months_since_last_seen"].isna().any())

    def test_non_role_recency_multiplier_order(self):
        self.assertLess(
            get_non_role_recency_multiplier(2),
            get_non_role_recency_multiplier(3),
        )
        self.assertLess(
            get_non_role_recency_multiplier(3),
            get_non_role_recency_multiplier(4),
        )
        self.assertLess(
            get_non_role_recency_multiplier(4),
            get_non_role_recency_multiplier(5),
        )
        self.assertLess(
            get_non_role_recency_multiplier(6),
            get_non_role_recency_multiplier(7),
        )
        self.assertGreater(
            get_non_role_recency_multiplier(None),
            get_non_role_recency_multiplier(7),
        )

    def test_non_role_score_uses_recency_multiplier(self):
        row = self.conditional[
            self.conditional["primary_category"].eq("non_role")
            & self.conditional["primary_sub_category"].eq("call_to_arms")
        ].iloc[0]
        self.assertNotEqual(float(row["cooldown_multiplier"]), 1.0)
        expected_score = (
            float(row["conditional_base_percent"])
            * float(row["cooldown_multiplier"])
            * float(row["seasonal_multiplier"])
        )
        self.assertAlmostEqual(float(row["adjusted_score"]), expected_score, places=3)


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


class SimpleUiHelperTest(unittest.TestCase):
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
        cls.final["display_label"] = cls.final["component"].apply(
            get_sub_category_display_label
        )

    def test_all_benefits_table_is_display_only(self):
        table = build_all_benefits_table(self.df)
        self.assertEqual(
            list(table.columns),
            ["Month", "Benefit Type", "Benefits Main Info", "Source"],
        )
        self.assertGreater(len(table), 0)

    def test_simple_primary_table_hides_debug_columns(self):
        table = build_simple_primary_prediction_table(
            self.primary_prediction.table,
            self.primary_prediction.latest_is_all_role,
        )
        self.assertEqual(list(table.columns), ["Category", "Probability %", "Reason"])
        self.assertNotIn("raw_score", table.columns)
        self.assertNotIn("domain_multiplier", table.columns)

    def test_simple_final_table_hides_debug_columns(self):
        table = build_simple_final_component_table(
            self.final,
            self.primary_prediction.latest_is_all_role,
        )
        self.assertEqual(
            list(table.columns),
            ["Rank", "Component", "Type", "Probability %", "Reason"],
        )
        self.assertNotIn("source_paths", table.columns)
        self.assertNotIn("applied_rules", table.columns)
        self.assertNotIn("historical_component_weight", table.columns)

    def test_normal_roles_do_not_show_seasonal_timing_reason(self):
        table = build_simple_final_component_table(
            self.final,
            self.primary_prediction.latest_is_all_role,
        )
        normal_role_labels = [
            "Bounty Hunter",
            "Collector",
            "Naturalist",
            "Moonshiner",
        ]
        role_reasons = table[
            table["Component"].isin(normal_role_labels)
        ].set_index("Component")["Reason"]

        for component in role_reasons.index:
            self.assertNotIn("Seasonal timing", role_reasons[component])

    def test_non_role_recency_reason_is_simple(self):
        table = build_simple_final_component_table(
            self.final,
            self.primary_prediction.latest_is_all_role,
        )
        call_to_arms_reason = table.loc[
            table["Component"].eq("Call To Arms"),
            "Reason",
        ].iloc[0]
        self.assertIn("Long time since last seen", call_to_arms_reason)
        self.assertNotIn("non_role_long_gap", call_to_arms_reason)

        synthetic_final = self.final.copy()
        synthetic_final.loc[
            synthetic_final["component"].eq("blood_money"),
            "applied_rules",
        ] = "non_role_cooldown=0.65"
        synthetic_table = build_simple_final_component_table(
            synthetic_final,
            self.primary_prediction.latest_is_all_role,
        )
        blood_money_reason = synthetic_table.loc[
            synthetic_table["Component"].eq("Blood Money"),
            "Reason",
        ].iloc[0]
        self.assertIn("Recent non-role cooldown", blood_money_reason)
        self.assertNotIn("non_role_cooldown", blood_money_reason)


if __name__ == "__main__":
    unittest.main()
