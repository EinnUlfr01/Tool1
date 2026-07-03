import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.monthly_update import (
    ROLE_OPTIONS,
    apply_monthly_update,
    build_new_month_row,
    close_happening_benefits,
    create_backup_path,
    default_close_end_date,
    get_subcategory_options,
    get_top_component,
)


CSV_COLUMNS = [
    "month_label",
    "start_date",
    "end_date",
    "primary_category",
    "primary_sub_category",
    "multiplier_info",
    "is_seasonal",
    "seasonal_type",
    "is_mixed",
    "mixed_components",
    "is_all_role",
    "all_role_components",
    "weight",
    "component_weight_rule",
    "extra_tags",
    "note",
    "source_name",
    "source_url",
    "confidence",
]


def base_row(month_label="2026-06", end_date="Happening"):
    return {
        "month_label": month_label,
        "start_date": "6/3/2026",
        "end_date": end_date,
        "primary_category": "role",
        "primary_sub_category": "naturalist",
        "multiplier_info": "Naturalist bonuses",
        "is_seasonal": "FALSE",
        "seasonal_type": "",
        "is_mixed": "FALSE",
        "mixed_components": "",
        "is_all_role": "FALSE",
        "all_role_components": "",
        "weight": "1",
        "component_weight_rule": "single",
        "extra_tags": "",
        "note": "",
        "source_name": "test",
        "source_url": "",
        "confidence": "high",
    }


class MonthlyUpdateHelperTest(unittest.TestCase):
    def test_new_row_builder_creates_one_pipe_joined_row(self):
        row = build_new_month_row(
            CSV_COLUMNS,
            month_label="2026-07",
            start_date=date(2026, 7, 1),
            primary_category="non_role",
            primary_sub_categories=["blood_money", "telegram"],
            note="Keep this note.",
        )
        self.assertEqual(row["month_label"], "2026-07")
        self.assertEqual(row["primary_sub_category"], "blood_money|telegram")
        self.assertEqual(row["end_date"], "Happening")
        self.assertEqual(row["confidence"], "high")
        self.assertEqual(row["weight"], "1")
        self.assertEqual(row["note"], "Keep this note.")

    def test_both_sets_mixed_components(self):
        row = build_new_month_row(
            CSV_COLUMNS,
            month_label="2026-07",
            start_date=date(2026, 7, 1),
            primary_category="both",
            primary_sub_categories=["collector", "free_roam"],
        )
        self.assertEqual(row["is_mixed"], "TRUE")
        self.assertEqual(row["mixed_components"], "collector|free_roam")

    def test_all_role_sets_all_role_schema(self):
        row = build_new_month_row(
            CSV_COLUMNS,
            month_label="2026-07",
            start_date=date(2026, 7, 1),
            primary_category="role",
            primary_sub_categories=ROLE_OPTIONS,
        )
        self.assertEqual(row["primary_category"], "all_role")
        self.assertEqual(row["primary_sub_category"], "all_role")
        self.assertEqual(row["is_all_role"], "TRUE")
        self.assertEqual(row["all_role_components"], "|".join(ROLE_OPTIONS))

    def test_explicit_all_role_sets_all_role_schema(self):
        row = build_new_month_row(
            CSV_COLUMNS,
            month_label="2026-07",
            start_date=date(2026, 7, 1),
            primary_category="all_role",
            primary_sub_categories=ROLE_OPTIONS,
        )
        self.assertEqual(row["primary_category"], "all_role")
        self.assertEqual(row["primary_sub_category"], "all_role")
        self.assertEqual(row["is_all_role"], "TRUE")
        self.assertEqual(row["all_role_components"], "|".join(ROLE_OPTIONS))

    def test_top_prediction_locks_role_and_non_role_options(self):
        role_top = get_top_component(
            pd.DataFrame(
                [
                    {
                        "component": "collector",
                        "component_type": "role",
                        "global_probability_percent": 30,
                    }
                ]
            )
        )
        non_role_top = get_top_component(
            pd.DataFrame(
                [
                    {
                        "component": "blood_money",
                        "component_type": "non_role",
                        "global_probability_percent": 30,
                    }
                ]
            )
        )
        self.assertEqual(role_top.primary_category, "role")
        self.assertEqual(get_subcategory_options(role_top.primary_category), ROLE_OPTIONS)
        self.assertEqual(non_role_top.primary_category, "non_role")
        self.assertIn("story_missions", get_subcategory_options("non_role"))

    def test_seasonal_top_sets_seasonal_type(self):
        top = get_top_component(
            pd.DataFrame(
                [
                    {
                        "component": "holiday_call_to_arms",
                        "component_type": "non_role",
                        "global_probability_percent": 30,
                    }
                ]
            )
        )
        self.assertEqual(top.primary_category, "seasonal")
        self.assertEqual(top.seasonal_type, "holiday")
        row = build_new_month_row(
            CSV_COLUMNS,
            month_label="2026-12",
            start_date=date(2026, 12, 1),
            primary_category=top.primary_category,
            primary_sub_categories=[top.component],
            is_seasonal=True,
            seasonal_type=top.seasonal_type,
        )
        self.assertEqual(row["is_seasonal"], "TRUE")
        self.assertEqual(row["seasonal_type"], "holiday")

    def test_other_non_role_requires_specific_raw_choice(self):
        top = get_top_component(
            pd.DataFrame(
                [
                    {
                        "component": "other_non_role",
                        "component_type": "non_role",
                        "global_probability_percent": 30,
                    }
                ]
            )
        )
        self.assertTrue(top.requires_raw_non_role)
        with self.assertRaises(ValueError):
            build_new_month_row(
                CSV_COLUMNS,
                month_label="2026-07",
                start_date=date(2026, 7, 1),
                primary_category="non_role",
                primary_sub_categories=["other_non_role"],
            )

    def test_close_happening_only_updates_happening_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "rdo_benefits_raw.csv"
            pd.DataFrame(
                [
                    base_row("2026-05", "6/2/2026"),
                    base_row("2026-06", "Happening"),
                ],
                columns=CSV_COLUMNS,
            ).to_csv(csv_path, index=False)
            backup_path = create_backup_path(csv_path)
            result = close_happening_benefits(
                csv_path,
                date(2026, 6, 30),
                backup_path=backup_path,
            )
            updated = pd.read_csv(csv_path, dtype=str)
            self.assertTrue(result.backup_path.exists())
            self.assertEqual(result.changed_rows, 1)
            self.assertEqual(updated.loc[0, "end_date"], "6/2/2026")
            self.assertEqual(updated.loc[1, "end_date"], "6/30/2026")

    def test_add_new_month_closes_happening_and_appends_one_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "rdo_benefits_raw.csv"
            pd.DataFrame([base_row()], columns=CSV_COLUMNS).to_csv(csv_path, index=False)
            new_row = build_new_month_row(
                CSV_COLUMNS,
                month_label="2026-07",
                start_date=date(2026, 7, 1),
                primary_category="role",
                primary_sub_categories=["collector"],
            )
            result = apply_monthly_update(
                csv_path,
                default_close_end_date(date(2026, 7, 1)),
                new_row,
                create_backup_path(csv_path),
            )
            updated = pd.read_csv(csv_path, dtype=str)
            self.assertEqual(result.changed_rows, 1)
            self.assertEqual(result.appended_rows, 1)
            self.assertEqual(len(updated), 2)
            self.assertEqual(updated.loc[0, "end_date"], "6/30/2026")
            self.assertEqual(updated.loc[1, "month_label"], "2026-07")

    def test_duplicate_month_label_is_blocked_before_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "rdo_benefits_raw.csv"
            pd.DataFrame([base_row()], columns=CSV_COLUMNS).to_csv(csv_path, index=False)
            duplicate_row = build_new_month_row(
                CSV_COLUMNS,
                month_label="2026-06",
                start_date=date(2026, 6, 29),
                primary_category="role",
                primary_sub_categories=["collector"],
            )
            with self.assertRaises(ValueError):
                apply_monthly_update(
                    csv_path,
                    None,
                    duplicate_row,
                    create_backup_path(csv_path),
                )


if __name__ == "__main__":
    unittest.main()
