import unittest

import pandas as pd

from src.candidate_history import build_candidate_history


ALL_ROLE_COMPONENTS = "bounty_hunter|trader|collector|moonshiner|naturalist"


def row(
    month_label: str,
    candidate: str,
    primary_category: str = "role",
    is_mixed: bool = False,
    mixed_components: str = "",
    is_all_role: bool = False,
    all_role_components: str = "",
    is_seasonal: bool = False,
    seasonal_type: str = "",
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
        "is_seasonal": is_seasonal,
        "seasonal_type": seasonal_type,
    }


def history_by_candidate(rows: list[dict], target_month: str) -> pd.DataFrame:
    history = build_candidate_history(pd.DataFrame(rows), target_month)
    return history.set_index("candidate")


class CandidateHistoryTest(unittest.TestCase):
    def test_direct_role(self):
        history = history_by_candidate(
            [row("2026-01", "bounty_hunter")],
            "2026-07",
        )

        bounty_hunter = history.loc["bounty_hunter"]
        self.assertEqual(str(bounty_hunter["last_direct_seen"]), "2026-01")
        self.assertEqual(bounty_hunter["months_since_direct"], 6)
        self.assertEqual(bounty_hunter["direct_count"], 1)

    def test_direct_non_role(self):
        history = history_by_candidate(
            [
                row(
                    "2026-02",
                    "call_to_arms",
                    primary_category="non_role",
                )
            ],
            "2026-07",
        )

        call_to_arms = history.loc["call_to_arms"]
        self.assertEqual(str(call_to_arms["last_direct_seen"]), "2026-02")
        self.assertEqual(call_to_arms["months_since_direct"], 5)
        self.assertEqual(call_to_arms["direct_count"], 1)

    def test_mixed_does_not_overwrite_direct(self):
        history = history_by_candidate(
            [
                row("2026-01", "bounty_hunter"),
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

        bounty_hunter = history.loc["bounty_hunter"]
        self.assertEqual(str(bounty_hunter["last_direct_seen"]), "2026-01")
        self.assertEqual(str(bounty_hunter["last_mixed_seen"]), "2026-06")
        self.assertEqual(str(bounty_hunter["last_any_seen"]), "2026-06")
        self.assertEqual(bounty_hunter["direct_count"], 1)
        self.assertEqual(bounty_hunter["mixed_count"], 1)

    def test_all_role_does_not_overwrite_role_direct(self):
        history = history_by_candidate(
            [
                row("2026-01", "bounty_hunter"),
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

        bounty_hunter = history.loc["bounty_hunter"]
        self.assertEqual(str(bounty_hunter["last_direct_seen"]), "2026-01")
        self.assertEqual(str(bounty_hunter["last_all_role_seen"]), "2026-06")
        self.assertEqual(str(bounty_hunter["last_any_seen"]), "2026-06")
        self.assertEqual(bounty_hunter["direct_count"], 1)
        self.assertEqual(bounty_hunter["all_role_inclusion_count"], 1)

    def test_all_role_candidate_direct(self):
        history = history_by_candidate(
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

        all_role = history.loc["all_role"]
        self.assertEqual(all_role["candidate_group"], "all_role")
        self.assertEqual(str(all_role["last_direct_seen"]), "2026-06")
        self.assertEqual(all_role["direct_count"], 1)
        self.assertEqual(all_role["all_role_inclusion_count"], 0)

    def test_target_month_rows_are_not_used(self):
        history = history_by_candidate(
            [
                row("2026-01", "bounty_hunter"),
                row("2026-07", "bounty_hunter"),
            ],
            "2026-07",
        )

        bounty_hunter = history.loc["bounty_hunter"]
        self.assertEqual(str(bounty_hunter["last_direct_seen"]), "2026-01")
        self.assertEqual(bounty_hunter["direct_count"], 1)

    def test_taxonomy_candidate_with_no_rows_is_kept(self):
        history = history_by_candidate(
            [row("2026-01", "bounty_hunter")],
            "2026-07",
        )

        naturalist = history.loc["naturalist"]
        self.assertEqual(naturalist["candidate_group"], "role")
        self.assertEqual(naturalist["direct_count"], 0)
        self.assertEqual(naturalist["mixed_count"], 0)
        self.assertEqual(naturalist["all_role_inclusion_count"], 0)
        self.assertEqual(naturalist["any_count"], 0)
        self.assertTrue(pd.isna(naturalist["last_direct_seen"]))
        self.assertTrue(pd.isna(naturalist["months_since_direct"]))

    def test_thanksgiving_marks_trader_direct_without_thanksgiving_candidate(self):
        history = history_by_candidate(
            [
                row(
                    "2025-11",
                    "trader",
                    is_seasonal=True,
                    seasonal_type="thanksgiving",
                )
            ],
            "2025-12",
        )

        trader = history.loc["trader"]
        self.assertEqual(str(trader["last_direct_seen"]), "2025-11")
        self.assertEqual(trader["direct_count"], 1)
        self.assertNotIn("thanksgiving", set(history.index))


if __name__ == "__main__":
    unittest.main()
