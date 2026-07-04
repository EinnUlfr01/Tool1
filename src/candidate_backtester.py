from __future__ import annotations

from statistics import median

import pandas as pd

from src.candidate_pool import get_candidate_group, normalize_candidate_key
from src.candidate_scorer import CandidateScoringConfig, score_candidates
from src.data_loader import parse_components


CANDIDATE_BACKTEST_COLUMNS = [
    "target_month",
    "target_type",
    "actual_candidates",
    "num_candidates",
    "num_actual_candidates",
    "top_candidate",
    "top_candidate_display_label",
    "top_candidate_probability",
    "top_candidate_probability_percent",
    "top3_candidates",
    "top5_candidates",
    "candidate_top1_hit",
    "candidate_top3_hit",
    "candidate_top5_hit",
    "actual_candidate_rank",
    "best_actual_candidate_rank",
    "average_actual_candidate_rank",
    "median_actual_candidate_rank",
    "average_hit_actual_candidate_rank",
    "median_hit_actual_candidate_rank",
    "capped_actual_candidate_rank",
    "average_capped_actual_candidate_rank",
    "miss_rate",
    "actual_candidate_probability",
    "actual_candidate_probability_percent",
    "average_actual_candidate_probability",
    "average_actual_candidate_probability_percent",
    "prior_strength",
    "notes",
]


def run_candidate_walk_forward_backtest(
    df: pd.DataFrame,
    min_train_months: int = 24,
    prior_strength: float = 3,
    config: CandidateScoringConfig | None = None,
) -> pd.DataFrame:
    """Walk forward through history with Candidate Scoring V2.1.

    Each target month is scored using only rows before that month. The target row
    and all future rows are excluded from training to avoid data leakage.
    """
    backtest_df = prepare_candidate_backtest_data(df)
    if backtest_df.empty:
        return empty_candidate_backtest_table()

    target_months = backtest_df["month_period"].drop_duplicates().sort_values()
    if len(target_months) <= min_train_months:
        return empty_candidate_backtest_table()

    rows = []
    for target_month in target_months.iloc[min_train_months:]:
        train_df = backtest_df[backtest_df["month_period"] < target_month].copy()
        actual_rows = backtest_df[backtest_df["month_period"] == target_month]
        target_month_label = str(target_month)
        ranking_df = score_candidates(
            train_df,
            target_month_label,
            prior_strength=prior_strength,
            config=config,
        )
        actual_candidates = extract_actual_candidates_from_rows(actual_rows)
        target_type = infer_target_type(actual_rows)
        rows.append(
            summarize_target_month(
                target_month_label,
                target_type,
                ranking_df,
                actual_candidates,
                prior_strength,
            )
        )

    return pd.DataFrame(rows, columns=CANDIDATE_BACKTEST_COLUMNS)


def prepare_candidate_backtest_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "month_label" not in df.columns:
        return pd.DataFrame()

    backtest_df = df.copy()
    month_dates = pd.to_datetime(
        backtest_df["month_label"],
        format="%Y-%m",
        errors="coerce",
    )
    backtest_df["month_period"] = month_dates.dt.to_period("M")
    backtest_df = backtest_df.dropna(subset=["month_period"])
    return backtest_df.sort_values("month_period").reset_index(drop=True)


def extract_actual_candidates(row: pd.Series) -> list[str]:
    """Extract canonical actual candidates from one CSV V2 row.

    All Roles is treated as the distinct all_role candidate. Mixed/both rows are
    evaluated by their component list, without creating a synthetic mixed
    candidate. Seasonal context such as Thanksgiving does not become a candidate.
    """
    if is_truthy(row.get("is_all_role", False)):
        return ["all_role"]

    if is_truthy(row.get("is_mixed", False)) or get_text(row, "primary_category") == "both":
        return dedupe_candidates(
            normalize_candidate_key(component)
            for component in parse_components(row.get("mixed_components", ""))
        )

    candidate = first_non_empty(
        row.get("effective_candidate", ""),
        row.get("effective_primary_sub_category", ""),
        row.get("primary_sub_category", ""),
    )
    return dedupe_candidates([normalize_candidate_key(candidate)])


def extract_actual_candidates_from_rows(actual_rows: pd.DataFrame) -> list[str]:
    candidates = []
    for _, row in actual_rows.iterrows():
        candidates.extend(extract_actual_candidates(row))
    return dedupe_candidates(candidates)


def get_actual_candidates(actual_rows: pd.DataFrame) -> list[str]:
    """Backward-compatible wrapper for callers/tests."""
    return extract_actual_candidates_from_rows(actual_rows)


def infer_target_type(actual_rows: pd.DataFrame) -> str:
    if actual_rows.empty:
        return ""
    if any(is_truthy(value) for value in actual_rows.get("is_all_role", [])):
        return "all_role"
    if any(is_truthy(value) for value in actual_rows.get("is_mixed", [])):
        return "mixed"
    categories = {
        normalize_candidate_key(value)
        for value in actual_rows.get("primary_category", [])
        if str(value or "").strip()
    }
    if "both" in categories:
        return "mixed"
    if len(categories) == 1:
        return next(iter(categories))
    return "|".join(sorted(categories))


def summarize_target_month(
    target_month: str,
    target_type: str,
    ranking_df: pd.DataFrame,
    actual_candidates: list[str],
    prior_strength: float,
) -> dict:
    if ranking_df.empty:
        return empty_target_month_summary(
            target_month,
            target_type,
            actual_candidates,
            prior_strength,
            "empty_ranking",
        )

    ranked_df = ranking_df.reset_index(drop=True)
    top_row = ranked_df.iloc[0]
    top3_candidates = ranked_df.head(3)["candidate"].tolist()
    top5_candidates = ranked_df.head(5)["candidate"].tolist()
    actual_set = set(actual_candidates)
    rank_by_candidate = dict(zip(ranked_df["candidate"], ranked_df["rank"]))
    probability_by_candidate = dict(zip(ranked_df["candidate"], ranked_df["final_probability"]))
    percent_by_candidate = dict(
        zip(ranked_df["candidate"], ranked_df["final_probability_percent"])
    )

    actual_ranks = [
        int(rank_by_candidate[candidate])
        for candidate in actual_candidates
        if candidate in rank_by_candidate
    ]
    actual_probability_values = [
        float(probability_by_candidate[candidate])
        for candidate in actual_candidates
        if candidate in probability_by_candidate
    ]
    actual_probability_percent_values = [
        float(percent_by_candidate[candidate])
        for candidate in actual_candidates
        if candidate in percent_by_candidate
    ]

    miss_count = len(actual_candidates) - len(actual_ranks)
    miss_rate = miss_count / len(actual_candidates) if actual_candidates else 0.0
    capped_rank = len(ranked_df) + 1
    capped_ranks = actual_ranks + [capped_rank] * miss_count
    best_rank = min(actual_ranks) if actual_ranks else 999
    average_hit_rank = sum(actual_ranks) / len(actual_ranks) if actual_ranks else 0.0
    median_hit_rank = median(actual_ranks) if actual_ranks else 0.0
    average_capped_rank = (
        sum(capped_ranks) / len(capped_ranks) if capped_ranks else 0.0
    )
    actual_probability = (
        max(actual_probability_values) if actual_probability_values else 0.0
    )
    actual_probability_percent = (
        max(actual_probability_percent_values)
        if actual_probability_percent_values
        else 0.0
    )
    average_actual_probability = (
        sum(actual_probability_values) / len(actual_probability_values)
        if actual_probability_values
        else 0.0
    )
    average_actual_probability_percent = (
        sum(actual_probability_percent_values) / len(actual_probability_percent_values)
        if actual_probability_percent_values
        else 0.0
    )
    missing_actuals = [
        candidate for candidate in actual_candidates if candidate not in rank_by_candidate
    ]
    notes = ""
    if missing_actuals:
        notes = "missing_actual_candidates=" + "|".join(missing_actuals)

    return {
        "target_month": target_month,
        "target_type": target_type,
        "actual_candidates": "|".join(actual_candidates),
        "num_candidates": len(ranked_df),
        "num_actual_candidates": len(actual_candidates),
        "top_candidate": str(top_row["candidate"]),
        "top_candidate_display_label": str(top_row["display_label"]),
        "top_candidate_probability": float(top_row["final_probability"]),
        "top_candidate_probability_percent": float(top_row["final_probability_percent"]),
        "top3_candidates": "|".join(top3_candidates),
        "top5_candidates": "|".join(top5_candidates),
        "candidate_top1_hit": int(str(top_row["candidate"]) in actual_set),
        "candidate_top3_hit": int(bool(actual_set.intersection(top3_candidates))),
        "candidate_top5_hit": int(bool(actual_set.intersection(top5_candidates))),
        "actual_candidate_rank": best_rank,
        "best_actual_candidate_rank": best_rank,
        "average_actual_candidate_rank": average_hit_rank,
        "median_actual_candidate_rank": median_hit_rank,
        "average_hit_actual_candidate_rank": average_hit_rank,
        "median_hit_actual_candidate_rank": median_hit_rank,
        "capped_actual_candidate_rank": min(capped_ranks) if capped_ranks else 0.0,
        "average_capped_actual_candidate_rank": average_capped_rank,
        "miss_rate": miss_rate,
        "actual_candidate_probability": actual_probability,
        "actual_candidate_probability_percent": actual_probability_percent,
        "average_actual_candidate_probability": average_actual_probability,
        "average_actual_candidate_probability_percent": average_actual_probability_percent,
        "prior_strength": prior_strength,
        "notes": notes,
    }


def summarize_candidate_backtest_results(results: pd.DataFrame) -> dict:
    if results.empty:
        return {
            "num_tests": 0,
            "candidate_top1_accuracy": 0,
            "candidate_top3_accuracy": 0,
            "candidate_top5_accuracy": 0,
            "average_actual_candidate_rank": 999,
            "median_actual_candidate_rank": 999,
            "average_hit_actual_candidate_rank": 0,
            "median_hit_actual_candidate_rank": 0,
            "miss_rate": 0,
            "average_capped_actual_candidate_rank": 0,
            "average_actual_probability_percent": 0,
            "average_top_candidate_probability_percent": 0,
            "top_candidate_group_distribution": {},
            "percentage_top1_role": 0,
            "percentage_top1_non_role": 0,
            "percentage_top1_all_role": 0,
            "percentage_top1_seasonal": 0,
            "prior_strength": 3,
        }

    top_groups = results["top_candidate"].apply(get_candidate_group)
    group_distribution = top_groups.value_counts(normalize=True).to_dict()
    hit_rank_values = results["average_hit_actual_candidate_rank"][
        results["average_hit_actual_candidate_rank"].gt(0)
    ]
    median_hit_rank_values = results["median_hit_actual_candidate_rank"][
        results["median_hit_actual_candidate_rank"].gt(0)
    ]
    summary = {
        "num_tests": len(results),
        "candidate_top1_accuracy": float(results["candidate_top1_hit"].mean()),
        "candidate_top3_accuracy": float(results["candidate_top3_hit"].mean()),
        "candidate_top5_accuracy": float(results["candidate_top5_hit"].mean()),
        "average_actual_candidate_rank": float(
            results["average_actual_candidate_rank"].mean()
        ),
        "median_actual_candidate_rank": float(
            results["median_actual_candidate_rank"].median()
        ),
        "average_hit_actual_candidate_rank": float(hit_rank_values.mean())
        if not hit_rank_values.empty
        else 0.0,
        "median_hit_actual_candidate_rank": float(median_hit_rank_values.median())
        if not median_hit_rank_values.empty
        else 0.0,
        "miss_rate": float(results["miss_rate"].mean()),
        "average_capped_actual_candidate_rank": float(
            results["average_capped_actual_candidate_rank"].mean()
        ),
        "average_actual_probability_percent": float(
            results["average_actual_candidate_probability_percent"].mean()
        ),
        "average_top_candidate_probability_percent": float(
            results["top_candidate_probability_percent"].mean()
        ),
        "top_candidate_group_distribution": group_distribution,
        "percentage_top1_role": float(group_distribution.get("role", 0.0)),
        "percentage_top1_non_role": float(group_distribution.get("non_role", 0.0)),
        "percentage_top1_all_role": float(group_distribution.get("all_role", 0.0)),
        "percentage_top1_seasonal": float(group_distribution.get("seasonal", 0.0)),
        "prior_strength": float(results["prior_strength"].iloc[0]),
    }
    for target_type, type_df in results.groupby("target_type"):
        if not target_type:
            continue
        prefix = f"{target_type}_"
        summary[prefix + "num_tests"] = len(type_df)
        summary[prefix + "top3_accuracy"] = float(
            type_df["candidate_top3_hit"].mean()
        )
        summary[prefix + "average_actual_candidate_rank"] = float(
            type_df["average_actual_candidate_rank"].mean()
        )
    return summary


def empty_target_month_summary(
    target_month: str,
    target_type: str,
    actual_candidates: list[str],
    prior_strength: float,
    notes: str,
) -> dict:
    return {
        "target_month": target_month,
        "target_type": target_type,
        "actual_candidates": "|".join(actual_candidates),
        "num_candidates": 0,
        "num_actual_candidates": len(actual_candidates),
        "top_candidate": "",
        "top_candidate_display_label": "",
        "top_candidate_probability": 0,
        "top_candidate_probability_percent": 0,
        "top3_candidates": "",
        "top5_candidates": "",
        "candidate_top1_hit": 0,
        "candidate_top3_hit": 0,
        "candidate_top5_hit": 0,
        "actual_candidate_rank": 999,
        "best_actual_candidate_rank": 999,
        "average_actual_candidate_rank": 999,
        "median_actual_candidate_rank": 999,
        "average_hit_actual_candidate_rank": 0,
        "median_hit_actual_candidate_rank": 0,
        "capped_actual_candidate_rank": 0,
        "average_capped_actual_candidate_rank": 0,
        "miss_rate": 1 if actual_candidates else 0,
        "actual_candidate_probability": 0,
        "actual_candidate_probability_percent": 0,
        "average_actual_candidate_probability": 0,
        "average_actual_candidate_probability_percent": 0,
        "prior_strength": prior_strength,
        "notes": notes,
    }


def empty_candidate_backtest_table() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_BACKTEST_COLUMNS)


def dedupe_candidates(candidates) -> list[str]:
    result = []
    seen = set()
    for candidate in candidates:
        normalized = normalize_candidate_key(candidate)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def first_non_empty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def get_text(row: pd.Series, column: str) -> str:
    return str(row.get(column, "") or "").strip().lower()


def is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}
