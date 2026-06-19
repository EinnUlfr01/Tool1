from dataclasses import dataclass

import pandas as pd

from src.backtester import prepare_backtest_data, run_walk_forward_backtest
from src.conditional_predictor import DEFAULT_COOLDOWN_CONFIG


PRIOR_STRENGTH_CANDIDATES = [2, 3, 5, 8]
COOLDOWN_CANDIDATES = [
    {"recent_1": 0.55, "recent_2": 0.75, "recent_3": 0.90},
    {"recent_1": 0.65, "recent_2": 0.80, "recent_3": 0.95},
    {"recent_1": 0.75, "recent_2": 0.90, "recent_3": 1.00},
]

OPTIMIZER_COLUMNS = [
    "prior_strength",
    "cooldown_recent_1",
    "cooldown_recent_2",
    "cooldown_recent_3",
    "category_top1_accuracy",
    "path_top3_accuracy",
    "component_top3_accuracy",
    "average_actual_path_rank",
    "rank_score",
    "final_backtest_score",
]


@dataclass
class OptimizationResult:
    enough_data: bool
    comparison_table: pd.DataFrame
    best_prior_strength: float
    best_cooldown_config: dict[str, float]
    best_final_backtest_score: float


def optimize_parameters(
    df: pd.DataFrame,
    min_train_months: int = 24,
    prior_strength_candidates: list[float] | None = None,
    cooldown_candidates: list[dict[str, float]] | None = None,
) -> OptimizationResult:
    """Try candidate configs and select the best backtested score."""
    prior_strength_candidates = prior_strength_candidates or PRIOR_STRENGTH_CANDIDATES
    cooldown_candidates = cooldown_candidates or COOLDOWN_CANDIDATES

    backtest_df = prepare_backtest_data(df)
    month_count = backtest_df["month_period"].nunique() if not backtest_df.empty else 0
    if month_count <= min_train_months:
        return OptimizationResult(
            enough_data=False,
            comparison_table=empty_optimizer_table(),
            best_prior_strength=3,
            best_cooldown_config=DEFAULT_COOLDOWN_CONFIG,
            best_final_backtest_score=0,
        )

    rows = []
    for prior_strength in prior_strength_candidates:
        for cooldown_config in cooldown_candidates:
            backtest_result_df = run_walk_forward_backtest(
                backtest_df,
                prior_strength=prior_strength,
                cooldown_config=cooldown_config,
                min_train_months=min_train_months,
            )
            rows.append(
                summarize_candidate_result(
                    backtest_result_df,
                    prior_strength,
                    cooldown_config,
                )
            )

    comparison_table = pd.DataFrame(rows)
    comparison_table = comparison_table.sort_values(
        ["final_backtest_score", "path_top3_accuracy", "category_top1_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best_row = comparison_table.iloc[0]
    best_cooldown_config = {
        "recent_1": float(best_row["cooldown_recent_1"]),
        "recent_2": float(best_row["cooldown_recent_2"]),
        "recent_3": float(best_row["cooldown_recent_3"]),
    }

    return OptimizationResult(
        enough_data=True,
        comparison_table=round_optimizer_table(comparison_table),
        best_prior_strength=float(best_row["prior_strength"]),
        best_cooldown_config=best_cooldown_config,
        best_final_backtest_score=float(best_row["final_backtest_score"]),
    )


def summarize_candidate_result(
    backtest_result_df: pd.DataFrame,
    prior_strength: float,
    cooldown_config: dict[str, float],
) -> dict:
    if backtest_result_df.empty:
        category_top1_accuracy = 0
        path_top3_accuracy = 0
        component_top3_accuracy = 0
        average_actual_path_rank = 999
    else:
        category_top1_accuracy = backtest_result_df["category_top1_hit"].mean()
        path_top3_accuracy = backtest_result_df["path_top3_hit"].mean()
        component_top3_accuracy = backtest_result_df["component_top3_hit"].mean()
        average_actual_path_rank = backtest_result_df["actual_path_rank"].mean()

    rank_score = 0
    if average_actual_path_rank > 0:
        rank_score = 1 / average_actual_path_rank

    final_backtest_score = (
        category_top1_accuracy * 0.4
        + component_top3_accuracy * 0.4
        + rank_score * 0.2
    )

    return {
        "prior_strength": prior_strength,
        "cooldown_recent_1": cooldown_config["recent_1"],
        "cooldown_recent_2": cooldown_config["recent_2"],
        "cooldown_recent_3": cooldown_config["recent_3"],
        "category_top1_accuracy": category_top1_accuracy,
        "path_top3_accuracy": path_top3_accuracy,
        "component_top3_accuracy": component_top3_accuracy,
        "average_actual_path_rank": average_actual_path_rank,
        "rank_score": rank_score,
        "final_backtest_score": final_backtest_score,
    }


def round_optimizer_table(df: pd.DataFrame) -> pd.DataFrame:
    rounded_df = df.copy()
    numeric_columns = [
        "category_top1_accuracy",
        "path_top3_accuracy",
        "component_top3_accuracy",
        "average_actual_path_rank",
        "rank_score",
        "final_backtest_score",
    ]
    rounded_df[numeric_columns] = rounded_df[numeric_columns].round(4)
    return rounded_df[OPTIMIZER_COLUMNS]


def empty_optimizer_table() -> pd.DataFrame:
    return pd.DataFrame(columns=OPTIMIZER_COLUMNS)
