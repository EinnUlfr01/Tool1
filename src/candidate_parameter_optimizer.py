from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.candidate_backtester import (
    run_candidate_walk_forward_backtest,
    summarize_candidate_backtest_results,
)
from src.candidate_scorer import CandidateScoringConfig


CANDIDATE_PRIOR_STRENGTH_CANDIDATES = [3, 4, 5]

CANDIDATE_OPTIMIZER_COLUMNS = [
    "prior_strength",
    "num_tests",
    "candidate_top1_accuracy",
    "candidate_top3_accuracy",
    "candidate_top5_accuracy",
    "average_actual_candidate_rank",
    "median_actual_candidate_rank",
    "average_hit_actual_candidate_rank",
    "median_hit_actual_candidate_rank",
    "miss_rate",
    "average_capped_actual_candidate_rank",
    "average_actual_probability_percent",
    "average_top_candidate_probability_percent",
    "rank_score",
    "probability_score",
    "final_candidate_backtest_score",
]


@dataclass
class CandidateOptimizationResult:
    enough_data: bool
    comparison_table: pd.DataFrame
    best_prior_strength: float
    best_final_candidate_backtest_score: float


def optimize_candidate_scoring_parameters(
    df: pd.DataFrame,
    min_train_months: int = 24,
    prior_strength_candidates: list[float] | None = None,
    config: CandidateScoringConfig | None = None,
) -> CandidateOptimizationResult:
    """Calibrate Candidate Scoring V2.1 prior strength.

    The final score is only for comparing Candidate Scoring V2.1 prior values:
    top3 accuracy * 0.35 + top5 accuracy * 0.25 + rank_score * 0.20 +
    probability_score * 0.10 + miss_score * 0.10. It is not comparable with
    the legacy optimizer.
    """
    prior_strength_candidates = (
        prior_strength_candidates or CANDIDATE_PRIOR_STRENGTH_CANDIDATES
    )

    rows = []
    for prior_strength in prior_strength_candidates:
        backtest_df = run_candidate_walk_forward_backtest(
            df,
            min_train_months=min_train_months,
            prior_strength=prior_strength,
            config=config,
        )
        if backtest_df.empty:
            continue
        rows.append(build_optimizer_row(backtest_df, prior_strength))

    if not rows:
        return CandidateOptimizationResult(
            enough_data=False,
            comparison_table=empty_candidate_optimizer_table(),
            best_prior_strength=3,
            best_final_candidate_backtest_score=0,
        )

    comparison_table = pd.DataFrame(rows)
    comparison_table = comparison_table.sort_values(
        [
            "final_candidate_backtest_score",
            "candidate_top3_accuracy",
            "candidate_top5_accuracy",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best_row = comparison_table.iloc[0]

    return CandidateOptimizationResult(
        enough_data=True,
        comparison_table=round_candidate_optimizer_table(comparison_table),
        best_prior_strength=float(best_row["prior_strength"]),
        best_final_candidate_backtest_score=float(
            best_row["final_candidate_backtest_score"]
        ),
    )


def build_optimizer_row(backtest_df: pd.DataFrame, prior_strength: float) -> dict:
    summary = summarize_candidate_backtest_results(backtest_df)
    average_rank = float(summary["average_hit_actual_candidate_rank"])
    rank_score = 0 if average_rank <= 0 else 1 / average_rank
    probability_score = float(summary["average_actual_probability_percent"]) / 100
    miss_score = 1 - float(summary["miss_rate"])
    final_score = (
        float(summary["candidate_top3_accuracy"]) * 0.35
        + float(summary["candidate_top5_accuracy"]) * 0.25
        + rank_score * 0.20
        + probability_score * 0.10
        + miss_score * 0.10
    )

    return {
        "prior_strength": prior_strength,
        "num_tests": int(summary["num_tests"]),
        "candidate_top1_accuracy": summary["candidate_top1_accuracy"],
        "candidate_top3_accuracy": summary["candidate_top3_accuracy"],
        "candidate_top5_accuracy": summary["candidate_top5_accuracy"],
        "average_actual_candidate_rank": summary["average_actual_candidate_rank"],
        "median_actual_candidate_rank": summary["median_actual_candidate_rank"],
        "average_hit_actual_candidate_rank": summary[
            "average_hit_actual_candidate_rank"
        ],
        "median_hit_actual_candidate_rank": summary["median_hit_actual_candidate_rank"],
        "miss_rate": summary["miss_rate"],
        "average_capped_actual_candidate_rank": summary[
            "average_capped_actual_candidate_rank"
        ],
        "average_actual_probability_percent": summary[
            "average_actual_probability_percent"
        ],
        "average_top_candidate_probability_percent": summary[
            "average_top_candidate_probability_percent"
        ],
        "rank_score": rank_score,
        "probability_score": probability_score,
        "final_candidate_backtest_score": final_score,
    }


def round_candidate_optimizer_table(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    numeric_columns = [
        "candidate_top1_accuracy",
        "candidate_top3_accuracy",
        "candidate_top5_accuracy",
        "average_actual_candidate_rank",
        "median_actual_candidate_rank",
        "average_hit_actual_candidate_rank",
        "median_hit_actual_candidate_rank",
        "miss_rate",
        "average_capped_actual_candidate_rank",
        "average_actual_probability_percent",
        "average_top_candidate_probability_percent",
        "rank_score",
        "probability_score",
        "final_candidate_backtest_score",
    ]
    result_df[numeric_columns] = result_df[numeric_columns].round(4)
    return result_df[CANDIDATE_OPTIMIZER_COLUMNS]


def empty_candidate_optimizer_table() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_OPTIMIZER_COLUMNS)
