import pandas as pd

from src.adjusted_predictor import calculate_adjusted_primary_category_prediction
from src.conditional_predictor import (
    DEFAULT_COOLDOWN_CONFIG,
    calculate_conditional_sub_category_prediction_with_config,
)
from src.data_loader import expand_components
from src.data_loader import VALID_PRIMARY_CATEGORIES
from src.global_predictor import (
    calculate_final_global_component_prediction,
    calculate_global_path_prediction,
)


BACKTEST_RESULT_COLUMNS = [
    "target_month",
    "actual_primary_category",
    "actual_primary_sub_category",
    "predicted_primary_category",
    "category_top1_hit",
    "path_top3_hit",
    "component_top3_hit",
    "actual_path_rank",
    "actual_component_rank",
]


def run_walk_forward_backtest(
    df: pd.DataFrame,
    prior_strength: float = 3,
    cooldown_config: dict[str, float] | None = None,
    min_train_months: int = 24,
) -> pd.DataFrame:
    """Walk forward through history and score each target month."""
    cooldown_config = cooldown_config or DEFAULT_COOLDOWN_CONFIG
    backtest_df = prepare_backtest_data(df)

    if len(backtest_df) <= min_train_months:
        return empty_backtest_result_table()

    target_months = backtest_df["month_period"].drop_duplicates().sort_values()
    if len(target_months) <= min_train_months:
        return empty_backtest_result_table()

    rows = []
    for target_month in target_months.iloc[min_train_months:]:
        train_df = backtest_df[backtest_df["month_period"] < target_month].copy()
        actual_rows = backtest_df[backtest_df["month_period"] == target_month]
        actual_row = actual_rows.iloc[0]

        adjusted_prediction = calculate_adjusted_primary_category_prediction(
            train_df,
            prior_strength=prior_strength,
        )
        adjusted_df = adjusted_prediction.table
        conditional_df = calculate_conditional_sub_category_prediction_with_config(
            train_df,
            cooldown_config=cooldown_config,
            parent_primary_prediction_df=adjusted_df,
            predicted_next_month=adjusted_prediction.predicted_next_month,
        )
        global_path_df = calculate_global_path_prediction(adjusted_df, conditional_df)
        final_component_df = calculate_final_global_component_prediction(
            conditional_df
        )

        actual_primary_category = actual_row["primary_category"]
        actual_primary_sub_categories = get_actual_sub_categories(actual_row)
        predicted_primary_category = get_top_primary_category(adjusted_df)
        actual_path_rank = get_actual_path_rank(
            global_path_df,
            actual_primary_category,
            actual_primary_sub_categories,
        )
        actual_component_rank = get_actual_component_rank(
            final_component_df,
            actual_primary_sub_categories,
        )

        rows.append(
            {
                "target_month": actual_row["month_label"],
                "actual_primary_category": actual_primary_category,
                "actual_primary_sub_category": "|".join(
                    actual_primary_sub_categories
                ),
                "predicted_primary_category": predicted_primary_category,
                "category_top1_hit": int(
                    predicted_primary_category == actual_primary_category
                ),
                "path_top3_hit": int(actual_path_rank <= 3),
                "component_top3_hit": int(actual_component_rank <= 3),
                "actual_path_rank": actual_path_rank,
                "actual_component_rank": actual_component_rank,
            }
        )

    return pd.DataFrame(rows, columns=BACKTEST_RESULT_COLUMNS)


def prepare_backtest_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "month_label",
        "primary_category",
        "primary_sub_category",
        "is_mixed",
        "mixed_components",
        "is_all_role",
        "all_role_components",
        "weight",
    ]
    if df.empty or any(column not in df.columns for column in required_columns):
        return pd.DataFrame()

    backtest_df = df.copy()
    backtest_df = backtest_df[
        backtest_df["primary_category"].isin(VALID_PRIMARY_CATEGORIES)
    ]

    month_dates = pd.to_datetime(
        backtest_df["month_label"],
        format="%Y-%m",
        errors="coerce",
    )
    backtest_df["month_period"] = month_dates.dt.to_period("M")
    backtest_df = backtest_df.dropna(subset=["month_period"])

    return backtest_df.sort_values("month_period").reset_index(drop=True)


def get_top_primary_category(adjusted_df: pd.DataFrame) -> str:
    if adjusted_df.empty:
        return ""

    return str(adjusted_df.iloc[0]["primary_category"])


def get_actual_sub_categories(actual_row: pd.Series) -> list[str]:
    return [record["component"] for record in expand_components(actual_row)]


def get_actual_path_rank(
    global_path_df: pd.DataFrame,
    actual_primary_category: str,
    actual_primary_sub_categories: list[str],
) -> int:
    if global_path_df.empty or not actual_primary_sub_categories:
        return 999

    ranked_df = global_path_df.reset_index(drop=True)
    matches = ranked_df[
        (ranked_df["primary_category"] == actual_primary_category)
        & (
            ranked_df["primary_sub_category"].isin(
                actual_primary_sub_categories
            )
        )
    ]
    if matches.empty:
        return 999

    return int(matches.index[0]) + 1


def get_actual_component_rank(
    final_component_df: pd.DataFrame,
    actual_primary_sub_categories: list[str],
) -> int:
    if final_component_df.empty or not actual_primary_sub_categories:
        return 999

    ranked_df = final_component_df.reset_index(drop=True)
    matches = ranked_df[
        ranked_df["component"].isin(actual_primary_sub_categories)
    ]
    if matches.empty:
        return 999
    return int(matches.index[0]) + 1


def empty_backtest_result_table() -> pd.DataFrame:
    return pd.DataFrame(columns=BACKTEST_RESULT_COLUMNS)
