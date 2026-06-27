from dataclasses import dataclass

import pandas as pd

from src.data_loader import VALID_PRIMARY_CATEGORIES


PREDICTION_COLUMNS = [
    "primary_category",
    "count",
    "base_probability_percent",
    "observed_transition_count",
    "total_transition_from_previous",
    "smoothed_transition_probability_percent",
    "transition_multiplier",
    "domain_multiplier",
    "final_category_multiplier",
    "raw_score",
    "adjusted_prediction_percent",
]

AFTER_ALL_ROLE_MULTIPLIERS = {
    "role": 0.45,
    "non_role": 1.45,
    "both": 1.15,
}
ALL_ROLE_REPEAT_PENALTY = 0.10
MIN_PRIOR_STRENGTH = 2.0
MAX_PRIOR_STRENGTH = 8.0


@dataclass
class AdjustedPrediction:
    latest_month: str
    predicted_next_month: str
    previous_category: str
    latest_is_all_role: bool
    table: pd.DataFrame


def calculate_adjusted_primary_category_prediction(
    df: pd.DataFrame,
    prior_strength: float = 3,
) -> AdjustedPrediction:
    """Predict primary categories using frequency and smoothed transitions."""
    sorted_df = prepare_prediction_data(df)
    if sorted_df.empty:
        return AdjustedPrediction("", "", "", False, empty_prediction_table())

    latest_row = sorted_df.iloc[-1]
    latest_month = latest_row["month_label"]
    predicted_next_month = str(latest_row["month_period"] + 1)
    previous_category = latest_row["primary_category"]
    latest_is_all_role = bool(latest_row["is_all_role"])
    prior_strength = clamp_prior_strength(prior_strength)

    base_table = build_base_probability_table(sorted_df)
    transitions = build_transition_pairs(sorted_df)
    total_transition_from_previous = int(
        (transitions["from_category"] == previous_category).sum()
    )

    rows = []
    for row in base_table.itertuples(index=False):
        category = row.primary_category
        base_probability = row.base_probability
        base_probability_percent = base_probability * 100
        observed_transition_count = count_observed_transition(
            transitions,
            previous_category,
            category,
        )
        smoothed_transition_probability = calculate_smoothed_transition_probability(
            observed_transition_count,
            total_transition_from_previous,
            prior_strength,
            base_probability,
        )
        transition_multiplier = (
            smoothed_transition_probability / base_probability
            if base_probability > 0
            else 0.0
        )
        domain_multiplier = get_domain_multiplier(latest_is_all_role, category)
        final_category_multiplier = transition_multiplier * domain_multiplier

        rows.append(
            {
                "primary_category": category,
                "count": int(row.count),
                "base_probability_percent": base_probability_percent,
                "observed_transition_count": observed_transition_count,
                "total_transition_from_previous": total_transition_from_previous,
                "smoothed_transition_probability_percent": (
                    smoothed_transition_probability * 100
                ),
                "transition_multiplier": transition_multiplier,
                "domain_multiplier": domain_multiplier,
                "final_category_multiplier": final_category_multiplier,
                "raw_score": base_probability_percent * final_category_multiplier,
            }
        )

    prediction_df = pd.DataFrame(rows)
    raw_score_total = float(prediction_df["raw_score"].sum())
    prediction_df["adjusted_prediction_percent"] = (
        prediction_df["raw_score"] / raw_score_total * 100
        if raw_score_total > 0
        else prediction_df["base_probability_percent"]
    )
    prediction_df = prediction_df.sort_values(
        "adjusted_prediction_percent",
        ascending=False,
    )

    return AdjustedPrediction(
        latest_month=latest_month,
        predicted_next_month=predicted_next_month,
        previous_category=previous_category,
        latest_is_all_role=latest_is_all_role,
        table=round_prediction_table(prediction_df),
    )


def prepare_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "month_label",
        "primary_category",
        "is_all_role",
        "weight",
    ]
    if df.empty or any(column not in df.columns for column in required_columns):
        return pd.DataFrame()

    prediction_df = df.copy()
    prediction_df = prediction_df[
        prediction_df["primary_category"].isin(VALID_PRIMARY_CATEGORIES)
    ]
    month_dates = pd.to_datetime(
        prediction_df["month_label"],
        format="%Y-%m",
        errors="coerce",
    )
    prediction_df["month_period"] = month_dates.dt.to_period("M")
    prediction_df = prediction_df.dropna(subset=["month_period"])
    return prediction_df.sort_values("month_period").reset_index(drop=True)


def build_base_probability_table(df: pd.DataFrame) -> pd.DataFrame:
    category_weights = (
        df.groupby("primary_category", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "weighted_count"})
    )
    category_counts = (
        df.groupby("primary_category", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    base_table = category_weights.merge(category_counts, on="primary_category")
    total_weight = float(base_table["weighted_count"].sum())
    base_table["base_probability"] = (
        base_table["weighted_count"] / total_weight if total_weight > 0 else 0.0
    )
    return base_table


def build_transition_pairs(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame(columns=["from_category", "to_category"])

    return pd.DataFrame(
        {
            "from_category": df["primary_category"].iloc[:-1].to_list(),
            "to_category": df["primary_category"].iloc[1:].to_list(),
        }
    )


def count_observed_transition(
    transitions: pd.DataFrame,
    previous_category: str,
    category: str,
) -> int:
    if transitions.empty:
        return 0
    return int(
        (
            (transitions["from_category"] == previous_category)
            & (transitions["to_category"] == category)
        ).sum()
    )


def calculate_smoothed_transition_probability(
    observed_transition_count: int,
    total_transition_from_previous: int,
    prior_strength: float,
    base_probability: float,
) -> float:
    denominator = total_transition_from_previous + prior_strength
    if denominator <= 0:
        return base_probability
    return (
        observed_transition_count + prior_strength * base_probability
    ) / denominator


def clamp_prior_strength(prior_strength: float) -> float:
    try:
        value = float(prior_strength)
    except (TypeError, ValueError):
        value = 3.0
    return min(MAX_PRIOR_STRENGTH, max(MIN_PRIOR_STRENGTH, value))


def get_domain_multiplier(latest_is_all_role: bool, category: str) -> float:
    if not latest_is_all_role:
        return 1.0
    return float(AFTER_ALL_ROLE_MULTIPLIERS.get(category, 1.0))


def round_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    rounded_df = df.copy()
    numeric_columns = [
        "base_probability_percent",
        "smoothed_transition_probability_percent",
        "transition_multiplier",
        "domain_multiplier",
        "final_category_multiplier",
        "raw_score",
        "adjusted_prediction_percent",
    ]
    rounded_df[numeric_columns] = rounded_df[numeric_columns].round(4)
    return rounded_df[PREDICTION_COLUMNS]


def empty_prediction_table() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)
