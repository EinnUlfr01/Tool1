from dataclasses import dataclass

import pandas as pd


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

ALL_ROLE_DOMAIN_MULTIPLIERS = {
    "non_role": 1.25,
    "mixed": 1.10,
    "role": 0.65,
    "all_role": 0.10,
}


@dataclass
class AdjustedPrediction:
    latest_month: str
    predicted_next_month: str
    previous_category: str
    table: pd.DataFrame


def calculate_adjusted_primary_category_prediction(
    df: pd.DataFrame,
    prior_strength: float = 3,
) -> AdjustedPrediction:
    """Build adjusted prediction scores without machine learning."""
    sorted_df = prepare_prediction_data(df)
    if sorted_df.empty:
        return AdjustedPrediction("", "", "", empty_prediction_table())

    latest_row = sorted_df.iloc[-1]
    latest_month = latest_row["month_label"]
    latest_period = latest_row["month_period"]
    predicted_next_month = str(latest_period + 1)
    previous_category = latest_row["primary_category"]

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

        if base_probability == 0:
            transition_multiplier = 0
        else:
            transition_multiplier = smoothed_transition_probability / base_probability

        domain_multiplier = get_domain_multiplier(previous_category, category)
        final_category_multiplier = transition_multiplier * domain_multiplier
        raw_score = base_probability_percent * final_category_multiplier

        rows.append(
            {
                "primary_category": category,
                "count": row.count,
                "base_probability_percent": base_probability_percent,
                "observed_transition_count": observed_transition_count,
                "total_transition_from_previous": total_transition_from_previous,
                "smoothed_transition_probability_percent": (
                    smoothed_transition_probability * 100
                ),
                "transition_multiplier": transition_multiplier,
                "domain_multiplier": domain_multiplier,
                "final_category_multiplier": final_category_multiplier,
                "raw_score": raw_score,
            }
        )

    prediction_df = pd.DataFrame(rows)
    raw_score_total = prediction_df["raw_score"].sum()
    if raw_score_total > 0:
        prediction_df["adjusted_prediction_percent"] = (
            prediction_df["raw_score"] / raw_score_total * 100
        )
    else:
        prediction_df["adjusted_prediction_percent"] = 0

    prediction_df = prediction_df.sort_values(
        "adjusted_prediction_percent",
        ascending=False,
    )

    return AdjustedPrediction(
        latest_month=latest_month,
        predicted_next_month=predicted_next_month,
        previous_category=previous_category,
        table=round_prediction_table(prediction_df),
    )


def prepare_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Use valid non-low rows sorted by month."""
    required_columns = ["month_label", "primary_category", "confidence"]
    if df.empty or any(column not in df.columns for column in required_columns):
        return pd.DataFrame()

    prediction_df = df.copy()
    prediction_df["confidence"] = prediction_df["confidence"].str.lower()
    prediction_df["primary_category"] = prediction_df["primary_category"].str.lower()
    prediction_df = prediction_df[prediction_df["confidence"] != "low"]
    prediction_df = prediction_df[prediction_df["primary_category"] != ""]

    month_dates = pd.to_datetime(
        prediction_df["month_label"],
        format="%Y-%m",
        errors="coerce",
    )
    prediction_df["month_period"] = month_dates.dt.to_period("M")
    prediction_df = prediction_df.dropna(subset=["month_period"])

    return prediction_df.sort_values("month_period").reset_index(drop=True)


def build_base_probability_table(df: pd.DataFrame) -> pd.DataFrame:
    total_months = len(df)
    base_table = (
        df["primary_category"]
        .value_counts()
        .rename_axis("primary_category")
        .reset_index(name="count")
    )
    base_table["base_probability"] = base_table["count"] / total_months
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
    return (
        observed_transition_count + prior_strength * base_probability
    ) / (total_transition_from_previous + prior_strength)


def get_domain_multiplier(previous_category: str, category: str) -> float:
    if previous_category != "all_role":
        return 1.0

    return ALL_ROLE_DOMAIN_MULTIPLIERS.get(category, 1.0)


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
