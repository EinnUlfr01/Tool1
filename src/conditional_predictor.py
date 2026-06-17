import pandas as pd

from src.seasonal_rules import is_seasonal_sub_category_allowed


ROLE_SUB_CATEGORIES = {
    "trader",
    "collector",
    "bounty_hunter",
    "moonshiner",
    "naturalist",
}

ALL_ROLE_ROLE_COOLDOWN_MULTIPLIER = 0.5

DEFAULT_COOLDOWN_CONFIG = {
    "recent_1": 0.2,
    "recent_2": 0.4,
    "recent_3": 0.6,
}

CONDITIONAL_COLUMNS = [
    "primary_category",
    "primary_sub_category",
    "count",
    "base_probability_percent",
    "conditional_base_percent",
    "conditional_sub_probability_percent",
    "cooldown_multiplier",
    "all_role_role_cooldown_multiplier",
    "final_cooldown_multiplier",
    "adjusted_score",
    "sub_raw_score",
    "normalized_sub_category_weight",
    "adjusted_conditional_sub_probability_percent",
    "parent_primary_adjusted_prediction_percent",
    "adjusted_prediction_percent",
]


def calculate_conditional_sub_category_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate layered sub-category prediction scores."""
    return calculate_conditional_sub_category_prediction_with_config(
        df,
        cooldown_config=DEFAULT_COOLDOWN_CONFIG,
    )


def calculate_conditional_sub_category_prediction_with_config(
    df: pd.DataFrame,
    cooldown_config: dict[str, float] | None = None,
    parent_primary_prediction_df: pd.DataFrame | None = None,
    predicted_next_month: str | None = None,
) -> pd.DataFrame:
    """Calculate layered sub-category prediction scores."""
    cooldown_config = normalize_cooldown_config(cooldown_config)
    sorted_df = prepare_conditional_data(df)
    if sorted_df.empty:
        return empty_conditional_table()

    latest_primary_category = sorted_df.iloc[-1]["primary_category"]
    recent_sub_categories = get_recent_sub_categories(sorted_df, recent_months=3)
    total_count = len(sorted_df)

    rows = []
    for primary_category in sorted(sorted_df["primary_category"].unique()):
        category_df = sorted_df[sorted_df["primary_category"] == primary_category]

        if primary_category in {"mixed", "all_role"}:
            rows.append(
                build_terminal_branch_row(
                    primary_category,
                    len(category_df),
                    total_count,
                )
            )
            continue

        rows.extend(
            build_conditional_rows(
                category_df=category_df,
                primary_category=primary_category,
                total_count=total_count,
                latest_primary_category=latest_primary_category,
                recent_sub_categories=recent_sub_categories,
                cooldown_config=cooldown_config,
            )
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return empty_conditional_table()

    result_df = apply_seasonal_prediction_rules(result_df, predicted_next_month)
    if result_df.empty:
        return empty_conditional_table()

    result_df = normalize_within_primary_category(result_df)
    result_df = apply_parent_adjusted_prediction(
        result_df,
        parent_primary_prediction_df,
    )
    return round_conditional_table(result_df)


def prepare_conditional_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "month_label",
        "primary_category",
        "primary_sub_category",
        "confidence",
    ]
    if df.empty or any(column not in df.columns for column in required_columns):
        return pd.DataFrame()

    conditional_df = df.copy()
    conditional_df["confidence"] = conditional_df["confidence"].str.lower()
    conditional_df["primary_category"] = conditional_df["primary_category"].str.lower()
    conditional_df["primary_sub_category"] = conditional_df[
        "primary_sub_category"
    ].str.lower()

    conditional_df = conditional_df[conditional_df["confidence"] != "low"]
    conditional_df = conditional_df[conditional_df["primary_category"] != ""]
    conditional_df = conditional_df[conditional_df["primary_sub_category"] != ""]

    month_dates = pd.to_datetime(
        conditional_df["month_label"],
        format="%Y-%m",
        errors="coerce",
    )
    conditional_df["month_period"] = month_dates.dt.to_period("M")
    conditional_df = conditional_df.dropna(subset=["month_period"])

    return conditional_df.sort_values("month_period").reset_index(drop=True)


def normalize_cooldown_config(
    cooldown_config: dict[str, float] | None = None,
) -> dict[str, float]:
    source_config = cooldown_config or DEFAULT_COOLDOWN_CONFIG
    normalized_config = {}

    for key, default_value in DEFAULT_COOLDOWN_CONFIG.items():
        try:
            normalized_config[key] = float(source_config.get(key, default_value))
        except (TypeError, ValueError):
            normalized_config[key] = float(default_value)

    return normalized_config


def build_terminal_branch_row(
    primary_category: str,
    count: int,
    total_count: int,
) -> dict:
    return {
        "primary_category": primary_category,
        "primary_sub_category": primary_category,
        "count": count,
        "base_probability_percent": calculate_percent(count, total_count),
        "conditional_base_percent": 100.0,
        "conditional_sub_probability_percent": 100,
        "cooldown_multiplier": 1.0,
        "all_role_role_cooldown_multiplier": 1.0,
        "final_cooldown_multiplier": 1.0,
        "adjusted_score": 100.0,
        "sub_raw_score": 100.0,
    }


def build_conditional_rows(
    category_df: pd.DataFrame,
    primary_category: str,
    total_count: int,
    latest_primary_category: str,
    recent_sub_categories: list[set[str]],
    cooldown_config: dict[str, float],
) -> list[dict]:
    category_total = len(category_df)
    counts = (
        category_df["primary_sub_category"]
        .value_counts()
        .rename_axis("primary_sub_category")
        .reset_index(name="count")
    )

    rows = []
    for row in counts.itertuples(index=False):
        sub_category = row.primary_sub_category
        base_probability_percent = calculate_percent(row.count, total_count)
        conditional_base_percent = calculate_percent(row.count, category_total)

        cooldown_multiplier = get_cooldown_multiplier(
            primary_category,
            sub_category,
            recent_sub_categories,
            cooldown_config,
        )
        all_role_role_cooldown_multiplier = get_all_role_role_cooldown_multiplier(
            latest_primary_category,
            primary_category,
            sub_category,
        )
        final_cooldown_multiplier = (
            cooldown_multiplier * all_role_role_cooldown_multiplier
        )
        adjusted_score = conditional_base_percent * final_cooldown_multiplier

        rows.append(
            {
                "primary_category": primary_category,
                "primary_sub_category": sub_category,
                "count": row.count,
                "base_probability_percent": base_probability_percent,
                "conditional_base_percent": conditional_base_percent,
                "conditional_sub_probability_percent": conditional_base_percent,
                "cooldown_multiplier": cooldown_multiplier,
                "all_role_role_cooldown_multiplier": (
                    all_role_role_cooldown_multiplier
                ),
                "final_cooldown_multiplier": final_cooldown_multiplier,
                "adjusted_score": adjusted_score,
                "sub_raw_score": adjusted_score,
            }
        )

    return rows


def calculate_percent(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator * 100


def apply_seasonal_prediction_rules(
    prediction_df: pd.DataFrame,
    predicted_next_month: str | None,
) -> pd.DataFrame:
    if prediction_df.empty or "primary_sub_category" not in prediction_df.columns:
        return prediction_df

    allowed_mask = prediction_df["primary_sub_category"].apply(
        lambda sub_category: is_seasonal_sub_category_allowed(
            sub_category,
            predicted_next_month,
        )
    )
    return prediction_df[allowed_mask].copy()


def get_recent_sub_categories(
    df: pd.DataFrame,
    recent_months: int,
) -> list[set[str]]:
    recent_periods = (
        df["month_period"].drop_duplicates().sort_values(ascending=False).head(recent_months)
    )

    recent_sets = []
    for period in recent_periods:
        month_df = df[df["month_period"] == period]
        recent_sets.append(set(month_df["primary_sub_category"].to_list()))

    return recent_sets


def get_cooldown_multiplier(
    primary_category: str,
    sub_category: str,
    recent_sub_categories: list[set[str]],
    cooldown_config: dict[str, float],
) -> float:
    if primary_category != "role" or sub_category not in ROLE_SUB_CATEGORIES:
        return 1.0

    months_since_seen = get_months_since_seen(sub_category, recent_sub_categories)
    return get_role_cooldown_multiplier(months_since_seen, cooldown_config)


def get_months_since_seen(
    sub_category: str,
    recent_sub_categories: list[set[str]],
) -> int | None:
    for months_since_seen, sub_categories in enumerate(
        recent_sub_categories[:3],
        start=1,
    ):
        if sub_category in sub_categories:
            return months_since_seen

    return None


def get_role_cooldown_multiplier(
    months_since_seen: int | None,
    cooldown_config: dict[str, float] | None,
) -> float:
    cooldown_config = normalize_cooldown_config(cooldown_config)

    if months_since_seen == 1:
        return cooldown_config["recent_1"]
    if months_since_seen == 2:
        return cooldown_config["recent_2"]
    if months_since_seen == 3:
        return cooldown_config["recent_3"]
    return 1.0


def get_all_role_role_cooldown_multiplier(
    latest_primary_category: str,
    primary_category: str,
    sub_category: str,
) -> float:
    if (
        latest_primary_category == "all_role"
        and primary_category == "role"
        and sub_category in ROLE_SUB_CATEGORIES
    ):
        return ALL_ROLE_ROLE_COOLDOWN_MULTIPLIER

    return 1.0


def normalize_within_primary_category(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    score_sum = normalized_df.groupby("primary_category")["adjusted_score"].transform(
        "sum"
    )
    count_sum = normalized_df.groupby("primary_category")["count"].transform("sum")

    normalized_df["normalized_sub_category_weight"] = 0.0
    has_score = score_sum > 0
    normalized_df.loc[has_score, "normalized_sub_category_weight"] = (
        normalized_df.loc[has_score, "adjusted_score"] / score_sum[has_score]
    )

    has_no_score = ~has_score
    has_count = has_no_score & (count_sum > 0)
    normalized_df.loc[has_count, "normalized_sub_category_weight"] = (
        normalized_df.loc[has_count, "count"] / count_sum[has_count]
    )

    normalized_df["adjusted_conditional_sub_probability_percent"] = 0.0
    normalized_df["adjusted_conditional_sub_probability_percent"] = (
        normalized_df["normalized_sub_category_weight"] * 100
    )

    return normalized_df.sort_values(
        ["primary_category", "adjusted_conditional_sub_probability_percent"],
        ascending=[True, False],
    )


def apply_parent_adjusted_prediction(
    conditional_df: pd.DataFrame,
    parent_primary_prediction_df: pd.DataFrame | None,
) -> pd.DataFrame:
    result_df = conditional_df.copy()
    parent_column = "parent_primary_adjusted_prediction_percent"
    result_df[parent_column] = 0.0
    result_df["adjusted_prediction_percent"] = 0.0

    required_columns = ["primary_category", "adjusted_prediction_percent"]
    if (
        parent_primary_prediction_df is None
        or parent_primary_prediction_df.empty
        or any(column not in parent_primary_prediction_df.columns for column in required_columns)
    ):
        return result_df

    parent_df = parent_primary_prediction_df[required_columns].rename(
        columns={"adjusted_prediction_percent": parent_column}
    )
    result_df = result_df.merge(parent_df, on="primary_category", how="left")
    merged_parent_column = f"{parent_column}_y"
    original_parent_column = f"{parent_column}_x"

    if merged_parent_column in result_df.columns:
        result_df[parent_column] = result_df[merged_parent_column].fillna(0.0)
        result_df = result_df.drop(
            columns=[original_parent_column, merged_parent_column],
            errors="ignore",
        )

    result_df[parent_column] = result_df[parent_column].astype(float)
    result_df["adjusted_prediction_percent"] = (
        result_df[parent_column] * result_df["normalized_sub_category_weight"]
    )

    return result_df.sort_values("adjusted_prediction_percent", ascending=False)


def round_conditional_table(df: pd.DataFrame) -> pd.DataFrame:
    rounded_df = df.copy()
    numeric_columns = [
        "base_probability_percent",
        "conditional_base_percent",
        "conditional_sub_probability_percent",
        "cooldown_multiplier",
        "all_role_role_cooldown_multiplier",
        "final_cooldown_multiplier",
        "adjusted_score",
        "sub_raw_score",
        "normalized_sub_category_weight",
        "adjusted_conditional_sub_probability_percent",
        "parent_primary_adjusted_prediction_percent",
        "adjusted_prediction_percent",
    ]
    rounded_df[numeric_columns] = rounded_df[numeric_columns].round(4)
    return rounded_df[CONDITIONAL_COLUMNS]


def empty_conditional_table() -> pd.DataFrame:
    return pd.DataFrame(columns=CONDITIONAL_COLUMNS)
