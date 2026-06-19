import pandas as pd

from src.data_loader import ROLE_COMPONENTS, expand_benefit_components
from src.seasonal_rules import is_seasonal_sub_category_allowed


ALL_ROLE_ROLE_COOLDOWN_MULTIPLIER = 0.75

DEFAULT_COOLDOWN_CONFIG = {
    "recent_1": 0.2,
    "recent_2": 0.4,
    "recent_3": 0.6,
}

CONDITIONAL_COLUMNS = [
    "primary_category",
    "component_type",
    "primary_sub_category",
    "count",
    "component_weight",
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
    """Predict expanded components and constrain them to their parent category."""
    cooldown_config = normalize_cooldown_config(cooldown_config)
    source_df, component_df = prepare_conditional_data(df)
    if source_df.empty or component_df.empty:
        return empty_conditional_table()

    latest_is_all_role = bool(source_df.iloc[-1]["is_all_role"])
    recent_sub_categories = get_recent_sub_categories(component_df, recent_months=3)
    total_component_weight = float(component_df["component_weight"].sum())
    rows = []

    for primary_category, category_df in component_df.groupby(
        "primary_category",
        sort=True,
    ):
        rows.extend(
            build_conditional_rows(
                category_df=category_df,
                primary_category=primary_category,
                total_component_weight=total_component_weight,
                latest_is_all_role=latest_is_all_role,
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


def prepare_conditional_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        return pd.DataFrame(), pd.DataFrame()

    source_df = df.copy()
    source_df["month_period"] = pd.to_datetime(
        source_df["month_label"],
        format="%Y-%m",
        errors="coerce",
    ).dt.to_period("M")
    source_df = source_df.dropna(subset=["month_period"])
    source_df = source_df[
        source_df["primary_category"].isin({"role", "non_role", "both"})
    ].sort_values("month_period")
    component_df = expand_benefit_components(source_df)
    if component_df.empty:
        return source_df, component_df

    component_df["month_period"] = pd.to_datetime(
        component_df["month_label"],
        format="%Y-%m",
        errors="coerce",
    ).dt.to_period("M")
    component_df = component_df.dropna(subset=["month_period"])
    return source_df.reset_index(drop=True), component_df.reset_index(drop=True)


def normalize_cooldown_config(
    cooldown_config: dict[str, float] | None = None,
) -> dict[str, float]:
    source_config = cooldown_config or DEFAULT_COOLDOWN_CONFIG
    normalized_config = {}
    for key, default_value in DEFAULT_COOLDOWN_CONFIG.items():
        try:
            normalized_config[key] = float(source_config.get(key, default_value))
        except (AttributeError, TypeError, ValueError):
            normalized_config[key] = float(default_value)
    return normalized_config


def build_conditional_rows(
    category_df: pd.DataFrame,
    primary_category: str,
    total_component_weight: float,
    latest_is_all_role: bool,
    recent_sub_categories: list[set[str]],
    cooldown_config: dict[str, float],
) -> list[dict]:
    category_total_weight = float(category_df["component_weight"].sum())
    grouped = (
        category_df.groupby(["component", "component_type"], as_index=False)
        .agg(
            count=("component", "size"),
            component_weight=("component_weight", "sum"),
        )
    )

    rows = []
    for row in grouped.itertuples(index=False):
        cooldown_multiplier = get_cooldown_multiplier(
            row.component,
            recent_sub_categories,
            cooldown_config,
        )
        all_role_multiplier = get_all_role_role_cooldown_multiplier(
            latest_is_all_role,
            row.component_type,
        )
        final_cooldown_multiplier = cooldown_multiplier * all_role_multiplier
        conditional_base_percent = calculate_percent(
            row.component_weight,
            category_total_weight,
        )
        adjusted_score = conditional_base_percent * final_cooldown_multiplier

        rows.append(
            {
                "primary_category": primary_category,
                "component_type": row.component_type,
                "primary_sub_category": row.component,
                "count": int(row.count),
                "component_weight": float(row.component_weight),
                "base_probability_percent": calculate_percent(
                    row.component_weight,
                    total_component_weight,
                ),
                "conditional_base_percent": conditional_base_percent,
                "conditional_sub_probability_percent": conditional_base_percent,
                "cooldown_multiplier": cooldown_multiplier,
                "all_role_role_cooldown_multiplier": all_role_multiplier,
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
    if prediction_df.empty:
        return prediction_df
    allowed_mask = prediction_df["primary_sub_category"].apply(
        lambda sub_category: is_seasonal_sub_category_allowed(
            sub_category,
            predicted_next_month,
        )
    )
    return prediction_df[allowed_mask].copy()


def get_recent_sub_categories(
    component_df: pd.DataFrame,
    recent_months: int,
) -> list[set[str]]:
    recent_periods = (
        component_df["month_period"]
        .drop_duplicates()
        .sort_values(ascending=False)
        .head(recent_months)
    )
    return [
        set(
            component_df.loc[
                component_df["month_period"] == period,
                "component",
            ].tolist()
        )
        for period in recent_periods
    ]


def get_cooldown_multiplier(
    sub_category: str,
    recent_sub_categories: list[set[str]],
    cooldown_config: dict[str, float],
) -> float:
    if sub_category not in ROLE_COMPONENTS:
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
    latest_is_all_role: bool,
    component_type: str,
) -> float:
    if latest_is_all_role and component_type == "role":
        return ALL_ROLE_ROLE_COOLDOWN_MULTIPLIER
    return 1.0


def normalize_within_primary_category(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    score_sum = normalized_df.groupby("primary_category")["adjusted_score"].transform(
        "sum"
    )
    weight_sum = normalized_df.groupby("primary_category")[
        "component_weight"
    ].transform("sum")
    normalized_df["normalized_sub_category_weight"] = 0.0

    has_score = score_sum > 0
    normalized_df.loc[has_score, "normalized_sub_category_weight"] = (
        normalized_df.loc[has_score, "adjusted_score"] / score_sum[has_score]
    )
    has_weight = (~has_score) & (weight_sum > 0)
    normalized_df.loc[has_weight, "normalized_sub_category_weight"] = (
        normalized_df.loc[has_weight, "component_weight"] / weight_sum[has_weight]
    )
    normalized_df["adjusted_conditional_sub_probability_percent"] = (
        normalized_df["normalized_sub_category_weight"] * 100
    )
    return normalized_df


def apply_parent_adjusted_prediction(
    conditional_df: pd.DataFrame,
    parent_primary_prediction_df: pd.DataFrame | None,
) -> pd.DataFrame:
    result_df = conditional_df.copy()
    parent_column = "parent_primary_adjusted_prediction_percent"
    result_df[parent_column] = 0.0

    required_columns = ["primary_category", "adjusted_prediction_percent"]
    if (
        parent_primary_prediction_df is not None
        and not parent_primary_prediction_df.empty
        and all(
            column in parent_primary_prediction_df.columns
            for column in required_columns
        )
    ):
        parent_map = parent_primary_prediction_df.set_index("primary_category")[
            "adjusted_prediction_percent"
        ]
        result_df[parent_column] = (
            result_df["primary_category"].map(parent_map).fillna(0.0).astype(float)
        )

    result_df["adjusted_prediction_percent"] = (
        result_df[parent_column] * result_df["normalized_sub_category_weight"]
    )
    return result_df.sort_values("adjusted_prediction_percent", ascending=False)


def round_conditional_table(df: pd.DataFrame) -> pd.DataFrame:
    rounded_df = df.copy()
    numeric_columns = [
        "component_weight",
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
