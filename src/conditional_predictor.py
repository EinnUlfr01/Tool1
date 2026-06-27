import pandas as pd

from src.data_loader import (
    ROLE_COMPONENTS,
    VALID_PRIMARY_CATEGORIES,
    expand_benefit_components,
)
from src.seasonal_rules import get_component_eligible_months
from src.seasonal_rules import get_seasonal_prediction_multiplier
from src.seasonal_rules import is_component_prediction_eligible


AFTER_ALL_ROLE_ROLE_COMPONENT_PENALTY = 0.60
ROLE_HISTORICAL_WEIGHT_POWER = 0.5
ROLE_RECENCY_MULTIPLIERS = {
    "recent_2_or_less": 0.35,
    "recent_3": 0.50,
    "recent_4": 0.70,
    "recent_5_to_6": 0.90,
    "older_7_plus": 1.15,
}
NON_ROLE_HISTORICAL_WEIGHT_POWER = 0.65
NON_ROLE_RECENCY_MULTIPLIERS = {
    "recent_2_or_less": 0.50,
    "recent_3": 0.65,
    "recent_4": 0.80,
    "recent_5_to_6": 1.00,
    "older_7_plus": 1.10,
    "unknown": 1.15,
}

DEFAULT_COOLDOWN_CONFIG = {
    "recent_1": 0.55,
    "recent_2": 0.75,
    "recent_3": 0.90,
}
RULE_DISPLAY_EPSILON = 0.0001

CONDITIONAL_COLUMNS = [
    "primary_category",
    "component_type",
    "primary_sub_category",
    "count",
    "component_weight",
    "historical_weight_adjusted",
    "base_probability_percent",
    "conditional_base_percent",
    "conditional_sub_probability_percent",
    "last_seen_month",
    "months_since_last_seen",
    "cooldown_multiplier",
    "all_role_role_cooldown_multiplier",
    "seasonal_multiplier",
    "final_cooldown_multiplier",
    "adjusted_score",
    "sub_raw_score",
    "applied_rules",
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

    component_df = filter_prediction_eligible_components(
        component_df,
        predicted_next_month,
    )
    if component_df.empty:
        return empty_conditional_table()

    latest_is_all_role = bool(source_df.iloc[-1]["is_all_role"])
    recent_sub_categories = get_recent_sub_categories(component_df, recent_months=3)
    role_last_seen_months = get_role_last_seen_months(
        component_df,
        ignore_all_role_rows=True,
    )
    non_role_last_seen_months = get_non_role_last_seen_months(component_df)
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
                role_last_seen_months=role_last_seen_months,
                non_role_last_seen_months=non_role_last_seen_months,
                cooldown_config=cooldown_config,
                predicted_next_month=predicted_next_month,
            )
        )

    result_df = pd.DataFrame(rows)
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
        "is_seasonal",
        "seasonal_type",
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
        source_df["primary_category"].isin(VALID_PRIMARY_CATEGORIES)
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


def filter_prediction_eligible_components(
    component_df: pd.DataFrame,
    predicted_next_month: str | None,
) -> pd.DataFrame:
    if component_df.empty:
        return component_df
    eligible_mask = component_df["component"].apply(
        lambda component: is_component_prediction_eligible(
            component,
            predicted_next_month,
        )
    )
    return component_df[eligible_mask].copy()


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
    role_last_seen_months: dict[str, pd.Period],
    non_role_last_seen_months: dict[str, pd.Period],
    cooldown_config: dict[str, float],
    predicted_next_month: str | None,
) -> list[dict]:
    adjusted_weight_by_component = {}
    for (component, component_type), component_df in category_df.groupby(
        ["component", "component_type"],
        sort=True,
    ):
        component_weight = float(component_df["component_weight"].sum())
        adjusted_weight_by_component[(component, component_type)] = (
            get_historical_weight_adjusted(component, component_type, component_weight)
        )
    category_total_adjusted_weight = float(
        sum(adjusted_weight_by_component.values())
    )
    rows = []
    target_period = parse_month_period(predicted_next_month)

    for (component, component_type), component_df in category_df.groupby(
        ["component", "component_type"],
        sort=True,
    ):
        component_weight = float(component_df["component_weight"].sum())
        historical_weight_adjusted = adjusted_weight_by_component[
            (component, component_type)
        ]
        last_seen_period = get_component_last_seen_period(
            component,
            component_type,
            role_last_seen_months,
            non_role_last_seen_months,
        )
        months_since_last_seen = calculate_month_delta(
            last_seen_period,
            target_period,
        )
        cooldown_multiplier = get_cooldown_multiplier(
            component,
            component_type,
            recent_sub_categories,
            cooldown_config,
            months_since_last_seen=months_since_last_seen,
        )
        all_role_multiplier = get_all_role_role_cooldown_multiplier(
            latest_is_all_role,
            component_type,
        )
        seasonal_multiplier = calculate_weighted_seasonal_multiplier(
            component_df,
            predicted_next_month,
        )
        final_multiplier = (
            cooldown_multiplier * all_role_multiplier * seasonal_multiplier
        )
        conditional_base_percent = calculate_percent(
            historical_weight_adjusted,
            category_total_adjusted_weight,
        )
        adjusted_score = conditional_base_percent * final_multiplier

        rows.append(
            {
                "primary_category": primary_category,
                "component_type": component_type,
                "primary_sub_category": component,
                "count": int(len(component_df)),
                "component_weight": component_weight,
                "historical_weight_adjusted": historical_weight_adjusted,
                "base_probability_percent": calculate_percent(
                    component_weight,
                    total_component_weight,
                ),
                "conditional_base_percent": conditional_base_percent,
                "conditional_sub_probability_percent": conditional_base_percent,
                "last_seen_month": (
                    str(last_seen_period) if last_seen_period is not None else ""
                ),
                "months_since_last_seen": months_since_last_seen,
                "cooldown_multiplier": cooldown_multiplier,
                "all_role_role_cooldown_multiplier": all_role_multiplier,
                "seasonal_multiplier": seasonal_multiplier,
                "final_cooldown_multiplier": final_multiplier,
                "adjusted_score": adjusted_score,
                "sub_raw_score": adjusted_score,
                "applied_rules": build_applied_rules(
                    component_type,
                    cooldown_multiplier,
                    all_role_multiplier,
                    seasonal_multiplier,
                ),
            }
        )
    return rows


def calculate_weighted_seasonal_multiplier(
    component_df: pd.DataFrame,
    predicted_next_month: str | None,
) -> float:
    total_weight = float(component_df["component_weight"].sum())
    if total_weight <= 0:
        return 1.0

    weighted_multiplier = 0.0
    for row in component_df.itertuples(index=False):
        if get_component_eligible_months(row.component) is None:
            multiplier = 1.0
            weighted_multiplier += float(row.component_weight) * multiplier
            continue
        multiplier = get_seasonal_prediction_multiplier(
            component=row.component,
            is_seasonal=row.is_seasonal,
            seasonal_type=row.seasonal_type,
            predicted_next_month=predicted_next_month,
        )
        weighted_multiplier += float(row.component_weight) * multiplier
    return weighted_multiplier / total_weight


def get_historical_weight_adjusted(
    component: str,
    component_type: str,
    component_weight: float,
) -> float:
    if component_type == "role" and component in ROLE_COMPONENTS:
        return component_weight ** ROLE_HISTORICAL_WEIGHT_POWER
    if component_type == "non_role":
        return component_weight ** NON_ROLE_HISTORICAL_WEIGHT_POWER
    return component_weight


def calculate_percent(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator * 100


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


def get_role_last_seen_months(
    component_df: pd.DataFrame,
    ignore_all_role_rows: bool = True,
) -> dict[str, pd.Period]:
    role_df = component_df[
        component_df["component_type"].eq("role")
        & component_df["component"].isin(ROLE_COMPONENTS)
    ].copy()
    if ignore_all_role_rows:
        role_df = role_df[~role_df["is_all_role"].eq(True)]
    if role_df.empty:
        return {}

    return {
        str(component): period
        for component, period in role_df.groupby("component")[
            "month_period"
        ].max().items()
    }


def get_non_role_last_seen_months(
    component_df: pd.DataFrame,
) -> dict[str, pd.Period]:
    non_role_df = component_df[component_df["component_type"].eq("non_role")]
    if non_role_df.empty:
        return {}

    return {
        str(component): period
        for component, period in non_role_df.groupby("component")[
            "month_period"
        ].max().items()
    }


def parse_month_period(month_label: str | None) -> pd.Period | None:
    if not month_label:
        return None
    month_date = pd.to_datetime(
        month_label,
        format="%Y-%m",
        errors="coerce",
    )
    if pd.isna(month_date):
        return None
    return month_date.to_period("M")


def get_component_last_seen_period(
    component: str,
    component_type: str,
    role_last_seen_months: dict[str, pd.Period],
    non_role_last_seen_months: dict[str, pd.Period],
) -> pd.Period | None:
    if component_type == "role" and component in ROLE_COMPONENTS:
        return role_last_seen_months.get(component)
    if component_type == "non_role":
        return non_role_last_seen_months.get(component)
    return None


def calculate_month_delta(
    last_seen_period: pd.Period | None,
    target_period: pd.Period | None,
) -> int | None:
    if last_seen_period is None or target_period is None:
        return None
    return (target_period.year - last_seen_period.year) * 12 + (
        target_period.month - last_seen_period.month
    )


def get_cooldown_multiplier(
    sub_category: str,
    component_type: str,
    recent_sub_categories: list[set[str]],
    cooldown_config: dict[str, float],
    months_since_last_seen: int | None = None,
) -> float:
    if component_type == "role" and sub_category in ROLE_COMPONENTS:
        return get_role_recency_multiplier(months_since_last_seen)
    if component_type == "non_role":
        return get_non_role_recency_multiplier(months_since_last_seen)
    return 1.0


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


def get_role_recency_multiplier(months_since_last_seen: int | None) -> float:
    if months_since_last_seen is None:
        return 1.0
    if months_since_last_seen <= 2:
        return ROLE_RECENCY_MULTIPLIERS["recent_2_or_less"]
    if months_since_last_seen == 3:
        return ROLE_RECENCY_MULTIPLIERS["recent_3"]
    if months_since_last_seen == 4:
        return ROLE_RECENCY_MULTIPLIERS["recent_4"]
    if months_since_last_seen in {5, 6}:
        return ROLE_RECENCY_MULTIPLIERS["recent_5_to_6"]
    return ROLE_RECENCY_MULTIPLIERS["older_7_plus"]


def get_non_role_recency_multiplier(months_since_last_seen: int | None) -> float:
    if months_since_last_seen is None:
        return NON_ROLE_RECENCY_MULTIPLIERS["unknown"]
    if months_since_last_seen <= 2:
        return NON_ROLE_RECENCY_MULTIPLIERS["recent_2_or_less"]
    if months_since_last_seen == 3:
        return NON_ROLE_RECENCY_MULTIPLIERS["recent_3"]
    if months_since_last_seen == 4:
        return NON_ROLE_RECENCY_MULTIPLIERS["recent_4"]
    if months_since_last_seen in {5, 6}:
        return NON_ROLE_RECENCY_MULTIPLIERS["recent_5_to_6"]
    return NON_ROLE_RECENCY_MULTIPLIERS["older_7_plus"]


def get_all_role_role_cooldown_multiplier(
    latest_is_all_role: bool,
    component_type: str,
) -> float:
    if latest_is_all_role and component_type == "role":
        return AFTER_ALL_ROLE_ROLE_COMPONENT_PENALTY
    return 1.0


def build_applied_rules(
    component_type: str,
    cooldown_multiplier: float,
    all_role_multiplier: float,
    seasonal_multiplier: float,
) -> str:
    rules = []
    if component_type == "role" and cooldown_multiplier < 1.0:
        rules.append(f"role_cooldown={cooldown_multiplier:.2f}")
    elif component_type == "non_role" and cooldown_multiplier < 1.0:
        rules.append(f"non_role_cooldown={cooldown_multiplier:.2f}")
    elif component_type == "non_role" and cooldown_multiplier > 1.0:
        rules.append(f"non_role_long_gap={cooldown_multiplier:.2f}")
    if all_role_multiplier < 1.0:
        rules.append(f"after_all_role_component={all_role_multiplier:.2f}")
    if seasonal_multiplier > 1.0 + RULE_DISPLAY_EPSILON:
        rules.append(f"in_season={seasonal_multiplier:.2f}")
    elif seasonal_multiplier < 1.0 - RULE_DISPLAY_EPSILON:
        rules.append(f"out_of_season={seasonal_multiplier:.2f}")
    return "|".join(rules) if rules else "none"


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
        "historical_weight_adjusted",
        "base_probability_percent",
        "conditional_base_percent",
        "conditional_sub_probability_percent",
        "months_since_last_seen",
        "cooldown_multiplier",
        "all_role_role_cooldown_multiplier",
        "seasonal_multiplier",
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
