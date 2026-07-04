from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.candidate_history import build_candidate_history, normalize_month_period
from src.candidate_pool import (
    HARD_SEASONAL_CANDIDATES,
    build_candidate_pool,
    normalize_candidate_key,
)
from src.data_loader import parse_components
from src.seasonal_rules import SEASONAL_MONTH_BOOST, parse_predicted_month
from src.taxonomy import VALID_NON_ROLE_COMPONENTS, VALID_ROLE_COMPONENTS


PRIOR_STRENGTH_OPTIONS = [3, 4, 5]

SCORING_COLUMNS = [
    "candidate",
    "candidate_group",
    "display_label",
    "valid_for_target_month",
    "base_score",
    "frequency_factor_raw",
    "frequency_factor_clamped",
    "frequency_factor",
    "recent_role_pressure",
    "recent_non_role_pressure",
    "recency_factor",
    "same_candidate_months_since",
    "same_candidate_cooldown_factor",
    "individual_non_role_recency_factor",
    "overdue_factor",
    "seasonal_factor",
    "all_role_candidate_factor",
    "non_role_context_factor",
    "other_non_role_context_factor",
    "other_non_role_cap_factor",
    "rare_bucket_factor",
    "smoothing_factor",
    "raw_score",
    "final_probability",
    "final_probability_percent",
    "rank",
    "last_direct_seen",
    "last_mixed_seen",
    "last_all_role_seen",
    "months_since_direct",
    "months_since_mixed",
    "months_since_all_role",
    "direct_count",
    "mixed_count",
    "all_role_inclusion_count",
    "direct_recency_factor",
    "mixed_recency_factor",
    "all_role_inclusion_recency_factor",
    "combined_recency_factor_before_clamp",
    "combined_recency_factor",
    "effective_count",
    "prior_strength",
    "excluded_reason",
]


@dataclass(frozen=True)
class CandidateScoringConfig:
    direct_weight: float = 1.00
    mixed_weight: float = 0.50
    all_role_inclusion_weight: float = 0.25
    minor_weight: float = 0.10
    none_weight: float = 0.00

    direct_recent_1: float = 0.55
    direct_recent_2: float = 0.75
    direct_recent_3: float = 0.90

    all_role_recent_lt_10: float = 0.10
    all_role_recent_10_11: float = 0.50
    all_role_recent_12_plus: float = 1.00

    role_overdue_5_plus: float = 1.10
    role_overdue_8_plus: float = 1.20
    role_overdue_never: float = 1.20

    non_role_overdue_5_plus: float = 1.03
    non_role_overdue_8_plus: float = 1.05
    non_role_overdue_never: float = 1.05

    non_role_context_role_streak_2: float = 1.10
    non_role_context_role_streak_3: float = 1.20
    non_role_context_previous_all_role: float = 1.05

    frequency_factor_min: float = 0.90
    frequency_factor_max: float = 1.10
    non_role_context_factor_min: float = 0.85
    non_role_context_factor_max: float = 1.20
    other_non_role_context_factor_max: float = 1.05
    other_non_role_rare_bucket_factor: float = 0.85
    individual_non_role_recency_factor_max: float = 1.15

    min_combined_recency_factor: float = 0.45
    seasonal_month_boost: float = SEASONAL_MONTH_BOOST


def score_candidates(
    df: pd.DataFrame,
    target_month: str | None,
    prior_strength: float = 3,
    config: CandidateScoringConfig | None = None,
) -> pd.DataFrame:
    config = config or CandidateScoringConfig()
    prior_strength = normalize_prior_strength(prior_strength)
    pool_df = build_candidate_pool(target_month)
    valid_pool_df = pool_df[pool_df["valid_for_target_month"].eq(True)].copy()
    if valid_pool_df.empty:
        return pd.DataFrame(columns=SCORING_COLUMNS)

    history_df = build_canonical_history(df, target_month)
    context = build_recent_context(df, target_month)
    scoring_df = valid_pool_df.merge(
        history_df,
        left_on="canonical_candidate",
        right_on="candidate",
        how="left",
        suffixes=("", "_history"),
    )
    scoring_df["candidate"] = scoring_df["canonical_candidate"]
    fill_history_defaults(scoring_df)

    scoring_df["base_score"] = 1.0
    scoring_df["effective_count"] = (
        scoring_df["direct_count"].astype(float) * config.direct_weight
        + scoring_df["mixed_count"].astype(float) * config.mixed_weight
        + scoring_df["all_role_inclusion_count"].astype(float)
        * config.all_role_inclusion_weight
    )
    mean_effective_count = float(scoring_df["effective_count"].mean())
    denominator = mean_effective_count + float(prior_strength)
    if denominator <= 0:
        scoring_df["frequency_factor_raw"] = 1.0
    else:
        scoring_df["frequency_factor_raw"] = 1 + (
            (scoring_df["effective_count"] - mean_effective_count) / denominator
        ) * 0.10
    scoring_df["frequency_factor_clamped"] = scoring_df[
        "frequency_factor_raw"
    ].clip(config.frequency_factor_min, config.frequency_factor_max)
    scoring_df["frequency_factor"] = scoring_df["frequency_factor_clamped"]

    scoring_df["recent_role_pressure"] = float(context.get("recent_role_pressure", 0.0))
    scoring_df["recent_non_role_pressure"] = float(
        context.get("recent_non_role_pressure", 0.0)
    )

    scoring_df["direct_recency_factor"] = scoring_df["months_since_direct"].apply(
        lambda value: get_direct_recency_factor(value, config)
    )
    scoring_df["mixed_recency_factor"] = scoring_df["months_since_mixed"].apply(
        lambda value: get_soft_recency_factor(value, config.mixed_weight, config)
    )
    scoring_df["all_role_inclusion_recency_factor"] = scoring_df[
        "months_since_all_role"
    ].apply(
        lambda value: get_soft_recency_factor(
            value,
            config.all_role_inclusion_weight,
            config,
        )
    )
    scoring_df["combined_recency_factor_before_clamp"] = (
        scoring_df["direct_recency_factor"]
        * scoring_df["mixed_recency_factor"]
        * scoring_df["all_role_inclusion_recency_factor"]
    )
    scoring_df["combined_recency_factor"] = scoring_df[
        "combined_recency_factor_before_clamp"
    ].clip(lower=config.min_combined_recency_factor)
    scoring_df["same_candidate_months_since"] = scoring_df.apply(
        get_same_candidate_months_since,
        axis=1,
    )
    scoring_df["same_candidate_cooldown_factor"] = scoring_df[
        "same_candidate_months_since"
    ].apply(get_same_candidate_cooldown_factor)
    scoring_df["individual_non_role_recency_factor"] = scoring_df.apply(
        lambda row: get_individual_non_role_recency_factor(
            row["candidate"],
            row["candidate_group"],
            row["same_candidate_months_since"],
            config,
        ),
        axis=1,
    )
    scoring_df["recency_factor"] = (
        scoring_df["same_candidate_cooldown_factor"]
        * scoring_df["individual_non_role_recency_factor"]
    )

    scoring_df["overdue_factor"] = scoring_df.apply(
        lambda row: get_overdue_factor(
            row["candidate_group"],
            row["months_since_direct"],
            config,
        ),
        axis=1,
    )
    scoring_df["seasonal_factor"] = scoring_df["candidate"].apply(
        lambda candidate: get_seasonal_factor(candidate, target_month, config)
    )
    scoring_df["all_role_candidate_factor"] = scoring_df.apply(
        lambda row: get_all_role_candidate_factor(
            row["candidate"],
            row["months_since_direct"],
            config,
        ),
        axis=1,
    )
    scoring_df["non_role_context_factor"] = scoring_df["candidate_group"].apply(
        lambda candidate_group: get_non_role_context_factor(
            candidate_group,
            context,
            config,
        )
    )
    scoring_df["other_non_role_context_factor"] = scoring_df.apply(
        lambda row: get_other_non_role_context_factor(
            row["candidate"],
            row["non_role_context_factor"],
            config,
        ),
        axis=1,
    )
    scoring_df["other_non_role_cap_factor"] = scoring_df.apply(
        lambda row: row["other_non_role_context_factor"]
        / row["non_role_context_factor"]
        if row["candidate"] == "other_non_role" and row["non_role_context_factor"]
        else 1.0,
        axis=1,
    )
    scoring_df["non_role_context_factor"] = scoring_df.apply(
        lambda row: row["other_non_role_context_factor"]
        if row["candidate"] == "other_non_role"
        else row["non_role_context_factor"],
        axis=1,
    )
    scoring_df["rare_bucket_factor"] = scoring_df["candidate"].apply(
        lambda candidate: config.other_non_role_rare_bucket_factor
        if candidate == "other_non_role"
        else 1.0
    )
    scoring_df["smoothing_factor"] = 1.0
    scoring_df["prior_strength"] = prior_strength
    scoring_df["raw_score"] = (
        scoring_df["base_score"]
        * scoring_df["frequency_factor"]
        * scoring_df["recency_factor"]
        * scoring_df["overdue_factor"]
        * scoring_df["seasonal_factor"]
        * scoring_df["all_role_candidate_factor"]
        * scoring_df["non_role_context_factor"]
        * scoring_df["rare_bucket_factor"]
        * scoring_df["smoothing_factor"]
    )
    scoring_df = normalize_scores(scoring_df)
    scoring_df = scoring_df.sort_values(
        ["final_probability", "candidate"],
        ascending=[False, True],
    ).reset_index(drop=True)
    scoring_df["rank"] = range(1, len(scoring_df) + 1)
    return round_scoring_table(scoring_df)


def build_canonical_history(df: pd.DataFrame, target_month: str | None) -> pd.DataFrame:
    history_df = build_candidate_history(df, target_month)
    if history_df.empty:
        return history_df

    canonical_history = history_df.copy()
    canonical_history["candidate"] = canonical_history["candidate"].apply(
        normalize_candidate_key
    )
    grouped_rows = []
    for candidate, candidate_df in canonical_history.groupby("candidate", sort=True):
        grouped_rows.append(
            {
                "candidate": candidate,
                "last_direct_seen": max_period(candidate_df["last_direct_seen"]),
                "last_mixed_seen": max_period(candidate_df["last_mixed_seen"]),
                "last_all_role_seen": max_period(
                    candidate_df["last_all_role_seen"]
                ),
                "months_since_direct": min_numeric(
                    candidate_df["months_since_direct"]
                ),
                "months_since_mixed": min_numeric(candidate_df["months_since_mixed"]),
                "months_since_all_role": min_numeric(
                    candidate_df["months_since_all_role"]
                ),
                "direct_count": int(candidate_df["direct_count"].sum()),
                "mixed_count": int(candidate_df["mixed_count"].sum()),
                "all_role_inclusion_count": int(
                    candidate_df["all_role_inclusion_count"].sum()
                ),
            }
        )
    return pd.DataFrame(grouped_rows)


def fill_history_defaults(scoring_df: pd.DataFrame) -> None:
    for column in ["direct_count", "mixed_count", "all_role_inclusion_count"]:
        scoring_df[column] = scoring_df[column].fillna(0).astype(int)
    for column in [
        "last_direct_seen",
        "last_mixed_seen",
        "last_all_role_seen",
        "months_since_direct",
        "months_since_mixed",
        "months_since_all_role",
    ]:
        if column not in scoring_df:
            scoring_df[column] = None


def get_direct_recency_factor(
    months_since_direct,
    config: CandidateScoringConfig,
) -> float:
    months_since = to_int_or_none(months_since_direct)
    if months_since == 1:
        return config.direct_recent_1
    if months_since == 2:
        return config.direct_recent_2
    if months_since == 3:
        return config.direct_recent_3
    return 1.0


def get_soft_recency_factor(
    months_since,
    weight: float,
    config: CandidateScoringConfig,
) -> float:
    base_factor = get_direct_recency_factor(months_since, config)
    return 1 - float(weight) * (1 - base_factor)


def get_overdue_factor(
    candidate_group: str,
    months_since_direct,
    config: CandidateScoringConfig,
) -> float:
    months_since = to_int_or_none(months_since_direct)
    if candidate_group == "role":
        if months_since is None:
            return config.role_overdue_never
        if months_since >= 8:
            return config.role_overdue_8_plus
        if months_since >= 5:
            return config.role_overdue_5_plus
        return 1.0

    return 1.0


def get_same_candidate_months_since(row: pd.Series) -> int | None:
    if row["candidate"] == "all_role":
        return to_int_or_none(row["months_since_direct"])
    return min_numeric(
        pd.Series([row["months_since_direct"], row["months_since_mixed"]])
    )


def get_same_candidate_cooldown_factor(months_since) -> float:
    months = to_int_or_none(months_since)
    if months is None:
        return 1.0
    if months == 1:
        return 0.30
    if months == 2:
        return 0.55
    if months == 3:
        return 0.70
    if 4 <= months <= 6:
        return 0.85
    return 1.0


def get_individual_non_role_recency_factor(
    candidate: str,
    candidate_group: str,
    same_candidate_months_since,
    config: CandidateScoringConfig,
) -> float:
    if candidate == "other_non_role":
        return 1.0
    if candidate_group != "non_role":
        return 1.0

    months = to_int_or_none(same_candidate_months_since)
    if months is None:
        return config.individual_non_role_recency_factor_max
    if months <= 6:
        return 1.0
    if months <= 12:
        return 1.08
    return config.individual_non_role_recency_factor_max


def get_seasonal_factor(
    candidate: str,
    target_month: str | None,
    config: CandidateScoringConfig,
) -> float:
    if candidate not in HARD_SEASONAL_CANDIDATES:
        return 1.0
    if parse_predicted_month(target_month) is None:
        return 1.0
    return config.seasonal_month_boost


def get_all_role_candidate_factor(
    candidate: str,
    months_since_direct,
    config: CandidateScoringConfig,
) -> float:
    if candidate != "all_role":
        return 1.0
    months_since = to_int_or_none(months_since_direct)
    if months_since is None:
        return 1.0
    if months_since < 10:
        return config.all_role_recent_lt_10
    if months_since in {10, 11}:
        return config.all_role_recent_10_11
    return config.all_role_recent_12_plus


def get_non_role_context_factor(
    candidate_group: str,
    context: dict[str, object],
    config: CandidateScoringConfig,
) -> float:
    if candidate_group != "non_role":
        return 1.0

    role_pressure = float(context.get("recent_role_pressure", 0.0))
    non_role_pressure = float(context.get("recent_non_role_pressure", 0.0))
    pressure_delta = role_pressure - non_role_pressure
    factor = 1.0 + pressure_delta * 0.08
    if int(context.get("months_since_non_role_group", 0)) >= 2:
        factor += 0.05
    if bool(context.get("previous_is_non_role", False)):
        factor -= 0.08
    return clamp(
        factor,
        config.non_role_context_factor_min,
        config.non_role_context_factor_max,
    )


def get_other_non_role_context_factor(
    candidate: str,
    non_role_context_factor: float,
    config: CandidateScoringConfig,
) -> float:
    if candidate != "other_non_role":
        return 1.0
    return min(float(non_role_context_factor), config.other_non_role_context_factor_max)


def build_recent_context(
    df: pd.DataFrame,
    target_month: str | None,
) -> dict[str, object]:
    if df.empty or "effective_primary_category" not in df.columns:
        return empty_recent_context()

    source_df = df.copy()
    source_df["month_period"] = get_source_month_period(source_df)
    source_df = source_df.dropna(subset=["month_period"])
    target_period = normalize_month_period(target_month)
    if target_period is not None:
        source_df = source_df[source_df["month_period"] < target_period]
    if source_df.empty:
        return empty_recent_context()

    latest_month = source_df["month_period"].max()
    recent_role_pressure = 0.0
    recent_non_role_pressure = 0.0
    decay_by_months_ago = {0: 1.0, 1: 0.70, 2: 0.40}
    for _, row in source_df.iterrows():
        months_ago = months_between_periods(row["month_period"], latest_month)
        if months_ago not in decay_by_months_ago:
            continue
        role_pressure, non_role_pressure = get_row_group_pressure(row)
        decay = decay_by_months_ago[months_ago]
        recent_role_pressure += role_pressure * decay
        recent_non_role_pressure += non_role_pressure * decay

    latest_rows = source_df[source_df["month_period"].eq(latest_month)]
    previous_is_all_role = latest_rows["effective_primary_category"].astype(str).eq(
        "all_role"
    ).any()
    previous_is_non_role = latest_rows["effective_primary_category"].astype(str).eq(
        "non_role"
    ).any()
    non_role_months = sorted(
        source_df[
            source_df.apply(row_has_non_role_pressure, axis=1)
        ]["month_period"].unique()
    )
    months_since_non_role_group = 0
    if non_role_months:
        months_since_non_role_group = months_between_periods(
            max(non_role_months),
            latest_month,
        )
    else:
        months_since_non_role_group = 99

    return {
        "role_streak": 0,
        "previous_is_all_role": previous_is_all_role,
        "previous_is_non_role": previous_is_non_role,
        "recent_role_pressure": round(recent_role_pressure, 8),
        "recent_non_role_pressure": round(recent_non_role_pressure, 8),
        "months_since_non_role_group": months_since_non_role_group,
    }


def empty_recent_context() -> dict[str, object]:
    return {
        "role_streak": 0,
        "previous_is_all_role": False,
        "previous_is_non_role": False,
        "recent_role_pressure": 0.0,
        "recent_non_role_pressure": 0.0,
        "months_since_non_role_group": 99,
    }


def get_row_group_pressure(row: pd.Series) -> tuple[float, float]:
    if get_bool_value(row.get("is_all_role", False)) or str(
        row.get("effective_primary_category", "")
    ) == "all_role":
        return 1.5, 0.0

    if get_bool_value(row.get("is_mixed", False)) or str(
        row.get("effective_primary_category", "")
    ) == "both":
        components = parse_components(row.get("mixed_components", ""))
        if not components:
            return 0.5, 0.5
        role_count = sum(component in VALID_ROLE_COMPONENTS for component in components)
        non_role_count = sum(
            component in VALID_NON_ROLE_COMPONENTS for component in components
        )
        total = role_count + non_role_count
        if total <= 0:
            return 0.5, 0.5
        return role_count / total, non_role_count / total

    category = str(row.get("effective_primary_category", "")).strip().lower()
    if category == "role":
        return 1.0, 0.0
    if category == "non_role":
        return 0.0, 1.0
    return 0.0, 0.0


def row_has_non_role_pressure(row: pd.Series) -> bool:
    _, non_role_pressure = get_row_group_pressure(row)
    return non_role_pressure > 0


def months_between_periods(start: pd.Period, end: pd.Period) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def get_bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def get_source_month_period(df: pd.DataFrame) -> pd.Series:
    if "month_period" in df.columns:
        return df["month_period"].apply(normalize_month_period)
    if "month_label" in df.columns:
        return df["month_label"].apply(normalize_month_period)
    if "month" in df.columns:
        return df["month"].apply(normalize_month_period)
    return pd.Series([pd.NaT] * len(df), index=df.index)


def normalize_scores(scoring_df: pd.DataFrame) -> pd.DataFrame:
    result_df = scoring_df.copy()
    total_raw_score = float(result_df["raw_score"].sum())
    if total_raw_score <= 0:
        result_df["final_probability"] = 1 / len(result_df) if len(result_df) else 0.0
    else:
        result_df["final_probability"] = result_df["raw_score"] / total_raw_score
    result_df["final_probability_percent"] = result_df["final_probability"] * 100
    return result_df


def normalize_prior_strength(prior_strength: float) -> float:
    try:
        value = int(prior_strength)
    except (TypeError, ValueError):
        return 3
    if value < min(PRIOR_STRENGTH_OPTIONS):
        return float(min(PRIOR_STRENGTH_OPTIONS))
    if value > max(PRIOR_STRENGTH_OPTIONS):
        return float(max(PRIOR_STRENGTH_OPTIONS))
    return float(value)


def max_period(values: pd.Series) -> pd.Period | None:
    periods = [
        value
        for value in values.tolist()
        if value is not None and not pd.isna(value)
    ]
    if not periods:
        return None
    return max(periods)


def min_numeric(values: pd.Series) -> int | None:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return None
    return int(numeric_values.min())


def to_int_or_none(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def format_period_value(value):
    if value is None or pd.isna(value):
        return ""
    return str(value)


def round_scoring_table(scoring_df: pd.DataFrame) -> pd.DataFrame:
    result_df = scoring_df.copy()
    for column in ["last_direct_seen", "last_mixed_seen", "last_all_role_seen"]:
        result_df[column] = result_df[column].apply(format_period_value)

    numeric_columns = [
        "base_score",
        "frequency_factor_raw",
        "frequency_factor_clamped",
        "frequency_factor",
        "recent_role_pressure",
        "recent_non_role_pressure",
        "recency_factor",
        "same_candidate_months_since",
        "same_candidate_cooldown_factor",
        "individual_non_role_recency_factor",
        "overdue_factor",
        "seasonal_factor",
        "all_role_candidate_factor",
        "non_role_context_factor",
        "other_non_role_context_factor",
        "other_non_role_cap_factor",
        "rare_bucket_factor",
        "smoothing_factor",
        "raw_score",
        "final_probability",
        "final_probability_percent",
        "direct_recency_factor",
        "mixed_recency_factor",
        "all_role_inclusion_recency_factor",
        "combined_recency_factor_before_clamp",
        "combined_recency_factor",
        "effective_count",
        "prior_strength",
    ]
    result_df[numeric_columns] = result_df[numeric_columns].round(8)
    return result_df[SCORING_COLUMNS]
