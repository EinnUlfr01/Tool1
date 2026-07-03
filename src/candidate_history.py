from __future__ import annotations

import pandas as pd

from src.data_loader import parse_components
from src.taxonomy import (
    VALID_ALL_ROLE_CANDIDATES,
    VALID_NON_ROLE_COMPONENTS,
    VALID_PREDICTION_COMPONENTS,
    VALID_ROLE_COMPONENTS,
    VALID_SEASONAL_COMPONENTS,
    canonical_component,
)


HISTORY_COLUMNS = [
    "candidate",
    "candidate_group",
    "last_direct_seen",
    "last_mixed_seen",
    "last_all_role_seen",
    "last_any_seen",
    "months_since_direct",
    "months_since_mixed",
    "months_since_all_role",
    "months_since_any",
    "direct_count",
    "mixed_count",
    "all_role_inclusion_count",
    "any_count",
]


def build_candidate_history(df: pd.DataFrame, target_month) -> pd.DataFrame:
    """Build per-candidate appearance history before the prediction month."""
    if df.empty:
        return empty_candidate_history()

    source_df = df.copy()
    source_df["month_period"] = get_month_period_series(source_df)
    source_df = source_df.dropna(subset=["month_period"])
    target_period = normalize_month_period(target_month)
    if target_period is not None:
        source_df = source_df[source_df["month_period"] < target_period]

    candidate_universe = get_candidate_universe(source_df)
    rows = [
        build_candidate_history_row(source_df, candidate, target_period)
        for candidate in candidate_universe
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def empty_candidate_history() -> pd.DataFrame:
    rows = [
        build_candidate_history_row(
            pd.DataFrame(columns=["month_period"]),
            candidate,
            None,
        )
        for candidate in get_candidate_universe(pd.DataFrame())
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def build_candidate_history_row(
    history_df: pd.DataFrame,
    candidate: str,
    target_period: pd.Period | None,
) -> dict:
    direct_months = []
    mixed_months = []
    all_role_months = []

    for row in history_df.itertuples(index=False):
        month_period = getattr(row, "month_period", None)
        if is_direct_candidate_row(row, candidate):
            direct_months.append(month_period)
        if is_mixed_candidate_row(row, candidate):
            mixed_months.append(month_period)
        if is_all_role_inclusion_row(row, candidate):
            all_role_months.append(month_period)

    any_months = direct_months + mixed_months + all_role_months
    last_direct_seen = find_last_seen(direct_months)
    last_mixed_seen = find_last_seen(mixed_months)
    last_all_role_seen = find_last_seen(all_role_months)
    last_any_seen = find_last_seen(any_months)

    return {
        "candidate": candidate,
        "candidate_group": get_candidate_group(candidate),
        "last_direct_seen": last_direct_seen,
        "last_mixed_seen": last_mixed_seen,
        "last_all_role_seen": last_all_role_seen,
        "last_any_seen": last_any_seen,
        "months_since_direct": months_between(last_direct_seen, target_period),
        "months_since_mixed": months_between(last_mixed_seen, target_period),
        "months_since_all_role": months_between(last_all_role_seen, target_period),
        "months_since_any": months_between(last_any_seen, target_period),
        "direct_count": len(direct_months),
        "mixed_count": len(mixed_months),
        "all_role_inclusion_count": len(all_role_months),
        "any_count": len(any_months),
    }


def get_candidate_universe(df: pd.DataFrame) -> list[str]:
    candidates = set(VALID_PREDICTION_COMPONENTS)

    for column in ["effective_candidate", "primary_sub_category"]:
        if column in df.columns:
            candidates.update(
                canonical_component(value)
                for value in df[column].dropna().tolist()
                if canonical_component(value)
            )

    for column, context in [
        ("mixed_components", "mixed"),
        ("all_role_components", "all_role"),
    ]:
        if column in df.columns:
            for value in df[column].dropna().tolist():
                candidates.update(parse_components(value, context=context))

    return sorted(candidate for candidate in candidates if candidate)


def get_candidate_group(candidate: str) -> str:
    if candidate in VALID_ALL_ROLE_CANDIDATES:
        return "all_role"
    if candidate in VALID_ROLE_COMPONENTS:
        return "role"
    if candidate in VALID_SEASONAL_COMPONENTS:
        return "seasonal"
    if candidate in VALID_NON_ROLE_COMPONENTS:
        return "non_role"
    return "unknown"


def is_direct_candidate_row(row, candidate: str) -> bool:
    is_all_role = get_bool(row, "is_all_role")
    if candidate in VALID_ALL_ROLE_CANDIDATES:
        return is_all_role or get_text(row, "effective_candidate") == candidate

    return (
        get_text(row, "effective_candidate") == candidate
        and not get_bool(row, "is_mixed")
        and not is_all_role
    )


def is_mixed_candidate_row(row, candidate: str) -> bool:
    if not get_bool(row, "is_mixed"):
        return False
    return candidate in parse_component_list(getattr(row, "mixed_components", ""))


def is_all_role_inclusion_row(row, candidate: str) -> bool:
    if candidate not in VALID_ROLE_COMPONENTS:
        return False
    if not get_bool(row, "is_all_role"):
        return False
    return candidate in parse_component_list(
        getattr(row, "all_role_components", ""),
        context="all_role",
    )


def parse_component_list(value: object, context: str = "mixed") -> list[str]:
    return parse_components(value, context=context)


def find_last_seen(months: list[pd.Period]) -> pd.Period | None:
    valid_months = [month for month in months if month is not None and not pd.isna(month)]
    if not valid_months:
        return None
    return max(valid_months)


def months_between(
    last_seen_month: pd.Period | None,
    target_month: pd.Period | None,
) -> int | None:
    if last_seen_month is None or target_month is None:
        return None
    return (target_month.year - last_seen_month.year) * 12 + (
        target_month.month - last_seen_month.month
    )


def get_month_period_series(df: pd.DataFrame) -> pd.Series:
    if "month_period" in df.columns:
        return df["month_period"].apply(normalize_month_period)
    if "month_label" in df.columns:
        return df["month_label"].apply(normalize_month_period)
    if "month" in df.columns:
        return df["month"].apply(normalize_month_period)
    return pd.Series([pd.NaT] * len(df), index=df.index)


def normalize_month_period(value) -> pd.Period | None:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    if pd.isna(value):
        return None

    text_value = str(value).strip()
    if not text_value:
        return None

    try:
        return pd.Period(text_value[:7], freq="M")
    except ValueError:
        month_date = pd.to_datetime(text_value, errors="coerce")
        if pd.isna(month_date):
            return None
        return month_date.to_period("M")


def get_bool(row, column: str) -> bool:
    value = getattr(row, column, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def get_text(row, column: str) -> str:
    return canonical_component(getattr(row, column, ""))
