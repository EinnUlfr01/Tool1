from __future__ import annotations

import pandas as pd

from src.seasonal_rules import get_component_eligible_months, parse_predicted_month
from src.taxonomy import (
    VALID_ALL_ROLE_CANDIDATES,
    VALID_NON_ROLE_COMPONENTS,
    VALID_ROLE_COMPONENTS,
    canonical_component,
)


CANDIDATE_POOL_COLUMNS = [
    "candidate",
    "canonical_candidate",
    "candidate_group",
    "display_label",
    "is_seasonal_candidate",
    "valid_for_target_month",
    "excluded_reason",
    "eligible_months",
]

CANDIDATE_ALIASES = {
    "telegram": "telegram_missions",
    "telegram_missions": "telegram_missions",
    "holiday_rewards": "holiday_call_to_arms",
}

HARD_SEASONAL_CANDIDATES = {
    "halloween",
    "halloween_call_to_arms",
    "holiday",
    "holiday_call_to_arms",
}

EXCLUDED_SEASONAL_CONTEXTS = {
    "thanksgiving",
    "valentines",
    "easter",
}

DISPLAY_LABELS = {
    "all_role": "All Roles",
    "telegram_missions": "Telegram Missions",
    "other_non_role": "Other Non-role",
    "halloween": "Halloween",
    "halloween_call_to_arms": "Halloween Call To Arms",
    "holiday": "Holiday",
    "holiday_call_to_arms": "Holiday Call To Arms",
}


def build_candidate_pool(target_month: str | None = None) -> pd.DataFrame:
    rows = [
        build_candidate_pool_row(candidate, target_month)
        for candidate in get_candidate_pool_universe()
    ]
    return pd.DataFrame(rows, columns=CANDIDATE_POOL_COLUMNS)


def get_valid_candidates(target_month: str | None = None) -> list[str]:
    pool_df = build_candidate_pool(target_month)
    valid_df = pool_df[pool_df["valid_for_target_month"].eq(True)]
    return valid_df["canonical_candidate"].drop_duplicates().tolist()


def build_candidate_pool_row(candidate: str, target_month: str | None) -> dict:
    canonical_candidate = normalize_candidate_key(candidate)
    eligible_months = get_candidate_eligible_months(canonical_candidate)
    valid_for_target_month = is_valid_for_target_month(
        canonical_candidate,
        target_month,
    )
    return {
        "candidate": canonical_candidate,
        "canonical_candidate": canonical_candidate,
        "candidate_group": get_candidate_group(canonical_candidate),
        "display_label": get_candidate_display_label(canonical_candidate),
        "is_seasonal_candidate": canonical_candidate in HARD_SEASONAL_CANDIDATES,
        "valid_for_target_month": valid_for_target_month,
        "excluded_reason": ""
        if valid_for_target_month
        else "out_of_season_target_month",
        "eligible_months": sorted(eligible_months) if eligible_months else [],
    }


def get_candidate_pool_universe() -> list[str]:
    candidates = set(VALID_ROLE_COMPONENTS)
    candidates.update(VALID_ALL_ROLE_CANDIDATES)
    candidates.update(
        normalize_candidate_key(candidate)
        for candidate in VALID_NON_ROLE_COMPONENTS
    )
    candidates.update(HARD_SEASONAL_CANDIDATES)
    candidates -= EXCLUDED_SEASONAL_CONTEXTS
    return sorted(candidate for candidate in candidates if candidate)


def normalize_candidate_key(value: str | None) -> str:
    candidate = canonical_component(value)
    return CANDIDATE_ALIASES.get(candidate, candidate)


def get_candidate_display_label(candidate: str) -> str:
    canonical_candidate = normalize_candidate_key(candidate)
    return DISPLAY_LABELS.get(
        canonical_candidate,
        canonical_candidate.replace("_", " ").title(),
    )


def get_candidate_group(candidate: str) -> str:
    canonical_candidate = normalize_candidate_key(candidate)
    if canonical_candidate in VALID_ALL_ROLE_CANDIDATES:
        return "all_role"
    if canonical_candidate in VALID_ROLE_COMPONENTS:
        return "role"
    if canonical_candidate in HARD_SEASONAL_CANDIDATES:
        return "seasonal"
    return "non_role"


def get_candidate_eligible_months(candidate: str) -> set[int] | None:
    if candidate not in HARD_SEASONAL_CANDIDATES:
        return None
    return get_component_eligible_months(candidate)


def is_valid_for_target_month(candidate: str, target_month: str | None) -> bool:
    eligible_months = get_candidate_eligible_months(candidate)
    if eligible_months is None:
        return True

    predicted_month = parse_predicted_month(target_month)
    if predicted_month is None:
        return True

    return predicted_month in eligible_months
