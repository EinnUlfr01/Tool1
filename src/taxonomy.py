from __future__ import annotations

import re
from dataclasses import dataclass


VALID_PRIMARY_CATEGORIES = {
    "role",
    "non_role",
    "both",
}

VALID_ROLE_COMPONENTS = {
    "bounty_hunter",
    "trader",
    "collector",
    "moonshiner",
    "naturalist",
}

VALID_NON_ROLE_COMPONENTS = {
    "free_roam",
    "races",
    "blood_money",
    "telegram",
    "call_to_arms",
    "featured_series",
    "other_non_role",
}

VALID_SEASONAL_COMPONENTS = {
    "halloween",
    "halloween_call_to_arms",
    "holiday",
    "holiday_call_to_arms",
    "holiday_rewards",
}

BENEFIT_INFO_TAGS = {
    "random_event",
    "posse_bonus",
    "camp_bonus",
    "stable_discount",
    "weapon_discount",
    "clothing_discount",
    "discount",
    "limited_time_reward",
    "bonus_reward",
}

KNOWN_RARE_NON_ROLE_ALIASES = {
    "story_missions": "other_non_role",
    "showdown": "other_non_role",
    "gang_hideouts": "other_non_role",
}

SUBCATEGORY_ALIASES = {
    "free_room": "free_roam",
    "free roam": "free_roam",
    "freeroam": "free_roam",
    "free_roam_events": "free_roam",
    "free roam events": "free_roam",
    "free_roam_missions": "free_roam",
    "free roam missions": "free_roam",
    "race": "races",
    "race_series": "races",
    "race series": "races",
    "blood money": "blood_money",
    "blood-money": "blood_money",
    "telegram missions": "telegram",
    "telegram_missions": "telegram",
    "cta": "call_to_arms",
    "call to arms": "call_to_arms",
    "call-to-arms": "call_to_arms",
    "featured series": "featured_series",
    "featured_series_pvp": "featured_series",
    "bounty": "bounty_hunter",
    "bounty hunter": "bounty_hunter",
    "bounty-hunter": "bounty_hunter",
    "other": "other_non_role",
    "other non role": "other_non_role",
    "other non-role": "other_non_role",
    "other_nonrole": "other_non_role",
    "holiday call to arms": "holiday_call_to_arms",
    "holiday-call-to-arms": "holiday_call_to_arms",
    "halloween call to arms": "halloween_call_to_arms",
    "halloween-call-to-arms": "halloween_call_to_arms",
    **KNOWN_RARE_NON_ROLE_ALIASES,
}

VALID_PREDICTION_COMPONENTS = (
    VALID_ROLE_COMPONENTS | VALID_NON_ROLE_COMPONENTS | VALID_SEASONAL_COMPONENTS
)


@dataclass(frozen=True)
class NormalizedComponent:
    raw_value: str
    normalized_value: str
    severity: str
    message: str
    suggested_fix: str


def normalize_token(value: object) -> str:
    """Normalize one loose token into lower snake_case form."""
    normalized = stringify_token(value).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in SUBCATEGORY_ALIASES:
        return normalized
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def stringify_token(value: object) -> str:
    if value is None:
        return ""
    if value != value:
        return ""
    return str(value).strip()


def canonical_component(value: object) -> str:
    token = normalize_token(value)
    return SUBCATEGORY_ALIASES.get(token, token)


def normalize_component(
    value: object,
    context: str = "mixed",
) -> NormalizedComponent:
    raw_value = stringify_token(value)
    token = normalize_token(raw_value)
    normalized_value = SUBCATEGORY_ALIASES.get(token, token)

    if not token:
        return NormalizedComponent(raw_value, "", "", "", "")

    if token in KNOWN_RARE_NON_ROLE_ALIASES:
        return NormalizedComponent(
            raw_value,
            normalized_value,
            "info",
            "Known rare non-role grouped under other_non_role for prediction.",
            "No change needed.",
        )

    if normalized_value == "other_non_role" and token == "other_non_role":
        return NormalizedComponent(
            raw_value,
            normalized_value,
            "warning",
            "Direct other_non_role found in CSV. Use the original raw non-role sub-category when known.",
            "Replace with raw sub-category such as story_missions, gang_hideouts, showdown, or another specific raw non-role if known.",
        )

    if token in SUBCATEGORY_ALIASES and normalized_value != token:
        return NormalizedComponent(
            raw_value,
            normalized_value,
            "info",
            "Alias normalized.",
            f"Use {normalized_value} in CSV.",
        )

    if normalized_value in BENEFIT_INFO_TAGS:
        return NormalizedComponent(
            raw_value,
            "other_non_role",
            "warning",
            "This looks like benefit info, not a prediction sub-category.",
            "Move to multiplier_info or benefits_info later.",
        )

    if normalized_value in VALID_PREDICTION_COMPONENTS:
        return NormalizedComponent(raw_value, normalized_value, "", "", "")

    if context in {"role", "all_role"}:
        return NormalizedComponent(
            raw_value,
            "",
            "error",
            "Unknown role component.",
            "Use one of VALID_ROLE_COMPONENTS.",
        )

    return NormalizedComponent(
        raw_value,
        "other_non_role",
        "warning",
        "Unknown non-role component mapped to other_non_role.",
        "Add taxonomy mapping if this becomes a recurring component.",
    )


def taxonomy_payload() -> dict[str, object]:
    return {
        "valid_primary_categories": sorted(VALID_PRIMARY_CATEGORIES),
        "valid_role_components": sorted(VALID_ROLE_COMPONENTS),
        "valid_non_role_components": sorted(VALID_NON_ROLE_COMPONENTS),
        "valid_seasonal_components": sorted(VALID_SEASONAL_COMPONENTS),
        "benefit_info_tags": sorted(BENEFIT_INFO_TAGS),
        "subcategory_aliases": dict(sorted(SUBCATEGORY_ALIASES.items())),
    }
