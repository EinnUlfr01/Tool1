KNOWN_SEASONAL_TYPES = {
    "halloween",
    "holiday",
    "thanksgiving",
    "valentines",
    "easter",
}
SEASONAL_TYPE_MONTHS = {
    "halloween": {10},
    "holiday": {12},
    "thanksgiving": {11},
}
SEASONAL_COMPONENT_MONTHS = {
    "halloween": {10},
    "halloween_call_to_arms": {10},
    "holiday": {12},
    "holiday_call_to_arms": {12},
    "holiday_rewards": {12},
}
SEASONAL_ALLOWED_MONTHS = {
    **SEASONAL_TYPE_MONTHS,
    **SEASONAL_COMPONENT_MONTHS,
}
SEASONAL_DISPLAY_GROUPS = {
    "halloween": "halloween",
    "halloween_call_to_arms": "halloween",
    "holiday": "holiday",
    "holiday_call_to_arms": "holiday",
    "holiday_rewards": "holiday",
    "thanksgiving": "thanksgiving",
    "valentines": "valentines",
    "easter": "easter",
}
SEASONAL_DISPLAY_LABELS = {
    "halloween": "Halloween",
    "holiday": "Holiday",
    "thanksgiving": "Thanksgiving",
    "valentines": "Valentines",
    "easter": "Easter",
    "holiday_call_to_arms": "Holiday Call To Arms",
    "halloween_call_to_arms": "Halloween Call To Arms",
    "holiday_rewards": "Holiday Rewards",
}

SEASONAL_MONTH_BOOST = 1.20
OUT_OF_SEASON_PENALTY = 0.70


def is_seasonal_sub_category_allowed(
    sub_category: str,
    predicted_next_month: str | None,
    seasonal_type: str | None = None,
) -> bool:
    seasonal_key = get_seasonal_key(sub_category, seasonal_type)
    allowed_months = SEASONAL_ALLOWED_MONTHS.get(seasonal_key)
    if allowed_months is None:
        return True

    predicted_month = parse_predicted_month(predicted_next_month)
    if predicted_month is None:
        return True

    return predicted_month in allowed_months


def is_component_prediction_eligible(
    component: str,
    predicted_next_month: str | None,
) -> bool:
    eligible_months = get_component_eligible_months(component)
    if eligible_months is None:
        return True

    predicted_month = parse_predicted_month(predicted_next_month)
    if predicted_month is None:
        return True

    return predicted_month in eligible_months


def get_component_eligible_months(component: str) -> set[int] | None:
    normalized_component = str(component or "").strip().lower()
    return SEASONAL_COMPONENT_MONTHS.get(normalized_component)


def get_component_exclusion_reason(
    component: str,
    predicted_next_month: str | None,
) -> str:
    if is_component_prediction_eligible(component, predicted_next_month):
        return ""
    return "out_of_season_target_month"


def get_seasonal_prediction_multiplier(
    component: str,
    is_seasonal: bool,
    seasonal_type: str | None,
    predicted_next_month: str | None,
) -> float:
    if not bool(is_seasonal):
        return 1.0

    if not is_component_prediction_eligible(component, predicted_next_month):
        return 0.0

    seasonal_key = get_component_seasonal_key(component, seasonal_type)
    if seasonal_key not in SEASONAL_COMPONENT_MONTHS:
        return 1.0

    if is_seasonal_sub_category_allowed(
        seasonal_key,
        predicted_next_month,
    ):
        return SEASONAL_MONTH_BOOST
    return OUT_OF_SEASON_PENALTY


def get_seasonal_key(
    component: str,
    seasonal_type: str | None = None,
) -> str:
    normalized_type = str(seasonal_type or "").strip().lower()
    if normalized_type:
        return normalized_type
    return str(component).strip().lower()


def get_component_seasonal_key(
    component: str,
    seasonal_type: str | None = None,
) -> str:
    normalized_component = str(component or "").strip().lower()
    normalized_type = str(seasonal_type or "").strip().lower()
    if normalized_component in SEASONAL_COMPONENT_MONTHS:
        return normalized_component
    if normalized_type in SEASONAL_COMPONENT_MONTHS:
        return normalized_type
    return normalized_component


def normalize_seasonal_display_group(value: str | None) -> str:
    normalized_value = str(value or "").strip().lower()
    return SEASONAL_DISPLAY_GROUPS.get(normalized_value, normalized_value)


def get_seasonal_display_label(value: str | None) -> str:
    group = normalize_seasonal_display_group(value)
    return SEASONAL_DISPLAY_LABELS.get(group, group.replace("_", " ").title())


def parse_predicted_month(predicted_next_month: str | None) -> int | None:
    if not predicted_next_month:
        return None

    try:
        return int(str(predicted_next_month).split("-")[1])
    except (IndexError, TypeError, ValueError):
        return None
