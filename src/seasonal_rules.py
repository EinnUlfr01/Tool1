SEASONAL_ALLOWED_MONTHS = {
    "halloween": {10, 11},
    "holiday": {12, 1},
    "christmas": {12, 1},
    "holiday_call_to_arms": {12, 1},
    "new_year": {1},
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


def get_seasonal_prediction_multiplier(
    component: str,
    is_seasonal: bool,
    seasonal_type: str | None,
    predicted_next_month: str | None,
) -> float:
    if not bool(is_seasonal):
        return 1.0

    seasonal_key = get_seasonal_key(component, seasonal_type)
    if seasonal_key not in SEASONAL_ALLOWED_MONTHS:
        return 1.0

    if is_seasonal_sub_category_allowed(
        component,
        predicted_next_month,
        seasonal_type=seasonal_type,
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


def parse_predicted_month(predicted_next_month: str | None) -> int | None:
    if not predicted_next_month:
        return None

    try:
        return int(str(predicted_next_month).split("-")[1])
    except (IndexError, TypeError, ValueError):
        return None
