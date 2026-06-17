SEASONAL_ALLOWED_MONTHS = {
    "halloween": {10, 11},
    "holiday_call_to_arms": {12, 1},
}


def is_seasonal_sub_category_allowed(
    sub_category: str,
    predicted_next_month: str | None,
) -> bool:
    sub_category = str(sub_category).strip().lower()
    allowed_months = SEASONAL_ALLOWED_MONTHS.get(sub_category)
    if allowed_months is None:
        return True

    predicted_month = parse_predicted_month(predicted_next_month)
    if predicted_month is None:
        return True

    return predicted_month in allowed_months


def parse_predicted_month(predicted_next_month: str | None) -> int | None:
    if not predicted_next_month:
        return None

    try:
        return int(str(predicted_next_month).split("-")[1])
    except (IndexError, TypeError, ValueError):
        return None
