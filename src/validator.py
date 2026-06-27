import pandas as pd

from src.seasonal_rules import SEASONAL_ALLOWED_MONTHS


def find_duplicate_months(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where month_label appears more than once."""
    if df.empty or "month_label" not in df.columns:
        return pd.DataFrame()

    duplicated = df[df["month_label"].duplicated(keep=False)]
    return duplicated.sort_values("month_label")


def find_missing_months(df: pd.DataFrame) -> list[str]:
    """Find missing YYYY-MM labels between the first and last valid month."""
    if df.empty or "month_label" not in df.columns:
        return []

    valid_months = pd.to_datetime(df["month_label"], format="%Y-%m", errors="coerce")
    if valid_months.dropna().empty:
        return []

    start = valid_months.min().to_period("M")
    end = valid_months.max().to_period("M")

    expected_months = [str(month) for month in pd.period_range(start, end, freq="M")]
    actual_months = set(valid_months.dropna().dt.to_period("M").astype(str))

    return [month for month in expected_months if month not in actual_months]


def find_happening_benefits(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows marked as ongoing."""
    if df.empty or "end_date" not in df.columns:
        return pd.DataFrame()

    return df[df["end_date"].astype(str).str.strip().str.lower() == "happening"]


def find_seasonal_quality_warnings(df: pd.DataFrame) -> pd.DataFrame:
    """Return non-blocking warnings for seasonal schema/month rule issues."""
    columns = ["month_label", "seasonal_type", "expected_month", "warning"]
    required_columns = [
        "month_label",
        "start_date",
        "primary_category",
        "is_seasonal",
        "seasonal_type",
    ]
    if df.empty or any(column not in df.columns for column in required_columns):
        return pd.DataFrame(columns=columns)

    warnings = []
    for _, row in df.iterrows():
        primary_category = str(row.get("primary_category", "")).strip().lower()
        is_seasonal = bool(row.get("is_seasonal", False))
        seasonal_type = str(row.get("seasonal_type", "")).strip().lower()
        month_label = str(row.get("month_label", "")).strip()

        if primary_category == "seasonal" and not is_seasonal:
            warnings.append(
                build_seasonal_warning(
                    month_label,
                    seasonal_type,
                    "",
                    "primary_category=seasonal but is_seasonal is FALSE",
                )
            )
        if primary_category == "seasonal" and not seasonal_type:
            warnings.append(
                build_seasonal_warning(
                    month_label,
                    seasonal_type,
                    "",
                    "primary_category=seasonal but seasonal_type is empty",
                )
            )
        if is_seasonal and not seasonal_type:
            warnings.append(
                build_seasonal_warning(
                    month_label,
                    seasonal_type,
                    "",
                    "is_seasonal is TRUE but seasonal_type is empty",
                )
            )
        if seasonal_type in SEASONAL_ALLOWED_MONTHS:
            actual_months = get_benefit_months(row)
            expected_months = SEASONAL_ALLOWED_MONTHS[seasonal_type]
            if actual_months and actual_months.isdisjoint(expected_months):
                warnings.append(
                    build_seasonal_warning(
                        month_label,
                        seasonal_type,
                        "/".join(str(month) for month in sorted(expected_months)),
                        "seasonal_type does not match its expected month",
                    )
                )

    return pd.DataFrame(warnings, columns=columns)


def build_seasonal_warning(
    month_label: str,
    seasonal_type: str,
    expected_month: str,
    warning: str,
) -> dict:
    return {
        "month_label": month_label,
        "seasonal_type": seasonal_type,
        "expected_month": expected_month,
        "warning": warning,
    }


def get_benefit_months(row: pd.Series) -> set[int]:
    months = set()
    start_date = pd.to_datetime(row.get("start_date", ""), errors="coerce")
    if pd.notna(start_date):
        months.add(int(start_date.month))

    month_label = pd.to_datetime(
        row.get("month_label", ""),
        format="%Y-%m",
        errors="coerce",
    )
    if pd.notna(month_label):
        months.add(int(month_label.month))
    return months
