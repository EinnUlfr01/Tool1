import pandas as pd


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
