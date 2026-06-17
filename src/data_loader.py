from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/rdo_benefits_raw.csv")

REQUIRED_COLUMNS = [
    "month_label",
    "start_date",
    "end_date",
    "primary_category",
    "primary_sub_category",
    "extra_tags",
    "source_name",
    "source_url",
    "confidence",
    "note",
]

CONFIDENCE_OPTIONS = ["all", "high", "medium", "low"]


def load_raw_benefits(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV without changing its schema."""
    csv_path = Path(path)

    if not csv_path.exists():
        raise ValueError(f"CSV file was not found: {csv_path}")

    if csv_path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {csv_path}") from exc

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError("CSV has headers but no data rows.")

    return clean_raw_benefits(df)


def clean_raw_benefits(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize values used for filtering and analysis."""
    clean_df = df.copy()

    for column in REQUIRED_COLUMNS:
        clean_df[column] = clean_df[column].fillna("").astype(str).str.strip()

    clean_df["confidence"] = clean_df["confidence"].str.lower()
    clean_df["primary_category"] = clean_df["primary_category"].str.lower()
    clean_df["primary_sub_category"] = clean_df["primary_sub_category"].str.lower()

    return clean_df


def get_available_years(df: pd.DataFrame) -> list[int]:
    month_dates = pd.to_datetime(df["month_label"], format="%Y-%m", errors="coerce")
    years = month_dates.dropna().dt.year.astype(int).unique()
    return sorted(years)


def filter_by_year(df: pd.DataFrame, selected_year: int | str) -> pd.DataFrame:
    if selected_year == "all":
        return df

    filtered_df = df.copy()
    month_dates = pd.to_datetime(
        filtered_df["month_label"], format="%Y-%m", errors="coerce"
    )
    return filtered_df[month_dates.dt.year == int(selected_year)]


def filter_by_confidence(df: pd.DataFrame, selected_confidence: str) -> pd.DataFrame:
    if selected_confidence == "all":
        return df

    return df[df["confidence"] == selected_confidence]


def get_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main analysis excludes low-confidence rows."""
    return df[df["confidence"] != "low"].copy()
