from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/rdo_benefits_raw.csv")

REQUIRED_COLUMNS = [
    "month_label",
    "start_date",
    "end_date",
    "primary_category",
    "primary_sub_category",
    "is_mixed",
    "mixed_components",
    "is_all_role",
    "all_role_components",
    "is_seasonal",
    "seasonal_type",
    "multiplier_info",
    "weight",
    "component_weight_rule",
    "source_name",
    "source_url",
    "confidence",
]

VALID_PRIMARY_CATEGORIES = {"role", "non_role", "both"}
ROLE_COMPONENTS = {
    "bounty_hunter",
    "trader",
    "collector",
    "naturalist",
    "moonshiner",
}
BOOLEAN_COLUMNS = ["is_mixed", "is_all_role", "is_seasonal"]
TRUE_VALUES = {"true", "1", "yes"}
FALSE_VALUES = {"false", "0", "no", ""}
CONFIDENCE_OPTIONS = ["all", "high", "medium", "low"]


def load_raw_benefits(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read and normalize the new benefits CSV schema."""
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
        legacy_hint = ""
        if {"month_label", "primary_category"}.issubset(df.columns):
            legacy_hint = (
                " CSV appears to use the old schema. Use the new schema with "
                "month_label, primary_category, is_mixed, is_all_role, "
                "component fields, and weight."
            )
        raise ValueError(
            "CSV is missing required new-schema columns: "
            + ", ".join(missing_columns)
            + legacy_hint
        )

    if df.empty:
        raise ValueError("CSV has headers but no data rows.")

    return clean_raw_benefits(df)


def clean_raw_benefits(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize values used by filtering, expansion, and prediction."""
    clean_df = df.copy()

    for column in clean_df.columns:
        if clean_df[column].dtype == object:
            clean_df[column] = clean_df[column].fillna("").astype(str).str.strip()

    clean_df["confidence"] = clean_df["confidence"].str.lower()
    clean_df["primary_category"] = clean_df["primary_category"].str.lower()
    clean_df["primary_sub_category"] = clean_df["primary_sub_category"].apply(
        normalize_component_string
    )
    clean_df["mixed_components"] = clean_df["mixed_components"].apply(
        normalize_component_string
    )
    clean_df["all_role_components"] = clean_df["all_role_components"].apply(
        normalize_component_string
    )
    clean_df["component_weight_rule"] = (
        clean_df["component_weight_rule"].str.lower()
    )
    clean_df["seasonal_type"] = clean_df["seasonal_type"].str.lower()

    for column in BOOLEAN_COLUMNS:
        clean_df[column] = clean_df[column].apply(
            lambda value: parse_boolean(value, column)
        )

    clean_df["weight"] = pd.to_numeric(clean_df["weight"], errors="coerce")
    invalid_weight_rows = clean_df.index[
        clean_df["weight"].isna() | (clean_df["weight"] < 0)
    ].tolist()
    if invalid_weight_rows:
        row_numbers = ", ".join(str(index + 2) for index in invalid_weight_rows[:10])
        raise ValueError(
            "CSV has invalid weight values at row(s): "
            f"{row_numbers}. Weight must be a non-negative number."
        )

    invalid_categories = sorted(
        set(clean_df["primary_category"]) - VALID_PRIMARY_CATEGORIES
    )
    if invalid_categories:
        raise ValueError(
            "CSV primary_category only accepts role, non_role, or both. "
            "Invalid value(s): "
            + ", ".join(invalid_categories)
        )

    invalid_months = pd.to_datetime(
        clean_df["month_label"],
        format="%Y-%m",
        errors="coerce",
    ).isna()
    if invalid_months.any():
        row_numbers = ", ".join(
            str(index + 2) for index in clean_df.index[invalid_months][:10]
        )
        raise ValueError(
            f"CSV has invalid month_label values at row(s): {row_numbers}. "
            "Expected YYYY-MM."
        )

    return clean_df


def parse_boolean(value: object, column_name: str = "boolean") -> bool:
    normalized_value = str(value).strip().lower()
    if normalized_value in TRUE_VALUES:
        return True
    if normalized_value in FALSE_VALUES:
        return False
    raise ValueError(
        f"CSV column {column_name} contains invalid boolean value: {value!r}. "
        "Accepted values are TRUE/FALSE, true/false, 1/0, or yes/no."
    )


def parse_components(value: object) -> list[str]:
    """Split a pipe-separated component field into normalized unique values."""
    components = []
    seen = set()
    for raw_component in str(value or "").split("|"):
        component = raw_component.strip().lower()
        if component and component not in seen:
            components.append(component)
            seen.add(component)
    return components


def normalize_component_string(value: object) -> str:
    return "|".join(parse_components(value))


def expand_components(row: pd.Series) -> list[dict]:
    """Expand one monthly row into weighted role/non-role component records."""
    if bool(row.get("is_all_role", False)):
        components = parse_components(row.get("all_role_components", ""))
    elif bool(row.get("is_mixed", False)):
        components = parse_components(row.get("mixed_components", ""))
    else:
        components = parse_components(row.get("primary_sub_category", ""))

    if not components:
        return []

    weight = float(row.get("weight", 0.0))
    component_weight = weight / len(components)

    return [
        {
            "month_label": row.get("month_label", ""),
            "primary_category": row.get("primary_category", ""),
            "component": component,
            "component_type": (
                "role" if component in ROLE_COMPONENTS else "non_role"
            ),
            "component_weight": component_weight,
            "is_mixed": bool(row.get("is_mixed", False)),
            "is_all_role": bool(row.get("is_all_role", False)),
            "is_seasonal": bool(row.get("is_seasonal", False)),
            "seasonal_type": row.get("seasonal_type", ""),
            "component_weight_rule": row.get("component_weight_rule", ""),
        }
        for component in components
    ]


def expand_benefit_components(df: pd.DataFrame) -> pd.DataFrame:
    """Expand every row while retaining its source primary category."""
    columns = [
        "month_label",
        "primary_category",
        "component",
        "component_type",
        "component_weight",
        "is_mixed",
        "is_all_role",
        "is_seasonal",
        "seasonal_type",
        "component_weight_rule",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for _, row in df.iterrows():
        records.extend(expand_components(row))

    return pd.DataFrame(records, columns=columns)


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
    """Return analysis rows without weighting or filtering by confidence."""
    return df.copy()
