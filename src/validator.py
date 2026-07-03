import pandas as pd

from src.seasonal_rules import KNOWN_SEASONAL_TYPES, SEASONAL_TYPE_MONTHS
from src.taxonomy import (
    VALID_PREDICTION_COMPONENTS,
    VALID_PRIMARY_CATEGORIES,
    VALID_ROLE_COMPONENTS,
    canonical_component,
    normalize_component,
    stringify_token,
)


VALIDATION_COLUMNS = [
    "row_index",
    "month_label",
    "field",
    "raw_value",
    "normalized_value",
    "severity",
    "message",
    "suggested_fix",
]


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
        if seasonal_type in SEASONAL_TYPE_MONTHS:
            actual_months = get_benefit_months(row)
            expected_months = SEASONAL_TYPE_MONTHS[seasonal_type]
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


def validate_sub_category_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate V2 prediction taxonomy without mutating source CSV data."""
    if df.empty:
        return pd.DataFrame(columns=VALIDATION_COLUMNS)

    records = []
    for index, row in df.iterrows():
        month_label = str(row.get("month_label", "") or "").strip()
        primary_category = str(row.get("primary_category", "") or "").strip().lower()
        is_mixed = bool(row.get("is_mixed", False))
        is_all_role = bool(row.get("is_all_role", False))
        is_seasonal = bool(row.get("is_seasonal", False))

        if primary_category not in VALID_PRIMARY_CATEGORIES:
            records.append(
                build_validation_record(
                    index,
                    month_label,
                    "primary_category",
                    raw_field_value(row, "primary_category"),
                    primary_category,
                    "error",
                    "Unknown primary category.",
                    "Use role, non_role, both, or all_role.",
                )
            )

        primary_raw = raw_field_value(row, "primary_sub_category")
        mixed_raw = raw_field_value(row, "mixed_components")
        all_role_raw = raw_field_value(row, "all_role_components")

        primary_components = split_component_values(primary_raw)
        mixed_components = split_component_values(mixed_raw)
        all_role_components = split_component_values(all_role_raw)

        if primary_category == "all_role" and not is_all_role:
            records.append(
                build_validation_record(
                    index,
                    month_label,
                    "is_all_role",
                    raw_field_value(row, "is_all_role"),
                    str(is_all_role),
                    "warning",
                    "primary_category=all_role but is_all_role is FALSE.",
                    "Set is_all_role=TRUE for all-role rows.",
                )
            )
        if is_all_role and primary_category not in {"all_role", "role", "both"}:
            records.append(
                build_validation_record(
                    index,
                    month_label,
                    "primary_category",
                    raw_field_value(row, "primary_category"),
                    primary_category,
                    "warning",
                    "is_all_role is TRUE but primary_category is not all_role.",
                    "Use primary_category=all_role for new all-role rows.",
                )
            )

        if primary_category == "role" and not primary_components:
            if not (is_all_role and all_role_components):
                records.append(
                    build_validation_record(
                        index,
                        month_label,
                        "primary_sub_category",
                        primary_raw,
                        "",
                        "error",
                        "primary_sub_category is required for role rows.",
                        "Fill a valid role component or all_role_components.",
                    )
                )
        if primary_category == "non_role" and not primary_components:
            records.append(
                build_validation_record(
                    index,
                    month_label,
                    "primary_sub_category",
                    primary_raw,
                    "",
                    "error",
                    "primary_sub_category is required for non_role rows.",
                    "Fill a valid non-role component.",
                )
            )
        if primary_category == "both" and not mixed_components:
            if is_all_role and all_role_components:
                pass
            else:
                records.append(
                    build_validation_record(
                        index,
                        month_label,
                        "mixed_components",
                        mixed_raw,
                        "",
                        "error",
                        "mixed_components is required for both rows.",
                        "Fill pipe-separated valid role/non-role components.",
                    )
                )
        if is_mixed and not mixed_components:
            records.append(
                build_validation_record(
                    index,
                    month_label,
                    "mixed_components",
                    mixed_raw,
                    "",
                    "error",
                    "is_mixed is TRUE but mixed_components is empty.",
                    "Fill pipe-separated valid mixed components.",
                )
            )
        if is_all_role and not all_role_components:
            records.append(
                build_validation_record(
                    index,
                    month_label,
                    "all_role_components",
                    all_role_raw,
                    "",
                    "error",
                    "is_all_role is TRUE but all_role_components is empty.",
                    "Fill all valid role components.",
                )
            )

        if primary_components and not should_skip_primary_components(
            primary_components,
            is_all_role,
            all_role_components,
        ):
            records.extend(
                validate_component_field(
                    index,
                    month_label,
                    "primary_sub_category",
                    primary_components,
                    component_context_for_primary(primary_category),
                )
            )
        if mixed_components:
            records.extend(
                validate_component_field(
                    index,
                    month_label,
                    "mixed_components",
                    mixed_components,
                    "mixed",
                )
            )
        if all_role_components:
            records.extend(
                validate_all_role_components(
                    index,
                    month_label,
                    all_role_components,
                )
            )

        seasonal_type = raw_field_value(row, "seasonal_type")
        normalized_seasonal_type = canonical_component(seasonal_type)
        if seasonal_type and normalized_seasonal_type not in KNOWN_SEASONAL_TYPES:
            records.append(
                build_validation_record(
                    index,
                    month_label,
                    "seasonal_type",
                    seasonal_type,
                    normalized_seasonal_type,
                    "warning",
                    "seasonal_type is not in current seasonal mapping.",
                    "Add it to known seasonal types if this is historical metadata, or add eligibility if recurring.",
                )
            )
        if (
            is_seasonal
            and normalized_seasonal_type in VALID_PREDICTION_COMPONENTS
            and normalized_seasonal_type not in SEASONAL_TYPE_MONTHS
        ):
            records.append(
                build_validation_record(
                    index,
                    month_label,
                    "seasonal_type",
                    seasonal_type,
                    normalized_seasonal_type,
                    "warning",
                    "seasonal_type is being used without eligibility mapping.",
                    "Add this seasonal type to seasonal_rules before applying timing rules.",
                )
            )

    return pd.DataFrame(records, columns=VALIDATION_COLUMNS)


def split_component_values(value: object) -> list[str]:
    return [
        stringify_token(component)
        for component in stringify_token(value).split("|")
        if stringify_token(component)
    ]


def raw_field_value(row: pd.Series, field: str) -> str:
    raw_field = f"raw_{field}"
    if raw_field in row:
        return stringify_token(row.get(raw_field, ""))
    return stringify_token(row.get(field, ""))


def should_skip_primary_components(
    primary_components: list[str],
    is_all_role: bool,
    all_role_components: list[str],
) -> bool:
    normalized = {canonical_component(component) for component in primary_components}
    return bool(is_all_role and all_role_components and normalized <= {"all_role"})


def component_context_for_primary(primary_category: str) -> str:
    if primary_category == "role":
        return "role"
    if primary_category == "non_role":
        return "non_role"
    if primary_category == "all_role":
        return "all_role"
    return "mixed"


def validate_component_field(
    row_index: int,
    month_label: str,
    field: str,
    components: list[str],
    context: str,
) -> list[dict]:
    records = []
    for component in components:
        result = normalize_component(component, context=context)
        if result.severity:
            records.append(
                build_validation_record(
                    row_index,
                    month_label,
                    field,
                    result.raw_value,
                    result.normalized_value,
                    result.severity,
                    result.message,
                    result.suggested_fix,
                )
            )
    return records


def validate_all_role_components(
    row_index: int,
    month_label: str,
    components: list[str],
) -> list[dict]:
    records = []
    for component in components:
        result = normalize_component(component, context="all_role")
        normalized_value = result.normalized_value or canonical_component(component)
        if result.severity:
            records.append(
                build_validation_record(
                    row_index,
                    month_label,
                    "all_role_components",
                    result.raw_value,
                    result.normalized_value,
                    result.severity,
                    result.message,
                    result.suggested_fix,
                )
            )
        elif normalized_value not in VALID_ROLE_COMPONENTS:
            records.append(
                build_validation_record(
                    row_index,
                    month_label,
                    "all_role_components",
                    component,
                    normalized_value,
                    "error",
                    "all_role_components only accepts role components.",
                    "Remove non-role components from all_role_components.",
                )
            )
    return records


def build_validation_record(
    row_index: int,
    month_label: str,
    field: str,
    raw_value: str,
    normalized_value: str,
    severity: str,
    message: str,
    suggested_fix: str,
) -> dict:
    return {
        "row_index": int(row_index),
        "month_label": month_label,
        "field": field,
        "raw_value": raw_value,
        "normalized_value": normalized_value,
        "severity": severity,
        "message": message,
        "suggested_fix": suggested_fix,
    }


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
