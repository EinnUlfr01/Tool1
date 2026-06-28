from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data_loader import DATA_PATH, load_raw_benefits
from src.seasonal_rules import (
    SEASONAL_COMPONENT_MONTHS,
    SEASONAL_TYPE_MONTHS,
    parse_predicted_month,
)
from src.taxonomy import VALID_SEASONAL_COMPONENTS
from src.validator import find_happening_benefits


ROLE_OPTIONS = [
    "bounty_hunter",
    "trader",
    "collector",
    "moonshiner",
    "naturalist",
]
NON_ROLE_OPTIONS = [
    "free_roam",
    "races",
    "blood_money",
    "telegram",
    "call_to_arms",
    "featured_series",
    "story_missions",
    "gang_hideouts",
    "showdown",
    "other_non_role",
]
SEASONAL_OPTIONS = [
    "halloween",
    "halloween_call_to_arms",
    "holiday",
    "holiday_call_to_arms",
    "holiday_rewards",
]
PRIMARY_CATEGORY_OPTIONS = ["role", "non_role", "both", "seasonal"]
SEASONAL_TYPE_OPTIONS = [
    "halloween",
    "holiday",
    "thanksgiving",
    "valentines",
    "easter",
]
HARD_SEASONAL_TYPES = {
    "halloween": "halloween",
    "halloween_call_to_arms": "halloween",
    "holiday": "holiday",
    "holiday_call_to_arms": "holiday",
    "holiday_rewards": "holiday",
}


@dataclass(frozen=True)
class TopComponent:
    component: str
    label: str
    component_type: str
    primary_category: str
    seasonal_type: str
    requires_raw_non_role: bool = False


@dataclass(frozen=True)
class CsvWriteResult:
    csv_path: Path
    backup_path: Path
    changed_rows: int
    appended_rows: int


def get_happening_rows(df: pd.DataFrame) -> pd.DataFrame:
    return find_happening_benefits(df)


def get_next_month_label(df: pd.DataFrame) -> str:
    if df.empty or "month_label" not in df.columns:
        raise ValueError("Cannot infer next month from an empty CSV.")

    month_values = pd.to_datetime(df["month_label"], format="%Y-%m", errors="coerce")
    latest_month = month_values.max()
    if pd.isna(latest_month):
        raise ValueError("Cannot infer next month because month_label values are invalid.")

    return str(latest_month.to_period("M") + 1)


def infer_component_category(component: str, component_type: str = "") -> str:
    normalized_component = str(component or "").strip().lower()
    normalized_type = str(component_type or "").strip().lower()
    if normalized_component in VALID_SEASONAL_COMPONENTS:
        return "seasonal"
    if normalized_type in {"role", "non_role"}:
        return normalized_type
    if normalized_component in ROLE_OPTIONS:
        return "role"
    return "non_role"


def get_top_component(final_component_df: pd.DataFrame) -> TopComponent | None:
    if final_component_df.empty or "component" not in final_component_df.columns:
        return None

    sorted_df = final_component_df.copy()
    if "rank" in sorted_df.columns:
        sorted_df = sorted_df.sort_values("rank", ascending=True)
    elif "Rank" in sorted_df.columns:
        sorted_df = sorted_df.sort_values("Rank", ascending=True)
    elif "global_probability_percent" in sorted_df.columns:
        sorted_df = sorted_df.sort_values("global_probability_percent", ascending=False)

    row = sorted_df.iloc[0]
    component = str(row.get("component", "")).strip().lower()
    label = str(row.get("display_label", "") or component.replace("_", " ").title())
    component_type = str(row.get("component_type", "")).strip().lower()
    primary_category = infer_component_category(component, component_type)
    seasonal_type = HARD_SEASONAL_TYPES.get(component, "")
    return TopComponent(
        component=component,
        label=label,
        component_type=component_type,
        primary_category=primary_category,
        seasonal_type=seasonal_type,
        requires_raw_non_role=component == "other_non_role",
    )


def get_subcategory_options(primary_category: str) -> list[str]:
    normalized_category = str(primary_category or "").strip().lower()
    if normalized_category == "role":
        return ROLE_OPTIONS.copy()
    if normalized_category == "non_role":
        return NON_ROLE_OPTIONS.copy()
    if normalized_category in {"both", "seasonal"}:
        return ROLE_OPTIONS + NON_ROLE_OPTIONS + SEASONAL_OPTIONS
    return []


def is_seasonal_entry_allowed(
    month_label: str,
    top_component: str = "",
    unlock_confirmed: bool = False,
) -> bool:
    normalized_top = str(top_component or "").strip().lower()
    if normalized_top in SEASONAL_COMPONENT_MONTHS:
        month = parse_predicted_month(month_label)
        return month in SEASONAL_COMPONENT_MONTHS[normalized_top]

    month = parse_predicted_month(month_label)
    if month is not None:
        allowed_months = set().union(*SEASONAL_TYPE_MONTHS.values())
        if month in allowed_months:
            return True

    return bool(unlock_confirmed)


def build_new_month_row(
    existing_columns: Iterable[str],
    month_label: str,
    start_date: date,
    primary_category: str,
    primary_sub_categories: list[str],
    multiplier_info: str = "",
    is_seasonal: bool = False,
    seasonal_type: str = "",
    note: str = "",
    source_name: str = "",
    source_url: str = "",
    confidence: str = "high",
    weight: float = 1.0,
    component_weight_rule: str = "equal_split",
) -> dict[str, str]:
    normalized_category = str(primary_category or "").strip().lower()
    selected_components = unique_clean_values(primary_sub_categories)
    if not month_label:
        raise ValueError("month_label is required.")
    if normalized_category not in PRIMARY_CATEGORY_OPTIONS:
        raise ValueError("primary_category must be role, non_role, both, or seasonal.")
    if not selected_components:
        raise ValueError("At least one primary_sub_category is required.")
    if "other_non_role" in selected_components and len(selected_components) == 1:
        raise ValueError(
            "other_non_role cannot be the only selected sub-category. Choose a specific raw non-role type when known."
        )

    primary_sub_category = "|".join(selected_components)
    is_mixed = normalized_category == "both"
    is_all_role = set(selected_components) == set(ROLE_OPTIONS)
    row = {column: "" for column in existing_columns}
    row.update(
        {
            "month_label": month_label,
            "start_date": format_csv_date(start_date),
            "end_date": "Happening",
            "primary_category": normalized_category,
            "primary_sub_category": primary_sub_category,
            "multiplier_info": str(multiplier_info or "").strip(),
            "is_seasonal": bool_to_csv(is_seasonal or normalized_category == "seasonal"),
            "seasonal_type": str(seasonal_type or "").strip().lower()
            if (is_seasonal or normalized_category == "seasonal")
            else "",
            "is_mixed": bool_to_csv(is_mixed),
            "mixed_components": primary_sub_category if is_mixed else "",
            "is_all_role": bool_to_csv(is_all_role),
            "all_role_components": "|".join(ROLE_OPTIONS) if is_all_role else "",
            "weight": format_weight(weight),
            "component_weight_rule": component_weight_rule,
            "note": str(note or "").strip(),
            "source_name": str(source_name or "").strip(),
            "source_url": str(source_url or "").strip(),
            "confidence": str(confidence or "high").strip().lower(),
        }
    )
    return row


def build_close_preview(df: pd.DataFrame, end_date: date) -> pd.DataFrame:
    happening_rows = get_happening_rows(df)
    if happening_rows.empty:
        return pd.DataFrame()
    preview = happening_rows.copy()
    preview["new_end_date"] = format_csv_date(end_date)
    return preview


def build_add_preview(
    df: pd.DataFrame,
    new_row: dict[str, str],
    close_current: bool,
    close_end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_preview = (
        build_close_preview(df, close_end_date)
        if close_current
        else pd.DataFrame()
    )
    return close_preview, pd.DataFrame([new_row])


def create_backup_path(csv_path: str | Path = DATA_PATH) -> Path:
    source_path = Path(csv_path)
    backup_dir = source_path.parent.parent / "backups"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return backup_dir / f"{source_path.stem}_{timestamp}{source_path.suffix}"


def backup_csv(csv_path: str | Path = DATA_PATH, backup_path: str | Path | None = None) -> Path:
    source_path = Path(csv_path)
    if not source_path.exists():
        raise ValueError(f"CSV file was not found: {source_path}")
    backup_path = Path(backup_path) if backup_path is not None else create_backup_path(source_path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, backup_path)
    return backup_path


def close_happening_benefits(
    csv_path: str | Path,
    end_date: date,
    backup_path: str | Path | None = None,
) -> CsvWriteResult:
    csv_path = Path(csv_path)
    raw_df = pd.read_csv(csv_path, dtype=str).fillna("")
    happening_mask = raw_df["end_date"].astype(str).str.strip().str.lower().eq("happening")
    changed_rows = int(happening_mask.sum())
    backup_path = backup_csv(csv_path, backup_path)
    if changed_rows:
        raw_df.loc[happening_mask, "end_date"] = format_csv_date(end_date)
        raw_df.to_csv(csv_path, index=False)
    return CsvWriteResult(csv_path, backup_path, changed_rows, 0)


def append_new_month_benefit(
    csv_path: str | Path,
    row: dict[str, str],
    backup_path: str | Path | None = None,
) -> CsvWriteResult:
    return apply_monthly_update(csv_path, None, row, backup_path)


def apply_monthly_update(
    csv_path: str | Path,
    close_end_date: date | None,
    new_row: dict[str, str] | None,
    backup_path: str | Path | None = None,
) -> CsvWriteResult:
    csv_path = Path(csv_path)
    raw_df = pd.read_csv(csv_path, dtype=str).fillna("")

    changed_rows = 0
    appended_rows = 0
    if new_row is not None:
        month_label = str(new_row.get("month_label", "")).strip()
        if raw_df["month_label"].astype(str).str.strip().eq(month_label).any():
            raise ValueError("Month already exists in CSV.")

    backup_path = backup_csv(csv_path, backup_path)
    if close_end_date is not None:
        happening_mask = raw_df["end_date"].astype(str).str.strip().str.lower().eq("happening")
        changed_rows = int(happening_mask.sum())
        if changed_rows:
            raw_df.loc[happening_mask, "end_date"] = format_csv_date(close_end_date)

    if new_row is not None:
        ordered_row = {column: str(new_row.get(column, "")) for column in raw_df.columns}
        raw_df = pd.concat([raw_df, pd.DataFrame([ordered_row])], ignore_index=True)
        appended_rows = 1

    raw_df.to_csv(csv_path, index=False)
    return CsvWriteResult(csv_path, backup_path, changed_rows, appended_rows)


def validate_csv_after_write(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.validator import find_seasonal_quality_warnings, validate_sub_category_data

    reloaded_df = load_raw_benefits(csv_path)
    return validate_sub_category_data(reloaded_df), find_seasonal_quality_warnings(reloaded_df)


def unique_clean_values(values: Iterable[str]) -> list[str]:
    cleaned = []
    seen = set()
    for value in values:
        normalized = str(value or "").strip().lower()
        if normalized and normalized not in seen:
            cleaned.append(normalized)
            seen.add(normalized)
    return cleaned


def bool_to_csv(value: bool) -> str:
    return "TRUE" if bool(value) else "FALSE"


def format_weight(value: float) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(numeric_value)


def format_csv_date(value: date) -> str:
    return f"{value.month}/{value.day}/{value.year}"


def default_close_end_date(start_date: date) -> date:
    return start_date - timedelta(days=1)
