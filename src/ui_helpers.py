import streamlit as st
import pandas as pd

from src.data_loader import (
    CONFIDENCE_OPTIONS,
    DATA_PATH,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    filter_by_confidence,
    filter_by_year,
    get_analysis_data,
    get_available_years,
    load_raw_benefits,
)
from src.probability import get_sub_category_display_label
from src.taxonomy import KNOWN_RARE_NON_ROLE_ALIASES


@st.cache_data
def load_cached_data():
    return load_raw_benefits(DATA_PATH)


def render_sidebar_filters(raw_df):
    with st.sidebar:
        st.header("Filters")

        available_years = get_available_years(raw_df)
        year_options = ["all"] + available_years
        selected_year = st.selectbox("Year", options=year_options, index=0)

        selected_confidence = st.selectbox(
            "Confidence",
            options=CONFIDENCE_OPTIONS,
            index=0,
        )

    return selected_year, selected_confidence


def prepare_filtered_data(raw_df, selected_year, selected_confidence):
    year_filtered_df = filter_by_year(raw_df, selected_year)
    filtered_df = filter_by_confidence(year_filtered_df, selected_confidence)
    analysis_df = get_analysis_data(year_filtered_df)
    full_history_analysis_df = get_analysis_data(raw_df)

    return filtered_df, analysis_df, full_history_analysis_df


def load_page_data():
    raw_df = load_cached_data()
    selected_year, selected_confidence = render_sidebar_filters(raw_df)
    filtered_df, analysis_df, full_history_analysis_df = prepare_filtered_data(
        raw_df,
        selected_year,
        selected_confidence,
    )

    return {
        "raw_df": raw_df,
        "selected_confidence": selected_confidence,
        "filtered_df": filtered_df,
        "analysis_df": analysis_df,
        "full_history_analysis_df": full_history_analysis_df,
    }


def render_debug_mode_toggle() -> bool:
    debug_mode = st.toggle("Debug Mode", value=False)
    st.caption("Debug Mode" if debug_mode else "Simple Mode")
    return debug_mode


def format_label(value: object) -> str:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return "-"
    return normalized_value.replace("_", " ").replace("|", " / ").title()


def format_category_label(value: object) -> str:
    category_labels = {
        "role": "Role",
        "non_role": "Non-role",
        "both": "Both",
    }
    normalized_value = str(value or "").strip().lower()
    return category_labels.get(normalized_value, format_label(normalized_value))


def format_component_label(value: object) -> str:
    normalized_value = str(value or "").strip().lower()
    if not normalized_value:
        return "-"
    return get_sub_category_display_label(normalized_value)


def format_raw_component_label(value: object) -> str:
    return format_component_label(value)


def format_component_list_label(value: object) -> str:
    components = [
        format_raw_component_label(component)
        for component in str(value or "").split("|")
        if str(component).strip()
    ]
    return " / ".join(components) if components else "-"


def build_raw_data_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = REQUIRED_COLUMNS + [
        column for column in OPTIONAL_COLUMNS if column in df.columns
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    display_df = df.copy()
    raw_field_map = {
        "primary_category": "raw_primary_category",
        "primary_sub_category": "raw_primary_sub_category",
        "mixed_components": "raw_mixed_components",
        "all_role_components": "raw_all_role_components",
        "seasonal_type": "raw_seasonal_type",
    }
    for column, raw_column in raw_field_map.items():
        if column in display_df.columns and raw_column in display_df.columns:
            display_df[column] = display_df[raw_column]

    visible_columns = [column for column in columns if column in display_df.columns]
    return display_df[visible_columns].reset_index(drop=True)


def build_all_benefits_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Month", "Benefit Type", "Benefits Main Info", "Note", "Source"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    display_df = df.copy()
    display_df["Month"] = display_df.get("month_label", "").astype(str)
    display_df["Benefit Type"] = display_df.apply(
        lambda row: build_benefit_type_label(row),
        axis=1,
    )
    display_df["Benefits Main Info"] = display_df.apply(
        lambda row: first_non_empty(
            row.get("multiplier_info", ""),
            default="-",
        ),
        axis=1,
    )
    note_values = (
        display_df["note"]
        if "note" in display_df.columns
        else pd.Series("", index=display_df.index)
    )
    display_df["Note"] = note_values.apply(
        lambda value: first_non_empty(value, default="-")
    )
    display_df["Source"] = display_df.apply(
        lambda row: first_non_empty(
            row.get("source_name", ""),
            row.get("source_url", ""),
            default="-",
        ),
        axis=1,
    )

    month_sort = pd.to_datetime(
        display_df.get("month_label", ""),
        format="%Y-%m",
        errors="coerce",
    )
    display_df["_month_sort"] = month_sort
    display_df = display_df.sort_values(
        ["_month_sort", "Month"],
        ascending=[False, False],
        na_position="last",
    )
    return display_df[columns].reset_index(drop=True)


def build_other_non_role_mapping_table() -> pd.DataFrame:
    mapping_rows = [
        ("story_missions", "Story Missions"),
        ("gang_hideouts", "Gang Hideouts"),
        ("showdown", "Showdown"),
    ]
    return pd.DataFrame(
        [
            {
                "Raw Type": display_label,
                "Counted As": get_sub_category_display_label(
                    KNOWN_RARE_NON_ROLE_ALIASES[raw_value]
                ),
            }
            for raw_value, display_label in mapping_rows
        ]
    )


def build_benefit_type_label(row: pd.Series) -> str:
    sub_category = first_non_empty(
        row.get("raw_primary_sub_category", ""),
        row.get("raw_mixed_components", "") if bool(row.get("is_mixed", False)) else "",
        row.get("primary_sub_category", ""),
        default="",
    )
    if sub_category:
        return format_component_list_label(sub_category)
    primary_category = first_non_empty(
        row.get("raw_primary_category", ""),
        row.get("primary_category", ""),
        default="",
    )
    return format_category_label(primary_category)


def first_non_empty(*values: object, default: str = "-") -> str:
    for value in values:
        normalized_value = str(value or "").strip()
        if normalized_value:
            return normalized_value
    return default


def build_simple_category_probability_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Category", "Historical Probability %", "Count"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    display_df = df.copy()
    display_df["Category"] = display_df["primary_category"].apply(
        format_category_label
    )
    display_df["Historical Probability %"] = display_df[
        "probability_percent"
    ].map(lambda value: f"{float(value):.2f}%")
    display_df["Count"] = display_df["count"].astype(int)
    return display_df[columns]


def build_simple_primary_prediction_table(
    df: pd.DataFrame,
    latest_is_all_role: bool,
) -> pd.DataFrame:
    columns = ["Category", "Probability %", "Reason"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    display_df = df.copy()
    display_df["Category"] = display_df["primary_category"].apply(
        format_category_label
    )
    display_df["Probability %"] = display_df[
        "adjusted_prediction_percent"
    ].map(lambda value: f"{float(value):.2f}%")
    display_df["Reason"] = display_df["primary_category"].apply(
        lambda category: build_primary_reason(category, latest_is_all_role)
    )
    return display_df[columns]


def build_primary_reason(category: object, latest_is_all_role: bool) -> str:
    if latest_is_all_role:
        reason_map = {
            "role": "Penalized after all-role",
            "non_role": "Boosted after all-role",
            "both": "Moderately boosted after all-role",
        }
        return reason_map.get(str(category), "Based on historical and transition pattern")
    return "Based on historical and transition pattern"


def build_simple_final_component_table(
    df: pd.DataFrame,
    latest_is_all_role: bool,
) -> pd.DataFrame:
    columns = ["Rank", "Component", "Type", "Probability %", "Reason"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    display_df = df.sort_values(
        "global_probability_percent",
        ascending=False,
    ).reset_index(drop=True).copy()
    display_df["Rank"] = display_df.index + 1
    display_df["Component"] = display_df.apply(
        lambda row: first_non_empty(
            row.get("display_label", ""),
            format_component_label(row.get("component", "")),
        ),
        axis=1,
    )
    display_df["Type"] = display_df["component_type"].apply(format_category_label)
    display_df["Probability %"] = display_df[
        "global_probability_percent"
    ].map(lambda value: f"{float(value):.2f}%")
    display_df["Reason"] = display_df.apply(
        lambda row: build_simple_component_reason(row, latest_is_all_role),
        axis=1,
    )
    return display_df[columns]


def build_simple_component_reason(
    row: pd.Series,
    latest_is_all_role: bool,
) -> str:
    reasons = []
    applied_rules = str(row.get("applied_rules", "") or "")
    source_paths = str(row.get("source_paths", "") or "")
    component_type = str(row.get("component_type", "") or "").lower()

    if "after_all_role_component" in applied_rules:
        reasons.append("Penalized after all-role")
    elif latest_is_all_role and component_type == "non_role":
        reasons.append("Boosted after all-role")
    if "role_cooldown" in applied_rules:
        reasons.append("Recent role cooldown")
    if "non_role_cooldown" in applied_rules:
        reasons.append("Recent non-role cooldown")
    if "non_role_long_gap" in applied_rules:
        reasons.append("Long time since last seen")
    if "in_season" in applied_rules:
        reasons.append("Seasonal timing")
    if "out_of_season" in applied_rules:
        reasons.append("Seasonal timing")
    if "|" in source_paths:
        reasons.append("Appears from multiple paths")

    if not reasons:
        reasons.append("Based on historical pattern")
    return "; ".join(reasons[:3])
