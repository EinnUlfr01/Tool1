import plotly.express as px
import streamlit as st

from src.probability import calculate_probability
from src.seasonal_rules import get_seasonal_display_label
from src.ui_helpers import (
    build_all_benefits_table,
    build_simple_category_probability_table,
    load_page_data,
    render_debug_mode_toggle,
)
from src.validator import (
    find_duplicate_months,
    find_happening_benefits,
    find_missing_months,
    find_seasonal_quality_warnings,
)


st.set_page_config(
    page_title="RDO Benefits Tool - Overview",
    page_icon="📊",
    layout="wide",
)

st.title("RDO Benefits Tool")
st.caption("Overview of Red Dead Online Benefits data.")
debug_mode = render_debug_mode_toggle()


try:
    page_data = load_page_data()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

filtered_df = page_data["filtered_df"]
analysis_df = page_data["analysis_df"]


st.header("A. Data Quality Check")

duplicate_months = find_duplicate_months(filtered_df)
missing_months = find_missing_months(filtered_df)

quality_col_1, quality_col_2, quality_col_3 = st.columns(3)
quality_col_1.metric("Visible rows", len(filtered_df))
quality_col_2.metric("Analysis rows", len(analysis_df))
quality_col_3.metric("Missing months", len(missing_months))

if duplicate_months.empty:
    st.success("No duplicated month_label values found in the current view.")
else:
    st.warning("Duplicated month_label values found in the current view.")
    if debug_mode:
        st.dataframe(duplicate_months, width="stretch", hide_index=True)

if missing_months:
    st.warning("Missing months found in the current month sequence.")
    if debug_mode:
        st.write(", ".join(missing_months))
else:
    st.success("No missing months found in the current month sequence.")

if analysis_df.empty:
    st.info("No rows are available for analysis after excluding confidence = low.")

seasonal_warnings = find_seasonal_quality_warnings(analysis_df)
if seasonal_warnings.empty:
    st.success("No seasonal data quality warnings found.")
else:
    st.warning("Seasonal data quality warnings found.")
    st.dataframe(seasonal_warnings, width="stretch", hide_index=True)


st.header("B. All Benefits")
all_benefits_table = build_all_benefits_table(filtered_df)
st.dataframe(all_benefits_table, width="stretch", hide_index=True)


if debug_mode:
    st.header("C. Raw Data")

    source_columns = ["source_name", "source_url", "confidence"]
    raw_display_columns = [
        column for column in filtered_df.columns if column not in source_columns
    ] + [column for column in source_columns if column in filtered_df.columns]
    st.dataframe(
        filtered_df[raw_display_columns],
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "The confidence filter changes Raw Data visibility only. Confidence is source "
        "metadata and does not change probability weights."
    )


st.header("D. Historical Category Probability" if debug_mode else "C. Historical Category Probability")
st.caption("This is historical distribution from the CSV, not next-month prediction.")

category_probability = calculate_probability(
    analysis_df,
    "primary_category",
    weight_column="weight",
)
if debug_mode:
    st.dataframe(category_probability, width="stretch", hide_index=True)
else:
    st.dataframe(
        build_simple_category_probability_table(category_probability),
        width="stretch",
        hide_index=True,
    )

if not category_probability.empty:
    category_fig = px.pie(
        category_probability,
        names="primary_category",
        values="probability_percent",
        title="Historical Category Probability",
    )
    category_fig.update_traces(textinfo="percent+label")
    st.plotly_chart(category_fig, width="stretch")


if debug_mode:
    st.header("E. Seasonal Display Probability")
    st.caption(
        "Historical seasonal display probability from CSV metadata, not next-month prediction."
    )

    seasonal_df = analysis_df[
        analysis_df["is_seasonal"].eq(True)
        & analysis_df["seasonal_type"].astype(str).str.strip().ne("")
    ].copy()
    if seasonal_df.empty:
        st.info("No seasonal rows are available for analysis.")
    else:
        seasonal_df["seasonal_display"] = seasonal_df["seasonal_type"].apply(
            get_seasonal_display_label
        )
        seasonal_probability = calculate_probability(
            seasonal_df,
            "seasonal_display",
            weight_column="weight",
        )
        st.dataframe(seasonal_probability, width="stretch", hide_index=True)
        seasonal_fig = px.pie(
            seasonal_probability,
            names="seasonal_display",
            values="probability_percent",
            title="Seasonal Probability by Display Group",
        )
        seasonal_fig.update_traces(textinfo="percent+label", sort=False)
        st.plotly_chart(seasonal_fig, width="stretch")


st.header("F. Happening Benefits" if debug_mode else "D. Happening Benefits")

happening_df = find_happening_benefits(filtered_df)

if happening_df.empty:
    st.info("No rows with end_date = Happening in the current view.")
else:
    st.dataframe(happening_df, width="stretch", hide_index=True)
