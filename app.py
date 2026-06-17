import plotly.express as px
import streamlit as st

from src.probability import calculate_probability
from src.ui_helpers import load_page_data
from src.validator import (
    find_duplicate_months,
    find_happening_benefits,
    find_missing_months,
)


st.set_page_config(
    page_title="RDO Benefits Tool - Overview",
    page_icon="📊",
    layout="wide",
)

st.title("RDO Benefits Tool")
st.caption("Overview of Red Dead Online Benefits data.")


try:
    page_data = load_page_data()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

filtered_df = page_data["filtered_df"]
analysis_df = page_data["analysis_df"]
selected_confidence = page_data["selected_confidence"]


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
    st.dataframe(duplicate_months, use_container_width=True, hide_index=True)

if missing_months:
    st.warning("Missing months found in the current month sequence.")
    st.write(", ".join(missing_months))
else:
    st.success("No missing months found in the current month sequence.")

if analysis_df.empty:
    st.info("No rows are available for analysis after excluding confidence = low.")


st.header("B. Raw Data")

st.dataframe(filtered_df, use_container_width=True, hide_index=True)

if selected_confidence == "all":
    st.caption(
        "Raw Data shows all confidence levels. Probability sections still exclude low-confidence rows."
    )


st.header("C. Primary Category Probability")

category_probability = calculate_probability(analysis_df, "primary_category")
st.dataframe(category_probability, use_container_width=True, hide_index=True)

if not category_probability.empty:
    category_fig = px.pie(
        category_probability,
        names="primary_category",
        values="probability_percent",
        title="Empirical Probability by Primary Category",
    )
    category_fig.update_traces(textinfo="percent+label")
    st.plotly_chart(category_fig, use_container_width=True)


st.header("D. Happening Benefits")

happening_df = find_happening_benefits(filtered_df)

if happening_df.empty:
    st.info("No rows with end_date = Happening in the current view.")
else:
    st.dataframe(happening_df, use_container_width=True, hide_index=True)
