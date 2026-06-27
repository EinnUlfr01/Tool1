import streamlit as st

from src.data_loader import (
    CONFIDENCE_OPTIONS,
    DATA_PATH,
    filter_by_confidence,
    filter_by_year,
    get_analysis_data,
    get_available_years,
    load_raw_benefits,
)

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
