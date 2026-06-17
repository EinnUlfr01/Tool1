import plotly.express as px
import streamlit as st

from src.adjusted_predictor import calculate_adjusted_primary_category_prediction
from src.conditional_predictor import (
    DEFAULT_COOLDOWN_CONFIG,
    calculate_conditional_sub_category_prediction_with_config,
)
from src.extra_tags_analyzer import calculate_extra_tag_probabilities
from src.global_predictor import calculate_global_path_prediction
from src.probability import (
    ALL_ROLE_ORDER,
    MIXED_ORDER,
    NON_ROLE_ORDER,
    ROLE_ORDER,
    SEASONAL_ORDER,
    collapse_to_top_n,
    order_sub_category_prediction_df,
)
from src.ui_helpers import load_page_data, run_parameter_optimizer


DEFAULT_PRIOR_STRENGTH = 3
PERCENT_TOLERANCE = 0.01

IMPORTANT_PRIMARY_SUB_CATEGORIES = [
    *ROLE_ORDER,
    *ALL_ROLE_ORDER,
    *NON_ROLE_ORDER,
    *SEASONAL_ORDER,
    *MIXED_ORDER,
]


st.set_page_config(
    page_title="RDO Benefits Tool - Next Month Prediction",
    page_icon="🎯",
    layout="wide",
)

st.title("Next Month Prediction")
st.caption(
    "Adjusted prediction score based on historical frequency, transition patterns, cooldown penalty, and backtesting."
)


def is_percent_close(value: float, expected: float = 100.0) -> bool:
    return abs(float(value) - expected) <= PERCENT_TOLERANCE


def validate_primary_prediction_total(primary_prediction_df):
    if (
        primary_prediction_df.empty
        or "adjusted_prediction_percent" not in primary_prediction_df.columns
    ):
        return

    total = primary_prediction_df["adjusted_prediction_percent"].sum()
    if not is_percent_close(total):
        st.warning(
            "Primary Category Prediction total is "
            f"{total:.4f}%, expected about 100%."
        )


def validate_sub_prediction_totals(primary_prediction_df, sub_prediction_df):
    required_primary_columns = ["primary_category", "adjusted_prediction_percent"]
    required_sub_columns = ["primary_category", "adjusted_prediction_percent"]
    if (
        primary_prediction_df.empty
        or sub_prediction_df.empty
        or any(column not in primary_prediction_df.columns for column in required_primary_columns)
        or any(column not in sub_prediction_df.columns for column in required_sub_columns)
    ):
        return

    sub_total = sub_prediction_df["adjusted_prediction_percent"].sum()
    if not is_percent_close(sub_total):
        st.warning(
            "Primary Sub-category Prediction total is "
            f"{sub_total:.4f}%, expected about 100%."
        )

    parent_totals = primary_prediction_df[required_primary_columns].rename(
        columns={"adjusted_prediction_percent": "parent_adjusted_prediction_percent"}
    )
    child_totals = (
        sub_prediction_df.groupby("primary_category", as_index=False)[
            "adjusted_prediction_percent"
        ]
        .sum()
        .rename(
            columns={
                "adjusted_prediction_percent": (
                    "child_adjusted_prediction_percent"
                )
            }
        )
    )
    comparison_df = parent_totals.merge(
        child_totals,
        on="primary_category",
        how="outer",
    ).fillna(0.0)
    comparison_df["difference"] = (
        comparison_df["child_adjusted_prediction_percent"]
        - comparison_df["parent_adjusted_prediction_percent"]
    ).abs()

    mismatched_categories = comparison_df[
        comparison_df["difference"] > PERCENT_TOLERANCE
    ]
    if not mismatched_categories.empty:
        st.warning(
            "Primary Sub-category Prediction does not match parent totals for: "
            + ", ".join(mismatched_categories["primary_category"].astype(str))
        )


try:
    page_data = load_page_data()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

filtered_df = page_data["filtered_df"]
prediction_analysis_df = page_data["full_history_analysis_df"]


st.header("A. Auto Backtesting & Parameter Optimization")

selected_prior_strength = DEFAULT_PRIOR_STRENGTH
selected_cooldown_config = DEFAULT_COOLDOWN_CONFIG.copy()

try:
    optimization_result = run_parameter_optimizer(prediction_analysis_df)
except Exception as exc:
    st.warning(
        f"Auto optimization failed. Using default parameters. Error: {exc}"
    )
    optimization_result = None

if optimization_result is None:
    st.info(
        "Using default parameters: prior_strength = 3, cooldown = {'recent_1': 0.2, 'recent_2': 0.4, 'recent_3': 0.6}."
    )
elif not optimization_result.enough_data:
    st.warning("Not enough historical data for backtesting. Using default parameters.")
    st.info(
        "Using default parameters: prior_strength = 3, cooldown = {'recent_1': 0.2, 'recent_2': 0.4, 'recent_3': 0.6}."
    )
else:
    selected_prior_strength = optimization_result.best_prior_strength
    selected_cooldown_config = optimization_result.best_cooldown_config

    st.subheader("Backtested Parameter Selection")
    st.dataframe(
        optimization_result.comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    best_config_df = {
        "best_prior_strength": [optimization_result.best_prior_strength],
        "best_cooldown_recent_1": [
            optimization_result.best_cooldown_config["recent_1"]
        ],
        "best_cooldown_recent_2": [
            optimization_result.best_cooldown_config["recent_2"]
        ],
        "best_cooldown_recent_3": [
            optimization_result.best_cooldown_config["recent_3"]
        ],
        "best_final_backtest_score": [
            round(optimization_result.best_final_backtest_score, 4)
        ],
    }
    st.dataframe(best_config_df, use_container_width=True, hide_index=True)
    st.caption("Auto-optimized parameters are being used for adjusted prediction.")


st.header("B. Adjusted Primary Category Prediction")

adjusted_prediction = calculate_adjusted_primary_category_prediction(
    prediction_analysis_df,
    prior_strength=selected_prior_strength,
)
adjusted_prediction_df = adjusted_prediction.table
validate_primary_prediction_total(adjusted_prediction_df)

if adjusted_prediction_df.empty:
    st.info("No rows are available for adjusted prediction after excluding confidence = low.")
else:
    prediction_col_1, prediction_col_2, prediction_col_3 = st.columns(3)
    prediction_col_1.metric("Latest month", adjusted_prediction.latest_month)
    prediction_col_2.metric(
        "Predicted next month",
        adjusted_prediction.predicted_next_month,
    )
    prediction_col_3.metric("Previous category", adjusted_prediction.previous_category)

    if adjusted_prediction.previous_category == "all_role":
        st.info(
            "Latest month is all_role. The tool applies the all_role domain adjustment to reduce repeated all_role/role outcomes and lift non_role/mixed branches."
        )

    st.dataframe(adjusted_prediction_df, use_container_width=True, hide_index=True)

    adjusted_fig = px.pie(
        adjusted_prediction_df,
        names="primary_category",
        values="adjusted_prediction_percent",
        title="Adjusted Prediction Score by Primary Category",
    )
    adjusted_fig.update_traces(textinfo="percent+label")
    st.plotly_chart(adjusted_fig, use_container_width=True)


st.header("C. Primary Sub-category Prediction")
st.caption(
    "Seasonal sub-categories such as halloween and holiday_call_to_arms are excluded from adjusted prediction when the predicted month is outside their usual season."
)

conditional_sub_prediction_df = calculate_conditional_sub_category_prediction_with_config(
    prediction_analysis_df,
    cooldown_config=selected_cooldown_config,
    parent_primary_prediction_df=adjusted_prediction_df,
    predicted_next_month=adjusted_prediction.predicted_next_month,
)
validate_sub_prediction_totals(adjusted_prediction_df, conditional_sub_prediction_df)

if conditional_sub_prediction_df.empty:
    st.info(
        "No rows are available for primary sub-category prediction after excluding confidence = low."
    )
else:
    st.dataframe(
        conditional_sub_prediction_df,
        use_container_width=True,
        hide_index=True,
    )

    conditional_sub_chart_df = collapse_to_top_n(
        conditional_sub_prediction_df,
        label_column="primary_sub_category",
        top_n=12,
        always_include=IMPORTANT_PRIMARY_SUB_CATEGORIES,
        value_column="adjusted_prediction_percent",
    )
    conditional_sub_chart_df = order_sub_category_prediction_df(
        conditional_sub_chart_df,
        predicted_next_month=adjusted_prediction.predicted_next_month,
        is_prediction=True,
    )

    if not conditional_sub_chart_df.empty:
        conditional_sub_fig = px.pie(
            conditional_sub_chart_df,
            names="primary_sub_category",
            values="adjusted_prediction_percent",
            title="Adjusted Prediction Score by Primary Sub-category",
        )
        conditional_sub_fig.update_traces(textinfo="percent+label", sort=False)
        st.plotly_chart(conditional_sub_fig, use_container_width=True)


st.header("D. Conditional Sub-category Prediction")

if conditional_sub_prediction_df.empty:
    st.info(
        "No rows are available for conditional sub-category prediction after excluding confidence = low."
    )
else:
    conditional_columns = [
        "primary_category",
        "primary_sub_category",
        "count",
        "conditional_base_percent",
        "cooldown_multiplier",
        "all_role_role_cooldown_multiplier",
        "adjusted_score",
        "normalized_sub_category_weight",
        "adjusted_conditional_sub_probability_percent",
    ]
    available_columns = [
        column
        for column in conditional_columns
        if column in conditional_sub_prediction_df.columns
    ]
    st.dataframe(
        conditional_sub_prediction_df[available_columns],
        use_container_width=True,
        hide_index=True,
    )


st.header("E. Global Path Prediction")

global_path_prediction_df = calculate_global_path_prediction(
    adjusted_prediction_df,
    conditional_sub_prediction_df,
)

if global_path_prediction_df.empty:
    st.info("No rows are available for global path prediction.")
else:
    st.dataframe(
        global_path_prediction_df,
        use_container_width=True,
        hide_index=True,
    )

    top_global_path_df = global_path_prediction_df.head(10)
    global_path_fig = px.bar(
        top_global_path_df,
        x="global_path_score",
        y="path_label",
        orientation="h",
        title="Top 10 Adjusted Global Path Scores",
    )
    global_path_fig.update_yaxes(
        categoryorder="array",
        categoryarray=top_global_path_df["path_label"].iloc[::-1].tolist(),
    )
    st.plotly_chart(global_path_fig, use_container_width=True)


st.header("F. Extra Tags Analysis")

if filtered_df.empty:
    st.info("No rows are available for extra tags analysis in the current view.")
else:
    tag_category_options = sorted(filtered_df["primary_category"].dropna().unique())

    tag_filter_col_1, tag_filter_col_2 = st.columns(2)
    with tag_filter_col_1:
        selected_tag_category = st.selectbox(
            "Extra tags primary_category",
            options=tag_category_options,
        )

    sub_category_options = sorted(
        filtered_df[filtered_df["primary_category"] == selected_tag_category][
            "primary_sub_category"
        ]
        .dropna()
        .unique()
    )

    with tag_filter_col_2:
        selected_tag_sub_category = st.selectbox(
            "Extra tags primary_sub_category",
            options=sub_category_options,
        )

    if selected_tag_category == "mixed":
        st.info(
            "mixed is treated as a terminal branch, so extra tag analysis is informational only."
        )

    extra_tag_probability_df = calculate_extra_tag_probabilities(
        filtered_df,
        selected_tag_category,
        selected_tag_sub_category,
    )

    if extra_tag_probability_df.empty:
        st.info(
            "No non-empty extra_tags were found for the selected primary_category and primary_sub_category."
        )
    else:
        st.dataframe(
            extra_tag_probability_df,
            use_container_width=True,
            hide_index=True,
        )
