import pandas as pd
import plotly.express as px
import streamlit as st

from src.adjusted_predictor import calculate_adjusted_primary_category_prediction
from src.conditional_predictor import (
    DEFAULT_COOLDOWN_CONFIG,
    calculate_conditional_sub_category_prediction_with_config,
)
from src.global_predictor import (
    calculate_final_global_component_prediction,
    calculate_global_path_prediction,
)
from src.optimization_cache import (
    get_cached_best_params,
    load_optimization_cache,
    save_optimization_cache,
)
from src.parameter_optimizer import optimize_parameters
from src.probability import (
    get_sub_category_display_label,
    order_sub_category_prediction_df,
)
from src.ui_helpers import load_page_data


DEFAULT_PRIOR_STRENGTH = 3
PERCENT_TOLERANCE = 0.01


st.set_page_config(
    page_title="RDO Benefits Tool - Next Month Prediction",
    page_icon="Target",
    layout="wide",
)

st.title("Next Month Prediction")
st.caption(
    "A practical weighted model using role/non_role/both history, monthly "
    "transitions, expanded components, and recent-role cooldown."
)


def is_percent_close(value: float, expected: float = 100.0) -> bool:
    return abs(float(value) - expected) <= PERCENT_TOLERANCE


def validate_prediction_totals(primary_df: pd.DataFrame, sub_df: pd.DataFrame) -> None:
    if primary_df.empty or sub_df.empty:
        return

    primary_total = float(primary_df["adjusted_prediction_percent"].sum())
    sub_total = float(sub_df["adjusted_prediction_percent"].sum())
    if not is_percent_close(primary_total):
        st.warning(f"Primary category total is {primary_total:.4f}%, expected 100%.")
    if not is_percent_close(sub_total):
        st.warning(f"Sub-category total is {sub_total:.4f}%, expected 100%.")

    parent_totals = primary_df.set_index("primary_category")[
        "adjusted_prediction_percent"
    ]
    child_totals = sub_df.groupby("primary_category")[
        "adjusted_prediction_percent"
    ].sum()
    mismatches = []
    for category in sorted(set(parent_totals.index) | set(child_totals.index)):
        difference = abs(
            float(parent_totals.get(category, 0.0))
            - float(child_totals.get(category, 0.0))
        )
        if difference > PERCENT_TOLERANCE:
            mismatches.append(category)
    if mismatches:
        st.warning("Parent/child prediction totals differ for: " + ", ".join(mismatches))


def validate_final_component_total(final_component_df: pd.DataFrame) -> None:
    if final_component_df.empty:
        return
    final_total = float(final_component_df["global_probability_percent"].sum())
    if not is_percent_close(final_total):
        st.warning(
            f"Final Global Component Prediction totals {final_total:.4f}%, "
            "expected 100%."
        )


def aggregate_component_predictions(
    prediction_df: pd.DataFrame,
    component_type: str,
    predicted_next_month: str,
) -> pd.DataFrame:
    component_df = prediction_df[
        prediction_df["component_type"] == component_type
    ].copy()
    if component_df.empty:
        return component_df

    aggregated_df = (
        component_df.groupby("primary_sub_category", as_index=False)
        .agg(
            count=("count", "sum"),
            historical_component_weight=("component_weight", "sum"),
            adjusted_prediction_percent=("adjusted_prediction_percent", "sum"),
        )
    )
    total = float(aggregated_df["adjusted_prediction_percent"].sum())
    aggregated_df["within_component_type_percent"] = (
        aggregated_df["adjusted_prediction_percent"] / total * 100
        if total > 0
        else 0.0
    )
    aggregated_df[
        ["historical_component_weight", "adjusted_prediction_percent", "within_component_type_percent"]
    ] = aggregated_df[
        ["historical_component_weight", "adjusted_prediction_percent", "within_component_type_percent"]
    ].round(4)
    aggregated_df["display_label"] = aggregated_df["primary_sub_category"].apply(
        get_sub_category_display_label
    )
    aggregated_df = aggregated_df.sort_values(
        "adjusted_prediction_percent",
        ascending=False,
    )
    return order_sub_category_prediction_df(
        aggregated_df,
        predicted_next_month=predicted_next_month,
        is_prediction=False,
    )


def render_component_prediction(
    title: str,
    component_df: pd.DataFrame,
) -> None:
    st.header(title)
    if component_df.empty:
        st.info("No expanded components are available for this prediction group.")
        return

    total_probability = float(component_df["adjusted_prediction_percent"].sum())
    st.caption(
        f"Combined contribution to the full next-month prediction: "
        f"{total_probability:.4f}%. Rows from mixed parent categories contribute "
        "to the matching component type."
    )
    st.dataframe(component_df, width="stretch", hide_index=True)

    figure = px.pie(
        component_df,
        names="display_label"
        if "display_label" in component_df.columns
        else "primary_sub_category",
        values="adjusted_prediction_percent",
        title=f"{title} composition",
    )
    figure.update_traces(textinfo="percent+label", sort=False)
    st.plotly_chart(figure, width="stretch")


try:
    page_data = load_page_data()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

prediction_analysis_df = page_data["full_history_analysis_df"]


st.header("A. Auto Backtesting & Parameter Optimization")

selected_prior_strength = DEFAULT_PRIOR_STRENGTH
selected_cooldown_config = DEFAULT_COOLDOWN_CONFIG.copy()
optimization_status = "default"
optimization_result = None
cache_state = load_optimization_cache()

if cache_state.is_valid:
    selected_prior_strength, selected_cooldown_config = get_cached_best_params(
        cache_state.cache
    )
    optimization_status = "cached"
else:
    optimization_status = cache_state.status
    st.warning(
        f"Optimization cache status: {cache_state.status}. "
        f"{cache_state.reason} Using default params until Run Optimization is pressed."
    )

run_optimization = st.button("Run Optimization")
if run_optimization:
    try:
        optimization_result = optimize_parameters(prediction_analysis_df)
        if optimization_result.enough_data:
            cache = save_optimization_cache(optimization_result)
            selected_prior_strength = optimization_result.best_prior_strength
            selected_cooldown_config = optimization_result.best_cooldown_config
            optimization_status = "newly optimized"
            st.success(
                "Optimization finished and cache was updated at "
                f"{cache['created_at']}."
            )
        else:
            optimization_status = "default"
            st.info("Not enough historical data for optimization. Using defaults.")
    except Exception as exc:
        optimization_status = "default"
        st.warning(f"Optimization failed. Using defaults. Error: {exc}")

st.metric("Optimization params", optimization_status)

if optimization_status == "cached" and cache_state.cache is not None:
    st.caption(
        "Using cached optimization params from "
        f"{cache_state.cache.get('created_at', 'unknown time')} "
        f"with score {float(cache_state.cache.get('score', 0.0)):.4f}."
    )
elif optimization_status in {"default", "stale"}:
    st.caption(
        "Using prior_strength=3 and cooldown={recent_1: 0.55, recent_2: 0.75, recent_3: 0.90}."
    )

if optimization_result is not None and optimization_result.enough_data:
    st.dataframe(
        optimization_result.comparison_table,
        width="stretch",
        hide_index=True,
    )
    best_col_1, best_col_2 = st.columns(2)
    best_col_1.metric("Best prior strength", selected_prior_strength)
    best_col_2.metric(
        "Best backtest score",
        f"{optimization_result.best_final_backtest_score:.4f}",
    )
    st.json(selected_cooldown_config)
else:
    st.json(selected_cooldown_config)


st.header("B. Primary Category Prediction")

adjusted_prediction = calculate_adjusted_primary_category_prediction(
    prediction_analysis_df,
    prior_strength=selected_prior_strength,
)
adjusted_prediction_df = adjusted_prediction.table

if adjusted_prediction_df.empty:
    st.info("No valid rows are available for prediction.")
    st.stop()

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Latest month", adjusted_prediction.latest_month)
metric_col_2.metric("Predicted next month", adjusted_prediction.predicted_next_month)
metric_col_3.metric("Previous category", adjusted_prediction.previous_category)

if adjusted_prediction.latest_is_all_role:
    st.info(
        "Latest month is marked is_all_role=TRUE. The tool keeps it under "
        "primary_category=role, but applies after-all-role rules: role is strongly "
        "penalized, non_role is strongly boosted, and both is moderately boosted. "
        "all_role is not treated as a standalone primary category."
    )

st.dataframe(adjusted_prediction_df, width="stretch", hide_index=True)
primary_figure = px.pie(
    adjusted_prediction_df,
    names="primary_category",
    values="adjusted_prediction_percent",
    title="Adjusted Prediction by Primary Category",
)
primary_figure.update_traces(textinfo="percent+label", sort=False)
st.plotly_chart(primary_figure, width="stretch")


conditional_prediction_df = calculate_conditional_sub_category_prediction_with_config(
    prediction_analysis_df,
    cooldown_config=selected_cooldown_config,
    parent_primary_prediction_df=adjusted_prediction_df,
    predicted_next_month=adjusted_prediction.predicted_next_month,
)
validate_prediction_totals(adjusted_prediction_df, conditional_prediction_df)

final_component_df = calculate_final_global_component_prediction(
    conditional_prediction_df
)
validate_final_component_total(final_component_df)

role_prediction_df = aggregate_component_predictions(
    conditional_prediction_df,
    "role",
    adjusted_prediction.predicted_next_month,
)
non_role_prediction_df = aggregate_component_predictions(
    conditional_prediction_df,
    "non_role",
    adjusted_prediction.predicted_next_month,
)

st.header("C. Final Global Component Prediction")
st.caption(
    "This is the main component result. It combines role and non-role components "
    "in one 100% probability system and merges the same component across parent paths."
)
if final_component_df.empty:
    st.info("No global component prediction is available.")
else:
    final_component_display_df = final_component_df.copy()
    final_component_display_df["display_label"] = final_component_display_df[
        "component"
    ].apply(get_sub_category_display_label)
    st.dataframe(final_component_display_df, width="stretch", hide_index=True)
    final_component_figure = px.pie(
        final_component_display_df,
        names="display_label",
        values="global_probability_percent",
        color="component_type",
        title="Final Global Component Probability",
    )
    final_component_figure.update_traces(textinfo="percent+label", sort=False)
    st.plotly_chart(final_component_figure, width="stretch")

render_component_prediction("D. Role Component Breakdown (Advanced)", role_prediction_df)
render_component_prediction(
    "E. Non-role Component Breakdown (Advanced)",
    non_role_prediction_df,
)


st.header("F. Calculation Details (Advanced)")
st.caption(
    "Each row has its historical weight. all_role_components, mixed_components, "
    "and pipe-separated primary_sub_category values are expanded and split equally. "
    "Component scores are normalized inside their source primary category, then "
    "multiplied by that category's adjusted prediction."
)

# TODO: Split this page into Simple Mode and Advanced / Debug Mode.
# Simple Mode should keep data summary, primary prediction, final component
# prediction, and happening benefits. Advanced Mode should keep multipliers,
# source_paths, applied_rules, breakdowns, and expanded path tables.

with st.expander("Expanded conditional prediction details"):
    st.dataframe(
        conditional_prediction_df,
        width="stretch",
        hide_index=True,
    )

global_path_prediction_df = calculate_global_path_prediction(
    adjusted_prediction_df,
    conditional_prediction_df,
)
with st.expander("Global category to component paths"):
    st.dataframe(
        global_path_prediction_df,
        width="stretch",
        hide_index=True,
    )
