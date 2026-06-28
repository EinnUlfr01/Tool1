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
from src.data_loader import DATA_PATH
from src.monthly_update import (
    PRIMARY_CATEGORY_OPTIONS,
    SEASONAL_TYPE_OPTIONS,
    TopComponent,
    apply_monthly_update,
    build_add_preview,
    build_close_preview,
    build_new_month_row,
    create_backup_path,
    default_close_end_date,
    get_happening_rows,
    get_next_month_label,
    get_subcategory_options,
    get_top_component,
    is_seasonal_entry_allowed,
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
from src.ui_helpers import (
    build_other_non_role_mapping_table,
    build_simple_final_component_table,
    build_simple_primary_prediction_table,
    load_page_data,
    render_debug_mode_toggle,
)


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
debug_mode = render_debug_mode_toggle()


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


def render_monthly_update_section(
    raw_df: pd.DataFrame,
    final_component_df: pd.DataFrame,
    debug_mode: bool,
) -> None:
    st.header("Monthly Update")
    st.caption(
        "Close ongoing benefits or append one new monthly CSV row after preview, confirmation, and backup."
    )

    top_component = get_top_component(final_component_df)
    action_col_1, action_col_2 = st.columns(2)
    if action_col_1.button("Close Current Benefits"):
        st.session_state["monthly_update_action"] = "close"
        st.session_state.pop("monthly_close_preview", None)
    if action_col_2.button("Add New Month Benefits"):
        st.session_state["monthly_update_action"] = "add"
        st.session_state.pop("monthly_add_preview", None)

    action = st.session_state.get("monthly_update_action", "")
    if action == "close":
        render_close_current_benefits(raw_df)
    elif action == "add":
        render_add_new_month_benefits(raw_df, top_component, debug_mode)


def render_close_current_benefits(raw_df: pd.DataFrame) -> None:
    with st.expander("Close Current Benefits", expanded=True):
        happening_rows = get_happening_rows(raw_df)
        if happening_rows.empty:
            st.info("No Happening benefits found.")
            return

        st.dataframe(happening_rows, width="stretch", hide_index=True)
        close_end_date = st.date_input(
            "End date",
            value=pd.Timestamp.today().date(),
            key="monthly_close_end_date",
        )
        if st.button("Preview Close Current Benefits"):
            backup_path = create_backup_path(DATA_PATH)
            st.session_state["monthly_close_preview"] = {
                "end_date": close_end_date,
                "backup_path": str(backup_path),
            }

        preview_state = st.session_state.get("monthly_close_preview")
        if not preview_state:
            return

        preview_df = build_close_preview(raw_df, preview_state["end_date"])
        st.write(f"CSV path: `{DATA_PATH}`")
        st.write(f"Backup path: `{preview_state['backup_path']}`")
        st.dataframe(preview_df, width="stretch", hide_index=True)
        confirmed = st.checkbox(
            "I confirm updating current Happening benefits end_date.",
            key="monthly_close_confirm",
        )
        if st.button("Save Close Current Benefits", disabled=not confirmed):
            try:
                result = apply_monthly_update(
                    DATA_PATH,
                    preview_state["end_date"],
                    None,
                    preview_state["backup_path"],
                )
                show_post_write_validation(result)
                st.session_state.pop("monthly_close_preview", None)
            except Exception as exc:
                st.error(f"Close Current Benefits failed: {exc}")


def render_add_new_month_benefits(
    raw_df: pd.DataFrame,
    top_component: TopComponent | None,
    debug_mode: bool,
) -> None:
    with st.expander("Add New Month Benefits", expanded=True):
        next_month_label = get_next_month_label(raw_df)
        default_start_date = pd.to_datetime(
            f"{next_month_label}-01",
            format="%Y-%m-%d",
        ).date()
        if top_component is None:
            st.info("No Final Global prediction is available. Enter the month manually.")
        else:
            st.write(f"Top Final Global prediction: **{top_component.label}**")

        with st.form("monthly_add_form"):
            use_top_prediction = "No / Enter manually"
            if top_component is not None:
                use_top_prediction = st.radio(
                    f"Is {top_component.label} the new month benefit?",
                    ["Yes", "No / Enter manually"],
                    horizontal=True,
                )
            start_date = st.date_input("Start date", value=default_start_date)
            close_current = st.checkbox(
                "Close current Happening benefits using start_date - 1",
                value=True,
            )

            locked_category = ""
            if top_component is not None and use_top_prediction == "Yes":
                locked_category = top_component.primary_category
                st.text_input(
                    "primary_category",
                    value=locked_category,
                    disabled=True,
                )
                if top_component.requires_raw_non_role:
                    st.warning(
                        "Top prediction is Other Non-role. Please choose a specific raw non-role type."
                    )
            unlock_seasonal = False
            unlock_seasonal_confirm = False
            if not locked_category:
                seasonal_allowed = is_seasonal_entry_allowed(next_month_label)
                if not seasonal_allowed:
                    unlock_seasonal = st.checkbox("Unlock seasonal entry")
                    if unlock_seasonal:
                        unlock_seasonal_confirm = st.checkbox(
                            "I confirm this is a new/exceptional seasonal benefit."
                        )
                category_options = [
                    category
                    for category in PRIMARY_CATEGORY_OPTIONS
                    if category != "seasonal"
                    or seasonal_allowed
                    or (unlock_seasonal and unlock_seasonal_confirm)
                ]
                primary_category = st.selectbox("primary_category", category_options)
            else:
                primary_category = locked_category

            options = get_subcategory_options(primary_category)
            default_components = default_selected_components(top_component, primary_category)
            primary_sub_categories = st.multiselect(
                "primary_sub_category",
                options=options,
                default=[value for value in default_components if value in options],
            )
            if "other_non_role" in primary_sub_categories:
                st.warning(
                    "Use other_non_role only when the specific raw non-role is unknown."
                )

            is_seasonal_default = primary_category == "seasonal" or (
                top_component is not None
                and use_top_prediction == "Yes"
                and bool(top_component.seasonal_type)
            )
            is_seasonal = st.checkbox("is_seasonal", value=is_seasonal_default)
            seasonal_type = ""
            if is_seasonal or primary_category == "seasonal":
                seasonal_type_default = default_seasonal_type(top_component)
                seasonal_type = st.selectbox(
                    "seasonal_type",
                    SEASONAL_TYPE_OPTIONS,
                    index=SEASONAL_TYPE_OPTIONS.index(seasonal_type_default)
                    if seasonal_type_default in SEASONAL_TYPE_OPTIONS
                    else 0,
                )

            multiplier_info = st.text_area("multiplier_info")
            note = st.text_area("note")
            source_name = st.text_input("source_name")
            source_url = st.text_input("source_url")
            confidence = st.selectbox("confidence", ["high", "medium", "low"], index=0)
            weight = 1.0
            component_weight_rule = "equal_split"
            if debug_mode:
                weight = st.number_input("weight", min_value=0.0, value=1.0, step=0.1)
                component_weight_rule = st.text_input(
                    "component_weight_rule",
                    value="equal_split",
                )
            preview_requested = st.form_submit_button("Preview Add New Month Benefits")

        if preview_requested:
            try:
                new_row = build_new_month_row(
                    pd.read_csv(DATA_PATH, dtype=str).columns,
                    month_label=next_month_label,
                    start_date=start_date,
                    primary_category=primary_category,
                    primary_sub_categories=primary_sub_categories,
                    multiplier_info=multiplier_info,
                    is_seasonal=is_seasonal,
                    seasonal_type=seasonal_type,
                    note=note,
                    source_name=source_name,
                    source_url=source_url,
                    confidence=confidence,
                    weight=weight,
                    component_weight_rule=component_weight_rule,
                )
                if raw_df["month_label"].astype(str).str.strip().eq(next_month_label).any():
                    st.error("Month already exists in CSV.")
                    return
                close_end_date = default_close_end_date(start_date)
                close_preview, new_row_preview = build_add_preview(
                    raw_df,
                    new_row,
                    close_current,
                    close_end_date,
                )
                st.session_state["monthly_add_preview"] = {
                    "new_row": new_row,
                    "close_current": close_current,
                    "close_end_date": close_end_date,
                    "backup_path": str(create_backup_path(DATA_PATH)),
                }
                st.session_state["monthly_add_preview_tables"] = {
                    "close_preview": close_preview,
                    "new_row_preview": new_row_preview,
                }
            except Exception as exc:
                st.error(f"Preview failed: {exc}")

        preview_state = st.session_state.get("monthly_add_preview")
        preview_tables = st.session_state.get("monthly_add_preview_tables", {})
        if not preview_state:
            return

        st.write(f"CSV path: `{DATA_PATH}`")
        st.write(f"Backup path: `{preview_state['backup_path']}`")
        close_preview = preview_tables.get("close_preview", pd.DataFrame())
        if preview_state["close_current"]:
            if close_preview.empty:
                st.info("No Happening benefits found. New month row will still be appended.")
            else:
                st.subheader("Rows to close")
                st.dataframe(close_preview, width="stretch", hide_index=True)
        st.subheader("New row to append")
        st.dataframe(preview_tables["new_row_preview"], width="stretch", hide_index=True)
        confirmed = st.checkbox(
            "I confirm appending this new month benefit row.",
            key="monthly_add_confirm",
        )
        if st.button("Save Add New Month Benefits", disabled=not confirmed):
            try:
                result = apply_monthly_update(
                    DATA_PATH,
                    preview_state["close_end_date"]
                    if preview_state["close_current"]
                    else None,
                    preview_state["new_row"],
                    preview_state["backup_path"],
                )
                show_post_write_validation(result)
                st.session_state.pop("monthly_add_preview", None)
                st.session_state.pop("monthly_add_preview_tables", None)
            except Exception as exc:
                st.error(f"Add New Month Benefits failed: {exc}")


def default_selected_components(
    top_component: TopComponent | None,
    primary_category: str,
) -> list[str]:
    if top_component is None:
        return []
    if top_component.requires_raw_non_role:
        return []
    if primary_category == top_component.primary_category:
        return [top_component.component]
    return []


def default_seasonal_type(top_component: TopComponent | None) -> str:
    if top_component is not None and top_component.seasonal_type:
        return top_component.seasonal_type
    return "halloween"


def show_post_write_validation(result) -> None:
    from src.monthly_update import validate_csv_after_write

    validation_report, seasonal_warnings = validate_csv_after_write(result.csv_path)
    st.cache_data.clear()
    error_report = (
        validation_report[validation_report["severity"].eq("error")]
        if not validation_report.empty
        else pd.DataFrame()
    )
    if not error_report.empty:
        st.error("CSV was written, but validation found errors.")
        st.dataframe(error_report, width="stretch", hide_index=True)
        return
    st.success(
        f"CSV updated. Backup: {result.backup_path}. "
        f"Closed rows: {result.changed_rows}. Appended rows: {result.appended_rows}."
    )
    warning_report = (
        validation_report[validation_report["severity"].eq("warning")]
        if not validation_report.empty
        else pd.DataFrame()
    )
    if not warning_report.empty:
        st.warning("Validation warnings were found.")
        st.dataframe(warning_report, width="stretch", hide_index=True)
    if not seasonal_warnings.empty:
        st.warning("Seasonal quality warnings were found.")
        st.dataframe(seasonal_warnings, width="stretch", hide_index=True)


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
    best_col_1, best_col_2 = st.columns(2)
    best_col_1.metric("Best prior strength", selected_prior_strength)
    best_col_2.metric(
        "Best backtest score",
        f"{optimization_result.best_final_backtest_score:.4f}",
    )
    if debug_mode:
        st.dataframe(
            optimization_result.comparison_table,
            width="stretch",
            hide_index=True,
        )
        st.json(selected_cooldown_config)
else:
    if debug_mode:
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

if debug_mode:
    st.dataframe(adjusted_prediction_df, width="stretch", hide_index=True)
else:
    st.dataframe(
        build_simple_primary_prediction_table(
            adjusted_prediction_df,
            adjusted_prediction.latest_is_all_role,
        ),
        width="stretch",
        hide_index=True,
    )
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
    if debug_mode:
        st.dataframe(final_component_display_df, width="stretch", hide_index=True)
    else:
        st.dataframe(
            build_simple_final_component_table(
                final_component_display_df,
                adjusted_prediction.latest_is_all_role,
            ),
            width="stretch",
            hide_index=True,
        )
    final_component_figure = px.pie(
        final_component_display_df,
        names="display_label",
        values="global_probability_percent",
        color="component_type",
        title="Final Global Component Probability",
    )
    final_component_figure.update_traces(textinfo="percent+label", sort=False)
    st.plotly_chart(final_component_figure, width="stretch")

    if final_component_display_df["component"].astype(str).eq("other_non_role").any():
        st.subheader("Other Non-role mapping")
        st.dataframe(
            build_other_non_role_mapping_table(),
            width="stretch",
            hide_index=True,
        )

if debug_mode:
    render_component_prediction("D. Role Component Breakdown", role_prediction_df)
    render_component_prediction(
        "E. Non-role Component Breakdown",
        non_role_prediction_df,
    )

    st.header("F. Calculation Details")
    st.caption(
        "Each row has its historical weight. all_role_components, mixed_components, "
        "and pipe-separated primary_sub_category values are expanded and split equally. "
        "Component scores are normalized inside their source primary category, then "
        "multiplied by that category's adjusted prediction."
    )

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

render_monthly_update_section(
    page_data["raw_df"],
    final_component_display_df if not final_component_df.empty else final_component_df,
    debug_mode,
)
