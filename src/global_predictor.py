import pandas as pd


GLOBAL_PATH_COLUMNS = [
    "primary_category",
    "primary_sub_category",
    "path_label",
    "category_adjusted_prediction_percent",
    "adjusted_conditional_sub_probability_percent",
    "global_path_score",
]

FINAL_COMPONENT_COLUMNS = [
    "component",
    "component_type",
    "global_probability_percent",
    "historical_component_weight",
    "source_paths",
    "applied_rules",
]


def calculate_global_path_prediction(
    adjusted_primary_df: pd.DataFrame,
    conditional_sub_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine adjusted category and conditional component scores."""
    required_primary_columns = ["primary_category", "adjusted_prediction_percent"]
    required_sub_columns = [
        "primary_category",
        "primary_sub_category",
        "adjusted_conditional_sub_probability_percent",
    ]

    if (
        adjusted_primary_df.empty
        or conditional_sub_df.empty
        or any(
            column not in adjusted_primary_df.columns
            for column in required_primary_columns
        )
        or any(
            column not in conditional_sub_df.columns
            for column in required_sub_columns
        )
    ):
        return empty_global_path_table()

    primary_scores = adjusted_primary_df[required_primary_columns].rename(
        columns={
            "adjusted_prediction_percent": "category_adjusted_prediction_percent"
        }
    )
    sub_scores = conditional_sub_df[required_sub_columns]
    global_df = sub_scores.merge(primary_scores, on="primary_category", how="inner")
    global_df["global_path_score"] = (
        global_df["category_adjusted_prediction_percent"]
        * global_df["adjusted_conditional_sub_probability_percent"]
        / 100
    )
    global_df["path_label"] = (
        global_df["primary_category"] + " -> " + global_df["primary_sub_category"]
    )
    global_df = global_df.sort_values("global_path_score", ascending=False)
    global_df["global_path_score"] = global_df["global_path_score"].round(4)
    return global_df[GLOBAL_PATH_COLUMNS]


def empty_global_path_table() -> pd.DataFrame:
    return pd.DataFrame(columns=GLOBAL_PATH_COLUMNS)


def calculate_final_global_component_prediction(
    conditional_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine duplicate components from primary-category parent paths."""
    required_columns = [
        "primary_category",
        "component_type",
        "primary_sub_category",
        "component_weight",
        "adjusted_prediction_percent",
    ]
    if conditional_df.empty or any(
        column not in conditional_df.columns for column in required_columns
    ):
        return empty_final_component_table()

    working_df = conditional_df.copy()
    if "applied_rules" not in working_df.columns:
        working_df["applied_rules"] = "none"

    grouped = (
        working_df.groupby(
            ["primary_sub_category", "component_type"],
            as_index=False,
        )
        .agg(
            global_probability_percent=("adjusted_prediction_percent", "sum"),
            historical_component_weight=("component_weight", "sum"),
            source_paths=("primary_category", join_unique_values),
            applied_rules=("applied_rules", join_applied_rules),
        )
        .rename(columns={"primary_sub_category": "component"})
    )
    grouped[
        ["global_probability_percent", "historical_component_weight"]
    ] = grouped[
        ["global_probability_percent", "historical_component_weight"]
    ].round(4)
    return grouped.sort_values(
        "global_probability_percent",
        ascending=False,
    ).reset_index(drop=True)[FINAL_COMPONENT_COLUMNS]


def join_unique_values(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values if str(value)}))


def join_applied_rules(values: pd.Series) -> str:
    rules = set()
    for value in values:
        rules.update(
            rule
            for rule in str(value).split("|")
            if rule and rule != "none"
        )
    return "|".join(sorted(rules)) if rules else "none"


def empty_final_component_table() -> pd.DataFrame:
    return pd.DataFrame(columns=FINAL_COMPONENT_COLUMNS)
