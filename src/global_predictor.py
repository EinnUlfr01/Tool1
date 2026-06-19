import pandas as pd


GLOBAL_PATH_COLUMNS = [
    "primary_category",
    "primary_sub_category",
    "path_label",
    "category_adjusted_prediction_percent",
    "adjusted_conditional_sub_probability_percent",
    "global_path_score",
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
