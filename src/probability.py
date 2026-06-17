import pandas as pd

from src.seasonal_rules import is_seasonal_sub_category_allowed


ROLE_ORDER = [
    "bounty_hunter",
    "trader",
    "collector",
    "moonshiner",
    "naturalist",
]

ALL_ROLE_ORDER = [
    "all_role",
]

NON_ROLE_ORDER = [
    "free_roam",
    "blood_money",
    "telegram_missions",
    "races",
    "call_to_arms",
    "showdown",
    "strange_tales",
]

MIXED_ORDER = [
    "mixed",
]

SEASONAL_ORDER = [
    "halloween",
    "holiday_call_to_arms",
]

OTHER_LABEL = "Other"


def calculate_probability(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Calculate empirical probability for one category column."""
    columns = [column_name, "count", "probability_percent"]

    if df.empty or column_name not in df.columns:
        return pd.DataFrame(columns=columns)

    valid_df = df[df[column_name].fillna("").astype(str).str.strip() != ""]
    if valid_df.empty:
        return pd.DataFrame(columns=columns)

    total = len(valid_df)
    result = (
        valid_df[column_name]
        .value_counts()
        .rename_axis(column_name)
        .reset_index(name="count")
    )
    result["probability_percent"] = (result["count"] / total * 100).round(2)

    return result[columns]


def collapse_to_top_n(
    probability_df: pd.DataFrame,
    label_column: str,
    top_n: int = 8,
    other_label: str = "Other",
    always_include: list[str] | None = None,
    value_column: str = "probability_percent",
) -> pd.DataFrame:
    """Keep top groups and combine the rest into Other."""
    if (
        probability_df.empty
        or len(probability_df) <= top_n
        or value_column not in probability_df.columns
    ):
        return probability_df

    always_include = always_include or []
    top_indexes = set(probability_df.head(top_n).index)
    always_include_indexes = set(
        probability_df[probability_df[label_column].isin(always_include)].index
    )
    keep_indexes = sorted(top_indexes | always_include_indexes)

    top_df = probability_df.loc[keep_indexes].copy()
    other_df = probability_df.drop(index=keep_indexes)

    if other_df.empty:
        return top_df.reset_index(drop=True)

    value_precision = 2 if value_column == "probability_percent" else 4
    other_row = pd.DataFrame(
        [
            {
                label_column: other_label,
                "count": int(other_df["count"].sum()),
                value_column: round(
                    float(other_df[value_column].sum()),
                    value_precision,
                ),
            }
        ]
    )

    return pd.concat([top_df, other_row], ignore_index=True)


def order_sub_category_prediction_df(
    df: pd.DataFrame,
    predicted_next_month: str | None = None,
    is_prediction: bool = True,
    label_column: str = "primary_sub_category",
) -> pd.DataFrame:
    if df.empty or label_column not in df.columns:
        return df

    ordered_df = df.copy()
    if is_prediction:
        allowed_mask = ordered_df[label_column].apply(
            lambda sub_category: is_seasonal_sub_category_allowed(
                sub_category,
                predicted_next_month,
            )
        )
        ordered_df = ordered_df[allowed_mask].copy()

    order = build_sub_category_order(predicted_next_month, is_prediction)
    order_rank = {label: index for index, label in enumerate(order)}
    fallback_rank = len(order)
    other_rank = fallback_rank + 1

    ordered_df["_sub_category_order"] = ordered_df[label_column].apply(
        lambda label: other_rank
        if label == OTHER_LABEL
        else order_rank.get(label, fallback_rank)
    )
    ordered_df["_original_order"] = range(len(ordered_df))

    ordered_df = ordered_df.sort_values(
        ["_sub_category_order", "_original_order"],
        kind="stable",
    ).drop(columns=["_sub_category_order", "_original_order"])

    return ordered_df.reset_index(drop=True)


def build_sub_category_order(
    predicted_next_month: str | None,
    is_prediction: bool,
) -> list[str]:
    seasonal_order = SEASONAL_ORDER
    if is_prediction:
        seasonal_order = [
            sub_category
            for sub_category in SEASONAL_ORDER
            if is_seasonal_sub_category_allowed(sub_category, predicted_next_month)
        ]

    return (
        ROLE_ORDER
        + ALL_ROLE_ORDER
        + NON_ROLE_ORDER
        + seasonal_order
        + MIXED_ORDER
    )
