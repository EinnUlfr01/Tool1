import pandas as pd


EXTRA_TAG_COLUMNS = ["extra_tag", "count", "tag_probability_percent"]


def calculate_extra_tag_probabilities(
    df: pd.DataFrame,
    primary_category: str,
    primary_sub_category: str,
) -> pd.DataFrame:
    """Calculate row-level tag frequency for a category/sub-category pair."""
    required_columns = ["primary_category", "primary_sub_category", "extra_tags"]
    if df.empty or any(column not in df.columns for column in required_columns):
        return empty_extra_tag_table()

    pair_df = df[
        (df["primary_category"] == primary_category)
        & (df["primary_sub_category"] == primary_sub_category)
    ]
    if pair_df.empty:
        return empty_extra_tag_table()

    total_rows = len(pair_df)
    tag_counts: dict[str, int] = {}

    for raw_tags in pair_df["extra_tags"]:
        tags = parse_extra_tags(raw_tags)
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if not tag_counts:
        return empty_extra_tag_table()

    result_df = pd.DataFrame(
        [
            {
                "extra_tag": tag,
                "count": count,
                "tag_probability_percent": round(count / total_rows * 100, 2),
            }
            for tag, count in tag_counts.items()
        ]
    )

    return result_df.sort_values(
        ["tag_probability_percent", "count", "extra_tag"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def parse_extra_tags(raw_tags: object) -> set[str]:
    if pd.isna(raw_tags):
        return set()

    return {
        tag.strip()
        for tag in str(raw_tags).split("|")
        if tag.strip()
    }


def empty_extra_tag_table() -> pd.DataFrame:
    return pd.DataFrame(columns=EXTRA_TAG_COLUMNS)
