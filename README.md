# RDO Benefits Tool

MVP Streamlit app for exploring Red Dead Online Benefits data from CSV.

## Features

- Validate and normalize the monthly benefits CSV
- Display data quality, raw data, empirical probability, and happening benefits
- Expand all-role, mixed, and pipe-separated benefit components
- Estimate next-month primary category and component probability
- Present one Final Global Component Prediction across all component types
- Backtest a small set of practical prediction parameters

## CSV Format

The app reads `data/raw/rdo_benefits_raw.csv`. Primary categories are:
`role`, `non_role`, and `both`. `all_role`, `mixed`, and `seasonal` are not
standalone Primary Categories; they are handled through their flag/component
columns. Legacy rows with `primary_category=seasonal`, `all_role`, or `mixed`
are normalized back to `role`, `non_role`, or `both` during loading.

CSV V2 columns:

```csv
month_label,start_date,end_date,primary_category,primary_sub_category,multiplier_info,is_seasonal,seasonal_type,is_mixed,mixed_components,is_all_role,all_role_components,weight,component_weight_rule,extra_tags,note,source_name,source_url,confidence
```

Pipe-separated component fields are expanded by the application. All-role,
mixed, and multi-component rows divide their monthly weight equally across
their components. `other_non_role` is the grouped non-role bucket for rare
non-role content, currently `story_missions`, `showdown`, and `gang_hideouts`.
Those legacy values are normalized to `other_non_role` in
`primary_sub_category` and `mixed_components`.

Seasonal rows use `is_seasonal=TRUE` and a non-empty `seasonal_type`.
Seasonal affects component/sub-category rules only, not Primary Category
Prediction. Month rules are:

- `halloween` / `halloween_call_to_arms`: October
- `holiday` / `holiday_call_to_arms`: December

Overview seasonal tables and charts display only two groups: Halloween and Holiday.
`holiday_call_to_arms` can still exist in the CSV as `seasonal_type`, but it is
displayed under Holiday. `halloween_call_to_arms` is displayed under Halloween.

`multiplier_info` is display metadata and does not affect probability.
`extra_tags` and `note` do not affect probability. This schema does not use
`include_in_probability` or `exclude_reason`.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Prediction Method

The MVP combines weighted historical category frequency, smoothed monthly
transitions, after-all-role rules, role/non-role cooldown, and seasonal metadata.
After an all-role month, `role` is strongly reduced, `non_role` is strongly
boosted, and `both` is moderately boosted without creating an `all_role`
primary category.

Component scores are normalized inside `role`, `non_role`, or `both`, then
constrained by the adjusted probability of that parent category. Seasonal data
can adjust component/sub-category rules, but it does not create a `seasonal`
parent category. The Final Global Component Prediction merges duplicate
components across parent paths and is the main 100% component result. Role and
non-role tables are supporting breakdowns only.

`multiplier_info`, `source_name`, `source_url`, `confidence`, `extra_tags`, and
`note` remain display metadata and do not affect probability.

## Optimization Cache

The prediction page does not run the full parameter optimization automatically
on every app load. It first checks `.cache/optimization_result.json`.

If the CSV hash, rule hash, search-space hash, and logic version match, cached
best params are used. If the cache is missing or stale, the page uses default
params until the user presses `Run Optimization`. Running optimization writes a
fresh cache file with the best params, score, and creation time.
