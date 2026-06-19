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

The app reads `data/raw/rdo_benefits_raw.csv`. Primary categories are limited
to `role`, `non_role`, and `both`.

Required columns:

```csv
month_label,start_date,end_date,primary_category,primary_sub_category,is_mixed,mixed_components,is_all_role,all_role_components,is_seasonal,seasonal_type,multiplier_info,weight,component_weight_rule,source_name,source_url,confidence
```

Pipe-separated component fields are expanded by the application. All-role,
mixed, and multi-component rows divide their monthly weight equally across
their components. `multiplier_info` and source fields are display metadata and
do not affect probability.

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
constrained by the adjusted probability of that parent category. The Final
Global Component Prediction merges duplicate components across parent paths
and is the main 100% component result. Role and non-role tables are supporting
breakdowns only.

`multiplier_info`, `source_name`, `source_url`, and `confidence` remain display
metadata and do not affect probability.
