# RDO Benefits Tool

RDO Benefits Tool is a local Streamlit app for storing, checking, and predicting
Red Dead Online monthly benefits from a historical CSV file.

V2 is a manual, CSV-first tool. It does not crawl Rockstar, Reddit, or any other
website. The app reads the local data, validates it, shows the current benefits,
and produces practical next-month probability tables that can help with manual
monthly updates.

## Current V2 Status

V2 includes:

- Simple Mode and Debug Mode
- Data Quality Check
- All Benefits display
- Raw Data display in Debug Mode
- Historical Category Probability
- Happening Benefits
- Next Month Prediction
- Final Global Component Prediction
- Compact Other Non-role mapping
- Monthly Update for closing and adding monthly benefits
- Optimization cache for backtesting parameters

## How To Run

From this project folder:

```powershell
cd D:\Tool\RDO_Benefits_Tool
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.port=8501
```

Then open:

```text
http://localhost:8501
```

## Project Structure

- `app.py`: overview page with data quality, all benefits, raw data, historical probability, and Happening benefits.
- `pages/02_next_month_prediction.py`: Next Month Prediction page, Final Global Component Prediction, debug breakdowns, and Monthly Update.
- `src/data_loader.py`: CSV loading, schema checks, raw-to-normalized data preparation, component expansion.
- `src/taxonomy.py`: V2 taxonomy, aliases, rare non-role grouping, and display normalization.
- `src/validator.py`: data quality checks for duplicates, missing months, categories, components, seasonal metadata, mixed rows, and all-role rows.
- `src/seasonal_rules.py`: seasonal type metadata and hard seasonal eligibility rules.
- `src/conditional_predictor.py`: role/non-role component prediction, recency multipliers, seasonal multipliers, and conditional component tables.
- `src/global_predictor.py`: merges conditional component paths into the Final Global Component Prediction.
- `src/optimization_cache.py`: cached best backtest parameters and cache staleness metadata.
- `src/monthly_update.py`: CSV-safe Monthly Update helpers for preview, backup, close, append, and validation.
- `data/raw/rdo_benefits_raw.csv`: source CSV used by the app.
- `data/backups/`: backup location used before CSV writes.
- `tests/`: unit tests and Streamlit AppTest checks.

## UI Pages

The Overview page contains:

- Data Quality Check
- All Benefits
- Raw Data in Debug Mode
- Historical Category Probability
- Seasonal Display Probability in Debug Mode
- Happening Benefits

The Next Month Prediction page contains:

- Auto Backtesting and Parameter Optimization
- Primary Category Prediction
- Final Global Component Prediction
- Other Non-role mapping when applicable
- Role and Non-role Component Breakdown in Debug Mode
- Calculation Details in Debug Mode
- Monthly Update

## CSV Schema V2

The app reads:

```text
data/raw/rdo_benefits_raw.csv
```

V2 columns:

```csv
month_label,start_date,end_date,primary_category,primary_sub_category,multiplier_info,is_seasonal,seasonal_type,is_mixed,mixed_components,is_all_role,all_role_components,weight,component_weight_rule,extra_tags,note,source_name,source_url,confidence
```

Main field behavior:

- `month_label`: month key in `YYYY-MM` format.
- `start_date`: benefit start date.
- `end_date`: benefit end date, or `Happening` for the current active month.
- `primary_category`: `role`, `non_role`, `both`, or a data-entry seasonal value that is normalized for prediction.
- `primary_sub_category`: one or more raw sub-categories, pipe-separated.
- `multiplier_info`: human-readable benefit details; it does not affect probability.
- `is_seasonal`: marks seasonal metadata.
- `seasonal_type`: `halloween`, `holiday`, `thanksgiving`, `valentines`, or `easter`.
- `is_mixed`: true when a row contains mixed components.
- `mixed_components`: pipe-separated components used for mixed rows.
- `is_all_role`: true when all five roles are represented.
- `all_role_components`: pipe-separated role components for all-role rows.
- `weight`: row weight, usually `1`.
- `component_weight_rule`: how components are split, usually `single`, `split_equal`, `mixed_manual`, or `equal_split`.
- `extra_tags`: legacy/display metadata; it does not affect probability.
- `note`: metadata only; it does not affect probability.
- `source_name` and `source_url`: manual source metadata.
- `confidence`: source confidence; manual Monthly Update rows default to `high`.

One monthly benefits period should be one CSV row. If a month has multiple
sub-categories, store them in `primary_sub_category` with pipe-separated values
such as `trader|free_roam|call_to_arms`. Do not create multiple rows for the
same `month_label`.

## Primary Categories

V2 prediction works with three parent categories:

- `role`
- `non_role`
- `both`

Seasonal data is handled through `is_seasonal`, `seasonal_type`, and
`seasonal_rules.py`. Some data-entry flows can write `primary_category=seasonal`
for hard seasonal benefits, but loading normalizes rows back into the prediction
model so seasonal does not become a fourth parent probability category.

Thanksgiving is not a hard seasonal prediction component. It is a known seasonal
type for November metadata and can appear with a normal role or non-role benefit.
For example, a Trader Thanksgiving benefit should generally be represented as
`primary_category=role`, `primary_sub_category=trader`,
`is_seasonal=TRUE`, and `seasonal_type=thanksgiving`.

Valentines and Easter are known historical seasonal types, but they do not add
hard prediction eligibility rules in V2.

## Role Taxonomy

Role prediction components:

- `bounty_hunter`
- `trader`
- `collector`
- `moonshiner`
- `naturalist`

## Non-role Taxonomy

Non-role prediction components:

- `free_roam`
- `races`
- `blood_money`
- `telegram`
- `call_to_arms`
- `featured_series`
- `other_non_role`

Known rare raw non-role values:

- `story_missions`
- `gang_hideouts`
- `showdown`

The CSV and All Benefits view can show Story Missions, Gang Hideouts, and
Showdown as raw benefit types. Internally, those rare raw values normalize to
`other_non_role` for prediction. They are not split into separate Final Global
prediction components because the data is sparse and the taxonomy would become
noisy.

The prediction page shows a compact mapping when `other_non_role` appears:

- Story Missions -> Other Non-role
- Gang Hideouts -> Other Non-role
- Showdown -> Other Non-role

Direct `other_non_role` is accepted as a fallback, but the validator warns
because a specific raw non-role value is better when known.

## Raw-first Data Flow

V2 keeps raw CSV values visible while using normalized values for prediction:

- The CSV stores the raw value entered by the user.
- Raw Data displays the raw CSV values.
- All Benefits displays readable raw labels.
- Prediction uses normalized components.

Examples:

- `story_missions` -> `Story Missions` in display -> `other_non_role` for prediction.
- `gang_hideouts` -> `Gang Hideouts` in display -> `other_non_role` for prediction.
- `showdown` -> `Showdown` in display -> `other_non_role` for prediction.

This keeps the data auditable without letting rare one-off labels fragment the
probability model.

## Mixed And Both Handling

Use `primary_category=both` when a month includes role and non-role components,
or when the monthly benefit is a mixed bundle that should contribute to multiple
component histories.

For both/mixed rows:

- `is_mixed=TRUE`
- `mixed_components` contains pipe-separated components.
- `primary_sub_category` can mirror the same pipe-separated component list.
- Components are expanded during analysis.
- Monthly weight is split across expanded components.

Monthly Update follows this convention by copying selected sub-categories into
`mixed_components` when `primary_category=both`.

## All-role Handling

All-role rows use:

- `is_all_role=TRUE`
- `all_role_components` with all five role components.

All-role rows are expanded for component probability, but they are not allowed to
pollute role recency. Role recency uses true last-seen role months, so an
all-role benefit does not make every role look newly featured in the same month.

After an all-role month, the primary prediction applies practical domain rules:
role is penalized, non-role is boosted, and both is moderately boosted. `all_role`
is not a standalone primary category.

## Seasonal Handling

Hard seasonal prediction components have month eligibility:

- Halloween hard seasonal components: October.
- Holiday hard seasonal components: December.

Hard seasonal examples include:

- `halloween`
- `halloween_call_to_arms`
- `holiday`
- `holiday_call_to_arms`
- `holiday_rewards`

If the target prediction month is outside a hard seasonal component's eligible
month, that component is excluded from Final Global Component Prediction.

Thanksgiving is a known seasonal type for November metadata, but it is not a hard
component that excludes normal Trader or role prediction. Valentines and Easter
are also known seasonal metadata types without V2 prediction eligibility.

## Prediction Logic Overview

The tool first calculates Primary Category Prediction across `role`, `non_role`,
and `both`. It then expands monthly rows into components and calculates
conditional component scores using historical weight, recency, after-all-role
rules, and seasonal rules.

Final Global Component Prediction is the main component result. It merges role,
non-role, and both paths into one 100% probability system. The same component
can appear under multiple parent paths, and those paths are merged into one final
component row.

Role and non-role breakdown tables in Debug Mode are supporting detail. They are
not separate competing 100% systems. Small total differences such as `99.9998`
or `100.0001` can happen because displayed values are rounded.

Display metadata such as `multiplier_info`, `source_name`, `source_url`,
`confidence`, `extra_tags`, and `note` does not change probability.

## Recency Rules

Role recency reduces scores for roles that appeared recently and boosts roles
that have not appeared for a longer time. All-role rows are ignored for role
last-seen calculations so they do not reset every role at once.

Non-role recency works similarly, but more lightly. Recent non-role components
can receive a cooldown, while long-gap or unknown non-role components can
receive a small boost.

The app also keeps a recent-sub-category cooldown path for practical smoothing.
These rules are intended to prevent the model from blindly repeating the most
recent benefit type.

## Optimization Cache

The prediction page does not automatically run heavy optimization every time the
app opens. It checks `.cache/optimization_result.json`.

The cache stores:

- `LOGIC_VERSION`
- `csv_hash`
- `rules_hash`
- `search_space_hash`
- best prior strength
- best cooldown config
- score
- creation time

If the CSV, rules, search space, or logic version changes, the cache becomes
stale and the page uses default parameters until the user presses
`Run Optimization`. There is normally no need to delete the cache manually unless
debugging.

## Data Quality And Validator

The validator checks:

- required columns
- duplicated `month_label`
- missing months
- primary categories
- role and non-role components
- aliases
- unknown components
- mixed row consistency
- all-role row consistency
- seasonal type metadata
- seasonal month quality warnings

Known aliases can be reported as info. Unknown non-role values normalize to
`other_non_role` with a warning. Unknown role values are errors. Direct
`other_non_role` is a warning because raw values such as `story_missions`,
`gang_hideouts`, or `showdown` are more useful when known.

`note` is preserved and displayed as metadata, but it does not affect prediction.

## Monthly Update

Monthly Update is at the bottom of the Next Month Prediction page. It writes to
the raw CSV only after preview, confirmation, and backup.

### Close Current Benefits

This action:

- finds rows with `end_date=Happening`
- lets the user choose the real end date
- previews the rows that will change
- shows the CSV path and backup path
- requires a confirmation checkbox
- backs up the CSV
- updates only `end_date` on Happening rows

If no Happening rows exist, the app shows a light message and does not append any
new data.

### Add New Month Benefits

This action:

- calculates the next `month_label` after the latest CSV month
- uses the top Final Global Component Prediction to ask whether that component is the new month benefit
- locks `primary_category` when the user answers Yes
- allows manual category selection when the user answers No
- uses a multiselect for `primary_sub_category`
- stores multiple sub-categories as pipe-separated values in one row
- can close current Happening benefits using `start_date - 1`
- previews changed rows and the new row
- requires confirmation
- backs up the CSV before writing
- blocks duplicate `month_label`

If the top prediction is `other_non_role`, the app asks the user to choose a
specific raw non-role type such as `story_missions`, `gang_hideouts`, or
`showdown` instead of blindly writing `other_non_role`.

Seasonal entry is guarded. Hard seasonal top predictions can lock seasonal entry
and set `is_seasonal` plus `seasonal_type`. Manual seasonal entry outside known
seasonal months requires an explicit unlock and confirmation for exceptional
events.

## Safety Notes

CSV writes always create a backup first under:

```text
data/backups/
```

If a manual test write needs rollback, restore from backup or manually remove
the new month row and set the previous month back to `end_date=Happening`.

The app does not commit or push Git changes. Source updates, CSV edits, staging,
commits, and pushes remain manual developer actions.

## Testing

Recommended checks:

```powershell
.\.venv\Scripts\python.exe -m compileall src app.py pages tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Streamlit AppTest is also used during development to check that `app.py` and
`pages/02_next_month_prediction.py` render without exceptions. These checks do
not require starting a real Streamlit server.

## V3 Roadmap

Possible V3 work:

- optional fetch/crawl support from trusted Rockstar, Reddit, or community sources
- update existing month support when a row already exists
- stronger source handling and source review workflow
- richer Monthly Update editing flow
- more taxonomy detail only if future data volume justifies it

Rare non-role values should stay grouped under `other_non_role` unless there is
enough historical data to make separate prediction components useful.
