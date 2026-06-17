# RDO Benefits Tool

MVP Streamlit app for exploring Red Dead Online Benefits data from CSV.

## Features

- Load benefits history from `data/raw/rdo_benefits_raw.csv`
- Display raw/cleaned benefits in Streamlit
- Calculate probability by benefit group
- Show simple Plotly charts
- Estimate next-month benefit group probability

## CSV Format

Required columns:

```csv
month,benefit_group,benefit_name
```

Optional columns:

```csv
description,start_date,end_date
```

Example:

```csv
month,benefit_group,benefit_name,description,start_date,end_date
2026-01,Role Bonus,Bounty Hunter XP Bonus,Example row,2026-01-01,2026-01-31
2026-02,Discount,Stable Discount,Example row,2026-02-01,2026-02-28
```

## Run

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Prediction Method

The MVP uses a simple weighted frequency model:

- 60% historical benefit group frequency
- 40% recent-month benefit group frequency

This is not a guaranteed Rockstar schedule prediction. It is a lightweight baseline
that becomes more useful as the CSV history grows.
