# Connect ESPN Schedule Ingestion to Prediction Without Synthetic Fallback

## Current diagnosis

Your stack already has most of the right pieces:

1. `PredictionService` caches:
   - team history
   - exact-match index
   - numeric medians
   - schedule data by season

2. `build_model_input_row(...)` already prefers exact dataset rows before constructing a synthetic row.

3. Your dataset builder already has postseason-aware constants:
   - `WC`
   - `DIV`
   - `CON`
   - `SB`
   - `POST`

The missing bridge is making sure the ESPN-ingested schedule produces rows with the same identity contract as the model dataset:

```txt
season + week + home_team + away_team
```

For postseason, that means model/nflverse weeks should be:

```txt
Wild Card = 19
Divisional = 20
Conference = 21
Super Bowl = 22
```

---

## Step 1 — Place the ingestion module

Copy:

```txt
schedule_ingestion.py
```

to:

```txt
backend/services/schedule_ingestion.py
```

---

## Step 2 — Generate the current season schedule

From repo root:

```powershell
python -m backend.services.schedule_ingestion `
  --season 2025 `
  --season-types 2,3 `
  --out-csv backend/data/Nfl_schedule_2025.csv `
  --out-parquet backend/data/schedules/nfl_schedule_2025.parquet `
  --raw-dir backend/data/raw/espn/scoreboards
```

This writes a CSV that your existing schedule discovery logic can find.

---

## Step 3 — Rebuild the feature dataset with future rows included

Use your canonical wrapper:

```powershell
python backend/builddataset.py `
  --start 2018 `
  --end 2025 `
  --out-dir backend/data/datasets `
  --encode onehot `
  --save-dominance-matrix
```

Your wrapper calls the builder with `include_future=config.include_future`, and the builder itself supports `include_future=True`.

---

## Step 4 — Verify exact match coverage

Add a quick smoke test:

```python
import pandas as pd
from backend.services.inference_row import build_exact_match_index
from backend.utils.team_codes import normalize_team_code

df = pd.read_csv("backend/data/datasets/latest_clean_or_promoted_dataset.csv")
idx = build_exact_match_index(df)

key = (
    2025,
    19,
    normalize_team_code("LAC"),
    normalize_team_code("PIT"),
)

print("exact match?", key in idx)
```

Replace the teams/week with an actual playoff matchup from your generated schedule.

---

## Step 5 — Expected prediction behavior

When `/predict` receives:

```json
{
  "season": 2025,
  "week": 19,
  "home_team": "LAC",
  "away_team": "PIT"
}
```

the desired internal path is:

```txt
build_model_input_row
  -> exact_match_index hit
  -> source = dataset_exact_index
```

The fallback path you want to avoid is:

```txt
build_model_input_row
  -> no exact row
  -> source = synthetic_model_assembly
```

---

## Step 6 — Add a debug assertion

Use `/debug/predict-input` or similar diagnostics and check:

```txt
selected_row_source should be dataset_exact_index or dataset_exact
missing_prior_count should be low
missing_after_impute should be low
```

If you see:

```txt
selected_row_source = synthetic
```

or:

```txt
prediction_source = synthetic_model_assembly
```

then your model-ready dataset is still missing the target game row.

---

## Step 7 — The critical invariant

The ESPN schedule CSV and feature dataset must agree on:

```txt
season
week
home_team
away_team
```

For postseason, the most common bug is:

```txt
ESPN week = 1
model dataset week = 19
```

The provided ingestion module fixes that by mapping postseason weeks into `19-22`.
