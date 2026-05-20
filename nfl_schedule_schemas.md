# NFL Schedule + Dataset Schemas

## 1. `raw_schedule` / ESPN snapshot layer

**Storage**
- JSON files under `backend/data/raw/espn/scoreboards/`
- Suggested filename: `season=2025_type=2_week=1.json`

**Purpose**
- Preserve the exact ESPN API payload before parsing.
- Use for debugging parser changes without refetching.

**Contract**
- No strict column schema; this is raw JSON.
- Never train directly from this layer.

---

## 2. `clean_schedule` / one row per game

Recommended files:

```txt
backend/data/Nfl_schedule_2025.csv
backend/data/schedules/nfl_schedule_2025.parquet
```

| Column | Type | Required | Notes |
|---|---:|---:|---|
| `season` | int | yes | NFL season year |
| `season_type` | int | yes | ESPN: `2=regular`, `3=postseason` |
| `game_type` | string | yes | `REG`, `WC`, `DIV`, `CON`, `SB`, `POST` |
| `week` | int | yes | Model/nflverse week. Postseason should be `19-22` |
| `espn_week` | int | yes | ESPN site API week number |
| `game_id` | string | yes | Canonical: `YYYY_WW_AWAY_HOME` |
| `espn_game_id` | string | yes | ESPN event ID |
| `gameday` | date string | yes-ish | `YYYY-MM-DD` |
| `game_date` | date string | yes-ish | Alias for builder compatibility |
| `gametime` | string | optional | `HH:MMZ` |
| `kickoff` | datetime string | optional | UTC ISO datetime |
| `kickoff_utc` | datetime string | optional | UTC ISO datetime |
| `home_team` | string | yes | Canonical abbreviation |
| `away_team` | string | yes | Canonical abbreviation |
| `home_team_name` | string | optional | Display name |
| `away_team_name` | string | optional | Display name |
| `home_score` | int/null | conditional | Must be null unless `completed=true` |
| `away_score` | int/null | conditional | Must be null unless `completed=true` |
| `completed` | bool | yes | Final/completed game flag |
| `status` | string | optional | ESPN status name |
| `status_detail` | string | optional | ESPN status detail |
| `neutral_site` | bool/null | optional | Useful for Super Bowl |
| `venue` | string/null | optional | Stadium |
| `roof` | string/null | optional | Future enrichment |
| `surface` | string/null | optional | Future enrichment |
| `spread_line` | float/null | optional | Pregame only; timestamp if possible |
| `total_line` | float/null | optional | Pregame only; timestamp if possible |
| `home_moneyline` | int/null | optional | Pregame only; timestamp if possible |
| `away_moneyline` | int/null | optional | Pregame only; timestamp if possible |
| `source` | string | yes | Example: `espn_site_scoreboard` |
| `ingested_at` | datetime string | yes | UTC ingestion timestamp |

**Hard validation rules**
- `game_id` must be unique.
- `home_team` and `away_team` cannot be blank or equal.
- Future rows must have `home_score = null` and `away_score = null`.
- Regular season and postseason must be included when building live/current season predictions.

---

## 3. `model_ready_games` / training + inference feature table

Recommended files:

```txt
backend/data/datasets/game_features_YYYYMMDD_clean.csv
backend/data/datasets/game_features_YYYYMMDD_clean.parquet
```

**Minimum identity columns**
| Column | Type | Required |
|---|---:|---:|
| `season` | int | yes |
| `week` | int | yes |
| `game_id` | string | yes |
| `home_team` | string | yes |
| `away_team` | string | yes |
| `game_type` | string | strongly recommended |
| `time_key` | int | recommended |

**Allowed feature families**
- Pregame schedule context: rest, neutral site, roof, surface, market lines.
- Team prior stats: `home_*prior*`, `away_*prior*`
- Rolling stats: `home_*rolling*`, `away_*rolling*`
- Difference features: `*_diff`
- Encoded team features if the model expects them.

**Outcome columns**
These can exist in the full dataset for training labels, but must be dropped from `X` before model fitting:

```txt
home_points_for
away_points_for
point_diff
winner
home_win
actual_winner
```

**Leakage rule**
All rolling/prior stats must be computed with a strict `shift(1)` or equivalent prior-game-only logic.

---

## 4. Serving contract

For `/predict`, the lookup key should be:

```python
(season, week, normalize(home_team), normalize(away_team))
```

Preferred source order:

```txt
1. dataset_exact_index
2. dataset_exact
3. schedule_enriched_rollforward
4. synthetic_model_assembly only as emergency fallback
```

If postseason games are missing from the model-ready dataset, `/predict` will not find exact rows and will fall back to synthetic assembly.
