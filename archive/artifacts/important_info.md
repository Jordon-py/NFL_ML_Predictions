# 🔑 Key Architectural Insights

_Last Updated:_ 2026-01-13 10:32:00

## Recent Changes (2026-01-13)

### Performance Optimization

- **Health polling**: Frontend now polls `/health` every 60 seconds (down from 15s) to reduce server load by 75%.
- **DataFrame fragmentation**: Fixed PerformanceWarning in `inference_row.py` by replacing iterative `.assign()` calls with single `pd.concat()` operation.
- **Inference simplification**: Removed nested `_map_stats` function, inlined stat mapping logic for better readability.
- **Imputation safety**: Added edge case handling for empty datasets and final NaN fallback in `_impute_remaining_missing`.

### Known Warnings (Safe to Ignore)

- **sklearn UserWarning**: "Skipping features without any observed values: ['neutral_site', 'kickoff_hour_utc', 'travel_distance_km']" — Expected when these columns have no data for a particular game. sklearn handles gracefully.
- **sklearn UserWarning**: "Skipping features without any observed values: ['kickoff']" — Same as above for categorical imputation.

## Core Architecture

### Prediction Pipeline

- Unified prediction response: frontend expects flat `UnifiedPredictionResponse` fields (home_score, home_win_probability, etc.).
- Schedule enrichment: backend supplies team names/logos per game and reads `team_logos.csv` from repo root.
- CORS config: backend uses `ALLOWED_ORIGINS` and `ALLOW_ORIGIN_REGEX` to allow localhost and Vercel previews.
- Legacy router: `backend/routes.py` is mounted under `/legacy` to preserve older endpoint shapes.

### Inference & Data Handling

- **Inference alignment**: `build_model_input_row` reindexes once and bulk-fills medians to avoid DataFrame fragmentation.
- **Roll-forward fills**: Stats now use batched assignments (`pd.concat()`) to avoid fragmentation warnings during synthetic rows.
- **Schedule ingestion**: `_load_schedule_df` trims CSV headers to normalize `home_team`/`away_team`/`week` fields.
- **Performance**: Per-team history cache avoids re-scanning the dataset for roll-forward fills.

### Models & Data

- **Models directory**: MODELS_DIR is set to `backend/20260102/models` for the latest artifacts.
- **Dataset alignment**: DATA_DIR defaults to `backend/data/datasets` to match model feature expectations.
- **Dataset enforcement**: `DATASET_PATH` can pin the exact CSV; startup fails if model features are missing.

### Documentation & Debug

- **Prediction endpoint map**: `docs/PREDICTION_ENDPOINT_MAP.md` and `docs/prediction_endpoint_map.svg` document the /predict flow.
- **Debug endpoint**: `/debug` uses `datetime`/`timezone` imports and shows model provenance.
- **Debug visibility**: `/debug/predict-input` reports missing/filled features for troubleshooting.

## Key Documentation

- [Artifacts System](artifacts_README.md)
- [Last 5 Tasks](last_5_tasks.md)
- [Next 5 Tasks](next_5_tasks.md)
- [Dataflow Map](../dataflow.md)
