# ?? Key Architectural Insights

_Last Updated:_ 2026-01-02 08:27:54

## Current Focus
- Unified prediction response: frontend expects flat `UnifiedPredictionResponse` fields (home_score, home_win_probability, etc.).
- Schedule enrichment: backend supplies team names/logos per game and now reads `team_logos.csv` from repo root.
- CORS config: backend uses `ALLOWED_ORIGINS` and `ALLOW_ORIGIN_REGEX` to allow localhost and Vercel previews.
- Documentation: standardized file headers are now present across backend/frontend source files.
- Legacy router: `backend/routes.py` is mounted under `/legacy` to preserve older endpoint shapes.
- Inference alignment: `build_model_input_row` reindexes once and bulk-fills medians to avoid DataFrame fragmentation.
- Roll-forward fills: priors/onehots now use batched assignments to avoid fragmentation warnings during synthetic rows.
- Schedule ingestion: `_load_schedule_df` trims CSV headers to normalize `home_team`/`away_team`/`week` fields.
- Prediction endpoint map: `docs/PREDICTION_ENDPOINT_MAP.md` and `docs/prediction_endpoint_map.svg` document the /predict flow.
- Debug endpoint: `/debug` now uses `datetime`/`timezone` imports to prevent NameError during calls/tests.
- predictionHelpers: `frontend/src/utils/predictionHelpers.js` is now used in `Dashboard.jsx` for prediction normalization.
- Models directory: MODELS_DIR is set to `backend/20260102/models` for the latest artifacts.
- Prediction UI: dashboard now stores predictions using the schedule-derived key and fills missing game fields via `toEntry`.
- Debug visibility: `/debug/predict-input` reports missing/filled features, and the dashboard header shows the active models directory.
- Inference quality: roll-forward logic now copies rolling/player/elo features from the latest prior game for each team.
- Dataset alignment: DATA_DIR now defaults to `backend/data/datasets` (relative to backend) to match model feature expectations.
- Dataset enforcement: `DATASET_PATH` can pin the exact CSV; startup fails if model features are missing.
- Performance: per-team history cache avoids re-scanning the dataset for roll-forward fills.

## Key Documentation

- [Artifacts System](artifacts_README.md)
- [Last 5 Tasks](last_5_tasks.md)
- [Next 5 Tasks](next_5_tasks.md)
- [Dataflow Map](../dataflow.md)
