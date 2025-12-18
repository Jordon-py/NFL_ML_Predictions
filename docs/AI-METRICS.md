# AI-METRICS.md

<<<<<<< HEAD
Last updated: 2025-11-02 18:45 UTC

App completion estimate: 98% (prod-ready; pending final smoke on varied predictions after latest fix)

Enhancement queue (next 2–3 sprints):

- Add prediction caching (in-memory/Redis) and simple request throttling
- Expand pytest suite for feature-builder and /predict payload validation
- Add CI smoke (health, predict, CORS) and minimal e2e via scripts/smoke_screenshots.js

## Metrics registry (concise)

- Variables (core globals): model_objects, dataset_df, DEFAULT_SCHEDULE, DEFAULT_DATASET, MODELS_DIR, DATA_DIR
- Functions (backend/main.py): lifespan, health, debug_info, report_training, report_calibration, get_current_nfl_context, get_next_week_schedule, build_game_mask, predict_game, predict_next_week, _build_future_row,_validate_dataset_schema, _sanity_predict, _validate_features_present, get_abbr
- Interactions: frontend calls /schedule/next-week and /predict; backend loads models via joblib, aligns features to estimator.feature_names_in_, sanitizes NaN/inf; returns PredictionResponse with provenance in prediction_source
- Data shapes: request {home_team, away_team, season, week}; response {home_score, away_score, home_win_probability, away_win_probability, point_diff, prediction_source, mode}

## Variables inventory (selected)

| Name | File | Type | Purpose |
|------|------|------|---------|
| model_objects | backend/main.py | Dict[str, Any] | Loaded artifacts: preprocessor, home/away regressors, calibrated win classifier, metadata |
| dataset_df | backend/main.py | pd.DataFrame | In-memory dataset for context/priors and schedule helpers |
| DEFAULT_DATASET | backend/main.py | Path | Default engineered dataset path (env-overridable) |
| DEFAULT_SCHEDULE | backend/main.py | Path | Default schedule CSV (env-overridable; resolver also finds latest Nfl_schedule_*.csv) |
| ALLOWED_ORIGINS | backend/main.py | List[str] | CORS allow-list parsed from env when RESTRICT_CORS=true |

## Functions inventory (grouped)

Backend service (FastAPI):

- lifespan(app): Load artifacts, dataset, run schema/feature validations, sanity predict
- health(): Returns status/mode and “models loaded” reason on success
- debug_info(): Dumps key runtime info (origins, features, artifact presence)
- report_training()/report_calibration(): Serve training/calibration JSON summaries

Prediction path:

- get_current_nfl_context(): Compute canonical season/week based on current date
- get_next_week_schedule(): Normalize schedule rows (abbrs/time), return upcoming games
- build_game_mask(df, season, week, home, away): Select matching game rows in dataset
- _build_future_row(df, home, away, season, week): Assemble engineered one-row feature vector using priors, trends, diffs, and cumulative pre_ metrics
- _validate_features_present(feature_names, row): Minimal identifier guard (home_team, away_team, home_game_date)
- _sanity_predict(model_objects, df): Tiny inference to validate pipeline after startup
- predict_game(payload): Align features to estimator expectations, sanitize values, infer scores and win prob; computes provenance prediction_source
- predict_next_week(): Batch predict over schedule for the upcoming week

Frontend API surface (client.js):

- getNextWeekSchedule(): GET /schedule/next-week
- predictGame(payload): POST /predict

## Data shapes

Request (POST /predict):
{
 home_team: string,  // NFL abbr
 away_team: string,  // NFL abbr
 season: number,
 week: number
}

Response (PredictionResponse):
{
 home_score: number,
 away_score: number,
 home_win_probability: number,
 away_win_probability: number,
 point_diff: number,
 prediction_source: "model" | "model+win_fallback" | "feature_fallback(+win_fallback)",
 mode: "production" | "development"
}

## Cross-file usage map

- backend/main.py → backend/models/* (joblib.load)
- backend/main.py → backend/data/* (pd.read_csv)
- frontend/src/api/client.js → FastAPI routes (/schedule/next-week, /predict)
- frontend/src/components/TeamGrid.jsx → client calls, shows prediction_source

## Risk radar (current view)

- 🟢 Startup validations: active; fail-fast on missing/invalid artifacts
- 🟡 Feature assembly edge cases: sparse datasets can still trigger feature_fallback; covered by imputer and guards
- 🟢 CORS: env-driven allow-list validated; Heroku config tested
- 🟡 Performance: batch /predict could benefit from caching and shared transforms

## Recent change highlights

- Fixed uniform predictions by consolidating _build_future_row.pre_cum return into a single dict that includes games, wins, win_rate, win_rate_l3, win_rate_l5 (previous early-return only included games → identical vectors)
- Strengthened classifier input alignment/sanitization to keep provenance = "model" (use feature_names_in_, fill NaN/±inf)
- Robust schedule resolution (env → default → latest matching file)

## Session log capsule

- 2025-11-02: pre_cum bug fixed; commit recorded; pending verification of varied predictions across matchups
- 2025-11-01: retrained classifier with leakage guard; metadata aligned; improved fallback behavior

## Suggestions (near-term)

- Add two unit tests: (1) pre_cum metrics length/keys invariant; (2) predict_game returns provenance "model" given a minimal synthetic row
- Add lightweight response cache keyed by (home, away, season, week)

## Appendix: Minimal contract

- Inputs: home_team, away_team, season, week (strings for teams per abbrs; ints for season/week)
- Output: scores, probabilities, provenance, mode
- Errors: 400 on missing identifiers; 503 on server schedule/data unavailability; 500 only on unexpected runtime failures
=======
Summary & Timestamp

- Updated: 2025-12-11 04:30 UTC. Fixed CORS preflight 400s by parsing ALLOWED_ORIGINS into a list and adding a catch-all OPTIONS responder. Predict now feeds raw feature columns to pipelines, eliminating constant 23.1/20.7 scores. App completion estimate: 98% (pending logo check + /history stub).

Variables Inventory

| name | file | line(s) | type/inferred | notes |
|------|------|---------|---------------|-------|
| model_objects | [backend/main.py](backend/main.py#L53-L59) | Dict[str, Any] | Loaded preprocessor/home/away/win models + metadata |
| dataset_df | [backend/main.py](backend/main.py#L54-L59) | pd.DataFrame | In-memory dataset for feature builders |
| MODELS_DIR | [backend/main.py](backend/main.py#L62-L73) | Path | Defaults to backend/data/prod-models/models; env override |
| DEFAULT_DATASET | [backend/main.py](backend/main.py#L75-L77) | Path | Engineered features CSV (20251210) |
| ALLOWED_ORIGINS | [backend/main.py](backend/main.py#L75-L111) | List[str] | Parsed allow list with localhost+Vercel defaults |
| ALLOW_ORIGIN_REGEX | [backend/main.py](backend/main.py#L107-L111) | str | Regex fallback for preview domains |
| ALLOW_FALLBACK_PREDICTIONS | [backend/main.py](backend/main.py#L112-L114) | bool | Gate for feature/win fallbacks |

Functions/Components Inventory

| name | file | line(s) | signature/props | called by | calls out to |
|------|------|---------|-----------------|-----------|--------------|
| lifespan | [backend/main.py](backend/main.py#L300-L520) | (app) -> AsyncGenerator | Startup loader (models/dataset), sanity checks | FastAPI app | joblib, pandas, _sanity_predict |
| preflight_ok | [backend/main.py](backend/main.py#L563-L571) | (rest_of_path: str) -> Response | Always 200 OPTIONS to avoid preflight 400s | CORS middleware chain | Response |
| health | [backend/main.py](backend/main.py#L1033-L1050) | () -> HealthResponse | Reports status/mode/reason | HTTP GET /health | model_objects presence |
| predict_game | [backend/main.py](backend/main.py#L1185-L1525) | (PredictionRequest) -> PredictionResponse | Assembles features, predicts scores/probabilities | HTTP POST /predict | _build_future_row,_predict_with_fill, _predict_proba_with_fill |
| get_next_week_schedule | [backend/main.py](backend/main.py#L1065-L1160) | () -> FullSchedule | Returns schedule with optional predictions | HTTP GET /schedule/next-week | nfl, predict_game |

Cross-File Usage Map

- [backend/main.py](backend/main.py) → joblib models under backend/data/prod-models/models; dataset CSV backend/data/prod-models/game_features_20251210.csv.
- [frontend/src/api/client.js](frontend/src/api/client.js) → backend /predict, /schedule/next-week.
- [scripts/test_endpoints.py](scripts/test_endpoints.py) exercises /schedule/next-week, /predict, /history (stub missing).

Risk Radar

- Bug/runtime: /history still unimplemented; callers will 404/501 unless stubbed.
- Bug/runtime: If ALLOWED_ORIGINS env set to a single string with commas, stripping is correct; but credentials remain disabled—confirm if cookies are ever needed.
- Style: predict_game still complex; consider extracting feature builders for testing coverage.

TODO/Aspirations

- Add smoke test for OPTIONS preflight on /health and /history when stubbed.
- Implement prediction history or respond 501 with guidance for UI clients.

Changed since last run

- Parsed ALLOWED_ORIGINS into a proper list and added catch-all OPTIONS handler to prevent 400 preflights [backend/main.py](backend/main.py#L75-L111, backend/main.py#L563-L571).
- Removed transformed-column alignment in win prob path; pipelines now consume raw feature names, eliminating constant predictions [backend/main.py](backend/main.py#L1375-L1495).
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
