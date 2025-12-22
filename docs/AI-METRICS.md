# AI-METRICS.md

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
