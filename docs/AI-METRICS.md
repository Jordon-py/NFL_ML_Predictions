<<<<<<< HEAD
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
=======
# AI-METRICS.md

## Summary & Timestamp

- **Date:** October 24, 2025
- **Time:** 23:00 UTC
- **App Completion Percentage:** 90%
- **Enhancement Ideas:** Add user authentication, implement caching for predictions, add more detailed error handling, integrate real-time NFL data feeds.

## Variables Inventory

| Name | File | Line(s) | Type/Inferred | Notes |
|------|------|---------|---------------|-------|
| model_objects | backend/main.py | 108 | Optional[Dict[str, Any]] | Global variable holding loaded ML models and metadata |
| dataset_df | backend/main.py | 109 | Optional[pd.DataFrame] | Global variable holding the loaded dataset |
| DEFAULT_CORS_ORIGINS | backend/main.py | 117-123 | List[str] | Default CORS origins for development |
| CORS_ORIGINS | backend/main.py | 125-127 | List[str] | Configured CORS origins from environment |
| CORS_ORIGIN_REGEX | backend/main.py | 128 | str | CORS origin regex pattern |
| TEAM_ABBREVIATIONS | backend/main.py | 131-152 | Dict[str, str] | Mapping of full team names to abbreviations |
| TEAM_CODE_FIX | backend/main.py | 153 | Dict[str, str] | Legacy team code corrections |
| VALID_ABBRS | backend/main.py | 154 | set | Valid team abbreviations |
| THIS_FILE | backend/main.py | 78 | Path | Path to the current file |
| BACKEND_DIR | backend/main.py | 79 | Path | Backend directory path |
| BASE_DIR | backend/main.py | 80 | Path | Base project directory |
| DATA_DIR | backend/main.py | 81 | Path | Data directory path |
| MODELS_DIR | backend/main.py | 82 | Path | Models directory path |
| LOG_DIR | backend/main.py | 83 | Path | Logs directory path |
| DEFAULT_DATASET | backend/main.py | 87 | Path | Default dataset file path |
| DEFAULT_SCHEDULE | backend/main.py | 88 | Path | Default schedule file path |
| FRONTEND_DIR | backend/main.py | 90 | Path | Frontend directory path |
| FRONTEND_BUILD | backend/main.py | 91 | Path | Frontend build directory |
| FRONTEND_DIST | backend/main.py | 92 | Path | Frontend dist directory |
| TRUTHY | backend/main.py | 94 | set | Truthy string values for boolean conversion |
| SERVE_FRONTEND | backend/main.py | 95 | bool | Whether to serve frontend static files |

## Functions/Components Inventory

| Name | File | Line(s) | Signature/Props | Called by | Calls out to |
|------|------|---------|----------------|-----------|-------------|
| _normalize_feature_cols | backend/main.py | 156-158 | (cols: Dict[str, List[str]]) -> List[str] | _validate_dataset_schema | - |
| get_abbr | backend/main.py | 160-183 | (name: str) -> str | predict_game, get_next_week_schedule | - |
| load_objects | backend/main.py | 186-218 | () -> Dict[str, Any] | lifespan | joblib.load |
| _validate_dataset_schema | backend/main.py | 221-235 | (df: pd.DataFrame, model_objects: Dict[str, Any]) -> None | lifespan | - |
| _sanity_predict | backend/main.py | 238-295 | (model_objects: Dict[str, Any], df: pd.DataFrame) -> None | lifespan | - |
| _coerce_bool | backend/main.py | 298-310 | (s: pd.Series) -> pd.Series | _ensure_home_away | - |
| _ensure_home_away | backend/main.py | 313-332 | (df: pd.DataFrame) -> pd.DataFrame | lifespan | _coerce_bool |
| lifespan | backend/main.py | 335-356 | (app: FastAPI) -> AsyncGenerator[None, None] | FastAPI app | load_objects,_ensure_home_away, _validate_dataset_schema,_sanity_predict |
| health | backend/main.py | 359-367 | () -> HealthResponse | FastAPI route | - |
| debug_info | backend/main.py | 370-382 | () -> Dict[str, Any] | FastAPI route | - |
| report_training | backend/main.py | 385-389 | () -> Dict[str, Any] | FastAPI route | - |
| report_calibration | backend/main.py | 392-403 | () -> Dict[str, Any] | FastAPI route | - |
| build_game_mask | backend/main.py | 406-413 | (df: pd.DataFrame, season: int, week: int, home_abbr: str, away_abbr: str) -> pd.Series | predict_game | - |
| get_current_nfl_context | backend/main.py | 416-447 | () -> Dict[str, Any] | predict_next_week | - |
| _validate_features_present | backend/main.py | 450-453 | (feature_names: List[str], row: pd.Series) -> List[str] | - | - |
| _build_future_row | backend/main.py | 456-530 | (df: pd.DataFrame, home: str, away: str, season: int, week: int) -> pd.Series | predict_game | - |
| get_next_week_schedule | backend/main.py | 533-575 | () -> List[ScheduleGame] | FastAPI route | get_abbr |
| predict_game | backend/main.py | 578-720 | (payload: PredictionRequest) -> PredictionResponse | FastAPI route | get_abbr, build_game_mask,_build_future_row |
| predict_next_week | backend/main.py | 723-760 | () -> Dict[str, Any] | FastAPI route | get_current_nfl_context, predict_game |
| App | frontend/src/App.jsx | 18-25 | () -> JSX.Element | index.jsx | DashBoard |
| ErrorBoundary | frontend/src/components/ErrorBoundary.jsx | 8-42 | (props) -> JSX.Element | index.jsx, App.jsx | - |
| DashBoard | frontend/src/components/DashBoard.jsx | 15-42 | ({state}) -> JSX.Element | App.jsx | TeamGrid, PredictionResult, HistoryChart, NavBar |
| HistoryChart | frontend/src/components/HistoryChart.jsx | 6-40 | ({state, history}) -> JSX.Element | DashBoard.jsx, PredictionResult.jsx | - |
| NavBar | frontend/src/components/NavBar/NavBar.jsx | 9-87 | () -> JSX.Element | DashBoard.jsx | - |
| PredictionResult | frontend/src/components/PredictionResult.jsx | 25-75 | ({entry}) -> JSX.Element | DashBoard.jsx | HistoryChart |
| TeamGrid | frontend/src/components/TeamGrid.jsx | 17-314 | ({state}) -> JSX.Element | DashBoard.jsx | getNextWeekSchedule, predictGame |
| PredictionProvider | frontend/src/PredictionContext.jsx | 58-75 | ({children}) -> JSX.Element | index.jsx | - |
| usePredictions | frontend/src/PredictionContext.jsx | 78-82 | () -> {state, actions} | DashBoard.jsx, TeamGrid.jsx | - |
| toEntry | frontend/src/PredictionContext.jsx | 40-55 | (data) -> Object | PredictionProvider | - |

## Cross-File Usage Map

- backend/main.py → backend/models/ (joblib.load for models)
- backend/main.py → backend/data/ (pd.read_csv for datasets)
- frontend/src/App.jsx → frontend/src/components/DashBoard.jsx
- frontend/src/components/DashBoard.jsx → frontend/src/components/TeamGrid.jsx, PredictionResult.jsx, HistoryChart.jsx, NavBar/
- frontend/src/components/TeamGrid.jsx → frontend/src/api/client.js (getNextWeekSchedule, predictGame)
- frontend/src/PredictionContext.jsx → frontend/src/components/ (used by multiple components)
- frontend/src/index.jsx → frontend/src/App.jsx, PredictionContext.jsx, ErrorBoundary.jsx

## Risk Radar

| Issue | File:Line | Category | Likelihood/Impact | Rationale | Suggested Fix |
|-------|----------|----------|-------------------|-----------|---------------|
| Import resolution errors | backend/main.py:61-68 | runtime | High/Medium | VS Code shows import errors but packages are installed | Restart Python language server or reload window |
| Double context providers | index.jsx/App.jsx (fixed) | runtime | Medium/Low | Fixed: removed duplicate PredictionProvider wrapping | Already fixed |
| Missing CSS file | ErrorBoundary.css (fixed) | styling | Low/Low | Fixed: created missing ErrorBoundary.css | Already fixed |
| Malformed comments | PredictionResult.jsx (fixed) | style | Low/Low | Fixed: cleaned up commented code blocks | Already fixed |
| HistoryChart state handling | HistoryChart.jsx (fixed) | logic | Medium/Medium | Fixed: corrected history array access | Already fixed |
| Invalid API route | backend/main.py:533 (fixed) | runtime | High/High | Fixed: added leading slash to /schedule/next-week endpoint | Already fixed |
| Backend startup failure | backend/main.py:335-356 (fixed) | runtime | High/High | Fixed: modified sanity check to handle unfitted preprocessor | Already fixed |
| Frontend proxy misconfiguration | frontend/vite.config.js (fixed) | runtime | High/Medium | Fixed: updated proxy to target localhost:5000 | Already fixed |

## TODO/Aspirations

- Implement user authentication for personalized predictions
- Add caching layer for repeated predictions
- Enhance error handling with more specific error messages
- Add unit tests for frontend components
- Implement data validation for API inputs

## Changed since last run

- Fixed double-wrapping of PredictionProvider and ErrorBoundary in index.jsx/App.jsx
- Created missing ErrorBoundary.css file with proper styling
- Fixed HistoryChart.jsx to properly handle history array instead of stringifying state
- Cleaned up malformed comments in PredictionResult.jsx
- Updated .github/copilot-instructions.md with current repository state
- Created docs/AI-METRICS.md for comprehensive codebase inventory
- Added leading slash to /schedule/next-week endpoint in backend/main.py
- Updated Vite proxy configuration to target localhost:5000 for dev API calls
- Modified _sanity_predict function to handle unfitted preprocessor during startup
- Enabled dataset schema validation and sanity checks in lifespan function
- Enhanced TeamGrid matchup cards with team logos, improved visual layout, fade-in animations, outline glows, and enhanced standout effects for predicted cards
- Implemented responsive flexbox layout for TeamGrid cards and structured card content with proper spacing and no overlapping stats
- Fixed kickoff time display to use user's local timezone instead of Pacific Time
>>>>>>> f3c92a29d (Complete NFL prediction system: dataset engineering with ELO/rolling/QB features, model training pipeline, UI fixes, and production-ready artifacts)
