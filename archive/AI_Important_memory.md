# AI_Important_memory.md

## Repository Map (Condensed)
- Backend entrypoint: backend/main.py (FastAPI app, lifespan loading, primary routes).
- Backend helpers: backend/main_helpers.py (model/dataset loading, history persistence).
- Backend routes: backend/routes.py (legacy router, schedule + logo helpers).
- Backend services: backend/services/prediction_service.py (core inference), backend/services/inference_row.py (feature row build), backend/services/live_predictor.py (live data row build).
- Schemas: backend/schemas.py (Pydantic contracts).
- Frontend entry: frontend/src/index.jsx, frontend/src/App.jsx (React Router).
- Frontend data layer: frontend/src/api/fetch.js + client.js.
- Frontend state: frontend/src/hooks/usePredictionState.js.
- Frontend UI: Dashboard + Card/TeamGrid components; StatsPage at frontend/src/pages/StatsPage.jsx + StatsPage.css.

## Key Data Flows
- Startup: backend/main.py lifespan loads model bundle (MODELS_DIR) + dataset (DATA_DIR/DATASET_PATH) then validates feature schema.
- Prediction: /predict -> PredictionService -> build_model_input_row -> preprocessor + regressors + win classifier.
- Schedule: /schedule/next-week uses schedule CSV loaders and enriches with team logos.
- Team logos: /teams/logos returns parsed team_logos.csv (primary source) with logo + color fields.
- Frontend: usePredictionState loads schedule + history + logos; TeamGrid renders Card components with logos + colors.

## Deployment Notes
- Heroku expects Linux paths. Model metadata sometimes contains absolute Windows paths; path resolution must treat drive-letter paths as absolute and fall back to basename in MODELS_DIR.
- Vercel rootDirectory = frontend. Vercel env requires VITE_API_BASE_URL; dev base can use VITE_API_DEV or VITE_DEV_ENV.
- Build artifacts are committed (frontend/dist).

## Hotspots / Watchouts
- backend/main_helpers.py: path resolution for model artifacts; dataset selection vs expected feature list.
- backend/services/live_predictor.py: internal inference helper previously referenced missing functions from backend/main.py.
- frontend/src/pages/StatsPage.jsx + StatsPage.css: only location where styling changes should be applied per UI instructions.
- frontend/src/api/fetch.js: base URL selection must handle dev/prod env naming.

## Clean Architecture Intent
- Keep backend endpoints thin; push transformation logic into helper/services.
- Keep frontend data normalization in hooks/utils; UI components should remain presentational.
