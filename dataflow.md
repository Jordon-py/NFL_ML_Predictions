# Dataflow Map - NFL Prediction App

This document maps the flow of data across the NFL Prediction App, from frontend interactions to backend processing and data storage.

## 1. High-Level Architecture

- **Frontend**: React (Vite) with `usePredictionState` in `App.jsx` and prop-driven state.
- **Backend**: FastAPI (`backend/main.py`) for ML inference, schedule service, LLM endpoints, plus legacy routes mounted under `/legacy`.
- **Data**: CSV datasets + joblib-serialized ML models.
- **Metadata**: `team_logos.csv` (repo root) or `backend/team_logos.csv` for team names/logos.

## 2. Dynamic Data Flows

### A. Game Prediction Flow

1. **Trigger**: User clicks a matchup card in `TeamGrid.jsx`.
2. **Frontend Call**: `api/client.js` -> `predictGame(payload)` sends POST to `/predict`.
3. **Backend Logic (`main.py`)**:
   - `predict(req)` calls `PredictionService.predict`.
   - `build_model_input_row` rolls forward team stats from the latest prior game, then aligns inputs to the model schema and fills numeric gaps from dataset medians.
   - Response is flattened into `UnifiedPredictionResponse` (home/away scores + probabilities).
4. **Response**: Unified, flat prediction payload (single shape used by UI components).
5. **State Update**: `Dashboard.jsx` normalizes new predictions via `toEntry` and pushes them into history.
6. **Persistence**: Backend appends a flat entry to `backend/Predictions/prediction_history.json`.

### B. Schedule Flow

1. **Trigger**: `usePredictionState` initial load.
2. **Frontend Call**: `getNextWeekSchedule()` sends GET to `/schedule/next-week`.
3. **Backend Logic (`main.py`)**:
   - Loads schedule (nflreadpy or CSV fallback) and trims schedule CSV headers.
   - Infers next week and enriches each game with `game_id`, `home_name`, `away_name`, and logo URLs.
4. **Response**: `{ games: [ ... ] }` with enriched schedule rows.
5. **State Update**: Schedule is normalized and stored in frontend state.

### C. Health + Status Flow

1. **Trigger**: `usePredictionState` polling loop.
2. **Frontend Call**: `/api/health` every 15s; `StatsPage.jsx` calls `/api/status/overview` and `/api/history`.
3. **Backend Logic (`main.py`)**: Returns model status, dataset stats, and history counts.
4. **Response**: Health/status payloads used for UI banners and metrics.

### D. Explain + Chat Flow

1. **Trigger**: User clicks "Explain This Prediction" or sends a chat message.
2. **Frontend Call**: `/api/predict/explain` or `/api/llm/chat` with optional prediction context.
3. **Backend Logic**: Uses Ollama integration to generate explanation/chat response.

### E. Legacy Router Flow

1. **Trigger**: Backward-compatible clients call legacy endpoints.
2. **Backend Entry**: `/legacy/*` routes (mounted from `backend/routes.py`).
3. **Behavior**: Returns older response shapes (nested prediction payloads, batch predictions) without altering unified endpoints.

### F. Debug Feature Fill Flow

1. **Trigger**: Developer posts a game context to `/debug/predict-input`.
2. **Backend Logic**: Builds the model input row and reports which columns were missing or median-filled.
3. **Response**: `{ models_dir, prediction_source, debug }` used to verify model artifacts and feature coverage.

## 3. Data Storage & Schema

- **Datasets**: `backend/data/*.csv` (engineered feature sets).
- **Models**: `backend/models/*.joblib` (regressors/classifiers).
- **History**: `backend/Predictions/prediction_history.json` (flat prediction entries with `ts`).
- **Metadata**: `backend/models/metadata.json` (artifact paths).

## 4. Environment Configuration

- `VITE_API_BASE_URL`: Frontend API target.
- `MODELS_DIR`: Backend model artifact location.
- `DATA_DIR`: Backend dataset location.
- `DATASET_PATH`: Optional explicit dataset file path (enforced single source).
- `OFFLINE_MODE`: Forces CSV schedule fallback.
- `ALLOWED_ORIGINS`: Comma-separated origins allowed by FastAPI CORS middleware.
- `ALLOW_ORIGIN_REGEX`: Regex for preview origins (e.g., `https://.*\.vercel\.app`).

## 5. Startup Validation

- Backend startup now validates that model feature names exist in the dataset.
- If features are missing, startup fails fast to prevent silent median-only predictions.

## 6. Reference Maps

- `docs/PREDICTION_ENDPOINT_MAP.md` provides a focused /predict endpoint map with line references.
