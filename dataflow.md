# Dataflow Map — NFL Prediction App

This document maps the flow of data across the NFL Prediction App, from frontend interactions to backend processing and data storage.

## 1. High-Level Architecture

- **Frontend**: React (Vite) + Context API for state management.
- **Backend**: FastAPI (Python) for ML inference and data serving.
- **Data**: CSV datasets + joblib-serialized ML models.

## 2. Dynamic Data Flows

### A. Game Prediction Flow

1. **Trigger**: User clicks "Predict" on a matchup card in `TeamGrid.jsx`.
2. **Frontend Call**: `api/client.js` -> `predictGame(payload)` sends POST to `/predict`.
3. **Backend Logic (`main.py`)**:
   - `predict_game(payload)` receives request.
   - `infer_prediction_from_dataset` finds matching game in `dataset_df`.
   - **Feature Engineering**: Rows are preprocessed using `InferenceBundle.preprocessor`.
   - **Model Inference**:
     - Home/Away score regressors predict final scores.
     - Win classifier predicts win probability (or uses fallback if missing features).
4. **Response**: Backend returns `PredictionResponse` (scores, probabilities, fallback status).
5. **State Update**: `TeamGrid.jsx` updates local state; `PredictionContext.js` updates `current` prediction and `history`.
6. **Persistence**: Backend appends prediction to `prediction_history.json`.

### B. Schedule Flow

1. **Trigger**: Dashboard mount.
2. **Frontend Call**: `api/client.js` -> `getNextWeekSchedule()` sends GET to `/schedule/next-week`.
3. **Backend Logic (`main.py`)**:
   - Fetches live data using `nflreadpy.fetch_nfl_schedule()`.
   - Fallback: Reads from local `NFL_Schedule.csv` if live fetch fails.
4. **Response**: Array of normalized game objects.

### C. Health & Metrics Flow

1. **Trigger**: `useDashboardEngine` hook (polling).
2. **Frontend Call**: `api/client.js` -> `getHealth()` sends GET to `/health`.
3. **Backend Logic (`main.py`)**:
   - Returns uptime, model status, dataset stats, and basic metrics.
4. **Response**: Health object displayed in Dashboard header.

## 3. Data Storage & Schema

- **Datasets**: `backend/data/*.csv` (Home/Away stats, EPA metrics).
- **Models**: `backend/models/*.joblib` (RandomForest/XGBoost regressors and classifiers).
- **History**: `backend/data/prediction_history.json`.
- **Metadata**: `backend/models/metadata.json` (defines artifact paths).

## 4. Environment Configuration

- `VITE_API_BASE_URL`: Frontend API target.
- `MODELS_DIR`: Backend model artifact location.
- `DATA_DIR`: Backend dataset location.
- `ALLOW_FALLBACK_PREDICTIONS`: Enables heuristic predictions when models fail.
