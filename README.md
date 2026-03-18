# NFL ML Predictions

Production-ready FastAPI backend serving NFL ML predictions with a Vite/React frontend.

## Quickstart

Backend (FastAPI):
```
cd backend
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend (Vite):
```
Last updated: March 10, 2026

NFL ML Predictions is a full-stack forecasting workspace for NFL matchups. It includes:

- A FastAPI backend for schedule loading, model inference, health/status, history, and LLM explanation endpoints.
- A React/Vite frontend with a premium landing page, local sign-in/sign-out, and protected dashboard routes.
- A dataset pipeline that builds clean game-level training data.
- A training pipeline that evaluates, calibrates, stores, and archives model artifacts.
- Per-user prediction history storage keyed to the signed-in frontend session.

## What Changed In This Refresh

- The landing page and protected app shell now support sign in and sign out.
- Prediction history is stored per user instead of one shared global ledger.
- The backend uses typed Pydantic contracts for API responses, chat/explain payloads, and persisted prediction records.
- `backend/builddataset.py` is now the canonical dataset build entrypoint.
- `backend/train_models.py` now defaults to the latest clean dataset, writes run manifests, archives model runs, and records a monthly in-season retraining cadence.
- The README now reflects the actual working scripts and storage layout.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 2. Build a clean dataset

```bash
python backend/builddataset.py --start 2018 --end 2025 --out-dir backend/data/datasets
```

What this does:

- Uses the current feature engineering logic in `backend/build_csv_datasets_v3.py`
- Writes a dated run under `backend/data/datasets/runs/<timestamp>/`
- Promotes the latest clean dataset to `backend/data/datasets/`
- Writes `latest_dataset.json` so training and runtime can discover the active dataset
- Supports `--with-calibration-rows` when you need legacy compatibility rows for older workflows

### 3. Train models

```bash
python backend/train_models.py
```

What this does:

- Uses the latest clean dataset by default
- Trains regression and win-probability models
- Writes active artifacts to `backend/models/`
- Archives the full run under `backend/models/runs/<timestamp>/`
- Writes `latest_training_run.json` and `training_schedule.json`
- Stores a canonical inference feature contract in `metadata.json` so runtime inference stays aligned

### 4. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Start the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## User Guide

### Sign in

- Sign in with your email and a password (six characters or more). The identity stays local to this device so you can return to the dashboard instantly.
- Your email is the key for per-user prediction history—everything you forecast is tied to that identity.

### Browse slates

- Use the week/season controls on the dashboard to pivot between the next live slate, archived weeks, or past seasons. The grid updates to match the slate you pick so there is always something to explore.
- Final scores sync automatically every Sunday, Monday, and Thursday night, so the matchup cards and History page always include the latest official results.

### Forecast & compare

- Tap any matchup, run the prediction, and the system saves your forecast in the backend (via the SQLite ledger under `backend/predictions.db`) and in your browser cache for fast access.
- Once final scores arrive, the cards and History chart compare your prediction to the actual outcome so you can see how each call landed.

## Developer Guide

## Canonical Entry Points

Use these scripts and endpoints as the source of truth:

- Dataset build: `python backend/builddataset.py`
- Model training: `python backend/train_models.py`
- Backend runtime: `uvicorn backend.main:app --reload --port 8000`
- Frontend runtime: `cd frontend && npm run dev`

Avoid older README references to `backend/build_csv_datasets.py`. The active builder is wrapped by `backend/builddataset.py`.

## Operator Endpoints

When `ENABLE_ADMIN=true`, the backend also exposes:

- `POST /admin/reload` to reload the active dataset and model artifacts without restarting the server
- `POST /admin/retrain` to run `backend/train_models.py` against the active dataset and hot-reload the result

`POST /admin/retrain` respects the monthly in-season freshness window by default. Pass `force: true` if you need to retrain immediately during the same freshness window.

## Repository Layout

```text
backend/
  builddataset.py              Canonical dataset build entrypoint
  build_csv_datasets_v3.py     Core feature engineering builder
  train_models.py              Model training + archiving entrypoint
  main.py                      FastAPI application
  schemas.py                   API Pydantic models
  pipeline_models.py           Pipeline/storage Pydantic models
  prediction_store.py          Per-user prediction history storage
  data/
    datasets/
      latest_dataset.json
      game_features_*_clean.csv
      runs/<timestamp>/
  models/
    metadata.json
    training_report.json
    training_schedule.json
    latest_training_run.json
    runs/<timestamp>/
  Predictions/
    users/<user-storage-key>/
      profile.json
      predictions.json

frontend/
  src/
    App.jsx
    hooks/useAuthSession.js
    hooks/usePredictionState.js
    pages/LandingPage.jsx
    components/DashBoard/Dashboard.jsx
```

## Backend API

### Core endpoints

- `GET /health`
- `GET /status/overview`
- `GET /status/models`
- `GET /schedule/next-week`
- `GET /teams/logos`
- `GET /history?limit=N`
- `POST /predict`
- `POST /predict/explain`
- `POST /llm/chat`

### User-scoped history

Prediction history is keyed by the `X-User-Id` request header.

- The frontend sends this automatically for prediction/history/status calls.
- If the header is missing, the backend falls back to an `anonymous` ledger.

### Example prediction request

```json
{
  "home_team": "KC",
  "away_team": "BUF",
  "season": 2025,
  "week": 15
}
```

## Data Pipeline

### Dataset outputs

Every `builddataset.py` run creates:

- A raw dated dataset inside a run directory
- A cleaned promoted dataset in `backend/data/datasets/`
- `dataset_manifest.json` in the run directory
- `latest_dataset.json` in `backend/data/datasets/`
- `build_csv_datasets.log` in the run directory
- Builder logs and metadata files generated by the underlying builder

### Cleaning rules added by the wrapper

- Strip BOM and whitespace from headers
- Drop fully blank rows
- Ensure `game_id` exists when enough schedule context is present
- Deduplicate by `game_id` using row completeness
- Sort the final dataset consistently by `season`, `week`, and `game_id`

## Training Pipeline

### Training outputs

Every `train_models.py` run now writes:

- Active model artifacts in `backend/models/`
- `training_report.json`
- `metadata.json`
- `training_schedule.json`
- `latest_training_run.json`
- `train_models_<timestamp>.log`
- `run_manifest.json` in the archived run folder
- Archived copies in `backend/models/runs/<timestamp>/`

### Retraining cadence

The training pipeline records a monthly in-season policy:

- In season: August through February
- Cadence: monthly
- Output: `training_schedule.json` contains the last train time and next recommended refresh

This does not yet start an OS scheduler automatically. It makes the schedule explicit and machine-readable for operators, tasks, or CI.

## Prediction Storage

### Backend

Per-user predictions are stored under:

```text
backend/Predictions/users/<user-storage-key>/predictions.json
```

Each record is validated by Pydantic before it is written.

### Frontend

Local browser cache is stored per user:

```text
prediction_history:<email>
```

This prevents one signed-in user from inheriting another user's cached history in the same browser.

## Pydantic Usage

Pydantic is now used across more than the HTTP API:

- API request and response schemas in `backend/schemas.py`
- Persisted prediction records in `backend/schemas.py`
- Dataset and training run manifests in `backend/pipeline_models.py`

This keeps runtime contracts, disk artifacts, and operator-facing metadata consistent.

The training metadata now records both:

- Canonical artifact names (`home_model`, `away_model`, `win_clf_calibrated`)
- A union `feature_names` contract used by runtime inference to keep score models and the win classifier aligned

## Verification

Recommended checks after pipeline or API changes:

```bash
python -m compileall backend frontend/src
pytest
cd frontend && npm run build
```

Runtime smoke checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status/overview -H "X-User-Id: analyst@example.com"
```

## Known Runtime Note

If `/health` reports an unhealthy preprocessing smoke test with a scikit-learn artifact mismatch, the server can still boot but model inference may be unreliable until:

- the environment is aligned to the artifact version in `requirements.txt`, or
- models are retrained in the current environment

## Troubleshooting

### Frontend shows CORS errors

- Run the frontend on `http://localhost:3000` or `http://localhost:5173`
- Confirm backend CORS settings in `backend/config.py`

### History appears empty after sign-in

- Make sure the frontend is signed in with an email
- Confirm the backend receives `X-User-Id`
- Check `backend/Predictions/users/` for the user ledger

### Training uses the wrong dataset

- Inspect `backend/data/datasets/latest_dataset.json`
- Override the dataset manually:

```bash
python backend/train_models.py --data backend/data/datasets/game_features_20260310_clean.csv
```

### Admin retrain reports that the active model is still current

- This is expected during the monthly in-season freshness window
- Use `python backend/train_models.py --force` or send `{"force": true}` to `POST /admin/retrain` when you intentionally want to retrain early

## Next Recommended Improvements

- Add real authentication instead of local-only sessions
- Add an automated scheduled job for monthly in-season retraining
- Add evaluation dashboards for archived model runs
- Add migration logic for old shared history files if they need to be preserved
