# NFL ML Predictions

Production-ready FastAPI backend serving NFL ML predictions with a Vite/React frontend.

## Quickstart

Backend (FastAPI):
```
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend (Vite):
```
cd frontend
npm install
```

Create `frontend/.env` for local dev:
```
VITE_API_DEV=http://127.0.0.1:8000
```

Then run:
```
npm run dev
```

## Environment Variables

Backend:
- `MODELS_DIR`: Override models directory (must contain `metadata.json` + artifacts).
- `DATA_DIR`: Override dataset directory (default resolves to `backend/data/datasets`).
- `DATASET_PATH`: Force a specific dataset CSV path.
- `ALLOWED_ORIGINS`: Comma-separated CORS allowlist.
- `ALLOW_ORIGIN_REGEX`: Regex for dynamic origins (defaults to Vercel previews).
- `RESTRICT_CORS`: `true` to restrict to allowlist only.
- `OFFLINE_MODE`: `true` to skip live schedule fetch (use CSV fallback).
- `SCHEDULE_PATH`: Override schedule CSV path.
- `POSTSEASON_SCHEDULE_PATH`: Override postseason schedule JSON (default `backend/post_schedule.json`).
- `ENABLE_ADMIN`: `true` to enable `/admin/*` endpoints.
- `OLLAMA_HOST`: Override Ollama host for LLM endpoints.

Frontend:
- `VITE_API_BASE_URL`: Production API base URL (Vercel).
- `VITE_API_DEV` or `VITE_API_BASE_DEV`: Local/dev API base URL.

## API Endpoints (Core)

Health + Status:
- `GET /health`
- `GET /status/overview`
- `GET /status/models`
- `GET /debug`

Predictions:
- `POST /predict` `{ home_team, away_team, season, week }`
- `GET /predict/next-week`
- `POST /predict/explain`
- `POST /llm/chat`

Schedule + Teams:
- `GET /schedule/next-week`
- `GET /teams/logos`

History:
- `GET /history?limit=100`

Admin (guarded by `ENABLE_ADMIN=true`):
- `POST /admin/reload`
- `POST /admin/retrain`

Note: Legacy `/legacy/*` routes have been removed; use the root endpoints above.

## Data + Models

- Models are loaded from `MODELS_DIR` (default `backend/models`).
- Dataset defaults to the newest `game_features_*.csv` under `DATA_DIR`
  (default `backend/data/datasets`). Set `DATASET_PATH` for an explicit file.
- Prediction history is persisted in `backend/Predictions/prediction_history.json`.
- `backend/build_csv_datasets_v3.py` writes datasets to `backend/data/datasets` by default.
- `backend/post_schedule.json` is used when regular-season schedules are exhausted.

## Deployment

- Procfile uses `gunicorn` with `uvicorn.workers.UvicornWorker`.
- Configure CORS explicitly in production (`ALLOWED_ORIGINS` + `ALLOW_ORIGIN_REGEX`).
