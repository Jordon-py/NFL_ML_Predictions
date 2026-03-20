# Environment Configuration

This repo runs a Vite frontend and FastAPI backend.

Backend env vars are centralized in `backend/app/core/settings.py`.
Frontend API base resolution is centralized in `frontend/src/api/client.js`.

## Backend Variables

| Name | Purpose | Default | Required in Prod | Example |
| --- | --- | --- | --- | --- |
| `APP_ENV` | Runtime environment label | `development` (local), `production` (Heroku dyno) | Yes | `production` |
| `LOG_LEVEL` | Backend log level | `INFO` | No | `INFO` |
| `RESTRICT_CORS` | Enforce allow-list CORS | `true` | Yes | `true` |
| `ALLOWED_ORIGINS` | Comma-separated exact browser origins | local+vercel defaults | Yes (if restricted) | `https://new-nfl-predict.vercel.app,http://localhost:5173` |
| `CORS_ORIGINS_REGEX` | Optional regex for preview origins | `^https://.*\.vercel\.app$` when preview support enabled | No | `^https://.*\.vercel\.app$` |
| `ALLOW_VERCEL_PREVIEWS` | Enable fallback preview regex | `true` | No | `true` |
| `DATASET_PATH` | Explicit dataset CSV path | auto-select latest `game_features*.csv` | No | `backend/data/game_features_latest.csv` |
| `SCHEDULE_PATH` | Explicit schedule CSV path | `backend/data/Nfl_schedule_2025.csv` | No | `backend/data/Nfl_schedule_2025.csv` |
| `MODELS_DIR` | Explicit model bundle directory | auto-discovery | No | `backend/models` |
| `PREDICT_CACHE_TTL_SEC` | Prediction cache TTL seconds | `900` | No | `900` |
| `PREDICT_CACHE_MAX_ITEMS` | Prediction cache max entries | `1000` | No | `1000` |
| `ENABLE_ADMIN` | Enable `/admin/*` routes | `false` | No | `true` |
| `ADMIN_TOKEN` | Bearer or `x-admin-token` for admin routes | empty | Required if admin exposed outside localhost | `<secret>` |

## Frontend Variables

| Name | Purpose | Default | Required in Prod | Example |
| --- | --- | --- | --- | --- |
| `VITE_API_BASE_URL` | Backend origin for API calls | `http://127.0.0.1:8000` in dev fallback | Yes | `https://nfl-predict-ecf5a5bd34fe.herokuapp.com` |
| `VITE_API_BASE_PATH` | Optional API prefix | empty | No | `/api` |

## Local Setup

1. Backend:
```bash
cp backend/.env.example backend/.env
```

2. Frontend:
```bash
cp frontend/.env.local.example frontend/.env.local
```

## Production Setup

1. Heroku (backend):
```bash
heroku config:set APP_ENV=production -a nfl-predict
heroku config:set RESTRICT_CORS=true -a nfl-predict
heroku config:set ALLOWED_ORIGINS=https://new-nfl-predict.vercel.app -a nfl-predict
heroku config:set CORS_ORIGINS_REGEX='^https://.*\.vercel\.app$' -a nfl-predict
```

2. Vercel (frontend):
```bash
vercel env add VITE_API_BASE_URL production
vercel env add VITE_API_BASE_PATH production
```

## Dataset Freshness

`backend/scripts/weekly_retrain.py` now rebuilds the dataset by default before training, then writes:

- `backend/reports/versioning/*`
- `backend/reports/drift/*`
- `backend/reports/automation/weekly_retrain_latest.json`

Use `--skip-dataset-build` only when intentionally reusing the current dataset.

