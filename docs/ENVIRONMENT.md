# Environment Configuration

This repo has two configuration entrypoints:

- Backend settings live in `backend/app/core/settings.py`.
- Frontend API resolution lives in `frontend/src/api/client.js`.

The goal of this document is to explain the active configuration behavior, not every legacy variable ever used in the repo.

## Backend Variables

| Name | Purpose | Default behavior | Example |
| --- | --- | --- | --- |
| `APP_ENV` | Runtime environment label | `development` locally, `production` on Heroku | `production` |
| `LOG_LEVEL` | Backend log level | `INFO` | `DEBUG` |
| `ALLOWED_ORIGINS` | Exact browser origins for CORS | Localhost + production defaults | `https://new-nfl-predict.vercel.app,http://localhost:3000` |
| `CORS_ORIGINS_REGEX` | Optional regex for preview domains | `^https://.*\\.vercel\\.app$` when preview support is enabled | `^https://.*\\.vercel\\.app$` |
| `RESTRICT_CORS` | Enforce allow-list CORS | `true` in code and in the canonical deploy setup | `true` |
| `ALLOW_VERCEL_PREVIEWS` | Enable preview-domain regex fallback | `true` | `true` |
| `DATASET_PATH` | Explicit dataset CSV override | Auto-discovers the promoted dataset from `backend/data/datasets/` | `backend/data/datasets/game_features_20260325_clean.csv` |
| `SCHEDULE_PATH` | Explicit schedule CSV override | Otherwise tries live schedule loading and then packaged CSV fallback | `data/Nfl_schedule_2025.csv` |
| `MODELS_DIR` | Explicit model bundle directory | Otherwise uses runtime auto-discovery | `data/models` |
| `TEAM_LOGOS_PATH` | Optional logos/name/color catalog | Auto-detects `backend/data/team_logos.csv` when present | `backend/data/team_logos.csv` |
| `PREDICT_CACHE_TTL_SEC` | Prediction cache TTL seconds | `900` | `900` |
| `PREDICT_CACHE_MAX_ITEMS` | Prediction cache max entries | `1000` | `1000` |
| `ENABLE_ADMIN` | Expose `/admin/*` routes | `false` | `true` |
| `ADMIN_TOKEN` | Token for admin routes | empty | `<secret>` |

## Runtime Path Resolution

### Model bundle resolution

If `MODELS_DIR` is not set, the backend resolves bundles in this order:

1. `backend/data/models/current`
2. `backend/data/models`
3. Other packaged complete bundles such as `backend/data/prod-models/models`
4. Dated bundles like `backend/YYYYMMDD/models`
5. `backend/models` as the final visible fallback

Why this matters:

- `backend/train_models.py` writes to `backend/models/` by default.
- Deployments usually point `MODELS_DIR` at `data/models` or ship a promoted bundle into `backend/data/models`.
- The runtime chooses the best complete serving bundle without assuming the newest training artifact is automatically deployed.

### Schedule resolution

For schedule endpoints, the backend tries:

1. Live schedule loading via `nflreadpy`
2. An explicit `SCHEDULE_PATH` override
3. Packaged schedule CSVs under `backend/data/`
4. Frontend public CSVs as a last local-development fallback

This is why `/schedule/next-week` can still return data in degraded or offline scenarios.

## Model Compatibility Notes

The active serving bundle can be older than the local Python environment.

- The backend records warnings when bundle metadata or scikit-learn versions drift.
- Startup no longer crashes on bundle incompatibility.
- `/predict` is the endpoint that becomes unavailable when required model artifacts are not ready.

Operational rule:

- If you upgrade scikit-learn or change training features, retrain and republish the serving bundle before treating the deployment as production-ready.

## Frontend Variables

| Name | Purpose | Default behavior | Example |
| --- | --- | --- | --- |
| `VITE_API_BASE_URL` | Backend origin used by production frontend builds | empty locally unless you intentionally test a remote API | `https://nfl-predict-ecf5a5bd34fe.herokuapp.com` |
| `VITE_API_BASE` | Older alias still honored by `client.js` | empty | `https://example.com` |
| `VITE_API_URL` | Older alias still honored by `client.js` | empty | `https://example.com` |
| `VITE_API_DEV` | Preferred local dev backend origin for the active client and legacy helper | empty | `http://127.0.0.1:8000` |
| `VITE_DEV_ENV` | Legacy alias for local builds | empty | `http://127.0.0.1:8000` |

Important frontend note:

- The main app uses `frontend/src/api/client.js`.
- `frontend/src/api/fetch.js` still exists, but it is not the primary transport path for the active app shell.

## Local Setup

### Backend

```bash
cp backend/.env.example backend/.env
```

Recommended local values:

```env
APP_ENV=development
MODELS_DIR=data/models
SCHEDULE_PATH=data/Nfl_schedule_2025.csv
```

### Frontend

```bash
cp frontend/.env frontend/.env.local
```

Recommended local value:

```env
VITE_API_DEV=http://127.0.0.1:8000
VITE_API_BASE_URL=https://nfl-predict-ecf5a5bd34fe.herokuapp.com
```

## Production Setup

### Heroku backend

```bash
heroku config:set APP_ENV=production -a nfl-predict
heroku config:set RESTRICT_CORS=true -a nfl-predict
heroku config:set ALLOWED_ORIGINS=https://new-nfl-predict.vercel.app -a nfl-predict
heroku config:set ALLOW_VERCEL_PREVIEWS=true -a nfl-predict
heroku config:set MODELS_DIR=data/models -a nfl-predict
heroku config:set SCHEDULE_PATH=data/Nfl_schedule_2025.csv -a nfl-predict
heroku config:set TEAM_LOGOS_PATH=backend/data/team_logos.csv -a nfl-predict
heroku config:set CORS_ORIGINS_REGEX='^https://.*\\.vercel\\.app$' -a nfl-predict
```

### Vercel frontend

```bash
vercel env add VITE_API_BASE_URL production
vercel env add VITE_API_BASE_URL preview
vercel env add VITE_API_BASE_URL development
```

Canonical Vercel layout:

- Project: `nfl-ml-predictions`
- Root Directory: `frontend/`
- Canonical production alias: `https://new-nfl-predict.vercel.app`
- Canonical backend origin: `https://nfl-predict-ecf5a5bd34fe.herokuapp.com`
- Frontend-only env vars in Vercel; do not store backend dataset/model paths there

## Operator Checks

Use these after config or deployment changes:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status/models
curl http://127.0.0.1:8000/schedule/next-week
```

If prediction traffic matters, also verify:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"home_team\":\"KC\",\"away_team\":\"BUF\",\"season\":2025,\"week\":15}"
```
