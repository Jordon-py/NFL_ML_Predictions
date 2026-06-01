# NFL ML Predictions README

```text
# ==========================================
# File: README.md
# Role: Project entry point and operator guide.
# Input Data: N/A
# Output Data: Setup instructions, deploy targets, and architecture overview.
# Dependencies: N/A
# Notes: Keep this aligned with real deploy targets and current setup steps.
# ==========================================
```

## NFL ML Predictions

Full-stack NFL forecasting workspace with a FastAPI backend, a React/Vite frontend, a dataset build pipeline, and a model training pipeline.

## Current Production Data Snapshot

- Active dataset target: `backend/data/datasets/game_features_20260531_clean.csv`
- Active schedule artifact: `backend/data/Nfl_schedule_2026.csv`
- Active model bundle: `backend/models` version `20260531T124903Z-prod-2026`
- Active dataset hash: `94bd8ca5e7e47ac5db5d4d583daaa93265313be24a20bf909848db68a18f188b`
- Dataset seasons: 2018-2026
- Future-game coverage: 272 leak-safe 2026 regular-season rows
- Future-row rule: scheduled games may include market/schedule context, but final scores and target columns stay null until completed
- Model readiness rule: `/predict` should only serve when `/health/pipeline` reports no blockers and the model bundle dataset hash matches `latest_dataset.json`

## Canonical Deploy Targets

- GitHub source branch: `master`
- Frontend: Vercel project `nfl-ml-predictions`
- Production frontend alias: `https://new-nfl-predict.vercel.app`
- Backend: Heroku app `nfl-predict`
- Canonical backend origin: `https://nfl-predict-ecf5a5bd34fe.herokuapp.com`

Deploy intent:

- GitHub Actions deploys from `master`
- Vercel should build from `frontend/`
- Heroku should serve the FastAPI backend with the buildpack + `Procfile` flow
- Production CORS should allow the canonical frontend origin plus `.vercel.app` previews

## What This Repo Actually Does

- Serves NFL schedule, health, status, prediction, and history endpoints from `backend/main.py`.
- Stores user-scoped prediction history in SQLite first, with JSON files as a fallback.
- Builds cleaned training datasets into `backend/data/datasets/`.
- Trains score and win-probability models and promotes bundles for serving.
- Ships a React app with a protected dashboard, history view, and status page.

## Quick Start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 2. Ingest the upcoming schedule

```bash
python -m backend.services.schedule_ingestion --season 2026 --season-types 2,3 ^
  --out-csv backend/data/Nfl_schedule_2026.csv ^
  --out-parquet backend/data/schedules/nfl_schedule_2026.parquet ^
  --raw-dir backend/data/raw/espn/scoreboards
```

The schedule ingestion layer keeps future games leak-safe by leaving scores null for non-completed games.

### 3. Build the canonical dataset

```bash
python backend/builddataset.py --start 2018 --end 2026 --out-dir backend/data/datasets --encode onehot --no-calibration-rows
```

What this writes:

- A dated run folder in `backend/data/datasets/runs/<timestamp>/`
- A promoted clean CSV in `backend/data/datasets/`
- Completed and future partitions in `backend/data/datasets/`
- `backend/data/datasets/latest_dataset.json`

### 4. Train models

```bash
python backend/train_models.py --data backend/data/datasets/game_features_20260531_clean.csv --out backend/models --production
```

What this writes by default:

- Promoted artifacts in `backend/models/`
- A staging bundle in `backend/models/staging/<run_id>/`
- `metadata.json`, `training_report.json`, and `run_summary.json`
- A dated mirror in `backend/YYYYMMDD/models/` when training uses the default output directory

Important runtime note:

- Training still writes to `backend/models/` by default.
- Serving prefers `MODELS_DIR` when set, then `backend/data/models/current`, then `backend/data/models`, then packaged fallbacks, and finally `backend/models`.
- That split is intentional so deployments can serve a promoted bundle while local training experiments stay isolated.

### 5. Start the backend

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 6. Start the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## Runtime Behavior That Matters

### Prediction readiness is allowed to degrade

The backend now boots even if models are missing or incompatible.

- `/health`, `/status/models`, `/schedule`, and `/history` still come up.
- `/predict` returns `503` with structured blockers when the active bundle is not ready.
- This makes deployments diagnosable instead of failing hard during startup.

### Runtime Enhancements (May 2026)

- Model hot-reload: the backend starts a lightweight background `model-watcher` thread that monitors the active models directory and reloads promoted bundles without requiring a full process restart. This improves promotion workflows and reduces downtime.
- In-process LRU cache: prediction responses are cached in-memory with TTL and max-items controlled by `PREDICT_CACHE_TTL_SEC` and `PREDICT_CACHE_MAX_ITEMS` (see `backend/main.py`) to reduce repeated identical inference cost during heavy UI refreshes.

### Premium reliability hardening (May 2026)

The backend prediction path was iterated in three passes:

1. **Weakpoint discovery pass**: identified duplicate in-memory history growth on cache hits and non-fail-fast team-code validation during `/predict`.
2. **Mitigation pass**: added bounded+deduplicated in-memory history recording to prevent repeated identical cache returns from crowding out useful recent history.
3. **Validation pass**: added runtime-backed team code validation (dataset + team map) so invalid abbreviations fail fast with actionable error hints.

### Schedule loading is queryable and postseason-safe

- `GET /schedule?season=<year>&week=<week>` returns a specific slate.
- `GET /schedule/next-week` remains the compatibility route for "next slate".
- When future postseason games exist, the backend keeps showing the next playoff slate.
- During true offseason, if the next season schedule is bundled or available through `nflreadpy`, the backend shows the upcoming season's earliest week instead of a stale archived slate.
- If no current or future season schedule exists anywhere, the backend falls back to the latest available archived slate rather than returning an empty schedule.

### History is user-scoped

The frontend sends `X-User-Id`, and the backend uses that to isolate prediction history.

- Primary store: SQLite-backed history and summary metrics
- Fallback: JSON ledgers under `backend/Predictions/users/<user-storage-key>/`
- The current session is local-device convenience state, not real server-side authentication

## Frontend Architecture In One Minute

- `frontend/src/App.jsx` creates the auth session and shared prediction state once.
- `frontend/src/hooks/usePredictionState.js` owns schedule, health, history, summary, logos, and prediction maps.
- `frontend/src/components/DashBoard/Dashboard.jsx` consumes that shared state instead of shadowing it locally.
- `frontend/src/api/client.js` is the supported transport and compatibility layer for the active app shell.

## Key Endpoints

### Health and status

- `GET /health`
- `GET /health/pipeline`
- `GET /status/overview`
- `GET /status/models`
- `GET /status/runtime`
- `GET /metadata/dataset`
- `GET /metadata/model-bundle`

### Schedule and prediction

- `GET /schedule`
- `GET /schedule/next-week`
- `GET /api/predict/next-week`
- `GET /teams/logos`
- `POST /predict`
- `POST /debug/predict-input`

### History

- `GET /history?limit=N`
- `GET /history/summary`

### Admin

When `ENABLE_ADMIN=true`:

- `POST /admin/retrain`
- `POST /admin/promote/{job_id}`

## Repository Map

```text
backend/
  main.py                      FastAPI app and runtime orchestration
  builddataset.py              Canonical dataset build entrypoint
  train_models.py              Canonical training entrypoint
  prediction_store.py          User-scoped history persistence
  sqlite_store.py              SQLite-backed prediction history
  app/core/settings.py         Environment settings and path resolution
  data/
    datasets/
      latest_dataset.json
      runs/<timestamp>/
    models/
      current/

frontend/
  src/
    App.jsx
    api/client.js
    hooks/usePredictionState.js
    components/DashBoard/Dashboard.jsx
    components/HistoryPage.jsx
    pages/StatsPage.jsx
  public/
    schedules/
```

logos/

```
    favicon.ico
```

### Premium enhancement iterations (May 2026)

Two weak points were prioritized and improved over three implementation iterations:

1. **CORS regex resilience and safety**
   - Iteration 1: detected that malformed `ALLOW_ORIGIN_REGEX` values could override safe defaults.
   - Iteration 2: normalized slash-delimited env regexes and rejected known-bad overmatching patterns.
   - Iteration 3: added backend tests to guarantee fallback to the canonical Vercel-origin regex in production.

2. **Frontend API reliability under transient failures**
   - Iteration 1: identified fetch calls as single-shot requests (no timeout, no retry).
   - Iteration 2: added request timeout + bounded retry logic for network/transient server errors.
   - Iteration 3: added client tests proving transient `503` retries recover successfully.

## Useful Docs

- [Environment configuration](docs/ENVIRONMENT.md)
- [Frontend prediction flow](docs/FRONTEND_PREDICTION_FLOW.md)

## Verification

Recommended checks after backend or frontend changes:

```bash
.venv\Scripts\python.exe -m pytest backend/tests -q
cd frontend && npm test -- --run && npm run build
python scripts/verify_api_cors.py --backend-url https://nfl-predict-ecf5a5bd34fe.herokuapp.com
```

Runtime smoke checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/pipeline
curl http://127.0.0.1:8000/status/overview -H "X-User-Id: analyst@example.com"
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"home_team\":\"LAC\",\"away_team\":\"ARI\",\"season\":2026,\"week\":1}"
```

## Troubleshooting

### `/predict` returns `503`

- Check `/status/models` for readiness blockers.
- Check `/health/pipeline` for dataset hash, stale dataset, and feature-contract blockers.
- Confirm `MODELS_DIR` points at a complete bundle.
- If `latest_dataset.json` changed after a dataset rebuild, retrain and promote a new model bundle before serving predictions.
- If the bundle was trained under a different scikit-learn version, retrain or align the runtime environment.

### The frontend loads but some pages look empty

- Confirm Vercel `VITE_API_BASE_URL` points at the canonical Heroku backend URL.
- In local dev, prefer `VITE_API_DEV=http://127.0.0.1:8000`.
- Older deployments may not expose `/history/summary` or queryable `/schedule`; the frontend now falls back, but a backend redeploy is still the clean fix.

### Training seems to use the wrong dataset

- Inspect `backend/data/datasets/latest_dataset.json`.
- Override explicitly when needed:

```bash
python backend/train_models.py --data backend/data/datasets/<your_clean_dataset>.csv
```

### Local schedule lookups return nothing

- Make sure `backend/data/Nfl_schedule_<upcoming-year>.csv` exists once the upcoming schedule is published.
- `SCHEDULE_PATH` can point to a preferred CSV, but the backend also scans sibling schedule CSVs so a stale explicit file does not hide a newer packaged season.
- The frontend also ships fallback CSVs under `frontend/public/schedules/` for compatibility with older backends.

### Dataset CSVs do not show up in `git status`

- Curated production dataset CSVs under `backend/data/datasets/game_features_*.csv` are intentionally allowed through `.gitignore`.
- Runtime databases and local prediction history remain ignored and should not be committed.
- If a newly generated schedule CSV is production-critical, add an explicit unignore rule such as `!backend/data/Nfl_schedule_2026.csv`.
