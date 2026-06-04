# REPO-INFO

## Scan Metadata

- Scan date: 2026-06-02
- Repo root: `C:\Users\goku\Documents\NFL_ML_Predictions`
- HEAD at scan start: `54697421f`
- Current branch: `master`
- GitHub remote: `https://github.com/Jordon-py/NFL_ML_Predictions.git`
- Backend deploy remote: `https://git.heroku.com/nfl-predict.git`
- Frontend deploy project: Vercel project `nfl-ml-predictions`
- Scan method: memory check, README review, root file map, targeted backend/frontend/deploy inspection, and focused cleanup verification.
- Working tree note: the tree was already dirty before this cleanup. Runtime DBs, build output, ignored env files, and temp permission folders were present together, so staging must stay selective.

## Executive Summary

This repository is a full-stack NFL prediction app with a FastAPI backend, React/Vite frontend, schedule ingestion tooling, dataset/model training scripts, and deployment wiring for Heroku plus Vercel.

The active runtime is simpler than the repo history suggests:

1. `backend/main.py` is the live FastAPI bootstrap; public route registration lives in `backend/routes/api.py`.
2. `backend/services/api_runtime.py` owns the route-facing business workflows.
3. `backend/app/core/settings.py` is the authoritative env, CORS, and path-resolution layer.
4. `frontend/src/api/client.js` is the frontend transport adapter.
5. `frontend/src/hooks/usePredictionState.js` is the dashboard/history shared state owner.
6. `backend/builddataset.py`, `backend/train_models.py`, and `backend/scripts/weekly_retrain.py` are the main ML ops entrypoints.

The biggest engineering risk is source-of-truth drift: active code, archived code, generated datasets, local runtime DBs, tracked build output, and permission-broken pytest temp directories all live near each other. Future changes should start from the active runtime map below, not from the archive folder or older service abstractions unless the code path is verified.

## Top-Level Structure

| Path | Role | Status |
| --- | --- | --- |
| Root files | Project metadata, deployment entrypoints, pytest config, README, and repo dossier | Keep intentionally small |
| `backend/` | FastAPI app, prediction helpers, training/data scripts, tests, runtime data | Active; owns backend scripts and generated feature data |
| `frontend/` | React/Vite app, client adapter, shared hook, dashboard/history/stats UI | Active |
| `docs/` | Environment, dataflow, schema, and integration docs | Active |
| `scripts/` | Repo-level operational checks | Active, small |
| `.github/workflows/` | CI and deploy automation | Active but secret-dependent |
| `archive/` | Historical repo snapshots and old docs/scripts | Historical, not runtime |
| `artifacts/` | Planning/task notes and generated QA assets | Supporting, not runtime |
| `.vercel/` | Linked Vercel project metadata | Active frontend deploy metadata |
| `.heroku/`, `Procfile`, `app.json` | Heroku/backend deploy metadata | Active backend deploy metadata |
| `tmp_pytest*`, `.runtime_*`, `pytest_basetemp_*` | Local pytest/temp folders | Local-only, currently permission-prone |

## Architecture Overview

### Backend

The backend serves one FastAPI app from `backend.main:app`. It loads dataset/model state during lifespan startup, keeps health/status endpoints available even when prediction readiness is degraded, and returns structured `503` blockers from `/predict` when the model bundle is not ready.

Primary layers:

- App bootstrap and router mounting: `backend/main.py`
- Route registration: `backend/routes/api.py`
- Route-facing workflows and runtime state: `backend/services/api_runtime.py`
- Environment and path settings: `backend/app/core/settings.py`
- Inference helpers: `backend/utils/functions_for_main.py`
- User prediction history: `backend/prediction_store.py` and `backend/sqlite_store.py`
- Dataset/model operations: `backend/builddataset.py`, `backend/train_models.py`, `backend/scripts/weekly_retrain.py`
- Schedule ingestion: `backend/services/schedule_ingestion.py`

### Frontend

The frontend is a Vite SPA. `App.jsx` owns routing/auth shell setup, while `usePredictionState.js` hydrates shared schedule, history, health, logos, summary, prediction maps, and season context for the dashboard and history page.

The dashboard now follows a backend-owned offseason contract:

`/offseason/status` -> explicit `/schedule?season=<season>&week=<week>` during offseason, otherwise `/schedule/next-week`.

`StatsPage.jsx` remains an exception because it fetches its own snapshot directly instead of consuming the shared hook.

### Deployment

Backend:

- Target: Heroku app `nfl-predict`
- Entrypoint: `Procfile` -> `uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1`
- Canonical backend origin in docs: `https://nfl-predict-ecf5a5bd34fe.herokuapp.com`

Frontend:

- Target: Vercel project `nfl-ml-predictions`
- Canonical alias: `https://new-nfl-predict.vercel.app`
- Build root: `frontend/`
- Build command: `npm run build`

## Important Files and Modules

| Path | Role / Purpose | Completeness | Important Units | Notes |
| --- | --- | --- | --- | --- |
| `README.md` | Operator quick start and repo map | Active | deploy targets, quick start, endpoints | Keep this aligned with real deploy targets |
| `REPO-INFO.md` | Durable repo intelligence dossier | Active | architecture map, risk map, safe-edit map | Update after meaningful runtime/deploy changes |
| `backend/main.py` | FastAPI app bootstrap and router mounting | Active, small | `app`, `create_app`, router include, exception handlers | Keep deployment-focused |
| `backend/routes/api.py` | Canonical route registration | Active | `/health`, `/status/*`, `/offseason/status`, `/schedule*`, `/history*`, `/predict`, `/admin/*` | Route map only; no business logic |
| `backend/services/api_runtime.py` | Route-facing runtime workflows | Active, dense | `AppState`, `lifespan`, prediction, schedule, history, admin, premium AI handlers | Highest-risk edit zone |
| `backend/app/core/settings.py` | Env, CORS, deploy-mode, path resolution | Active | `Settings`, `allowed_origins`, `effective_allow_origin_regex`, `resolved_*` properties | Start here for config/deploy bugs |
| `backend/utils/functions_for_main.py` | Schedule time parsing, row lookup, model prep, prediction helpers | Active | `_add_kickoff_utc_datetime`, `_get_game_row_with_source`, `_prepare_inputs`, `_predict_score`, `_roll_forward_missing_player_stats` | Fixed to parse both naive and UTC schedule times |
| `backend/services/schedule_ingestion.py` | ESPN scoreboard schedule ingestion | New active tool | `ingest_schedule`, `clean_schedule_frame`, `validate_schedule_frame`, `save_schedule` | Produces leak-safe future rows |
| `backend/prediction_store.py` | User-scoped history facade | Active | context builder, append/load/summary helpers | SQLite-first, JSON fallback |
| `backend/sqlite_store.py` | SQLite prediction history and score persistence | Active | DB setup, persist/query/upsert functions | `backend/predictions.db` is runtime state |
| `backend/builddataset.py` | Canonical dataset build wrapper | Active | CLI, cleaning, manifest write | Produces `backend/data/datasets/*` |
| `backend/train_models.py` | Training, evaluation, staging/promotion | Active, complex | model pipelines, reports, metadata | Must stay aligned with serving bundle contract |
| `backend/scripts/weekly_retrain.py` | Dataset rebuild + train automation | Active | CLI orchestration | Good operator entrypoint |
| `backend/ollama/llm_ollama.py` | Premium AI agent facade | Active | `NFLAgent`, CLI `chat` | Delegates memory and client behavior |
| `backend/ollama/memory.py` | Premium AI dataset memory | Active | `NFLMemory` | Loads feature CSV and builds bounded prompt context |
| `backend/ollama/client.py` | Premium AI Ollama client helpers | Active | `OllamaClient`, `chat_messages`, `explain_prediction` | Owns env config, auth headers, timeout, and model fallbacks |
| `backend/scripts/audit_inference.py` | Manual inference-row audit | Active utility | sample game diagnostics | Reads legacy/current feature CSVs and model bundles |
| `backend/scripts/sync_data.py` | Schedule and score sync helper | Active utility | schedule CSV writes, score-sync job | Backend-owned because it writes backend data |
| `backend/scripts/sync_direct.py` | nflverse schedule CSV fetcher | Active utility | yearly schedule CSV writes | Backend-owned because it writes backend data |
| `backend/scripts/sync_season.py` | Completed-score backfill helper | Active utility | ESPN date scan, SQLite upsert | Backend-owned because it writes backend score state |
| `frontend/src/api/client.js` | Frontend transport and normalization | Active | `fetchJson`, `getOffseasonStatus`, `getScheduleForWeek`, `predictGame`, history helpers | Compatibility layer for old deployments |
| `frontend/src/hooks/usePredictionState.js` | Shared dashboard/history state owner | Active | init hydration, `refreshHistory`, `loadScheduleForWeek`, `setPrediction`, `pushHistory` | Main frontend state contract |
| `frontend/src/App.jsx` | Router/auth/app shell | Active | protected shell, route wiring | Creates one shared hook instance |
| `frontend/src/components/DashBoard/Dashboard.jsx` | Main prediction UI | Active | slate controls, predict one/all actions | User-facing primary workflow |
| `frontend/src/pages/StatsPage.jsx` | Status/history/schedule view | Active but drift-prone | independent fetch hydration | Should eventually reuse shared hook or shared service |
| `scripts/verify_api_cors.py` | Live API/CORS verifier | Active | health/status/predict/CORS probes | Use after deploy |
| `.github/workflows/ci.yml` | GitHub CI | Active | backend tests, frontend tests/build, optional CORS check | CI uses `backend/requirements.txt` |
| `.github/workflows/deploy.yml` | GitHub deploy workflow | Active but secret-dependent | Heroku git push, Vercel action | Requires GitHub secrets |

## Backend Map

### Public API Surface

- `GET /health`
- `GET /status`
- `GET /status/overview`
- `GET /status/models`
- `GET /status/runtime`
- `GET /status/dataset-versioning`
- `GET /status/performance-drift`
- `GET /offseason/status` and `/api/offseason/status`
- `GET /schedule`
- `GET /schedule/next-week`
- `GET /predict/next-week` and `/api/predict/next-week`
- `GET /teams/logos` and `/api/teams/logos`
- `GET /history`
- `GET /history/summary`
- `POST /predict` and `/api/predict`
- Admin-only when enabled: `POST /admin/retrain`, `GET /admin/retrain/{job_id}`, `POST /admin/promote/{job_id}`

### Runtime Data Boundaries

| Path | Meaning | Edit Policy |
| --- | --- | --- |
| `backend/data/datasets/` | Promoted clean datasets and manifests | Generated; commit only intentional curated artifacts |
| `backend/data/models/` | Packaged runtime model bundle | Generated/model artifact; do not hand-edit |
| `backend/data/Nfl_schedule_2025.csv` | Current packaged schedule CSV | Curated runtime asset |
| `backend/data/Nfl_schedule_2026.csv` | Current packaged 2026 schedule CSV | Curated runtime asset |
| `backend/data/schedules/*.parquet` | Schedule ingestion parquet outputs | Generated evidence/possible future runtime asset |
| `backend/predictions.db` | SQLite runtime prediction history | Runtime state; avoid committing |
| `backend/Predictions/` | JSON fallback history | Runtime state; avoid committing |

### Schedule Selection Contract

`backend/main.py` tries live `nflreadpy` schedule loading first, then non-empty packaged CSV fallbacks. It should prefer future playoff games, then a current/future season if rows exist, and only then fall back to the latest archived slate.

`/offseason/status` must not advertise a season/week that the schedule endpoint cannot populate. If the only available slate is archived, it should point the frontend at that actual archived season/week and mark offseason mode as true.

## Frontend Map

### Active Flow

1. `App.jsx` creates the local auth session and protected routes.
2. `usePredictionState.js` calls `/offseason/status`.
3. During offseason, the hook calls `/schedule?season=<current_season>&week=<current_week>`.
4. During normal season/postseason, the hook calls `/schedule/next-week`.
5. `client.js` normalizes schedule/history/prediction responses.
6. `Dashboard.jsx` renders games and calls `predictGame`.
7. `/predict` responses update per-card prediction maps and user-scoped history.
8. `HistoryPage.jsx` reads the same shared state.

### Frontend Compatibility Fallbacks

- If `/history/summary` is unavailable, `client.js` derives metrics from `/history`.
- If `/schedule?season=&week=` is unavailable, `client.js` falls back to public schedule CSVs.
- If `/schedule/next-week` is unavailable, `client.js` also tries local schedule assets.
- If `/offseason/status` is unavailable, `client.js` returns a safe non-offseason fallback.

## Data Flow / Lifecycle Map

### Schedule Lifecycle

ESPN scoreboard API
-> `backend/services/schedule_ingestion.py`
-> `ScheduleRow` dataclass records
-> `clean_schedule_frame`
-> CSV/parquet schedule artifacts
-> `backend/main.py` schedule loader
-> `_add_kickoff_utc_datetime`
-> `_select_schedule_slice`
-> `/schedule*` JSON rows
-> `frontend/src/api/client.js`
-> `usePredictionState.js`
-> dashboard cards.

### Prediction Lifecycle

Dashboard game row
-> `buildPredictPayload`
-> `client.predictGame`
-> `POST /predict`
-> readiness gate in `backend/main.py`
-> `_get_game_row_with_source`
-> `_roll_forward_missing_player_stats`
-> `_prepare_inputs`
-> `_predict_score` and win-probability logic
-> `PredictionResponse`
-> SQLite/JSON history persistence
-> frontend prediction map and history refresh.

### Dataset / Model Lifecycle

Raw NFL data and schedule assets
-> `backend/builddataset.py`
-> clean dataset and `latest_dataset.json`
-> `backend/train_models.py`
-> model artifacts, metadata, reports, staging bundle
-> optional promotion to runtime models directory
-> `backend/main.py` startup model loading
-> `/status/models` and `/predict`.

### History Lifecycle

Browser local session email
-> `X-User-Id`
-> `prediction_store.build_prediction_user_context`
-> SQLite `user_predictions`
-> score backfill from completed schedule rows
-> `/history` and `/history/summary`
-> dashboard/history/stats UI.

## Command Surface Map

### Backend

```powershell
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
python backend/builddataset.py --start 2018 --end 2025 --out-dir backend/data/datasets
python backend/train_models.py
python backend/scripts/audit_inference.py
python backend/scripts/sync_data.py
python -m backend.services.schedule_ingestion --season 2025 --season-types 2,3 --out-csv backend/data/Nfl_schedule_2025.csv --out-parquet backend/data/schedules/nfl_schedule_2025.parquet
python scripts/verify_api_cors.py --backend-url https://nfl-predict-ecf5a5bd34fe.herokuapp.com
```

### Tests

```powershell
python -m py_compile backend/main.py backend/utils/functions_for_main.py backend/services/schedule_ingestion.py
.\nflenv\Scripts\python.exe -m pytest backend\tests -q -o addopts=''
cd frontend
npm test -- --run
npm run build
```

Local Windows note: pytest temp folders in this checkout can raise `PermissionError: [WinError 5] Access is denied`. When that happens, rerun focused tests with `-o addopts=''` and report the temp-ACL blocker honestly.

### Deployment

```powershell
git push origin master
git push heroku main
heroku config:get SCHEDULE_PATH -a nfl-predict
heroku config:get MODELS_DIR -a nfl-predict
heroku logs --tail -a nfl-predict
```

## Risk / Debt / Confusion Hotspots

- `backend/main.py` is too large and owns too many responsibilities.
- `StatsPage.jsx` bypasses shared prediction state and can drift from dashboard behavior.
- `frontend/package-lock.json` can pick up accidental churn from dirty `node_modules`; inspect before staging.
- `frontend/dist/index.html` should remain ignored build output and should not be part of normal deploy commits.
- `backend/predictions.db` is runtime state and should not be part of normal deploy commits.
- `backend/data/Nfl_schedule_2026.csv` is now a populated curated runtime asset.
- Root `requirements.txt` and `backend/requirements.txt` differ. Heroku uses the root file; CI currently installs `backend/requirements.txt`.
- Root is intentionally limited to project/deploy/test metadata plus README/REPO-INFO; backend operations live under `backend/scripts/`, frontend code under `frontend/`, and durable docs under `docs/`.
- `.slugignore` excludes docs/tests from Heroku, which is fine for slug size but means deploy debugging must rely on source locally/GitHub.
- Multiple temp folders currently produce Windows permission warnings during git/pytest scans.
- `archive/` contains many old files with names similar to active files. Do not use archive code as runtime evidence unless a current import path proves it.

## Change-Safety Map

### Safer Edit Zones

- `docs/*.md`
- `README.md`
- `scripts/verify_api_cors.py`
- Focused frontend presentational components
- Narrow tests under `backend/tests/` or `frontend/src/**/*.test.*`
- Small compatibility additions inside `frontend/src/api/client.js`

### Medium-Risk Edit Zones

- `frontend/src/hooks/usePredictionState.js`
- `frontend/src/api/client.js`
- `backend/app/core/settings.py`
- `backend/services/schedule_ingestion.py`
- `backend/utils/functions_for_main.py`

### High-Risk Edit Zones

- `backend/main.py`
- `backend/train_models.py`
- `backend/builddataset.py`
- model artifact directories
- runtime history DB/files
- deploy config files (`Procfile`, `app.json`, `.github/workflows/deploy.yml`, `.slugignore`)

## Likely Incomplete or Placeholder Areas

- Real authentication is not implemented. `useAuthSession.js` is local-device identity only.
- 2026 schedule CSV/parquet files are populated curated runtime assets.
- Admin retrain/promote endpoints exist but require careful production token/config handling.
- Frontend public schedule fallback assets are compatibility-only; backend schedule assets are the runtime source of truth.
- Stats page does not yet share the dashboard hook's offseason routing behavior.
- Schedule ingestion produces good clean schedule rows, but feature-dataset exact-match coverage still needs verification after ingestion changes.

## Recommended Next Steps

1. Keep `master` as the canonical GitHub source branch and GitHub Actions deploy trigger.
2. Refactor `StatsPage.jsx` to consume shared season/schedule context or a shared schedule service.
3. Keep generated `frontend/dist/*`, `backend/predictions.db`, and local artifact screenshots out of Git.
4. Add backend tests specifically for `/offseason/status` matching an available `/schedule?season=&week=` response.
5. Keep future scripts and docs in their owner directories: backend data/runtime helpers in `backend/scripts/`, repo-level external verifiers in `scripts/`, and durable markdown in `docs/`.
6. Align Heroku and CI dependency surfaces by resolving the root `requirements.txt` versus `backend/requirements.txt` split.

## Open Questions / Uncertainties

- Should the repo commit generated parquet schedule artifacts, or keep only canonical CSV schedules in Git?
- Should `origin/main` remain as a compatibility branch for any external deploy integration, or can it be deleted after `master` deployments are verified?
- Is the current Heroku config still `SCHEDULE_PATH=data/Nfl_schedule_2025.csv`, or has it moved to the populated 2026 schedule?
- Should GitHub Actions deploy both frontend and backend on every `master` push, or should backend deploy remain a manual Heroku push?
