# REPO-INFO

## Scan Metadata

- Scan date: 2026-06-06
- Repo root: `C:\Users\iProg\Documents\NFL_ML_Predictions`
- Current branch: `master`
- HEAD: `c69f8815b` (`docs: record enhancement verification`)
- Working tree at final refresh: modified workflow/package/docs/toolchain files plus an unrelated modified `backend/ollama/chat.ipynb`.
- GitHub remote: `https://github.com/Jordon-py/NFL_ML_Predictions.git`
- Heroku remote: `https://git.heroku.com/nfl-predict.git`
- Frontend deploy project in docs: Vercel project `nfl-ml-predictions`
- Scan method: repo-info scanner, README review, git status/remotes, backend route inspection, frontend route inspection, manifest/config reads, runtime asset reads, CI/deploy workflow reads, official GitHub Actions release checks, and local verification.
- Scope: current operational refresh after CI/toolchain updates. Intentional edited surfaces are `.github/workflows/*.yml`, `frontend/package*.json`, `pyproject.toml`, `backend/pyproject.toml`, and this file.

## Executive Summary

This is a full-stack NFL forecasting app: FastAPI backend, React/Vite frontend, dataset build scripts, model training/promotion scripts, Heroku backend deployment, and Vercel frontend deployment.

The active runtime is smaller than the repository history suggests:

1. `backend/main.py` creates `backend.main:app`, configures CORS, exception handlers, and mounts the API router.
2. `backend/routes/api.py` is the canonical backend route registry.
3. `backend/services/api_runtime.py` owns route-facing runtime state, schedule/model loading, prediction, history, admin, and premium AI workflows.
4. `frontend/src/App.jsx` owns the SPA shell and route wiring.
5. `frontend/src/hooks/usePredictionState.js` owns shared dashboard/history state.
6. `frontend/src/api/client.js` owns API base resolution, request timeout/retry behavior, response normalization, and frontend compatibility fallbacks.
7. `backend/builddataset.py` and `backend/train_models.py` are the main dataset/model lifecycle entrypoints.

The main repo risk is source-of-truth drift: `archive/` still contains old code and docs, dependency manifests diverge between local/Heroku and CI, schedule defaults still mention 2025 in some deploy/docs surfaces while the active data snapshot includes 2026, and the frontend client has schedule CSV fallback code even though `frontend/public/schedules/` is not present in this checkout.

## Repo Snapshot

- Scanner result: 643 files, about 205 MB.
- Largest drift zone: `archive/` has 257 files and about 60.71 MB. Treat it as historical unless a current import path proves runtime use.
- Top-level active areas:
  - `backend/` - FastAPI app, runtime services, tests, data/model assets, ML scripts.
  - `frontend/` - React/Vite SPA, frontend tests, Vercel config.
  - `docs/` - 8 active markdown docs.
  - `scripts/` - repo-level operational checks, currently including `scripts/verify_api_cors.py`.
  - `.github/workflows/` - CI, Heroku deploy, and scheduled retrain automation.

## Active Runtime Map

### Backend

| Path | Role | Notes |
| --- | --- | --- |
| `backend/main.py` | FastAPI app bootstrap | Small by design; mounts `api_router` and delegates runtime behavior. |
| `backend/routes/api.py` | Canonical public route map | Declarative route registration only. |
| `backend/services/api_runtime.py` | Runtime workflows | Dense, high-risk owner for app state, models, schedules, predictions, history, admin, premium AI. |
| `backend/app/core/settings.py` | Env/CORS/path settings | Reads `SCHEDULE_PATH`, `MODELS_DIR`, `ENABLE_ADMIN`, CORS env aliases. |
| `backend/builddataset.py` | Dataset build entrypoint | Writes clean/completed/future datasets and manifests. |
| `backend/train_models.py` | Training and promotion entrypoint | Writes model bundles, metadata, reports, and staging/promotion artifacts. |
| `backend/ollama/client.py` | Premium AI client config | Handles cloud/local Ollama host selection, bearer auth, timeout, model fallback. |

Backend route groups registered in `backend/routes/api.py`:

- Health/status: `GET /health`, `/status`, `/status/overview`, `/health/pipeline`, `/status/models`, `/status/runtime`, `/status/dataset-versioning`, `/status/performance-drift`.
- Metadata/debug: `GET /debug`, `/metadata/dataset`, `/metadata/model-bundle`, `/debug/dataset`, `POST /debug/predict-input`, plus `/api/*` debug aliases.
- Schedule/teams: `GET /schedule`, `/schedule/next-week`, `/predict/next-week`, `/api/predict/next-week`, `/teams/logos`, `/api/teams/logos`.
- History: `GET /history`, `GET /history/summary`, `DELETE /history`, `GET /history/summary/memory`.
- Premium AI: `POST /premium/explain`, `/api/premium/explain`, `/premium/chat`, `/api/premium/chat`.
- Prediction: `POST /predict`, `/api/predict`.
- Admin: `POST /admin/retrain`, `GET /admin/retrain/{job_id}`, `POST /admin/promote/{job_id}`. Routes are mounted; access is controlled in runtime code and `ENABLE_ADMIN`/request checks.

### Frontend

| Path | Role | Notes |
| --- | --- | --- |
| `frontend/src/App.jsx` | SPA shell and route wiring | Uses `BrowserRouter`, protected shell, lazy pages, and one shared prediction hook instance. |
| `frontend/src/hooks/usePredictionState.js` | Shared app state | Owns schedule, week, predictions, history, summary, logos, health, season context. |
| `frontend/src/api/client.js` | API transport | Resolves local/prod backend base URL, retries transient failures, applies premium timeout. |
| `frontend/src/components/DashBoard/Dashboard.jsx` | Primary prediction UI | Consumes shared state and calls prediction/premium actions. |
| `frontend/src/components/HistoryPage.jsx` | History UI | Consumes shared history state. |
| `frontend/src/pages/StatsPage.jsx` | Stats/status page | Still fetches its own snapshot; keep this on the drift watchlist. |
| `frontend/src/pages/LandingPage.jsx` | Public landing/sign-in page | Root route. |

Frontend routes registered in `App.jsx`:

- `/` - landing page.
- `/*` - protected app shell; unauthenticated users redirect to `/`.
- `/app` - dashboard.
- `/history` - prediction history.
- `/stats` - stats/status.
- `/settings` - local account/session settings.
- Protected wildcard - signed-in 404 page.

## Data And Model State

Current dataset manifest: `backend/data/datasets/latest_dataset.json`

- Run id: `20260531T124903Z`
- Seasons: 2018-2026
- Rows/columns: 2499 rows, 242 columns
- Completed/future rows: 2227 completed, 272 future
- Clean dataset: `backend/data/datasets/game_features_20260531_clean.csv`
- Dataset hash: `94bd8ca5e7e47ac5db5d4d583daaa93265313be24a20bf909848db68a18f188b`

Current schedules under `backend/data/`:

- `Nfl_schedule_2025.csv`
- `Nfl_schedule_2026.csv`

Current model bundle under `backend/models/`:

- Pipeline artifacts: `home_pipe.joblib`, `away_pipe.joblib`, `win_pipe.joblib`
- Legacy/estimator artifacts: `home_model.joblib`, `away_model.joblib`, `win_clf_calibrated.joblib`
- Preprocessors: `preprocessor.joblib`, `score_preprocessor.joblib`, `win_preprocessor.joblib`
- Metadata/reports: `metadata.json`, `feature_manifest.json`, `training_report.json`, `run_summary.json`
- `metadata.json` reports scikit-learn `1.7.2`, training timestamp `2026-06-04T21:16:55.724480+00:00`, and the same dataset hash as `latest_dataset.json`.
- `run_summary.json` reports status `PROMOTED` and a passed model gate.

Important drift note: `frontend/src/api/client.js` can fall back to bundled public schedule CSVs, but `frontend/public/` currently contains only `nfl_ham2.png` and `nfl_pic.png`; there is no `frontend/public/schedules/` directory in this checkout.

## Manifest And Deploy Surfaces

| Path | Observed role | Current note |
| --- | --- | --- |
| `requirements.txt` | Root Python dependency surface, used by Heroku-style root builds | Fully pinned package list, includes FastAPI `0.124.2`, scikit-learn `1.7.2`, Ollama `0.6.1`. |
| `backend/requirements.txt` | CI backend dependency surface | Range-pinned production deps, includes FastAPI `>=0.104.0,<0.110.0`; this diverges materially from root requirements. |
| `pyproject.toml` | Root Python metadata and uv workspace | Python `~=3.12.0`, no declared deps, `backend` workspace member. |
| `backend/pyproject.toml` | Backend package metadata | Python `~=3.12.0`, no declared deps; aligned with root metadata. |
| `frontend/package.json` | Frontend scripts and npm deps | Scripts: `dev`, `start`, `build`, `preview`, `test`; engines allow Node `>=22.12.0 <23`, aligned with CI and `frontend/.nvmrc`. |
| `frontend/package-lock.json` | Locked npm dependency graph | Root package engine metadata is aligned with `frontend/package.json`. |
| `Procfile` | Heroku ASGI entrypoint | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1`. |
| `app.json` | Heroku app metadata/default env | Defaults still show `SCHEDULE_PATH=data/Nfl_schedule_2025.csv` and `MODELS_DIR=models`. |
| `frontend/vercel.json` | Vercel SPA config | Vite framework, `dist` output, `npm ci`, SPA rewrite to `/index.html`. |
| `.slugignore` | Heroku slug pruning | Excludes docs/tests/frontend build noise; explicitly keeps `backend/models/**` and `backend/data/datasets/**`. |

No root `vercel.json` exists. Vercel config is in `frontend/vercel.json`.

## Test And Build Expectations

README recommended checks:

```powershell
.venv\Scripts\python.exe -m pytest backend/tests -q
cd frontend
npm test -- --run
npm run build
python scripts/verify_api_cors.py --backend-url https://nfl-predict-ecf5a5bd34fe.herokuapp.com
```

`pytest.ini`:

- `testpaths = backend/tests backend`
- `addopts = --basetemp=tmp_pytest`

GitHub Actions CI (`.github/workflows/ci.yml`):

- Runs on pushes to `main` and `master`, plus pull requests.
- Uses `actions/checkout@v6`, `actions/setup-python@v6`, and `actions/setup-node@v6`.
- Backend job uses `PYTHON_VERSION=3.12`, installs `backend/requirements.txt`, installs pytest `>=9,<10`, verifies `backend/models/metadata.json` scikit-learn version matches installed runtime, then runs `python -m pytest backend/tests -q`.
- Frontend job uses `NODE_VERSION=22`, npm cache against `frontend/package-lock.json`, then runs `npm ci`, `npm test -- --run`, and `npm run build` from `frontend/`.
- CORS job runs `scripts/verify_api_cors.py` against `CI_BACKEND_URL` or the documented Heroku fallback.

GitHub Actions deploy (`.github/workflows/deploy.yml`):

- Runs on `master` push and manual dispatch.
- Uses `actions/checkout@v6` and `actions/setup-python@v6`.
- Deploys backend to Heroku by pushing `HEAD:main`.
- Verifies required model bundle files before deploy.
- After deploy, polls `/status/models` and sends a sample `/predict` request.
- It does not deploy Vercel; Vercel appears to rely on the linked frontend project/config.

Scheduled retrain (`.github/workflows/scheduled-retrain.yml`):

- Uses `actions/checkout@v6`, `actions/setup-python@v6`, and `actions/upload-artifact@v7`.
- Runs focused contract tests before rebuilding the dataset/model bundle.
- Uploads retrain reports, dataset manifests, and model artifacts without auto-promoting.

Current friction to keep visible:

- Root `requirements.txt` and `backend/requirements.txt` disagree on backend dependency versions. That can let CI pass under one FastAPI/Starlette surface while Heroku runs another.
- Local `.venv` is Python 3.11.6 while project metadata and CI target Python 3.12. `.venv312` is an ignored Python 3.12 environment created for release-confidence local backend checks.
- Local Node is currently v25.0.0, which is outside the strict frontend engine range. Frontend checks can run if `engine-strict` is not enabled, but release-confidence frontend verification should use Node 22.

## Runtime Versus Archive Drift

Use active files first:

- Backend runtime: `backend/main.py`, `backend/routes/api.py`, `backend/services/api_runtime.py`, `backend/app/core/settings.py`.
- Frontend runtime: `frontend/src/App.jsx`, `frontend/src/api/client.js`, `frontend/src/hooks/usePredictionState.js`, route page/component files.
- Data/model runtime: `backend/data/datasets/latest_dataset.json`, `backend/data/Nfl_schedule_2025.csv`, `backend/data/Nfl_schedule_2026.csv`, `backend/models/*`.
- Deploy/runtime config: `Procfile`, `app.json`, `.slugignore`, `frontend/vercel.json`, `.github/workflows/*.yml`.

Treat these as historical unless verified by current imports or deploy config:

- `archive/**`
- old test files under `archive/**`
- archived backend copies under `archive/backend/**`
- old planning outputs under `artifacts/**` and `review/**`

Do not use archive code to infer current route behavior. The active route registry is `backend/routes/api.py`.

## Known Watch Items

1. Dependency source split: align or document why root `requirements.txt` and `backend/requirements.txt` intentionally diverge.
2. Local Python runtime split: `.venv` is currently Python 3.11.6 while metadata and CI require Python 3.12.
3. Schedule default split: README and active data emphasize 2026, but `app.json` and docs examples still default `SCHEDULE_PATH` to 2025. Runtime sibling-schedule scanning reduces the risk, but docs/deploy defaults still look stale.
4. Frontend schedule fallback split: either add the expected public schedule CSVs or document that local CSV fallback is code-only and unavailable in this checkout.
5. `StatsPage.jsx` still fetches independently; dashboard/history share `usePredictionState.js`.
6. Admin routes are mounted; keep production access checks and `ENABLE_ADMIN` behavior explicit when changing them.
7. Heroku slug excludes docs/tests, so production debugging must use local checkout, GitHub, logs, and live endpoint probes.

## Recommended Next Steps

1. Resolve the dependency split before backend feature work: decide whether CI should install root `requirements.txt`, or whether Heroku should build from `backend/requirements.txt`.
2. Recreate the default `.venv` with Python 3.12 once it is safe to replace the existing Python 3.11 environment; until then, `.venv312` is available for backend verification.
3. Refresh `docs/ENVIRONMENT.md` and `app.json` schedule examples if 2026 is now the intended packaged default.
4. Decide whether `frontend/public/schedules/` should exist; if not, remove or clearly label schedule CSV fallback assumptions in docs.
5. Keep route changes focused in `backend/routes/api.py` plus `backend/services/api_runtime.py`, then verify `backend/tests` and frontend client tests.

## Command Log

Material commands/facts used for this refresh:

```powershell
git rev-parse --show-toplevel
git status --short --branch
git log -1 --oneline --decorate
git remote get-url origin
git remote get-url heroku
python C:\Users\iProg\.codex\skills\repo-info\scripts\repo_scan.py --repo C:\Users\iProg\Documents\NFL_ML_Predictions --format markdown
Get-Content -Raw README.md
Get-Content -Raw REPO-INFO.md
Get-Content -Raw frontend\package.json
Get-Content -Raw pyproject.toml
Get-Content -Raw backend\pyproject.toml
Get-Content -Raw requirements.txt
Get-Content -Raw backend\requirements.txt
rg -n "APIRouter|@router\.|@app\.|include_router|FastAPI\(" backend -g "*.py" -g "!archive/**"
rg -n "BrowserRouter|Routes|Route|Navigate|react-router|createBrowserRouter|path=|NavLink|Link" frontend\src -g "*.jsx" -g "*.js"
Get-Content -Raw backend\main.py
Get-Content -Raw backend\routes\api.py
Get-Content -Raw frontend\src\App.jsx
Get-Content -Raw frontend\src\api\client.js
Get-Content -Raw frontend\src\hooks\usePredictionState.js
Get-Content -Raw pytest.ini
Get-Content -Raw .github\workflows\ci.yml
Get-Content -Raw .github\workflows\deploy.yml
Get-Content -Raw Procfile
Get-Content -Raw app.json
Get-Content -Raw frontend\vercel.json
Get-Content -Raw .slugignore
rg --files backend\tests frontend\src -g "*test*" -g "!**/node_modules/**"
Get-ChildItem backend\data\datasets -File
Get-ChildItem backend\data -File -Filter "Nfl_schedule_*.csv"
Get-ChildItem backend\models -File
Get-ChildItem frontend\public -Force
Get-ChildItem archive -Recurse -File -ErrorAction SilentlyContinue | Measure-Object
Get-ChildItem archive -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
Get-Content -Raw backend\data\datasets\latest_dataset.json
Get-Content -Raw backend\models\metadata.json
Get-Content -Raw backend\models\run_summary.json
gh --version
gh auth status
node --version
npm --version
.venv312\Scripts\python.exe -m pytest -q backend\tests
npm --prefix frontend test -- --run
npm --prefix frontend run build
git diff --check
```

Commands attempted but corrected to tighter reads:

- `Get-Content -Raw vercel.json` failed because no root `vercel.json` exists; the active file is `frontend/vercel.json`.
- A broad `rg --files backend\data frontend\public ...` scan timed out on the large data/model tree; targeted `Get-ChildItem` reads were used instead.

Verification completed after the CI/toolchain update: workflow YAML parsed with PyYAML, Python 3.12 backend tests passed from `.venv312` (`25 passed`), frontend tests passed (`14 passed`), Vite production build passed, and `git diff --check` returned success with only line-ending warnings.
