# Architecture Map

This repository has three different kinds of material:

1. Active runtime code that powers the current app.
2. Alternate or staged modules that may support future refactors.
3. Historical snapshots under `archive/`.

If you want to change the live product safely, start with the active runtime path
below and treat everything else as reference material until proven otherwise.

## 1. Active Runtime Path

| Layer | Primary Files | Current Responsibility |
|---|---|---|
| Frontend boot | `frontend/src/index.jsx`, `frontend/src/App.jsx` | Starts React, registers routes, and loads high-level status for the nav. |
| Prediction UI | `frontend/src/components/DashBoard/Dashboard.jsx` | Fetches the next-week schedule and owns per-game prediction state. |
| Matchup rendering | `frontend/src/components/Card/TeamGrid.jsx`, `frontend/src/components/Card/Card.jsx` | Renders game cards and shows loading/errors/predictions. |
| Status/history UI | `frontend/src/pages/StatsPage.jsx`, `frontend/src/components/HistoryPage.jsx` | Reads backend status/history endpoints and renders reporting views. |
| Frontend API layer | `frontend/src/api/client.js` | Active fetch wrapper and response normalization for the live UI. |
| Frontend data-shape helpers | `frontend/src/utils/gameUtils.js` | Canonical season/week/team normalization and matchup key generation. |
| Backend entrypoint | `backend/main.py` | FastAPI app, startup loading, live endpoints, and current inference flow. |
| Backend support helpers | `backend/utils/functions_for_main.py`, `backend/app/core/settings.py` | Team normalization, feature alignment, dataset/model path resolution, settings. |
| Model artifacts | `backend/models/`, dated `backend/20*/models/` folders | Joblib bundles discovered by `backend/main.py`. |

## 2. Request Flow

### Dashboard flow

1. `Dashboard.jsx` calls `getNextWeekSchedule()` from `frontend/src/api/client.js`.
2. `backend/main.py` serves `GET /schedule/next-week`.
3. When the user clicks a card, `Dashboard.jsx` builds a normalized payload from `gameUtils.js`.
4. `frontend/src/api/client.js` sends `POST /predict`.
5. `backend/main.py` returns the prediction response.
6. `TeamGrid.jsx` and `Card.jsx` render the result using the same matchup key strategy.

### Status/history flow

1. `App.jsx` calls `getStatusOverview()` to drive the nav health indicator.
2. `StatsPage.jsx` calls `getStatusOverview()`, `getPredictionHistory()`, and `getNextWeekSchedule()`.
3. `backend/main.py` serves `GET /status/overview`, `GET /history`, and `GET /schedule/next-week`.

## 3. Active vs Nearby Modules

The following files exist in the repo but are not on the current `App.jsx` + `backend.main`
execution path:

| Path | Status | Practical Guidance |
|---|---|---|
| `backend/routes.py` | Alternate router module | Useful reference, but the live API surface is currently defined in `backend/main.py`. |
| `backend/services/` | Staged/alternate inference layer | Read for ideas; do not assume edits here affect the running app today. |
| `frontend/src/api/fetch.js` | Unused alternate fetch wrapper | The live frontend imports `frontend/src/api/client.js` instead. |
| `frontend/src/hooks/usePredictionState.js` | Unwired state abstraction | Educational/reference code, not the current dashboard state owner. |
| `archive/` | Historical code/docs | Valuable context, but easy to confuse with live code. |

Rule of thumb:

- Live frontend behavior: start in `App.jsx`, `Dashboard.jsx`, `StatsPage.jsx`, `client.js`, or `gameUtils.js`.
- Live backend behavior: start in `backend/main.py`.
- Unsure whether a file is active: search for imports before editing it.

## 4. Highest-Signal Edit Points

| Goal | Start Here | Why |
|---|---|---|
| Change schedule or card behavior | `frontend/src/components/DashBoard/Dashboard.jsx` | Owns schedule loading, predict actions, and reset logic. |
| Change matchup identity or prediction payload shaping | `frontend/src/utils/gameUtils.js` | Shared source of truth for the active UI flow. |
| Change API URL handling or frontend response normalization | `frontend/src/api/client.js` | Single live fetch client. |
| Change prediction math or row selection | `backend/main.py`, `backend/utils/functions_for_main.py` | Current inference path lives here. |
| Change runtime configuration | `backend/app/core/settings.py` | Centralized backend settings model. |

## 5. Environment Variables That Matter

| Variable | Scope | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | Frontend | Backend base URL for production builds. |
| `ALLOWED_ORIGINS` / `CORS_ORIGINS` | Backend | Explicit CORS allowlist. |
| `CORS_ORIGINS_REGEX` / `ALLOW_ORIGIN_REGEX` | Backend | Regex origin allowance, mainly for preview environments. |
| `DATASET_PATH` | Backend | Override the dataset used for inference. |
| `SCHEDULE_PATH` | Backend | Override the schedule CSV fallback path. |
| `MODELS_DIR` / `MODELS_PATH` / `MODEL_DIR` | Backend | Override where trained artifacts are loaded from. |

## 6. Practical Commands

- Frontend dev server: run from `frontend/` with `npm run dev`
- Frontend production build: run from `frontend/` with `npm run build`
- Backend app: `uvicorn backend.main:app --reload --port 8000`
- Targeted backend smoke tests: `pytest backend/tests/test_endpoints.py backend/tests/test_api_endpoints.py`
