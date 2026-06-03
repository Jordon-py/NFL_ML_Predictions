# Backend Map

Updated: 2026-03-28

## Status Rubric

- `done`: Active in the current backend path and supported by current wiring or tests.
- `needs work`: Active or important, but incomplete, weakly covered, duplicated, or carrying known issues.
- `review only`: Not moved in this pass because it may still matter, but it is not part of the live backend path or it needs manual review.
- `moved to review`: High-confidence junk or scratch files that were relocated into `review/`.

## Live Backend Path

- `backend/main.py`: Live FastAPI application and route definitions.
- `main.py`: Repo-root ASGI shim for `backend.main:app`.
- `backend/app/core/settings.py`: Environment, CORS, model path, and schedule path resolution.
- `backend/utils/functions_for_main.py`: Prediction, schedule, and feature-prep helpers used by `backend/main.py`.
- `backend/utils/ops_reporting.py`: Dataset/version/drift helpers used by serving and retraining flows.

## Feature Map

| Status | Feature | Key files | Notes |
| --- | --- | --- | --- |
| `done` | App bootstrap and runtime config | `backend/main.py`, `main.py`, `backend/app/core/settings.py` | `backend/main.py` is the live app; the repo-root `main.py` is only a compatibility launcher. |
| `done` | Health, status, debug, and drift APIs | `backend/main.py`, `backend/tests/test_routes_smoke.py`, `backend/tests/test_api_endpoints.py`, `backend/tests/test_endpoints.py` | Covers `/health`, `/status`, `/status/overview`, `/status/runtime`, `/status/dataset-versioning`, `/status/performance-drift`, `/debug`, `/debug/dataset`, and `/debug/predict-input`. |
| `done` | Schedule loading and next-week slate | `backend/main.py`, `backend/utils/functions_for_main.py` | Uses `nflreadpy` first, then falls back to local CSV paths and logo lookup files. |
| `done` | Prediction inference | `backend/main.py`, `backend/utils/functions_for_main.py`, `backend/tests/test_predict_two_stage.py` | Current live path for `/predict`, `/api/predict`, `/predict/next-week`, and `/api/predict/next-week`. |
| `done` | In-memory prediction history | `backend/main.py`, `backend/tests/test_endpoints.py` | `/history` is currently backed by `state.history`, not the SQLite/history sidecar files. |
| `done` | Admin retrain and promotion flow | `backend/main.py`, `backend/scripts/weekly_retrain.py`, `backend/utils/ops_reporting.py` | Background retrain jobs, gating, and staged model promotion are wired in the live app. |
| `done` | Model training stack | `backend/train_models.py`, `backend/tests/test_train_models_stack.py` | Current trainer writes staged/promoted bundles, metadata, and reports. |
| `needs work` | Dataset build path | `backend/builddataset.py`, `backend/build_csv_datasetsv3.py`, `backend/build_csv_datasets_v3.py` | The typed wrapper looks current, but there are two similarly named builders and one duplicate file has merge markers. |
| `needs work` | Startup coverage | `backend/tests/test_startup_checks.py` | This test file is effectively empty, so lifespan/startup behavior has weak direct coverage. |
| `needs work` | Conflicted or unstable files | `backend/pipeline_enhanced_v3.py`, `backend/build_csv_datasets_v3.py`, `backend/pbp_cache.csv` | These files contain merge-conflict markers or corrupted content and should not be treated as canonical. |

## Review Only

### Legacy or Competing Backend Stacks

- `backend/routes.py`: Older parallel router layer. No current `include_router(...)` call was found in the live app.
- `backend/services/prediction_service.py`
- `backend/services/inference_row.py`
- `backend/schemas.py`

These files form an alternate prediction stack that is not wired into `backend/main.py`.

### Dormant Persistence Experiments

- `backend/prediction_store.py`
- `backend/sqlite_store.py`
- `backend/predictions.db`
- `backend/Predictions/prediction-history.md`

The live `/history` route does not currently use this persistence path.

### Historical or Manual Analysis Assets

- `backend/OUTDIR/`
- `backend/build_row.ipynb`
- `backend/get_logos.ipynb`
- `backend/jupyter_feats.ipynb`
- `backend/merge.ipynb`
- `backend/sched.ipynb`
- `backend/Team_cache.ipynb`
- `backend/pipeline_enhanced.py`
- `backend/pipeline_enhanced2.py`
- `backend/NFL Prediction Model Analysis.docx`
- `backend/reflexion_ds_full_run_package.md`

These look like research, intermediate analysis, or older refactor artifacts. They were not moved in this conservative pass.

### Dated Model Runs and Runtime-Sensitive Bundles

- `backend/20251109/`
- `backend/20251110/`
- `backend/20251111/`
- `backend/20251117/`
- `backend/20251123/`
- `backend/20251215/`
- `backend/20260115/`
- `backend/20260208/`
- `backend/20260326/`
- `backend/20260327/`
- `backend/models/`
- `backend/prod-models/`

These were intentionally left in place. `backend/main.py` discovers model bundles dynamically, so moving dated run folders could change runtime behavior unless `MODELS_DIR` is pinned.

### Duplicate Schedule and Logo Sources

- `backend/Nfl_schedule_2025.csv`
- `backend/schedule_2025.csv`
- `backend/schedules.csv`
- `NFL_Schedule.csv`
- `backend/team_logo.csv`
- `backend/team_logos.csv`
- `team_logos.csv`

The live backend probes multiple fallback paths for schedules and logos, so these duplicates need a deliberate canonicalization pass instead of an automatic move.

## Moved To Review

The following backend-adjacent files were moved into `review/` because they are high-confidence junk, logs, or scratch artifacts:

- `backend/uvicorn.out` -> `review/backend/uvicorn.out`
- `backend/uvicorn.err` -> `review/backend/uvicorn.err`
- `tmp_uvicorn.log` -> `review/tmp_uvicorn.log`
- `backend/jup.py` -> `review/backend/jup.py`
- `backend/Untitled-1.ipynb` -> `review/backend/Untitled-1.ipynb`
- `backend/Untitled-1.ps1` -> `review/backend/Untitled-1.ps1`

## Test Signals

- `backend/tests/test_routes_smoke.py`: Broad public-route smoke coverage against `backend.main.app`.
- `backend/tests/test_api_endpoints.py`: Health, schedule, debug, and prediction shape checks.
- `backend/tests/test_endpoints.py`: Status overview, history, and prediction endpoint checks.
- `backend/tests/test_predict_two_stage.py`: Verifies the two-stage stacked prediction path in the live backend.
- `backend/tests/test_train_models_stack.py`: Verifies training outputs, metadata, and stacked training behavior.
- `backend/tests/test_feature_leak_guard.py` and `backend/tests/test_leakage.py`: Guardrails for feature leakage in the training/data pipeline.
