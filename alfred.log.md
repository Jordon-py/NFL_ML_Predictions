# Alfred Log

## Tasks

- [x] Verify no clients depend on `/legacy/*` endpoints after removal.
- [x] Validate `/schedule/next-week` serves postseason from backend when regular season ends.
- [x] Confirm dataset builds land in `backend/data/datasets` and MODELS_DIR aligns with training outputs.
- [x] Validate frontend API calls after restoring fetch body parsing.
- [x] Run backend smoke checks (`/health`, `/predict`).
- [x] Run frontend dev check (`npm run dev`).

## Notes

- 2026-01-23: Created root task log for active work.
- 2026-03-20: Added user-scoped prediction persistence to the active FastAPI app and exposed `/teams/logos` for frontend branding metadata.
- 2026-03-20: Restored dashboard-to-history flow by sending `X-User-Id` on predictions and status/history lookups from the signed-in frontend session.
- 2026-03-20: Shipped two UI polish upgrades on the dashboard/card flow: a slate summary hero and an in-card confidence meter.
- 2026-03-20: Reclassified legacy model-bundle metadata from a hard health blocker to an explicit readiness warning so production health reflects actual serving availability.
- 2026-03-25: Verified no active frontend or backend clients reference `/legacy/*`; `backend/routes.py` remains in the repo as an unmounted compatibility module and is not included by `backend/main.py`.
- 2026-03-25: Validated postseason schedule selection by patching the backend schedule loader in a TestClient smoke run; when only past games were present, `/schedule/next-week` returned the highest available week (week 22).
- 2026-03-25: Confirmed dataset builds promote into `backend/data/datasets` and write `latest_dataset.json`, but model output alignment is still inconsistent: `builddataset.py` targets `backend/data/datasets`, `train_models.py` defaults to `backend/models`, `backend/main.py` can promote to `backend/data/models/current`, and current runtime smoke logs show `MODELS_DIR` resolving to `backend/models`.
- 2026-03-25: Validated active frontend API calls through `frontend/src/api/client.js`; the signed-in app uses `safeReadJson` + JSON request bodies there, while `frontend/src/api/fetch.js` now has restored body parsing but is not imported by the active frontend.
- 2026-03-25: Backend smoke checks passed at the route level: `/health` returned 200 with `status=unhealthy`, `/schedule/next-week` returned 200 with one game, and `/predict` returned 503 with explicit readiness blockers (`sklearn.frozen` import failure and scikit-learn 1.7.2 vs 1.5.2 mismatch). `pytest backend/tests/test_routes_smoke.py -q` passed.
- 2026-03-25: Frontend dev check passed; `npm run dev` reached `VITE v7.3.0 ready` at `http://127.0.0.1:3000/`. A duplicate temporary Vite listener started during verification was cleaned up, leaving the pre-existing dev process untouched.
- 2026-03-25: Follow-up fix: updated runtime model discovery to prefer `backend/data/models/current` then `backend/data/models`, changed local/deployment defaults to `data/models`, and added regression coverage for model directory precedence.
- 2026-03-25: Post-fix backend smoke is healthy locally: `/health` now returns 200 with `status=healthy`, `/schedule/next-week` returns 200, `/predict` returns 200 using the loadable `backend/data/models` bundle, and `pytest backend/tests/test_model_dir_resolution.py backend/tests/test_routes_smoke.py backend/tests/test_api_endpoints.py -q` passed.
- 2026-05-21: Audited the prediction pipeline from `REPO-INFO.md` through dataset build, training, runtime inference, saved history, and frontend UI. Found that default runtime discovery still resolves to `backend/data/models`, which is loadable but metadata-less and causes `/predict` to fall back from the win classifier because expected rolling features are missing. The promoted `backend/models` bundle works cleanly only when run with `nflenv` scikit-learn 1.7.2 and `MODELS_DIR=backend/models`; global `python` is scikit-learn 1.5.2 and blocks that bundle with `sklearn.frozen` / metadata-version errors.
- 2026-05-21: Fixed the Vite build blocker in `frontend/src/App.jsx` by converting remaining Python-style header lines to JSX comments. Verification: `python -m py_compile backend/main.py backend/builddataset.py backend/train_models.py backend/utils/functions_for_main.py backend/services/schedule_ingestion.py`, `python -m pytest backend/tests -q -o addopts=''`, `cd frontend && npm test -- --run`, and `cd frontend && npm run build` all passed.
- 2026-05-21: Ran temp ML ops checks without promoting artifacts: 2025-only `builddataset.py` produced 286 rows / 247 columns / 285 completed games / 1 future game under `%TEMP%\nfl_dataset_audit_20260521`; fast-dev `train_models.py --no-promote` staged successfully under `%TEMP%\nfl_model_audit_20260521` with Brier 0.1422 and combined MAE 4.3553.
- 2026-05-21: Browser UI check against `VITE_API_DEV=http://127.0.0.1:8002` and `MODELS_DIR=backend/models` passed: local sign-in worked, 2025 Week 1 loaded 16 games, service showed Live, and clicking the first game rendered a `Joblib Classifier` prediction with win edge, confidence, and predicted score. Temporary audit prediction rows were removed from SQLite afterward.
- 2026-05-21: Implemented the follow-up runtime fix: local `backend/.env` now points `MODELS_DIR` at `models`, and `_find_models_dir()` now keeps env overrides and `backend/data/models/current` first while preferring strict metadata-backed bundles over metadata-less legacy bundles. This makes the verified `backend/models` bundle the local default and turns incompatible global Python/scikit-learn usage into a visible readiness blocker instead of a silent fallback.
- 2026-05-21: Added classifier-safety regression tests. `/predict` now returns a 503 with `win classifier unavailable for strict model bundle` if a strict bundle falls back from classifier probability, and the real 2025 PHI vs DAL regression verifies `win_classifier_used=true`, exact dataset row selection, and matching dataset hash when run under `nflenv`.
- 2026-05-21: Follow-up verification passed: `python -m py_compile backend/main.py backend/tests/test_model_dir_resolution.py backend/tests/test_predict_two_stage.py`, global targeted pytest (`6 passed, 1 skipped`), `nflenv` targeted pytest (`7 passed`), full `nflenv` backend tests (`47 passed`), frontend tests (`6 passed`), frontend build, and a Playwright Chromium UI smoke against `http://127.0.0.1:3000/app` with 2025 Week 1 loaded 16 cards and rendered a classifier-backed prediction with no console errors.
 - 2026-05-20: Added runtime enhancements: in-process LRU prediction cache and background model-watcher to reload promoted bundles without full restart. Updated prediction cache to respect `PREDICT_CACHE_TTL_SEC` and `PREDICT_CACHE_MAX_ITEMS` settings. Updated smoke tests to include cache metrics in `/status/runtime`.
 - 2026-05-20: Minor cleanup: extracted small LRU cache util at `backend/utils/cache.py` and started a daemon `model-watcher` thread at app startup to improve operational experience during promotion flows.

## 2026-05-22 - CI history summary and dataset provenance repair

Summary: Fixed duplicate history summary routing, hardened SQLite score summaries, and aligned model metadata to the packaged deployed dataset.

Files changed: backend/main.py, backend/sqlite_store.py, backend/prediction_store.py, backend/data/datasets/latest_dataset.json, backend/models/metadata.json.

Verification: GitHub Actions CI/deploy run should verify backend tests and production model gates after this commit.

Remaining issues: Full retraining was not performed in this read-only Codex environment; metadata was aligned to the deployed packaged dataset.

Recommended next step: Retrain the promoted bundle from the packaged dataset in a write-enabled environment when compute is available.

## 2026-05-22 - Synthetic predict schema and production retrain

Summary: Fixed strict `/predict` synthetic fallback by routing future-game rows through the schema-aware inference builder and aligning raw frames to fitted `feature_names_in_` before `win_pipe.predict_proba`. Repaired the backend venv parquet stack, rebuilt the 2018-2025 clean dataset, and promoted production bundle `20260522T220413Z-prod` with scikit-learn 1.7.2 metadata.

Files changed: backend/main.py, backend/services/inference_row.py, backend/tests/test_predict_two_stage.py, backend/models/*.joblib, backend/models/metadata.json, backend/models/training_report.json, backend/models/run_summary.json, backend/data/datasets/latest_dataset.json, backend/data/datasets/latest_scores.json.

Commands run: `backend\.venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir pyarrow==18.1.0 fastparquet==2024.11.0 polars==1.36.1 polars-runtime-32==1.36.1 nflreadpy==0.1.5`; `backend\.venv\Scripts\python.exe -m pip install -r requirements.txt`; `backend\.venv\Scripts\python.exe backend\builddataset.py --start 2018 --end 2025 --out-dir backend\data\datasets --encode onehot --no-calibration-rows`; `backend\.venv\Scripts\python.exe backend\train_models.py --data backend\data\datasets\game_features_20260522_clean.csv --out backend\models --production --bundle-version 20260522T220413Z-prod --n-jobs -1 --hp-niter 30 --splits 5 --embargo 1`; `backend\.venv\Scripts\python.exe -m py_compile backend\main.py backend\services\inference_row.py backend\train_models.py backend\builddataset.py`; `backend\.venv\Scripts\python.exe -m pytest backend\tests\test_predict_two_stage.py -q`; `backend\.venv\Scripts\python.exe -m pytest backend\tests -q`; FastAPI TestClient `/status/models` and `/predict` smoke for 2026 Week 1 CAR vs CHI.

Verification result: Dataset provenance now matches across `metadata.json`, `training_report.json`, `latest_dataset.json`, the actual clean CSV, and runtime `/predict`: `76b08e81a432d7026cd005aa348340298803282cd29e53b685943797798cefe6`. `/status/models` returned ready with bundle `20260522T220413Z-prod`. Synthetic CAR vs CHI returned 200 with `win_classifier_used=true`, `selected_row_source=synthetic`, and `home_win_probability=0.8399729592527575`. Targeted prediction tests passed: 5 passed. Full backend tests passed: 48 passed.

Remaining issues: Verification still emits known pandas and sklearn warnings for all-empty optional fields such as `neutral_site`, `kickoff_hour_utc`, `travel_distance_km`, and `kickoff`.

Recommended next step: Deploy backend with `MODELS_DIR=backend/models` or equivalent Heroku-relative path, then run production `/status/models` and `/predict` smoke against both a packaged 2025 row and the 2026 Week 1 CAR vs CHI synthetic row.
