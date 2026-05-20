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
- 2026-05-20: Added runtime enhancements: in-process LRU prediction cache and background model-watcher to reload promoted bundles without full restart. Updated prediction cache to respect `PREDICT_CACHE_TTL_SEC` and `PREDICT_CACHE_MAX_ITEMS` settings. Updated smoke tests to include cache metrics in `/status/runtime`.
- 2026-05-20: Minor cleanup: extracted small LRU cache util at `backend/utils/cache.py` and started a daemon `model-watcher` thread at app startup to improve operational experience during promotion flows.
- 2026-05-20: Exposed `/predict` fallback diagnostics (`row_source`, `fallback_used`, `fallback_reasons`, `probability_source`) and verified the real-model smoke path for 2025 week 1 BUF vs BAL now uses the win classifier. Added compatibility handling for legacy rolling feature names so the active model bundle can consume the current prior-feature dataset without dropping to logistic fallback.
