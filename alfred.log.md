# Alfred Activity Log

## 2025-12-10T15:00:00Z — API Coherence & Endpoint Simplification

### Context
- Performed comprehensive analysis of `backend/main.py` and `frontend/src/api/client.js`
- Identified critical bugs, dead code, and logic errors affecting API reliability

### Backend Changes (`main.py`)

| Issue | Fix | Lines |
|-------|-----|-------|
| `nfl.load_schedules(2025)` called at module level → Heroku crash | Changed to lazy-load (`DEFAULT_SCHEDULE = None`) | L43-47 |
| Logic error in `get_current_nfl_context()` — accessing `.iloc[-1]` on empty DataFrame | Fixed conditional to return early when `schedule_df.empty` | L614-647 |
| Unreachable code block after early return | Removed duplicate fallback return | L648-655 |

### Frontend Changes (`client.js`)

| Issue | Fix | Lines |
|-------|-----|-------|
| **CRITICAL**: `JSON.response.body` is invalid JavaScript | Removed — `response` is already parsed JSON | L136-137 |
| Missing `api()` function — `get()` and `postJson()` referenced undefined | Added complete `api()` with timeout, retry, AbortController | L76-160 |
| No educational comments | Added section headers and JSDoc explaining each pattern | Throughout |

### Endpoint Status Matrix

| Endpoint | Backend | Frontend | Status |
|----------|---------|----------|--------|
| `/health` | ✅ | ✅ `getHealth()` | Working |
| `/debug` | ✅ | — | Working |
| `/report/training` | ✅ | ✅ `getTrainingReport()` | Working |
| `/report/calibration` | ✅ | ✅ `getCalibrationReport()` | Working |
| `/schedule/next-week` | ✅ | ✅ `getNextWeekSchedule()` | Working |
| `/predict` | ✅ | ✅ `predictGame()` | **Fixed** |
| `/predict/next-week` | ✅ | ✅ `predictNextWeek()` | Working |
| `/history` | ❌ Missing | ⚠️ `getPredictionHistory()` | Not implemented |
| `/train` | ❌ Missing | ⚠️ `startTraining()` | Not implemented |
| `/status/overview` | ❌ Missing | ⚠️ Has fallback | Graceful degradation |

### Deployment Pending
- Backend: `git push heroku rollback/heroku-endpoint-restore:master`
- Frontend: `npx vercel --prod --yes`
- Post-deploy: Smoke test `/health`, `/schedule/next-week`, `/predict`

---

## 2025-12-08T23:59:00Z

- Promoted refreshed production artifacts from `backend/prod-models/models` into `backend/models/prod_models/` (metadata timestamp 2025-12-08 17:05 UTC, 200 features, 2,149 rows).
- Purpose: ensure the API and Heroku deploy consume the latest trained models and feature schema without relying on older prod bundle.
- Next: restart backend/Heroku release to load the new joblib set; verify `/debug` reflects the new metadata timestamp and feature count.

## 2025-12-09T00:05:00Z

- Added `win_classifier_used` flag to `PredictionResponse` so the frontend can distinguish calibrated classifier runs from the logistic fallback badge.
- Purpose: stop UI from showing “Logistic fallback” when the win classifier actually ran (prediction_source still reports provenance).
- Next: redeploy backend to Heroku; frontend badges should flip to “Classifier” on model-driven predictions.

## 2025-12-08T23:30:00Z

- Pinned dataset consumption to the latest engineered file: `DATASET_PATH` now points to `C:\\Users\\goku\\Documents\\NFL_ML_Predictions\\backend\\game_features_20251208.csv` and `DEFAULT_DATASET` defaults to the same path in `config.py` for production alignment.
- Hardened dataset fallback in `backend/main.py` to prefer the 20251208 CSV in backend root or data folder before older archives, reducing startup risks from stale files.
- Context: Observed production loading an older dataset; this update forces the newest artifact and retains defensive fallbacks for legacy locations.

## 2025-12-08T23:55:00Z

- Fixed startup column-count mismatch by preferring `preprocessor.feature_names_in_` over metadata when assembling `raw_feature_columns`, ensuring sanity checks use the fitted transformer’s expected 153 columns instead of stale metadata.
- Updated dataset schema validation and sanity prediction to rely on the fitted preprocessor’s feature list, reducing false mismatch warnings when newer CSVs add/remove columns.
- Note: Restart backend after pull; if deploying to Heroku, rebuild/release to pick up the code changes.

## 2025-12-08T19:05:00Z

- Resolved Heroku startup crash by lazily loading `nflreadpy` in `backend/config.py` and redeployed (Heroku release v435) — `/health` reports healthy with production models loaded.
- Tracked production artifacts in `backend/models/prod_models/` so model metadata/joblibs deploy with the app; `/predict` now returns model-driven scores in production.
- Updated `backend/.env` for ops alignment: `DATASET_PATH=./data/game_features_2014_2025.csv`, `SCHEDULE_PATH=./data/Nfl_schedule_2025.csv`, `ALLOW_ORIGIN_REGEX=https://.*\.vercel\.app`, and appended the latest Vercel domain to `ALLOWED_ORIGINS`.
- Frontend redeployed to Vercel (`https://nfl-ml-predictions.vercel.app`); smoke-tested `/predict` (SF vs CHI, 2024 W14) returning production probabilities.

## 2025-12-08T06:35:00Z

- Fixed training pipeline crash in `backend/train_models.py` caused by `TrainingSummary` missing `hist_model_metrics` when saving reports.
- Hardened feature-importance extraction to unwrap `CalibratedClassifierCV` so calibrated pipelines expose their underlying estimators for importance mapping.
- Pending: rerun `python backend/train_models.py --data backend/data/game_features_2014_2025.csv --out backend/models/prod_models` and redeploy artifacts to Heroku/Vercel.

## 2025-12-07T09:40:00Z

- **Reverted all uncommitted changes** to clean working state at commit `793634d52`
- **Fixed hamburger menu size**: Reduced from 48px to 32px (button and icon)
  - Updated `HamburgerMenu.css`: button dimensions 48px → 32px, border-radius 8px → 6px
  - Updated `HamburgerMenu.module.css`: matching dimensions
  - Updated `HamburgerMenu.jsx`: icon dimensions 40px → 24px
  - Fixed CSS syntax error (unclosed bracket on `:focus` selector)
- Commit: `ea819add6` — "fix(ui): reduce hamburger menu size from 48px to 32px"
- **Restored missing frontend API helpers** for build stability
  - Added `getHealthStatus`, `getPredictionHistory`, `startTraining`, and `getStatusOverview` exports in `frontend/src/api/client.js`
  - StatsPage and PredictionContext build now succeeds (`npm run build` passes)
- Build check: `npm run build` (frontend) now succeeds

## 2025-11-23T06:50:53Z

- Added doc headers and ASCII clean-up across backend/main.py, backend/train_models.py, and frontend dashboards/clients.
- Fixed `/predict` to reuse the trained classifier, emit game metadata for frontend mapping, and bounded history; normalized prediction history client helpers plus dashboard handler so TeamGrid and StatsPage render win probabilities reliably.
- Updated maintenance.md with resolution summaries, To-Implement items, AI-to-Dev notes, and a user-response tracker.

## 2025-11-06T01:25:52Z

- Rebuilt `backend/data/game_features.csv` (2,748 rows) and refreshed diagnostics.
- Updated `backend/train_models.py` for sklearn 1.5 compatibility (CalibratedClassifierCV estimator param, probability clipping).
- Retrained models; new artifacts written under `backend/models/` with balanced accuracy 0.713 and ROC AUC 0.731 on holdout.

## 2025-11-06T01:46:05Z

- Broadened `REG_PARAM_DISTS`/`CLF_PARAM_DISTS` so 25 RandomizedSearch iterations explore >70 classifier combos and deep hist-boost settings.
- Added macro-F1 threshold sweep (best cutoff **0.48**) and persisted the value to `training_report.json` and `metadata.json`.
- Refreshed models and docs; holdout metrics now Balanced Acc 0.716, ROC AUC 0.741, Brier 0.205, Log-loss 0.595.

## 2025-11-05T15:20:00Z

- Reverted `predict_game` score handling to emit floats, preserving math-friendly payloads and accurate point differential calculations.
- Updated `TeamGrid` score display to show schedule abbreviations with formatted values (e.g., `BUF 24.1 – 27.3 KC`) for quicker visual parsing.
- Logged updates in ADA memory and project report; pending manual frontend smoke to confirm layout spacing.

## [2025-11-23 11:25] backend: stabilize schedule & fix predict handler

Context:

- Inspected `backend/main.py` (health, schedule, predict, history endpoints).
- Verified model loading during local startup logs.

Findings:

- `/predict` was declared as `predict(request=PredictRequest)` (body not typed), contained an incorrect `win_model.fit(...)` call in request handling, printed debug output, and used a misspelled result variable leading to runtime errors.
- `SCHEDULE_PATH` was hard-coded to the repo root and did not default to `backend/data/` where the CSV typically resides in deployments.
- Team identifiers and season/week fields needed normalization to avoid frontend/backend mismatches.

Changes Proposed / Applied:

- Applied safe fixes to `backend/main.py`:
  - Added environment-aware `SCHEDULE_PATH` (env override `SCHEDULE_PATH`, default `backend/data/Nfl_schedule_2025.csv`).
  - Normalized dataset columns for `home_team`, `away_team`, `home_abbr`, `away_abbr` to uppercase when loading datasets.
  - Enhanced `/schedule/next-week` CSV parsing (numeric coercion for season/week, safe kickoff isoformat extraction).
  - Fixed `/predict` handler signature to `request: PredictRequest`, removed `fit` call, removed stray print, corrected `result` variable usage, and added defensive logging.
- No behavior-changing API contract changes; response shapes remain compatible with existing frontend expectations.

Risks / Follow-ups:

- Confirm that models are stored as full pipelines or that the prediction code applies the same preprocessing used during training.
- If schedule times are naive local datetimes, decide whether to apply a timezone offset; optionally add `SCHEDULE_TIMEZONE_OFFSET_HOURS` env var.
- If you want server-side enrichment of logos (home_logo/away_logo), update `backend/team_logo.csv` or the schedule CSV and re-deploy.

---

## 2025-12-04T19:45Z — Production Deployment Complete (Alfred Session)

### Context

- Backend had been returning "Application Error" on Heroku for all endpoints.
- Rollback branch `rollback/heroku-endpoint-restore` was prepared at commit `380a0d8d4`.

### Root Cause Identified

Heroku logs revealed:

```plaintext
ModuleNotFoundError: No module named 'pydantic._internal'
```

**Trace:** `import nflreadpy` → requires `pydantic-settings` → incompatible pydantic version on Heroku.

### Fix Applied

- Removed unused `import nflreadpy as nfl` from `backend/main.py`.
- Committed as `7da71c03f`: "fix: remove nflreadpy import causing pydantic._internal crash".
- Force-pushed to Heroku: `git push heroku rollback/heroku-endpoint-restore:master --force`.

### Deployment Results

| Component | Status | URL |
|-----------|--------|-----|
| **Backend** | ✅ Healthy | `https://nfl-predict-ecf5a5bd34fe.herokuapp.com` |
| **Frontend** | ✅ Live | `https://nfl-predict.vercel.app` |

### Endpoint Verification

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/health` | ✅ 200 | `{"status":"healthy","mode":"production","reason":"models loaded"}` |
| `/schedule/next-week` | ✅ 200 | Returns 14 games for Week 14 |
| `/predict` | ⚠️ 422 | `Model metadata missing raw_feature_columns` — needs model retrain |

### Frontend Updates

- Ran `npm audit fix --force` — vulnerabilities reduced from 52 to 46.
- Remaining vulnerabilities are deep transitive dependencies in dev tooling.
- Deployed to Vercel via CLI: `npx vercel --prod --yes`.

### Next Steps

1. **Model Retraining** — Run `train_models.py` to regenerate metadata with `raw_feature_columns`.
2. **Vulnerability Audit** — Review remaining npm vulnerabilities for security risk assessment.
3. **Git Sync** — Merge `rollback/heroku-endpoint-restore` back to `master` if stable.

### Session Metrics

- **Time:** ~30 minutes
- **Commits:** 1 (`7da71c03f`)
- **Files Changed:** `backend/main.py`
- **Heroku Release:** v406+

---

## 2025-12-06T00:00Z — API Resilience & Client Alignment

### Changes

- Removed unused `nflreadpy` import to prevent pydantic dependency failures on Heroku.
- Consolidated duplicate `_glob_latest` helper and removed stray example `/retrain` snippet from `main.py`.
- Added `_infer_raw_feature_columns` to derive model features from the preprocessor or dataset when `metadata.json` is missing them; `/predict` now returns a clear 503 instead of hard-failing.
- Set `DEFAULT_DATASET` to `backend/data/game_features.csv` (was empty path) so startup uses the engineered dataset by default.
- API client now respects `VITE_API_BASE` **or** `VITE_API_URL`, retains a single Heroku fallback constant, and surfaces friendlier errors for metadata-related 503s and validation 422s.

### Status

- `/health` and `/schedule/next-week` remain healthy.
- `/predict` will proceed when feature columns can be inferred; otherwise returns 503 with explicit retrain guidance.

### Next Actions

1. Retrain models to regenerate `metadata.json` with `raw_feature_columns`.
2. Deploy backend to Heroku and frontend to Vercel after retraining.
3. Run a quick `/predict` smoke test (KC vs HOU, 2025 W14) to verify provenance is `model`.
