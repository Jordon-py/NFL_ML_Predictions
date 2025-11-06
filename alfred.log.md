# alfred.log.md — Operational Log

Date: 2025-11-06
Branch: alfred/deploy-20251105

Activity:

- Deployed frontend to Vercel Production successfully after spinner dependency removal.
- Prepared repository cleanup DRY RUN to reduce footprint and remove tracked installs.

Proposed Cleanup (awaiting confirmation):

- Keep: backend API code and models; frontend src/public; deployment configs; core docs.
- Archive/Delete: notebooks, legacy data, training-only scripts (or move to scripts/), tracked node_modules, venv, duplicate docs.
- Add .gitignore to prevent reintroduction of installed artifacts.

Next Steps on CONFIRM: CLEAN-20251106

1) Create branch `chore/cleanup-2025-11-06`.
2) Add `.gitignore` and remove tracked `node_modules` and `.venv`.
3) Move notebooks/legacy data to `archive/` (or delete per confirmation).
4) Run health checks: pytest, dev server, and Vite build.
5) Open PR with detailed summary and file list.

Quality Gates (current session):

- Build: PASS (Vercel prod build completed)
- Lint/Typecheck: N/A (not executed this step)
- Tests: N/A (not executed this step)

# Alfred Activity Log

## 2025-11-06T03:45:00Z

- Removed the `react-spinners` dependency from StatsPage and replaced it with an in-house CSS spinner to stabilize Vercel builds.
- Cleaned `frontend/package.json` and regenerated the lock file via `npm install --prefix frontend` (engine warning noted: repo prefers Node 20.x).
- Documented the fix across `docs/report.md` and `.debug_memory.json`; redeploy pending to verify production.

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
