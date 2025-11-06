# Repository Cleanup — Dry Run (2025-11-06)

Purpose: Propose a minimal, production-focused file set. This is a NON-DESTRUCTIVE preview. No files are deleted until you reply with the confirmation token shown below.

## Keep (runtime/deploy critical)

- Backend (API)
  - `backend/main.py`, `backend/__init__.py`
  - `backend/requirements.txt`
  - `backend/models/` → `home_model.joblib`, `away_model.joblib`, `win_clf_calibrated.joblib`, `preprocessor.joblib`, `metadata.json`, `training_report.json`
  - `backend/data/game_features.csv` (and small CSVs required for startup), exclude heavy/legacy datasets
  - `backend/tests/` and `backend/test_*.py`

- Frontend (UI)
  - `frontend/package.json`, `frontend/vite.config.js`, `frontend/tsconfig.json`, `frontend/vercel.json`
  - `frontend/public/**/*`
  - `frontend/src/**/*`

- Deployment & Root Config
  - `Procfile`, `heroku.yml`, `app.json`, `vercel.json`
  - Root `README.md`, `requirements.txt` (if used by platform), `requirements-lock.txt`
  - `.github/**/*` project automation

## Archive/Delete Candidates (not needed for runtime)

- Large/interactive artifacts
  - `backend/merge.ipynb`, `backend/trainer_mock.ipynb`, `backend/data/legacy_data/**/*`, `backend/data/Pred_history_data.ipynb`

- Training-only utilities (move to scripts/ or archive)
  - `backend/build_csv_datasets.py`, `backend/analyze_merge_datasets.py`, `backend/transform_dataset.py`, `backend/ts_split.py`, `backend/train_models.py`

- Installed artifacts accidentally tracked
  - `node_modules/**/*` (should be git-ignored; Vercel/Heroku install from package.json)
  - `backend/.venv/**/*` (should be git-ignored)

- Redundant docs/misc
  - `test_schedule.html`, duplicated maintenance docs (retain a single canonical copy in `docs/`)

If you prefer not to delete, we can move these to `archive/` for safekeeping.

## Actions this plan would perform

1) Add a `.gitignore` to ignore `node_modules/`, `backend/.venv/`, `.pytest_cache/`, `__pycache__/`, and build outputs.
2) Remove tracked `node_modules/` from the repository history (working tree delete + new commit). Size and diff will shrink substantially.
3) Delete or move to `archive/` the listed Jupyter notebooks and legacy datasets not needed by the running app.
4) Keep a concise `docs/` set: `README.md`, `IMPLEMENTATION_SUMMARY.md`, `report.md`. Move niche docs to `docs/archive/`.

## Safety checks

- Backend: run `pytest -q` and `uvicorn backend.main:app` smoke checks after cleanup.
- Frontend: run `npm run build` and open Vercel preview.

## Confirmation

Reply with the token below to proceed with the cleanup in a new branch and open a PR:

CONFIRM: CLEAN-20251106

On apply, I will also write a summary to `./alfred.log.md` and update `docs/report.md` with the change log.
