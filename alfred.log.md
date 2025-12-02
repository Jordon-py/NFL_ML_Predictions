# Alfred Activity Log

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
