# Documentation Consolidated Overview

Last updated: 2025-11-02 18:45 UTC

This single page summarizes the most important information from the docs folder and points to the canonical sources after cleanup. Redundant or historical documents have been archived under `docs/legacy/`.

## What you need most days

- API endpoints and contracts: see `docs/AI-METRICS.md` (Data shapes, functions, interactions)
- How the system flows end-to-end: `docs/DATA_FLOW.md` and `docs/ARCH_MAP.md`
- Runbook and operations: `docs/RUNBOOK.md`
- Recent changes and engineering highlights: `docs/report.md` and `docs/session_completion_report.md`

## CORS and environment

The project previously had multiple CORS docs (guide, summary, checklist, quick ref). These are now unified:

- Canonical: Configure CORS via `RESTRICT_CORS=true` and `ALLOWED_ORIGINS` (comma-separated origins) in backend; frontend uses `VITE_API_BASE` in production and Vite proxy in dev.
- Quick reference: `docs/RUNBOOK.md` contains the short operational commands for verification (health, OPTIONS preflight, predict).
- Archived, for historical detail: see `docs/legacy/` (CORS_* and API_CORS_CHECKLIST).

## Key contracts (frontend ↔ backend)

- POST /predict
  - Request: { home_team, away_team, season, week }
  - Response: { home_score, away_score, home_win_probability, away_win_probability, point_diff, prediction_source, mode }

- GET /schedule/next-week
  - Response: Array<{ home_team, away_team, season, week, kickoff_local, id }>

## Recent fix of note

- Feature assembly bug in `_build_future_row` fixed (pre_cum metrics now return all five fields together). This resolves uniform predictions and increases `prediction_source: "model"` coverage.

## Archived documents

The following docs were consolidated and moved to `docs/legacy/` to reduce duplication:

- API_CORS_CHECKLIST.md
- CORS_API_CONFIGURATION.md
- CORS_CONFIGURATION_SUMMARY.md
- CORS_QUICK_REFERENCE.md
- MODEL_FIX_SUMMARY.md
- SCHEDULE_FIX_SUMMARY.md
- TRAIN_MODELS_REFACTOR.md
- enhancement_workflow.md

## Pointers

- Training and evaluation: `backend/enhanced_pipeline.py`, reports under `backend/reports/`
- Artifacts at runtime: `backend/models/` (preprocessor, regressors, calibrated classifier, metadata)
- Data inputs: `backend/data/` (engineered datasets, schedule CSV)

— This page will remain the stable index for maintainers. See `docs/legacy/` for full historical references.
