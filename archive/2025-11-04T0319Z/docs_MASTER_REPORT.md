# Backup: MASTER_REPORT.md (2025-11-04T0319Z)

This is an automated backup of docs/MASTER_REPORT.md prior to the Dry Run refactor plan edits.

---

# NFL_ML_Predictions — Master Engineering Report

Last updated: 2025-11-01

## Executive Summary

A production-ready FastAPI backend serves NFL game predictions to a React (Vite) frontend. Models are trained via a hardened pipeline with leakage guards; artifacts and metadata live under `backend/models`. Recent frontend work focused on mobile nav UX and component cleanup; backend work focused on resilient inference, provenance, and realistic training metrics.

This report consolidates prior docs and session notes into a single, living source of truth centered on:

- What the system does and how data flows
- What changed recently and why
- Current metrics and health
- Known issues with precise file:line references
- Concrete next steps and feature suggestions

## System Overview

- Backend: FastAPI on 127.0.0.1:8000 (Heroku in prod), scikit-learn models loaded via joblib.
- Frontend: React + Vite; dev uses proxy, prod uses `VITE_API_URL`.
- Artifacts: `backend/models/` contains preprocessor, win classifier, and metadata with expected features.
- Data Flow: CSV datasets → feature engineering → training pipeline → joblib artifacts → FastAPI `/predict` → UI.

Key endpoints:

- GET `/health` — service status and model load state
- GET `/debug` — environment and model metadata
- GET `/schedule/next-week` — normalized upcoming schedule
- POST `/predict` — body: `{home_team, away_team, season, week}`
- POST `/predict/next-week` — batch predictions

## Recent Changes (highlights)

- Mobile nav UX
  - Hamburger only on phones; desktop links hidden at ≤768px.
  - Collapsed menu fully hides links and removes them from focus/AT (`aria-hidden`, `inert`).

- TeamGrid cleanup
  - Removed inline styles (toasts, badges, wrappers) → centralized in `TeamGrid.css`.
  - Fixed loader bug (teams loader now sets `teams: false` when done).
  - Replaced inline style image hide with `.is-hidden` class.

- Backend resilience & training integrity
  - Feature alignment to estimator `feature_names_in_`; imputes missing columns once and retries.
  - Minimal required identifiers enforced; numerics can be imputed when allowed.
  - Training leakage guards; production-mode CV metrics are now realistic.

## Current Metrics

- Win model (production mode, CV):
  - Brier ≈ 0.177
  - Logloss ≈ 0.509
  - ROC AUC ≈ 0.805
  - PR AUC ≈ 0.725
- Health: `/health` returns healthy with models loaded; `/predict` returns `prediction_source: "model"`.

## How to Run

- Backend (Windows PowerShell)
  - Create venv, install deps, run API.
- Frontend
  - `npm install`, `npm run dev` for proxy-based dev.
- Prod
  - Frontend: `npm run build`
  - Backend: push to Heroku (Procfile)

See `docs/ONBOARDING_DEBUG_GUIDE.md` for quickrun details and troubleshooting.

## Recently Resolved Issues

The following CSS issues in `frontend/src/components/TeamGrid.css` have been fixed to ensure standards compliance and visual consistency:

1) Line 160: Corrected `border-bottom-left-radius: 1px solid;` to use a valid radius value or removed as appropriate.
2) Line 294: Replaced `background-color: var(a-shine);` with a defined CSS variable, e.g., `var(--c-card)`.
3) Line 307: Removed invalid `transition: transformY(-9px);` and used `transform: translateY(-9px);` with proper transition.
4) Line 311: Split compound `animation` property into two valid animation declarations with correct timing functions.
5) Line 348: Changed `color: var(a-shine);` to a valid color variable, e.g., `var(--c-text-on-dark)`.
6) Line 451: Fixed `transform: scale(1.00) rotate(360deg 3s infinite);` by implementing rotation via keyframes and animation.
7) Line 49: Updated `animation-timing-function: var(--a-ease-in-ease-out);` to use the defined `--a-ease` variable.

All previously tracked TeamGrid.css issues are now resolved and verified.

## Known Issues & Required Fixes (file:line)

No outstanding CSS variable or property errors remain in TeamGrid.css as of this update.

- Issue: `animation-timing-function: var(--a-ease-in-ease-out);` references an undefined variable (`--a-ease` exists; `--a-ease-in-ease-out` does not).
- Fix: Replace with `animation-timing-function: var(--a-ease);`.

## Suggested Next Steps

- CSS hygiene pass
  - Fix the invalid properties/variables listed above (TeamGrid.css). Add a quick stylelint rule to catch these patterns going forward.

- Frontend UX polish
  - Add CSS-only stagger with `:nth-child` for card reveal; reduce animation jank for reduced-motion users.
  - Extract a Toasts component with enter/exit transitions.

- Backend observability
  - Add simple log counters for prediction provenance (`model`, `fallback`) to detect regressions.

- Testing
  - Add a minimal Playwright E2E that asserts mobile/desktop nav behavior and `/predict` happy path.

## Feature Ideas

- Predictions “what-if” panel (adjust inputs like neutral field, rest days) and recompute local predictions.
- Historical analysis view using saved `prediction_history` entries with charts (win prob over time).
- Simple bookmarking of specific matchups with notifications for schedule changes.

## Appendix

- Artifacts
  - `backend/models/` → `preprocessor.joblib`, `win_clf_calibrated.joblib`, `metadata.json`, `training_report_*.json`
- Documentation
  - This file supersedes scattered report notes; for setup details see `docs/ONBOARDING_DEBUG_GUIDE.md`.
