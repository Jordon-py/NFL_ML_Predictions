# Onboarding & Debug Guide

This guide helps you spin up the NFL_ML_Predictions project fast, understand how things fit together, and troubleshoot common issues.

## Overview

- Backend: FastAPI (Python) serving predictions and schedule endpoints
- Frontend: React (Vite) consuming backend via REST
- Models: scikit-learn artifacts under `backend/models/` with `metadata.json`
- Data flow: CSV datasets → training pipeline → joblib artifacts → FastAPI `/predict` → UI

## Quick Start

- Backend (Windows PowerShell):
  1. Create/activate venv: `cd backend; python -m venv .venv; .\.venv\Scripts\Activate.ps1`
  2. Install deps: `python -m pip install -r requirements.txt`
  3. Run API: `python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`
- Frontend:
  1. `cd frontend; npm install`
  2. Dev server: `npm run dev` (Vite proxies API calls in dev)

Prod builds:

- Frontend: `cd frontend && npm run build`
- Heroku: push to the Heroku remote to build and release (Procfile present)

## Key Endpoints

- GET `/health` — status and model load info
- GET `/debug` — environment and model metadata
- GET `/schedule/next-week` — normalized next week schedule
- POST `/predict` — body: `{ home_team, away_team, season, week }`
- POST `/predict/next-week` — batch predict upcoming games

## Configuration

- Backend `.env` (in repo root and/or `backend/.env`):
  - `DATASET_PATH` — CSV used for schema checks and future-row assembly
  - `ALLOW_FALLBACK_PREDICTIONS` — allow imputation-based predictions when engineered columns are missing
  - `CORS_ORIGINS` — comma-separated list for FastAPI CORS
- Frontend: uses Vite proxy in dev; in prod, set `VITE_API_BASE_URL` or configure `frontend/src/api/client.js`

## Common Issues & Fixes

- Failed to fetch in dev:
  - Ensure backend is on 127.0.0.1:8000 and Vite is proxying or API base is set correctly.
- 400 Bad Request on `/predict` (missing columns):
  - Identifiers must exist: `home_team`, `away_team`, `home_game_date`.
  - Numeric feature gaps are imputed when ALLOW_FALLBACK_PREDICTIONS=true.
- Model feature mismatch errors (sklearn):
  - The server aligns inputs to `model.feature_names_in_`; check `/debug` for loaded feature counts.
- Casing mismatch on artifact filenames (Linux/Heroku):
  - The loader resolves files case-insensitively, but prefer consistent casing in `metadata.json`.

## Debugging Workflow

1. Sanity check the backend:
   - Start API → visit `/health` and `/debug`
2. Smoke the schedule and one prediction:
   - GET `/schedule/next-week`, then POST `/predict` for one game
3. Verify provenance:
   - Check `prediction_source` in responses (`model`, `model+win_fallback`, etc.)
4. If predictions fallback frequently:
   - Confirm `DATASET_PATH` points to engineered features (e.g., `merge_dominance.csv`)
   - Consider retraining to update `metadata.json` feature columns

## Training (Backend)

- Runner: `backend/enhanced_pipeline.py`
- Outputs: `backend/models/` (joblibs, `metadata.json`), `backend/reports/`
- Notes:
  - Leakage guard filters target-derived features
  - Production mode trains on all rows and reports CV metrics

## Frontend Tips

- Hamburger menu is mobile-only (hidden ≥768px) via CSS
- TeamGrid shows inline errors per-card; full-page errors only on bootstrap failures
- Use browser devtools Network tab to inspect `/predict` and `/schedule` calls

## Where to Look

- `backend/main.py` — endpoints, CORS, model loading
- `backend/models/metadata.json` — feature schema and artifact paths
- `frontend/src/api/client.js` — API base logic
- `docs/` — change logs and architecture notes

## Useful Scripts

- `scripts/verify_api_cors.py` — quick CORS probe
- `backend/tests/` — pytest-based startup checks

## Support Checklist

- [ ] `/health` returns healthy with models loaded
- [ ] `/schedule/next-week` returns games
- [ ] `/predict` returns probabilities with `prediction_source: "model"`
- [ ] Frontend dev server proxies API in dev
- [ ] Heroku release shows vX with successful build logs
