# Frontend Audit & Simplification (React/Vite)

## Goals
- Keep the UI production-ready while reducing complexity and removing dead code.
- Align the frontend strictly to the backend endpoints that exist in `backend/main.py`.
- Avoid `useContext`/`useMemo` unless they clearly add value.

## API Contract (Verified Against `backend/main.py`)
The frontend now calls only these endpoints:
- `GET /schedule/next-week` → list of games
- `POST /predict` → single-game prediction
- `GET /history?limit=...` → recent predictions
- `GET /status/overview` → health + dataset + history metrics

Notes:
- The previous UI referenced `/teams/logos` and `/predict/next-week`; those routes are not exposed by `backend/main.py`, so the frontend no longer calls them.
- “Predict All Games” now performs one `/predict` call per game with a small concurrency limit.

## What Changed (High-Level)
- Removed `PredictionContext` and related context-based docs/components to avoid global state and stale abstractions.
- Simplified the prediction flow:
  - `Dashboard` owns schedule + predictions state.
  - `TeamGrid` is presentational and delegates actions via callbacks.
  - `Card` is a small, stable presentational component using the existing `TeamGrid.css` styles.
- Moved `NavBar` into `App.jsx` so every route has consistent navigation.
- Removed unused hooks/utilities/components (`useNextWeekSchedule`, training hooks, debug log, unused buttons, etc.).
- Removed unused frontend dependencies (`@material/web`, `papaparse`).
- Added `frontend/vite.config.js` to filter a known-safe Rollup warning (`"use client"` directives from React Router) so `npm run build` is warning-free.
- Updated `npm test` to pass when no tests exist: `vitest --passWithNoTests`.

## Environment Variables
This is a Vite app (not CRA). Use:
- `VITE_API_BASE` (recommended)

Examples:
- Local dev backend: `VITE_API_BASE=http://127.0.0.1:8000`
- Production (Vercel): `VITE_API_BASE=https://<your-heroku-app>.herokuapp.com`

## Build / Run Locally
```bash
cd frontend
npm install
npm run build
npm run preview
```

## Deployment Notes (Heroku + Vercel)
### Heroku (Backend)
This repo’s Heroku deployment is a Python/FastAPI app:
- `Procfile` is `web: gunicorn ... backend.main:app`
- Buildpack is `heroku/python`

Recommended verification commands (run from a machine with Heroku CLI auth):
```bash
heroku login
heroku apps
heroku git:remote -a <app-name>
heroku config -a <app-name>
heroku logs --tail -a <app-name>
```

### Vercel (Frontend)
- Push to GitHub; Vercel typically auto-builds from `frontend/`.
- Ensure `VITE_API_BASE` is set in Vercel project env vars.

## Quick Smoke Checks
From any terminal:
```bash
curl -sS https://<api-host>/status/overview | head
curl -sS https://<api-host>/schedule/next-week | head
curl -sS "https://<api-host>/history?limit=5" | head
```

