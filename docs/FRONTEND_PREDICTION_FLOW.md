# Frontend Prediction Flow

This is the smallest accurate mental model for the current frontend.

## The Main Idea

The app now has one shared prediction state owner for the primary app shell.

- `frontend/src/App.jsx` creates auth state and route structure.
- `frontend/src/hooks/usePredictionState.js` owns shared schedule, history, summary, logos, health, and per-matchup request state.
- `frontend/src/components/DashBoard/Dashboard.jsx` reads that shared state and triggers user actions.
- `frontend/src/components/HistoryPage.jsx` renders the same shared history state.

That architecture matters because older versions of the app duplicated schedule and prediction state inside the dashboard, which made pages drift out of sync.

## Active Request Flow

1. `App.jsx`
   Creates the auth session and mounts the protected app shell.

2. `usePredictionState.js`
   Hydrates the next slate, history, history summary, logos, and health status.

3. `client.js`
   Handles HTTP transport, normalizes backend response shapes, and applies compatibility fallbacks.

4. `Dashboard.jsx`
   Uses the shared state to load a slate, predict one game, or predict the whole board.

5. `HistoryPage.jsx`
   Receives the shared history and summary through props.

6. `StatsPage.jsx`
   Is the exception: it fetches its own status snapshot directly instead of consuming the shared hook.

## Which Client File Is Real?

The active frontend transport layer is:

- `frontend/src/api/client.js`

The active app shell no longer keeps a second legacy fetch wrapper in the supported path. Transport quirks and compatibility fallbacks belong in `client.js`.

## Shared Data Contracts

The frontend works with several "game-like" payloads:

- schedule rows from `/schedule` or `/schedule/next-week`
- prediction responses from `/predict`
- history entries from `/history`

Those shapes are similar, but not identical. The frontend keeps them aligned with:

- `frontend/src/utils/gameUtils.js` for matchup keys and prediction payloads
- `frontend/src/api/client.js` for transport-level normalization
- `frontend/src/hooks/usePredictionState.js` for shared app state

## Why Matchup Keys Matter

The dashboard keeps parallel maps for:

- predictions
- loading flags
- per-card errors

If one screen builds keys differently, the UI looks broken even when the network call succeeded. The current flow avoids that by using the same season-week-home-away key logic across schedule rows, predictions, and history.

## Compatibility Fallbacks

The frontend now tolerates older backend deployments.

### History summary fallback

If `/history/summary` is missing, `client.js` derives summary metrics from `/history`.

### Queryable schedule fallback

If `/schedule?season=<year>&week=<week>` is missing, `client.js` falls back to bundled CSVs in `frontend/public/schedules/`.

### Next-slate fallback

If `/schedule/next-week` is missing, the client also falls back to local schedule assets.

### Status overview fallback

If `/status/overview` is unavailable, the UI uses a safe fallback object instead of crashing the app shell.

## Backend Touchpoints

The frontend primarily depends on:

- `GET /health`
- `GET /status/overview`
- `GET /schedule`
- `GET /schedule/next-week`
- `GET /teams/logos`
- `GET /history`
- `GET /history/summary`
- `POST /predict`

The main backend implementation lives in `backend/main.py`.

## Auth Boundary

- `useAuthSession.js` stores a local-device session in browser storage.
- The frontend derives `X-User-Id` from that local session email.
- This supports user-scoped history in the current app, but it is not a real authentication boundary.

## Safe Extension Rules

If you change the frontend later, keep these boundaries:

- Put transport quirks and legacy response handling in `client.js`
- Keep shared route-level prediction state in `usePredictionState.js`
- Keep `Dashboard.jsx` focused on user actions and rendering decisions
- Keep presentational card components unaware of transport details

That separation is what keeps the dashboard, history page, and status page coherent.
