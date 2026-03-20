# Frontend Prediction Flow

This note explains the smallest useful mental model for the current React app.

## Why this document exists

The frontend has a simple runtime path, but several files touch the same data:

- the dashboard loads a schedule
- the user predicts a matchup
- the status pages read history and health
- the UI has to keep schedule rows, prediction responses, and history entries aligned

The important rule is that the frontend now uses one shared normalization layer in [frontend/src/utils/gameUtils.js](../frontend/src/utils/gameUtils.js).

## Request flow

1. `frontend/src/App.jsx`
   Loads the router and asks `getStatusOverview()` for a lightweight health snapshot used by the nav.

2. `frontend/src/components/DashBoard/Dashboard.jsx`
   Owns the "next week schedule" and the per-game prediction state.

3. `frontend/src/api/client.js`
   Wraps HTTP calls and normalizes older backend response shapes into stable frontend shapes.

4. `frontend/src/components/Card/TeamGrid.jsx`
   Receives schedule rows plus prediction/loading/error maps and renders one `Card` per game.

5. `frontend/src/pages/StatsPage.jsx`
   Reads schedule, history, and overview data again for a status-oriented page.

## Shared frontend contract

The UI repeatedly receives "game-like" objects from different sources:

- schedule rows from `/schedule/next-week`
- prediction responses from `/predict`
- history entries from `/history`

Those objects are similar, but not identical. `gameUtils.js` keeps the following rules in one place:

- `normalizeTeamCode(value)` converts team identifiers into the uppercase format expected by the API.
- `getGameWeek(gameLike)` and `getGameSeason(gameLike)` tolerate older field aliases such as `week_num`.
- `buildMatchupKey(gameLike)` creates the composite key used by dashboard state maps.
- `buildPredictPayload(gameLike)` builds the exact body expected by `POST /predict`.
- `normalizeMatchup(gameLike)` reshapes a raw schedule row into the smaller card-friendly format.

## Why the key matters

The dashboard stores three parallel maps:

- `predictions`
- `loadingMap`
- `errorsMap`

Each map is keyed by the same season-week-home-away composite string. If one screen invents a slightly different key, the UI appears "out of sync" even though the network call succeeded. Centralizing key generation removes that class of bug.

## Backend touchpoints

The current frontend depends primarily on these backend endpoints:

- `GET /status/overview`
- `GET /health`
- `GET /schedule/next-week`
- `GET /history`
- `POST /predict`

The main backend implementation lives in [backend/main.py](../backend/main.py), while prediction orchestration is explained in [backend/services/prediction_service.py](../backend/services/prediction_service.py).

## Safe extension points

If you need to change behavior later, start here:

- add new schedule/prediction field aliases in `gameUtils.js`
- keep `client.js` responsible for transport and response-shape cleanup
- keep `Dashboard.jsx` responsible for user actions and per-card state
- keep `TeamGrid.jsx` and `Card.jsx` focused on rendering
