# Dataflow Mapping

## Overview

This application is a full-stack NFL forecasting tool that uses a shared-state pattern for data distribution.

### Frontend

- **State Initialization**: `App.jsx` orchestrates global routing and initializes `usePredictionState.js`.
- **Data Fetching**: `usePredictionState.js` fetches hydration data (initial predictions, matchups, and summary stats) from the backend.
- **Dashboard Component**: `Dashboard.jsx` takes the prediction state and passes `matchup` and `prediction` data down to the `TeamGrid.jsx` / `Card.jsx` grid. `TeamGrid.jsx` now owns local slate filters for team/stadium search and prediction status, then passes the visible slate back to `Dashboard.jsx` when the user runs "Predict visible".
- **Card Component**: `Card.jsx` takes `matchup` and `prediction` props. It derives UI labels and visual stats via `derivePredictionMeta()`.
- **History Chart**: `HistoryChart.jsx` visualizes the array of history events. It provides functional UI filters (Resolved/Pending/Search) and sorts by Date, Confidence, or Margin Delta.
- **Stats Overview**: `StatsPage.jsx` fetches schedule, history, `/status/overview`, and `/history/summary` independently, then renders the shared `NavBar`, readiness summary cards, schedule coverage, and history feedback chart.

### Backend

- **API (FastAPI)**: Serves endpoints for fetching schedules, making predictions, and reading prediction history.
- **Models**: Predictions and History rely on CSV features (`game_features_*.csv`).

## Key Data Transformations

- `Card.jsx`: `(matchup, prediction) -> derivePredictionMeta(matchup, prediction)` computes probability arrays and win/loss states.
- `TeamGrid.jsx`: `(games, predictions, loading, errors, local filters) -> filteredGames` controls visible cards and the target list passed into bulk prediction.
- `HistoryChart.jsx`: `(historyArray) -> normalizeHistoryRow()` formats timestamps and calculates spread deltas before filtering and sorting.
- `StatsPage.jsx`: `(schedule, history, status overview, history summary) -> overview metrics` shows backend readiness, dataset rows, upcoming slate count, and forecast history quality.
