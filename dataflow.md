# Dataflow Mapping

## Overview
This application is a full-stack NFL forecasting tool that uses a shared-state pattern for data distribution.

### Frontend
- **State Initialization**: `App.jsx` orchestrates global routing and initializes `usePredictionState.js`.
- **Data Fetching**: `usePredictionState.js` fetches hydration data (initial predictions, matchups, and summary stats) from the backend.
- **Dashboard Component**: `Dashboard.jsx` takes the prediction state and passes `matchup` and `prediction` data down to the `Card.jsx` grid.
- **Card Component**: `Card.jsx` takes `matchup` and `prediction` props. It derives UI labels and visual stats via `derivePredictionMeta()`.
- **History Chart**: `HistoryChart.jsx` visualizes the array of history events. It provides functional UI filters (Resolved/Pending/Search) and sorts by Date, Confidence, or Margin Delta.

### Backend
- **API (FastAPI)**: Serves endpoints for fetching schedules, making predictions, and reading prediction history.
- **Models**: Predictions and History rely on CSV features (`game_features_*.csv`).

## Key Data Transformations
- `Card.jsx`: `(matchup, prediction) -> derivePredictionMeta(matchup, prediction)` computes probability arrays and win/loss states.
- `HistoryChart.jsx`: `(historyArray) -> normalizeHistoryRow()` formats timestamps and calculates spread deltas before filtering and sorting.
