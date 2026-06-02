# Dataflow Mapping

## Overview

This application is a full-stack NFL forecasting tool that uses a shared-state pattern for data distribution.

### Frontend

- **State Initialization**: `App.jsx` orchestrates global routing and initializes `usePredictionState.js`.
- **Data Fetching**: `usePredictionState.js` fetches hydration data (initial predictions, matchups, and summary stats) from the backend.
- **Dashboard Component**: `Dashboard.jsx` takes the prediction state and passes `matchup` and `prediction` data down to the `TeamGrid.jsx` / `Card.jsx` grid. `TeamGrid.jsx` now owns local slate filters for team/stadium search and prediction status.
- **Card Component**: `Card.jsx` takes `matchup` and `prediction` props. It derives UI labels and visual stats via `derivePredictionMeta()`. On forecast completion, users can trigger the Premium AI breakdown drawer.
- **Premium Chat**: `Dashboard.jsx` hosts the conversational Premium AI Coach floating panel and sends user questions through `premiumChat()`.
- **History Chart**: `HistoryChart.jsx` visualizes the array of history events, providing functional UI filters (Resolved/Pending/Search) and sorting by Date, Confidence, or Margin Delta.
- **Stats Overview**: `StatsPage.jsx` fetches schedule, history, `/status/overview`, and `/history/summary` independently, rendering readiness summaries and forecast history quality.

### Backend

- **API (FastAPI)**: Serves endpoints for fetching schedules, making predictions, reading prediction history, and Premium AI explain/chat.
- **LLM Agent**: `NFLAgent` in `backend/ollama/llm_ollama.py` handles dataset-aware prompts, Ollama authentication headers, model fallback, and legacy `explain_prediction` / `chat_messages` helpers.
- **History safety**: Premium endpoints call the prediction path with request-scoped persistence disabled so AI enrichment does not create duplicate history records.

## Key Data Transformations

- `Card.jsx`: `(matchup, prediction) -> getPremiumExplanation(matchup) -> POST /premium/explain` generates structured game breakdowns.
- `Dashboard.jsx`: `(chatInput, season, week) -> premiumChat(chatInput) -> POST /premium/chat` returns chat responses enriched with live model predictions.
- `Card.jsx`: `(matchup, prediction) -> derivePredictionMeta(matchup, prediction)` computes probability arrays and win/loss states.
- `TeamGrid.jsx`: `(games, predictions, loading, errors, local filters) -> filteredGames` controls visible cards and the target list passed into bulk prediction.
- `HistoryChart.jsx`: `(historyArray) -> normalizeHistoryRow()` formats timestamps and calculates spread deltas before filtering and sorting.
- `StatsPage.jsx`: `(schedule, history, status overview, history summary) -> overview metrics` shows backend readiness, dataset rows, upcoming slate count, and forecast history quality.
