# API Endpoint Map

## Overview

This document outlines the connection between frontend and backend endpoints, corrected for accuracy and simplified for clarity. It serves as the source of truth for the NFL Prediction App's API ecosystem.

---

## Endpoint Flow Diagram

```mermaid
graph TD
    %% Frontend Components
    UI_Health[Status/Health UI]
    UI_Schedule[Schedule/Home UI]
    UI_Predict[Prediction Card UI]
    UI_Chat[LLM Chat UI]
    UI_History[History Page]

    %% Client Layer
    Client[api/client.js]

    %% Backend Layer
    API_Core[FastAPI: main.py]
    
    %% Backend Subsystems
    Sub_Models[InferenceBundle (Models)]
    Sub_Data[Dataset (Pandas DataFrame)]
    Sub_Ollama[Ollama Integration]
    Sub_History[History (JSON File)]

    %% Edges
    UI_Health -->|getHealthStatus| Client
    UI_Schedule -->|getNextWeekSchedule| Client
    UI_Predict -->|predictGame| Client
    UI_Chat -->|chatLLM| Client
    UI_History -->|getPredictionHistory| Client

    Client -->|GET /health| API_Core
    Client -->|GET /schedule/next-week| API_Core
    Client -->|POST /predict| API_Core
    Client -->|POST /llm/chat| API_Core
    Client -->|GET /history| API_Core
    Client -->|GET /status/overview| API_Core

    API_Core -->|Read| Sub_Models
    API_Core -->|Read| Sub_Data
    API_Core -->|Chat/Explain| Sub_Ollama
    API_Core -->|Read/Write| Sub_History
```

---

## Validated Endpoints

### 🟢 Core System

#### `GET /health`

**Description:** Health check to verify if models and datasets are loaded.

- **Frontend Caller:** `getHealthStatus()`
- **Backend Handler:** `health()`
- **Response:**

  ```json
  { "status": "healthy", "mode": "production", "reason": "models and dataset loaded" }
  ```

- **Errors:** `503 Service Unavailable` (if not initialized)

#### `GET /status/overview`

**Description:** Detailed system status including dataset stats and history counts.

- **Frontend Caller:** `getStatusOverview()`
- **Backend Handler:** `get_status_overview()`
- **Response:** `{ health, dataset, history }`

---

### 🏈 Prediction & Schedule

#### `GET /schedule/next-week`

**Description:** Fetches upcoming games for the current/next week. Factors in live API data with CSV fallbacks.

- **Frontend Caller:** `getNextWeekSchedule()`
- **Backend Handler:** `get_next_week_schedule()`
- **Response:**

  ```json
  { "games": [{ "home_team": "KC", "away_team": "BUF", "season": 2025, ... }] }
  ```

#### `POST /predict`

**Description:** Generates win probability and score predictions for a specific matchup.

- **Frontend Caller:** `predictGame(payload)`
- **Backend Handler:** `predict_game(payload)`
- **Input:**

  ```json
  { "home_team": "KC", "away_team": "BUF", "season": 2025, "week": 11 }
  ```

- **Output:** `PredictionResponse` (includes `home_win_probability`, `home_score`, `simulation_metrics`)

#### `GET /history`

**Description:** Retrieves recent predictions stored in the backend's local JSON history.

- **Frontend Caller:** `getPredictionHistory(limit)`
- **Backend Handler:** `get_history(limit)`
- **Query Params:** `limit` (default: 100)
- **Response:** `{ "entries": [...], "total": 42 }`

---

### 🧠 Intelligence & Assets

#### `POST /llm/chat`

**Description:** Conversational interface often provided with prediction context.

- **Frontend Caller:** `chatLLM(payload)`
- **Backend Handler:** `llm_chat(payload)`
- **Input:** `{ "messages": [...], "prediction": {...} }`
- **Output:** `{ "reply": "...", "used_llm": true }`

#### `GET /teams/{team_abbr}` (Backend Only)

**Description:** Retrieves branding assets (logos, colors) for a team.

- **Frequency:** **Backend Only** (Client logic uses internal mapping or static assets).
- **Status:** ✅ **Fixed**. Function calls and imports corrected.

#### `POST /predict/explain`

**Description:** Standalone explanation endpoint.

- **Frontend Caller:** `explainPrediction(payload)` (Added in v1.2)
- **Status:** ✅ **Fixed**. Client wrapper added.

---

## Summary

| Metric | Value |
| :--- | :--- |
| **Total Endpoints** | 8 |
| **Active (Synced)** | 7 |
| **Backend Only** | 1 |
| **Sync Status** | � 100% |
| **Date** | 2025-12-29 |

### 🛠️ Correction Notes

1. **Fixed**: `GET /teams/{team_abbr}` now correctly imports `normalize_abbr` and calls `get_team_asset`.
2. **Fixed**: Added `explainPrediction` to `client.js`.
3. **Fixed**: `POST /predict` handler patched (removed invalid `await` and defined missing `bundle`).
4. **Verified**: `client.js` functions map cleanly to backend endpoints.
5. **Fixed**: "LA" logo missing -> Aliased "LA" to "LAR" in frontend.
6. **Fixed**: Prediction failure for "LA" -> Added `normalize_abbr` to backend prediction logic.
