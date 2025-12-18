# Prediction State (No Context)

Note: This file name is legacy; the hook is `usePredictionState`.

## Overview

The frontend now uses a simple state hook in `App.jsx` and passes data down via props.
No React Context is used for predictions.

## What It Does

- **State Access**: current prediction, history, schedule, teams, and health
- **Actions**: update prediction maps, history, and loading/error flags
- **Persistence**: localStorage sync for prediction history
- **Effects**: schedule load + health polling + team metadata fetch

## Syntax & Usage

```javascript
import { usePredictionState } from "./hooks/usePredictionState";

function App() {
  const {
    history,
    current,
    setPrediction,
    pushHistory,
  } = usePredictionState();

  // Pass the state and actions to pages/components via props.
  return <Dashboard history={history} current={current} />;
}
```

## Data Structure

```javascript
{
  current: PredictionEntry | null,
  history: PredictionEntry[],
  schedule: Array,
  teams: Record<string, { name: string, logoUrl: string }>,
  health: { status: string, mode: string, reason: string },
  predictions: Record<string, PredictionEntry>,
  loading: Record<string, boolean>,
  errors: Record<string, string | null>
}
```

<<<<<<< HEAD
## Best Practices

- Keep prediction state in the App layer to avoid prop drilling across routes.
- Use the provided handlers (setPrediction, pushHistory, setLoading, setError).
- Pass only the props needed by each component.
=======
### PredictionEntry Shape

```javascript
{
  ts: string,           // ISO timestamp
  source: string,       // 'teamgrid' or other
  game: {
    season: number,
    week: number,
    home_abbr: string,
    away_abbr: string
  },
  metrics: {
    home_score: number,
    away_score: number,
    point_diff: number
  },
  probs: {
    home: number,       // Home win probability (0-1)
    away: number,       // Away win probability (0-1)
    ensemble: number    // Combined probability
  }
}
```

## Interaction Diagram

```mermaid
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Components    │────│ usePredictions   │────│ PredictionContext│
│                 │    │                  │    │                 │
│ • TeamGrid      │    │ • state          │    │ • Reducer       │
│ • PredictionResult│  │ • actions        │    │ • Actions       │
│ • HistoryChart  │    │ • selectors      │    │ • Persistence   │
│ • NavBar        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │     localStorage       │
                    │ "prediction_history"   │
                    └─────────────────────────┘
```

## Key Interactions

1. **Components** → `usePredictions()` → **Context Provider**
   - Components call the hook to access shared state
   - Hook validates context availability (throws if not wrapped)

2. **Actions Flow**:
   - Component calls `actions.setCurrent(entry)`
   - Hook dispatches to reducer
   - Reducer updates state immutably
   - All consuming components re-render

3. **Persistence**:
   - State changes trigger `useEffect`
   - History array serialized to localStorage
   - Hydrated on app startup

4. **Selectors**:
   - Computed values (count, latest) derived from state
   - Automatically update when state changes

## Best Practices

- **Always wrap in PredictionProvider**: App must be wrapped at root level
- **Use actions for updates**: Never mutate state directly
- **Leverage selectors**: For derived/computed values
- **Handle loading states**: Check `state.current` for null before rendering
- **Error boundaries**: Wrap components that use predictions

## Common Patterns

```javascript
// Pattern 1: Check for current prediction
const { state } = usePredictions();
if (!state.current) return <div>No prediction yet</div>;

// Pattern 2: Add new prediction
const { actions } = usePredictions();
const newEntry = toEntry(predictionData);
actions.setCurrent(newEntry);
actions.pushHistory(newEntry);

// Pattern 3: Reset on logout
const { actions } = usePredictions();
actions.resetHistory();
```

## Educational Notes

- **Why Context + Reducer?** Provides predictable state updates without prop drilling
- **Immutability**: All updates create new state objects for React's reconciliation
- **Memoization**: Actions and selectors are memoized to prevent unnecessary re-renders
- **Hydration**: Safe localStorage loading prevents crashes on malformed data
- **Type Safety**: Runtime checks ensure proper usage within provider tree
>>>>>>> f3c92a29d (Complete NFL prediction system: dataset engineering with ELO/rolling/QB features, model training pipeline, UI fixes, and production-ready artifacts)
