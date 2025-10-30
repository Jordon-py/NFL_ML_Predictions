# usePredictions Hook

## Overview

`usePredictions` is a custom React hook that provides centralized access to the NFL prediction application's state management. It follows the Context + Reducer pattern for predictable, testable state updates across components.

## What It Does

- **State Access**: Provides current prediction and historical predictions
- **Actions**: Exposes methods to update state (set current, add to history, reset)
- **Selectors**: Computed values like history count and latest entry
- **Persistence**: Automatically syncs with localStorage
- **Type Safety**: Ensures usage within PredictionProvider context

## Syntax & Usage

```javascript
import { usePredictions } from '../PredictionContext.jsx';

function MyComponent() {
  const { state, actions, selectors } = usePredictions();

  // Access current state
  const currentPrediction = state.current;
  const predictionHistory = state.history;

  // Use actions
  const handleNewPrediction = (predictionData) => {
    actions.setCurrent(predictionData);
    actions.pushHistory(predictionData);
  };

  // Use selectors
  const totalPredictions = selectors.count;
  const mostRecent = selectors.latest;

  return (
    <div>
      <p>Total predictions: {totalPredictions}</p>
      {/* Component JSX */}
    </div>
  );
}
```

## Data Structure

### State Object

```javascript
{
  current: PredictionEntry | null,  // Latest prediction result
  history: PredictionEntry[]        // Array of past predictions (newest first)
}
```

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
