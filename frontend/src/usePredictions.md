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

## Best Practices

- Keep prediction state in the App layer to avoid prop drilling across routes.
- Use the provided handlers (setPrediction, pushHistory, setLoading, setError).
- Pass only the props needed by each component.
