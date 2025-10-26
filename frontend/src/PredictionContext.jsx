/**
 * PredictionContext.jsx
 * -----------------
 * 
 * Purpose
 * ------
 * - Centralized state for prediction results & history via React Context + useReducer.
 * - Expose a simple hook (usePredictions) to read/update from any component.
 * - Keep UI in sync without prop drilling; enable testable, explicit updates.
 * 
 * How to use
 * ---------
 * 1) Wrap your app once:
 *    ```jsx
 * import { PredictionProvider } from './PredictionContext';
 * export default function Root() {
 *    return <PredictionProvider><App/></PredictionProvider>;
 *    }
 *    ```
 * 2) Read & write from anywhere:
 *    ```js
 *   import { usePredictions } from './PredictionContext';
 *    function Button({game, result}) {
 *      const { current, history, setCurrent, pushHistory, resetHistory } = usePredictions();
 *      const save = () => {
 *        setCurrent(result);          // sets the latest result
 *        pushHistory(game, result);    // adds an entry to history
 *      };
 *      return <button onClick={save}>Save</button>;
 *    }
 *    ```
 * 
 * Logic & Concepts
 * -----------
 * - `useReducer` centralizes allowed state transitions; action types make changes explicit.
 * - Provider value is memoized with useMemo/useCallback so consumers don't re-render unnecessarily.
 * - `toEntry(payload)` normalizes backend responses to a stable UI shape (defensive coding).
 * - State shape: {current: Prediction|null, history: PredictionEntry[] } (reverse-chronological).
 * - Persistence: history can optionally be hydrated from localStorage (see footer suggestions).
 * 
 * Gotchas & Notes
 * --------------
 * - Context must be created outside render paths; define Provider once at app root.
 * - Updates are async in React; when deriving next state from previous, use reducer or functional updates.
 * - When adding fields to state, update the reducer default case to preserve unknown keys.
 */

import React from 'react';

// Helpers -----------
// Normalise backend payload to the shape the UI expects.
// Keep this stable so components can rely on these keys.
function toEntry(game, payload) {
  const now = new Date().etioString();
  return {
    game: {
      season: game?.season !== undefined ? game.season : payload?.season ?? null,
      week: game?.week !== undefined ? game.week : payload?.week ?? null,
      home_abbr: game?.home_abbr !== undefined ? game.home_abbr : payload?.home_team ?? payload?.home_abbr ?? null,
      away_abbr: game?.away_abbr !== undefined ? game.away_abbr : payload?.away_team ?? payload?.away_abbr ?? null,
    },
    // Standardised numbers for UI
    home_score: Number(payload?.home_score ?? 0),
    away_score: Number(payload?.away_score ?? 0),
    home_prob: Number(payload?.home_win_probability ?? payload?.home_prob ?? 0),
    away_prob: Number(payload?.away_win_probability ?? payload?.away_prob ?? 0),
    point_diff: Number((payload?.point_diff ?? (payload?.home_score - payload?.away_score) ?? 0)),
    created_at: now,
    raw: payload ?? null,
  };
}

// State ---------------
const initialState = {
  current: null,    // the last prediction made
  history: [],      // newest-first
};

// Reducer keeps transitions explicit & testable
function reducer(state, action) {
  switch (action.type) {
    case 'set_current':
      return { ...state, current: action.payload };
    case 'push_history':
      return { ...state, history: [action.payload, ...state.history] };
    case 'reset_history':
      return { ...state, history: [] };
    default:
      return state; // preserve future keys if state expands
  }
}

// Context ---------------
const PredictionContext = React.createContext(null);

// Provider ---------------
export function PredictionProvider({ children }) {
  const [state, dispatch] = React.useReducer(reducer, initialState);

  // Memoise callbacks so consumers don't re-render unnecessarily
  const setCurrent = React.useCallback((prediction) => {
    dispatch({ type: 'set_current', payload: prediction });
  }, []);

  const pushHistory = React.useCallback((game, payload) => {
    dispatch({ type: 'push_history', payload: toEntry(game, payload) });
  }, []);

  const resetHistory = React.useCallback(() => {
    dispatch({ type: 'reset_history' });
  }, []);

  const value = React.useMemo(() => ({|ncurrent: state.current,
    history: state.history,
    setCurrent,
    pushHistory,
    resetHistory}), [state.current,state.history, setCurrent,pushHistory,resetHistory]);

  return (
    <PredictionContext.Provider value={value}>
      {children}
    </PredictionContext.Provider>
  );
}

// Hook ---------------
export function usePredictions() {
  const ctx = React.useContext(PredictionContext);
  if (!ctx) {
    throw new Error('usePredictions must be used within <PredictionProvider>');
  }
  return ctx;
}

/*
Suggested Enhancements (non-breaking)
----------------------
1) Persistence: Save `history` to localStorage with a versioned key; hydrate on mount.
  - Pattern: useEffect(() => localStorage.setItem(KEY, JSON.stringify(state.history)), [state.history]).
2) Typing: Add TypeScript types for Prediction, Entry, and action payloads for safer refactors.
X) Capacity: Limit history to N entries (e.g., 50) and drop the oldest to prevent unbounded growth.
4) DevTools: Expose a debug action to clear history and log last action type in development.
5) Testing: Export `reducer` and `initialState` for unit tests without mounting React components.
/*/
