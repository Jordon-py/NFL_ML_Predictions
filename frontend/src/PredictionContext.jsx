/**
 * PredictionContext.jsx
 * --------------------
 * Component Purpose:
 *   Provide a shared prediction store (current result + historical list)
 *   using React Context + Reducer so all views stay in sync.
 *
 * Core Logic Overview:
 *   - `initialState` tracks the most recent prediction (`current`) and a
 *     reverse-chronological `history` array.
 *   - `reducer` responds to explicit action types so updates are predictable
 *     and testable.
 *   - Action creators (`setCurrent`, `pushHistory`, `resetHistory`) are
 *     memoised callbacks exposed through context consumers.
 *   - `toEntry` normalises any backend payload into the shape the UI expects.
 *
 * Modification Guide:
 *   - Add new action types inside the reducer, then expose a matching
 *     callback in the provider so components never call `dispatch` directly.
 *   - Extend `history` trimming/deduping here instead of inside components to
 *     keep presentation logic simple.
 *   - When adding fields to entries, update `toEntry` so downstream renderers
 *     see the new data in a predictable structure.
 */
// frontend/src/PredictionContext.js
import React, {
  createContext,
  useContext,
  useMemo,
  useReducer,
  useCallback,
} from 'react';

/**
 * PredictionContext
 * Purpose: Shared prediction store using React Context + Reducer.
 * State shape:
 *   {
 *     current: { ... } | null,
 *     history: Array<{ ... }>
 *   }
 */

const PredictionContext = createContext(null);

// Simple, destructurable state
const initialState = {
  current: null,
  history: [],
};

// Pure reducer (no mutations)
function reducer(state, action) {
  switch (action.type) {
    case 'SET_CURRENT':
      return { ...state, current: action.payload };
    case 'PUSH_HISTORY':
      return { ...state, history: [action.payload, ...state.history] };
    case 'RESET_HISTORY':
      return { ...state, history: [] };
    default:
      return state; // no-op for unknown actions
  }
}

/** Normalize any backend response into a UI-friendly entry */
export function toEntry({
  source = 'teamgrid',
  season,
  week,
  home_abbr,
  away_abbr,
  home_score,
  away_score,
  point_diff,
  home_win_probability,
  away_win_probability,
  ensemble_probability,
}) {
  return {
    ts: new Date().toISOString(),
    source,
    game: { season, week, home_abbr, away_abbr },
    metrics: {
      home_score,
      away_score,
      point_diff,
    },
    probs: {
      home: home_win_probability,
      away: away_win_probability,
      ensemble: ensemble_probability,
    },
  };
}

export function PredictionProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Stable action creators
  const setCurrent = useCallback(
    (entry) => dispatch({ type: 'SET_CURRENT', payload: entry }),
    []
  );
  const pushHistory = useCallback(
    (entry) => dispatch({ type: 'PUSH_HISTORY', payload: entry }),
    []
  );
  const resetHistory = useCallback(
    () => dispatch({ type: 'RESET_HISTORY' }),
    []
  );

  const actions = useMemo(() => ({ setCurrent, pushHistory, resetHistory }), [setCurrent, pushHistory, resetHistory]);
  
  const value = useMemo(() => ({ state, actions }), [state, actions]);

  // Use React.createElement to avoid JSX in .js files (fixes previous parse error)
  return React.createElement(
    PredictionContext.Provider,
    { value },
    children
  );
}

/** Primary hook */
export function usePredictions() {
  const ctx = useContext(PredictionContext);
  if (!ctx) {
    throw new Error('usePredictions must be used within a <PredictionProvider>');
  }
  return ctx;
}

/** Alias hook for callers importing singular form */
export function usePrediction() {
  return usePredictions();
}

// Optional direct context export if you need it elsewhere
export { PredictionContext };
