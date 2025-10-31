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
<<<<<<< HEAD

import React, {
  createContext, useContext, useMemo,
  useReducer, useCallback, useEffect
} from 'react';
import { toEntry } from './utils/predictionHelpers';

const KEY = "prediction_history";
const MAX_HISTORY = 100;

// Action types
const SET_CURRENT = 'SET_CURRENT';
const PUSH_HISTORY = 'PUSH_HISTORY';
const RESET_HISTORY = 'RESET_HISTORY';

const initialState = { current: null, history: [] };

function reducer(state, action) {
  switch (action.type) {
    case SET_CURRENT:
      return { ...state, current: action.payload };
    case PUSH_HISTORY:
      return { ...state, history: [action.payload, ...state.history].slice(0, MAX_HISTORY) };
    case RESET_HISTORY:
      return { ...state, history: [] };
    default:
      return state;
  }
}

// Safe hydration from localStorage
function loadHistory() {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

const Ctx = createContext(null);

export function PredictionProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState, (s) => ({
    ...s, history: loadHistory()
  }));

  // Actions
  const setCurrent = useCallback((e) => dispatch({ type: SET_CURRENT, payload: e }), []);
  const pushHistory = useCallback((e) => dispatch({ type: PUSH_HISTORY, payload: e }), []);
  const resetHistory = useCallback(() => dispatch({ type: RESET_HISTORY }), []);

  // Persist
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(state.history));
    } catch { }
  }, [state.history]);

  // Tiny dev logger
  useEffect(() => {
    if (typeof window !== "undefined" && import.meta && import.meta.env && import.meta.env.DEV) {
      console.debug("[PredictionContext] state:", state);
    }
  }, [state]);


  // Selectors
  const count = state.history.length;
  const latest = state.history[0] ?? null;

  const value = useMemo(() => ({
    setCurrent,
    pushHistory,
    resetHistory,
    count,
    latest,
    current: state.current
  }), [state]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const usePredictions = () => {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("usePredictions must be used within PredictionProvider");
  return ctx;
};
=======
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
>>>>>>> c6845983cfbfd1be9afb17b5b47b7331808ca550
