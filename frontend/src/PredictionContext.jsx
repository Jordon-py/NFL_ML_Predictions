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
// /frontend/src/state/PredictionContext.jsx
import React, {
  createContext, useContext, useMemo,
  useReducer, useCallback, useEffect
} from 'react';

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
  try { const raw = localStorage.getItem(KEY); return Array.isArray(JSON.parse(raw)) ? JSON.parse(raw) : []; }
  catch { return []; }
}

export function toEntry({ source='teamgrid', season, week, home_abbr, away_abbr,
  home_score, away_score, point_diff, home_win_probability, away_win_probability, ensemble_probability }) {
  return {
    ts: new Date().toISOString(),
    source,
    game: { season, week, home_abbr, away_abbr },
    metrics: { home_score, away_score, point_diff },
    probs: {
      home: home_win_probability,
      away: away_win_probability,
      ensemble: ensemble_probability ?? home_win_probability
    }
  };
}

const Ctx = createContext(null);

export function PredictionProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState, (s) => ({
    ...s, history: loadHistory()
  }));

  // Actions
  const setCurrent   = useCallback((e) => dispatch({ type: SET_CURRENT,  payload: e }), []);
  const pushHistory  = useCallback((e) => dispatch({ type: PUSH_HISTORY, payload: e }), []);
  const resetHistory = useCallback(()  => dispatch({ type: RESET_HISTORY }), []);

  // Persist
  useEffect(() => { try {
    localStorage.setItem(KEY, JSON.stringify(state.history));
  } catch {} }, [state.history]);

  // Tiny dev logger
  useEffect(() => {
    if (import.meta.env?.DEV) console.debug("[PredictionContext] state:", state);
  }, [state]);

  // Selectors
  const count = state.history.length;
  const latest = state.history[0] ?? null;

  const value = useMemo(() => ({
    state, actions: { setCurrent, pushHistory, resetHistory },
    selectors: { count, latest }
  }), [state, setCurrent, pushHistory, resetHistory, count, latest]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const usePredictions = () => {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("usePredictions must be used within PredictionProvider");
  return ctx;
};
