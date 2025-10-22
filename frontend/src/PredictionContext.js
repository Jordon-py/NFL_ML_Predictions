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

import React, {createContext, useContext, useMemo, useReducer, useCallback} from 'react';

const PredictionContext = createContext(null);

// We keep a simple state shape so components can destructure easily.
const initialState = {
  current: null,
  history: [],
};

// Reducers should stay pure: given the previous state and an action we return
// the next state object without mutating the previous state.
function reducer(state, action) {
  switch (action.type) {
    case 'SET_CURRENT':
      return {...state, current: action.payload};
    case 'PUSH_HISTORY':
      return {...state, history: [action.payload, ...state.history]};
    case 'RESET_HISTORY':
      return {...state, history: []};
    default:
      // Returning the existing state ensures unknown actions are no-ops.
      return state;
  }
}

/** Helper to create a normalized entry from any backend response */
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
  ensemble_probability
}) {
  return {
    ts: new Date().toISOString(),
    source,
    game: {season, week, home_abbr, away_abbr},
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

export function PredictionProvider({children}) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Actions are stable callbacks. Co-locate logic here to keep views dumb.
  const setCurrent = useCallback((entry) => dispatch({type: 'SET_CURRENT', payload: entry}), []);
  const pushHistory = useCallback((entry) => dispatch({type: 'PUSH_HISTORY', payload: entry}), []);
  const resetHistory = useCallback(() => dispatch({type: 'RESET_HISTORY'}), []);

  // Expose a stable bundle of actions; include every dependency to avoid stale closures.
  const actions = useMemo(() => ({setCurrent, pushHistory, resetHistory}), [setCurrent, pushHistory, resetHistory]);

  // Memoise context value so consumers only re-render when state/actions change.
  const value = useMemo(() => ({state, actions}), [state, actions]);

  return <PredictionContext.Provider value={value}>{children}</PredictionContext.Provider>;

}

export function usePredictions() {
  const ctx = useContext(PredictionContext);
  if (!ctx) throw new Error('usePredictions must be used within a PredictionProvider');
  return ctx;
}
