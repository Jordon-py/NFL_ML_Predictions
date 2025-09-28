/**
 * PredictionContext
 * ------------------
 * Purpose:
 *   Centralize prediction-related state and actions without Redux.
 *   Matches the user's preference for React Context + hooks only.
 *
 * Shared State Shape:
 *   {
 *     current: {
 *       ts: ISOString,
 *       source: 'teamgrid' | 'form' | string,
 *       game: { season, week, home_abbr, away_abbr },
 *       metrics: { home_score, away_score, point_diff },
 *       probs: { home: number|null, away: number|null, ensemble: number|null }
 *     } | null,
 *     history: Array<same shape as current>
 *   }
 *
 * Why this helps (Layer 2: State Management):
 *   - Single source of truth for "current" and "history".
 *   - Eliminates duplicated local states across components.
 *   - History entries use one canonical schema consumed by charts/UI.
 *
 * Usage:
 *   <PredictionProvider>...children...</PredictionProvider>
 *   const { state, actions } = usePredictions();
 */

import React, { createContext, useContext, useMemo, useReducer, useCallback } from 'react';

const PredictionContext = createContext(null);

const initialState = {
  current: null,
  history: [],
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_CURRENT':
      return { ...state, current: action.payload };
    case 'PUSH_HISTORY':
      return { ...state, history: [action.payload, ...state.history] };
    case 'RESET_HISTORY':
      return { ...state, history: [] };
    default:
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
  home_win_probability = null,
  away_win_probability = null,
  ensemble_probability = null,
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

  // Actions are stable callbacks. Co-locate logic here to keep views dumb.
  const setCurrent = useCallback((entry) => dispatch({ type: 'SET_CURRENT', payload: entry }), []);
  const pushHistory = useCallback((entry) => dispatch({ type: 'PUSH_HISTORY', payload: entry }), []);
  const resetHistory = useCallback(() => dispatch({ type: 'RESET_HISTORY' }), []);

  const actions = useMemo(() => ({ setCurrent, pushHistory, resetHistory }), [setCurrent, pushHistory]);

  const value = useMemo(() => ({ state, actions }), [state, actions]);

  return <PredictionContext.Provider value={value}>{children}</PredictionContext.Provider>;

}

export function usePredictions() {
  const ctx = useContext(PredictionContext);
  if (!ctx) throw new Error('usePredictions must be used within a PredictionProvider');
  return ctx;
}
  /* @returns {Promise<{home_score: number, away_score: number, point_diff: number, home_win_probability: number, away_win_probability: number}>}
   * Fetch prediction data.
   * @returns {Promise<{home_score: number, away_score: number, point_diff: number, home_win_probability: number, away_win_probability: number}>}
   */