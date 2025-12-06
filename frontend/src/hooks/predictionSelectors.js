// predictionSelectors.js
import { usePredictions } from '../PredictionContext.jsx';

/**
 * usePredictionStateSafe
 * Ensures you always get a non-null prediction state.
 */
export function usePredictionStateSafe() {
  const state = usePredictions();

  if (!state) {
    // This guard helps catch configuration bugs early
    throw new Error('usePredictionStateSafe must be used within a PredictionProvider');
  }

  return state;
}

/**
 * usePredictionHistory
 * Always returns a safe array for `history`.
 */
export function usePredictionHistory() {
  const state = usePredictionStateSafe();
  const { history } = state;

  return Array.isArray(history) ? history : [];
}
