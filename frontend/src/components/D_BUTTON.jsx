/**
 * D_BUTTON.jsx — Clear History Button
 * -----------------------------------
 * Purpose:
 *   Provide a single, accessible control to clear the prediction history.
 *   Uses the context action (resetHistory) and also syncs localStorage.
 *
 * Contract:
 *   - No props required.
 *   - When clicked, clears history and announces the action to screen readers.
 *
 * Notes:
 *   - The PredictionProvider persists history under KEY = "prediction_history".
 *   - We call resetHistory() and then proactively write [] to localStorage to
 *     ensure immediate UI consistency for components reading from storage.
 */
import { useCallback } from 'react';
import { usePredictions } from '../PredictionContext.jsx';

const KEY = 'prediction_history';

export default function D_BUTTON() {
  const { resetHistory, count } = usePredictions();

  const onClear = useCallback(() => {
    try {
      resetHistory();
      // Force storage sync for consumers that hydrate from localStorage
      localStorage.setItem(KEY, JSON.stringify([]));
      // Optional console note in dev
      if (import.meta?.env?.DEV) console.info('[D_BUTTON] Cleared prediction history');
    } catch (err) {
      console.error('Failed to clear history:', err);
    }
  }, [resetHistory]);

  return (
    <button
      type="button"
      className="clear-history-button"
      onClick={onClear}
      aria-label="Clear prediction history"
      title="Clear prediction history"
      disabled={count === 0}
    >
      Clear History
    </button>
  );
}
