/**
 * HistoryPage.jsx
 * ----------------
 * Purpose:
 *   Standalone route that renders the HistoryChart using data from
 *   the global PredictionContext. This allows the chart to be opened
 *   directly at /history without relying on the dashboard layout.
 *
 * Contract:
 *   - Reads `state` from usePredictions().
 *   - Passes `state` and a safe `history` array to <HistoryChart/>.
 *
 * Complexity notes:
 *   - Render cost is dominated by the chart, typically O(n) over history length.
 *   - Re-renders when context `state.history` changes.
 */
import { usePredictions } from '../PredictionContext.jsx';
import HistoryChart from './HistoryChart.jsx';
import NavBar from './NavBar/NavBar.jsx';

export default function HistoryPage() {
  const { state } = usePredictions();

  return (
    <>
      <NavBar />
        <HistoryChart
          state={state}
          history={Array.isArray(state?.history) ? state.history : []}
        />
    
    </>
  );
}