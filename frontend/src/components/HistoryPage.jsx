/**
 * HistoryPage.jsx
 * ----------------
 * Purpose:
 *   Standalone route that renders the HistoryChart using data from
 *   the global PredictionContext. This allows the chart to be opened
 *   directly at /history without relying on the dashboard layout.
 *
 * Contract:
 *   - Reads prediction state via selector hooks.
 *   - Supplies a safe `history` array to <HistoryChart/>.
 *
 * Notes:
 *   - Chart render cost is roughly O(n) over `history.length`.
 *   - Page re-renders when `state.history` changes in context.
 */
import HistoryChart from './HistoryChart.jsx';
import NavBar from './NavBar/NavBar.jsx';

import { useEffect, useState } from 'react';
import { getPredictionHistory } from '../api/client.js';

export default function HistoryPage() {
  // lightweight client-backed history loader — avoids missing selector hooks
  const [history, setHistory] = useState([]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await getPredictionHistory(100);
        if (!mounted) return;
        setHistory(Array.isArray(res.entries) ? res.entries : res.entries ?? res ?? []);
      } catch (err) {
        setHistory([]);
      }
    })();
    return () => (mounted = false);
  }, []);

  // NavBar can display static info; HistoryChart reads `history` prop
  return (
    <>
      <NavBar />
      <HistoryChart history={history} />
    </>
  );
}
