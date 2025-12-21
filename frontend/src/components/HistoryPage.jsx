<<<<<<< HEAD
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
import React from 'react';
import NavBar from './NavBar/NavBar.jsx';
import React from 'react'
import { useState, useEffect } from 'react';
import { usePredictions } from '../PredictionContext.jsx';
import HistoryChart from './HistoryChart.jsx';


export default function HistoryPage() {
  // 1. Get the full, validated prediction state (never null here)
  const predictionState = usePredictionStateSafe();

  // 2. Get a guaranteed array for history ( [] if missing/non-array )
  const history = usePredictionHistory();

  return (
    <>
      {/* NavBar can either:
          - read context itself, or
          - receive exactly the slice it needs */}
      <NavBar state={predictionState} />

      {/* HistoryChart now receives data that is already safe and normalized */}
      <HistoryChart state={predictionState} history={history} />
    </>
  );
}

=======
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
import React from 'react';
import NavBar from './NavBar/NavBar.jsx';
import HistoryChart from './HistoryChart.jsx';

import {
  usePredictionStateSafe,
  usePredictionHistory,
} from '../hooks/predictionSelectors.js';

export default function HistoryPage() {
  // 1. Get the full, validated prediction state (never null here)
  const predictionState = usePredictionStateSafe();

  // 2. Get a guaranteed array for history ( [] if missing/non-array )
  const history = usePredictionHistory();

  return (
    <>
      {/* NavBar can either:
          - read context itself, or
          - receive exactly the slice it needs */}
      <NavBar state={predictionState} />

      {/* HistoryChart now receives data that is already safe and normalized */}
      <HistoryChart state={predictionState} history={history} />
    </>
  );
}
>>>>>>> e830235acc489dc81436e2a91bd690360a3d1c62
