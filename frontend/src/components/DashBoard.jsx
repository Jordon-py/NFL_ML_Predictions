/**
 * DashBoard.jsx
 * -------------
 * Purpose:
 *   Example container for TeamGrid + PredictionResult + HistoryChart.
 *   Reads and writes via PredictionContext actions.
 *
 * Notes:
 *   Adjust layout as your project requires. This file shows how "current" and
 *   "history" flow through the app using Context actions.
 */

import React from 'react';
import TeamGrid from './TeamGrid.jsx';
import PredictionResult from './PredictionResult.jsx';
import HistoryChart from './HistoryChart.jsx';
import { usePredictions } from '../PredictionContext.js';

export default function DashBoard() {
  const { state } = usePredictions();

  return (
    <main className="dashboard">
    <header>
      <div className="team-grid-header">
        <h2>Next Week's NFL Matchups</h2>
        <p>Click any matchup to see predicted scores</p>
      </div>
    </header>
      <section>
        <TeamGrid />
      </section>

      <section aria-live="polite">
        <PredictionResult entry={state.current} />
      </section>

      <section>
        <HistoryChart history={state.history} />
      </section>
    </main>
  );
}
