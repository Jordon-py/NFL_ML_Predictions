/**
 * DashBoard.jsx
<<<<<<< HEAD
 * -----------------------------------------------------------------------------
 * PURPOSE
 *   Compose the primary dashboard view: navigation, matchup grid, latest
 *   prediction panel, and historical trend chart.
 *
 * INPUTS
 *   - Reads shared prediction data from usePredictions() context.
 *     NOTE: Implementation currently exposes { current, latest, actions... } and
 *           not a raw { state } object; the docs mention a different shape.
 *           We handle both shapes defensively.
 *
 * OUTPUTS
 *   - Renders:
 *       <NavBar/>            : receives a minimal "state-like" object
 *       <TeamGrid/>          : interactive matchup cards (predict on click)
 *       <HistoryChart/>      : uses historical predictions
 *       <PredictionResult/>  : shows the latest/current prediction
 *
 * DEPENDENCIES
 *   - Prediction context/provider (see PredictionContext.jsx)
 *   - Children components listed above
 *   - localStorage "prediction_history" for hydration fallback
 *
 * USAGE
 *   Place inside a tree wrapped by <PredictionProvider>. The component is
 *   layout-focused; keep data transformations in the provider or dedicated
 *   utilities so children receive ready-to-render props.
 * -----------------------------------------------------------------------------
 */

import React, { useMemo } from 'react';
import { usePredictions } from '../PredictionContext.jsx';
=======
 * -------------
 * Component Purpose:
 *   Compose the primary dashboard layout: it renders the grid of matchups,
 *   the latest prediction, and historical trend in one place.
 *
 * Core Logic Overview:
 *   - Reads shared state from `usePredictions()` (context provider).
 *   - Delegates user interactions to child components; this component stays
 *     focused on layout and accessibility semantics.
 *
 * Modification Guide:
 *   - To inject new sections (e.g. filters, leaderboards), add `<section>`
 *     blocks so screen readers understand the layout.
 *   - Keep data transformations in the context/provider layer—children should
 *     receive ready-to-render props.
 */
import {usePredictions} from '../PredictionContext.jsx';

>>>>>>> c6845983cfbfd1be9afb17b5b47b7331808ca550
import TeamGrid from './TeamGrid.jsx';
import PredictionResult from './PredictionResult.jsx';
import HistoryChart from './HistoryChart.jsx';
import NavBar from './NavBar/NavBar.jsx';
<<<<<<< HEAD
import HamburgerMenu from './HamburgerMenu.jsx';

// LocalStorage key aligns with provider for safe hydration fallback.
const LS_KEY = 'prediction_history';

// Safe local history loader (guards malformed JSON / unavailable storage).
function loadHistoryLocal() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export default function DashBoard() {
  // Pull whatever the context actually exposes today.
  // Impl (PredictionContext.jsx) exposes: { current, latest, setCurrent, ... }.
  // Docs (usePredictions.md) suggest: { state, actions, selectors }.
  const ctx = usePredictions();

  // Resolve "current" in a backward/forward compatible way:
  // - Prefer ctx.current (implementation)
  // - Else ctx.state?.current (doc shape)
  // - Else ctx.latest (selector for newest history entry)
  const current = ctx.current ?? ctx?.state?.current ?? ctx?.latest ?? null;

  // Resolve "history":
  // - Prefer ctx.state?.history if present (doc shape)
  // - Else hydrate directly from localStorage (kept by provider/TeamGrid)
  const history = useMemo(
    () => ctx?.state?.history ?? loadHistoryLocal(),
    [ctx?.state?.history]
  );

  // Minimal "state-like" object for components that expect a combined shape.
  // Keeps coupling low while preserving existing prop contracts.
  const navState = useMemo(
    () => ({ current, history }),
    [current, history]
  );

  return (
    <>
      {/* Global nav; gets a compact state snapshot for badges/counters, etc. */}
      <NavBar state={navState} />

      <main className="dashboard">
        <header>
          <div className="team-grid-header">
            <h2 className="nfl-matchups">Next Week&apos;s NFL Matchups</h2>
            <p>Click any matchup to see predicted scores</p>
          </div>
        </header>

        {/* Main content: interactive grid + historical trend side-by-side */}
        <section className="team-main">
          {/* TeamGrid handles fetching schedule and issuing predictions */}
          <TeamGrid />
          {/* HistoryChart expects (history, latestPred). Provide both explicitly. */}
          <HistoryChart
            className="history-chart"
            history={history}
            latestPred={current}
          />
        </section>

        {/* Live region to politely announce latest result changes to AT users */}
        <section aria-live="polite">
          <PredictionResult entry={current} />
=======
import './TeamGrid.css';

export default function DashBoard() {
  // `state` exposes { current, history } for the entire app.
  const {state} = usePredictions();

  return (
    <>
      <NavBar state={state} />
      <main className="dashboard">
        <header>
          <div className="team-grid-header">
            <h2 className="nfl-matchups">Next Week's NFL Matchups</h2>
            <p>Click any matchup to see predicted scores</p>
          </div>
        </header>
        <section>
          <TeamGrid state={state} />
        </section>

        <section aria-live="polite">
          {/* Pass the current prediction entry; component handles the empty state. */}
          <PredictionResult entry={state.current} />
        </section>

        <section>
          {/* Historical predictions show trend data to the user. */}
          <HistoryChart history={state.history} />
>>>>>>> c6845983cfbfd1be9afb17b5b47b7331808ca550
        </section>
      </main>
    </>
  );
}
