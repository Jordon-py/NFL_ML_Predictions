/*
File: DashBoard.jsx
Purpose: Main dashboard container; integrates PredictionContext state, renders TeamGrid/HistoryChart/PredictionResult, manages prediction history from context or localStorage.
Functions: DashBoard (React component), loadHistoryLocal (localStorage helper)
Variables: LS_KEY (localStorage key), history (prediction history array), navState (for NavBar)
Interacts With: PredictionContext (state + actions), TeamGrid/HistoryChart/PredictionResult/NavBar (child components), DashBoard.module.css (CSS Modules)
*/
import React, { useMemo, useEffect } from "react";
import { usePredictions } from '../../PredictionContext';
import TeamGrid from "../Card/TeamGrid.jsx";
import PredictionResult from "../PredictionResult.jsx";
import HistoryChart from "../HistoryChart.jsx";
import NavBar from "../NavBar/NavBar.jsx";
import styles from "./DashBoard.module.css";

/**
 * Dashboard — layout-only container using CSS Modules.
 * Cleans up context shape handling and avoids inline styles.
 */
const LS_KEY = "prediction_history";

function loadHistoryLocal() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * DashBoard component
 * 
 * Purpose:
 *   - Serves as the main dashboard container for the NFL prediction frontend.
 *   - Integrates context state from PredictionContext, displays the current and historical predictions,
 *     and renders the TeamGrid, HistoryChart, and PredictionResult components.
 *   - Fetches and logs the NFL schedule CSV for potential future use.
 * 
 * Key Logic Flow:
 *   - Loads prediction history from context or localStorage.
 *   - Computes navigation state for NavBar.
 *   - Renders the main dashboard layout using CSS Modules for styling.
 * 
 * Dependencies:
 *   - PredictionContext (for state and actions)
 *   - TeamGrid, PredictionResult, HistoryChart, NavBar components
 *   - Dashboard.module.css for layout and style
 */
export default function DashBoard() {
  const {
    current,
    history = [],
    schedule,
    week,
    teams,
    predictions,
    loading,
    errors,
    makePrediction,
    health
  } = usePredictions() || {};

  const latestFromHistory = history.length ? history[0] : null;

  const navState = useMemo(
    () => ({ current: current ?? latestFromHistory, latest: latestFromHistory, count: history.length, health }),
    [current, latestFromHistory, history.length, health]
  );
  // Removed legacy CSV fetch — schedule now loaded centrally in PredictionContext.
  // If we later need raw CSV for analytics, add a dedicated hook instead of inline effect.
  return (
    <>
      <NavBar state={navState} />

      <main className={styles.dashboard}>
        <header className={styles.header}>
          <div className={styles.teamGridHeader}>
            <h2 className={styles.title}>Next Week&apos;s NFL Matchups</h2>
            <p className={styles.subtitle}>
              {health?.status !== 'healthy'
                ? `Models loading... predictions are temporarily disabled. Reason: ${health?.reason || 'initializing'}`
                : "Click any matchup to see predicted scores"
              }
            </p>
          </div>
        </header>

        <section className={styles.teamMain}>
          <TeamGrid
            games={schedule}
            week={week}
            teams={teams}
            predictions={predictions}
            loading={loading}
            errors={errors}
            onPredict={makePrediction}
          />
          <HistoryChart className={styles.historyChart} history={history} latestPred={current ?? latestFromHistory} />
        </section>

        <section aria-live="polite">
          <PredictionResult entry={current ?? latestFromHistory} />
        </section>
      </main>
    </>
  );
}

