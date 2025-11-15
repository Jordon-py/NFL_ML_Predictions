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
import styles from "./Dashboard.module.css";

/**
 * Dashboard — layout-only container using CSS Modules.
 * Cleans up context shape handling and avoids inline styles.
 */
const PREDICTION_HISTORY_KEY = "prediction_history";

function loadPredictionHistoryFromLocalStorage() {
  try {
    const rawHistoryData = localStorage.getItem(PREDICTION_HISTORY_KEY);
    const parsedHistory = JSON.parse(rawHistoryData);
    return Array.isArray(parsedHistory) ? parsedHistory : [];
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
    current: currentPrediction,
    history: predictionHistory = [],
    schedule: upcomingGames,
    week: currentWeek,
    teams: teamMetadata,
    predictions: gamePredictions,
    loading: loadingStates,
    errors: errorStates,
    makePrediction: handlePredictionRequest,
    health: backendHealth
  } = usePredictions() || {};

  const mostRecentPrediction = predictionHistory.length ? predictionHistory[0] : null;
  const displayedPrediction = currentPrediction ?? mostRecentPrediction;

  const navigationBarState = useMemo(
    () => ({ 
      current: displayedPrediction, 
      latest: mostRecentPrediction, 
      count: predictionHistory.length, 
      health: backendHealth 
    }),
    [displayedPrediction, mostRecentPrediction, predictionHistory.length, backendHealth]
  );
  const isBackendHealthy = backendHealth?.status === 'healthy';
  const healthMessage = isBackendHealthy
    ? "Click any matchup to see predicted scores"
    : `Models loading... predictions are temporarily disabled. Reason: ${backendHealth?.reason || 'initializing'}`;

  return (
    <>
      <NavBar state={navigationBarState} />

      <main className={styles.dashboard}>
        <header className={styles.header}>
          <div className={styles.teamGridHeader}>
            <h2 className={styles.title}>Next Week&apos;s NFL Matchups</h2>
            <p className={styles.subtitle}>{healthMessage}</p>
          </div>
        </header>

        <section className={styles.teamMain}>
          <TeamGrid
            games={upcomingGames}
            week={currentWeek}
            teams={teamMetadata}
            predictions={gamePredictions}
            loading={loadingStates}
            errors={errorStates}
            onPredict={handlePredictionRequest}
          />
          <HistoryChart 
            className={styles.historyChart} 
            history={predictionHistory} 
            latestPred={displayedPrediction} 
          />
        </section>

        <section aria-live="polite">
          <PredictionResult entry={displayedPrediction} />
        </section>
      </main>
    </>
  );
}

