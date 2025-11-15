/*
File: Dashboard.jsx
Purpose:
  Main dashboard container; integrates PredictionContext state, renders
  TeamGrid/HistoryChart/PredictionResult, and builds a small nav "summary"
  object for the NavBar.

Key ideas:
  - PredictionContext is the single source of truth for schedule + predictions.
  - localStorage is used only as a fallback history source if context is empty.
  - Layout and styling come from Dashboard.module.css via CSS Modules.
*/

import React, { useMemo } from "react";
import { usePredictions } from "../../PredictionContext";
import TeamGrid from "../Card/TeamGrid.jsx";
import PredictionResult from "../PredictionResult.jsx";
import HistoryChart from "../HistoryChart.jsx";
import NavBar from "../NavBar/NavBar.jsx";
// @ts-ignore -- CSS module is resolved by Vite; types are not required here.
import styles from "./Dashboard.module.css";

/**
 * Dashboard — layout-only container using CSS Modules.
 * Cleans up context shape handling and avoids inline styles.
 */
const PREDICTION_HISTORY_KEY = "prediction_history";

function loadPredictionHistoryFromLocalStorage() {
  try {
    if (typeof window === "undefined" || typeof window.localStorage === "undefined") {
      // Server-side render or non-browser environment: nothing to load.
      return [];
    }
    const rawHistoryData = window.localStorage.getItem(PREDICTION_HISTORY_KEY);
    if (!rawHistoryData) return [];
    const parsedHistory = JSON.parse(rawHistoryData);
    return Array.isArray(parsedHistory) ? parsedHistory : [];
  } catch {
    // Corrupt or missing data should never crash the UI.
    return [];
  }
}

/**
 * Dashboard component
 *
 * Responsibility:
 *  - Read state from PredictionContext (current prediction, schedule, etc.)
 *  - Provide a "nav state" snapshot to NavBar
 *  - Lay out TeamGrid + HistoryChart side-by-side
 *  - Render the currently selected prediction in PredictionResult
 */
export default function DashBoard() {
  /**
   * Raw prediction context state.
   *
   * Typed as `any` to keep JSX ergonomics while Pylance/TS remain active
   * in this JS project.
   */
  /** @type {any} */
  const predictionState = usePredictions() || {};
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
    health: backendHealth,
  } = predictionState;

  const historyList = useMemo(
    () =>
      Array.isArray(predictionHistory) && predictionHistory.length
        ? predictionHistory
        : loadPredictionHistoryFromLocalStorage(),
    [predictionHistory]
  );

  const mostRecentPrediction = historyList.length ? historyList[0] : null;
  const displayedPrediction = currentPrediction ?? mostRecentPrediction;

  const navigationBarState = useMemo(
    () => ({
      current: displayedPrediction,
      latest: mostRecentPrediction,
      count: historyList.length,
      health: backendHealth,
    }),
    [displayedPrediction, mostRecentPrediction, historyList.length, backendHealth]
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
          {/* TeamGrid:
              - "schedule" comes from PredictionContext and is the canonical
                view of games to be predicted. */}
          <TeamGrid
            games={upcomingGames}
            week={currentWeek}
            teams={teamMetadata}
            predictions={gamePredictions}
            loading={loadingStates}
            errors={errorStates}
            onPredict={handlePredictionRequest}
          />
          <div className={styles.historyChart}>
            <HistoryChart history={historyList} state={predictionState} />
          </div>
        </section>

        {/* PredictionResult reads a single "entry" and renders it in a
            human-friendly layout. aria-live ensures screen readers are
            notified when the selected game changes. */}
        <section aria-live="polite">
          <PredictionResult entry={displayedPrediction} />
        </section>
      </main>
    </>
  );
};
