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

import React, { useEffect, useMemo, useState } from "react";
import { usePredictions } from "../../PredictionContext";
import TeamGrid from "../Card/TeamGrid.jsx";
import PredictionResult from "../PredictionResult.jsx";
import HistoryChart from "../HistoryChart.jsx";
import NavBar from "../NavBar/NavBar.jsx";
// @ts-ignore - CSS module import for JS/JSX file
import styles from "./Dashboard.module.css";
import { predictGame, getNextWeekSchedule } from "../../api/client.js";
import { useParams } from "react-router-dom";


const LS_KEY = "prediction_history";

/**
 * loadHistoryLocal
 * ----------------
 * Helper that safely retrieves any previously stored prediction history
 * from localStorage. This lets the dashboard still render "something"
 * if the context is empty (e.g. on a fresh reload before the backend responds).
 */
function loadHistoryLocal() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
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
  const [thisWeekSchedule, setThisWeekSchedule] = useState([]);
  const predictionState = usePredictions();
  const {
    current,
    history: ctxHistory,
    schedule,
    week,
    teams,
    predictions,
    loading,
    errors,
    makePrediction,
    health,
  } = predictionState;

  // CSS Modules mapping – keeps JSX readable while avoiding global class names.
  const {
    dashboard,
    header,
    teamGridHeader,
    title,
    subtitle,
    teamMain,
    historyChart,
  } = styles;

  // Effective history: use context first, fall back to localStorage
  const history = useMemo(() => {
    // Prefer the canonical history from context; fall back to any locally
    // persisted history if the context has not yet hydrated from the backend.
    if (Array.isArray(ctxHistory)) return ctxHistory;
    return loadHistoryLocal();
  }, [ctxHistory]);

  // The "latest" entry is the first element of history (assumes newest-first order).
  const latestFromHistory = history.length ? history[0] : null;

  /**
   * navState:
   * - current: the currently selected prediction (if any),
   *            defaulting to the latest history entry
   * - latest:  a mirror of the latest history entry for quick access
   * - count:   how many predictions we've stored (for small UI badges)
   * - health:  backend / model health summary for global status display
   *
   * useMemo is used purely to avoid re-creating this object on every render.
   */
  const navState = useMemo(
    () => ({
      current: current ?? latestFromHistory,
      latest: latestFromHistory,
      count: history.length,
      health,
    }),
    [current, latestFromHistory, history.length, health]
  );

  // Derived health strings keep JSX simple and play nicely with loose JS tooling types.
  const healthStatus = health?.status ?? "unknown";
  const healthReason = health?.reason ?? "initializing";

  useEffect(() => {
    const fetchNextWeekSchedule = async () => {
      try {
        const nextWeekSchedule = await getNextWeekSchedule();
        console.log("Next week schedule:", nextWeekSchedule);
        setThisWeekSchedule(nextWeekSchedule || []);
      } catch (error) {
        console.error("Failed to fetch next week schedule:", error);
        setThisWeekSchedule([]);
      }
    };

    fetchNextWeekSchedule();
  }, []);

  return (
    <>
      <NavBar state={navState} />

      <main className={dashboard}>
        <header className={header}>
          <div className={teamGridHeader}>
            <h2 className={title}>Next Week&apos;s NFL Matchups</h2>
            <p className={subtitle}>
              {healthStatus !== "healthy"
                ? `Models loading... predictions are temporarily disabled. Reason: ${healthReason}`
                : "Click any matchup to see predicted scores"}
            </p>
          </div>
        </header>

        <section className={teamMain}>
          {/* TeamGrid:
              - "schedule" comes from PredictionContext and is the canonical
                view of games to be predicted. */}
          <TeamGrid
            games={schedule}
            week={week}
            teams={teams}
            predictions={predictions}
            loading={loading}
            errors={errors}
            onPredict={() => predictGame(useParams)}
          />

          {/* HistoryChart visualizes model performance over past predictions. */}
          <div className={historyChart}>
            <HistoryChart history={history} state={predictionState} />
          </div>
        </section>

        {/* PredictionResult reads a single "entry" and renders it in a
            human-friendly layout. aria-live ensures screen readers are
            notified when the selected game changes. */}
        <section aria-live="polite">
          <PredictionResult entry={current ?? latestFromHistory} />
        </section>
      </main>
    </>
  );
};
