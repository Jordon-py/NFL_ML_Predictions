<<<<<<< HEAD
// File: frontend/src/pages/StatsPage.jsx
// Purpose: Status and history dashboard showing backend health, dataset stats, schedule, and recent predictions.
// Functions: toGameKey(28), LoadingSpinner(33), SummaryCard(53), StatsPage(63)
// Variables: historyMap(139), overviewData(165)
// Interacts With: api/client for health/status/history/schedule.
// StatsPage.jsx - Status + History dashboard
// -----------------------------------------
// Pulls real-time health, dataset, and prediction history metrics from the backend
// while still falling back to local context data when offline. Serves as the
// "status page" requested by stakeholders.

import { useEffect, useState } from "react";
=======
// File: frontend/src/pages/StatsPage.jsx
// Purpose: Status and history dashboard showing backend health, dataset stats, schedule, and recent predictions.
// Functions: toGameKey(28), LoadingSpinner(33), SummaryCard(53), StatsPage(63)
// Variables: historyMap(139), overviewData(165)
// Interacts With: api/client for health/status/history/schedule, PredictionContext for fallback state.
// StatsPage.jsx - Status + History dashboard
// -----------------------------------------
// Pulls real-time health, dataset, and prediction history metrics from the backend
// while still falling back to local context data when offline. Serves as the
// "status page" requested by stakeholders.

import React, { useState, useEffect, useMemo } from "react";
import { usePredictions } from "../PredictionContext";
import NavBar from "../components/NavBar/NavBar.jsx";
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
import HistoryChart from "../components/HistoryChart.jsx";
import {
  getNextWeekSchedule,
  getPredictionHistory,
  getStatusOverview,
} from "../api/client";
<<<<<<< HEAD
// @ts-ignore - CSS module import for JS/JSX file
import styles from "./StatsPage.module.css";

/**
 * Builds a stable "game key" from either a schedule row or a prediction entry.
 * We intentionally support both schedule objects and prediction objects here.
 */
const toGameKey = (game) =>
  [game?.season, game?.week, game?.home_abbr || game?.home_team, game?.away_abbr || game?.away_team]
    .filter(Boolean)
    .join("-");

function LoadingSpinner({ label = "Loading" }) {
  return (
    <div className={styles.loadingContainer} role="status" aria-live="polite">
      <span className={styles.loadingSpinner} aria-hidden="true" />
      <p>{label}...</p>
    </div>
  );
}

/**
 * SummaryCard - small KPI card used for health / dataset / history metrics.
 * `intent` is a semantic style hook ("ok" | "error" | "default").
 *
 * @param {{
 *   title: string;
 *   value: string | number | null | undefined;
 *   subtext?: string;
 *   intent?: "ok" | "error" | "default";
 * }} props
 */
function SummaryCard({ title, value, subtext, intent = "default" }) {
  return (
    <article className={`${styles.summaryCard} ${styles[intent] ?? ""}`}>
      <p className={styles.summaryLabel}>{title}</p>
      <strong className={styles.summaryValue}>{value ?? "-"}</strong>
      {subtext && <small className={styles.summarySubtext}>{subtext}</small>}
    </article>
  );
}

export default function StatsPage() {
  // Remote payloads from the backend
  const [schedule, setSchedule] = useState(/** @type {any[]} */([]));
  const [history, setHistory] = useState(/** @type {any[]} */([]));
  const [overview, setOverview] = useState(/** @type {any | null} */(null));
=======
// @ts-ignore - CSS module import for JS/JSX file
import "./StatsPage.css";

/**
 * Builds a stable "game key" from either a schedule row or a prediction entry.
 * We intentionally support both schedule objects and prediction objects here.
 */
const toGameKey = (game) =>
  [game?.season, game?.week, game?.home_abbr || game?.home_team, game?.away_abbr || game?.away_team]
    .filter(Boolean)
    .join("-");

function LoadingSpinner({ label = "Loading" }) {
  return (
    <div className={loadingContainer} role="status" aria-live="polite">
      <span className={loadingSpinner} aria-hidden="true" />
      <p>{label}...</p>
    </div>
  );
}

/**
 * SummaryCard - small KPI card used for health / dataset / history metrics.
 * `intent` is a semantic style hook ("ok" | "error" | "default").
 *
 * @param {{
 *   title: string;
 *   value: string | number | null | undefined;
 *   subtext?: string;
 *   intent?: "ok" | "error" | "default";
 * }} props
 */
function SummaryCard({ title, value, subtext, intent = "default" }) {
  return (
    <article className={`${summaryCard} ${styles[intent] ?? ""}`}>
      <p className={summaryLabel}>{title}</p>
      <strong className={summaryValue}>{value ?? "-"}</strong>
      {subtext && <small className={summarySubtext}>{subtext}</small>}
    </article>
  );
}


export default function StatsPage() {
  /** @type {any} */
  const predictionState = usePredictions();

  // Remote payloads from the backend
  const [schedule, setSchedule] = useState(
    /** @type {any[]} */([])
  );
  const [historyPayload, setHistoryPayload] = useState({ entries: [], total: 0 });
  const [overview, setOverview] = useState(null);
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75

  // Local UI state
  const [isPageLoading, setIsPageLoading] = useState(true);
  const [pageError, setPageError] = useState(/** @type {string | null} */(null));
<<<<<<< HEAD

  /**
   * Initial hydration:
   * - schedule: upcoming games (for "Next Week Schedule")
   * - history: last N prediction entries
   * - overview: health + dataset + history summary metrics
   */
  useEffect(() => {
    let active = true;

    const hydrate = async () => {
      try {
        setIsPageLoading(true);
        const [scheduleData, historyResponse, overviewData] = await Promise.all([
          getNextWeekSchedule(),
          getPredictionHistory(50),
          getStatusOverview(),
        ]);

        if (!active) return;

        setSchedule(Array.isArray(scheduleData) ? scheduleData : []);
        setHistory(Array.isArray(historyResponse?.entries) ? historyResponse.entries : []);
=======

  /**
   * Initial hydration:
   * - schedule: upcoming games (for "Next Week Schedule")
   * - history: last N prediction entries
   * - overview: health + dataset + history summary metrics
   */
  useEffect(() => {
    let active = true;

    const hydrate = async () => {
      try {
        setIsPageLoading(true);
        const [scheduleData, historyResponse, overviewData] = await Promise.all([
          getNextWeekSchedule(),
          getPredictionHistory(50),
          getStatusOverview(),
        ]);

        if (!active) return;

        setSchedule(Array.isArray(scheduleData) ? scheduleData : []);
        setHistoryPayload(historyResponse || { entries: [], total: 0 });
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
        setOverview(overviewData || null);
        setPageError(null);
      } catch (err) {
        if (!active) return;
<<<<<<< HEAD
        console.error("[StatsPage] loadPageData failed", err);
        setPageError("Failed to load status data. Backend may be offline.");
      } finally {
        if (active) setIsPageLoading(false);
      }
    };

    hydrate();
    return () => {
      // guard so we don't update state on an unmounted component
      active = false;
    };
  }, []);

  const historyMap = new Map();
  history.forEach((entry) => {
    if (!entry) return;
    if (entry.game_id) historyMap.set(entry.game_id, entry);
    const compositeKey = toGameKey(entry);
    if (compositeKey) historyMap.set(compositeKey, entry);
  });

  const overviewData = overview || {};
  const health = overviewData.health || { status: "unknown" };
  const datasetStatistics = overviewData.dataset || {};
  const historyMetrics = overviewData.history?.metrics || { total_predictions: history.length };
  /** @type {any[]} */
  const scheduleList = Array.isArray(schedule) ? schedule : [];

  const predictionWinRate = typeof historyMetrics?.win_rate === "number" ? `${Math.round(historyMetrics.win_rate * 100)}%` : "n/a";

  /**
   * Renders "Next Week Schedule" list:
   * - Each row shows matchup + kickoff time
   * - If we have a matching prediction, show win probabilities and margin
   */
  const renderScheduleList = () => {
    if (isPageLoading) return <LoadingSpinner label="Loading status" />;
    if (pageError) return <div className={styles.error}>{pageError}</div>;
    if (scheduleList.length === 0) {
      return (
        <p className={styles.empty}>
          No future games detected in the schedule file.
        </p>
      );
    }

    return (
      <ul className={styles.scheduleList}>
        {scheduleList.map((game) => {
          const idKey = game?.game_id ?? game?.id;
          const compositeKey = toGameKey(game);

          // Try to resolve the prediction by canonical ID first, then by composite key
          const prediction =
            (idKey && historyMap.get(idKey)) ||
            (compositeKey && historyMap.get(compositeKey));

          const kickoffDate = game?.kickoff ? new Date(game.kickoff) : null;
          const kickoffLabel = kickoffDate
            ? kickoffDate.toLocaleString()
            : "TBD";

          return (
            <li key={idKey || compositeKey} className={styles.scheduleItem}>
              <div className={styles.gameInfo}>
                <span>
                  {game.away_abbr || game.away_team} @{" "}
                  {game.home_abbr || game.home_team}
                </span>
                <span className={styles.kickoffTime}>{kickoffLabel}</span>
              </div>

              {prediction ? (
                <div className={styles.predictionDetails}>
                  <p>
                    Home win:{" "}
                    {Math.round(
                      (prediction.home_win_probability ?? 0) * 100
                    )}
                    %
                  </p>
                  <p>
                    Away win:{" "}
                    {Math.round(
                      (prediction.away_win_probability ?? 0) * 100
                    )}
                    %
                  </p>
                  <p className={styles.pointDiff}>
                    Diff:{" "}
                    {prediction.point_diff?.toFixed?.(1) ??
                      prediction.point_diff}{" "}
                    pts
                  </p>
                </div>
              ) : (
                <p className={styles.pendingNote}>
                  No prediction recorded yet.
                </p>
              )}
            </li>
          );
        })}
      </ul>
    );
=======
        console.error("[StatsPage] loadPageData failed", err);
        setPageError("Failed to load status data. Backend may be offline.");
      } finally {
        if (active) setIsPageLoading(false);
      }
    };

    hydrate();
    return () => {
      // guard so we don't update state on an unmounted component
      active = false;
    };
  }, []);

  /**
   * History source:
   * - Prefer backend-provided history payload.
   * - Fall back to context history if backend history is empty/offline.
   */
  const history = useMemo(() => {
    if (Array.isArray(historyPayload?.entries) && historyPayload.entries.length) {
      return historyPayload.entries;
    }
    return Array.isArray(predictionState?.history) ? predictionState.history : [];
  }, [historyPayload, predictionState?.history]);

  /**
   * historyMap:
   * We key predictions by BOTH:
   *  - canonical game_id (backend ID)
   *  - composite key (season-week-home-away) via toGameKey(entry)
   *
   * That way, schedule rows can be matched regardless of whether they carry
   * a game_id or just the team/season/week fields.
   */
  const historyMap = useMemo(() => {
    const map = new Map();

    history.forEach(
      /** @param {any} entry */
      (entry) => {
        if (!entry) return;

        // Primary key: stable ID from the backend, if present
        if (entry.game_id) {
          map.set(entry.game_id, entry);
        }

        // Fallback: composite key constructed from season/week/home/away
        const compositeKey = toGameKey(entry);
        if (compositeKey) {
          map.set(compositeKey, entry);
        }
      }
    );

    return map;
  }, [history]);

  // Prefer backend overview; fall back to context health if overview is missing.
  /** @type {any} */
  const overviewData = overview || {};
  // @ts-ignore - overviewData is a plain JSON-like object from the backend.
  const health = overviewData.health || predictionState?.health;
  const datasetStatistics = overviewData.dataset || {};
  const historyMetrics = overviewData.history?.metrics || {
    total_predictions: history.length,
  };
  /** @type {any[]} */
  const scheduleList = Array.isArray(schedule) ? schedule : [];

  const predictionHistoryEntries = history;
  const predictionWinRate =
    typeof historyMetrics?.win_rate === "number"
      ? `${Math.round(historyMetrics.win_rate * 100)}%`
      : "n/a";

  /**
   * Renders "Next Week Schedule" list:
   * - Each row shows matchup + kickoff time
   * - If we have a matching prediction, show win probabilities and margin
   */
  const renderScheduleList = () => {
    if (isPageLoading) return <LoadingSpinner label="Loading status" />;
    if (pageError) return <div className={error}>{pageError}</div>;
    if (scheduleList.length === 0) {
      return (
        <p className={empty}>
          No future games detected in the schedule file.
        </p>
      );
    }

    return (
      <ul className={scheduleList}>
        {scheduleList.map((game) => {
          const idKey = game?.game_id ?? game?.id;
          const compositeKey = toGameKey(game);

          // Try to resolve the prediction by canonical ID first, then by composite key
          const prediction =
            (idKey && historyMap.get(idKey)) ||
            (compositeKey && historyMap.get(compositeKey));

          const kickoffDate = game?.kickoff ? new Date(game.kickoff) : null;
          const kickoffLabel = kickoffDate
            ? kickoffDate.toLocaleString()
            : "TBD";

          return (
            <li key={idKey || compositeKey} className={scheduleItem}>
              <div className={gameInfo}>
                <span>
                  {game.away_abbr || game.away_team} @{" "}
                  {game.home_abbr || game.home_team}
                </span>
                <span className={kickoffTime}>{kickoffLabel}</span>
              </div>

              {prediction ? (
                <div className={predictionDetails}>
                  <p>
                    Home win:{" "}
                    {Math.round(
                      (prediction.home_win_probability ?? 0) * 100
                    )}
                    %
                  </p>
                  <p>
                    Away win:{" "}
                    {Math.round(
                      (prediction.away_win_probability ?? 0) * 100
                    )}
                    %
                  </p>
                  <p className={pointDiff}>
                    Diff:{" "}
                    {prediction.point_diff?.toFixed?.(1) ??
                      prediction.point_diff}{" "}
                    pts
                  </p>
                </div>
              ) : (
                <p className={pendingNote}>
                  No prediction recorded yet.
                </p>
              )}
            </li>
          );
        })}
      </ul>
    );
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
  };

  return (
    <>
<<<<<<< HEAD
      <div className={styles.statsPage}>
        <header className={styles.pageHeader}>
          <h1 className={styles.h1}>Prediction Status Page</h1>
          <p className={styles.pageLead}>
            Live backend health, dataset stats, and recorded predictions.
          </p>
        </header>

        <section className={styles.summaryGrid}>
          <SummaryCard
            title="Backend Health"
            value={health?.status ?? "unknown"}
            subtext={health?.reason}
            intent={
              health?.status === "healthy"
                ? "ok"
                : health?.status === "unhealthy"
                  ? "error"
                  : "default"
            }
          />
          <SummaryCard
            title="Dataset rows"
            value={datasetStatistics?.rows ?? "-"}
            subtext={datasetStatistics?.path ?? "path unknown"}
          />
          <SummaryCard
            title="Predictions recorded"
            value={historyMetrics?.total_predictions ?? history.length}
            subtext={`Win rate: ${predictionWinRate}`}
          />
        </section>

        <section className={styles.scheduleSection}>
          <h2 className={styles.h2}>Next Week Schedule</h2>
          {renderScheduleList()}
        </section>

        <section className={styles.historySection}>
          <h2 className={styles.h2}>Historical Predictions</h2>
          {/* HistoryChart still receives the raw history array plus context state */}
          <HistoryChart history={history} />
=======
      {/* Propagate latest health into the NavBar so the whole app reflects status */}
      {/* @ts-ignore - predictionState is a JS object from context; we allow spreading here. */}
      <NavBar state={{ ...predictionState, health }} />

      <div className={statsPage}>
        <header className={pageHeader}>
          <h1 className={h1}>Prediction Status Page</h1>
          <p className={pageLead}>
            Live backend health, dataset stats, and recorded predictions.
          </p>
        </header>

        <section className={summaryGrid}>
          <SummaryCard
            title="Backend Health"
            value={health?.status ?? "unknown"}
            subtext={health?.reason}
            intent={
              health?.status === "healthy"
                ? "ok"
                : health?.status === "unhealthy"
                  ? "error"
                  : "default"
            }
          />
          <SummaryCard
            title="Dataset rows"
            value={datasetStatistics?.rows ?? "-"}
            subtext={datasetStatistics?.path ?? "path unknown"}
          />
          <SummaryCard
            title="Predictions recorded"
            value={historyMetrics?.total_predictions ?? predictionHistoryEntries.length}
            subtext={`Win rate: ${predictionWinRate}`}
          />
        </section>

        <section className={scheduleSection}>
          <h2 className={h2}>Next Week Schedule</h2>
          {renderScheduleList()}
        </section>

        <section className={historySection}>
          <h2 className={h2}>Historical Predictions</h2>
          {/* HistoryChart still receives the raw history array plus context state */}
          <HistoryChart history={history} state={predictionState} />
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
        </section>
      </div>
    </>
  );
}
