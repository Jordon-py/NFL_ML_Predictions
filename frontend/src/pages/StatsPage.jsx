// File: frontend/src/pages/StatsPage.jsx
// Purpose: Status and history dashboard showing backend health, dataset stats, schedule, and recent predictions.
// Key helpers: buildHistoryLookup(), formatWinProbability(), renderScheduleList()
// Interacts With: api/client for health/status/history/schedule.
// StatsPage.jsx - Status + History dashboard
// -----------------------------------------
// Pulls real-time health, dataset, and prediction history metrics from the backend
// while still falling back to local context data when offline. Serves as the
// "status page" requested by stakeholders.

import { useEffect, useState } from "react";
import HistoryChart from "../components/HistoryChart.jsx";
import {
  getNextWeekSchedule,
  getPredictionHistory,
  getStatusOverview,
} from "../api/client";
import { buildMatchupKey } from "../utils/gameUtils.js";
// @ts-ignore - CSS module import for JS/JSX file
import styles from "./StatsPage.module.css";

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

/**
 * Build a lookup table that can resolve predictions by either canonical game_id
 * or by the fallback season-week-home-away composite key.
 */
function buildHistoryLookup(entries) {
  const lookup = new Map();

  for (const entry of Array.isArray(entries) ? entries : []) {
    if (!entry) continue;
    if (entry.game_id) lookup.set(entry.game_id, entry);

    const compositeKey = buildMatchupKey(entry);
    if (compositeKey) lookup.set(compositeKey, entry);
  }

  return lookup;
}

function formatWinProbability(probability) {
  return typeof probability === "number" ? `${Math.round(probability * 100)}%` : "n/a";
}

export default function StatsPage() {
  // Remote payloads from the backend
  const [schedule, setSchedule] = useState(/** @type {any[]} */([]));
  const [history, setHistory] = useState(/** @type {any[]} */([]));
  const [overview, setOverview] = useState(/** @type {any | null} */(null));

  // Local UI state
  const [isPageLoading, setIsPageLoading] = useState(true);
  const [pageError, setPageError] = useState(/** @type {string | null} */(null));

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
        setOverview(overviewData || null);
        setPageError(null);
      } catch (err) {
        if (!active) return;
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

  const historyMap = buildHistoryLookup(history);

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
            const compositeKey = buildMatchupKey(game);

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
                  <p>Home win: {formatWinProbability(prediction.home_win_probability)}</p>
                  <p>Away win: {formatWinProbability(prediction.away_win_probability)}</p>
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
  };

  return (
    <>
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
        </section>
      </div>
    </>
  );
}
