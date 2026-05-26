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
import NavBar from "../components/NavBar/NavBar.jsx";
import {
  getHistorySummary,
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

function formatInteger(value) {
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "-";
}

function formatKickoff(value) {
  if (!value) return "TBD";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "TBD";
  return date.toLocaleString([], {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function healthIntent(status) {
  if (status === "healthy") return "ok";
  if (status === "unhealthy") return "error";
  return "default";
}

export default function StatsPage({ authSession = null, onSignOut }) {
  // Remote payloads from the backend
  const [schedule, setSchedule] = useState(/** @type {any[]} */([]));
  const [history, setHistory] = useState(/** @type {any[]} */([]));
  const [overview, setOverview] = useState(/** @type {any | null} */(null));
  const [historySummary, setHistorySummary] = useState(/** @type {any | null} */(null));

  // Local UI state
  const [isPageLoading, setIsPageLoading] = useState(true);
  const [pageError, setPageError] = useState(/** @type {string | null} */(null));
  const userId = authSession?.userId || null;

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
        const [scheduleData, historyResponse, overviewData, summaryData] = await Promise.all([
          getNextWeekSchedule(),
          getPredictionHistory(50, userId),
          getStatusOverview(userId),
          getHistorySummary(userId),
        ]);

        if (!active) return;

        setSchedule(Array.isArray(scheduleData) ? scheduleData : []);
        setHistory(Array.isArray(historyResponse?.entries) ? historyResponse.entries : []);
        setOverview(overviewData || null);
        setHistorySummary(summaryData || null);
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
  }, [userId]);

  const historyMap = buildHistoryLookup(history);

  const overviewData = overview || {};
  const health = overviewData.health || { status: "unknown" };
  const datasetStatistics = overviewData.dataset || {};
  const historyMetrics = historySummary || overviewData.history?.metrics || { total_predictions: history.length };
  /** @type {any[]} */
  const scheduleList = Array.isArray(schedule) ? schedule : [];
  const healthStatus = health?.status ?? "unknown";
  const healthTone = healthIntent(healthStatus);
  const statusBadgeClass =
    healthStatus === "healthy"
      ? styles.statusOk
      : healthStatus === "unhealthy"
        ? styles.statusError
        : styles.statusDefault;
  const serviceReady = healthStatus === "healthy";
  const firstKickoff = scheduleList[0]?.kickoff || null;
  const latestPrediction = history[0]?.ts || history[0]?.timestamp || null;

  const predictionWinRate = typeof historyMetrics?.win_rate === "number" ? `${Math.round(historyMetrics.win_rate * 100)}%` : "n/a";
  const spreadError =
    typeof historyMetrics?.avg_abs_spread_error === "number"
      ? `${historyMetrics.avg_abs_spread_error.toFixed(1)} pts`
      : "n/a";
  const overviewMessage = pageError
    ? "Status data could not fully load. Retry from the dashboard after backend health returns."
    : serviceReady
      ? "Backend models, schedule data, and history metrics are available for review."
      : "The overview can still show cached or partial data, but new forecasts may be blocked.";

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

          const kickoffLabel = formatKickoff(game?.kickoff);

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
      <NavBar
        authSession={authSession}
        onSignOut={onSignOut}
        state={{
          health,
          title: "Service Overview",
          heroSubtitle: "Backend readiness, dataset coverage, and forecast history in one review surface.",
          subtitle: `${scheduleList.length} upcoming game${scheduleList.length === 1 ? "" : "s"} tracked`,
          healthLabel: serviceReady ? "Service: Live" : `Service: ${healthStatus}`,
        }}
      />

      <main className={styles.statsPage} aria-label="NFL prediction service overview">
        <header className={styles.pageHeader}>
          <div className={styles.pageHeaderCopy}>
            <p className={styles.pageEyebrow}>Operations overview</p>
            <h1 className={styles.h1}>Prediction readiness, schedule context, and model history.</h1>
            <p className={styles.pageLead}>{overviewMessage}</p>
          </div>

          <aside className={styles.pageStatusPanel} aria-label="Current service state">
            <span className={`${styles.statusBadge} ${statusBadgeClass}`}>
              {serviceReady ? "Ready" : healthStatus}
            </span>
            <strong>{serviceReady ? "Forecasting available" : "Check backend readiness"}</strong>
            <span>{health?.reason || "No backend blocker reported."}</span>
          </aside>
        </header>

        <section className={styles.summaryGrid} aria-label="Overview summary">
          <SummaryCard
            title="Backend Health"
            value={healthStatus}
            subtext={health?.reason || "Health endpoint reachable when service is live"}
            intent={healthTone}
          />
          <SummaryCard
            title="Dataset rows"
            value={formatInteger(datasetStatistics?.rows)}
            subtext={datasetStatistics?.path ?? "dataset path unknown"}
          />
          <SummaryCard
            title="Upcoming games"
            value={scheduleList.length}
            subtext={`First kickoff: ${formatKickoff(firstKickoff)}`}
          />
          <SummaryCard
            title="Prediction record"
            value={formatInteger(historyMetrics?.total_predictions ?? history.length)}
            subtext={`Win rate: ${predictionWinRate} | spread error: ${spreadError}`}
          />
        </section>

        <section className={styles.statusNarrative} aria-label="Status notes">
          <span>Latest prediction: {latestPrediction ? formatKickoff(latestPrediction) : "none yet"}</span>
          <span>Resolved games: {formatInteger(historyMetrics?.resolved_games ?? 0)}</span>
          <span>History scope: {history.length} loaded entries</span>
        </section>

        <div className={styles.contentGrid}>
          <section className={styles.scheduleSection}>
            <div className={styles.sectionHeader}>
              <div>
                <p className={styles.pageEyebrow}>Next slate</p>
                <h2 className={styles.h2}>Schedule and prediction coverage</h2>
              </div>
              <span>{scheduleList.length} games</span>
            </div>
            {renderScheduleList()}
          </section>

          <section className={styles.historySection}>
            <div className={styles.sectionHeader}>
              <div>
                <p className={styles.pageEyebrow}>Model feedback</p>
                <h2 className={styles.h2}>Historical predictions</h2>
              </div>
              <span>{predictionWinRate} win rate</span>
            </div>
            <HistoryChart history={history} summary={historyMetrics} />
          </section>
        </div>
      </main>
    </>
  );
}
