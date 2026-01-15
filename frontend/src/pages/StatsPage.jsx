// ==========================================
// File: frontend/src/pages/StatsPage.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ../components/NavBar/NavBar.jsx, ../components/HistoryChart.jsx, ./StatsPage.css
// Notes: Presentation-focused component.
// ==========================================

// File: frontend/src/pages/StatsPage.jsx
//
// Purpose:
//   A simple "Status + History" dashboard that reads everything from the backend.
//   Primary endpoints (client.js prefers these):
//   - /status/overview        (health + dataset + history metrics)
//   - /api/games/next-week    (upcoming games)
//   - /api/history?limit=N    (recent prediction entries; stable envelope)
//
//   Back-compat fallbacks (client.js will use these if needed):
//   - /schedule/next-week
//   - /history?limit=N
//
// Design goals:
//   - NO context hook or offline fallback.
//   - Keep it easy to reason about (one component, minimal derived state).
//   - Avoid runtime errors from CSS-module style variables by using string classNames.
//
// Depends on:
//   - ../api/client: getNextWeekSchedule, getPredictionHistory, getStatusOverview
//   - NavBar, HistoryChart
//
// Notes:
//   - client.js normalizes getPredictionHistory() into { entries, total, limit }
//     and getStatusOverview() into a safe shape.

import React, { useEffect, useMemo, useState } from "react";
import NavBar from "../components/NavBar/NavBar.jsx";
import HistoryChart from "../components/HistoryChart.jsx";
import {
  getNextWeekSchedule,
  getPredictionHistory,
  getStatusOverview,
} from "../api/client";
import "./StatsPage.css";

// Keep this small so the page stays fast, but large enough for trend charts.
const HISTORY_LIMIT = 500;

/** Build a stable composite key that works for BOTH schedule rows and history entries. */
function toGameKey(game) {
  const season = game?.season ?? "";
  const week = game?.week ?? "";
  const home = (game?.home_abbr || game?.home_team || "").toString().trim().toUpperCase();
  const away = (game?.away_abbr || game?.away_team || "").toString().trim().toUpperCase();

  // Example: "2025-15-KC-BUF"
  return [season, week, home, away].filter(Boolean).join("-");
}

/** Safe percent label for probabilities in [0, 1]. */
function toPercentLabel(prob) {
  const n = Number(prob);
  if (!Number.isFinite(n)) return "n/a";
  return `${Math.round(n * 100)}%`;
}

function LoadingSpinner({ label = "Loading" }) {
  return (
    <div className="stats-loading" role="status" aria-live="polite">
      <span className="stats-loading__spinner" aria-hidden="true" />
      <p className="stats-loading__label">{label}...</p>
    </div>
  );
}

/**
 * SummaryCard - small KPI card for quick metrics.
 *
 * intent: "ok" | "error" | "default"
 */
function SummaryCard({ title, value, subtext, intent = "default" }) {
  return (
    <article className={`summary-card summary-card--${intent}`}>
      <p className="summary-card__label">{title}</p>
      <strong className="summary-card__value">{value ?? "-"}</strong>
      {subtext ? <small className="summary-card__subtext">{subtext}</small> : null}
    </article>
  );
}

export default function StatsPage() {
  // Remote payloads (backend-only)
  const [schedule, setSchedule] = useState([]);
  const [historyPayload, setHistoryPayload] = useState({ entries: [], total: 0, limit: 0 });
  const [overview, setOverview] = useState(null);

  // Local UI state
  const [isLoading, setIsLoading] = useState(true);
  const [pageError, setPageError] = useState(null);

  /**
   * Load everything in parallel.
   * We use Promise.allSettled so ONE failing endpoint does not blank the whole page.
   */
  useEffect(() => {
    let active = true;

    async function hydrate() {
      setIsLoading(true);
      setPageError(null);

      const [scheduleRes, historyRes, overviewRes] = await Promise.allSettled([
        getNextWeekSchedule(),
        getPredictionHistory(HISTORY_LIMIT),
        getStatusOverview(),
      ]);

      if (!active) return;

      const scheduleData =
        scheduleRes.status === "fulfilled" && Array.isArray(scheduleRes.value)
          ? scheduleRes.value
          : [];
      if (scheduleRes.status === "rejected") {
        console.warn("[StatsPage] schedule fetch failed", scheduleRes.reason);
      }
      setSchedule(scheduleData);

      const historyData =
        historyRes.status === "fulfilled"
          ? historyRes.value || { entries: [], total: 0, limit: 0 }
          : { entries: [], total: 0, limit: 0 };
      if (historyRes.status === "rejected") {
        console.warn("[StatsPage] history fetch failed", historyRes.reason);
      }
      setHistoryPayload(historyData);

      const overviewData = overviewRes.status === "fulfilled" ? overviewRes.value || null : null;
      if (overviewRes.status === "rejected") {
        console.warn("[StatsPage] overview fetch failed", overviewRes.reason);
      }
      setOverview(overviewData);

      const failures = [
        scheduleRes.status === "rejected" ? "schedule" : null,
        historyRes.status === "rejected" ? "history" : null,
        overviewRes.status === "rejected" ? "overview" : null,
      ].filter(Boolean);

      if (failures.length === 3) {
        setPageError("Failed to load status data (schedule, history, overview). Backend may be offline.");
      } else if (failures.length > 0) {
        setPageError(`Some data failed to load: ${failures.join(", ")}.`);
      }
      setIsLoading(false);
    }

    hydrate();

    return () => {
      active = false; // guard against setState on unmounted component
    };
  }, []);

  // Normalize backend shapes into predictable view-model values.
  // client.js guarantees arrays/objects, so we can trust them more here.
  const history = useMemo(
    () => (Array.isArray(historyPayload?.entries) ? historyPayload.entries : []),
    [historyPayload]
  );
  const safeOverview = overview || {};

  const health = safeOverview.health || { status: "unknown" };
  const dataset = safeOverview.dataset || {};
  const historyMetrics = safeOverview.history?.metrics || {};

  const totalPredictions =
    Number.isFinite(Number(historyMetrics.total_predictions))
      ? Number(historyMetrics.total_predictions)
      : history.length;

  const winRateLabel =
    typeof historyMetrics.win_rate === "number"
      ? `${Math.round(historyMetrics.win_rate * 100)}%`
      : "n/a";

  const historyMap = useMemo(() => {
    const map = new Map();
    for (const entry of history) {
      if (!entry) continue;
      if (entry.game_id) map.set(entry.game_id, entry);
      const key = toGameKey(entry);
      if (key) map.set(key, entry);
    }
    return map;
  }, [history]);

  const scheduleRows = useMemo(() => (Array.isArray(schedule) ? schedule : []), [schedule]);
  const currentWeek = useMemo(() => {
    const weekValue = scheduleRows[0]?.week;
    return Number.isFinite(Number(weekValue)) ? Number(weekValue) : null;
  }, [scheduleRows]);

  const navState = useMemo(
    () => ({
      title: "Prediction Status",
      heroSubtitle: "Live backend health, dataset stats, and recorded predictions.",
      subtitle: `${totalPredictions} historical predictions stored`,
      weekLabel: currentWeek ? `Week ${currentWeek}` : "Week ?",
      healthLabel:
        health?.status === "healthy"
          ? "Backend: Healthy"
          : `Backend: ${health?.status ?? "unknown"}`,
    }),
    [currentWeek, health?.status, totalPredictions]
  );

  function renderSchedule() {
    if (isLoading) return <LoadingSpinner label="Loading status" />;
    if (pageError) return <div className="stats-error">{pageError}</div>;

    if (scheduleRows.length === 0) {
      return <p className="stats-empty">No future games detected in the schedule file.</p>;
    }

    return (
      <ul className="stats-schedule__list">
        {scheduleRows.map((game, idx) => {
          const idKey = game?.game_id ?? game?.id;
          const compositeKey = toGameKey(game);
          const prediction =
            (idKey && historyMap.get(idKey)) || (compositeKey && historyMap.get(compositeKey));

          const kickoffDate = game?.kickoff ? new Date(game.kickoff) : null;
          const kickoffLabel = kickoffDate ? kickoffDate.toLocaleString() : "TBD";
          const awayCode = (game?.away_abbr || game?.away_team || "").toString().trim();
          const homeCode = (game?.home_abbr || game?.home_team || "").toString().trim();
          const awayLogo = game?.away_logo;
          const homeLogo = game?.home_logo;
          const rowKey = idKey || compositeKey || `${idx}`;

          return (
            <li key={rowKey} className="stats-schedule__item">
              <div className="stats-schedule__game">
                <div className="stats-schedule__teams">
                  <span className="stats-schedule__team">
                    {awayLogo ? (
                      <img className="stats-schedule__logo" src={awayLogo} alt={`${awayCode} logo`} />
                    ) : null}
                    <span className="stats-schedule__abbr">{awayCode || "AWAY"}</span>
                  </span>
                  <span className="stats-schedule__at">@</span>
                  <span className="stats-schedule__team">
                    {homeLogo ? (
                      <img className="stats-schedule__logo" src={homeLogo} alt={`${homeCode} logo`} />
                    ) : null}
                    <span className="stats-schedule__abbr">{homeCode || "HOME"}</span>
                  </span>
                </div>
                <span className="stats-schedule__kickoff">{kickoffLabel}</span>
              </div>

              {prediction ? (
                <div className="stats-schedule__prediction">
                  <p>
                    <span>Home win:</span>
                    <strong>{toPercentLabel(prediction.home_win_probability)}</strong>
                  </p>
                  <p>
                    <span>Away win:</span>
                    <strong>{toPercentLabel(prediction.away_win_probability)}</strong>
                  </p>
                  <p className="stats-schedule__diff">
                    <span>Diff:</span>
                    <span>{prediction.point_diff?.toFixed?.(1) ?? prediction.point_diff ?? "n/a"} pts</span>
                  </p>
                </div>
              ) : (
                <p className="stats-schedule__pending">No prediction recorded yet.</p>
              )}
            </li>
          );
        })}
      </ul>
    );
  }

  return (
    <>
      <NavBar state={navState} />

      <div className="stats-page">
        <header className="stats-page__header">
          <h1 className="stats-page__title">Prediction Status Page</h1>
          <p className="stats-page__lead">
            Live backend health, dataset stats, and recorded predictions.
          </p>
        </header>

        <section className="stats-summary">
          <div className="stats-summary__grid">
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
              value={dataset?.rows ?? "-"}
              subtext={dataset?.path ? "Path loaded" : "Path unknown"}
            />
            <SummaryCard
              title="Predictions"
              value={totalPredictions}
              subtext={`Win rate: ${winRateLabel}`}
            />
          </div>
        </section>

        <section className="stats-section">
          <h2 className="stats-section__title">Next Week Schedule</h2>
          {renderSchedule()}
        </section>

        <section className="stats-section">
          <h2 className="stats-section__title">Historical Predictions</h2>
          <HistoryChart history={history} state={{ health }} />
        </section>
      </div>
    </>
  );
}
