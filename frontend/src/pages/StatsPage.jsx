// @ts-nocheck
/**
 * StatsPage.jsx — Status + History dashboard
 * -----------------------------------------
 * Pulls real-time health, dataset, and prediction history metrics from the backend
 * while still falling back to local context data when offline. Serves as the
 * "status page" requested by stakeholders.
 */
import React, { useState, useEffect, useMemo } from "react";
import { usePredictions } from "../PredictionContext";
import NavBar from "../components/NavBar/NavBar.jsx";
import HistoryChart from "../components/HistoryChart";
import { getNextWeekSchedule, getPredictionHistory, getStatusOverview } from "../api/client";
import styles from "./StatsPage.module.css";

const toGameKey = (game) => [game?.season, game?.week, game?.home_abbr || game?.home_team, game?.away_abbr || game?.away_team]
  .filter(Boolean)
  .join("-");

function LoadingSpinner({ label = "Loading" }) {
  return (
    <div className={styles.loadingContainer} role="status" aria-live="polite">
      <span className={styles.loadingSpinner} aria-hidden="true" />
      <p>{label}…</p>
    </div>
  );
}

function SummaryCard({ title, value, subtext, intent = "default" }) {
  return (
    <article className={`${styles.summaryCard} ${styles[intent] ?? ""}`}>
      <p className={styles.summaryLabel}>{title}</p>
      <strong className={styles.summaryValue}>{value ?? "—"}</strong>
      {subtext && <small className={styles.summarySubtext}>{subtext}</small>}
    </article>
  );
}

export default function StatsPage() {
  const predictionState = usePredictions();
  const [schedule, setSchedule] = useState([]);
  const [historyPayload, setHistoryPayload] = useState({ entries: [], total: 0 });
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    const hydrate = async () => {
      try {
        setLoading(true);
        const [scheduleData, historyData, overviewData] = await Promise.all([
          getNextWeekSchedule(),
          getPredictionHistory(50),
          getStatusOverview(),
        ]);
        if (!active) return;
        setSchedule(Array.isArray(scheduleData) ? scheduleData : []);
        setHistoryPayload(historyData || { entries: [], total: 0 });
        setOverview(overviewData || null);
        setError(null);
      } catch (err) {
        if (!active) return;
        console.error("[StatsPage] hydrate failed", err);
        setError("Failed to load status data. Backend may be offline.");
      } finally {
        if (active) setLoading(false);
      }
    };
    hydrate();
    return () => { active = false; };
  }, []);

  const history = useMemo(() => {
    if (Array.isArray(historyPayload?.entries) && historyPayload.entries.length) return historyPayload.entries;
    return Array.isArray(predictionState?.history) ? predictionState.history : [];
  }, [historyPayload, predictionState?.history]);

  const historyMap = useMemo(() => {
    const map = new Map();
    history.forEach((entry) => {
      if (entry?.game_id) map.set(entry.game_id, entry);
    });
    return map;
  }, [history]);

  const health = overview?.health || predictionState?.health;
  const datasetStats = overview?.dataset || {};
  const historyMetrics = overview?.history?.metrics || { total_predictions: history.length };
  const scheduleList = Array.isArray(schedule) ? schedule : [];

  const renderSchedule = () => {
    if (loading) return <LoadingSpinner label="Loading status" />;
    if (error) return <div className={styles.error}>{error}</div>;
    if (scheduleList.length === 0) return <p className={styles.empty}>No future games detected in the schedule file.</p>;

    return (
      <ul className={styles.scheduleList}>
        {scheduleList.map((game) => {
          const key = toGameKey(game);
          const prediction = historyMap.get(key);
          const kickoffDate = game?.kickoff ? new Date(game.kickoff) : null;
          const kickoffLabel = kickoffDate ? kickoffDate.toLocaleString() : "TBD";

          return (
            <li key={key} className={styles.scheduleItem}>
              <div className={styles.gameInfo}>
                <span>{game.away_abbr || game.away_team} @ {game.home_abbr || game.home_team}</span>
                <span className={styles.kickoffTime}>{kickoffLabel}</span>
              </div>
              {prediction ? (
                <div className={styles.predictionDetails}>
                  <p>Home win: {Math.round((prediction.home_win_probability ?? 0) * 100)}%</p>
                  <p>Away win: {Math.round((prediction.away_win_probability ?? 0) * 100)}%</p>
                  <p className={styles.pointDiff}>Diff: {prediction.point_diff?.toFixed?.(1) ?? prediction.point_diff} pts</p>
                </div>
              ) : (
                <p className={styles.pendingNote}>No prediction recorded yet.</p>
              )}
            </li>
          );
        })}
      </ul>
    );
  };

  const winRate = typeof historyMetrics?.win_rate === "number"
    ? `${Math.round(historyMetrics.win_rate * 100)}%`
    : "n/a";

  return (
    <>
      <NavBar state={{ ...predictionState, health }} />
      <div className={styles.statsPage}>
        <header className={styles.pageHeader}>
          <h1 className={styles.h1}>Prediction Status Page</h1>
          <p className={styles.pageLead}>Live backend health, dataset stats, and recorded predictions.</p>
        </header>

        <section className={styles.summaryGrid}>
          <SummaryCard
            title="Backend Health"
            value={health?.status ?? "unknown"}
            subtext={health?.reason}
            intent={health?.status === "healthy" ? "ok" : health?.status === "unhealthy" ? "error" : "default"}
          />
          <SummaryCard
            title="Dataset rows"
            value={datasetStats?.rows ?? "—"}
            subtext={datasetStats?.path ?? "path unknown"}
          />
          <SummaryCard
            title="Predictions recorded"
            value={historyMetrics?.total_predictions ?? history.length}
            subtext={`Win rate: ${winRate}`}
          />
        </section>

        <section className={styles.scheduleSection}>
          <h2 className={styles.h2}>Next Week Schedule</h2>
          {renderSchedule()}
        </section>

        <section className={styles.historySection}>
          <h2 className={styles.h2}>Historical Predictions</h2>
          <HistoryChart history={history} state={predictionState} />
        </section>
      </div>
    </>
  );
}
