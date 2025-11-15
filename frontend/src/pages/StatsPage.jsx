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

const generateGameKey = (game) => [game?.season, game?.week, game?.home_abbr || game?.home_team, game?.away_abbr || game?.away_team]
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
  const [upcomingSchedule, setUpcomingSchedule] = useState([]);
  const [historyData, setHistoryData] = useState({ entries: [], total: 0 });
  const [statusOverview, setStatusOverview] = useState(null);
  const [isPageLoading, setIsPageLoading] = useState(true);
  const [pageError, setPageError] = useState(null);

  useEffect(() => {
    let isComponentMounted = true;
    
    const loadPageData = async () => {
      try {
        setIsPageLoading(true);
        const [scheduleData, historyResponse, overviewData] = await Promise.all([
          getNextWeekSchedule(),
          getPredictionHistory(50),
          getStatusOverview(),
        ]);
        if (!isComponentMounted) return;
        
        setUpcomingSchedule(Array.isArray(scheduleData) ? scheduleData : []);
        setHistoryData(historyResponse || { entries: [], total: 0 });
        setStatusOverview(overviewData || null);
        setPageError(null);
      } catch (err) {
        if (!isComponentMounted) return;
        console.error("[StatsPage] loadPageData failed", err);
        setPageError("Failed to load status data. Backend may be offline.");
      } finally {
        if (isComponentMounted) setIsPageLoading(false);
      }
    };
    
    loadPageData();
    return () => { isComponentMounted = false; };
  }, []);

  const predictionHistoryEntries = useMemo(() => {
    if (Array.isArray(historyData?.entries) && historyData.entries.length) return historyData.entries;
    return Array.isArray(predictionState?.history) ? predictionState.history : [];
  }, [historyData, predictionState?.history]);

  const predictionsByGameKey = useMemo(() => {
    const keyToPredictionMap = new Map();
    predictionHistoryEntries.forEach((entry) => {
      if (entry?.game_id) keyToPredictionMap.set(entry.game_id, entry);
    });
    return keyToPredictionMap;
  }, [predictionHistoryEntries]);

  const backendHealth = statusOverview?.health || predictionState?.health;
  const datasetStatistics = statusOverview?.dataset || {};
  const historyMetrics = statusOverview?.history?.metrics || { total_predictions: predictionHistoryEntries.length };
  const scheduleGames = Array.isArray(upcomingSchedule) ? upcomingSchedule : [];

  const renderScheduleList = () => {
    if (isPageLoading) return <LoadingSpinner label="Loading status" />;
    if (pageError) return <div className={styles.error}>{pageError}</div>;
    if (scheduleGames.length === 0) return <p className={styles.empty}>No future games detected in the schedule file.</p>;

    return (
      <ul className={styles.scheduleList}>
        {scheduleGames.map((game) => {
          const gameKey = generateGameKey(game);
          const gamePrediction = predictionsByGameKey.get(gameKey);
          const kickoffDate = game?.kickoff ? new Date(game.kickoff) : null;
          const kickoffDisplayText = kickoffDate ? kickoffDate.toLocaleString() : "TBD";

          return (
            <li key={gameKey} className={styles.scheduleItem}>
              <div className={styles.gameInfo}>
                <span>{game.away_abbr || game.away_team} @ {game.home_abbr || game.home_team}</span>
                <span className={styles.kickoffTime}>{kickoffDisplayText}</span>
              </div>
              {gamePrediction ? (
                <div className={styles.predictionDetails}>
                  <p>Home win: {Math.round((gamePrediction.home_win_probability ?? 0) * 100)}%</p>
                  <p>Away win: {Math.round((gamePrediction.away_win_probability ?? 0) * 100)}%</p>
                  <p className={styles.pointDiff}>Diff: {gamePrediction.point_diff?.toFixed?.(1) ?? gamePrediction.point_diff} pts</p>
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

  const predictionWinRate = typeof historyMetrics?.win_rate === "number"
    ? `${Math.round(historyMetrics.win_rate * 100)}%`
    : "n/a";

  return (
    <>
      <NavBar state={{ ...predictionState, health: backendHealth }} />
      <div className={styles.statsPage}>
        <header className={styles.pageHeader}>
          <h1 className={styles.h1}>Prediction Status Page</h1>
          <p className={styles.pageLead}>Live backend health, dataset stats, and recorded predictions.</p>
        </header>

        <section className={styles.summaryGrid}>
          <SummaryCard
            title="Backend Health"
            value={backendHealth?.status ?? "unknown"}
            subtext={backendHealth?.reason}
            intent={backendHealth?.status === "healthy" ? "ok" : backendHealth?.status === "unhealthy" ? "error" : "default"}
          />
          <SummaryCard
            title="Dataset rows"
            value={datasetStatistics?.rows ?? "—"}
            subtext={datasetStatistics?.path ?? "path unknown"}
          />
          <SummaryCard
            title="Predictions recorded"
            value={historyMetrics?.total_predictions ?? predictionHistoryEntries.length}
            subtext={`Win rate: ${predictionWinRate}`}
          />
        </section>

        <section className={styles.scheduleSection}>
          <h2 className={styles.h2}>Next Week Schedule</h2>
          {renderScheduleList()}
        </section>

        <section className={styles.historySection}>
          <h2 className={styles.h2}>Historical Predictions</h2>
          <HistoryChart history={predictionHistoryEntries} state={predictionState} />
        </section>
      </div>
    </>
  );
}
