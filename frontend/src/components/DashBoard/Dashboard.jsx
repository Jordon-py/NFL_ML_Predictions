// File: frontend/src/components/DashBoard/DashBoard.jsx
// Purpose: Main dashboard container connecting PredictionContext to UI components and prediction actions.
// Functions: useDashboardData, useNavigationState, Dashboard
// Interacts With: PredictionContext, api/client.predictGame, TeamGrid, HistoryChart, NavBar, PredictionResult, LoadingState, ErrorDisplay.

import React, { useCallback } from "react";
import { usePredictions } from "../../PredictionContext";
import { predictGame } from "../../api/nfl.js";
import TeamGrid from "../Card/TeamGrid";
import PredictionResult from "../PredictionResult";
import HistoryChart from "../HistoryChart";
import NavBar from "../NavBar/NavBar";
import LoadingState from "../LoadingState";
import ErrorDisplay from "../ErrorDisplay";

/* ----------------------------- small helpers ----------------------------- */

// Clamp a number into [min, max]
const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));

// Normalize a team code to uppercase abbrev-like string
function normalizeTeamCode(value) {
  return (value || "").toString().trim().toUpperCase();
}

// Normalize schedule rows into a consistent shape the UI can rely on
function normalizeSchedule(schedule, fallbackWeek = 1) {
  if (!Array.isArray(schedule)) return [];

  return schedule.map((game) => {
    const homeCode = normalizeTeamCode(game.home_abbr || game.home_team);
    const awayCode = normalizeTeamCode(game.away_abbr || game.away_team);

    const gameWeek = Number.isFinite(Number(game.week))
      ? Number(game.week)
      : Number(fallbackWeek || 1);

    const id =
      game.id ||
      game.game_id ||
      `${game.season ?? ""}-${gameWeek}-${homeCode}-${awayCode}`;

    return {
      ...game,
      id,
      home_abbr: homeCode || game.home_abbr,
      away_abbr: awayCode || game.away_abbr,
      week: gameWeek,
    };
  });
}

/* ------------------------ Calibration Logic ---------------------------- */

// Simple "no-op" calibration model builder for now.
// Can be expanded to use historical data for Isotonic Regression if needed.
function buildCalibrationModel(history) {
  return { type: "identity", version: 1 };
}

// Apply calibration model to raw probabilities.
// Currently just passes through the raw values (identity).
function calibrateWinProbabilities(rawHome, rawAway, model) {
  // If no raw probs, return nulls
  if (rawHome === null || rawHome === undefined) {
    return { home: null, away: null, meta: model };
  }
  
  // Basic normalization so they sum to 1 if both exist
  let h = Number(rawHome);
  let a = rawAway !== null && rawAway !== undefined ? Number(rawAway) : 1 - h;
  
  return {
    home: clamp(h),
    away: clamp(a),
    meta: model,
  };
}

/* --------------------------- derived view-model --------------------------- */

/**
 * Derive the dashboard-friendly view of PredictionContext.
 * (No useMemo: this recalculates each render for clarity.)
 */
function useDashboardData() {
  const context = usePredictions();

  const {
    schedule,
    week,
    history,
    current,
    teams = {},
    predictions = {},
    loading = {},
    errors = {},
    health,

    // actions from PredictionContext
    setPrediction,
    setLoading,
    setError,
    pushHistory,
  } = context;

  const upcomingGames = normalizeSchedule(schedule, week);
  const currentWeek = Number.isFinite(Number(week))
    ? Number(week)
    : Number(upcomingGames[0]?.week || 1);

  const isLoading = Boolean(loading.global) && upcomingGames.length === 0;
  const error = errors.global || null;

  const mergedHealth = health || { status: "unknown", reason: "no health info" };
  const predictionHistory = Array.isArray(history) ? history : [];

  return {
    isLoading,
    error,
    data: {
      upcomingGames,
      currentWeek,
      teamMetadata: teams,
      gamePredictions: predictions,
      predictionHistory,
      loadingMap: loading,
      errorMap: errors,
      currentPrediction: current,
      health: mergedHealth,
      actions: { setPrediction, setLoading, setError, pushHistory },
    },
  };
}

/**
 * Compact nav summary for <NavBar /> (no useMemo).
 */
function useNavigationState(data) {
  if (!data) {
    return {
      title: "NFL Prediction Dashboard",
      heroSubtitle: "Initializing...",
      subtitle: "Loading...",
      weekLabel: "Week ?",
      healthLabel: healthStatus,
    };
  }

  const { currentWeek, predictionHistory, health } = data;

  const count = Array.isArray(predictionHistory) ? predictionHistory.length : 0;
  const healthStatus = health?.status || "unknown";

  return {
    title: "Read the Field. Beat the Line.",
    heroSubtitle:
      "AI-Powered NFL Game Predictions — Select a matchup to generate live, model-backed probabilities.",
    subtitle: `${count} historical predictions stored`,
    weekLabel: `Week ${currentWeek}`,
    healthLabel: healthStatus === "healthy" ? "Backend: Healthy" : `Backend: ${healthStatus}`,
  };
}

/* -------------------------------- Dashboard ------------------------------- */

export default function Dashboard() {
  const { isLoading, error, data } = useDashboardData();
  const navState = useNavigationState(data);

  if (isLoading) {
    return <LoadingState message="Loading schedule and predictions..." error={error} />;
  }

  if (error || !data) {
    return <ErrorDisplay error={error || new Error("No dashboard data available.")} />;
  }

  const {
    upcomingGames: games = [],
    currentWeek,
    teamMetadata: teams = {},
    gamePredictions: predictions = {},
    predictionHistory = [],
    currentPrediction,
    health,
    loadingMap: loading = {},
    errorMap: errors = {},
    actions = {},
  } = data;

  const { setPrediction, setLoading, setError, pushHistory } = actions;

  const isBackendHealthy = health?.status === "healthy";
  const healthMessage = isBackendHealthy
    ? "Backend is healthy. Click a matchup to request fresh predictions."
    : health?.reason || "Backend is not ready for predictions yet.";

  // No useMemo: compute calibration model each render for now (simple + predictable)
  const calibrationModel = buildCalibrationModel(predictionHistory);

  // Keep game-key generation consistent everywhere
  const deriveGameKey = useCallback(
    (game) => {
      if (!game) return "";

      const home = normalizeTeamCode(game.home_abbr || game.home_team);
      const away = normalizeTeamCode(game.away_abbr || game.away_team);

      const season = Number(game.season || 2025);
      const weekValue = Number.isFinite(Number(game.week))
        ? Number(game.week)
        : Number(currentWeek || 1);

      return game.game_id || `${season}-${weekValue}-${home}-${away}`;
    },
    [currentWeek]
  );

  /**
   * Click handler for <TeamGrid />:
   * - normalize payload
   * - call backend
   * - store prediction + history
   */
  const handlePredictionRequest = useCallback(
    async (game) => {
      if (!game) return;

      const home = normalizeTeamCode(game.home_abbr || game.home_team);
      const away = normalizeTeamCode(game.away_abbr || game.away_team);

      const season = Number(game.season || 2025);
      const week = Number.isFinite(Number(game.week)) ? Number(game.week) : Number(currentWeek || 1);

      const gameKey = deriveGameKey({ ...game, home_abbr: home, away_abbr: away, season, week });
      if (!gameKey) return;

      // Even if health is “not healthy”, we still attempt the request.
      // If backend truly can’t respond, we catch and store error below.
      setLoading?.(gameKey, true);
      setError?.(gameKey, null);

      try {
        const payload = { home_team: home, away_team: away, season, week };

        const rawPrediction = await predictGame(payload);
        console.log("[Dashboard] Prediction received:", rawPrediction);

        // Pull probabilities from common keys
        const rawHomeProb =
          rawPrediction?.home_win_probability ??
          rawPrediction?.prob_home ??
          rawPrediction?.probs?.home ??
          null;

        const rawAwayProb =
          rawPrediction?.away_win_probability ??
          rawPrediction?.prob_away ??
          rawPrediction?.probs?.away ??
          null;

        const calibrated = calibrateWinProbabilities(rawHomeProb, rawAwayProb, calibrationModel);

        // Scores: prefer backend outputs, keep minimal fallbacks
        const homeScore = rawPrediction?.home_score ?? rawPrediction?.home_score_pred ?? null;
        const awayScore = rawPrediction?.away_score ?? rawPrediction?.away_score_pred ?? null;

        const pointDiff =
          rawPrediction?.point_diff ??
          (typeof homeScore === "number" && typeof awayScore === "number" ? homeScore - awayScore : null);

        const enrichedPrediction = {
          ...rawPrediction,
          game_id: rawPrediction?.game_id || gameKey,
          season: rawPrediction?.season ?? season,
          week: rawPrediction?.week ?? week,
          home_team: rawPrediction?.home_team || home,
          away_team: rawPrediction?.away_team || away,

          home_win_probability: calibrated.home,
          away_win_probability: calibrated.away,

          home_score: homeScore,
          away_score: awayScore,
          point_diff: pointDiff,

          calibration_meta: calibrated.meta,
        };

        setPrediction?.(gameKey, enrichedPrediction);

        pushHistory?.({
          ...enrichedPrediction,
          timestamp: new Date().toISOString(),
          game: { ...game, home_abbr: home, away_abbr: away, week, season },
        });
      } catch (err) {
        setError?.(gameKey, err instanceof Error ? err.message : "Prediction request failed.");
      } finally {
        setLoading?.(gameKey, false);
      }
    },
    [
      calibrationModel,
      currentWeek,
      deriveGameKey,
      setError,
      setLoading,
      setPrediction,
      pushHistory,
    ]
  );

  const handleResetPrediction = useCallback(
    (game) => {
      const gameKey = deriveGameKey(game);
      if (!gameKey) return;

      setPrediction?.(gameKey, null);
      setError?.(gameKey, null);
      setLoading?.(gameKey, false);
    },
    [deriveGameKey, setError, setLoading, setPrediction]
  );

  return (
    <div className="dashboard-layout">
      <NavBar state={navState} />

      <main className="dashboard-main">
        <header className="dashboard-header">
          <div className="dashboard-header-content">
            <h1 className="dashboard-title">NFL Prediction Dashboard</h1>
            <p className="dashboard-subtitle">{healthMessage}</p>
          </div>
        </header>

        <section className="dashboard-content">
          <div className="content-grid">
            <div className="team-grid-section">
              <TeamGrid
                week={currentWeek}
                games={games}
                teams={teams}
                predictions={predictions}
                loading={loading}
                errors={errors}
                onPredict={handlePredictionRequest}
                onReset={handleResetPrediction}
              />
            </div>

            <div className="history-section">
              <HistoryChart history={predictionHistory} state={currentPrediction} />
            </div>
          </div>
        </section>

        <section className="prediction-results-section" aria-live="polite">
          <PredictionResult entry={currentPrediction} />
        </section>
      </main>
    </div>
  );
}
