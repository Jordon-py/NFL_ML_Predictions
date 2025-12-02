// File: frontend/src/components/DashBoard/DashBoard.jsx
// Purpose: Main dashboard container connecting PredictionContext to UI components and prediction actions.
// Functions: useDashboardData, useNavigationState, Dashboard
// Interacts With: PredictionContext, api/client.predictGame, Card/TeamGrid, HistoryChart, NavBar, PredictionResult, LoadingState, ErrorDisplay.

/*
  Dashboard.jsx (simplified + fixed)

  - Derives a clean "view model" from PredictionContext.
  - Wires TeamGrid clicks to the /predict backend via api/client.predictGame.
  - Ensures prediction results are stored in PredictionContext and rendered
    in both the grid and the history chart.
*/

import React, { useMemo, useCallback } from "react";
import { usePredictions } from "../../PredictionContext";
import { predictGame } from "../../api/client";
import TeamGrid from "../Card/TeamGrid";
import PredictionResult from "../PredictionResult";
import HistoryChart from "../HistoryChart";
import NavBar from "../NavBar/NavBar";
import LoadingState from "../LoadingState";
import ErrorDisplay from "../ErrorDisplay";

/**
 * Shape of the derived dashboard data returned by useDashboardData.
 *
 * @typedef {Object} DashboardData
 * @property {Array<any>} upcomingGames
 * @property {number} currentWeek
 * @property {Record<string, any>} teamMetadata
 * @property {Record<string, any>} gamePredictions
 * @property {Array<any>} predictionHistory
 * @property {Record<string, boolean>} loadingMap
 * @property {Record<string, any>} errorMap
 * @property {any} currentPrediction
 * @property {{ status: string, reason?: string }} health
 * @property {{
 *   setPrediction: (key: string, prediction: any) => void,
 *   setLoading: (key: string, loading: boolean) => void,
 *   setError: (key: string, error: string | null) => void,
 *   pushHistory: (entry: any) => void,
 * }} actions
 */

/**
 * Derive the dashboard-friendly view of PredictionContext.
 * This keeps the component tree simple and isolates shape-munging here.
 *
 * @returns {{ isLoading: boolean, error: Error | null, data: DashboardData | null }}
 */
function useDashboardData() {
  const context = /** @type {any} */ (usePredictions());

  return useMemo(() => {
    if (!context) {
      return {
        isLoading: true,
        error: new Error(
          "Prediction context not available. Please refresh the page and try again."
        ),
        data: null,
      };
    }

    const {
      current,
      history = [],
      schedule = [],
      week,
      teams = {},
      predictions = {},
      loading = {},
      errors = {},
      health,
      // actions exposed by PredictionContext
      setPrediction,
      setLoading,
      setError,
      pushHistory,
    } = context;

    const isGlobalLoading = Boolean(loading.global);
    const mergedHealth = health || {
      status: "unknown",
      reason: "no health info",
    };

    const transformedHistory = Array.isArray(history) ? history : [];
    console.log('transformedhistory dashboard', transformedHistory)
    console.log('history', history)
    const transformedGames = Array.isArray(schedule)
      ? schedule.map((game) => {
        const homeCode = (game.home_abbr || game.home_team || "")
          .toString()
          .trim()
          .toUpperCase();
        const awayCode = (game.away_abbr || game.away_team || "")
          .toString()
          .trim()
          .toUpperCase();
        const gameWeek = Number.isFinite(Number(game.week))
          ? Number(game.week)
          : Number(week || 1);

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
      })
      : [];

    const currentWeek = Number.isFinite(Number(week))
      ? Number(week)
      : Number(transformedGames[0]?.week || 1);

    return {
      isLoading: Boolean(isGlobalLoading && transformedGames.length === 0),
      error: errors.global || null,
      data: {
        upcomingGames: transformedGames,
        currentWeek,
        teamMetadata: teams,
        gamePredictions: predictions,
        predictionHistory: transformedHistory,
        currentPrediction: current,
        health: mergedHealth,
        loadingMap: loading,
        errorMap: errors,
        actions: {
          setPrediction,
          setLoading,
          setError,
          pushHistory,
        },
      },
    };
  }, [context]);
}

/**
 * Build a compact navigation summary from dashboard data
 * to keep <NavBar /> fairly dumb and presentation-only.
 *
 * @param {DashboardData | null} data
 */
function useNavigationState(data) {
  return useMemo(() => {
    if (!data) {
      return {
        title: "NFL Prediction Dashboard",
        subtitle: "Initializing...",
        weekLabel: "Week ?",
        healthLabel: "Unknown",
      };
    }

    const { currentWeek, predictionHistory, health } = data;
    const count = Array.isArray(predictionHistory)
      ? predictionHistory.length
      : 0;
    const healthStatus = health?.status || "unknown";

    return {
      title: "NFL Prediction Dashboard",
      subtitle: `${count} historical predictions stored`,
      weekLabel: `Week ${currentWeek}`,
      healthLabel:
        healthStatus === "healthy"
          ? "Backend: Healthy"
          : `Backend: ${healthStatus}`,
    };
  }, [data]);
}

/**
 * Top-level dashboard page component.
 */
function Dashboard() {
  const { isLoading, error, data } = useDashboardData();
  const navState = useNavigationState(data);

  // Early returns for loading and error states
  if (isLoading) {
    return <LoadingState message="Loading schedule and predictions..." />;
  }

  if (error || !data) {
    return (
      <ErrorDisplay
        error={error || new Error("No dashboard data available.")}
      />
    );
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

  const deriveGameKey = useCallback(
    (game) => {
      if (!game) return "";

      const home = (game.home_abbr || game.home_team || "")
        .toString()
        .trim()
        .toUpperCase();
      const away = (game.away_abbr || game.away_team || "")
        .toString()
        .trim()
        .toUpperCase();
      const season = Number(game.season || new Date().getFullYear());
      const weekValue = Number.isFinite(Number(game.week))
        ? Number(game.week)
        : Number(currentWeek || 1);

      return game.game_id || `${season}-${weekValue}-${home}-${away}`;
    },
    [currentWeek]
  );

  /**
   * Click handler for <TeamGrid />.
   * - Normalizes the game row into the payload expected by /predict.
   * - Calls api/client.predictGame.
   * - Stores the result into PredictionContext (predictions + history + current).
   *
   * NOTE: This fixes the previous bug where the handler returned early and
   * never called /predict when a valid game was provided.
   */
  const handlePredictionRequest = useCallback(
    async (game) => {
      // Debugging: log when Dashboard receives a request from TeamGrid/Card
      try {
        // eslint-disable-next-line no-console
        console.debug('[Dashboard] handlePredictionRequest called', { game });
      } catch (_err) { }
      if (!game) {
        console.warn(
          "[Dashboard] handlePredictionRequest called without a game."
        );
        return;
      }

      const home = (game.home_abbr || game.home_team || "")
        .toString()
        .trim()
        .toUpperCase();
      const away = (game.away_abbr || game.away_team || "")
        .toString()
        .trim()
        .toUpperCase();
      const season = Number(game.season || new Date().getFullYear());
      const week = Number.isFinite(Number(game.week))
        ? Number(game.week)
        : Number(currentWeek || 1);

      const baseGameForKey = {
        ...game,
        home_abbr: home,
        away_abbr: away,
        season,
        week,
      };
      const gameKey = deriveGameKey(baseGameForKey);

      if (!gameKey) {
        console.warn("[Dashboard] Could not derive game key for", game);
        return;
      }

      if (!isBackendHealthy) {
        // Do not short-circuit here. Previously we returned early which
        // prevented any network call from being attempted when the health
        // probe had not yet reported 'healthy' (race conditions on startup).
        // Instead, log a warning and continue to attempt the prediction; if
        // the backend truly cannot answer the request the API call will fail
        // and we will set an error in the catch block below.
        // eslint-disable-next-line no-console
        console.warn('[Dashboard] backend health is not healthy; attempting prediction anyway', health);
      }

      // Mark this game as "loading" in context
      setLoading?.(gameKey, true);
      setError?.(gameKey, null);

      try {
        const payload = {
          home_team: home,
          away_team: away,
          season,
          week,
        };

        // Debug: show the payload about to be sent so we can inspect it in DevTools.
        // eslint-disable-next-line no-console
        console.debug('[Dashboard] predictGame payload', payload);

        // Persist a tiny marker so we can confirm the UI attempted a prediction
        // even when network logs are hard to observe (e.g., proxy or CORS). This
        // is safe for dev and will be ignored in production environments.
        try {
          localStorage.setItem(
            'nfl_last_predict_attempt',
            JSON.stringify({ gameKey, payload, ts: new Date().toISOString() })
          );
        } catch (_e) { /* ignore localStorage failures */ }

        const rawPrediction = await predictGame(payload);

        // Normalize a few common fields so Card/TeamGrid can rely on them.
        const homeProb =
          rawPrediction?.home_win_probability ??
          rawPrediction?.prob_home ??
          rawPrediction?.probs?.home ??
          null;

        const awayProb =
          rawPrediction?.away_win_probability ??
          rawPrediction?.prob_away ??
          rawPrediction?.probs?.away ??
          (typeof homeProb === "number" ? 1 - homeProb : null);

        const homeScore =
          rawPrediction?.home_score ?? rawPrediction?.home_score_pred ?? null;

        const awayScore =
          rawPrediction?.away_score ?? rawPrediction?.away_score_pred ?? null;

        const pointDiff =
          rawPrediction?.point_diff ??
          (typeof homeScore === "number" && typeof awayScore === "number"
            ? homeScore - awayScore
            : null);

        const enrichedPrediction = {
          ...rawPrediction,
          game_id: rawPrediction?.game_id || gameKey,
          season: rawPrediction?.season ?? season,
          week: rawPrediction?.week ?? week,
          home_team: rawPrediction?.home_team || home,
          away_team: rawPrediction?.away_team || away,
          home_win_probability: homeProb,
          away_win_probability: awayProb,
          home_score: homeScore,
          away_score: awayScore,
          point_diff: pointDiff,
        };
        console.log('ENRICHEDPREDICTIONS', enrichedPrediction)
        // Store in context keyed by gameKey so TeamGrid can look it up.
        setPrediction?.(gameKey, enrichedPrediction);

        // Push into history with an attached timestamp + game context.
        pushHistory?.({
          ...enrichedPrediction,
          timestamp: new Date().toISOString(),
          game: { ...game, home_abbr: home, away_abbr: away, week, season },
        });
      } catch (err) {
        console.error("[Dashboard] Prediction request failed", err);
        setError?.(
          gameKey,
          err instanceof Error ? err.message : "Prediction request failed."
        );
      } finally {
        setLoading?.(gameKey, false);
      }
    },
    [
      currentWeek,
      deriveGameKey,
      health,
      isBackendHealthy,
      setPrediction,
      setLoading,
      setError,
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
              <HistoryChart
                history={predictionHistory}
                state={currentPrediction}
              />
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

export default Dashboard;
