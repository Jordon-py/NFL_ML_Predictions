// ==========================================
// File: frontend/src/components/DashBoard/Dashboard.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ../../api/client.js, ../Card/TeamGrid, ../PredictionResult
// Notes: Presentation-focused component.
// ==========================================

/**
 * NFL Prediction Dashboard (Simplified)
 * -------------------------------------
 * Straightforward container for predictions, schedule display, history, and LLM chat.
 */

import { useState, useEffect, useMemo } from "react";
import { getDebugInfo, predictGame } from "../../api/client.js";
import TeamGrid from "../Card/TeamGrid";
import PredictionResult from "../PredictionResult";
import HistoryChart from "../HistoryChart";
import NavBar from "../NavBar/NavBar";
import ErrorDisplay from "../ErrorDisplay";
import LLMChat from "../LLMChat/LLMChat";
import { API_BASE } from "../../api/fetch";
import { buildGameKey } from "../../utils/predictionContextUtils";
import { toEntry } from "../../utils/predictionHelpers";

export default function Dashboard({
  schedule,
  week,
  predictions,
  loading,
  errors,
  current,
  history,
  health,
  seasonContext,
  setPrediction,
  setLoading,
  setError,
  pushHistory,
}) {
  const [debugInfo, setDebugInfo] = useState(null);
  const [showcase, setShowcase] = useState(null);
  const [showcaseLoading, setShowcaseLoading] = useState(false);
  const [showcaseError, setShowcaseError] = useState("");

  const TEAM_POOL = useMemo(
    () => [
      "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
      "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
      "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
      "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    ],
    []
  );

  useEffect(() => {
    let active = true;
    getDebugInfo()
      .then((data) => {
        if (active) setDebugInfo(data);
      })
      .catch(() => { });
    return () => {
      active = false;
    };
  }, []);

  const handlePredict = async (game) => {
    if (!game) return;

    const key = buildGameKey(game);
    if (!key) return;

    setLoading(key, true);
    setError(key, null);

    try {
      const home = game.home_team || game.home_abbr;
      const away = game.away_team || game.away_abbr;
      const season = game.season;
      const week = game.week;

      const rawPrediction = await predictGame(home, away, season, week);
      const entry = toEntry({ prediction: rawPrediction, game, source: "teamgrid" });


      const normalizedEntry = {
        ...entry,
        game_id: key || entry.game_id,
      };

      setPrediction(key, normalizedEntry);
      pushHistory(normalizedEntry);
    } catch (error) {
      const detail = error?.body?.detail || error?.message || "Prediction request failed";
      setError(key, detail);
    } finally {
      setLoading(key, false);
    }
  };

  const handleReset = (game) => {
    if (!game) return;
    const key = buildGameKey(game);
    if (!key) return;
    setPrediction(key, null);
    setError(key, null);
    setLoading(key, false);
  };

  const runOffseasonShowcase = async () => {
    if (showcaseLoading) return;
    setShowcaseError("");
    setShowcaseLoading(true);
    try {
      const homeIdx = Math.floor(Math.random() * TEAM_POOL.length);
      let awayIdx = Math.floor(Math.random() * TEAM_POOL.length);
      if (awayIdx === homeIdx) awayIdx = (awayIdx + 7) % TEAM_POOL.length;
      const home = TEAM_POOL[homeIdx];
      const away = TEAM_POOL[awayIdx];
      const season = Number(seasonContext?.current_season) || new Date().getFullYear();
      const kickoffWeek = Number(seasonContext?.display_week) || 1;

      const rawPrediction = await predictGame(home, away, season, kickoffWeek);
      const syntheticGame = {
        home_team: home,
        away_team: away,
        home_abbr: home,
        away_abbr: away,
        season,
        week: kickoffWeek,
      };
      const gameKey = buildGameKey(syntheticGame);
      const entry = toEntry({
        prediction: rawPrediction,
        game: syntheticGame,
        source: "offseason-showcase",
      });
      const normalizedEntry = {
        ...entry,
        game_id: gameKey || entry.game_id,
      };
      setShowcase(normalizedEntry);
      if (gameKey) {
        setPrediction(gameKey, normalizedEntry);
      }
      pushHistory(normalizedEntry);
    } catch (error) {
      const detail = error?.body?.detail || error?.message || "Could not generate offseason showcase matchup.";
      setShowcaseError(detail);
    } finally {
      setShowcaseLoading(false);
    }
  };

  const healthStatus = health?.status;
  const backendHealthy = healthStatus === "healthy";
  const scheduleEmpty = !schedule || schedule.length === 0;
  const seasonPhase = seasonContext?.phase || "offseason";
  const seasonLabel = seasonContext?.label || "Offseason";
  const seasonMessage = seasonContext?.message || "No live slate currently available.";
  const nextKickoffLabel = seasonContext?.next_kickoff
    ? new Date(seasonContext.next_kickoff).toLocaleString()
    : "TBD";
  const isOffseasonMode = seasonPhase === "offseason";
  const shouldShowBackendError =
    scheduleEmpty && Boolean(healthStatus) && healthStatus !== "loading" && !backendHealthy;

  if (shouldShowBackendError) {
    const reason = health?.reason || health?.mode || `Backend status: ${healthStatus}`;
    const message = API_BASE ? `${reason} (API: ${API_BASE})` : reason;
    return (
      <ErrorDisplay
        error={new Error(message)}
        onRetry={() => window.location.reload()}
      />
    );
  }

  const modelDir = debugInfo?.config?.models_dir;
  const modelLabel = modelDir
    ? modelDir.split(/[\\/]/).slice(-2).join("/")
    : null;

  return (
    <div className="dashboard-layout advanced">
      <NavBar state={{ health }} />

      <main className="dashboard-main advanced">
        <header className="dashboard-header advanced">
          <div className="dashboard-header-content">
            <h1 className="dashboard-title">NFL Prediction Dashboard</h1>
            {modelLabel && (
              <p className="dashboard-subtitle">Model: {modelLabel}</p>
            )}
            <div className={`season-context-ribbon phase-${seasonPhase}`}>
              <strong className="season-context-pill">{seasonLabel}</strong>
              <span className="season-context-message">{seasonMessage}</span>
              <span className="season-context-kickoff">Next kickoff: {nextKickoffLabel}</span>
            </div>
          </div>
        </header>

        <section className="dashboard-content advanced">
          <div className="content-grid advanced">
            {isOffseasonMode ? (
              <section className="offseason-mode-panel" aria-live="polite">
                <h2>Offseason Mode</h2>
                <p>
                  Live games are paused, so this mode runs showcase predictions for potential matchups
                  and keeps model exploration useful between seasons.
                </p>
                <div className="offseason-mode-actions">
                  <button
                    type="button"
                    onClick={runOffseasonShowcase}
                    disabled={showcaseLoading}
                    className="offseason-mode-button"
                  >
                    {showcaseLoading ? "Generating matchup..." : "Generate Offseason Showcase Matchup"}
                  </button>
                </div>
                {showcaseError ? <p className="offseason-mode-error">{showcaseError}</p> : null}
                <PredictionResult entry={showcase || current} />
              </section>
            ) : (
              <div className="team-grid-section enhanced">
                <TeamGrid
                  week={week}
                  games={schedule}
                  predictions={predictions}
                  loading={loading}
                  errors={errors}
                  onPredict={handlePredict}
                  onReset={handleReset}
                />
              </div>
            )}

            <div className="history-section enhanced">
              <HistoryChart history={history} />
            </div>
          </div>
        </section>

        {!isOffseasonMode && (
          <section className="prediction-results-section advanced" aria-live="polite">
            <PredictionResult entry={current} />
          </section>
        )}

        <section className="llm-chat-section">
          <LLMChat prediction={current} />
        </section>
      </main>
    </div>
  );
}
