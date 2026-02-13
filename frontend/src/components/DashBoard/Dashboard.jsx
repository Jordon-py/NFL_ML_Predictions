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

import { useState, useEffect } from "react";
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
  setPrediction,
  setLoading,
  setError,
  pushHistory,
}) {
  const [debugInfo, setDebugInfo] = useState(null);

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
        game_id: predictionKey || entry.game_id,
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

  const healthStatus = health?.status;
  const backendHealthy = healthStatus === "healthy";
  const scheduleEmpty = !schedule || schedule.length === 0;
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
          </div>
        </header>

        <section className="dashboard-content advanced">
          <div className="content-grid advanced">
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

            <div className="history-section enhanced">
              <HistoryChart history={history} />
            </div>
          </div>
        </section>

        <section className="prediction-results-section advanced" aria-live="polite">
          <PredictionResult entry={current} />
        </section>

        <section className="llm-chat-section">
          <LLMChat prediction={current} />
        </section>
      </main>
    </div>
  );
}
