/**
 * NFL Prediction Dashboard (Simplified)
 * -------------------------------------
 * Straightforward container for predictions, schedule display, history, and LLM chat.
 */

import React from "react";
import { predictGame } from "../../api/client.js";
import TeamGrid from "../Card/TeamGrid";
import PredictionResult from "../PredictionResult";
import HistoryChart from "../HistoryChart";
import NavBar from "../NavBar/NavBar";
import ErrorDisplay from "../ErrorDisplay";
import LLMChat from "../LLMChat/LLMChat";
import { buildGameKey } from "../../utils/predictionContextUtils";

export default function Dashboard({
  schedule,
  week,
  teams,
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

  const handlePredict = async (game) => {
    if (!game) return;

    const key = game.game_id || buildGameKey(game);
    if (!key) return;

    setLoading(key, true);
    setError(key, null);

    try {
      const payload = {
        home_team: game.home_team || game.home_abbr,
        away_team: game.away_team || game.away_abbr,
        season: game.season,
        week: game.week,
      };

      const rawPrediction = await predictGame(payload);

      const entry = {
        ...rawPrediction,
        timestamp: new Date().toISOString(),
        game: {
          season: rawPrediction.season ?? game.season,
          week: rawPrediction.week ?? game.week,
          home_abbr: rawPrediction.home_team || game.home_team || game.home_abbr,
          away_abbr: rawPrediction.away_team || game.away_team || game.away_abbr,
        },
      };

      setPrediction(key, entry);
      pushHistory(entry);
    } catch (error) {
      const detail = error?.body?.detail || error?.message || "Prediction request failed";
      setError(key, detail);
    } finally {
      setLoading(key, false);
    }
  };

  const handleReset = (game) => {
    if (!game) return;
    const key = game.game_id || buildGameKey(game);
    if (!key) return;
    setPrediction(key, null);
    setError(key, null);
    setLoading(key, false);
  };

  if (health?.status === "unhealthy" && (!schedule || schedule.length === 0)) {
    return (
      <ErrorDisplay
        error={new Error(health?.reason || "Backend is unhealthy")}
        recoveryOptions={[
          { label: "Reload", action: () => window.location.reload() },
        ]}
      />
    );
  }

  return (
    <div className="dashboard-layout advanced">
      <NavBar state={{ health }} />

      <main className="dashboard-main advanced">
        <header className="dashboard-header advanced">
          <div className="dashboard-header-content">
            <h1 className="dashboard-title">NFL Prediction Dashboard</h1>
            <p className="dashboard-subtitle">
              {week ? `Week ${week} matchups` : "Upcoming matchups"}
            </p>
          </div>
        </header>

        <section className="dashboard-content advanced">
          <div className="content-grid advanced">
            <div className="team-grid-section enhanced">
              <TeamGrid
                week={week}
                games={schedule}
                teams={teams}
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
      )}

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
