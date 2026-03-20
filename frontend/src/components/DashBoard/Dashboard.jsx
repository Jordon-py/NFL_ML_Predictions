// File: frontend/src/components/DashBoard/Dashboard.jsx
/**
 * Dashboard
 * ---------
 * Loads the next-week schedule and owns prediction state for the grid.
 *
 * Design choices:
 * - No context or custom hooks: plain `useState` + `useEffect`.
 * - No batch endpoint dependency: "Predict All" calls `/predict` per game
 *   with a small concurrency limit.
 */

import { useEffect, useState } from "react";
import TeamGrid from "../Card/TeamGrid.jsx";
import { getNextWeekSchedule, predictGame } from "../../api/client.js";
import {
  buildMatchupKey,
  buildPredictPayload,
  getGameWeek,
} from "../../utils/gameUtils.js";

function removeKey(map, key) {
  if (!key || !Object.prototype.hasOwnProperty.call(map, key)) return map;
  const next = { ...map };
  delete next[key];
  return next;
}

// -------------------------
// Dashboard component
// -------------------------

export default function Dashboard() {
  const [games, setGames] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // These maps all share the same canonical matchup key so the grid can look up
  // prediction, loading, and error state with one identifier.
  const [predictions, setPredictions] = useState({});
  const [loadingMap, setLoadingMap] = useState({});
  const [errorsMap, setErrorsMap] = useState({});
  const [isBulkLoading, setIsBulkLoading] = useState(false);

  const safeGames = Array.isArray(games) ? games : [];

  const loadSchedule = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const schedule = await getNextWeekSchedule();
      setGames(Array.isArray(schedule) ? schedule : []);
    } catch (e) {
      setGames([]);
      setError(e?.message ?? "Failed to load schedule");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadSchedule();
  }, []);

  /**
   * Called when the user clicks a card.
   * This is the "fix": we call predictGame(), then store the result so <Card /> can render it.
   */
  const onPredict = async (game) => {
    const key = buildMatchupKey(game);
    if (!key || loadingMap[key]) return;

    setErrorsMap((prev) => removeKey(prev, key));

    setLoadingMap((prev) => ({ ...prev, [key]: true }));
    try {
      const payload = buildPredictPayload(game);
      const prediction = await predictGame(payload);
      const predictionKey = buildMatchupKey(prediction);
      setPredictions((prev) => ({
        ...prev,
        [key]: prediction,
        ...(predictionKey && predictionKey !== key ? { [predictionKey]: prediction } : {}),
      }));
    } catch (e) {
      setErrorsMap((prev) => ({
        ...prev,
        [key]: e?.message ?? "Prediction failed",
      }));
    } finally {
      setLoadingMap((prev) => removeKey(prev, key));
    }
  };

  async function runWithLimit(items, limit, worker) {
    const queue = [...items];
    const workers = Array.from({ length: Math.max(1, limit) }, async () => {
      while (queue.length) {
        const item = queue.shift();
        if (item == null) return;
        await worker(item);
      }
    });
    await Promise.all(workers);
  }

  const onPredictAll = async () => {
    if (isBulkLoading) return;
    setIsBulkLoading(true);
    try {
      const targets = safeGames.filter((g) => {
        const key = buildMatchupKey(g);
        return key && !predictions[key] && !loadingMap[key];
      });
      await runWithLimit(targets, 4, onPredict);
    } finally {
      setIsBulkLoading(false);
    }
  };

  /**
   * Reset handler called by Card's "Reset" button.
   * Removes prediction + per-game error/loading for that game key.
   */
  const onReset = (gameOrMatchup) => {
    const key = buildMatchupKey(gameOrMatchup);
    if (!key) return;

    setPredictions((prev) => removeKey(prev, key));
    setErrorsMap((prev) => removeKey(prev, key));
    setLoadingMap((prev) => removeKey(prev, key));
  };

  const weekValue = getGameWeek(safeGames[0]);
  const weekLabel = weekValue != null ? `Week ${weekValue}` : "Next Week";

  return (
    <main className="dashboard" aria-label="NFL Predict Dashboard">
      {/* Header / Controls */}
      <header className="dashboard__header">
        <div className="dashboard__titleWrap">
          <h2 className="dashboard__title">Dashboard</h2>
          <p className="dashboard__subtitle">
            Schedule: <strong>{weekLabel}</strong>
          </p>
        </div>

        <div className="dashboard__actions">
          <button
            type="button"
            className="dashboard__btn"
            onClick={loadSchedule}
            disabled={isLoading}
            aria-busy={isLoading ? "true" : "false"}
          >
            {isLoading ? "Refreshing..." : "Refresh Schedule"}
          </button>
        </div>
      </header>

      {/* Global error from schedule/logos hook */}
      {error && (
        <section className="dashboard__notice dashboard__notice--error" role="alert">
          <p>
            <strong>Schedule load failed:</strong> {error}
          </p>
          <button type="button" className="dashboard__btn" onClick={loadSchedule}>
            Try again
          </button>
        </section>
      )}

      {/* Main grid */}
      <TeamGrid
        week={weekValue ?? undefined}
        games={safeGames}
        isLoading={Boolean(isLoading)}
        predictions={predictions}
        loading={loadingMap}
        errors={errorsMap}
        onPredict={onPredict}
        onReset={onReset}
        onPredictAll={onPredictAll}
        isBulkLoading={isBulkLoading}
      />
    </main>
  );
}
