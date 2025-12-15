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

// -------------------------
// Small pure helpers (no React)
// -------------------------

/** Normalise a team identifier into an uppercase abbreviation string. */
function normalizeAbbr(value) {
  return (value ?? "").toString().trim().toUpperCase();
}

/**
 * Build a stable key for a game.
 * Must match TeamGrid's key strategy so our predictions map lines up with its lookups.
 */
function toGameKey(game) {
  const season = game?.season ?? game?.season_num ?? "";
  const week = game?.week ?? game?.week_num ?? "";
  const home = normalizeAbbr(game?.home_abbr ?? game?.home_team);
  const away = normalizeAbbr(game?.away_abbr ?? game?.away_team);
  return [season, week, home, away].filter(Boolean).join("-");
}

/** Build the payload expected by the predictGame backend API. */
function buildPredictPayload(game) {
  const homeAbbr = normalizeAbbr(game?.home_abbr ?? game?.home_team);
  const awayAbbr = normalizeAbbr(game?.away_abbr ?? game?.away_team);

  return {
    home_team: homeAbbr,
    away_team: awayAbbr,
    season: game?.season ?? game?.season_num ?? null,
    week: game?.week ?? game?.week_num ?? null,
  };
}

// -------------------------
// Dashboard component
// -------------------------

export default function Dashboard() {
  const [games, setGames] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // 2) Prediction state: these maps are keyed by the same `toGameKey(game)` used by TeamGrid
  const [predictions, setPredictions] = useState({});
  const [loadingMap, setLoadingMap] = useState({});
  const [errorsMap, setErrorsMap] = useState({});
  const [isBulkLoading, setIsBulkLoading] = useState(false);

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
    const key = toGameKey(game);
    if (!key || loadingMap[key]) return;

    setErrorsMap((prev) => {
      if (!prev[key]) return prev;
      const copy = { ...prev };
      delete copy[key];
      return copy;
    });

    setLoadingMap((prev) => ({ ...prev, [key]: true }));
    try {
      const payload = buildPredictPayload(game);
      const prediction = await predictGame(payload);
      const predictionKey = toGameKey(prediction);
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
      setLoadingMap((prev) => {
        const copy = { ...prev };
        delete copy[key];
        return copy;
      });
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
      const targets = (Array.isArray(games) ? games : []).filter((g) => {
        const key = toGameKey(g);
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
    const key = toGameKey(gameOrMatchup);
    if (!key) return;

    setPredictions((prev) => {
      if (!prev[key]) return prev;
      const copy = { ...prev };
      delete copy[key];
      return copy;
    });
    setErrorsMap((prev) => {
      if (!prev[key]) return prev;
      const copy = { ...prev };
      delete copy[key];
      return copy;
    });
    setLoadingMap((prev) => {
      if (!prev[key]) return prev;
      const copy = { ...prev };
      delete copy[key];
      return copy;
    });
  };

  const weekValue =
    games?.[0]?.week ?? games?.[0]?.week_num ?? games?.[0]?.week_number ?? null;
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
        week={games?.[0]?.week ?? games?.[0]?.week_num ?? undefined}
        games={Array.isArray(games) ? games : []}
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
