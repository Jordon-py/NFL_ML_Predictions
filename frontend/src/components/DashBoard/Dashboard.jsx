// File: frontend/src/components/DashBoard/Dashboard.jsx
/**
 * Dashboard.jsx
 * -------------
 * Purpose:
 *   The "home screen" that:
 *   1) loads the next-week schedule + team logos (context data)
 *   2) renders the schedule as a clickable <TeamGrid />
 *   3) owns prediction state (predictions/loading/errors) so cards can actually show results
 *
 * Why this fixes common Dashboard bugs:
 * - TeamGrid can *call* predictGame() on click, but it cannot store the response for you.
 *   If Dashboard doesn't pass `onPredict`, cards will stay stuck on "Predictions not available yet".
 * - This file wires the hook -> TeamGrid props correctly and normalizes team/logo shapes.
 *
 * References (your uploaded files):
 * - useNextWeekSchedule(): returns { games, teams, isLoading, error, refresh }.
 * - TeamGrid: expects { games, teams, isLoading, predictions, loading, errors, onPredict, onReset }.
 * - Card: renders prediction details when prediction fields exist.
 */

import React, { useCallback, useMemo, useState } from "react";
import { getNextWeekPredictions, predictGame } from "../../api/client.js";
import { useNextWeekSchedule } from "../../hooks/useNextWeekSchedule.js";
import TeamGrid from "../Card/TeamGrid.jsx";
import NavBar from "../NavBar/NavBar.jsx";
import "./Dashboard.module.css";

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
  return (
    game?.game_id ??
    [
      game?.season,
      game?.week,
      game?.home_abbr || game?.home_team,
      game?.away_abbr || game?.away_team,
    ]
      .filter(Boolean)
      .join("-")
  );
}

/**
 * Normalize whatever the backend returns for teams/logos into:
 *   { [ABBR]: { name?: string, logoUrl?: string } }
 *
 * This matters because TeamGrid tries: teams[abbr].logoUrl (see your TeamGrid.jsx).
 */
function normalizeTeamsMap(rawTeams) {
  const out = {};

  if (!rawTeams || typeof rawTeams !== "object") return out;

  for (const [rawKey, rawVal] of Object.entries(rawTeams)) {
    const abbr = normalizeAbbr(rawKey);

    // Case A: rawTeams["KC"] = "https://.../kc.png"
    if (typeof rawVal === "string") {
      out[abbr] = { logoUrl: rawVal };
      continue;
    }

    // Case B: rawTeams["KC"] = { logoUrl: "...", name: "..." }
    if (rawVal && typeof rawVal === "object") {
      const logoUrl =
        rawVal.logoUrl ??
        rawVal.logo_url ??
        rawVal.logo ??
        rawVal.url ??
        null;

      const name = rawVal.name ?? rawVal.fullName ?? rawVal.team ?? null;

      out[abbr] = {
        ...(name ? { name } : {}),
        ...(logoUrl ? { logoUrl } : {}),
      };
    }
  }

  return out;
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
  // 1) Context data: schedule + logos
  const { games, teams, isLoading, error, refresh } = useNextWeekSchedule();

  // 2) Prediction state: these maps are keyed by the same `toGameKey(game)` used by TeamGrid
  const [predictions, setPredictions] = useState({});
  const [loadingMap, setLoadingMap] = useState({});
  const [errorsMap, setErrorsMap] = useState({});
  const [bulkLoading, setBulkLoading] = useState(false);
  // Ref-based guard for in-flight predictions to avoid stale-closure issues
  const inFlightRef = React.useRef(new Set());

  // Normalize team map shape so TeamGrid can reliably find logos at teams[abbr].logoUrl
  const normalizedTeams = useMemo(() => normalizeTeamsMap(teams), [teams]);

  // Derive current week label for UI (fallback: "Next Week")
  const weekLabel = useMemo(() => {
    const w =
      games?.[0]?.week ??
      games?.[0]?.week_num ??
      games?.[0]?.week_number ??
      null;
    return w != null ? String(w) : "Next Week";
  }, [games]);

  /**
   * Called when the user clicks a card.
   * This is the "fix": we call predictGame(), then store the result so <Card /> can render it.
   */
  const onPredict = useCallback(
    async (game) => {
      const key = toGameKey(game);

      // Guard: don't double-fire a prediction for the same game while it's in-flight
      if (inFlightRef.current.has(key)) return;
      inFlightRef.current.add(key);

      // Clear any previous error for this game
      setErrorsMap((prev) => {
        if (!prev[key]) return prev;
        const copy = { ...prev };
        delete copy[key];
        return copy;
      });

      const payload = buildPredictPayload(game);

      try {
        setLoadingMap((prev) => ({ ...prev, [key]: true }));

        // Backend call
        const prediction = await predictGame(payload);

        // Store prediction so Card can display confidence/scores/etc.
        setPredictions((prev) => ({ ...prev, [key]: prediction }));
      } catch (err) {
        const message =
          err?.message ??
          "Prediction request failed. Check backend logs/CORS and payload shape.";

        setErrorsMap((prev) => ({ ...prev, [key]: message }));
      } finally {
        inFlightRef.current.delete(key);
        setLoadingMap((prev) => {
          const copy = { ...prev };
          delete copy[key];
          return copy;
        });
      }
    },
    []
  );

  /**
   * Reset handler called by Card's "Reset" button.
   * Removes prediction + per-game error/loading for that game key.
   */
  const onReset = useCallback((gameOrMatchup) => {
    const key = toGameKey(gameOrMatchup);

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
  }, []);

  const onPredictAll = useCallback(async () => {
    if (bulkLoading) return;
    setBulkLoading(true);
    try {
      const games = await getNextWeekPredictions();
      if (!Array.isArray(games)) return;

      const newPreds = {};
      for (const g of games) {
        const key = toGameKey(g);
        if (g && g.prediction) {
          newPreds[key] = g.prediction;
        }
      }
      setPredictions((prev) => ({ ...prev, ...newPreds }));
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Predict all failed", err);
    } finally {
      setBulkLoading(false);
    }
  }, [bulkLoading]);

  return (
    <>
      <NavBar />
      <main className="dashboard" aria-label="NFL Predict Dashboard">
        {/* Header / Controls */}
        <header className="dashboard__header">
          <div className="dashboard__titleWrap">
            <h2 className="dashboard__title">Dashboard</h2>
            <p className="dashboard__subtitle">
              Schedule: <strong>Week {weekLabel}</strong>
            </p>
          </div>

          <div className="dashboard__actions">
            <button
              type="button"
              className="dashboard__btn"
              onClick={refresh}
              disabled={isLoading}
              aria-busy={isLoading ? "true" : "false"}
            >
              {isLoading ? "Refreshing..." : "Refresh Schedule"}
            </button>
            <button
              type="button"
              className="dashboard__btn"
              onClick={onPredictAll}
              disabled={bulkLoading || isLoading}
              aria-busy={bulkLoading ? "true" : "false"}
              style={{ marginLeft: 8 }}
            >
              {bulkLoading ? "Predicting..." : "Predict All Games"}
            </button>
          </div>
        </header>

        {/* Global error from schedule/logos hook */}
        {error && (
          <section className="dashboard__notice dashboard__notice--error" role="alert">
            <p>
              <strong>Schedule load failed:</strong> {error}
            </p>
            <button type="button" className="dashboard__btn" onClick={refresh}>
              Try again
            </button>
          </section>
        )}

        {/* Main grid */}
        <TeamGrid
          week={games?.[0]?.week ?? games?.[0]?.week_num ?? undefined}
          games={Array.isArray(games) ? games : []}
          isLoading={Boolean(isLoading)}
          teams={normalizedTeams}
          predictions={predictions}
          loading={loadingMap}
          errors={errorsMap}
          onPredict={onPredict}
          onReset={onReset}
          onPredictAll={onPredictAll}
          isBulkLoading={bulkLoading}
        />
      </main>
    </>
    );
  }

