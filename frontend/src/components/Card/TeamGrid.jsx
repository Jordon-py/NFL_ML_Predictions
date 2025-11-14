// placeholder
// TeamGrid.jsx — Week view integrated with prop-driven Card
import React, { useMemo } from "react";
import Card from "./Card.jsx";
import "./TeamGrid.css";

/**
 * TeamGrid (Week View)
 * Renders all games for the selected week using the futuristic Card component.
 * 
 * Visual/UX intent: Presents a responsive, animated grid of matchup cards, each styled with modern effects to highlight NFL games and prediction status for an engaging, informative user experience.
 *
 * Props:
 *  - games: Array<Game>  // full schedule array (season, week, home_abbr, away_abbr, kickoff_* etc.)
 *  - week?: number       // defaults to 10
 *  - teams?: Record<string, { logoUrl?: string, name?: string }>
 *  - predictions?: Record<string, Prediction>         // keyed by game key/id
 *  - loading?: Record<string, boolean>                // keyed by game key/id
 *  - errors?: Record<string, string>                  // keyed by game key/id
 *  - onPredict?: (game: Game) => void                 // trigger prediction for a game
 *
 * Game shape (expected):
 *  { game_id?: string, season: number, week: number,
 *    home_abbr: string, away_abbr: string,
 *    kickoff_ts_utc?: string, kickoff_iso?: string, kickoff?: string }
 *
 * Prediction shape (example):
 *  { home_win_probability: number, away_win_probability: number,
 *    home_score?: number, away_score?: number, point_diff?: number }
 */

/**
 * getKey
 * Generates a unique, stable key for a game object.
 * Prefers backend-provided game_id; falls back to a composite key of season, week, home_abbr, and away_abbr.
 * @param {Object} g - Game object
 * @returns {string} Unique key for the game
 */
export function getKey(g) {
  return g?.game_id ?? [g?.season, g?.week, g?.home_abbr, g?.away_abbr].filter(Boolean).join("-");
}

export default function TeamGrid({
  games = [],
  week = 4,
  teams = {},
  predictions = {},
  loading = {},
  errors = {},
  onPredict,
}) {
  // Filter to the requested week and keep order stable
  // NOTE: For optimal memoization, ensure 'games' is a stable reference upstream (e.g., useMemo/useState in parent).
  const weekGames = useMemo(() => {
    const filtered = games.filter((g) => Number(g?.week) === Number(week));
    // Sort by kickoff ascending if timestamp present; fallback keeps original order
    return [...filtered].sort((a, b) => {
      const ta = Date.parse(a.kickoff || a.kickoff_ts_utc || a.kickoff_iso || '') || 0;
      const tb = Date.parse(b.kickoff || b.kickoff_ts_utc || b.kickoff_iso || '') || 0;
      return ta - tb;
    });
  }, [games, week]);

  return (
    <section className="team-grid">
      <header className="team-grid__header">
        <h2 className="team-grid__title">Week {week} Games</h2>
        <p className="team-grid__subtitle">
          {weekGames.length ? `${weekGames.length} matchups` : "No games found for this week"}
        </p>
      </header>

      <div className="team-grid-cards" role="list">
        {weekGames.map((game, index) => {
          const key = getKey(game);
          const isLoading = loading[key] || false;
          const error = errors[key] || null;

          const matchup = {
            away_team: game.away_abbr,
            home_team: game.home_abbr,
            // Prefer kickoff_ts_utc (most precise, UTC timestamp), then kickoff_iso (ISO string), then fallback to kickoff (legacy/local format) if others unavailable.
            kickoff: game.kickoff_ts_utc || game.kickoff_iso || game.kickoff || "",
            away_logo: teams[game.away_abbr]?.logoUrl,
            home_logo: teams[game.home_abbr]?.logoUrl,
          };

          const prediction = predictions[key] || null;
          // Extract status logic for clarity and maintainability
          const status = isLoading
            ? "Predicting…"
            : error
              ? "Error"
              : prediction
                ? "Predicted"
                : "Ready";
          // The CSS variable '--i' is used by TeamGrid.css to enable staggered animations or grid ordering effects for each card.
          return (
            <div key={key} className="grid-item" style={{ "--i": index }} role="listitem">
              <Card
                index={index}
                matchup={matchup}
                prediction={prediction}
                loading={isLoading}
                error={error}
                // Optional cosmetics you can customize or remove:
                title={`${game.away_abbr} @ ${game.home_abbr}`}
                status={status}
                // Click triggers prediction if provided
                onClick={onPredict ? () => onPredict(game) : undefined}
              />
            </div>
          );
        })}
      </div>
    </section>
  );
}
