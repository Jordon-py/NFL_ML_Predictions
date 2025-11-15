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
 * generateGameKey
 * Generates a unique, stable identifier for a game object.
 * Prefers backend-provided game_id; falls back to a composite key of season, week, home_abbr, and away_abbr.
 * @param {Object} game - Game object
 * @returns {string} Unique key for the game
 */
export function generateGameKey(game) {
  return game?.game_id ?? [game?.season, game?.week, game?.home_abbr, game?.away_abbr].filter(Boolean).join("-");
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
  const gamesForCurrentWeek = useMemo(() => {
    const filteredGames = games.filter((game) => Number(game?.week) === Number(week));
    // Sort by kickoff ascending if timestamp present; fallback keeps original order
    return [...filteredGames].sort((gameA, gameB) => {
      const kickoffA = Date.parse(gameA.kickoff || gameA.kickoff_ts_utc || gameA.kickoff_iso || '') || 0;
      const kickoffB = Date.parse(gameB.kickoff || gameB.kickoff_ts_utc || gameB.kickoff_iso || '') || 0;
      return kickoffA - kickoffB;
    });
  }, [games, week]);

  const totalGamesMessage = gamesForCurrentWeek.length 
    ? `${gamesForCurrentWeek.length} matchups` 
    : "No games found for this week";

  return (
    <section className="team-grid">
      <header className="team-grid__header">
        <h2 className="team-grid__title">Week {week} Games</h2>
        <p className="team-grid__subtitle">{totalGamesMessage}</p>
      </header>

      <div className="team-grid-cards" role="list">
        {gamesForCurrentWeek.map((game, index) => {
          const gameKey = generateGameKey(game);
          const isGameLoading = loading[gameKey] || false;
          const gameError = errors[gameKey] || null;
          const gamePrediction = predictions[gameKey] || null;

          const matchupData = {
            away_team: game.away_abbr,
            home_team: game.home_abbr,
            // Prefer kickoff_ts_utc (most precise, UTC timestamp), then kickoff_iso (ISO string), then fallback to kickoff (legacy/local format) if others unavailable.
            kickoff: game.kickoff_ts_utc || game.kickoff_iso || game.kickoff || "",
            away_logo: teams[game.away_abbr]?.logoUrl,
            home_logo: teams[game.home_abbr]?.logoUrl,
          };

          // Extract status logic for clarity and maintainability
          const predictionStatus = isGameLoading
            ? "Predicting…"
            : gameError
              ? "Error"
              : gamePrediction
                ? "Predicted"
                : "Ready";

          // The CSS variable '--i' is used by TeamGrid.css to enable staggered animations or grid ordering effects for each card.
          return (
            <div key={gameKey} className="grid-item" style={{ "--i": index }} role="listitem">
              <Card
                index={index}
                matchup={matchupData}
                prediction={gamePrediction}
                loading={isGameLoading}
                error={gameError}
                title={`${game.away_abbr} @ ${game.home_abbr}`}
                status={predictionStatus}
                onClick={onPredict ? () => onPredict(game) : undefined}
              />
            </div>
          );
        })}
      </div>
    </section>
  );
}
