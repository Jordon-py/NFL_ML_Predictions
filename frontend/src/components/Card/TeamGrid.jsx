// TeamGrid.jsx
// ------------------------------------------------------
// Grid layout for all games in a given NFL week.
// - Receives a `week` number and an array of `games`.
// - Renders a responsive CSS grid of <Card /> components.
// - Keeps empty / loading states friendly and clear.
// ------------------------------------------------------

import React from 'react';
import Card from './Card.jsx';        // adjust path if needed
import './TeamGrid.css';
import { getNextWeekSchedule, predictGame } from '../../api/client.js';

/** Build a stable key for a game, mirroring PredictionContext/getKey + StatsPage/toGameKey. */
const toGameKey = (game) =>
  game?.game_id ?? [
    game?.season,
    game?.week,
    game?.home_abbr || game?.home_team,
    game?.away_abbr || game?.away_team,
  ]
    .filter(Boolean)
    .join('-');

/**
 * @param {{
 *   week?: number,
 *   games?: Array<any>,
 *   isLoading?: boolean,
 *   teams?: Record<string, { name?: string; logoUrl?: string }>,
 *   predictions?: Record<string, any>,
 *   loading?: Record<string, boolean>,
 *   errors?: Record<string, any>,
 *   onPredict?: (game: any) => void,
 * }} [props]
 */
function TeamGrid({
  week = 10,
  games = [],
  isLoading = false,
  teams = {},
  predictions = {},
  loading = {},
  errors = {},
  onPredict,
} = {}) {
  const safeWeek = week ?? (games[0]?.week ?? 10);

  if (isLoading) {
    return (
      <section className="team-grid" aria-busy="true">
        <header className="team-grid__header">
          <h2 className="team-grid__title">Week {safeWeek} Games</h2>
          <p className="team-grid__subtitle">Loading schedule…</p>
        </header>
        <div className="team-grid__empty">
          <div className="team-grid__spinner" />
        </div>
      </section>
    );
  }

  if (!games || games.length === 0) {
    return (
      <section className="team-grid">
        <header className="team-grid__header">
          <h2 className="team-grid__title">Week {safeWeek} Games</h2>
          <p className="team-grid__subtitle">
            No games found for Week {safeWeek}. Try refreshing or checking your API.
          </p>
        </header>
        <div className="team-grid__empty">
          <p className="team-grid__empty-text">
            Once the schedule loads, all Week {safeWeek} matchups will appear here.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section
      className="team-grid"
      aria-label={`NFL Week ${safeWeek} games`}
      data-week={safeWeek}
    >
      <header className="team-grid__header">
        <div className="team-grid__heading">
          <span className="team-grid__badge">Week {safeWeek}</span>
          <h2 className="team-grid__title">Week {safeWeek} Games</h2>
        </div>
        <p className="team-grid__subtitle">
          Showing <strong>{games.length}</strong> games scheduled.
        </p>
      </header>

      <div className="team-grid__grid">
        {games.map((game, index) => {
          const rawKey = toGameKey(game) || String(index);

          // Look up any existing prediction + request state for this game.
          const prediction = predictions?.[rawKey];
          const isGameLoading = Boolean(loading?.[rawKey]);
          const errorMessage = errors?.[rawKey] ?? null;

          // Enrich the schedule row with team metadata (logos + pretty names) when available.
          const homeAbbr = game.home_abbr || game.home_team;
          const awayAbbr = game.away_abbr || game.away_team;
          const homeMeta = homeAbbr && teams && teams[homeAbbr] ? teams[homeAbbr] : null;
          const awayMeta = awayAbbr && teams && teams[awayAbbr] ? teams[awayAbbr] : null;

          const enrichedGame = {
            ...game,
            home_logo: homeMeta?.logoUrl,
            away_logo: awayMeta?.logoUrl,
          };

          if (prediction) {
            enrichedGame.home_pred_score =
              prediction.home_score ?? prediction.home_score_pred ?? null;
            enrichedGame.away_pred_score =
              prediction.away_score ?? prediction.away_score_pred ?? null;
            enrichedGame.home_win_probability =
              prediction.home_win_probability ?? prediction.probs?.home ?? null;
          }

          const handleClick = () => {
            if (typeof onPredict === 'function' && !isGameLoading) {
              onPredict(game);
            }
          };

          return (
            <Card
              key={rawKey}
              game={enrichedGame}
              isLoading={isGameLoading}
              error={errorMessage}
              onClick={handleClick}
            />
          );
        })}
      </div>
    </section>
  );
}

export default TeamGrid;
