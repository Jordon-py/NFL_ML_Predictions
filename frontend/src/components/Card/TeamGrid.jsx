// File: frontend/src/components/Card/TeamGrid.jsx
// ------------------------------------------------------------------------------------
// High-level purpose
// ------------------
// Renders a grid of NFL games for a single week and wires those games up to the
// prediction flow.
//
// Responsibilities:
// - Display a grid of <Card /> components (one per matchup).
// - Surface loading and error states per game.
// - Pass the correct game metadata and prediction values into each <Card />.
// - Trigger prediction requests when a card is clicked, either via:
//     • parent-provided onPredict, or
//     • local fallback calling predictGame() directly.
// ------------------------------------------------------------------------------------

import React from 'react';
import Card from './Card.jsx';
import './TeamGrid.css';
import { predictGame } from '../../api/nfl.js';

/**
 * Build a stable key for a game.
 *
 * Strategy:
 * - Prefer backend-provided game_id.
 * - Otherwise, use "<season>-<week>-<HOME_ABBR>-<AWAY_ABBR>".
 */
const toGameKey = (game) =>
  game?.game_id ??
  [
    game?.season,
    game?.week,
    game?.home_abbr || game?.home_team,
    game?.away_abbr || game?.away_team,
  ]
    .filter(Boolean)
    .join('-');

/** Normalise a team identifier into an uppercase abbreviation string. */
const normalizeAbbr = (value) =>
  (value ?? '')
    .toString()
    .trim()
    .toUpperCase();

/** Safely resolve team metadata (name, logo, etc.) from the teams map. */
const getTeamMeta = (teams, abbr) =>
  abbr && teams && teams[abbr] ? teams[abbr] : null;

/**
 * Build the enriched game object passed to <Card />.
 * This keeps the map() body small and self-documenting.
 */
const buildEnrichedGame = (game, teams, prediction) => {
  const homeAbbr = normalizeAbbr(game.home_abbr ?? game.home_team);
  const awayAbbr = normalizeAbbr(game.away_abbr ?? game.away_team);

  const homeMeta = getTeamMeta(teams, homeAbbr);
  const awayMeta = getTeamMeta(teams, awayAbbr);

  const enrichedGame = {
    ...game,
    home_logo: game.home_logo || homeMeta?.logoUrl || null,
    away_logo: game.away_logo || awayMeta?.logoUrl || null,
    // Normalize kickoff variants to a single field
    kickoff: game.kickoff || game.game_day || game.gameday || null,
    home_abbr: homeAbbr,
    away_abbr: awayAbbr,
  };

  if (prediction) {
    // Which team is predicted to win?
    enrichedGame.home_win =
      prediction.home_win === 1 ? homeAbbr : awayAbbr;

    enrichedGame.confidence_score =
      prediction.confidence_score ?? null;

    // Normalize scores into *_pred_score
    enrichedGame.home_pred_score =
      prediction.home_score ?? prediction.home_score_pred ?? null;
    enrichedGame.away_pred_score =
      prediction.away_score ?? prediction.away_score_pred ?? null;

    // Normalize win probability
    enrichedGame.home_win_probability =
      prediction.home_win_probability ?? prediction.probs?.home ?? null;
  }

  return { enrichedGame, homeAbbr, awayAbbr };
};

/** Build the payload expected by the predictGame backend API. */
const buildPredictPayload = (game, homeAbbr, awayAbbr) => ({
  home_team: homeAbbr,
  away_team: awayAbbr,
  season: game?.season ?? game?.season_num ?? null,
  week: game?.week ?? game?.week_num ?? null,
});

/**
 * TeamGrid component
 *
 * @param {Object} [props]
 * @param {number} [props.week]
 * @param {Array<any>} [props.games]
 * @param {boolean} [props.isLoading]
 * @param {Record<string, { name?: string; logoUrl?: string }>} [props.teams]
 * @param {Record<string, any>} [props.predictions]
 * @param {Record<string, boolean>} [props.loading]
 * @param {Record<string, any>} [props.errors]
 * @param {(game: any) => void} [props.onPredict]
 * @param {(game: any) => void} [props.onReset]
 */
export default function TeamGrid({
  week = 10,
  games = [],
  isLoading = false,
  teams = {},
  predictions = {},
  loading = {},
  errors = {},
  onPredict,
  onReset,
} = {}) {
  // Prefer explicit week; fall back to first game; default to 10 as safe placeholder.
  const safeWeek = week ?? games[0]?.week ?? 10;

  // Local loading map used when this component calls predictGame directly.
  const [localLoadingMap, setLocalLoadingMap] = React.useState({});

  // 1. Global "isLoading" state: show skeleton/spinner and skip the grid entirely.
  if (isLoading) {
    return (
      <section className="team-grid" aria-busy="true">
        <header className="team-grid__header">
          <h2 className="team-grid__title">Week {safeWeek} Games</h2>
          <p className="team-grid__subtitle">Loading schedule...</p>
        </header>
        <div className="team-grid__empty">
          <div className="team-grid__spinner" />
        </div>
      </section>
    );
  }

  // 2. Empty schedule state: show a helpful message instead of an empty grid.
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

  // 3. Normal, "loaded" path: render the header and the card grid.
  return (
    <section
      className="team-grid"
      aria-label={`NFL Week ${safeWeek}`}
      data-week={safeWeek}
    >
      <header className="team-grid__header">
        <div className="team-grid__heading">
          <span className="team-grid__badge">Week {safeWeek}</span>
          <h2 className="team-grid__title">Week {safeWeek}</h2>
        </div>
        <p className="team-grid__subtitle">
          Showing <strong>{games.length}</strong> games scheduled.
        </p>
      </header>

      <div className="team-grid__grid">
        {games.map((game, index) => {
          // Stable key for React + lookup maps
          const rawKey = toGameKey(game) || String(index);
          const prediction = predictions?.[rawKey];

          // Combine parent loading with local fallback loading
          const isLocalLoading = Boolean(localLoadingMap?.[rawKey]);
          const isGameLoading = Boolean(loading?.[rawKey]) || isLocalLoading;

          const errorMessage = errors?.[rawKey] ?? null;

          // Build the enriched game object (logos, abbrevs, derived fields)
          const { enrichedGame, homeAbbr, awayAbbr } = buildEnrichedGame(
            game,
            teams,
            prediction
          );

          /**
           * Click handler for a single card.
           *
           * Guard pattern:
           * - If a request is already in-flight, bail.
           * - Prefer parent onPredict; otherwise fall back to predictGame() + local loading.
           */
          const handleClick = async () => {
            console.debug('[TeamGrid] card clicked', { rawKey, game, isGameLoading });

            if (isGameLoading) return;

            if (typeof onPredict === 'function') {
              try {
                onPredict(game);
              } catch (err) {
                // eslint-disable-next-line no-console
                console.error('[TeamGrid] onPredict handler threw', err);
              }
              return;
            }

            const payload = buildPredictPayload(game, homeAbbr, awayAbbr);

            try {
              setLocalLoadingMap((prev) => ({ ...prev, [rawKey]: true }));
              await predictGame(payload);
            } catch (err) {
              // eslint-disable-next-line no-console
              console.error('[TeamGrid] predictGame failed', err);
            } finally {
              setLocalLoadingMap((prev) => {
                const copy = { ...prev };
                delete copy[rawKey];
                return copy;
              });
            }
          };

          return (
            <Card
              key={rawKey}
              matchup={enrichedGame}
              prediction={prediction}
              loading={isGameLoading}
              error={errorMessage}
              index={index}
              onClick={handleClick}
              // Only pass onReset if provided. Card will call onReset(matchup).
              onReset={onReset ? () => onReset(enrichedGame) : undefined}
            />
          );
        })}
      </div>
    </section>
  );
}
