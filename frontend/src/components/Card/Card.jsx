// Card/Card.jsx
// ------------------------------------------------------
// Presentation component for a single NFL game.
// - Expects a `game` object with team + kickoff info.
// - Designed to look clean inside the TeamGrid layout.
// - Shows matchup, time, venue, and optional prediction.
// ------------------------------------------------------

import React from 'react';
import './Card.module.css'; // optional: or you can keep all styles in TeamGrid.css

/**
 * @typedef {Object} Game
 * @property {number|string|null} [week]
 * @property {string|null} [home_team]
 * @property {string|null} [away_team]
 * @property {string|null} [home]
 * @property {string|null} [away]
 * @property {string|null} [venue]
 * @property {string|number|Date|null} [kickoff]
 * @property {string|number|Date|null} [start_time]
 * @property {string|null} [home_logo]
 * @property {string|null} [away_logo]
 * @property {number|null} [home_win_probability]
 * @property {number|null} [home_pred_score]
 * @property {number|null} [away_pred_score]
 */

/**
 * @param {string|number|Date|null|undefined} kickoffRaw
 * @returns {string}
 */
function formatKickoff(kickoffRaw) {
  if (!kickoffRaw) return 'TBD';

  const date = new Date(kickoffRaw);
  if (Number.isNaN(date.getTime())) {
    // If we can’t parse it, just show the raw string
    return typeof kickoffRaw === 'string' ? kickoffRaw : String(kickoffRaw);
  }

  // Example: Sun, Nov 9 · 10:00 AM
  return date.toLocaleString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

/**
 * @param {number|null|undefined} prob
 * @returns {string|null}
 */
function toPercent(prob) {
  if (prob === null || prob === undefined) return null;
  const pct = Math.round(prob * 100);
  return `${pct}%`;
}

/**
 * @param {{
 *   game: Game,
 *   isLoading?: boolean,
 *   error?: string | null,
 *   onClick?: () => void,
 * }} props
 */
function Card({ game, isLoading = false, error = null, onClick }) {
  const {
    week,
    home_team,
    away_team,
    home,
    away,
    venue,
    kickoff,
    start_time,
    home_logo,
    away_logo,
    home_win_probability,
    home_pred_score,
    away_pred_score,
  } = game;

  const displayWeek = week ?? '–';
  const homeTeam = home_team || home || 'HOME';
  const awayTeam = away_team || away || 'AWAY';
  const kickoffLabel = formatKickoff(kickoff || start_time);
  const winProbLabel = toPercent(home_win_probability);

  const hasPrediction =
    home_pred_score !== undefined && away_pred_score !== undefined;

  const rootClassName = [
    'game-card',
    isLoading ? 'game-card--loading' : '',
    error ? 'game-card--error' : '',
  ]
    .filter(Boolean)
    .join(' ');

  const handleKeyDown = (event) => {
    if (!onClick) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onClick();
    }
  };

  return (
    <article
      className={rootClassName}
      aria-label={`${awayTeam} at ${homeTeam}`}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <header className="game-card__header">
        <span className="game-card__week">W{displayWeek}</span>
        <span className="game-card__kickoff">{kickoffLabel}</span>
      </header>

      <div className="game-card__teams">
        <div className="game-card__team game-card__team--away">
          {away_logo && (
            <img
              src={away_logo}
              alt={`${awayTeam} logo`}
              className="game-card__logo"
              loading="lazy"
            />
          )}
          <span className="game-card__label">Away</span>
          <span className="game-card__name">{awayTeam}</span>
        </div>

        <div className="game-card__vs"> @ </div>

        <div className="game-card__team game-card__team--home">
          {home_logo && (
            <img
              src={home_logo}
              alt={`${homeTeam} logo`}
              className="game-card__logo"
              loading="lazy"
            />
          )}
          <span className="game-card__label">Home</span>
          <span className="game-card__name">{homeTeam}</span>
        </div>
      </div>

      {venue && (
        <div className="game-card__venue" title={venue}>
          {venue}
        </div>
      )}

      <footer className="game-card__footer">
        {isLoading ? (
          <span className="game-card__prediction game-card__prediction--loading">
            Fetching prediction...
          </span>
        ) : hasPrediction ? (
          <div className="game-card__prediction">
            <span className="game-card__prediction-label">Model Score</span>
            <div className="game-card__prediction-row">
              <span className="game-card__prediction-team">
                {awayTeam}: <strong>{
                  away_pred_score == null
                    ? ''
                    : (away_pred_score.toFixed?.(1) ?? away_pred_score)
                }</strong>
              </span>
              <span className="game-card__prediction-team">
                {homeTeam}: <strong>{
                  home_pred_score == null
                    ? ''
                    : (home_pred_score.toFixed?.(1) ?? home_pred_score)
                }</strong>
              </span>
            </div>
          </div>
        ) : (
          <span className="game-card__prediction game-card__prediction--empty">
            Predictions not available yet
          </span>
        )}

        {winProbLabel && (
          <div className="game-card__winprob">
            <span className="game-card__winprob-label">Home win chance</span>
            <span className="game-card__winprob-value">{winProbLabel}</span>
            <div
              className="game-card__winprob-bar"
              aria-hidden="true"
            >
              <div
                className="game-card__winprob-fill"
                style={{ width: winProbLabel }}
              />
            </div>
          </div>
        )}

        {error && !isLoading && (
          <div className="game-card__error" role="status">
            {error}
          </div>
        )}
      </footer>
    </article>
  );
}

export default Card;
