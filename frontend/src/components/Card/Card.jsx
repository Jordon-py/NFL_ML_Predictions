// Card/Card.jsx
// ------------------------------------------------------
// Presentation component for a single NFL game.
// - Expects a `game` object with team + kickoff info.
// - Designed to look clean inside the TeamGrid layout.
// - Shows matchup, time, venue, and optional prediction.
// ------------------------------------------------------

import React from 'react';
import styles from './Card.module.css'; // CSS module styles

/**
 * @typedef {Object} Matchup
 * @property {string} away_team
 * @property {string} home_team
 * @property {string | number | Date} [kickoff]
 * @property {string} [away_logo]
 * @property {string} [home_logo]
 */

/**
 * @typedef {Object} Prediction
 * @property {number} [home_win_probability]
 * @property {number} [away_win_probability]
 * @property {number} [home_score]
 * @property {number} [away_score]
 * @property {number} [point_diff]
 */

/**
 * @typedef {Object} CardProps
 * @property {Matchup} matchup
 * @property {Prediction} [prediction]
 * @property {string} [title]
 * @property {string} [status]
 * @property {React.ReactNode} [icon]
 * @property {number} [progress]
 * @property {boolean} [loading]
 * @property {string} [error]
 * @property {number} [index]
 * @property {() => void} [onClick]
 */

/**
 * Card v2 — Prop-driven & motion-aware
 * Props:
 *  - matchup: { away_team, home_team, kickoff, away_logo, home_logo }
 *  - prediction?: { home_win_probability, away_win_probability, home_score?, away_score?, point_diff? }
 *  - title?: string  (e.g., "AI Node")
 *  - status?: "Active" | "Idle" | "Error" | string
 *  - icon?: ReactNode
 *  - progress?: number (0..100)
 *  - loading?: boolean
 *  - error?: string
 *  - index?: number (stagger animation index)
 *  - onClick?: () => void
 *
 * @param {CardProps} props
 */
export default function Card({
  matchup,
  prediction,
  title,
  status,
  icon,
  progress,
  loading = false,
  error,
  index = 0,
  onClick,
}) {
  React.useDebugValue('Card');

  if (!matchup) return null;

  const { away_team, home_team, kickoff, away_logo, home_logo } = matchup;
  const hasPrediction = !!prediction;

  /**
   * @param {number | null | undefined} probabilityValue
   * @returns {number | null}
   */
  const formatProbabilityAsPercentage = (probabilityValue) =>
    typeof probabilityValue === 'number' && isFinite(probabilityValue)
      ? Math.round(probabilityValue * 100)
      : null;

  const cardClassNames = [
    styles.card,
    hasPrediction ? styles.hasPrediction : '',
    loading ? styles.isLoading : '',
    error ? styles.isError : '',
  ]
    .filter(Boolean)
    .join(' ');

  const kickoffDisplayTime = kickoff ? new Date(kickoff).toLocaleString() : 'TBD';
  const shouldShowTopBar = title || status || icon;
  const hasScoreDetails =
    prediction?.home_score != null ||
    prediction?.away_score != null ||
    prediction?.point_diff != null;

  /** @param {any} event */
  const handleKeyDown = (event) => {
    if (!onClick) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onClick();
    }
  };

  return (
    <article
      className={cardClassNames}
      // Custom CSS variable used by Card.module.css for staggered animations.
      // @ts-ignore - allow custom property name in inline style.
      style={{ '--i': index }}
      aria-pressed={loading ? 'true' : 'false'}
      onClick={onClick}
      onKeyDown={onClick ? handleKeyDown : undefined}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : -1}
    >
      {/* Top bar: optional icon/title/status */}
      {shouldShowTopBar && (
        <div className={styles.topBar}>
          <div className={styles.left}>
            {icon && (
              <span className={styles.icon} aria-hidden>
                {icon}
              </span>
            )}
            {title && <strong className={styles.title}>{title}</strong>}
          </div>
          {status && <span className={styles.status}>{status}</span>}
        </div>
      )}

      {/* Matchup row */}
      <div className={styles.matchupRow}>
        <div className="game-card__team game-card__team--away">
          {away_logo && (
            <img
              src={away_logo}
              alt={`${away_team} logo`}
              className="game-card__logo"
              loading="lazy"
            />
          )}
          <span className="game-card__label">Away</span>
          <span className="game-card__name">{away_team}</span>
        </div>

        <div className="game-card__vs">@</div>

        <div className="game-card__team game-card__team--home">
          {home_logo && (
            <img
              src={home_logo}
              alt={`${home_team} logo`}
              className="game-card__logo"
              loading="lazy"
            />
          )}
          <span className="game-card__label">Home</span>
          <span className="game-card__name">{home_team}</span>
        </div>
      </div>

      <div className={styles.meta}>
        <time
          className={styles.kickoff}
          dateTime={kickoff ? new Date(kickoff).toISOString() : undefined}
        >
          {kickoffDisplayTime}
        </time>
      </div>

      <footer className="game-card__footer">
        {loading ? (
          <span className="game-card__prediction game-card__prediction--loading">
            Fetching prediction...
          </span>
        ) : hasPrediction ? (
          <div className={styles.predictionBody}>
            <div className={styles.probRow}>
              <span>Home</span>
              <b>
                {formatProbabilityAsPercentage(
                  prediction.home_win_probability
                )}
                %
              </b>
            </div>
            <div className={styles.probRow}>
              <span>Away</span>
              <b>
                {formatProbabilityAsPercentage(
                  prediction.away_win_probability
                )}
                %
              </b>
            </div>
            {/* Optional numeric details if present */}
            {hasScoreDetails && (
              <div className={styles.detailRow}>
                <span>Score</span>
                <b>
                  {away_team} {prediction.away_score ?? '—'} –{' '}
                  {prediction.home_score ?? '—'} {home_team}
                  {prediction.point_diff != null && (
                    <em className={styles.diff}>
                      {' '}
                      • Δ{' '}
                      {typeof prediction.point_diff === 'number' &&
                      typeof prediction.point_diff.toFixed === 'function'
                        ? prediction.point_diff.toFixed(1)
                        : prediction.point_diff}
                    </em>
                  )}
                </b>
              </div>
            )}
          </div>
        ) : (
          <span className="game-card__prediction game-card__prediction--empty">
            Predictions not available yet
          </span>
        )}
      </footer>

      {/* Optional progress meter */}
      {typeof progress === 'number' && isFinite(progress) && (
        <div className={styles.progressTrack} aria-hidden>
          <div
            className={styles.progressBar}
            style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
          />
        </div>
      )}
    </article>
  );
}
 