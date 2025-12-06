// File: frontend/src/components/Card/Card.jsx
// Purpose: Presentational card for a single NFL matchup with probabilities
//          and optional score / model details. Used inside TeamGrid as the
//          main "game tile" for each matchup.
//
// Interacts With:
//   - TeamGrid (parent grid layout).
//   - PredictionContext (provides `matchup` + `prediction` data).
//   - Backend /predict responses (home/away win probabilities, scores).

import React from 'react';
import styles from './Card.module.css'; // CSS module styles

/**
 * Local lookup of NFL team abbreviations → full team names.
 * Shared by all Card instances; not recreated per render.
 */
const NFL_TEAMS_MAP = {
  ARI: 'Arizona Cardinals',
  ATL: 'Atlanta Falcons',
  BAL: 'Baltimore Ravens',
  BUF: 'Buffalo Bills',
  CAR: 'Carolina Panthers',
  CHI: 'Chicago Bears',
  CIN: 'Cincinnati Bengals',
  CLE: 'Cleveland Browns',
  DAL: 'Dallas Cowboys',
  DEN: 'Denver Broncos',
  DET: 'Detroit Lions',
  GB: 'Green Bay Packers',
  HOU: 'Houston Texans',
  IND: 'Indianapolis Colts',
  JAX: 'Jacksonville Jaguars',
  KC: 'Kansas City Chiefs',
  LV: 'Las Vegas Raiders',
  LAC: 'Los Angeles Chargers',
  LAR: 'Los Angeles Rams',
  MIA: 'Miami Dolphins',
  MIN: 'Minnesota Vikings',
  NE: 'New England Patriots',
  NO: 'New Orleans Saints',
  NYG: 'New York Giants',
  NYJ: 'New York Jets',
  PHI: 'Philadelphia Eagles',
  PIT: 'Pittsburgh Steelers',
  SF: 'San Francisco 49ers',
  SEA: 'Seattle Seahawks',
  TB: 'Tampa Bay Buccaneers',
  TEN: 'Tennessee Titans',
  WSH: 'Washington Commanders',
};

/**
 * Normalize a 0–1 probability into a rounded 0–100 integer.
 * Returns `null` when the input is missing or invalid.
 */
const formatProbabilityAsPercentage = (probabilityValue) =>
  typeof probabilityValue === 'number' && isFinite(probabilityValue)
    ? Math.round(probabilityValue * 100)
    : null;

/** Decide which team-name map to use (prop override vs local default). */
const getTeamNameMap = (nfl_teams) =>
  nfl_teams && Object.keys(nfl_teams).length > 0 ? nfl_teams : NFL_TEAMS_MAP;

/** Build the main card CSS class string. */
const buildCardClassNames = ({ hasPrediction, loading, error }) =>
  [
    styles.card,
    hasPrediction ? styles.hasPrediction : '',
    loading ? styles.isLoading : '',
    error ? styles.isError : '',
  ]
    .filter(Boolean)
    .join(' ');

/** Compute the human-readable kickoff time string. */
const getKickoffDisplayTime = (kickoff) =>
  kickoff ? new Date(kickoff).toLocaleString() : 'TBD';

/**
 * Derive convenient flags and values from the prediction object.
 * Keeps the render section clean and self-documenting.
 */
const derivePredictionMeta = (prediction) => {
  const hasScoreDetails =
    prediction?.home_score != null ||
    prediction?.away_score != null ||
    prediction?.point_diff != null;

  const classifierUsed = prediction?.win_classifier_used === true;

  const maxConfidence = formatProbabilityAsPercentage(
    Math.max(
      prediction?.home_win_probability ?? 0,
      prediction?.away_win_probability ?? 0
    )
  );

  return { hasScoreDetails, classifierUsed, maxConfidence };
};

/**
 * @typedef {Object} Matchup
 * @property {string} away_team   - Abbreviation for away team, e.g. "KC".
 * @property {string} home_team   - Abbreviation for home team, e.g. "DAL".
 * @property {string | number | Date} [kickoff] - Kickoff timestamp.
 * @property {string} [away_logo] - URL for away team logo.
 * @property {string} [home_logo] - URL for home team logo.
 */

/**
 * @typedef {Object} Prediction
 * @property {number} [home_win_probability] - 0–1 probability home team wins.
 * @property {number} [away_win_probability] - 0–1 probability away team wins.
 * @property {number} [home_score]
 * @property {number} [away_score]
 * @property {number} [point_diff]
 * @property {boolean} [win_classifier_used]
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
 * @property {(matchup: Matchup) => void} [onReset]
 * @property {Record<string, string>} [nfl_teams]
 */

/**
 * Card v2 - Prop-driven & motion-aware.
 *
 * @param {CardProps} props
 */
export default function Card({
  nfl_teams,
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
  onReset,
}) {
  React.useDebugValue('Card');

  // If we do not have a matchup, nothing to render.
  if (!matchup) return null;

  const { away_team, home_team, kickoff, away_logo, home_logo } = matchup;
  const hasPrediction = !!prediction;

  // 1) Derived data that does not depend on component state
  const teamNameMap = getTeamNameMap(nfl_teams);
  const awayFullName = teamNameMap[away_team] || away_team;
  const homeFullName = teamNameMap[home_team] || home_team;

  const cardClassNames = buildCardClassNames({
    hasPrediction,
    loading,
    error,
  });

  const kickoffDisplayTime = getKickoffDisplayTime(kickoff);

  const { hasScoreDetails, classifierUsed, maxConfidence } =
    derivePredictionMeta(prediction);

  // Local debug state for a brief visual click cue
  const [debugClicked, setDebugClicked] = React.useState(false);

  /**
   * Main click handler wrapper.
   * - Adds debug instrumentation.
   * - Guards against handler exceptions.
   * - Preserves original semantics: call onClick() with no event arg.
   */
  const handleArticleClick = () => {
    if (typeof onClick !== 'function') return;

    try {
      // Visual dev cue
      setDebugClicked(true);
      setTimeout(() => setDebugClicked(false), 700);

      // eslint-disable-next-line no-console
      console.debug('[Card] article clicked', { matchup });

      // Dev-only observable hooks (window event + localStorage)
      try {
        if (typeof window !== 'undefined') {
          let isDev = false;
          try {
            // eslint-disable-next-line no-undef
            isDev = !!(import.meta && import.meta.env && import.meta.env.DEV);
          } catch (_e) {
            isDev = false;
          }

          if (isDev) {
            try {
              window.dispatchEvent(
                new CustomEvent('nfl-card-click', { detail: { matchup } })
              );
            } catch (_e) {
              /* ignore */
            }
          }

          try {
            localStorage.setItem(
              'nfl_last_card_click',
              JSON.stringify({
                game_id: matchup?.game_id || null,
                ts: new Date().toISOString(),
              })
            );
          } catch (_e) {
            /* ignore localStorage failures */
          }
        }
      } catch (_err) {
        /* noop */
      }

      onClick();
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('[Card] onClick handler threw', err);
    }
  };

  /** Keyboard accessibility: allow Enter/Space to trigger the same click path. */
  const handleKeyDown = (event) => {
    if (!onClick) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleArticleClick();
    }
  };

  /**
   * Handle click on the "Reset" button.
   * - Stop the click from bubbling up to the card’s <article> onClick.
   * - Notify the parent via `onReset(matchup)` so it can clear prediction state.
   */
  const handleReset = (event) => {
    try {
      event?.stopPropagation?.();
    } catch (_e) {
      /* noop */
    }

    if (typeof onReset === 'function') {
      try {
        onReset(matchup);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('[Card] onReset handler threw', err);
      }
    }
    // Reset is distinct from "open" – no fallback to onClick.
  };

  return (
    <article
      className={[
        cardClassNames,
        debugClicked ? styles.debugClicked : '',
      ]
        .filter(Boolean)
        .join(' ')}
      // Custom CSS var used by Card.module.css for staggered animations.
      // @ts-ignore - allow custom property name in inline style.
      style={{ '--i': index }}
      aria-pressed={loading ? 'true' : 'false'}
      onClick={handleArticleClick}
      onKeyDown={onClick ? handleKeyDown : undefined}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : -1}
    >
      {/* Loading indicator overlay (small spinner) */}
      {loading && (
        <div className={styles.loadingIndicator} aria-hidden="true">
          <div className={styles.spinner} />
        </div>
      )}

      {/* Top bar: optional icon/title/status */}
      {(title || status || icon) && (
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

      {/* Matchup row: logos + full team names */}
      <div className={styles.matchupRow}>
        <div id="away" className={styles.gameCard}>
          {away_logo && (
            <img
              src={away_logo}
              alt={`${away_team} logo`}
              id="away-logo"
              className={styles.gameCardLogo}
              loading="lazy"
            />
          )}

          <span className="game-card__label">Away</span>
          <span className="game-card__name">{awayFullName}</span>
        </div>

        <div className="game-card__vs">VS</div>

        <div className="game-card__team game-card__team--home">
          {home_logo && (
            <img
              src={home_logo}
              alt={`${home_team} logo`}
              id="home-logo"
              className={styles.gameCardLogo}
              loading="lazy"
            />
          )}

          <span className={styles.gameCardLabel}>Home</span>
          <span className={styles.gameCardName}>{homeFullName}</span>
        </div>
      </div>

      {/* Kickoff metadata */}
      <div className={styles.kickoffRow}>
        <time
          className={styles.kickoffTime}
          dateTime={kickoff ? new Date(kickoff).toISOString() : undefined}
        >
          {kickoffDisplayTime}
        </time>
      </div>

      {/* Prediction body / footer */}
      <footer className="game-card__footer">
        {loading ? (
          <span className="game-card__prediction--loading">
            <span className={styles.spinner} aria-label="Loading" /> Fetching
            prediction...
          </span>
        ) : hasPrediction ? (
          <>
            {/* Reset button: clears prediction via parent handler */}
            <button
              id="reset-button"
              type="button"
              className={styles.resetButton}
              onClick={handleReset}
              aria-label="Reset prediction"
            >
              Reset
            </button>

            <div className={styles.predictionBody}>
              <div className={styles.badgeRow}>
                <span className={styles.badge}>
                  {classifierUsed ? 'Classifier' : 'Logistic fallback'}
                </span>
                {typeof maxConfidence === 'number' && (
                  <span className={styles.badge}>
                    Confidence {maxConfidence}%
                  </span>
                )}
              </div>

              {/* Optional numeric details if present */}
              {hasScoreDetails && (
                <div className={styles.predScore}>
                  <span>Score:</span>
                  <b>
                    {away_team}{' '}
                    {typeof prediction.away_score === 'number'
                      ? Math.round(prediction.away_score)
                      : '-'}{' '}
                    - {home_team}{' '}
                    {typeof prediction.home_score === 'number'
                      ? Math.round(prediction.home_score)
                      : '-'}{' '}
                    {prediction.point_diff != null && (
                      <em className={styles.diff}>
                        {' '}
                        diff:{' '}
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
          </>
        ) : (
          <span className="game-card__prediction game-card__prediction--empty">
            Predictions not available yet
          </span>
        )}
      </footer>

      {/* Optional progress meter (e.g., confidence, pipeline completeness) */}
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
