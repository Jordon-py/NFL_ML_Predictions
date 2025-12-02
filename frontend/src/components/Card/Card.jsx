// File: frontend/src/components/Card/Card.jsx
// Purpose: Presentational card for a single NFL matchup with probabilities
//          and optional score / model details. Used inside TeamGrid as the
//          main "game tile" for each matchup.
// Functions:
//   - Card(...)                          (export default) — main React component.
//   - formatProbabilityAsPercentage(...) (inner helper)   — normalizes 0–1 to 0–100%.
// Data:
//   - NFL_TEAMS_MAP — local lookup from team abbreviation → full franchise name.
// Interacts With:
//   - TeamGrid (parent grid layout).
//   - PredictionContext (provides `matchup` + `prediction` data).
//   - Backend /predict responses (home/away win probabilities, scores).

import React from 'react';
import styles from './Card.module.css'; // CSS module styles
import { MdFilledButton } from '@material/web/button/filled-button';

/**
 * Local lookup of NFL team abbreviations → full team names.
 * This is intentionally defined at the module level so it is:
 * - shared by all Card instances
 * - not recreated on every render
 *
 * Callers MAY still pass an `nfl_teams` prop to override/extend this map
 * (e.g., for historical teams), but if they do not, we fall back here.
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
 * @typedef {Object} Matchup
 * @property {string} away_team   - Abbreviation for away team, e.g. "KC".
 * @property {string} home_team   - Abbreviation for home team, e.g. "DAL".
 * @property {string | number | Date} [kickoff] - Kickoff timestamp (any Date-compatible type).
 * @property {string} [away_logo] - URL for away team logo.
 * @property {string} [home_logo] - URL for home team logo.
 */

/**
 * @typedef {Object} Prediction
 * @property {number} [home_win_probability] - 0–1 probability home team wins.
 * @property {number} [away_win_probability] - 0–1 probability away team wins.
 * @property {number} [home_score]           - Predicted or actual home score.
 * @property {number} [away_score]           - Predicted or actual away score.
 * @property {number} [point_diff]           - Predicted margin (home - away, usually).
 * @property {boolean} [win_classifier_used] - Whether a classifier model was used.
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
 * @property {(matchup: Matchup) => void} [onReset] - Optional callback when the reset button is clicked.
 * @property {Record<string, string>} [nfl_teams]   - Optional override for team name map.
 */

/**
 * Card v2 - Prop-driven & motion-aware.
 *
 * Props:
 *  - matchup: { away_team, home_team, kickoff, away_logo, home_logo }
 *  - prediction?: {
 *        home_win_probability,
 *        away_win_probability,
 *        home_score?,
 *        away_score?,
 *        point_diff?,
 *        win_classifier_used?
 *    }
 *  - title?: string   (e.g., "AI Node")
 *  - status?: string  (e.g., "Active" | "Idle" | "Error")
 *  - icon?: ReactNode
 *  - progress?: number (0..100)
 *  - loading?: boolean
 *  - error?: string
 *  - index?: number     (stagger animation index)
 *  - onClick?: () => void
 *  - onReset?: (matchup: Matchup) => void
 *  - nfl_teams?: Record<abbr, fullName> override map
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

  if (!matchup) return null;

  const { away_team, home_team, kickoff, away_logo, home_logo } = matchup;
  const hasPrediction = !!prediction;

  /**
   * Normalize a 0–1 probability into a rounded 0–100 integer.
   * Returns `null` when the input is missing or invalid, so the caller
   * can decide whether/how to render it.
   *
   * @param {number | null | undefined} probabilityValue
   * @returns {number | null}
   */
  const formatProbabilityAsPercentage = (probabilityValue) =>
    typeof probabilityValue === 'number' && isFinite(probabilityValue)
      ? Math.round(probabilityValue * 100)
      : null;

  // Choose the active team-name map:
  // - Use the prop `nfl_teams` if provided (and non-empty),
  // - otherwise fall back to the local NFL_TEAMS_MAP defined above.
  const teamNameMap =
    nfl_teams && Object.keys(nfl_teams).length > 0 ? nfl_teams : NFL_TEAMS_MAP;

  // Human-friendly team names used in the UI.
  const awayFullName = teamNameMap[away_team] || away_team;
  const homeFullName = teamNameMap[home_team] || home_team;

  // Build the card's CSS class string in a readable way.
  const cardClassNames = [
    styles.card,
    hasPrediction ? styles.hasPrediction : '',
    loading ? styles.isLoading : '',
    error ? styles.isError : '',
  ]
    .filter(Boolean)
    .join(' ');

  // Kickoff time: use a human-readable locale string or fallback label.
  const kickoffDisplayTime = kickoff
    ? new Date(kickoff).toLocaleString()
    : 'TBD';

  // Simple boolean helpers that clarify rendering conditions.
  const shouldShowTopBar = title || status || icon;
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

  /** Keyboard accessibility: allow Enter/Space to trigger onClick. */
  const handleKeyDown = (event) => {
    if (!onClick) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      // Use the article click wrapper so keyboard and pointer events
      // share the same instrumentation path.
      handleArticleClick?.();
    }
  };

  // Local debug state for visual click feedback during dev-only troubleshooting.
  const [debugClicked, setDebugClicked] = React.useState(false);

  /**
   * Wrapper for the provided onClick prop that adds lightweight
   * debug instrumentation and guards against handler exceptions.
   */
  const handleArticleClick = (event) => {
    if (typeof onClick !== 'function') return;
    try {
      // Small visual cue for devs so clicks are visible without DevTools.
      setDebugClicked(true);
      setTimeout(() => setDebugClicked(false), 700);

      // eslint-disable-next-line no-console
      console.debug('[Card] article clicked', { matchup });

      // Dev-only observable: write a click marker to window + localStorage.
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

      // Preserve original semantics: call onClick with no DOM event argument.
      onClick();
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('[Card] onClick handler threw', err);
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
        // Pass the matchup so the parent knows which game to reset.
        onReset(matchup);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('[Card] onReset handler threw', err);
      }
    }
    // No fallback to onClick: reset is a distinct action from "open card".
  };

  return (
    <article
      className={[cardClassNames, debugClicked ? styles.debugClicked : '']
        .filter(Boolean)
        .join(' ')}
      // Custom CSS variable used by Card.module.css for staggered animations.
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

      {/* Matchup row: logos + abbreviations + full team names */}
      <div className={styles.matchupRow}>
        <div className="game-card__team game-card__team--away">
          {away_logo && (
            <img
              src={away_logo}
              alt={`${away_team} logo`}
              id={styles.away_logo}
              className="game-card__logo"
              loading="lazy"
            />
          )}

          <span className="game-card__label">Away</span>

          {/* Abbreviation + full name (e.g., "KC — Kansas City Chiefs") */}
          <span className="game-card__name">
           {awayFullName}
          </span>
        </div>

        <div className="game-card__vs">@</div>

        <div className="game-card__team game-card__team--home">
          {home_logo && (
            <img
              src={home_logo}
              alt={`${home_team} logo`}
              id={styles.home_logo}
              className={styles.game-card_logo}
              loading="lazy"
            />
          )}

          <span className="game-card__label">Home</span>

          <span className="game-card__name">
           {homeFullName}
          </span>
        </div>
      </div>

      {/* Kickoff metadata */}
      <div className={styles.meta}>
        <time
          className={styles.kickoff}
          dateTime={kickoff ? new Date(kickoff).toISOString() : undefined}
        >
          {kickoffDisplayTime}
        </time>
      </div>

      {/* Prediction body / footer */}
      <footer className="game-card__footer">
        {loading ? (
          <span className="game-card__prediction game-card__prediction--loading">
            Fetching prediction...
          </span>
        ) : hasPrediction ? (
          <>
            {/* Reset button: clears prediction via parent handler */}
            <button
              type="button"
              className={styles.reset}
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
                <div className={styles.detailRow}>
                  <span>Predicted Score:</span>
                  <b>
                    {away_team}{' '}
                    {typeof prediction.away_score === 'number'
                      ? Math.round(prediction.away_score)
                      : '-'}{' '}
                    -{' '}
                    {typeof prediction.home_score === 'number'
                      ? Math.round(prediction.home_score)
                      : '-'}{' '}
                    {home_team}
                    {prediction.point_diff != null && (
                      <em className={styles.diff}>
                        {' '}
                        diff{' '}
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

      {/* Optional progress meter (e.g., pipeline completeness, confidence, etc.) */}
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
