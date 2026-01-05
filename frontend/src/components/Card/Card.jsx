// ==========================================
// File: frontend/src/components/Card/Card.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ./Card.module.css
// Notes: Presentation-focused component.
// ==========================================

// File: frontend/src/components/Card/Card.jsx
// Purpose: Presentational card for a single NFL matchup with probabilities
//          and optional score / model details. Used inside TeamGrid as the
//          main "game tile" for each matchup.
//
// Interacts With:
//   - TeamGrid (parent grid layout).
//   - Parent container (provides `matchup` + `prediction` data).
//   - Backend /predict responses (home/away win probabilities, scores).

import React from 'react';
import styles from './Card.module.css'; // CSS module styles

/**
 * Local lookup of NFL team abbreviations - full team names.
 * This is intentionally defined at the module level so it is:
 * - shared by all Card instances
 * - not recreated on every render
 *
 * When the backend does not provide team names, we fall back to this map.
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
 * Normalize a 0-1 probability into a rounded 0-100 integer.
 * Returns `null` when the input is missing or invalid.
 */
const formatProbabilityAsPercentage = (probabilityValue) =>
  typeof probabilityValue === 'number' && isFinite(probabilityValue)
    ? Math.round(probabilityValue * 100)
    : null;

/** Build the main card CSS class string. */
const buildCardClassNames = ({ hasPrediction, loading, error, debugClicked }) =>
  [
    styles.card,
    'game-card',
    hasPrediction ? styles.hasPrediction : '',
    loading ? styles.isLoading : '',
    error ? styles.isError : '',
    debugClicked ? styles.debugClicked : '',
  ]
    .filter(Boolean)
    .join(' ');

/** Compute the human-readable kickoff time string. */
const getKickoffDisplayTime = (kickoff) => {
  if (!kickoff) return 'TBD';
  const d = new Date(kickoff);
  if (isNaN(d.getTime())) {
    // Attempt to handle "YYYY-MM-DD HH:MM" format if raw Date fails
    return kickoff;
  }
  return d.toLocaleString([], { 
    weekday: 'short', 
    month: 'short', 
    day: 'numeric', 
    hour: '2-digit', 
    minute: '2-digit' 
  });
};

/**
 * Derive convenient flags and values from the prediction object.
 * Keeps the render section clean and self-documenting.
 */
const derivePredictionMeta = (prediction) => {
  const homeScore = prediction?.home_score;
  const awayScore = prediction?.away_score;
  const pointDiff =
    prediction?.point_diff ??
    (homeScore != null && awayScore != null ? homeScore - awayScore : null);

  const hasScoreDetails =
    homeScore != null ||
    awayScore != null ||
    pointDiff != null;

  const sim = prediction?.simulation_metrics;
  const isExpert = !!sim;
  const classifierUsed = !isExpert && prediction?.win_classifier_used === true;

  const homeProb = prediction?.home_win_probability;
  const awayProb = prediction?.away_win_probability;

  const maxConfidence =
    typeof homeProb === 'number' || typeof awayProb === 'number'
      ? formatProbabilityAsPercentage(Math.max(homeProb ?? 0, awayProb ?? 0))
      : null;

  return { hasScoreDetails, classifierUsed, isExpert, maxConfidence, sim, homeScore, awayScore };
};

/**
 * @typedef {Object} Matchup
 * @property {string} away_team   - Abbreviation for away team, e.g. "KC".
 * @property {string} home_team   - Abbreviation for home team, e.g. "DAL".
 * @property {string | number | Date} [kickoff] - Kickoff timestamp (any Date-compatible type).
 * @property {string} [away_logo] - URL for away team logo.
 * @property {string} [home_logo] - URL for home team logo.
 * @property {string} [away_color] - Primary color for away team (hex).
 * @property {string} [home_color] - Primary color for home team (hex).
 * @property {string} [away_color2] - Secondary color for away team (hex).
 * @property {string} [home_color2] - Secondary color for home team (hex).
 * @property {string} [away_wordmark] - Wordmark URL for away team.
 * @property {string} [home_wordmark] - Wordmark URL for home team.
 */

/**
 * @typedef {Object} Prediction
 * @property {number} [home_win_probability] - 0-1 probability home team wins.
 * @property {number} [away_win_probability] - 0-1 probability away team wins.
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
  onReset,
}) {
  if (!matchup) return null;

  const awayTeam = matchup.away_team ?? '';
  const homeTeam = matchup.home_team ?? '';
  const awayFullName = matchup.away_name ?? NFL_TEAMS_MAP[awayTeam] ?? awayTeam;
  const homeFullName = matchup.home_name ?? NFL_TEAMS_MAP[homeTeam] ?? homeTeam;
  const awayLogo = matchup.away_logo;
  const homeLogo = matchup.home_logo;
  const awayColor = matchup.away_color;
  const homeColor = matchup.home_color;
  const awayColor2 = matchup.away_color2;
  const homeColor2 = matchup.home_color2;
  const awayWordmark = matchup.away_wordmark;
  const homeWordmark = matchup.home_wordmark;

  const kickoff = matchup.kickoff ? new Date(matchup.kickoff) : null;
  const kickoffDisplayTime = getKickoffDisplayTime(matchup.kickoff);
  const kickoffDateTime =
    kickoff && !Number.isNaN(kickoff.getTime()) ? kickoff.toISOString() : undefined;

  // Local debug state for a brief visual click cue
  const [debugClicked, setDebugClicked] = React.useState(false);

  const { hasScoreDetails, classifierUsed, isExpert, maxConfidence, sim, homeScore, awayScore } =
    derivePredictionMeta(prediction);
  const hasPrediction =
    Boolean(prediction) &&
    (hasScoreDetails ||
      prediction?.home_win_probability != null ||
      prediction?.away_win_probability != null);

  const cardClassName = buildCardClassNames({ hasPrediction, loading, error, debugClicked });
  const cardStyle = {
    "--i": index,
    ...(awayColor ? { "--away-color": awayColor } : {}),
    ...(homeColor ? { "--home-color": homeColor } : {}),
    ...(awayColor2 ? { "--away-color-2": awayColor2 } : {}),
    ...(homeColor2 ? { "--home-color-2": homeColor2 } : {}),
  };
  const awayTeamStyle = awayWordmark ? { "--team-wordmark": `url(${awayWordmark})` } : undefined;
  const homeTeamStyle = homeWordmark ? { "--team-wordmark": `url(${homeWordmark})` } : undefined;

  /**
   * Main click handler wrapper.
   * - Adds debug instrumentation.
   * - Guards against handler exceptions.
   * - Preserves original semantics: call onClick() with no event arg.
   */
  const handleArticleClick = () => {
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

      if (typeof onClick === 'function') {
        onClick();
      }
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
      onClick();
    }
  };

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
    // Reset is distinct from "open" - no fallback to onClick.
  };

  return (
    <article
      className={cardClassName + (loading ? 'game-card--loading' : '') + (error ? 'game-card--error' : '')}
      // @ts-ignore custom property used by some animations
      style={cardStyle}
      onClick={handleArticleClick}
      onKeyDown={onClick ? handleKeyDown : undefined}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : -1}
      aria-busy={loading ? 'true' : undefined}
    >
      <header className="game-card__header">
        <span className="game-card__week">Week {matchup.week ?? '-'}</span>
        <time className="game-card__kickoff" dateTime={kickoff?.toISOString?.()}>
          {kickoffDisplayTime}
        </time>
      </header>

      {/* Card content lives in a single inner wrapper so overlays + progress bars can be positioned safely */}
      <div className={styles.cardInner}>
        {/* Header: optional title/status + kickoff time */}
        <header className={styles.head}>
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

          <div className={styles.kickoffRow}>
            <time
              className={styles.kickoffTime}
              dateTime={kickoffDateTime}
            >
              {kickoffDisplayTime}
            </time>
          </div>
        </header>

        {/* Matchup row: logos + full team names */}
        <div className={styles.teamRow}>
          <div
            id="away"
            className={`${styles.gameCard} game-card__team game-card__team--away`}
            style={awayTeamStyle}
          >
            {awayLogo && (
              <img
                src={awayLogo}
                alt={`${awayTeam} logo`}
                id="away-logo"
                className={styles.gameCardLogo}
                loading="lazy"
              />
            )}

            <span className={`${styles.gameCardLabel} game-card__label`}>Away</span>
            <span className={`${styles.gameCardName} game-card__name`}>{awayFullName}</span>
          </div>

          <div className={`${styles.vs} game-card__vs`}>VS</div>

          <div
            className={`${styles.gameCard} game-card__team game-card__team--home`}
            style={homeTeamStyle}
          >
            {homeLogo && (
              <img
                src={homeLogo}
                alt={`${homeTeam} logo`}
                id="home-logo"
                className={styles.gameCardLogo}
                loading="lazy"
              />
            )}

            <span className={`${styles.gameCardLabel} game-card__label`}>Home</span>
            <span className={`${styles.gameCardName} game-card__name`}>{homeFullName}</span>
          </div>
        </div>

        {/* Prediction body / footer */}
        <footer className={`${styles.gameCardFooter} game-card__footer`}>
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
                  {isExpert ? (
                    <span className={`${styles.badge} ${styles.expertBadge}`}>
                      Ensemble Mixture (ML + MC)
                    </span>
                  ) : (
                    <span className={styles.badge}>
                      {classifierUsed ? 'Joblib Classifier' : 'Logistic fallback'}
                    </span>
                  )}
                  {prediction?.prediction_source && (
                    <span className={styles.badge}>
                      Source {prediction.prediction_source}
                    </span>
                  )}
                  {typeof maxConfidence === 'number' && (
                    <span className={styles.badge}>
                      Confidence {maxConfidence}%
                    </span>
                  )}
                </div>

                {/* Optional numeric details if present */}
                <div className={styles.predScore}>
                  <div className={styles.scoreRow}>
                    <span>Predicted Score:</span>
                    <b>
                      {awayTeam} {Math.round(awayScore ?? 0)} -{' '}
                      {homeTeam} {Math.round(homeScore ?? 0)}
                    </b>
                  </div>
                  {isExpert && (
                    <div className={styles.expertRange}>
                      <span>Range:</span>
                      <em>
                        {Math.round(sim.sim_away_score - 1.28 * sim.sim_away_sd)}-
                        {Math.round(sim.sim_away_score + 1.28 * sim.sim_away_sd)}
                      </em>
                      <span> vs </span>
                      <em>
                        {Math.round(sim.sim_home_score - 1.28 * sim.sim_home_sd)}-
                        {Math.round(sim.sim_home_score + 1.28 * sim.sim_home_sd)}
                      </em>
                    </div>
                  )}
                </div>
              </div>
            </>
          ) : (
            <span className="game-card__prediction game-card__prediction--empty">
              Predictions not available yet
            </span>
          )}
        </footer>
      </div>

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
