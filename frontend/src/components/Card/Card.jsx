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
import { getPremiumExplanation } from '../../api/client.js';

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
  LA: 'Los Angeles Rams',
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
  SD: 'Los Angeles Chargers',
  SF: 'San Francisco 49ers',
  SEA: 'Seattle Seahawks',
  STL: 'Los Angeles Rams',
  TB: 'Tampa Bay Buccaneers',
  TEN: 'Tennessee Titans',
  WAS: 'Washington Commanders',
  WSH: 'Washington Commanders',
};

const TEAM_CODE_ALIASES = {
  LA: 'LAR',
  STL: 'LAR',
  SD: 'LAC',
  OAK: 'LV',
  WSH: 'WAS',
  COMMANDERS: 'WAS',
  RAMS: 'LAR',
  RAIDERS: 'LV',
  CHARGERS: 'LAC',
  JAGUARS: 'JAX',
};

const NFLVERSE_ASSET_ROOT = 'https://raw.githubusercontent.com/nflverse/nflverse-pbp/master';
const GENERIC_LEAGUE_LOGO = `${NFLVERSE_ASSET_ROOT}/NFL.png`;
const SPECIAL_LOGO_FALLBACKS = {
  AFC: `${NFLVERSE_ASSET_ROOT}/AFC.png`,
  NFC: `${NFLVERSE_ASSET_ROOT}/NFC.png`,
  NFL: GENERIC_LEAGUE_LOGO,
};

const normalizeLogoTeamCode = (value) => {
  const raw = (value ?? '').toString().trim().toUpperCase();
  if (!raw) return '';
  return TEAM_CODE_ALIASES[raw] ?? raw;
};

const buildTeamLogoCandidates = (teamCode, explicitLogo) => {
  const normalizedCode = normalizeLogoTeamCode(teamCode);
  const candidates = [explicitLogo];

  if (normalizedCode && SPECIAL_LOGO_FALLBACKS[normalizedCode]) {
    candidates.push(SPECIAL_LOGO_FALLBACKS[normalizedCode]);
  } else if (normalizedCode && /^[A-Z]{2,3}$/.test(normalizedCode)) {
    candidates.push(`${NFLVERSE_ASSET_ROOT}/squared_logos/${normalizedCode}.png`);
  }

  candidates.push(GENERIC_LEAGUE_LOGO);

  return Array.from(new Set(candidates.filter(Boolean)));
};

const buildTeamStyleVars = (wordmark, primaryColor, secondaryColor) => ({
  ...(wordmark ? { '--team-wordmark': `url(${wordmark})` } : {}),
  ...(primaryColor ? { '--team-primary': primaryColor } : {}),
  ...((secondaryColor || primaryColor)
    ? { '--team-secondary': secondaryColor || primaryColor }
    : {}),
});

const getTeamBadgeText = (teamCode, teamName) => {
  const normalizedCode = normalizeLogoTeamCode(teamCode);
  if (normalizedCode) return normalizedCode;

  const initials = (teamName ?? '')
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part[0])
    .join('')
    .slice(0, 3)
    .toUpperCase();

  return initials || 'NFL';
};

function TeamLogoBadge({ teamCode, teamName, logoSrc }) {
  const logoCandidates = React.useMemo(
    () => buildTeamLogoCandidates(teamCode, logoSrc),
    [teamCode, logoSrc]
  );
  const [candidateIndex, setCandidateIndex] = React.useState(0);

  React.useEffect(() => {
    setCandidateIndex(0);
  }, [logoCandidates]);

  const activeLogo = logoCandidates[candidateIndex];
  const badgeText = getTeamBadgeText(teamCode, teamName);
  const accessibleTeamLabel = teamName || teamCode || 'NFL team';

  if (!activeLogo) {
    return (
      <div className={styles.logoFrame} aria-hidden="true">
        <span className={styles.logoFallback}>{badgeText}</span>
      </div>
    );
  }

  return (
    <div className={styles.logoFrame}>
      <img
        src={activeLogo}
        alt={`${accessibleTeamLabel} logo`}
        className={styles.gameCardLogo}
        loading="lazy"
        onError={() => setCandidateIndex((current) => current + 1)}
      />
    </div>
  );
}

/**
 * Normalize a 0-1 probability into a rounded 0-100 integer.
 * Returns `null` when the input is missing or invalid.
 */
const formatProbabilityAsPercentage = (probabilityValue) =>
  typeof probabilityValue === 'number' && isFinite(probabilityValue)
    ? Math.round(probabilityValue * 100)
    : null;

const extractPremiumErrorMessage = (error) => {
  const detail = error?.body?.detail;
  if (typeof detail === 'string') return detail;
  if (detail?.message) return detail.message;
  if (error?.body?.error?.message) return error.body.error.message;
  return error?.message || 'Failed to load premium breakdown';
};

const getExpertReasoningText = (prediction) => {
  const reasoning =
    prediction?.expert_reasoning ??
    prediction?.expert_prediction?.reasoning ??
    prediction?.expert_prediction?.reasoning_sentences;

  if (Array.isArray(reasoning)) {
    return reasoning.map((line) => String(line).trim()).filter(Boolean).join(' ');
  }

  return typeof reasoning === 'string' ? reasoning.trim() : '';
};

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
  const homePct = formatProbabilityAsPercentage(homeProb);
  const awayPct = formatProbabilityAsPercentage(awayProb);

  const maxConfidence =
    typeof homeProb === 'number' || typeof awayProb === 'number'
      ? formatProbabilityAsPercentage(Math.max(homeProb ?? 0, awayProb ?? 0))
      : null;

  return {
    hasScoreDetails,
    classifierUsed,
    isExpert,
    maxConfidence,
    sim,
    homeScore,
    awayScore,
    homePct,
    awayPct,
  };
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
 * @property {Object} [actualScore] - Optional final result ({home_score, away_score, status}) returned from scoreboard sync.
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
  actualScore,
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

  // Premium AI breakdown state
  const [premiumExplain, setPremiumExplain] = React.useState(null);
  const [loadingPremium, setLoadingPremium] = React.useState(false);
  const [premiumError, setPremiumError] = React.useState(null);
  const [showPremiumExplain, setShowPremiumExplain] = React.useState(false);
  const expertReasoningText = React.useMemo(
    () => getExpertReasoningText(prediction),
    [prediction]
  );
  const expertLayerUsed = prediction?.expert_prediction?.used_llm === true;
  const premiumDisplayText = premiumExplain || expertReasoningText;

  React.useEffect(() => {
    setPremiumExplain(null);
    setPremiumError(null);
    setShowPremiumExplain(Boolean(expertReasoningText));
  }, [
    expertReasoningText,
    prediction?.game_id,
    prediction?.prediction_source,
    homeTeam,
    awayTeam,
  ]);

  const { hasScoreDetails, classifierUsed, isExpert, maxConfidence, sim, homeScore, awayScore, homePct, awayPct } =
    derivePredictionMeta(prediction);
  const hasPrediction =
    Boolean(prediction) &&
    (hasScoreDetails ||
      prediction?.home_win_probability != null ||
      prediction?.away_win_probability != null);
  const predictedDiff =
    hasPrediction && typeof homeScore === "number" && typeof awayScore === "number"
      ? Math.round(homeScore - awayScore)
      : null;
  const actualHasScore =
    actualScore?.home_score != null && actualScore?.away_score != null;
  const actualDiff =
    actualHasScore && typeof actualScore.home_score === "number" && typeof actualScore.away_score === "number"
      ? Math.round(actualScore.home_score - actualScore.away_score)
      : null;
  const finalDelta =
    actualDiff != null && predictedDiff != null ? actualDiff - predictedDiff : null;
  const hasProbabilityMeter = Number.isFinite(homePct) && Number.isFinite(awayPct);
  const favoriteTeamCode = hasProbabilityMeter && homePct >= awayPct ? homeTeam : awayTeam;
  const favoriteTeamName = hasProbabilityMeter && homePct >= awayPct ? homeFullName : awayFullName;
  const confidenceSummary = hasProbabilityMeter ? Math.max(homePct, awayPct) : null;

  const cardClassName = buildCardClassNames({ hasPrediction, loading, error, debugClicked });
  const cardStyle = {
    "--i": index,
    ...(awayColor ? { "--away-color": awayColor } : {}),
    ...(homeColor ? { "--home-color": homeColor } : {}),
    ...(awayColor2 ? { "--away-color-2": awayColor2 } : {}),
    ...(homeColor2 ? { "--home-color-2": homeColor2 } : {}),
  };
  const awayTeamStyle = buildTeamStyleVars(awayWordmark, awayColor, awayColor2);
  const homeTeamStyle = buildTeamStyleVars(homeWordmark, homeColor, homeColor2);

  const handlePremiumExplain = async (event) => {
    try {
      event?.stopPropagation?.();
      if (loadingPremium) return;
      if (expertReasoningText) {
        setPremiumError(null);
        setShowPremiumExplain((current) => !current);
        return;
      }
      if (premiumExplain) {
        setShowPremiumExplain(!showPremiumExplain);
        return;
      }

      setLoadingPremium(true);
      setPremiumError(null);
      const res = await getPremiumExplanation({
        home_team: homeTeam,
        away_team: awayTeam,
        season: matchup.season || 2025,
        week: matchup.week || 1,
      });
      setPremiumExplain(String(res?.reply || 'Premium AI did not return an explanation.'));
      setShowPremiumExplain(true);
    } catch (err) {
      console.error("[Card] premiumExplain failed", err);
      setPremiumError(extractPremiumErrorMessage(err));
    } finally {
      setLoadingPremium(false);
    }
  };

  /**
   * Main click handler wrapper.
   * - Adds debug instrumentation.
   * - Guards against handler exceptions.
   * - Preserves original semantics: call onClick() with no event arg.
   */
  const handlePredictAction = (event) => {
    try {
      event?.stopPropagation?.();
      // Brief visual click cue for dev feedback
      setDebugClicked(true);
      setTimeout(() => setDebugClicked(false), 700);

      if (typeof onClick === 'function') {
        onClick();
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('[Card] onClick handler threw', err);
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
      aria-busy={loading ? 'true' : undefined}
    >


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
            <TeamLogoBadge teamCode={awayTeam} teamName={awayFullName} logoSrc={awayLogo} />

            <div className={styles.teamMetaRow}>
              <span className={`${styles.gameCardLabel} game-card__label`}>Away</span>
              <span className={styles.teamCodePill}>{normalizeLogoTeamCode(awayTeam) || awayTeam}</span>
            </div>
            <span className={`${styles.gameCardName} game-card__name`}>{awayFullName}</span>
          </div>

          <div className={`${styles.vs} game-card__vs`}>VS</div>

          <div
            className={`${styles.gameCard} game-card__team game-card__team--home`}
            style={homeTeamStyle}
          >
            <TeamLogoBadge teamCode={homeTeam} teamName={homeFullName} logoSrc={homeLogo} />

            <div className={styles.teamMetaRow}>
              <span className={`${styles.gameCardLabel} game-card__label`}>Home</span>
              <span className={styles.teamCodePill}>{normalizeLogoTeamCode(homeTeam) || homeTeam}</span>
            </div>
            <span className={`${styles.gameCardName} game-card__name`}>{homeFullName}</span>
          </div>
        </div>

        {/* Prediction body / footer */}
        <footer className={`${styles.gameCardFooter} game-card__footer`}>
          {loading ? (
            <div className={styles.actionPanel} role="status" aria-live="polite">
              <span className="game-card__prediction--loading">
                <span className={styles.spinner} aria-label="Loading" /> Running model pipeline...
              </span>
              <span className={styles.actionHint}>Preparing score, winner probability, and confidence signals.</span>
            </div>
          ) : error ? (
            <div className={`${styles.actionPanel} ${styles.errorPanel}`} role="alert">
              <strong>Prediction unavailable</strong>
              <span>{error}</span>
              <button
                type="button"
                className={styles.predictButton}
                onClick={handlePredictAction}
                disabled={typeof onClick !== 'function'}
              >
                Try again
              </button>
            </div>
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
                  {expertLayerUsed ? (
                    <span className={`${styles.badge} ${styles.expertBadge}`}>
                      Gemma Cloud Expert Layer
                    </span>
                  ) : isExpert ? (
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

                {hasProbabilityMeter && (
                  <div className={styles.probabilityPanel}>
                    <div className={styles.probabilityHeader}>
                      <span>Win edge</span>
                      <strong>
                        {favoriteTeamCode} {confidenceSummary}%
                      </strong>
                    </div>
                    <div className={styles.probabilityTrack} aria-hidden="true">
                      <div
                        className={`${styles.probabilityFill} ${styles.probabilityFillAway}`}
                        style={{ width: `${awayPct}%` }}
                      />
                      <div
                        className={`${styles.probabilityFill} ${styles.probabilityFillHome}`}
                        style={{ width: `${homePct}%` }}
                      />
                    </div>
                    <div className={styles.probabilityScale}>
                      <span>{awayTeam} {awayPct}%</span>
                      <span>{favoriteTeamName}</span>
                      <span>{homeTeam} {homePct}%</span>
                    </div>
                  </div>
                )}

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

                {/* Premium AI Breakdown Section */}
                <div className={styles.premiumSection}>
                  <button
                    type="button"
                    className={styles.premiumButton}
                    onClick={handlePremiumExplain}
                    disabled={loadingPremium}
                  >
                    {loadingPremium ? (
                      <>
                        <span className={styles.premiumSpinner} />
                        ✨ Analyzing Matchup...
                      </>
                    ) : (
                      <>
                        ✨ {showPremiumExplain ? 'Hide Premium AI Breakdown' : 'View Premium AI Breakdown'}
                      </>
                    )}
                  </button>

                  {premiumError && (
                    <div className={styles.premiumError}>
                      ⚠️ {premiumError}
                    </div>
                  )}

                  {showPremiumExplain && premiumDisplayText && (
                    <div className={styles.premiumContent}>
                      <div className={styles.premiumContentHeader}>
                        <span>{expertReasoningText ? 'Gemma Cloud Expert Reasoning' : '🤖 Premium AI Analyst Report'}</span>
                      </div>
                      <div className={styles.premiumText}>
                        {premiumDisplayText.split('\n').map((line, idx) => {
                          const trimmed = line.trim();
                          if (trimmed.startsWith('###')) {
                            return <h4 key={idx} className={styles.premiumH4}>{trimmed.replace('###', '')}</h4>;
                          } else if (trimmed.startsWith('##')) {
                            return <h3 key={idx} className={styles.premiumH3}>{trimmed.replace('##', '')}</h3>;
                          } else if (trimmed.startsWith('*') || trimmed.startsWith('-')) {
                            return <li key={idx} className={styles.premiumLi}>{trimmed.substring(1).trim()}</li>;
                          } else if (trimmed) {
                            return <p key={idx} className={styles.premiumP}>{trimmed}</p>;
                          }
                          return <div key={idx} style={{ height: '6px' }} />;
                        })}
                      </div>
                    </div>
                  )}
                </div>
                {actualHasScore && (
                  <div className={styles.actualRow}>
                    <span className={styles.actualLabel}>Final</span>
                    <strong>
                      {actualScore.home_score} - {actualScore.away_score}
                    </strong>
                    {actualScore.status && (
                      <span className={styles.actualStatus}>{actualScore.status}</span>
                    )}
                    {finalDelta != null && (
                      <span className={styles.actualDiff}>
                        Diff vs prediction {finalDelta >= 0 ? "+" : ""}
                        {finalDelta}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className={styles.actionPanel}>
              <div>
                <strong>Ready to forecast</strong>
                <span>Generate projected score, win edge, and model confidence for this matchup.</span>
              </div>
              <button
                type="button"
                className={styles.predictButton}
                onClick={handlePredictAction}
                disabled={typeof onClick !== 'function'}
              >
                Generate prediction
              </button>
            </div>
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
