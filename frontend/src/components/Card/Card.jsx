// File: frontend/src/components/Card/Card.jsx
// Presentational card for a single matchup (rendered inside TeamGrid).

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
  matchup,
  prediction,
  loading = false,
  error,
  index = 0,
  onClick,
  onReset,
}) {
  if (!matchup) return null;

  const awayTeam = matchup.away_team ?? '';
  const homeTeam = matchup.home_team ?? '';
  const awayName = NFL_TEAMS_MAP[awayTeam] ?? awayTeam;
  const homeName = NFL_TEAMS_MAP[homeTeam] ?? homeTeam;

  const kickoff = matchup.kickoff ? new Date(matchup.kickoff) : null;
  const kickoffLabel = kickoff && !Number.isNaN(kickoff.getTime()) ? kickoff.toLocaleString() : 'TBD';

  const hasPrediction = Boolean(prediction);
  const homeWinProb = typeof prediction?.home_win_probability === 'number' ? prediction.home_win_probability : null;
  const awayWinProb = typeof prediction?.away_win_probability === 'number' ? prediction.away_win_probability : null;
  const homeWinPct = homeWinProb != null ? Math.round(homeWinProb * 100) : null;
  const awayWinPct = awayWinProb != null ? Math.round(awayWinProb * 100) : null;

  const homeScore = typeof prediction?.home_score === 'number' ? Math.round(prediction.home_score) : null;
  const awayScore = typeof prediction?.away_score === 'number' ? Math.round(prediction.away_score) : null;
  const diff = typeof prediction?.point_diff === 'number' ? prediction.point_diff : null;
  const modelLabel = prediction?.win_classifier_used ? 'Classifier' : 'Logistic fallback';

  const handleKeyDown = (event) => {
    if (!onClick) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onClick();
    }
  };

  const handleReset = (event) => {
    event?.stopPropagation?.();
    onReset?.();
  };

  return (
    <article
      className={[
        'game-card',
        loading ? 'game-card--loading' : '',
        error ? 'game-card--error' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      // @ts-ignore custom property used by some animations
      style={{ '--i': index }}
      onClick={onClick}
      onKeyDown={onClick ? handleKeyDown : undefined}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : -1}
      aria-busy={loading ? 'true' : undefined}
    >
      <header className="game-card__header">
        <span className="game-card__week">Week {matchup.week ?? '—'}</span>
        <time className="game-card__kickoff" dateTime={kickoff?.toISOString?.()}>
          {kickoffLabel}
        </time>
      </header>

      <div className="game-card__teams">
        <div className="game-card__team game-card__team--away">
          {matchup.away_logo && (
            <img
              className="game-card__logo"
              src={matchup.away_logo}
              alt={`${awayTeam} logo`}
              loading="lazy"
            />
          )}
          <span className="game-card__label">Away</span>
          <span className="game-card__name">{awayName}</span>
        </div>

        <div className="game-card__vs">@</div>

        <div className="game-card__team game-card__team--home">
          {matchup.home_logo && (
            <img
              className="game-card__logo"
              src={matchup.home_logo}
              alt={`${homeTeam} logo`}
              loading="lazy"
            />
          )}
          <span className="game-card__label">Home</span>
          <span className="game-card__name">{homeName}</span>
        </div>
      </div>

      <footer className="game-card__footer">
        {error && (
          <div className="game-card__error" role="alert">
            {error}
          </div>
        )}

        {loading ? (
          <span className="game-card__prediction game-card__prediction--loading">
            Fetching prediction...
          </span>
        ) : hasPrediction ? (
          <>
            {typeof onReset === 'function' && (
              <button type="button" className="game-card__reset" onClick={handleReset}>
                Reset
              </button>
            )}

            <div className="game-card__winprob">
              <span className="game-card__winprob-label">Win Probability ({modelLabel})</span>
              <span className="game-card__winprob-value">
                {homeWinPct != null && awayWinPct != null
                  ? `${homeTeam} ${homeWinPct}% • ${awayTeam} ${awayWinPct}%`
                  : 'n/a'}
              </span>
              {homeWinPct != null && (
                <div className="game-card__winprob-bar" aria-hidden="true">
                  <div className="game-card__winprob-fill" style={{ width: `${homeWinPct}%` }} />
                </div>
              )}
            </div>

            <div className="game-card__prediction">
              <span className="game-card__prediction-label">Predicted Score</span>
              <div className="game-card__prediction-row">
                <span className="game-card__prediction-team">
                  {awayTeam} {awayScore ?? '-'}
                </span>
                <span className="game-card__prediction-team">
                  {homeTeam} {homeScore ?? '-'}
                </span>
              </div>
              {diff != null && (
                <div className="game-card__prediction-row">
                  <span className="game-card__prediction-team">Diff</span>
                  <span className="game-card__prediction-team">{diff.toFixed?.(1) ?? diff}</span>
                </div>
              )}
            </div>
          </>
        ) : (
          <span className="game-card__prediction game-card__prediction--empty">
            Click to predict
          </span>
        )}
      </footer>
    </article>
  );
}
