// File: frontend/src/components/Card/Card.jsx
// Purpose: Presentational card for a single NFL matchup with probabilities and optional score details.
// Functions: Card(65)
// Variables: formatProbabilityAsPercentage(88)
// Interacts With: TeamGrid, PredictionContext data, backend /predict responses.
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
 * Card v2 - Prop-driven & motion-aware
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
export default function Card( {
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
} )
{
  React.useDebugValue( 'Card' );

  if ( !matchup ) return null;

  const { away_team, home_team, kickoff, away_logo, home_logo } = matchup;
  const hasPrediction = !!prediction;

  /**
   * @param {number | null | undefined} probabilityValue
   * @returns {number | null}
   */
  const formatProbabilityAsPercentage = ( probabilityValue ) =>
    typeof probabilityValue === 'number' && isFinite( probabilityValue )
      ? Math.round( probabilityValue * 100 )
      : null;

  const cardClassNames = [
    styles.card,
    hasPrediction ? styles.hasPrediction : '',
    loading ? styles.isLoading : '',
    error ? styles.isError : '',
  ]
    .filter( Boolean )
    .join( ' ' );

  const kickoffDisplayTime = kickoff ? new Date( kickoff ).toLocaleString() : 'TBD';
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

  /** @param {any} event */
  const handleKeyDown = ( event ) =>
  {
    if ( !onClick ) return;
    if ( event.key === 'Enter' || event.key === ' ' ) {
      event.preventDefault();
      // Use the article click wrapper so keyboard and pointer events share the same instrumentation.
      handleArticleClick?.();
    }
  };

  // Local debug state for visual click feedback during dev-only troubleshooting.
  const [ debugClicked, setDebugClicked ] = React.useState( false );

  /**
   * Wrapper for the provided onClick prop that adds lightweight
   * debug instrumentation and guards against handler exceptions.
   */
  const handleArticleClick = ( event ) =>
  {
    if ( typeof onClick !== 'function' ) return;
    try {
      // Provide a small visual cue so developers can see the click registered
      // even when the console is not open. This state auto-clears after 700ms.
      setDebugClicked( true );
      setTimeout( () => setDebugClicked( false ), 700 );
      // eslint-disable-next-line no-console
      console.debug( '[Card] article clicked', { matchup } );
      // Dev-only observable: dispatch a window event and persist a tiny
      // marker in localStorage so devs can verify a click even when the
      // console is closed or network is being proxied.
      try {
        if ( typeof window !== 'undefined' ) {
          // Dispatch a custom event for UI-driven smoke tests / puppeteer.
          try {
            // Guard import.meta usage in case tests or older runtimes don't provide it.
            let isDev = false;
            try {
              // import.meta is available in ESM (Vite) environments; guard access in case
              // tests or legacy runtimes do not expose it.
              // eslint-disable-next-line no-undef
              isDev = !!( import.meta && import.meta.env && import.meta.env.DEV );
            } catch ( _e ) { isDev = false; }
            if ( isDev ) {
              try { window.dispatchEvent( new CustomEvent( 'nfl-card-click', { detail: { matchup } } ) ); } catch ( _e ) { /* ignore */ }
            }

            // Always write a short localStorage key so manual inspection can
            // confirm the click without opening DevTools -> Console.
            try {
              localStorage.setItem( 'nfl_last_card_click', JSON.stringify( { game_id: matchup?.game_id || null, ts: new Date().toISOString() } ) );
            } catch ( _e ) { /* ignore localStorage failures */ }
          } catch ( _e ) { /* defensive */ }
        }
      } catch ( _err ) { /* noop */ }
      // Keep previous semantics: call the passed handler without the raw DOM event
      // to match callers that expect a simple function (e.g., TeamGrid.handleClick)
      onClick();
    } catch ( err ) {
      // eslint-disable-next-line no-console
      console.error( '[Card] onClick handler threw', err );
    }
  };

  return (
    <article
      className={ [ cardClassNames, debugClicked ? styles.debugClicked : '' ].filter( Boolean ).join( ' ' ) }
      // Custom CSS variable used by Card.module.css for staggered animations.
      // @ts-ignore - allow custom property name in inline style.
      style={ { '--i': index } }
      aria-pressed={ loading ? 'true' : 'false' }
      onClick={ handleArticleClick }
      onKeyDown={ onClick ? handleKeyDown : undefined }
      role={ onClick ? 'button' : undefined }
      tabIndex={ onClick ? 0 : -1 }
    >
      {/* Loading indicator overlay (small spinner) */ }
      { loading && (
        <div className={ styles.loadingIndicator } aria-hidden="true">
          <div className={ styles.spinner } />
        </div>
      ) }
      {/* Top bar: optional icon/title/status */ }
      { shouldShowTopBar && (
        <div className={ styles.topBar }>
          <div className={ styles.left }>
            { icon && (
              <span className={ styles.icon } aria-hidden>
                { icon }
              </span>
            ) }
            { title && <strong className={ styles.title }>{ title }</strong> }
          </div>
          { status && <span className={ styles.status }>{ status }</span> }
        </div>
      ) }

      {/* Matchup row */ }
      <div className={ styles.matchupRow }>
        <div className="game-card__team game-card__team--away">
          { away_logo && (
            <img
              src={ away_logo }
              alt={ `${away_team} logo` }
              className="game-card__logo"
              loading="lazy"
            />
          ) }
          <span className="game-card__label">Away</span>
          <span className="game-card__name">{ away_team }</span>
        </div>

        <div className="game-card__vs">@</div>

        <div className="game-card__team game-card__team--home">
          { home_logo && (
            <img
              src={ home_logo }
              alt={ `${home_team} logo` }
              className="game-card__logo"
              loading="lazy"
            />
          ) }
          <span className="game-card__label">Home</span>
          <span className="game-card__name">{ home_team }</span>
        </div>
      </div>

      <div className={ styles.meta }>
        <time
          className={ styles.kickoff }
          dateTime={ kickoff ? new Date( kickoff ).toISOString() : undefined }
        >
          { kickoffDisplayTime }
        </time>
      </div>

      <footer className="game-card__footer">
        { loading ? (
          <span className="game-card__prediction game-card__prediction--loading">
            Fetching prediction...
          </span>
        ) : hasPrediction ? (
          <div className={ styles.predictionBody }>
            <div className={ styles.probRow }>
              <span>Home</span>
              <b>
                { formatProbabilityAsPercentage(
                  prediction.home_win_probability
                ) }
                %
              </b>
            </div>
            <div className={ styles.probRow }>
              <span>Away</span>
              <b>
                { formatProbabilityAsPercentage(
                  prediction.away_win_probability
                ) }
                %
              </b>
            </div>
            <div className={ styles.badgeRow }>
              <span className={ styles.badge }>
                { classifierUsed ? "Classifier" : "Logistic fallback" }
              </span>
              { typeof maxConfidence === "number" && (
                <span className={ styles.badge }>
                  Confidence { maxConfidence }%
                </span>
              ) }
            </div>
            {/* Optional numeric details if present */ }
            { hasScoreDetails && (
              <div className={ styles.detailRow }>
                <span>Score</span>
                <b>
                  { away_team } { prediction.away_score ?? '-' } -{ ' ' }
                  { prediction.home_score ?? '-' } { home_team }
                  { prediction.point_diff != null && (
                    <em className={ styles.diff }>
                      { ' ' }
                      diff{ ' ' }
                      { typeof prediction.point_diff === 'number' &&
                        typeof prediction.point_diff.toFixed === 'function'
                        ? prediction.point_diff.toFixed( 1 )
                        : prediction.point_diff }
                    </em>
                  ) }
                </b>
              </div>
            ) }
          </div>
        ) : (
          <span className="game-card__prediction game-card__prediction--empty">
            Predictions not available yet
          </span>
        ) }
      </footer>

      {/* Optional progress meter */ }
      { typeof progress === 'number' && isFinite( progress ) && (
        <div className={ styles.progressTrack } aria-hidden>
          <div
            className={ styles.progressBar }
            style={ { width: `${Math.max( 0, Math.min( 100, progress ) )}%` } }
          />
        </div>
      ) }
    </article>
  );
}
