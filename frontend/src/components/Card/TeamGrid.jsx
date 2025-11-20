// File: frontend/src/components/Card/TeamGrid.jsx
// Purpose: Render weekly matchup grid and surface predictions/loading/errors for each game.
// Functions: toGameKey(25), TeamGrid(47)
// Variables: none
// Interacts With: Card component, Dashboard onPredict handler, PredictionContext state.
// TeamGrid.jsx
// ------------------------------------------------------
// Grid layout for all games in a given NFL week.
// - Receives a `week` number and an array of `games`.
// - Renders a responsive CSS grid of <Card /> components.
// - Keeps empty / loading states friendly and clear.
// ------------------------------------------------------

import React from 'react';
import Card from './Card.jsx';        // presentational card for a single matchup
import './TeamGrid.css';
import { predictGame } from '../../api/client.js';

/**
 * Build a stable key for a game, mirroring PredictionContext/generateGameKey
 * and StatsPage/toGameKey.
 *
 * @param {any} game
 * @returns {string}
 */
const toGameKey = ( game ) =>
  game?.game_id ?? [
    game?.season,
    game?.week,
    game?.home_abbr || game?.home_team,
    game?.away_abbr || game?.away_team,
  ]
    .filter( Boolean )
    .join( '-' );

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
function TeamGrid( {
  week = 10,
  games = [],
  isLoading = false,
  teams = {},
  predictions = {},
  loading = {},
  errors = {},
  onPredict,
} = {} )
{
  const safeWeek = week ?? ( games[ 0 ]?.week ?? 10 );

  // Local loading map used when this component invokes predictGame directly
  // (fallback when no parent onPredict handler is provided). We keep this
  // separate from the parent-provided `loading` prop so we don't collide
  // with external state management (Dashboard / PredictionContext).
  const [ localLoadingMap, setLocalLoadingMap ] = React.useState( {} );

  if ( isLoading ) {
    return (
      <section className="team-grid" aria-busy="true">
        <header className="team-grid__header">
          <h2 className="team-grid__title">Week { safeWeek } Games</h2>
          <p className="team-grid__subtitle">Loading schedule...</p>
        </header>
        <div className="team-grid__empty">
          <div className="team-grid__spinner" />
        </div>
      </section>
    );
  }

  if ( !games || games.length === 0 ) {
    return (
      <section className="team-grid">
        <header className="team-grid__header">
          <h2 className="team-grid__title">Week { safeWeek } Games</h2>
          <p className="team-grid__subtitle">
            No games found for Week { safeWeek }. Try refreshing or checking your API.
          </p>
        </header>
        <div className="team-grid__empty">
          <p className="team-grid__empty-text">
            Once the schedule loads, all Week { safeWeek } matchups will appear here.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section
      className="team-grid"
      aria-label={ `NFL Week ${safeWeek}` }
      data-week={ safeWeek }
    >
      <header className="team-grid__header">
        <div className="team-grid__heading">
          <span className="team-grid__badge">Week { safeWeek }</span>
          <h2 className="team-grid__title">Week { safeWeek }</h2>
        </div>
        <p className="team-grid__subtitle">
          Showing <strong>{ games.length }</strong> games scheduled.
        </p>
      </header>

      <div className="team-grid__grid">
        { games.map( ( game, index ) =>
        {
          const rawKey = toGameKey( game ) || String( index );

          // Look up any existing prediction + request state for this game.
          const prediction = predictions?.[ rawKey ];
          // Combine external loading state (from context) with any local
          // loading footprints generated when this component called
          // predictGame directly (fallback path).
          const isLocalLoading = Boolean( localLoadingMap?.[ rawKey ] );
          const isGameLoading = Boolean( loading?.[ rawKey ] ) || isLocalLoading;
          const errorMessage = errors?.[ rawKey ] ?? null;

          // Enrich the schedule row with team metadata (logos + pretty names) when available.
          // Normalize abbreviations (defensive) and prefer server-provided
          // logos when available; otherwise fall back to the local teams map.
          const homeAbbr = ( game.home_abbr || game.home_team || '' ).toString().trim().toUpperCase();
          const awayAbbr = ( game.away_abbr || game.away_team || '' ).toString().trim().toUpperCase();
          const homeMeta = homeAbbr && teams && teams[ homeAbbr ] ? teams[ homeAbbr ] : null;
          const awayMeta = awayAbbr && teams && teams[ awayAbbr ] ? teams[ awayAbbr ] : null;

          const enrichedGame = {
            ...game,
            home_logo: game.home_logo || homeMeta?.logoUrl || null,
            away_logo: game.away_logo || awayMeta?.logoUrl || null,
            home_abbr: homeAbbr,
            away_abbr: awayAbbr,
          };

          if ( prediction ) {
            enrichedGame.home_win = prediction.home_win === 1 ? homeAbbr : awayAbbr
            enrichedGame.confidence_score = prediction.confidence_score ?? null;
            enrichedGame.home_pred_score =
              prediction.home_score ?? prediction.home_score_pred ?? null;
            enrichedGame.away_pred_score =
              prediction.away_score ?? prediction.away_score_pred ?? null;
            enrichedGame.home_win_probability =
              prediction.home_win_probability ?? prediction.probs?.home ?? null;
          }

          const handleClick = async () =>
          {
            // Debug instrumentation: log clicks so we can confirm UI events reach TeamGrid.
            // Keep this lightweight — it will be removed once we verify behavior.
            try {
              // rawKey is available in this lexical scope.
              // Use console.debug to reduce noise in production consoles.
              // eslint-disable-next-line no-console
              console.debug( '[TeamGrid] card clicked', { rawKey, game, isGameLoading } );
            } catch ( _err ) { }
            // No-op while a prediction request for this game is in-flight
            if ( isGameLoading ) return;

            // Prefer an injected onPredict handler (from parent). If present
            // the Dashboard will handle the request/state update. Otherwise
            // fall back to calling the client helper directly.
            if ( typeof onPredict === 'function' ) {
              try {
                // Pass the original schedule/game row to the parent so it can
                // construct the canonical request payload and update context.
                onPredict( game );
              } catch ( err ) {
                console.error( '[TeamGrid] onPredict handler threw', err );
              }
              return;
            }

            // Fire-and-forget client fallback: construct the canonical payload
            // the backend expects and call the published api helper. While the
            // request is in-flight we set a local loading flag so the UI can
            // show a visual indicator on the card.
            const payload = {
              home_team: homeAbbr,
              away_team: awayAbbr,
              season: game?.season ?? game?.season_num ?? null,
              week: game?.week ?? game?.week_num ?? null,
            };

            try {
              setLocalLoadingMap( ( prev ) => ( { ...prev, [ rawKey ]: true } ) );
              await predictGame( payload );
            } catch ( err ) {
              console.error( '[TeamGrid] predictGame failed', err );
            } finally {
              setLocalLoadingMap( ( prev ) =>
              {
                const copy = { ...prev };
                delete copy[ rawKey ];
                return copy;
              } );
            }
          };

          return (
            <Card
              role='button'
              key={ rawKey }
              matchup={ enrichedGame }
              prediction={ prediction }
              loading={ isGameLoading }
              error={ errorMessage }
              index={ index }
              onClick={ handleClick }
            />
          );
        } ) }
      </div>
    </section>
  );
}

export default TeamGrid;
