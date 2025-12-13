// File: frontend/src/components/Card/TeamGrid.jsx
// Location: ./frontend/src/components/Card/TeamGrid.jsx
// ------------------------------------------------------------------------------------
// High-level purpose
// ------------------
// This file defines the "TeamGrid" React component, which renders a grid of NFL games
// for a single week and wires those games up to the prediction flow.
//
// It is responsible for:
// - Displaying a clean grid of <Card /> components (one per matchup).
// - Surfacing loading and error states per game.
// - Passing the correct game metadata and prediction values into each <Card />.
// - Triggering prediction requests when a card is clicked, either by delegating to a
//   parent onPredict handler or by calling the predictGame() API helper directly.
//
// Function index (line numbers are approximate and will shift as you edit the file):
// - toGameKey (≈45): builds a stable string key for a game row.
// - TeamGrid   (≈69): main React component that renders the weekly game grid.
//
// Data & dependencies
// -------------------
// - Props:
//   - week:        the NFL week number.
//   - games:       array of schedule rows from the backend.
//   - teams:       map of team abbreviations to metadata (name, logoUrl, etc.).
//   - predictions: map from game key -> prediction payload.
//   - loading:     map from game key -> boolean in-flight state.
//   - errors:      map from game key -> error message or object.
//   - onPredict:   optional callback invoked when a game card is clicked.
// - External modules:
//   - <Card />:   presentational component for a single matchup card.
//   - predictGame: API helper in ../../api/client.js (fallback when onPredict is absent).
// - Styling:
//   - TeamGrid.css defines the layout and visual treatment for the grid and states.
// ------------------------------------------------------------------------------------

import React from 'react';
import Card from './Card.jsx';        // Presentational card for a single matchup
import './TeamGrid.css';
import { predictGame } from '../../api/nfl.js';

/**
 * Helper: build a stable key for a game.
 *
 * Why this matters:
 * - React lists need a "key" prop that is stable between renders.
 * - We mirror the same strategy used in PredictionContext/StatsPage so that the
 *   same game shares a consistent identifier across the app.
 *
 * Pattern:
 * - Prefer an explicit game_id if the backend generated one.
 * - Otherwise, fall back to a composite key:
 *   "<season>-<week>-<HOME_ABBR>-<AWAY_ABBR>".
 *
 * @param {any} game - schedule or prediction object
 * @returns {string} - unique-ish identifier for a single game
 */
const toGameKey = (game) =>
  game?.game_id ??
  [
    game?.season,
    game?.week,
    game?.home_abbr || game?.home_team,
    game?.away_abbr || game?.away_team,
  ]
    // filter(Boolean) removes null/undefined/empty string so we do not
    // accidentally end up with "undefined" segments in the key.
    .filter(Boolean)
    .join('-');

/**
 * TeamGrid component
 * -------------------
 * Renders a section containing a header + responsive grid of matchup cards.
 *
 * Key patterns to notice:
 * - We use default values in the function parameter list so callers can omit props
 *   without causing runtime errors.
 * - All "lookup" state (predictions, loading, errors) is keyed by toGameKey().
 * - We keep a *local* loading map as a fallback when the parent does not supply
 *   its own onPredict handler.
 *
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
 *   onReset?: (game: any) => void,   // <— add this
 * }} [props]
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
  onReset,            // <— add this
} = {}) {
  // ...rest of file unchanged up to the map()

  {
    // "safeWeek":
    // - Prefer the explicit week prop.
    // - If not provided, try to infer from the first game.
    // - Fall back to NFL week 10 as a harmless default.
    const safeWeek = week ?? (games[0]?.week ?? 10);

    // Local loading map used when this component invokes predictGame directly
    // (fallback when no parent onPredict handler is provided).
    //
    // Pattern:
    // - The key is the same game key used in global state (toGameKey).
    // - The value is a boolean meaning "a request for this game is in-flight".
    //
    // This is separate from the parent-provided `loading` prop so we do not
    // accidentally overwrite or collide with external state management
    // (Dashboard / PredictionContext).
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
            <h2 className="team-grid__title">Matchups</h2>
          </div>
          <p className="team-grid__subtitle">
            Showing <strong>{games.length}</strong> games scheduled.
          </p>
        </header>

        <div className="team-grid__grid">
          {games.map((game, index) => {
            // Build a stable key for React and for lookup maps.
            const rawKey = toGameKey(game) || String(index);

            // Look up any existing prediction + request state for this game.
            const prediction = predictions?.[rawKey];

            // Combine external loading state (from context) with local fallback
            // state used when this component calls predictGame directly.
            const isLocalLoading = Boolean(localLoadingMap?.[rawKey]);
            const isGameLoading = Boolean(loading?.[rawKey]) || isLocalLoading;

            const errorMessage = errors?.[rawKey] ?? null;

            // Normalize team abbreviations:
            // - We accept either home_abbr or home_team (same for away).
            // - We defensively convert to string, trim whitespace, then uppercase.
            //   This keeps keys consistent with how teams are stored in the
            //   `teams` metadata map and in backend datasets.
            const homeAbbr = (game.home_abbr || game.home_team || '')
              .toString()
              .trim()
              .toUpperCase();
            const awayAbbr = (game.away_abbr || game.away_team || '')
              .toString()
              .trim()
              .toUpperCase();

            // Try to find optional team metadata (logo + pretty name).
            // Pattern:
            // - Nullish coalescing (??) avoids throwing when `teams[abbr]` is undefined.
            // - We explicitly default to null so consumers can test truthiness.
            const homeMeta = homeAbbr && teams && teams[homeAbbr] ? teams[homeAbbr] : null;
            const awayMeta = awayAbbr && teams && teams[awayAbbr] ? teams[awayAbbr] : null;

            // "enrichedGame" is the object actually passed down to <Card />.
            // We take the raw schedule row and overlay any UI-friendly fields:
            // - logos from either the backend or local `teams` map
            // - normalized abbreviations
            // - predicted scores / win probabilities (if available)
            const enrichedGame = {
              ...game,
              home_logo: game.home_logo || homeMeta?.logoUrl || null,
              away_logo: game.away_logo || awayMeta?.logoUrl || null,
              // Backwards/forwards-compatible kickoff field: some backends use
              // `kickoff`, `game_day`, or `gameday`. Normalize to `kickoff` so
              // Card.jsx can always read a single field.
              kickoff: game.kickoff || game.game_day || game.gameday || null,
              home_abbr: homeAbbr,
              away_abbr: awayAbbr,
            };

            // If we already have a prediction in props, attach it to the
            // enriched game so the Card can render the numbers and confidence.
            if (prediction) {
              enrichedGame.home_win =
                prediction.home_win === 1 ? homeAbbr : awayAbbr;

              enrichedGame.confidence_score = prediction.confidence_score ?? null;

              // Some backends expose `home_score`, others `home_score_pred`.
              // We normalize both cases into `home_pred_score`/`away_pred_score`.
              enrichedGame.home_pred_score =
                prediction.home_score ?? prediction.home_score_pred ?? null;
              enrichedGame.away_pred_score =
                prediction.away_score ?? prediction.away_score_pred ?? null;

              // Same idea for probabilities: prefer explicit "home_win_probability",
              // otherwise read from prediction.probs.home.
              enrichedGame.home_win_probability =
                prediction.home_win_probability ?? prediction.probs?.home ?? null;
            }

            /**
             * Click handler for a single card.
             *
             * This uses a "guard clause" pattern:
             * - If a request is already in-flight for this game, we bail early.
             * - Otherwise, we try the parent-supplied onPredict callback first.
             *   If that does not exist, we fall back to calling predictGame()
             *   directly and manage the local loading flag ourselves.
             */
            const handleClick = async () => {
              // Lightweight debug log so we can confirm clicks are reaching this
              console.debug('[TeamGrid] card clicked', { rawKey, game, isGameLoading });

              // No-op while a prediction request for this game is in-flight.
              if (isGameLoading) return;

              // Prefer an injected onPredict handler (from parent). If present
              // the Dashboard/PredictionContext will handle the request and all
              // state updates.
              if (typeof onPredict === 'function') {
                try {
                  // Pass the original schedule row so the parent can build the
                  // canonical API payload (season/week/teams).
                  onPredict(game);
                } catch (err) {
                  // eslint-disable-next-line no-console
                  console.error('[TeamGrid] onPredict handler threw', err);
                }
                return;
              }

              // Fallback path: construct the payload that the backend expects
              // and call the API helper directly.
              const payload = {
                home_team: homeAbbr,
                away_team: awayAbbr,
                // We support both season/week and season_num/week_num naming.
                season: game?.season ?? game?.season_num ?? null,
                week: game?.week ?? game?.week_num ?? null,
              };

              try {
                // Functional setState pattern:
                // - We use the callback form of setLocalLoadingMap(prev => next)
                //   to avoid bugs when multiple clicks happen in quick succession.
                setLocalLoadingMap((prev) => ({ ...prev, [rawKey]: true }));
                await predictGame(payload);
              } catch (err) {
                // eslint-disable-next-line no-console
                console.error('[TeamGrid] predictGame failed', err);
              } finally {
                // Clean up the local loading flag when the request completes,
                // regardless of success or failure.
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
                // Use the existing click handler that already implements the prediction flow
                onClick={handleClick}
                // Pass an onReset handler ONLY if the parent provided one.
                // The Card will call onReset(matchup) when the Reset button is clicked.
                onReset={onReset ? () => onReset(enrichedGame) : undefined}
              />
            );


          })}
        </div>
      </section>
    );
  }
}