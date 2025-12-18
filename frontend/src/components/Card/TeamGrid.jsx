/**
 * NFL Prediction App — TeamGrid Component (Expert v2.0)
 * ====================================================
 * 
 * Orchestrates the display of NFL matchups for a specific week.
 * This component acts as a smart container that maps raw game data
 * to highly-interactive Card components, managing prediction state 
 * and user interactions with optimized render cycles.
 */

import React from 'react';
import Card from './Card';
import './TeamGrid.css';
import cardStyles from './Card.module.css';
import { buildGameKey } from '../../utils/predictionContextUtils';

/**
 * TeamGrid Component
 * 
 * @param {Object} props
 * @param {number} props.week - Current NFL week being displayed.
 * @param {Array} props.games - List of matchups for the week.
 * @param {Object} props.predictions - Map of processed predictions keyed by game ID.
 * @param {Object} props.loading - Map of loading states keyed by game ID.
 * @param {Object} props.errors - Map of error messages keyed by game ID.
 * @param {Function} props.onPredict - Callback to trigger a new prediction.
 * @param {Function} props.onReset - Callback to clear a prediction.
 * @param {Object} props.features - Feature toggles (queueAware, confidenceDisplay, etc).
 */
export default function TeamGrid({
  week,
  games = [],
  predictions = {},
  loading = {},
  errors = {},
  onPredict,
  onReset,
  features = {}
}) {
  const gameItems = (games || []).map((game, index) => {
    const gkey = buildGameKey(game);

    const homeTeam = game.home_abbr || game.home_team;
    const awayTeam = game.away_abbr || game.away_team;

    return {
      key: gkey,
      index,
      matchup: {
        game_id: gkey,
        home_team: homeTeam,
        away_team: awayTeam,
        home_name: game.home_name,
        away_name: game.away_name,
        kickoff: game.kickoff,
        home_logo: game.home_logo,
        away_logo: game.away_logo,
        season: game.season,
        week: game.week
      },
      prediction: predictions[gkey],
      isLoading: !!loading[gkey],
      error: errors[gkey]
    };
  });

  // Empty State Handling
  if (!games || games.length === 0) {
    return (
      <div className="team-grid--empty">
        <div className="empty-message">
          <h3>No games found for Week {week}</h3>
          <p>The schedule might not be loaded or the week is invalid.</p>
        </div>
      </div>
    );
  }

  return (
    <section className="team-grid" aria-label='NFL Week Matchups'>
      <header className="team-grid__header">
        <h2 className="team-grid__title">Week {week} Matchups</h2>
        {features.queueAware && (
          <div className="team-grid__status-bar">
            <span>{gameItems.filter(g => g.prediction).length} / {gameItems.length} Predictions Ready</span>
          </div>
        )}
      </header>

      <div className={`team-grid__grid ${cardStyles.cardGrid}`}>
        {gameItems.map((item) => (
          <Card
            key={item.key}
            index={item.index}
            matchup={item.matchup}
            prediction={item.prediction}
            loading={item.isLoading}
            error={item.error}
            onClick={() => onPredict && onPredict(item.matchup)}
            onReset={() => onReset && onReset(item.matchup)}
            // Enhanced features
            status={item.isLoading ? "Crunching data..." : item.prediction ? "Predicted" : "Pending"}
            progress={item.isLoading ? 100 : item.prediction ? 100 : 0}
          />
        ))}
      </div>

      <footer className="team-grid__footer">
        <p className="footer-note">
          Predictions are generated using a multi-variate regression ensemble trained on historical NFL data.
        </p>
      </footer>
    </section>
  );
}
