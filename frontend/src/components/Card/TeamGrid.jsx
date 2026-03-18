/**
 * NFL Prediction App — TeamGrid Component (Expert v2.0)
 * ====================================================
 * 
 * Orchestrates the display of NFL matchups for a specific week.
 * This component acts as a smart container that maps raw game data
 * to highly-interactive Card components, managing prediction state 
 * and user interactions with optimized render cycles.
 */

import { buildGameKey } from '../../utils/predictionContextUtils';
import Card from './Card';
import cardStyles from './Card.module.css';
import './TeamGrid.css';

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
 * @param {number} [props.season] - Season label used when no slate is present.
 */
export default function TeamGrid({
  week,
  season = null,
  games = [],
  predictions = {},
  loading = {},
  errors = {},
  onPredict,
  onReset,
  features = {},
  actualScores = {},
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
          home_color: game.home_color,
          away_color: game.away_color,
          home_color2: game.home_color2,
          away_color2: game.away_color2,
          home_wordmark: game.home_wordmark,
          away_wordmark: game.away_wordmark,
          season: game.season,
          week: game.week,
          game_type: game.game_type,
        },
        prediction: predictions[gkey],
        isLoading: !!loading[gkey],
        error: errors[gkey],
        actualScore: actualScores[gkey],
      };
  });

  // Empty State Handling
  if (!games || games.length === 0) {
    return (
      <div className="team-grid__empty">
        <div className="team-grid__empty-text">
          <h3>No matchups are available right now.</h3>
          <p>
            Try another week or season from the controls above, and the grid will refresh with more
            matchups for Season {season ?? new Date().getFullYear()}.
          </p>
        </div>
      </div>
    );
  }

  const isPostseason = games.some(g => g.game_type && g.game_type !== 'REG');
  const gridTitle = isPostseason ? "Postseason Matchups" : `Week ${week} Matchups`;

  return (
    <section className="team-grid" aria-label='NFL Matchups'>
      <header className="team-grid__header">
        <h2 className="team-grid__title">{gridTitle}</h2>
        {features.queueAware && (
          <div className="team-grid__status-bar">
            <span>{gameItems.filter(g => g.prediction).length} / {gameItems.length} forecasts ready</span>
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
          status={
            item.isLoading
              ? "Crunching data..."
              : item.actualScore?.status
                ? item.actualScore.status
                : item.prediction
                  ? "Forecast ready"
                  : "Pending prediction"
          }
          progress={item.isLoading || item.prediction ? 100 : 0}
        />
        ))}
      </div>

      <footer className="team-grid__footer">
        <p className="footer-note">
          Select any matchup to generate a forecast and open the detailed breakdown below.
        </p>
      </footer>
    </section>
  );
}
