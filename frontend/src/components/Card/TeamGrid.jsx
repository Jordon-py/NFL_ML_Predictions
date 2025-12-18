// File: frontend/src/components/Card/TeamGrid.jsx
// Renders the weekly schedule grid and delegates prediction actions to its parent.

import Card from './Card.jsx';
import './TeamGrid.css';

const normalizeAbbr = (value) => (value ?? '').toString().trim().toUpperCase();

const toGameKey = (game) => {
  const season = game?.season ?? game?.season_num ?? '';
  const week = game?.week ?? game?.week_num ?? '';
  const home = normalizeAbbr(game?.home_abbr ?? game?.home_team);
  const away = normalizeAbbr(game?.away_abbr ?? game?.away_team);
  return [season, week, home, away].filter(Boolean).join('-');
};

/**
 * TeamGrid component
 *
 * @param {Object} [props]
 * @param {number} [props.week]
 * @param {Array<any>} [props.games]
 * @param {boolean} [props.isLoading]
 * @param {Record<string, any>} [props.predictions]
 * @param {Record<string, boolean>} [props.loading]
 * @param {Record<string, any>} [props.errors]
 * @param {(game: any) => void} [props.onPredict]
 * @param {(game: any) => void} [props.onReset]
 * @param {() => Promise<void> | void} [props.onPredictAll]
 * @param {boolean} [props.isBulkLoading]
 */
export default function TeamGrid({
  week = 10,
  games = [],
  isLoading = false,
  predictions = {},
  loading = {},
  errors = {},
  onPredict,
  onReset,
  onPredictAll,
  isBulkLoading = false,
} = {}) {
  // Prefer explicit week; fall back to first game; default to 10 as safe placeholder.
  const safeWeek = week ?? games?.[0]?.week ?? games?.[0]?.week_num ?? 10;

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
          <h2 className="team-grid__title">Week {safeWeek}</h2>
        </div>
        <div className="team-grid__actions">
          <button
            type="button"
            className="team-grid__btn"
            onClick={typeof onPredictAll === 'function' ? onPredictAll : undefined}
            disabled={isLoading || isBulkLoading || typeof onPredictAll !== 'function'}
            aria-busy={isBulkLoading ? 'true' : 'false'}
          >
            {isBulkLoading ? 'Predicting...' : 'Predict All Games'}
          </button>
        </div>
        <p className="team-grid__subtitle">
          Showing <strong>{games.length}</strong> games scheduled.
        </p>
      </header>

      <div className="team-grid__grid">
        {games.map((game, index) => {
          // Stable key for React + lookup maps
          const key = toGameKey(game) || String(index);
          const prediction = predictions?.[key];
          const isGameLoading = Boolean(loading?.[key]);
          const errorMessage = errors?.[key] ?? null;

          const homeAbbr = normalizeAbbr(game.home_abbr ?? game.home_team);
          const awayAbbr = normalizeAbbr(game.away_abbr ?? game.away_team);
          const matchup = {
            ...game,
            season: game?.season ?? game?.season_num,
            week: game?.week ?? game?.week_num,
            home_team: homeAbbr,
            away_team: awayAbbr,
            home_logo: game?.home_logo ?? null,
            away_logo: game?.away_logo ?? null,
            kickoff: game?.kickoff ?? null,
          };

          return (
            <Card
              key={key}
              matchup={matchup}
              prediction={prediction}
              loading={isGameLoading}
              error={errorMessage}
              index={index}
              onClick={typeof onPredict === 'function' ? () => onPredict(game) : undefined}
              onReset={typeof onReset === 'function' ? () => onReset(game) : undefined}
            />
          );
        })}
      </div>
    </section>
  );
}
