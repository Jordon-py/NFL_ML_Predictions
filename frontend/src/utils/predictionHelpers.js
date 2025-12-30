/**
 * predictionHelpers.js
 * ---------------------
 * Utility functions for normalizing prediction data across the app.
 * Separated from UI components to keep Vite Fast Refresh stable.
 */

/**
 * Normalizes backend prediction payload into a UI-friendly entry object.
 *
 * @param {Object} params - Prediction data from backend or UI.
 * @param {string} [params.source='teamgrid'] - Source of the prediction (e.g., 'teamgrid', 'api').
 * @param {number} params.season - NFL season year.
 * @param {number} params.week - NFL week number.
 * @param {string} params.home_abbr - Home team abbreviation.
 * @param {string} params.away_abbr - Away team abbreviation.
 * @param {number} params.home_score - Predicted home team score.
 * @param {number} params.away_score - Predicted away team score.
 * @param {number} params.point_diff - Predicted point differential.
 * @param {number} params.home_win_probability - Probability home team wins.
 * @param {number} params.away_win_probability - Probability away team wins.
 * @param {number} [params.ensemble_probability] - Optional ensemble model probability.
 * @returns {Object} Normalized prediction entry for UI/history.
 */
export function toEntry({
  source = 'teamgrid',
  season,
  week,
  home_abbr,
  away_abbr,
  home_score,
  away_score,
  point_diff,
  home_win_probability,
  away_win_probability,
  ensemble_probability
}) {
  return {
    ts: new Date().toISOString(),
    source,
    game: { season, week, home_abbr, away_abbr },
    metrics: { home_score, away_score, point_diff },
    probs: {
      home: home_win_probability,
      away: away_win_probability,
      ensemble: ensemble_probability
    }
  };
}
