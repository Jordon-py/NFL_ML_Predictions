// ==========================================
// File: frontend/src/utils/predictionHelpers.js
// Role: Frontend utility helpers.
// Input Data: Function parameters.
// Output Data: Transformed values.
// Dependencies: None
// Notes: Shared across UI modules.
// ==========================================
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
 * @param {Object} [params.prediction] - Raw backend prediction payload (flat or nested fields).
 * @param {Object} [params.game] - Schedule/game context used for fallback info.
 * @param {string} [params.source='teamgrid'] - Source of the prediction (e.g., 'teamgrid', 'api').
 * @returns {Object} Normalized prediction entry for UI/history.
 */
export function toEntry({
  source = "teamgrid",
  prediction,
  game,
  ...payload
}) {
  const rawPrediction = prediction && typeof prediction === "object" ? prediction : {};
  const nestedPrediction =
    rawPrediction.prediction && typeof rawPrediction.prediction === "object"
      ? rawPrediction.prediction
      : null;

  const base = {
    ...rawPrediction,
    ...(nestedPrediction || {}),
    ...payload,
  };

  const entry = {
    ...base,
    source: base.source ?? source,
    ts: base.ts ?? new Date().toISOString(),
  };

  const fallbackGame = game && typeof game === "object" ? game : null;
  if (fallbackGame) entry.game = fallbackGame;

  entry.season ??= fallbackGame?.season;
  entry.week ??= fallbackGame?.week;
  entry.home_team ??= fallbackGame?.home_team ?? fallbackGame?.home_abbr;
  entry.away_team ??= fallbackGame?.away_team ?? fallbackGame?.away_abbr;
  entry.home_name ??= fallbackGame?.home_name;
  entry.away_name ??= fallbackGame?.away_name;

  if (entry.point_diff == null && entry.home_score != null && entry.away_score != null) {
    entry.point_diff = Number(entry.home_score) - Number(entry.away_score);
  }

  return entry;
}

/** Convert a probability in [0..1] to an integer percentage, or null if invalid. */
export function toWholePercent(prob) {
  const n = Number(prob);
  if (!Number.isFinite(n)) return null;
  return Math.round(n * 100);
}
