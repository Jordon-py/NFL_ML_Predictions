/**
 * PredictionResult.jsx
 * --------------------
 * Component Purpose:
 *   Present the most recent prediction entry supplied by the context.
 *
 * Core Logic Overview:
 *   - Guard against `null` entries and return a helpful empty message.
 *   - Compute display-friendly percentages using basic rounding.
 *   - Render structured sections (meta, scores, probabilities) for clarity.
 *
 * Modification Guide:
 *   - Keep heavy calculations out of the component—normalise data inside the
 *     context or dedicated selectors.
 *   - When adding new metrics, ensure you handle `null`/`undefined` so the UI
 *     never crashes while the backend evolves.
 */

import React from 'react';

/**
 * @typedef {Object} PredictionGame
 * @property {number} [week]
 * @property {number} [season]
 * @property {string} [home_abbr]
 * @property {string} [away_abbr]
 * @property {string} [home_team]
 * @property {string} [away_team]
 */

/**
 * @typedef {Object} PredictionMetrics
 * @property {number|null} [home_score]
 * @property {number|null} [away_score]
 * @property {number|null} [home_score_pred]
 * @property {number|null} [away_score_pred]
 * @property {number|null} [point_diff]
 */

/**
 * @typedef {Object} PredictionProbabilities
 * @property {number|null} [home]
 * @property {number|null} [away]
 * @property {number|null} [ensemble]
 * @property {number|null} [home_win_probability]
 * @property {number|null} [away_win_probability]
 */

/**
 * @typedef {Object} PredictionEntry
 * @property {PredictionGame} [game]
 * @property {PredictionMetrics} [metrics]
 * @property {PredictionProbabilities} [probs]
 * @property {number} [week]
 * @property {number} [season]
 * @property {number} [home_score]
 * @property {number} [away_score]
 * @property {number} [home_score_pred]
 * @property {number} [away_score_pred]
 * @property {number} [point_diff]
 * @property {number} [home_win_probability]
 * @property {number} [away_win_probability]
 * @property {string} [home_abbr]
 * @property {string} [away_abbr]
 * @property {string} [home_team]
 * @property {string} [away_team]
 */

/** @type {PredictionEntry} */
const EMPTY_ENTRY = {};

/**
 * PredictionResult
 * ----------------
 * Displays the most recent prediction entry, including game metadata, scores, and win probabilities.
 *
 * @param {{ entry: PredictionEntry | null }} props
 *   - entry.game: {Object} Game metadata (week, season, home/away abbreviations).
 *   - entry.metrics: {Object} Prediction metrics (scores, point_diff).
 *   - entry.probs: {Object} Probability metrics (home, away, ensemble).
 *
 * Renders structured prediction details, handling missing/null data gracefully.
 */
export default function PredictionResult({ entry }) {
  if (!entry) {
    return (
      <div className="prediction-result" aria-live="polite">
        <h3>Prediction</h3>
        <p>No prediction selected yet.</p>
      </div>
    );
  }

  /** @type {PredictionEntry} */
  const base = entry ?? EMPTY_ENTRY;

  // Normalise "game" metadata regardless of whether it comes nested (entry.game)
  // or from a backend history payload (top-level season/week/home/away fields).
  const rawGame = base.game || {};
  const game = {
    week: rawGame.week ?? base.week,
    season: rawGame.season ?? base.season,
    home_abbr:
      rawGame.home_abbr ??
      rawGame.home_team ??
      base.home_abbr ??
      base.home_team,
    away_abbr:
      rawGame.away_abbr ??
      rawGame.away_team ??
      base.away_abbr ??
      base.away_team,
  };

  // Normalise score + margin fields across shapes:
  //   - legacy: entry.metrics.{home_score,away_score,point_diff}
  //   - /predict response: {home_score, away_score, point_diff}
  //   - /history entry: {home_score_pred, away_score_pred, point_diff}
  const metrics = {
    home_score:
      (base.metrics && base.metrics.home_score) ??
      base.home_score ??
      base.home_score_pred ??
      null,
    away_score:
      (base.metrics && base.metrics.away_score) ??
      base.away_score ??
      base.away_score_pred ??
      null,
    point_diff:
      (base.metrics && base.metrics.point_diff) ??
      base.point_diff ??
      null,
  };

  // Normalise probability fields:
  //   - legacy: entry.probs.{home,away,ensemble}
  //   - /predict + /history: {home_win_probability, away_win_probability}
  const probs = {
    home:
      (base.probs && base.probs.home) ??
      base.home_win_probability ??
      null,
    away:
      (base.probs && base.probs.away) ??
      base.away_win_probability ??
      null,
    ensemble: base.probs?.ensemble ?? null,
  };

  // Compute display-friendly percentages, rounding if values exist.
  const homePct = probs.home != null ? Math.round(probs.home * 100) : null;
  const awayPct = probs.away != null ? Math.round(probs.away * 100) : null;
  const ensemblePct = probs.ensemble != null ? Math.round(probs.ensemble * 100) : null;

  return (
    <>

      <div className="prediction-result" aria-live="polite">
        <h3>Prediction</h3>
        <div className="meta">
          <span>
            {game.week != null && game.season != null
              ? `Week ${game.week} • ${game.season}`
              : 'Game info unavailable'}
          </span>
        </div>
        <div className="scores">
          <div className="score-line">
            <strong>
              {game.home_abbr} (Home):{' '}
              {metrics.home_score != null ? metrics.home_score : '-'}
            </strong>
            <span className="score-sep"> — </span>
            <strong>
              {game.away_abbr} (Away):{' '}
              {metrics.away_score != null ? metrics.away_score : '-'}
            </strong>
          </div>
          <div className="score-diff">
            <strong>{game.home_abbr}</strong>{' '}
            {metrics.home_score != null ? metrics.home_score : '-'} —{' '}
            {metrics.away_score != null ? metrics.away_score : '-'}{' '}
            <strong>{game.away_abbr}</strong>
            <span className="separator">•</span>
            <span>Diff: {metrics.point_diff != null ? metrics.point_diff : '-'}</span>
          </div>
        </div>
        <div className="probs">
          {homePct != null && <span>Home win: {homePct}%</span>}
          {awayPct != null && <span>Away win: {awayPct}%</span>}
          {ensemblePct != null && <span>Ensemble: {ensemblePct}%</span>}
          {[homePct, awayPct, ensemblePct].every(v => v == null) && (
            <span>No probability data available.</span>
          )}
        </div>
      </div>
    </>
  );
}