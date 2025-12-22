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

import './PredictionResult.css';

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

  // Normalise "game" metadata
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

  const homePct = probs.home != null ? Math.round(probs.home * 100) : null;
  const awayPct = probs.away != null ? Math.round(probs.away * 100) : null;

  // Expert Simulation Metrics
  const sim = base.simulation_metrics;
  const showExpert = !!sim;

  // Compute confidence ranges (10th - 90th percentile ≈ mean ± 1.28 * std)
  const homeRange = showExpert ? [
    Math.round(sim.sim_home_score - 1.28 * sim.sim_std_home),
    Math.round(sim.sim_home_score + 1.28 * sim.sim_std_home)
  ] : null;
  const awayRange = showExpert ? [
    Math.round(sim.sim_away_score - 1.28 * sim.sim_std_away),
    Math.round(sim.sim_away_score + 1.28 * sim.sim_std_away)
  ] : null;

  return (
    <div className={`prediction-result-container ${showExpert ? 'expert-mode' : ''}`} aria-live="polite">
      <div className="prediction-header">
        <div className="header-text">
          <h3>{showExpert ? 'Ensemble Mixture Analysis' : 'Single Game Prediction'}</h3>
          <span className="meta-text">
            {game.week != null && game.season != null
              ? `Week ${game.week} • ${game.season}`
              : 'Match Details'}
          </span>
        </div>
        {showExpert && (
          <div className="expert-badge">
            <span className="pulse-icon"></span>
            Ensemble Mixture (ML + MC)
          </div>
        )}
      </div>

      <div className="expert-content">
        <div className="team-row">
          <div className="team-block home">
            <span className="team-name">{game.home_abbr}</span>
            <span className="score-main">{Math.round(metrics.home_score)}</span>
            {showExpert && (
              <div className="range-box">
                <span className="range-label">Expected Range</span>
                <span className="range-val">{Math.max(0, homeRange[0])}–{homeRange[1]}</span>
              </div>
            )}
          </div>

          <div className="vs-divider">
            <div className="vs-circle">VS</div>
            <div className="line"></div>
          </div>

          <div className="team-block away">
            <span className="team-name">{game.away_abbr}</span>
            <span className="score-main">{Math.round(metrics.away_score)}</span>
            {showExpert && (
              <div className="range-box">
                <span className="range-label">Expected Range</span>
                <span className="range-val">{Math.max(0, awayRange[0])}–{awayRange[1]}</span>
              </div>
            )}
          </div>
        </div>

        <div className="win-probability-expert">
          <div className="prob-header">
            <span>Win Probability</span>
            {showExpert && <span className="sim-meta">{sim.n_sims.toLocaleString()} trials</span>}
          </div>
          <div className="prob-bar-wrapper">
            <div className="prob-bar-base">
              <div 
                className="prob-fill home" 
                style={{ width: `${homePct}%` }}
              >
                <span className="prob-inner-text">{homePct}%</span>
              </div>
              <div 
                className="prob-fill away" 
                style={{ width: `${100 - homePct}%` }}
              >
                <span className="prob-inner-text">{100 - homePct}%</span>
              </div>
            </div>
          </div>
        </div>

        {showExpert && (
          <div className="expert-brief">
            <p>
              Model Confidence: <strong>High</strong>. Outcomes modeled using Gaussian distribution based on historical RMSE ({sim.sim_std_home?.toFixed(1)}).
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
