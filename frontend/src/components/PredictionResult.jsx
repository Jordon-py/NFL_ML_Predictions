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

import * as React from 'react';

export default function PredictionResult({entry}) {
  // Provide a friendly fallback so the area never collapses visually.
  if (!entry) return <div className="prediction-result">No prediction yet.</div>;

  // Defensive destructuring: fallback to empty objects to avoid runtime errors if entry is malformed.
  const {game = {}, metrics = {}, probs = {}} = entry;
  // Convert probabilities to whole-number percentages for readability.
  const homePct = probs.home != null ? Math.round(probs.home * 100) : null;
  const awayPct = probs.away != null ? Math.round(probs.away * 100) : null;
  const ensemblePct = probs.ensemble != null ? Math.round(probs.ensemble * 100) : null;

  return (
    <div className="prediction-result" aria-live="polite">
      <h3>Prediction</h3>
      <div className="meta">
        <span>Week {game.week} • {game.season}</span>
        <span>{game.away_abbr} @ {game.home_abbr}</span>
      </div>
      <div className="scores">
        <div className="score-line">
          <b>{game.home_abbr} (Home): {metrics.home_score}</b>
          <span className="score-sep"> — </span>
          <b>{game.away_abbr} (Away): {metrics.away_score}</b>
        </div>
        <div className="score-diff">
          <strong>{game.home_abbr}</strong> {metrics.home_score} — {metrics.away_score} <strong>{game.away_abbr}</strong>
          <span className="separator">•</span>
          <span>Diff: {metrics.point_diff}</span>
        </div>
      </div>
      <div className="probs">
        {homePct != null && <span>Home win: {homePct}%</span>}
        {awayPct != null && <span>Away win: {awayPct}%</span>}
        {ensemblePct != null && <span>Ensemble: {ensemblePct}%</span>}
        {homePct == null && awayPct == null && ensemblePct == null && (
          <span>No probability data available.</span>
        )}
      </div>
    </div>
  );
}
