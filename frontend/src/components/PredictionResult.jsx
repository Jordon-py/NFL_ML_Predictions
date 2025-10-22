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

export default function PredictionResult({entry}) {
// Checks entry prop for the predictions
  console.log(`entry: ${JSON.stringify(entry)}, rawEntry in predictionResult.jsx: ${entry}`);
  if (!entry) return (
      <div className="prediction-result">No prediction yet.</div>  
      // Provide a friendly fallback so the area never collapses visually.
  ); 

  // Defensive destructuring: fallback to empty objects to avoid runtime errors if entry is malformed.
  const {game = {}, metrics = {}, probs = {}} = entry;
  // Convert probabilities to whole-number percentages for readability.
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
            <strong>{game.home_abbr} (Home): {metrics.home_score != null ? metrics.home_score : '-'}</strong>
            <span className="score-sep"> — </span>
            <strong>{game.away_abbr} (Away): {metrics.away_score != null ? metrics.away_score : '-'}</strong>
          </div>
          <div className="score-diff">
            <strong>{game.home_abbr}</strong> {metrics.home_score != null ? metrics.home_score : '-'} — {metrics.away_score != null ? metrics.away_score : '-'} <strong>{game.away_abbr}</strong>
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
