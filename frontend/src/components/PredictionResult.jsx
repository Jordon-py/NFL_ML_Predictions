/**
 * PredictionResult.jsx
 * --------------------
 * Purpose:
 *   Present the latest prediction ("current") in a consistent schema.
 *
 * Layer 1 Fix:
 *   - Aligns with unified entry: metrics + probs.
 *
 * Layer 2:
 *   - Stateless display. All logic lives in Context or containers.
 */

import React from 'react';

export default function PredictionResult({ entry }) {
  if (!entry) return <div className="prediction-result">No prediction yet.</div>;

  const { game, metrics, probs } = entry;
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
        <strong>{game.home_abbr}</strong> {metrics.home_score} — {metrics.away_score} <strong>{game.away_abbr}</strong>
        <span className="separator">•</span>
        <span>Diff: {metrics.point_diff}</span>
      </div>

      <div className="probs">
        {homePct != null && <span>Home win: {homePct}%</span>}
        {awayPct != null && <span>Away win: {awayPct}%</span>}
        {ensemblePct != null && <span>Ensemble: {ensemblePct}%</span>}
      </div>
    </div>
  );
}
