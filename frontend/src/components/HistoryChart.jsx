/**
 * HistoryChart.jsx
 * ----------------
 * Render a compact history view for prediction entries.
 * Accepts either a single entry or an array of entries as `historyData`.
 */

import React, { useMemo } from 'react';

export default function HistoryChart({ state, history }) {
  // Handle different prop patterns
  const historyArray = state?.history || history || [];

  const points = useMemo(() => {
    return historyArray.map((e, idx) => {
      const ts = e?.ts ?? e?.game?.ts ?? null;
      const rawProb = e?.probs?.ensemble ?? e?.probs?.home ?? e?.probs?.away ?? null;
      const pct = rawProb != null && typeof rawProb === 'number' ? Math.round(rawProb * 100) : null;
      const label = `${e?.game?.away_abbr ?? 'Away'} @ ${e?.game?.home_abbr ?? 'Home'}`;
      const time = ts ? new Date(ts) : null;
      return {
        x: time,
        y: pct,
        label,
        originalIndex: idx,
      };
    });
  }, [historyArray]);

  if (points.length === 0) {
    return <div className="history-chart">No history yet.</div>;
  }

  return (
    <div className="history-chart" aria-live="polite">
      <h3>Prediction History</h3>
      <ul>
        {points.map((p) => (
          <li key={p.originalIndex}>
            <code>{p.x ? p.x.toLocaleString() : '—'}</code>
            {' — '}
            <strong>{p.y != null ? `${p.y}%` : 'n/a'}</strong>
            {' '}
            <em>({p.label})</em>
          </li>
        ))}
      </ul>
    </div>
  );
}
