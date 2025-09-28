/**
 * HistoryChart.jsx
 * ----------------
 * Purpose:
 *   Consume the unified history entries and render a simple trend.
 *
 * Layer 1 + 2 Fix:
 *   - Stop assuming `ensemble_proba` exists on root.
 *   - Use `entry.probs.ensemble ?? entry.probs.home ?? null` safely.
 *
 * Note:
 *   Replace the placeholder rendering with your chart lib as needed.
 */

import React, { useMemo } from 'react';

export default function HistoryChart({ history }) {
  const points = useMemo(() => {
    return (history ?? []).map((e) => {
      const y = (e?.probs?.ensemble ?? e?.probs?.home ?? null);
      return {
        x: e?.ts ?? '',
        y: y != null ? Math.round(y * 100) : null,
        label: `${e?.game?.away_abbr} @ ${e?.game?.home_abbr}`,
      };
    });
  }, [history]);

  if (!points.length) return <div className="history-chart">No history yet.</div>;

  // Minimal textual fallback. Swap for a chart library in your stack.
  return (
    <div className="history-chart">
      <h3>History (prob %)</h3>
      <ul>
        {points.map((p, i) => (
          <li key={i}>
            <code>{p.x}</code> — <strong>{p.y ?? 'n/a'}%</strong> <em>({p.label})</em>
          </li>
        ))}
      </ul>
    </div>
  );
}

// PropTypes removed: types are managed by upstream context or TypeScript where applicable.
