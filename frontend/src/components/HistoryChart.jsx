/**
 * HistoryChart.jsx
 * ----------------
 * Component Purpose:
 *   Consume the shared history entries and render a minimal trend list.
 *
 * Core Logic Overview:
 *   - Derive `points` with `useMemo` so we only recompute when history updates.
 *   - Safely pick the best probability value (`ensemble` ➜ `home`) and convert
 *     it to a human-readable percentage.
 *   - Provide a text-based fallback list you can later replace with a chart lib.
 *
 * Modification Guide:
 *   - Swap the `<ul>` for your favourite chart component; just plug in `points`.
 *   - If you need more metrics, extend the map function and keep null checks so
 *     missing data never crashes the render.
 */

import React, {useMemo} from 'react';

export default function HistoryChart({history}) {
  // Transform history entries into chart-friendly tuples.
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

  // Empty states should still return semantic markup for screen readers.
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
