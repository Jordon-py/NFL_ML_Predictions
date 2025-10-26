// /frontend/src/components/HistoryChart.jsx
// @ts-nocheck
import React, { useMemo } from 'react';

export default function HistoryChart({ history = [] }) {
  const items = Array.isArray(history) ? history : [];

  const points = useMemo(() => {
    return items.map((e, i) => {
      const ts = e.ts || e.time || null;
      const label = e?.game
        ? `${e.game.season} W${e.game.week} ${e.game.away_abbr}@${e.game.home_abbr}`
        : `Entry ${i + 1}`;
      const homeProb = e?.probs?.home ?? e?.probs?.ensemble ?? e?.home_win_probability ?? null;
      return {
        x: ts ? new Date(ts) : null,
        y: homeProb != null ? Math.round(homeProb * 100) : null,
        label,
        idx: i
      };
    });
  }, [items]);

  const summary = useMemo(() => {
    const ys = points.map(p => p.y).filter(v => typeof v === "number");
    const avg = ys.length ? (ys.reduce((a, b) => a + b, 0) / ys.length) : null;
    return {
      count: items.length,
      last: items[0]?.ts || null,
      avgHomeWinPct: avg != null ? Math.round(avg) : null
    };
  }, [points, items]);

  return (
    <section className="history-chart">
      <header>
        <h3>Prediction History</h3>
        <small>
          {summary.count} item(s)
          {summary.last ? <> • last: {new Date(summary.last).toLocaleString()}</> : null}
          {summary.avgHomeWinPct != null ? <> • avg home win: {summary.avgHomeWinPct}%</> : null}
        </small>
      </header>

      {points.length === 0 ? (
        <p>No history yet. Make some predictions to populate this view.</p>
      ) : (
        <ol className="history-points">
          {points.map((p) => (
            <li key={p.idx} title={p.label}>
              <code>{p.x ? p.x.toLocaleString() : "—"}</code>
              {" — "}
              <strong>{p.y != null ? `${p.y}%` : "n/a"}</strong>
              {" "}
              <em>({p.label})</em>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
