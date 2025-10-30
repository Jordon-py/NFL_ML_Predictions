// /frontend/src/components/HistoryChart.jsx
// @ts-nocheck

/**
 * HistoryChart — Educational Overview
 *
 * Purpose:
 *   Display a simple, accessible “history feed” of prediction events. Each row shows:
 *     • When the prediction was made (timestamp)
 *     • Home team win probability (as a whole percent)
 *     • A human-readable label for the game (e.g., "2025 W7 MIA@BUF")
 *
 * Data contract (input):
 *   <HistoryChart history={arrayOfEvents} />
 *   Each event may come from different parts of the app, so we normalize the fields.
 *
 *   Expected event shape (loose union — missing fields are tolerated):
 *     {
 *       ts?: string | number | Date,               // primary timestamp
 *       time?: string | number | Date,             // optional alt timestamp
 *       probs?: { home?: number, ensemble?: number },
 *       home_win_probability?: number,             // optional alt prob (0..1)
 *       game?: {
 *         season: number,
 *         week: number,
 *         away_abbr: string,                       // e.g., "MIA"
 *         home_abbr: string                        // e.g., "BUF"
 *       }
 *     }
 *
 * Key ideas (why the code looks this way):
 *   • Normalization helpers (extractTimestamp / extractHomeWinProbability / buildGameLabel) keep render logic clean.
 *   • useMemo caches derived arrays/aggregates so we don’t recompute on every render.
 *   • We fail gracefully: when fields are missing, we render "—" or "n/a" instead of breaking.
 *
 * Teaching notes (React patterns used):
 *   • “Derived state” via useMemo: when a value can be computed from props, prefer memoized derivation over useState.
 *   • “Defensive rendering”: optional chaining (?.) + nullish checks keep UI robust against partial data.
 *   • “Single responsibility” helpers: keeps the map() clean and self-documenting.
 */

import React, { useMemo } from "react";

/* ---------- tiny utilities ---------- */

/** Return the first non-nullish value (null/undefined are skipped). */
const firstNonNullish = (...values) => values.find((v) => v != null);

/** Convert a probability in [0..1] to an integer percentage, or null if invalid. */
const toWholePercent = (prob) => (typeof prob === "number" ? Math.round(prob * 100) : null);

/** Safely coerce many timestamp shapes to a Date, or null if not present. */
const toDateOrNull = (value) => {
  if (value == null) return null;
  const d = value instanceof Date ? value : new Date(value);
  return isNaN(d.getTime()) ? null : d;
};

/* ---------- normalization helpers (keep render logic clean) ---------- */

/** Prefer `ts`, then `time`; return a Date or null. */
const extractTimestamp = (event) => toDateOrNull(firstNonNullish(event?.ts, event?.time, null));

/** Prefer `probs.home`, fallback to `probs.ensemble`, then `home_win_probability`; return [0..1] or null. */
const extractHomeWinProbability = (event) =>
  firstNonNullish(event?.probs?.home, event?.probs?.ensemble, event?.home_win_probability, null);

/** Build a readable game label, or a generic entry label if no game info exists. */
const buildGameLabel = (event, index) => {
  const g = event?.game;
  return g
    ? `${g.season} W${g.week} ${g.away_abbr}@${g.home_abbr}`
    : `Entry ${index + 1}`;
};

export default function HistoryChart({ history }) {
  // Normalize the input early so the rest of the code can assume an array.
  const historyItems = Array.isArray(history) ? history : [];

  /**
   * chartPoints: normalized, render-ready rows
   *   - index: stable key/index
   *   - timestamp: Date|null
   *   - homeWinPercent: integer percent or null
   *   - label: string
   */
  const chartPoints = useMemo(() => {
    return historyItems.map((event, index) => {
      const timestamp = extractTimestamp(event);
      const prob = extractHomeWinProbability(event);
      return {
        index,
        timestamp,
        homeWinPercent: toWholePercent(prob),
        label: buildGameLabel(event, index),
      };
    });
  }, [historyItems]);

  /**
   * statsSummary: small header stats we can show users
   *   - totalCount: number of events
   *   - mostRecentDate: Date|null (based on first item in the array)
   *   - averageHomeWinPercent: integer percent or null
   */
  const statsSummary = useMemo(() => {
    const percentValues = chartPoints
      .map((p) => p.homeWinPercent)
      .filter((n) => typeof n === "number");

    const averageHomeWinPercent =
      percentValues.length
        ? Math.round(percentValues.reduce((a, b) => a + b, 0) / percentValues.length)
        : null;

    const mostRecentDate = historyItems[0] ? extractTimestamp(historyItems[0]) : null;

    return {
      totalCount: historyItems.length,
      mostRecentDate,
      averageHomeWinPercent,
    };
  }, [chartPoints, historyItems]);

  /* ---------- render ---------- */

  return (
    <section className="history-chart">
      <header>
        <h2>Prediction History</h2>
        <small>
          {statsSummary.totalCount} item(s)
          {statsSummary.mostRecentDate && <> • last: {statsSummary.mostRecentDate.toLocaleString()}</>}
          {statsSummary.averageHomeWinPercent != null && <> • avg home win: {statsSummary.averageHomeWinPercent}%</>}
        </small>
      </header>

      {chartPoints.length === 0 ? (
        <p>No history yet. Make some predictions to populate this view.</p>
      ) : (
          <ol className="history-points a-text-fade-slide">
          {console.log(`CHARTPOINTS: ${JSON.stringify(chartPoints)}`)}
          {chartPoints.slice(0, 16).map((row) => (
            <React.Fragment key={row.index}>
              <li title={row.label}>
                <code>{row.timestamp ? row.timestamp.toLocaleString() : "—"}</code>
                {" — "}
                <strong>{row.homeWinPercent != null ? `${row.homeWinPercent}%` : "n/a"}</strong>{" "}
                <em>({row.label})</em>
              </li>
              <br />
            </React.Fragment>
          ))}
        </ol>
      )}
    </section>
  );
}
