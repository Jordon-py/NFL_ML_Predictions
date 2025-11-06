// /frontend/src/components/HistoryChart.jsx
/**
 * TypeScript checking enabled. All major props and helpers are now typed for maintainability.
 */

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
 *   <HistoryChart history={arrayOfEvents} state={controllerState} />
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
import D_BUTTON from "./D_BUTTON";


/**
 * @typedef {Object} Game
 * @property {number} season
 * @property {number} week
 * @property {string} away_abbr
 * @property {string} home_abbr
 * @property {string|number|Date=} ts
 */

/**
 * @typedef {Object} Event
 * @property {string|number|Date=} ts
 * @property {string|number|Date=} time
 * @property {{home?: number, ensemble?: number}=} probs
 * @property {number=} home_win_probability
 * @property {Game=} game
 */

/**
 * @typedef {Object} HistoryChartProps
 * @property {Event[]|undefined} history
 * @property {Object=} state
 * @property {Function=} onClearHistory
 */

/* ---------- tiny utilities ---------- */

/** Return the first non-nullish value (null/undefined are skipped). */
const firstNonNullish = (...values) => values.find((v) => v != null);

/** Convert a probability in [0..1] to an integer percentage, or null if invalid. */
const toWholePercent = (prob) => (
  typeof prob === "number" ?
    Math.round(prob * 100) : null);


/** Safely coerce many timestamp shapes to a Date, or null if not present. */
const toDateOrNull = (value) => {
  if (value == null) return null;
  const d = value instanceof Date ? value : new Date(value);
  return isNaN(d.getTime()) ? null : d;
};

/* ---------- normalization helpers (keep render logic clean) ---------- */

/** Prefer `ts`, then `time`; return a Date or null. */
const extractTimestamp = (event) => toDateOrNull(firstNonNullish(event?.ts, event?.time, event?.game?.ts, null));

/** Prefer `probs.home`, fallback to other sources; return [0..1] or null. */
const extractHomeWinProbability = (event) =>
  firstNonNullish(
    event?.probs?.home,            // primary source per contract
    event?.home_win_probability,   // fallback used elsewhere in app
    event?.probs?.ensemble,        // optional ensemble prob
    null
  );

/** Build a readable game label, or a generic entry label if no game info exists. */
const buildGameLabel = (event, index) => {
  const g = event?.game;
  if (g?.season && g?.week && g?.away_abbr && g?.home_abbr) {
    return `${g.season} W${g.week} ${g.away_abbr}@${g.home_abbr}`;
  }
  if (g?.away_abbr || g?.home_abbr) {
    return `${g?.away_abbr ?? "Away"} @ ${g?.home_abbr ?? "Home"}`;
  }
  return `Entry ${index + 1}`;
};

/**
 * HistoryChart
 * @param {Object} props
 * @param {Array} props.history - Array of prediction events
 * @param {Object} props.state - Optional controller state
 * @param {Function} [props.onClearHistory] - Callback to clear history (provided by parent)
 */
/**
 * HistoryChart
 * @param {HistoryChartProps} props
 */
export default function HistoryChart({ history, state, onClearHistory }) {
  // Normalize the input early so the rest of the code can assume an array.
  const historyItems = useMemo(() => {
    if (Array.isArray(history)) {
      return history;
    }
    if (Array.isArray(state?.history)) {
      return state.history;
    }
    return [];
  }, [history, state?.history]);

  /**
   * chartPoints: normalized, render-ready rows
   *   - index: stable key/index
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

  // Remove useEffect and handleClick here; expect onClearHistory prop from parent.
  /* ---------- render ---------- */

  if (chartPoints.length === 0) {
    return (
      <section className="history-chart" aria-live="polite">
        <header>
          <h2>Prediction History</h2>
          <small>0 item(s)</small>
        </header>
        <p>No history yet. Make some predictions to populate this view.</p>
      </section>
    );
  }

  // Note: Clear History control is self-contained via D_BUTTON (uses context)

  return (
    <section className="history-chart" aria-live="polite">
      <header>
        <div className="history-chart-header-controls">
          <h2>Prediction History</h2>
          <D_BUTTON />
        </div>
        <br />
        <small>
          {statsSummary.totalCount} item(s)
          {statsSummary.mostRecentDate && <> • last: {statsSummary.mostRecentDate.toLocaleString()}</>}
          {statsSummary.averageHomeWinPercent != null && <> • avg home win: {statsSummary.averageHomeWinPercent}%</>}
        </small>
      </header>
      <ol className="history-points a-text-fade-slide">
        {chartPoints.slice(0, 16).map((row) => (
          <li key={row.index} title={row.label}>
            <code>{row.timestamp ? row.timestamp.toLocaleString() : "—"}</code>
            {" — "}
            <strong>{row.homeWinPercent != null ? `${row.homeWinPercent}%` : "n/a"}</strong>{" "}
            <em>({row.label})</em>
          </li>
        ))}
      </ol>
    </section>
  );
}
