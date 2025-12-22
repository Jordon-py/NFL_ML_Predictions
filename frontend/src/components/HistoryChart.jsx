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
 *   Primary usage:
 *     <HistoryChart />
 *     - Reads history from PredictionContext via selector hooks.
 *
 *   Optional override:
 *     <HistoryChart history={arrayOfEvents} />
 *     - If `history` is provided and is an array, it is used instead of context.
 *
 *   Expected event shape (loose union — missing fields are tolerated):
 *     {
 *       ts?: string | number | Date,               // primary timestamp
 *       time?: string | number | Date,             // optional alt timestamp
 *       probs?: { home?: number, ensemble?: number, away?: number },
 *       home_win_probability?: number,             // optional alt prob (0..1)
 *       game?: {
 *         season: number,
 *         week: number,
 *         away_abbr: string,                       // e.g., "MIA"
 *         home_abbr: string                        // e.g., "BUF"
 *       }
 *     }
 *
 * Key ideas:
 *   • Selector hooks (usePredictionHistory) hide context + defensive checks.
 *   • Normalization helpers (extractTimestamp / extractHomeWinProbability / buildGameLabel) keep render logic clean.
 *   • useMemo caches derived rows and summary stats so we don’t recompute on every render.
 *   • We fail gracefully: missing fields render "—" or "n/a" instead of throwing.
 */
import React, { useMemo } from "react";
import { usePredictionHistory } from "../hooks/predictionSelectors.js";

/* ---------- tiny utilities ---------- */

/** Return the first non-nullish value (null/undefined are skipped). */
const firstNonNullish = (...values) => values.find((v) => v != null);

/** Convert a probability in [0..1] to an integer percentage, or null if invalid. */
const toWholePercent = (prob) =>
  typeof prob === "number" ? Math.round(prob * 100) : null;
/** Safely coerce many timestamp shapes to a Date, or null if not present. */
const toDateOrNull = (value) => {
  if (value == null) return null;
  const d = value instanceof Date ? value : new Date(value);
  return isNaN(d.getTime()) ? null : d;
};

/* ---------- normalization helpers (keep render logic clean) ---------- */

/** Prefer `ts`, then `time`, then `game.ts`; return a Date or null. */
const extractTimestamp = (event) =>
  toDateOrNull(
    firstNonNullish(event?.ts, event?.time, event?.game?.ts, null)
  );

/**
 * Prefer `probs.home`, fallback to other sources; return [0..1] or null.
 * Note: we include probs.ensemble and probs.away so we can still render *some*
 * probability even if the shape varies between callers.
 */
const extractHomeWinProbability = (event) =>
  firstNonNullish(
    event?.probs?.home,
    event?.probs?.ensemble,
    event?.probs?.away,
    event?.home_win_probability,
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
 * HistoryChart component
 *
 * Props:
 *   - history (optional): if provided and is an array, it overrides context history.
 *     Otherwise we use history from PredictionContext via usePredictionHistory().
 */
export default function HistoryChart({ history: historyOverride }) {
  // 1) Get a normalized history array, preferring explicit `history` if valid.
  //    The selector hides the context access + "is this an array?" checks.
  const historyItems = usePredictionHistory(historyOverride);

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

    const averageHomeWinPercent = percentValues.length
      ? Math.round(
          percentValues.reduce((a, b) => a + b, 0) / percentValues.length
        )
      : null;

    const mostRecentDate = historyItems[0]
      ? extractTimestamp(historyItems[0])
      : null;

    return {
      totalCount: historyItems.length,
      mostRecentDate,
      averageHomeWinPercent,
    };
  }, [chartPoints, historyItems]);

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

  return (
    <section className="history-chart" aria-live="polite">
      <header>
        <h2>Prediction History</h2>
        <small>
          {statsSummary.totalCount} item(s)
          {statsSummary.mostRecentDate && (
            <>
              {" "}
              • last: {statsSummary.mostRecentDate.toLocaleString()}
            </>
          )}
          {statsSummary.averageHomeWinPercent != null && (
            <> • avg home win: {statsSummary.averageHomeWinPercent}%</>
          )}
        </small>
      </header>

      <ol className="history-points a-text-fade-slide">
        {chartPoints.slice(0, 16).map((row) => (
          <li key={row.index} title={row.label}>
            <code>
              {row.timestamp ? row.timestamp.toLocaleString() : "—"}
            </code>
            {" — "}
            <strong>
              {row.homeWinPercent != null ? `${row.homeWinPercent}%` : "n/a"}
            </strong>{" "}
            <em>({row.label})</em>
          </li>
        ))}
      </ol>
    </section>
  );
}
