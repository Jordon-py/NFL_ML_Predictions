// ==========================================
// File: frontend/src/components/HistoryChart.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react
// Notes: Presentation-focused component.
// ==========================================

// /frontend/src/components/HistoryChart.jsx
// @ts-nocheck

/**
 * HistoryChart - Educational Overview
 *
 * Purpose:
 *   Display a simple, accessible "history feed" of prediction events. Each row shows:
 *     - When the prediction was made (timestamp)
 *     - Home team win probability (as a whole percent)
 *     - A human-readable label for the game (e.g., "2025 W7 MIA@BUF")
 *
 * Data contract (input):
 *   Primary usage:
 *     <HistoryChart />
 *     - Reads history from props (empty array fallback).
 *
 *   Optional override:
 *     <HistoryChart history={arrayOfEvents} />
 *     - If `history` is provided and is an array, it is used directly.
 *
 *   Expected event shape (unified):
 *     {
 *       ts?: string | number | Date,               // primary timestamp
 *       time?: string | number | Date,             // optional alt timestamp
 *       home_win_probability?: number,             // probability in [0..1]
 *       season?: number,
 *       week?: number,
 *       away_team?: string,                        // e.g., "MIA"
 *       home_team?: string                         // e.g., "BUF"
 *     }
 *
 * Key ideas:
 *   - Simple props keep render logic easy to follow.
 *   - Normalization helpers (extractTimestamp / extractHomeWinProbability / buildGameLabel) keep render logic clean.
 *   - Derived rows and summary stats are computed inline for simplicity.
 *   - We fail gracefully: missing fields render "-" or "n/a" instead of throwing.
 */
import React from "react";

import { toWholePercent } from "../utils/predictionHelpers";

/* ---------- tiny utilities ---------- */

/** Safely coerce many timestamp shapes to a Date, or null if not present. */


const toDateOrNull = (value) => {
  if (value == null) return null;
  const d = value instanceof Date ? value : new Date(value);
  return isNaN(d.getTime()) ? null : d;
}

function extractHomeWinProbability(event) {
  if (!event) return null;
  return typeof event.home_win_probability === "number" ? event.home_win_probability : null;
}

function buildGameLabel(event, index) {
  if (!event) return `prediction-${index}`;
  const season = event.season ?? "";
  const weekRaw = event.week ?? "";
  const week = weekRaw ? `W${weekRaw}` : "";
  const away = event.away_team || "away";
  const home = event.home_team || "home";
  return `${season} ${week} ${away}@${home}`.trim();
}


/** Safely extract a Date object from various event shapes. */
function extractTimestamp(event) {
  if (!event) return null;
  const val = event.ts || event.timestamp || event.time || null;
  return toDateOrNull(val);
}

/**
 * HistoryChart component
 *
 * Props:
 *   - history (optional): if provided and is an array, it is used directly.
 */
export default function HistoryChart({ history: historyOverride = [] }) {
  const historyItems = Array.isArray(historyOverride) ? historyOverride : [];

  /**
   * chartPoints: normalized, render-ready rows
   */
  const chartPoints = historyItems.map((event, index) => {
    const timestamp = extractTimestamp(event);
    const prob = extractHomeWinProbability(event);
    const predictedHome = typeof event.home_score === "number" ? event.home_score : null;
    const predictedAway = typeof event.away_score === "number" ? event.away_score : null;
    const actualHome =
      typeof event.final_home_score === "number" ? event.final_home_score : null;
    const actualAway =
      typeof event.final_away_score === "number" ? event.final_away_score : null;
    const actualLabel =
      actualHome != null && actualAway != null ? `${actualHome}-${actualAway}` : null;
    const actualStatus = event.game_status;

    return {
      index,
      timestamp,
      homeWinPercent: toWholePercent(prob),
      label: buildGameLabel(event, index),
      predictedHome,
      predictedAway,
      actualLabel,
      actualStatus,
    };
  });

  /**
   * statsSummary: small header stats we can show users
   */
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

  const statsSummary = {
    totalCount: historyItems.length,
    mostRecentDate,
    averageHomeWinPercent,
  };

  /* ---------- render ---------- */

  if (chartPoints.length === 0) {
    return (
      <section className="history-chart" aria-live="polite">
        <header>
          <h2>Prediction History</h2>
          <small>0 saved</small>
        </header>
        <p>No saved predictions yet. Generate a forecast to start your history.</p>
      </section>
    );
  }

  return (
    <section className="history-chart" aria-live="polite">
      <header>
        <h2>Prediction History</h2>

        <small>
          {statsSummary.totalCount} saved

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
            <div className="history-score">
              <span>
                Pred: {row.predictedHome ?? "—"}-{row.predictedAway ?? "—"}
              </span>
              {row.actualLabel && (
                <span>
                  Final: {row.actualLabel}
                  {row.actualStatus ? ` (${row.actualStatus})` : ""}
                </span>
              )}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
