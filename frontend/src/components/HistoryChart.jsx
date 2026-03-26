import React, { useMemo, useState } from "react";

const PAGE_SIZE = 12;

function toDateOrNull(value) {
  if (value == null) return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function toFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatPercent(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "n/a";
}

function buildGameLabel(event, index) {
  if (!event) return `prediction-${index}`;
  const season = event.season ?? "";
  const week = event.week != null ? `W${event.week}` : "";
  const away = event.away_team || "away";
  const home = event.home_team || "home";
  return `${season} ${week} ${away}@${home}`.trim();
}

function normalizeHistoryRow(event, index) {
  const predictedHome = toFiniteNumber(event?.home_score);
  const predictedAway = toFiniteNumber(event?.away_score);
  const actualHome = toFiniteNumber(event?.final_home_score);
  const actualAway = toFiniteNumber(event?.final_away_score);
  const homeWinProbability = toFiniteNumber(event?.home_win_probability);
  const awayWinProbability = toFiniteNumber(event?.away_win_probability);
  const predictedDiff =
    predictedHome != null && predictedAway != null ? predictedHome - predictedAway : null;
  const actualDiff =
    actualHome != null && actualAway != null ? actualHome - actualAway : null;
  const resolved = actualHome != null && actualAway != null;
  const predictedHomeWins =
    homeWinProbability != null || awayWinProbability != null
      ? (homeWinProbability ?? 0) >= (awayWinProbability ?? 0)
      : predictedDiff != null
        ? predictedDiff >= 0
        : null;
  const actualHomeWins = resolved ? actualHome > actualAway : null;
  const correct =
    resolved && predictedHomeWins != null && actualHomeWins != null
      ? predictedHomeWins === actualHomeWins
      : null;

  return {
    id: event?.game_id || `${buildGameLabel(event, index)}-${index}`,
    timestamp: toDateOrNull(event?.ts || event?.timestamp || event?.time || null),
    label: buildGameLabel(event, index),
    predictedHome,
    predictedAway,
    actualHome,
    actualAway,
    predictedDiff,
    actualDiff,
    confidence:
      homeWinProbability != null || awayWinProbability != null
        ? Math.max(homeWinProbability ?? 0, awayWinProbability ?? 0)
        : null,
    homeWinProbability,
    awayWinProbability,
    resolved,
    correct,
    gameStatus: event?.game_status || null,
  };
}

function SummaryTile({ label, value, detail }) {
  return (
    <article className="history-chart__summaryTile">
      <span className="history-chart__summaryLabel">{label}</span>
      <strong className="history-chart__summaryValue">{value}</strong>
      {detail ? <span className="history-chart__summaryDetail">{detail}</span> : null}
    </article>
  );
}

export default function HistoryChart({ history = [], summary = null }) {
  const [filter, setFilter] = useState("all");
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const historyItems = Array.isArray(history) ? history : [];

  const rows = useMemo(
    () => historyItems.map((event, index) => normalizeHistoryRow(event, index)),
    [historyItems]
  );

  const filteredRows = useMemo(() => {
    if (filter === "resolved") return rows.filter((row) => row.resolved);
    if (filter === "pending") return rows.filter((row) => !row.resolved);
    return rows;
  }, [filter, rows]);

  const displayRows = filteredRows.slice(0, visibleCount);
  const mostRecent = rows[0]?.timestamp || null;
  const resolvedCount =
    typeof summary?.resolved_games === "number"
      ? summary.resolved_games
      : rows.filter((row) => row.resolved).length;
  const totalPredictions =
    typeof summary?.total_predictions === "number" ? summary.total_predictions : rows.length;
  const winRate = typeof summary?.win_rate === "number" ? `${Math.round(summary.win_rate * 100)}%` : "n/a";
  const avgSpreadError =
    typeof summary?.avg_abs_spread_error === "number"
      ? `${summary.avg_abs_spread_error.toFixed(1)} pts`
      : "n/a";

  if (rows.length === 0) {
    return (
      <section className="history-chart" aria-live="polite">
        <header className="history-chart__header">
          <h2>Prediction History</h2>
          <small>0 saved</small>
        </header>
        <p>No saved predictions yet. Generate a forecast to start your history.</p>
      </section>
    );
  }

  return (
    <section className="history-chart" aria-live="polite">
      <header className="history-chart__header">
        <div>
          <h2>Prediction History</h2>
          <small>
            {totalPredictions} saved
            {mostRecent ? ` • last: ${mostRecent.toLocaleString()}` : ""}
          </small>
        </div>
      </header>

      <section className="history-chart__summary" aria-label="History summary">
        <SummaryTile label="Resolved" value={resolvedCount} detail="Games with final scores" />
        <SummaryTile label="Win rate" value={winRate} detail="Correct winner calls" />
        <SummaryTile label="Spread error" value={avgSpreadError} detail="Average absolute margin miss" />
        <SummaryTile
          label="Confidence"
          value={formatPercent(summary?.avg_confidence ?? null)}
          detail="Average top-side probability"
        />
      </section>

      <div className="history-chart__toolbar">
        <div className="history-chart__filters" role="tablist" aria-label="History filters">
          {[
            { key: "all", label: "All" },
            { key: "resolved", label: "Resolved" },
            { key: "pending", label: "Pending" },
          ].map((option) => (
            <button
              key={option.key}
              type="button"
              className={`history-chart__filter ${filter === option.key ? "is-active" : ""}`}
              onClick={() => {
                setFilter(option.key);
                setVisibleCount(PAGE_SIZE);
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
        <p className="history-chart__toolbarText">
          Showing {displayRows.length} of {filteredRows.length}
        </p>
      </div>

      <ol className="history-points a-text-fade-slide">
        {displayRows.map((row) => {
          const resultLabel = !row.resolved ? "Pending" : row.correct ? "Correct" : "Missed";
          const marginDelta =
            row.actualDiff != null && row.predictedDiff != null
              ? `${(row.actualDiff - row.predictedDiff).toFixed(1)} pts`
              : "n/a";
          return (
            <li key={row.id} className="history-chart__item" title={row.label}>
              <div className="history-chart__itemHeader">
                <code>{row.timestamp ? row.timestamp.toLocaleString() : "—"}</code>
                <span className={`history-chart__pill history-chart__pill--${resultLabel.toLowerCase()}`}>
                  {resultLabel}
                </span>
              </div>

              <strong className="history-chart__label">{row.label}</strong>

              <div className="history-score">
                <span>Pred: {row.predictedHome ?? "—"}-{row.predictedAway ?? "—"}</span>
                <span>Final: {row.resolved ? `${row.actualHome}-${row.actualAway}` : "Pending"}</span>
                <span>Confidence: {formatPercent(row.confidence)}</span>
                <span>Margin delta: {marginDelta}</span>
                {row.gameStatus ? <span>Status: {row.gameStatus}</span> : null}
              </div>
            </li>
          );
        })}
      </ol>

      {displayRows.length < filteredRows.length ? (
        <div className="history-chart__footer">
          <button
            type="button"
            className="history-chart__moreButton"
            onClick={() => setVisibleCount((count) => count + PAGE_SIZE)}
          >
            Show more
          </button>
        </div>
      ) : null}
    </section>
  );
}
