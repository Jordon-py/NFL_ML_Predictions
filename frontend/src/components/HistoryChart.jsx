import React, { useMemo, useState, useEffect } from "react";
import { Link } from "react-router-dom";

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

function buildReadableMatchup(event) {
  const away = event?.away_team || "Away";
  const home = event?.home_team || "Home";
  return {
    away,
    home,
    meta: [event?.season, event?.week != null ? `Week ${event.week}` : null]
      .filter(Boolean)
      .join(" • "),
  };
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
    matchup: buildReadableMatchup(event),
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
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [sortConfig, setSortConfig] = useState({ key: "date", direction: "desc" });
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const historyItems = Array.isArray(history) ? history : [];

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const rows = useMemo(
    () => historyItems.map((event, index) => normalizeHistoryRow(event, index)),
    [historyItems]
  );

  const filteredAndSortedRows = useMemo(() => {
    let result = rows;
    
    // Status Filter
    if (filter === "resolved") result = result.filter((row) => row.resolved);
    if (filter === "pending") result = result.filter((row) => !row.resolved);
    
    // Search Filter
    if (debouncedQuery.trim() !== "") {
      const q = debouncedQuery.toLowerCase();
      result = result.filter(row => 
        row.label.toLowerCase().includes(q)
      );
    }

    // Sort Logic
    result = [...result].sort((a, b) => {
      let aVal, bVal;
      
      switch (sortConfig.key) {
        case "confidence":
          aVal = a.confidence ?? 0;
          bVal = b.confidence ?? 0;
          break;
        case "margin":
          // Absolute margin delta - lower is better (more accurate)
          aVal = (a.actualDiff != null && a.predictedDiff != null) 
            ? Math.abs(a.actualDiff - a.predictedDiff) 
            : Infinity; // Unresolved or missing go to bottom
          bVal = (b.actualDiff != null && b.predictedDiff != null) 
            ? Math.abs(b.actualDiff - b.predictedDiff) 
            : Infinity;
          break;
        case "date":
        default:
          aVal = a.timestamp ? a.timestamp.getTime() : 0;
          bVal = b.timestamp ? b.timestamp.getTime() : 0;
          break;
      }

      if (aVal < bVal) return sortConfig.direction === "asc" ? -1 : 1;
      if (aVal > bVal) return sortConfig.direction === "asc" ? 1 : -1;
      return 0;
    });
    
    return result;
  }, [filter, debouncedQuery, sortConfig, rows]);

  const displayRows = filteredAndSortedRows.slice(0, visibleCount);
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
  const analysisNote =
    resolvedCount > 0
      ? `${resolvedCount} resolved games can be reviewed against final scores.`
      : "Pending predictions will become more useful once final scores sync.";

  if (rows.length === 0) {
    return (
      <section className="history-chart" aria-live="polite">
        <header className="history-chart__header">
          <div>
            <h2>Prediction History</h2>
            <small>0 saved</small>
          </div>
        </header>
        <div className="history-chart__empty">
          <strong>No saved predictions yet.</strong>
          <p>Generate a forecast from the dashboard to build a reviewable model history.</p>
          <Link className="history-chart__emptyAction" to="/app">Go to dashboard</Link>
        </div>
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
        <p className="history-chart__insight">{analysisNote}</p>
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
        <div className="history-chart__search">
          <div className="history-chart__searchWrapper">
            <input 
              type="text" 
              placeholder="Search by team (e.g. KC, BUF)" 
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setVisibleCount(PAGE_SIZE);
              }}
              className="history-chart__searchInput"
              aria-label="Search predictions by team"
            />
            {searchQuery && (
              <button 
                type="button" 
                className="history-chart__clearSearch" 
                onClick={() => {
                  setSearchQuery("");
                  setDebouncedQuery("");
                }}
                aria-label="Clear search"
              >
                ✕
              </button>
            )}
          </div>
        </div>

        <div className="history-chart__controls">
          <select 
            className="history-chart__sortSelect"
            value={`${sortConfig.key}-${sortConfig.direction}`}
            onChange={(e) => {
              const [key, direction] = e.target.value.split("-");
              setSortConfig({ key, direction });
              setVisibleCount(PAGE_SIZE);
            }}
            aria-label="Sort predictions"
          >
            <option value="date-desc">Newest First</option>
            <option value="date-asc">Oldest First</option>
            <option value="confidence-desc">Highest Confidence</option>
            <option value="confidence-asc">Lowest Confidence</option>
            <option value="margin-asc">Most Accurate (Margin)</option>
            <option value="margin-desc">Least Accurate (Margin)</option>
          </select>
        </div>

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
          Showing {displayRows.length} of {filteredAndSortedRows.length}
        </p>
      </div>

      <div className="history-chart__grid">
        {displayRows.map((row) => {
          const resultLabel = !row.resolved ? "Pending" : row.correct ? "Correct" : "Missed";
          const marginDelta =
            row.actualDiff != null && row.predictedDiff != null
              ? `${(row.actualDiff - row.predictedDiff).toFixed(1)} pts`
              : "n/a";
          const predictedWinner =
            row.homeWinProbability != null || row.awayWinProbability != null
              ? row.homeWinProbability >= row.awayWinProbability
                ? row.matchup.home
                : row.matchup.away
              : row.predictedDiff != null
                ? row.predictedDiff >= 0
                  ? row.matchup.home
                  : row.matchup.away
                : "n/a";
          const confidencePercent =
            typeof row.confidence === "number" ? Math.round(row.confidence * 100) : null;
          return (
            <article key={row.id} className="history-card" title={row.label}>
              <div className="history-card__header">
                <span className="history-card__date">{row.timestamp ? row.timestamp.toLocaleString() : "No timestamp"}</span>
                <span className={`history-card__badge history-card__badge--${resultLabel.toLowerCase()}`}>
                  {resultLabel}
                </span>
              </div>

              <div className="history-card__matchup">
                <span className="history-card__team">{row.matchup.away}</span>
                <span className="history-card__at">@</span>
                <span className="history-card__team">{row.matchup.home}</span>
              </div>
              <span className="history-card__meta">{row.matchup.meta}</span>
              <span className="history-card__compactLabel">{row.label}</span>

              <div className="history-card__takeaway">
                <span>Projected winner</span>
                <strong>{predictedWinner}</strong>
                {confidencePercent != null ? (
                  <div className="history-card__confidenceTrack" aria-label={`Confidence ${confidencePercent}%`}>
                    <span style={{ width: `${confidencePercent}%` }} />
                  </div>
                ) : null}
              </div>

              <div className="history-card__stats">
                <div className="history-card__statRow">
                  <span className="history-card__statLabel">Prediction</span>
                  <span className="history-card__statValue">{row.predictedHome ?? "—"} - {row.predictedAway ?? "—"}</span>
                </div>
                <div className="history-card__statRow">
                  <span className="history-card__statLabel">Final</span>
                  <span className="history-card__statValue">{row.resolved ? `${row.actualHome} - ${row.actualAway}` : "Pending"}</span>
                </div>
                <div className="history-card__statRow">
                  <span className="history-card__statLabel">Confidence</span>
                  <span className="history-card__statValue">{formatPercent(row.confidence)}</span>
                </div>
                <div className="history-card__statRow">
                  <span className="history-card__statLabel">Margin delta</span>
                  <span className="history-card__statValue">{marginDelta}</span>
                </div>
                {row.gameStatus && (
                  <div className="history-card__statRow">
                    <span className="history-card__statLabel">Status</span>
                    <span className="history-card__statValue">{row.gameStatus}</span>
                  </div>
                )}
              </div>
            </article>
          );
        })}
        {displayRows.length === 0 && (
          <div className="history-chart__noResults">
            <strong>No matching predictions.</strong>
            <p>Try a different team code, clear search, or switch the result filter.</p>
            <button
              type="button"
              className="history-chart__moreButton"
              onClick={() => {
                setSearchQuery("");
                setDebouncedQuery("");
                setFilter("all");
              }}
            >
              Reset filters
            </button>
          </div>
        )}
      </div>

      {displayRows.length < filteredAndSortedRows.length ? (
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
