// @ts-nocheck
// /frontend/src/components/TeamGrid.jsx
// -----------------------------------------------------------------------------
// TeamGrid — Next-Week Matchups with One-Click Predictions
//
// PURPOSE
//   Render the upcoming NFL matchups as interactive cards. When a user
//   clicks (or presses Enter/Space) on a card, we call the prediction API,
//   display the result (win prob, point diff, score), and persist a
//   lightweight history for later charts.
//
// INPUTS (via dependencies)
//   - getNextWeekSchedule(): Promise<Game[]>
//       Game: { season, week, home_abbr, away_abbr, kickoff_* ... }
//   - predictGame(payload): Promise<{
//       home_score, away_score, point_diff,
//       home_win_probability, away_win_probability
//     }>
//
// SIDE DATA / PERSISTENCE
//   - Fetches team metadata CSV: /data/myteamdescriptions.csv
//       Format (headerless): teamName,abbr,logoUrl
//   - Reads/writes localStorage key "prediction_history" (kept to last 100)
//
// OUTPUTS
//   - Accessible card grid; each card shows teams, kickoff time, and either
//     a CTA (“Click to predict”) or the model’s prediction once available.
//
// ERROR & LOADING BEHAVIOR
//   - Distinguishes bootstrap loading (teams/schedule) from per-card loading.
//   - Displays a generic error with a “Retry” if any bootstrap step fails.
//
// ACCESSIBILITY
//   - Cards are keyboard-activable with role="button" and onKeyDown handler.
//
// -----------------------------------------------------------------------------
// NOTES ON EXTENSION
//   - Safe to add: market odds overlays, filters, pagination, skeleton states.
//   - If CSV might contain commas inside names, consider a robust CSV parser.
// -----------------------------------------------------------------------------

import { getNextWeekSchedule, predictGame, ApiError } from '../api/client.js';
import { addLog } from '../api/debugLog.js';
import React, { useState, useEffect, useCallback } from 'react';
import PredictionResult from './PredictionResult.jsx';
import './TeamGrid.css';

let teamNames = {
  "ARI": "Arizona Cardinals",
  "ATL": "Atlanta Falcons",
  "BAL": "Baltimore Ravens",
  "BUF": "Buffalo Bills",
  "CAR": "Carolina Panthers",
  "CHI": "Chicago Bears",
  "CIN": "Cincinnati Bengals",
  "CLE": "Cleveland Browns",
  "DAL": "Dallas Cowboys",
  "DEN": "Denver Broncos",
  "DET": "Detroit Lions",
  "GB": "Green Bay Packers",
  "HOU": "Houston Texans",
  "IND": "Indianapolis Colts",
  "JAX": "Jacksonville Jaguars",
  "KC": "Kansas City Chiefs",
  "LAC": "Los Angeles Chargers",
  "LAR": "Los Angeles Rams",
  "LV": "Las Vegas Raiders",
  "MIA": "Miami Dolphins",
  "MIN": "Minnesota Vikings",
  "NE": "New England Patriots",
  "NO": "New Orleans Saints",
  "NYG": "New York Giants",
  "NYJ": "New York Jets",
  "PHI": "Philadelphia Eagles",
  "PIT": "Pittsburgh Steelers",
  "SEA": "Seattle Seahawks",
  "SF": "San Francisco 49ers",
  "TB": "Tampa Bay Buccaneers",
  "TEN": "Tennessee Titans",
  "WAS": "Washington Commanders"
}


// ---- Small stateless utilities (safe at module scope) -----------------------

// Stable game key: used for maps (loading/predictions) and history joins.
const makeKey = (g) => `${g.season}-${g.week}-${g.home_abbr}-${g.away_abbr}`;

// Keyboard activation helper: support Enter and Space to trigger a card "click".
const isActionKey = (key) => key === 'Enter' || key === ' ';

/**
 * @param {Object} row
 * @returns {string}
 */
// Attempts to format the kickoff time using several possible fields; 
// if the date is malformed or missing, returns the raw value as a fallback.
const formatKickoffTime = (row) => {
  const iso = row.kickoff_ts_utc || row.kickoff_iso || row.kickoff || null;
  try {
    return kickoffFormatter.format(new Date(iso));
  } catch {
    return iso ? String(iso) : 'TBD';
  }
};

/**
 * @param {string} csvText
 * @returns {Object}
 */
const parseTeamCsv = (csvText) =>
  csvText.trim().split('\n').slice(1).reduce((acc, line) => {
    const [teamName, abbr, logoUrl] = line.split(',').map((v) => v.trim());
    if (abbr) acc[abbr] = { name: teamName, abbr, logoUrl };
    return acc;
  }, {});

// Format numeric scores to one decimal place when possible.
const formatScore = (value) => {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(1) : value;
};

// Add a stable local-time formatter used by formatKickoffTime
const kickoffFormatter = new Intl.DateTimeFormat(undefined, {
  weekday: 'short',
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
});

// -----------------------------------------------------------------------------

/**
 * TeamGrid Component
 * ------------------
 * Renders a grid of upcoming NFL matchups for the next week, allowing users to view team details and request machine learning-based game predictions.
 * - Loads team metadata from a CSV file and schedule data from the backend API.
 * - Handles prediction requests for each matchup, displaying scores and win probabilities.
 * - Manages loading and error states for robust user experience.
 * Dependencies: React, getNextWeekSchedule, predictGame (API client).
 */
export default function TeamGrid() {
  const [teams, setTeams] = useState({});
  const [schedule, setSchedule] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [predictErrors, setPredictErrors] = useState({}); // per-game prediction errors (do not nuke the whole grid)
  const [history, setHistory] = useState([]);
  // Lightweight toast notifications (stack top-right)
  const [toasts, setToasts] = useState([]);

  // Auto-dismiss toasts after ~5.5s
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      setToasts((ts) => ts.filter((t) => now - t.createdAt < 5500));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // Loading flags:
  // - "teams"/"schedule" for bootstrap steps
  // - dynamic keys per game for in-flight predictions
  const [loading, setLoading] = useState({ teams: true, schedule: true });

  // Single error string surfaced to the page-level error panel.
  const [error, setError] = useState(null);

  // 1) Load team metadata (CSV) on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/data/myteamdescriptions.csv');
        if (!res.ok) throw new Error('Failed to load team data');
        const text = await res.text();
        if (!cancelled) setTeams(parseTeamCsv(text));
      } catch (err) {
        console.error('[TeamGrid] loadTeams:', err);
        if (!cancelled) setError('Failed to load team data');
      } finally {
        if (!cancelled) setLoading((s) => ({ ...s, teams: false }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Load persisted history from localStorage on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem("prediction_history");
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) setHistory(parsed.slice(-100));
      }
    } catch (err) {
      // Non-fatal: localStorage can be unavailable or corrupted.
      console.debug('[TeamGrid] history restore failed:', err);
    }
  }, []);

  // 3) Load next week schedule on mount.
  useEffect(() => {
    (async () => {
      try {
        const games = await getNextWeekSchedule();
        setSchedule(Array.isArray(games) ? games : []);
      } catch (err) {
        console.error('[TeamGrid] loadSchedule:', err);
        setError('Failed to load schedule');
        setSchedule([]);
      } finally {
        setLoading((s) => ({ ...s, schedule: false }));
      }
    })();
  }, []);

  // Predict handler (click/keyboard). Guard against concurrent duplicate requests.
  const handlePredict = useCallback(
    async (game) => {
      const key = makeKey(game);
      // If this game is already in-flight, ignore subsequent triggers.
      if (loading[key]) return;

      // Mark this specific game as loading.
      setLoading((s) => ({ ...s, [key]: true }));
      setError(null);

      // Compute payload up-front so we can log it in catch
      const payload = {
        home_team: game.home_abbr || game.home_team,
        away_team: game.away_abbr || game.away_team,
        season: Number(game.season),
        week: Number(game.week),
      };

      try {
        // Client-side validation to avoid obvious 400s
        if (!payload.home_team || !payload.away_team) {
          throw new ApiError('Missing team abbreviations', { status: 400 });
        }
        if (payload.home_team === payload.away_team) {
          throw new ApiError('Home and away teams must differ', { status: 400 });
        }
        if (!Number.isFinite(payload.season) || !Number.isFinite(payload.week)) {
          throw new ApiError('Season and week must be numbers', { status: 400 });
        }

        // Call the model API and coerce numeric fields defensively.
        const res = await predictGame(payload);
        const result = {
          home_score: Number(res.home_score),
          away_score: Number(res.away_score),
          point_diff: Number(res.point_diff),
          home_win_probability: Number(res.home_win_probability),
          away_win_probability: Number(res.away_win_probability),
          mode: String(res.mode || ''),
          prediction_source: String(res.prediction_source || ''),
          // New: classifier usage telemetry from backend
          win_classifier_used: typeof res.win_classifier_used === 'boolean' ? res.win_classifier_used : undefined,
          win_probability_source: typeof res.win_probability_source === 'string' ? res.win_probability_source : '',
          win_threshold_used: (res.win_threshold_used ?? null) !== null ? Number(res.win_threshold_used) : null,
        };

        // Clear any prior error for this game on success and update prediction map.
        setPredictErrors((errs) => {
          const { [key]: _old, ...rest } = errs;
          return rest;
        });
        // Update in-memory predictions map.
        setPredictions((p) => ({ ...p, [key]: result }));

        // Persist to history (bounded to last 100).
        const entry = {
          ts: new Date().toISOString(),
          game: {
            season: payload.season,
            week: payload.week,
            home_abbr: payload.home_team,
            away_abbr: payload.away_team,
          },
          probs: {
            home_score: result.home_score,
            away_score: result.away_score,
            home: result.home_win_probability,
            away: result.away_win_probability,
            // Maintain "ensemble" for downstream charts expecting this key.
            ensemble: result.home_win_probability,
          },
        };

        setHistory((hist) => {
          // FIX: correct array spread; previously `[hist, ...entry]` caused a runtime error
          const next = [...hist, entry].slice(-100);
          try {
            localStorage.setItem('prediction_history', JSON.stringify(next));
          } catch {
            // Non-fatal: storage quota or availability.
          }
          return next;
        });
      } catch (e) {
        console.error('[TeamGrid] predictGame:', e);
        // Do not set the global page error for per-card prediction failures; show inline card error instead.
        let msg = 'Failed to get prediction';
        if (e instanceof ApiError) {
          if (e.details && typeof e.details === 'object' && e.details.detail) msg = String(e.details.detail);
          else if (typeof e.details === 'string' && e.details.trim()) msg = e.details;
          else if (e.status === 400) msg = 'Bad request — please check teams, season, and week.';
          if (/fallback/i.test(msg)) {
            msg = 'Prediction requires server fallback and is currently disabled.';
          }
        } else if (e && typeof e.message === 'string' && e.message.trim()) {
          msg = e.message;
        }
        setPredictErrors((m) => ({ ...m, [key]: msg }));
        try {
          addLog({
            level: 'error',
            where: 'TeamGrid.handlePredict',
            key,
            payload, // payload is now always in scope
            message: msg,
          });
        } catch { }
        // Toast (auto-dismiss)
        setToasts((t) => [
          ...t,
          { id: `${Date.now()}-${Math.random()}`, type: 'error', message: msg, createdAt: Date.now() },
        ]);
      } finally {
        // Clear per-game loading flag.
        setLoading((s) => ({ ...s, [key]: false }));
      }
    },
    [loading]
  );

  // ---- Render guards --------------------------------------------------------

  if (error) {
    return (
      <div className="team-grid-error">
        <h3>Error Loading Data</h3>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  // While bootstrap schedule is loading, show a friendly spinner.
  if (loading.schedule) {
    return (
      <div className="team-grid-loading">
        <div className="loading-spinner" />
        <p>Loading next week’s matchups…</p>
      </div>
    );
  }

  // If schedule is empty (and not loading), show an empty state rather than a spinner.
  if (!schedule.length) {
    return (
      <div className="team-grid-empty">
        <p>No upcoming matchups found.</p>
      </div>
    );
  }

  // ---- Main grid ------------------------------------------------------------

  return (
    <div className="team-grid-section">
      {/* Toasts container (top-right) */}
      <div className="toast-container">
        {toasts.map((t) => (
          <div
            key={t.id}
            role="status"
            className={`toast ${t.type === 'error' ? 'toast--error' : 'toast--info'}`}
          >
            {t.message}
          </div>
        ))}
      </div>
      <div className="team-grid-cards a-shine">
        {schedule.map((game, index) => {
          const key = makeKey(game);
          const prediction = predictions[key];
          const isLoading = !!loading[key];

          return (
            // CSS var --i allows staggered or shimmer animations if desired
            <div key={key} className="grid-item">
              <div
                className={`card hover inner-card sb3__content ${prediction ? 'has-prediction' : ''
                  } ${isLoading ? 'loading' : ''}`}
                onClick={() => handlePredict(game)}
                onKeyDown={(e) => {
                  if (isActionKey(e.key)) {
                    e.preventDefault();
                    handlePredict(game);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-pressed={isLoading}
                aria-label={`Predict ${game.away_abbr} at ${game.home_abbr}`}
              >
                <header className="matchup-head">
                  <div className="teams-row">
                    {/* AWAY team */}
                    <div className="team-info away">
                      {/* Hide broken logos to avoid layout jank */}
                      <img
                        src={teams[game.away_abbr]?.logoUrl}
                        alt={`${game.away_abbr} logo`}
                        className="team-logo"
                        onError={(e) => {
                          try { e.currentTarget.classList.add('is-hidden'); } catch { }
                        }}
                      />
                      <strong>{game.away_abbr}</strong>
                    </div>

                    <span className="at-symbol">@</span>

                    {/* HOME team */}
                    <div className="team-info home">
                      <img
                        src={teams[game.home_abbr]?.logoUrl}
                        alt={`${game.home_abbr} logo`}
                        className="team-logo"
                        onError={(e) => {
                          try { e.currentTarget.classList.add('is-hidden'); } catch { }
                        }}
                      />
                      <strong>{game.home_abbr}</strong>
                    </div>
                  </div>

                  <span className="kickoff">{formatKickoffTime(game)}</span>
                </header>

                {/* Body switches between CTA, loading, and prediction result */}
                {isLoading ? (
                  <div className="prediction loading-line">Predicting…</div>
                ) : prediction ? (
                  <div className="prediction">
                    {/* tiny source badge for transparency */}
                    {prediction.prediction_source ? (
                      <div
                        className="source-badge"
                        title={`Mode: ${prediction.mode || 'production'}`}
                      >
                        {prediction.prediction_source}
                      </div>
                    ) : null}
                    {/* classifier usage badge: 'clf' when model produced probability, 'legacy' if fallback */}
                    {typeof prediction.win_classifier_used === 'boolean' ? (
                      <div
                        className="source-badge"
                        title={`Win prob via: ${prediction.win_probability_source || 'unknown'}${typeof prediction.win_threshold_used === 'number' ? ` • thr=${prediction.win_threshold_used.toFixed(2)}` : ''
                          }`}
                      >
                        {prediction.win_classifier_used ? 'clf' : 'legacy'}
                      </div>
                    ) : null}
                    <div>
                      Home win: {(prediction.home_win_probability * 100).toFixed(0)}%
                    </div>
                    <div>Point diff: {prediction.point_diff.toFixed(1)}</div>
                    <div>
                      Score: {game.away_abbr} {formatScore(prediction.away_score)} – {formatScore(prediction.home_score)} {game.home_abbr}
                    </div>
                  </div>
                ) : predictErrors[key] ? (
                  <div className="prediction error" role="status" aria-live="polite">
                    <div style={{ color: '#b00020' }}>{predictErrors[key]}</div>
                    <button
                      className="retry-btn"
                      onClick={(e) => { e.stopPropagation(); handlePredict(game); }}
                    >
                      Retry
                    </button>
                  </div>
                ) : (
                  <div className="cta">Click to predict</div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Saved prediction history */}
      {history.length > 0 && history.some((h) => h && Object.keys(h).length > 0) && (
        <div className="prediction-history">
          <h3>Saved Predictions</h3>
          <div className="history-list">
            {history
              .filter(
                (hist) =>
                  hist &&
                  typeof hist === 'object' &&
                  Object.keys(hist).length > 0 &&
                  hist.game &&
                  hist.probs
              )
              .map((hist, i) => (
                <div key={`history-${i}`} className="history-entry">
                  <PredictionResult entry={{ ...hist }} />
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
// Change Log (2025-02-14): Streamlined metadata loading, reinstated PredictionResult import, and tightened JSX boundaries to restore type-safe rendering.
