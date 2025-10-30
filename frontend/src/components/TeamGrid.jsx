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

import { getNextWeekSchedule, predictGame } from '../api/client.js';
import React, { useState, useEffect, useCallback } from 'react';

// ---- Small stateless utilities (safe at module scope) -----------------------

// Stable game key: used for maps (loading/predictions) and history joins.
const makeKey = (g) => `${g.season}-${g.week}-${g.home_abbr}-${g.away_abbr}`;

// Keyboard activation helper: support Enter and Space to trigger a card "click".
const isActionKey = (key) => key === 'Enter' || key === ' ';

// Intl date formatter (US-style short labels). Keep outside component to avoid re-creation.
const kickoffFormatter = new Intl.DateTimeFormat('en-US', {
  weekday: 'short',
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
  hour12: true,
});

// Accepts a schedule row and returns a human-friendly kickoff string.
// Falls back to the raw value if the date cannot be parsed.
const formatKickoffTime = (row) => {
  const iso = row.kickoff_ts_utc || row.kickoff_iso || row.kickoff || null;
  if (!iso) return 'TBA';
  try {
    const d = new Date(iso);
    // Guard against "Invalid Date" (NaN time).
    if (Number.isNaN(+d)) return String(iso);
    return kickoffFormatter.format(d);
  } catch {
    return String(iso);
  }
};

// Minimal CSV parser for "teamName,abbr,logoUrl" (headerless).
// Trims whitespace, skips blanks, and ignores lines without an abbreviation.
const parseTeamCsv = (csvText) =>
  csvText
    .trim()
    .split('\n')
    .slice(1) // skip header row if present; harmless if not
    .reduce((acc, rawLine) => {
      const line = rawLine.trim();
      if (!line) return acc; // skip empty lines
      const parts = line.split(',').map((v) => v.trim());
      const [teamName, abbr, logoUrl] = parts;
      if (abbr) acc[abbr] = { name: teamName || abbr, abbr, logoUrl };
      return acc;
    }, {});

// ---- Component --------------------------------------------------------------

export default function TeamGrid() {
  // Team metadata map, upcoming games, prediction map, and persisted history.
  const [teams, setTeams] = useState({});
  const [schedule, setSchedule] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [history, setHistory] = useState([]);

  // Loading flags:
  // - "teams"/"schedule" for bootstrap steps
  // - dynamic keys per game for in-flight predictions
  const [loading, setLoading] = useState({ teams: true, schedule: true });

  // Single error string surfaced to the page-level error panel.
  const [error, setError] = useState(null);

  // 1) Load team metadata (CSV) on mount.
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/data/myteamdescriptions.csv');
        if (!res.ok) throw new Error('Failed to load team data');
        const text = await res.text();
        setTeams(parseTeamCsv(text));
      } catch (err) {
        console.error('[TeamGrid] loadTeams:', err);
        setError('Failed to load team data');
      } finally {
        // Mark the bootstrap step as done regardless of outcome.
        setLoading((s) => ({ ...s, teams: false }));
      }
    })();
  }, []);

  // 2) Hydrate recent prediction history from localStorage (best-effort, no error UI).
  useEffect(() => {
    try {
      const raw = localStorage.getItem('prediction_history');
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

      try {
        // Normalize payload to API expectations (prefer *_abbr).
        const payload = {
          home_team: game.home_abbr || game.home_team,
          away_team: game.away_abbr || game.away_team,
          season: Number(game.season),
          week: Number(game.week),
        };

        // Call the model API and coerce numeric fields defensively.
        const res = await predictGame(payload);
        const result = {
          home_score: Number(res.home_score),
          away_score: Number(res.away_score),
          point_diff: Number(res.point_diff),
          home_win_probability: Number(res.home_win_probability),
          away_win_probability: Number(res.away_win_probability),
        };

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
            home: result.home_win_probability,
            away: result.away_win_probability,
            // Maintain "ensemble" for downstream charts expecting this key.
            ensemble: result.home_win_probability,
          },
        };
        setHistory((h) => {
          const next = [...h, entry].slice(-100);
          try {
            localStorage.setItem('prediction_history', JSON.stringify(next));
          } catch {
            // Non-fatal: storage quota or availability.
          }
          return next;
        });
      } catch (e) {
        console.error('[TeamGrid] predictGame:', e);
        setError('Failed to get prediction');
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
      <div className="team-grid-cards a-shine">
        {schedule.map((game, index) => {
          const key = makeKey(game);
          const prediction = predictions[key];
          const isLoading = !!loading[key];

          return (
            // CSS var --i allows staggered or shimmer animations if desired
            <div key={key} style={{ '--i': index }}>
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
                          e.currentTarget.style.display = 'none';
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
                          e.currentTarget.style.display = 'none';
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
                    <div>
                      Home win: {(prediction.home_win_probability * 100).toFixed(0)}%
                    </div>
                    <div>Point diff: {prediction.point_diff.toFixed(1)}</div>
                    <div>
                      Score: {prediction.home_score}–{prediction.away_score}
                    </div>
                  </div>
                ) : (
                  <div className="cta">Click to predict</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
