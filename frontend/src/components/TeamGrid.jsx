// @ts-nocheck
// TeamGrid.jsx
import React, {useState, useEffect} from 'react';
import {getNextWeekSchedule, predictGame} from '../api/client.js';
import PredictionResult from './PredictionResult';

const kickoffFormatter = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/Los_Angeles',
  weekday: 'short',
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
  hour12: true,
});

/**
 * @param {Object} row
 * @returns {string}
 */
// Attempts to format the kickoff time using several possible fields; 
// if the date is malformed or missing, returns the raw value as a fallback.
const formatKickoffTime = (row) => {
  const iso = row.kickoff_ts_utc || row.kickoff_iso || row.kickoff || null;
  try {return kickoffFormatter.format(new Date(iso));} catch {return String(iso);}

};

/**
 * @param {string} csvText
 * @returns {Object}
 */
const parseTeamCsv = (csvText) =>
  csvText.trim().split('\n').slice(1).reduce((acc, line) => {
    const [teamName, abbr, logoUrl] = line.split(',').map((v) => v.trim());
    if (abbr) acc[abbr] = {name: teamName, abbr, logoUrl};
    return acc;
  }, {});

const normalizeSchedulePayload = (data) => {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.games)) return data.games;
  throw new Error('Schedule payload is malformed.');
};

const isActionKey = (key) => key === 'Enter' || key === ' ';

const makeKey = (g) => `${g.season}-${g.week}-${g.home_abbr}-${g.away_abbr}`;

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
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/data/myteamdescriptions.csv');
        if (!res.ok) throw new Error('Failed to load team data');
        setTeams(parseTeamCsv(await res.text()));
      } catch (err) {
        console.error('[TeamGrid] loadTeams:', err);
        setError('Failed to load team data');
      }
    })();
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
      console.debug("Failed to load prediction history from localStorage", err);
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        setSchedule(normalizeSchedulePayload(await getNextWeekSchedule()));
      } catch (err) {
        console.error('[TeamGrid] loadSchedule:', err);
        setError('Failed to load schedule');
        setSchedule([]);
      }
    })();
  }, []);

  const handlePredict = async (game) => {
    const key = makeKey(game);
    if (loading[key]) return;
    setLoading((s) => ({...s, [key]: true}));
    setError(null);

    try {
      const payload = {
        // Send both abbreviation (preferred) and fallback full name. Backend will normalize.
        // Also include raw fields some backend builds expect (rest days) so the
        // prediction endpoint can construct features if available.
        home_abbr: game.home_abbr || game.home_team,
        away_abbr: game.away_abbr || game.away_team,
        season: Number(game.season),
        week: Number(game.week),
      };
      const {home_score, away_score, home_win_probability, away_win_probability, point_diff} = await predictGame(payload);
      const result = {
        home_score: Number(home_score),
        away_score: Number(away_score),
        point_diff: Number(point_diff),
        home_win_probability: Number(home_win_probability),
        away_win_probability: Number(away_win_probability),
      };
      setPredictions((p) => ({...p, [key]: result}));
    } catch (e) {
      console.error('[TeamGrid] predictGame:', e);
      setError('Failed to get prediction');
    } finally {
      setLoading((s) => ({...s, [key]: false}));
    }
  };

  const renderTeam = (abbr, isHome) => {
    const team = teams[abbr];
    if (!team) {
      return (
        <div className="team-placeholder">
          <div className="team-logo-placeholder">{abbr}</div>
          <span className="team-name">{abbr}</span>
        </div>
      );
    }
    return (
      <div className="team">
        <img
          src={team.logoUrl}
          alt={`${team.name} logo`}
          className="team-logo"
          onError={(e) => {
            e.currentTarget.classList.add('is-hidden');
            const fallback = e.currentTarget.nextElementSibling;
            if (fallback) fallback.classList.remove('is-hidden');
          }}
        />
        <div className="team-logo-placeholder is-hidden">{team.abbr}</div>
        <div className="team-info">
          <span className="team-name">{team.name}</span>
          <span className="team-abbr">{team.abbr}</span>
          {isHome && <span className="home-indicator">(Home)</span>}
        </div>
      </div>
    );
  };

  if (error) {
    return (
      <div className="team-grid-error">
        <h3>Error Loading Data</h3>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  if (schedule.length === 0) {
    return (
      <div className="team-grid-loading">
        <div className="loading-spinner"></div>
        <p>Loading next week's matchups...</p>
      </div>
    );
  }

  return (
    <div className="team-grid-section">
      <div className="team-grid-cards a-shine">
        {schedule.map((game, index) => {
          const key = makeKey(game);
          const prediction = predictions[key];
          const isLoading = !!loading[key];

          return (
            <div key={key} style={{'--i': index}}>
              <div
                className={`matchup-card inner-card sb3__content ${prediction ? 'has-prediction' : ''} ${isLoading ? 'loading' : ''}`}
                onClick={() => handlePredict(game)}
                onKeyDown={(e) => {
                  if (isActionKey(e.key)) {
                    e.preventDefault();
                    handlePredict(game);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-label={`Predict ${game.away_abbr} at ${game.home_abbr}`}
              >
                <div className="matchup-teams inner-card">
                  <div className="away-team">{renderTeam(game.away_abbr || game.away_team, false)}</div>
                  <div className="vs-indicator inner-card"><strong>VS</strong></div>
                  <div className="home-team inner-card">{renderTeam(game.home_abbr || game.home_team, true)}</div>
                </div>

                <div className="matchup-time inner-card">{formatKickoffTime(game)}</div>

                {isLoading && (
                  <div className="prediction-loading">
                    <div className="loading-spinner small"></div>
                    <span>Predicting...</span>
                  </div>
                )}

                {prediction && (
                  <div className="prediction-result inner-card">
                    <div className="predicted-scores inner-card a-shine">
                      <span className="away-team-abbr">{game.away_abbr}</span>
                      <span className="score away-score inner-card">{prediction.away_score.toFixed(1)}</span>
                      <span className="score-separator inner-card">{'VS'}</span>
                      <span className="home-team-abbr inner-card">{game.home_abbr || game.home_team}</span>
                      <span className="score home-score inner-card">{prediction.home_score.toFixed(1)}</span>
                      <br />
                    </div>
                    <div className="inner-card point-diff a-shine">
                      <div className="point-diff">
                        Spread: {prediction.point_diff > 0 ? '+' : ''}{prediction.point_diff.toFixed(1)}<br />
                      </div>
                    </div>
                    <div className="point-diff">
                      Home Win Probability: {prediction.home_win_probability.toFixed(2)}
                    </div>

                    {/* Render PredictionResult with a normalized entry shape */}
                    <PredictionResult
                      entry={{
                        game,
                        metrics: {
                          home_score: prediction.home_score,
                          away_score: prediction.away_score,
                          point_diff: prediction.point_diff,
                        },
                        probs: {
                          home: prediction.home_win_probability,
                          away: prediction.away_win_probability,
                        },
                      }}
                    />

                    <div className="save-history">
                      <button
                        type="button"
                        onClick={(ev) => {
                          ev.stopPropagation();
                          const entry = {
                            game,
                            metrics: {
                              home_score: prediction.home_score,
                              away_score: prediction.away_score,
                              point_diff: prediction.point_diff,
                            },
                            probs: {
                              home: prediction.home_win_probability,
                              away: prediction.away_win_probability,
                            },
                          };
                          // Persist with dedupe and cap
                          const makeKey = (e) => `${e.game.season}-${e.game.week}-${e.game.home_abbr || e.game.home_team}-${e.game.away_abbr || e.game.away_team}`;
                          setHistory((h) => {
                            const key = makeKey(entry);
                            const filtered = h.filter((x) => makeKey(x) !== key);
                            const next = [...filtered, entry].slice(-100);
                            try {
                              localStorage.setItem("prediction_history", JSON.stringify(next));
                            } catch (err) {
                              console.debug("Failed to save prediction history", err);
                            }
                            return next;
                          });
                        }}
                      >
                        Save to History
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Saved prediction history */}
      {history.length > 0 && (
        <div className="prediction-history">
          <h3>Saved Predictions</h3>
          <div className="history-list">
            {history.map((entry, i) => (
              <div key={`history-${i}`} className="history-entry">
                <PredictionResult entry={entry} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
