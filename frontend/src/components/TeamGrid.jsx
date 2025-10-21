// @ts-nocheck
// TeamGrid.jsx
import React, {useState, useEffect} from 'react';
import {getNextWeekSchedule, predictGame} from '../api/client.js';

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
const formatKickoffTime = (row) => {
  const iso = row.kickoff_ts_utc || row.kickoff_iso || row.kickoff || null;
  if (!iso) return 'TBD';
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

export default function TeamGrid() {
  const [teams, setTeams] = useState({});
  const [schedule, setSchedule] = useState([]);
  const [predictions, setPredictions] = useState({});
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
        home_team: game.home_abbr || game.home_team,
        away_team: game.away_abbr || game.away_team,
        // Also include raw fields some backend builds expect (rest days) so the
        // prediction endpoint can construct features if available.
        home_abbr: game.home_abbr || game.home_team,
        away_abbr: game.away_abbr || game.away_team,
        home_rest: game.home_rest ?? 7,
        away_rest: game.away_rest ?? 7,
        season: Number(game.season),
        week: Number(game.week),
      };
      const {home_score, away_score, home_abbr, away_abbr} = await predictGame(payload);
      const result = {
        home_score: Number(home_score),
        away_score: Number(away_score),
        point_diff: Number(home_score) - Number(away_score),
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
                  <div className="vs-indicator inner-card">@</div>
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
                    <div className="predicted-scores inner-card">
                      <span className="score away-score inner-card">{prediction.away_score.toFixed(1)}</span>
                      <span className="score-separator inner-card">{'<->'}</span>
                      <span className="score home-score inner-card">{prediction.home_score.toFixed(1)}</span>
                    </div>
                    <div className="point-diff">
                      Spread: {prediction.point_diff > 0 ? '+' : ''}{prediction.point_diff.toFixed(1)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
