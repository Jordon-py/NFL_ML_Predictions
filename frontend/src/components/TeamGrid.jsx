/**
 * TeamGrid Component
 * ------------------
 * Purpose: Display upcoming NFL matchups and trigger model predictions.
 * Flow: Fetch team metadata -> fetch schedule -> render cards -> request predictions per selection.
 * Dependencies: React hooks, local CSV data, API client helpers (getNextWeekSchedule, predictGame).
 */
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

const formatKickoffTime = (isoString) => {
  try {
    return kickoffFormatter.format(new Date(isoString));
  } catch {
    return isoString;
  }
};

const parseTeamCsv = (csvText) =>
  csvText
    .trim()
    .split('\n')
    .slice(1)
    .reduce((acc, line) => {
      const [teamName, abbr, logoUrl] = line.split(',').map((value) => value.trim());
      if (abbr) acc[abbr] = {name: teamName, abbr, logoUrl};
      return acc;
    }, {});

const normalizeSchedulePayload = (data) => {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.games)) return data.games;
  throw new Error('Schedule payload is malformed.');
};

const isActionKey = (key) => key === 'Enter' || key === ' ';

/** Build a stable key per game */
const makeKey = (g) => `${g.season}-${g.week}-${g.home_abbr}-${g.away_abbr}`;

export default function TeamGrid() {
  const [teams, setTeams] = useState({});
  const [schedule, setSchedule] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  // Load team metadata from CSV
  useEffect(() => {
    const loadTeams = async () => {
      try {
        const response = await fetch('/data/myteamdescriptions.csv');
        if (!response.ok) throw new Error('Failed to load team data');
        setTeams(parseTeamCsv(await response.text()));
      } catch (err) {
        console.error('[TeamGrid] Failed to load teams:', err);
        setError('Failed to load team data');
      }
    };
    loadTeams();
  }, []);

  // Load next week's schedule from API
  useEffect(() => {
    const loadSchedule = async () => {
      try {
        setSchedule(normalizeSchedulePayload(await getNextWeekSchedule()));
      } catch (err) {
        console.error('[TeamGrid] Failed to load schedule:', err);
        setError('Failed to load schedule');
        setSchedule([]);
      }
    };
    loadSchedule();
  }, []);

  // Predict a matchup
  const handlePredict = async (game) => {

    const key = makeKey(game);
    if (loading[key]) return;
    setLoading((prev) => ({...prev, [key]: true}));
    setError(null);
    const payload = {
      home_team: game.home_abbr,
      away_team: game.away_abbr,
      season: game.season,
      week: game.week,
    };
    try {
      const res = await predictGame(payload);
      const {home_score, away_score} = res;
      const result = {
        home_score: Number(home_score),
        away_score: Number(away_score),
        point_diff: Number(home_score) - Number(away_score),
      };
      return setPredictions((prev) => ({...prev, [key]: result}));
    } catch (e) {
      console.error('[TeamGrid] predictGame failed:', e);
      setError('Failed to get prediction');
    } finally {
      setLoading((prev) => ({...prev, [key]: false}));
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
                  <div className="away-team">{renderTeam(game.away_abbr, false)}</div>
                  <div className="vs-indicator inner-card">@</div>
                  <div className="home-team inner-card">{renderTeam(game.home_abbr, true)}</div>
                </div>

                <div className="matchup-time inner-card">{formatKickoffTime(game.kickoff_iso)}</div>

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
