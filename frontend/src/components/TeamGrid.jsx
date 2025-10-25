
import {getNextWeekSchedule, predictGame} from '../api/client.js';
import React, {useState, useEffect} from 'react';

// NOTE: hooks (like useCallback) must only be called inside React function
// components or custom hooks. Creating a memoized callback at module scope
// causes runtime errors. Use a plain utility function here instead.
const makeKey = (g) => `${g.season}-${g.week}-${g.home_abbr}-${g.away_abbr}`;

const isActionKey = (key) => key === 'Enter' || key === ' ';

const kickoffFormatter = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/Los_Angeles', weekday: 'short', month: 'short',
  day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true,
});

const formatKickoffTime = (row) => {
  const iso = row.kickoff_ts_utc || row.kickoff_iso || row.kickoff || null;
  try { return kickoffFormatter.format(new Date(iso)); } catch { return String(iso); }
};

const parseTeamCsv = (csvText) =>
  csvText.trim().split('\n').slice(1).reduce((acc, line) => {
    const [teamName, abbr, logoUrl] = line.split(',').map((v) => v.trim());
    if (abbr) acc[abbr] = {name: teamName, abbr, logoUrl};
    return acc;
  }, {});



export default function TeamGrid() {
  const [teams, setTeams] = useState({});
  const [schedule, setSchedule] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  useEffect(() => { (async () => {
    try {
      const res = await fetch('/data/myteamdescriptions.csv');
      if (!res.ok) throw new Error('Failed to load team data');
      setTeams(parseTeamCsv(await res.text()));
    } catch (err) {
      console.error('[TeamGrid] loadTeams:', err);
      setError('Failed to load team data');
    }
  })(); }, []);

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

  useEffect(() => { (async () => {
    try { setSchedule(await getNextWeekSchedule()); }
    catch (err) {
      console.error('[TeamGrid] loadSchedule:', err);
      setError('Failed to load schedule'); setSchedule([]);
    }
  })(); }, []);

  const handlePredict = async (game) => {
    const key = makeKey(game);
    if (loading[key]) return;
    setLoading((s) => ({ ...s, [key]: true }));
    setError(null);

    try {
      const payload = {
        home_abbr: game.home_abbr || game.home_team,
        away_abbr: game.away_abbr || game.away_team,
        season: Number(game.season), week: Number(game.week),
      };
      const res = await predictGame(payload);
      const result = {
        home_score: Number(res.home_score),
        away_score: Number(res.away_score),
        point_diff: Number(res.point_diff),
        home_win_probability: Number(res.home_win_probability),
        away_win_probability: Number(res.away_win_probability),
      };
      setPredictions((p) => ({ ...p, [key]: result }));

      // persist to history
      const entry = {
        ts: new Date().toISOString(),
        game: { season: payload.season, week: payload.week, home_abbr: payload.home_abbr, away_abbr: payload.away_abbr },
        probs: { home: result.home_win_probability, away: result.away_win_probability, ensemble: result.home_win_probability },
      };
      setHistory((h) => {
        const next = [entry, ...h].slice(0, 100);
        localStorage.setItem("prediction_history", JSON.stringify(next));
        return next;
      });
    } catch (e) {
      console.error('[TeamGrid] predictGame:', e);
      setError('Failed to get prediction');
    } finally {
      setLoading((s) => ({ ...s, [key]: false }));
    }
  };

  if (error) return (<div className="team-grid-error"><h3>Error Loading Data</h3><p>{error}</p><button onClick={()=>window.location.reload()}>Retry</button></div>);
  if (schedule.length === 0) return (<div className="team-grid-loading"><div className="loading-spinner"></div><p>Loading next week’s matchups…</p></div>);

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
                onKeyDown={(e) => { if (isActionKey(e.key)) { e.preventDefault(); handlePredict(game); } }}
                tabIndex={0} role="button" aria-pressed={isLoading}
                aria-label={`Predict ${game.away_abbr} at ${game.home_abbr}`}
              >
                <header className="matchup-head">
                  <strong>{game.away_abbr}</strong> @ <strong>{game.home_abbr}</strong>
                  <span className="kickoff">{formatKickoffTime(game)}</span>
                </header>
                {prediction ? (
                  <div className="prediction">
                    <div>Home win: {(prediction.home_win_probability*100).toFixed(0)}%</div>
                    <div>Point diff: {prediction.point_diff.toFixed(1)}</div>
                    <div>Score: {prediction.home_score}–{prediction.away_score}</div>
                  </div>
                ) : <div className="cta">Click to predict</div>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
