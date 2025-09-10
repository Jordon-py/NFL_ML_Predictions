import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { getNextWeekSchedule, predictGame } from '../api/client.js';
import './TeamGrid.css';

/**
 * TeamGrid Component - Displays NFL matchups for next week with prediction capabilities
 *
 * Features:
 * - Loads team metadata from CSV
 * - Fetches next week's schedule from API
 * - Displays matchup cards with team logos and kickoff times
 * - Handles prediction requests on card clicks
 * - Shows loading states and prediction results
 * - Responsive grid layout with accessibility features
 *
 * @component
 * @param {object} props - Component props
 * @param {Function} props.onPrediction - Callback when prediction is made
 * @returns {JSX.Element} TeamGrid component
 */
function TeamGrid({ onPrediction }) {
  const [teams, setTeams] = useState({});
  const [schedule, setSchedule] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  /**
   * Load team metadata from CSV file
   * Parses team descriptions and creates lookup table by abbreviation
   */
  useEffect(() => {
    const loadTeams = async () => {
      try {
        const response = await fetch('/data/myteamdescriptions.csv');
        if (!response.ok) {
          throw new Error('Failed to load team data');
        }

        const csvText = await response.text();
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());

        const teamData = {};
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map(v => v.trim());
          if (values.length >= 3) {
            const [teamName, abbr, logoUrl] = values;
            teamData[abbr] = {
              name: teamName,
              abbr,
              logoUrl,
            };
          }
        }

        setTeams(teamData);
      } catch (err) {
        console.error('[TeamGrid] Failed to load teams:', err);
        setError('Failed to load team data');
      }
    };

    loadTeams();
  }, []);

  /**
   * Load next week's schedule from API
   */
  useEffect(() => {
    const loadSchedule = async () => {
      try {
        const scheduleData = await getNextWeekSchedule();
        setSchedule(scheduleData);
      } catch (err) {
        console.error('[TeamGrid] Failed to load schedule:', err);
        setError('Failed to load schedule');
      }
    };

    if (Object.keys(teams).length > 0) {
      loadSchedule();
    }
  }, [teams]);

  /**
   * Handle prediction request for a matchup
   * @param {object} game - Game object with season, week, home_abbr, away_abbr
   */
  const handlePredict = async (game) => {
    const gameKey = `${game.home_abbr}-${game.away_abbr}`;

    // Don't make duplicate requests
    if (loading[gameKey] || predictions[gameKey]) {
      return;
    }

    setLoading(prev => ({ ...prev, [gameKey]: true }));
    setError(null);

    try {
      const result = await predictGame({
        home_team: game.home_abbr,
        away_team: game.away_abbr,
        season: game.season,
        week: game.week,
      });

      setPredictions(prev => ({ ...prev, [gameKey]: result }));

      // Notify parent component
      if (onPrediction) {
        onPrediction(game, result);
      }
    } catch (err) {
      console.error('[TeamGrid] Prediction failed:', err);
      setError(`Failed to predict ${game.home_abbr} vs ${game.away_abbr}`);
    } finally {
      setLoading(prev => ({ ...prev, [gameKey]: false }));
    }
  };

  /**
   * Format kickoff time to local timezone (America/Los_Angeles)
   * @param {string} isoString - ISO timestamp string
   * @returns {string} Formatted local time
   */
  const formatKickoffTime = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString('en-US', {
        timeZone: 'America/Los_Angeles',
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      });
    } catch (err) {
      console.warn('[TeamGrid] Failed to format time:', err);
      return isoString;
    }
  };

  /**
   * Render team logo and name
   * @param {string} abbr - Team abbreviation
   * @param {boolean} isHome - Whether this is the home team
   * @returns {JSX.Element} Team display element
   */
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
            e.target.style.display = 'none';
            e.target.nextSibling.style.display = 'block';
          }}
        />
        <div className="team-logo-placeholder" style={{ display: 'none' }}>
          {team.abbr}
        </div>
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
        <button onClick={() => window.location.reload()}>
          Retry
        </button>
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
    <div className="team-grid">
      <div className="team-grid-header">
        <h2>Next Week's NFL Matchups</h2>
        <p>Click any matchup to see predicted scores</p>
      </div>

      <div className="matchups-grid">
        {schedule.map((game, index) => {
          const gameKey = `${game.home_abbr}-${game.away_abbr}`;
          const prediction = predictions[gameKey];
          const isLoading = loading[gameKey];

          return (
            <div
              key={`${game.season}-${game.week}-${index}`}
              className={`matchup-card ${prediction ? 'has-prediction' : ''} ${isLoading ? 'loading' : ''}`}
              onClick={() => handlePredict(game)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handlePredict(game);
                }
              }}
              tabIndex={0}
              role="button"
              aria-label={`Predict ${game.away_abbr} at ${game.home_abbr}`}
            >
              <div className="matchup-teams">
                <div className="away-team">
                  {renderTeam(game.away_abbr, false)}
                </div>
                <div className="vs-indicator">@</div>
                <div className="home-team">
                  {renderTeam(game.home_abbr, true)}
                </div>
              </div>

              <div className="matchup-time">
                {formatKickoffTime(game.kickoff_iso)}
              </div>

              {isLoading && (
                <div className="prediction-loading">
                  <div className="loading-spinner small"></div>
                  <span>Predicting...</span>
                </div>
              )}

              {prediction && (
                <div className="prediction-result">
                  <div className="predicted-scores">
                    <span className="score home-score">{prediction.home_score.toFixed(1)}</span>
                    <span className="score-separator">-</span>
                    <span className="score away-score">{prediction.away_score.toFixed(1)}</span>
                  </div>
                  <div className="point-diff">
                    Spread: {prediction.point_diff > 0 ? '+' : ''}{prediction.point_diff.toFixed(1)}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

TeamGrid.propTypes = {
  onPrediction: PropTypes.func,
};

export default TeamGrid;
