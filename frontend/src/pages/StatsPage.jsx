/**
 * StatsPage.jsx
 * ----------------
 * Purpose:
 *   Dedicated route that pairs the next-week schedule with stored prediction history.
 *   Keeps dependencies internal (no react-spinners) so Vercel builds stay lean.
 *
 * Key ideas:
 *   - Leverages PredictionContext history while lazily fetching the schedule.
 *   - Provides a minimal, accessible loading indicator implemented via local CSS.
 *   - Reuses the global <NavBar/> to keep navigation consistent.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { usePredictions } from '../PredictionContext';
import { getNextWeekSchedule } from '../api/client';
import HistoryChart from '../components/HistoryChart';
import NavBar from '../components/NavBar/NavBar.jsx';
import './StatsPage.css';

function LoadingSpinner() {
  return (
    <div className="loading-container" role="status" aria-live="polite">
      <span className="loading-spinner" aria-hidden="true" />
      <p>Loading schedule...</p>
    </div>
  );
}

const StatsPage = () => {
  const { history } = usePredictions();
  const [schedule, setSchedule] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const getSchedule = async () => {
      try {
        setLoading(true);
        const response = await getNextWeekSchedule();
        if (response && response.games) {
          setSchedule(response.games);
        } else {
          setSchedule([]);
        }
        setError(null);
      } catch (err) {
        console.error('Error fetching schedule:', err);
        setError('Failed to load schedule. The API might be down.');
        setSchedule([]);
      } finally {
        setLoading(false);
      }
    };

    getSchedule();
  }, []);

  const predictionMap = useMemo(() => {
    if (!history) return new Map();
    return new Map(history.map(p => [p.game_id, p]));
  }, [history]);

  const renderSchedule = () => {
    if (loading) {
      return <LoadingSpinner />;
    }

    if (error) {
      return <div className="error-message">{error}</div>;
    }

    if (schedule.length === 0) {
      return <p>No schedule available for the next week.</p>;
    }

    return (
      <ul className="schedule-list">
        {schedule.map(game => {
          const gamePrediction = predictionMap.get(game.game_id);
          const kickoffTime = new Date(game.kickoff).toLocaleString();

          return (
            <li key={game.game_id} className="schedule-item">
              <div className="game-info">
                <span>{game.away_team} @ {game.home_team}</span>
                <span className="kickoff-time">{kickoffTime}</span>
              </div>
              {gamePrediction && (
                <div className="prediction-details">
                  <p>Home: {Math.round(gamePrediction.home_win_probability * 100)}%</p>
                  <p>Away: {Math.round(gamePrediction.away_win_probability * 100)}%</p>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    );
  };

  return (
    <>
      <NavBar />
      <div className="stats-page">
        
        <h1>Next Week Schedule & Predictions</h1>
        {renderSchedule()}

        <h2>Historical Predictions</h2>
        <HistoryChart />
      </div>
    </>
  );
};

export default StatsPage;
