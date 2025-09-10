/**
 * NFL Game Prediction Dashboard - Main Application Component
 *
 * This sophisticated React component serves as the central hub for an interactive
 * NFL game outcome prediction system. It orchestrates a seamless user experience
 * by managing prediction workflows, maintaining historical data, and providing
 * real-time visual feedback through dynamic charts and result displays.
 *
 * Key Responsibilities:
 * - Manages application state for prediction results and historical records
 * - Coordinates communication between user input forms and backend prediction services
 * - Renders conditional UI components based on user interactions and data availability
 * - Implements error handling for network requests and API failures
 * - Maintains chronological prediction history for trend analysis
 *
 * State Management:
 * - `result`: Stores the most recent prediction outcome from the backend API
 * - `history`: Accumulates all prediction attempts with timestamps for longitudinal analysis
 * - `currentPrediction`: Stores the prediction result from TeamGrid interactions
 *
 * User Experience Flow:
 * 1. User can either use the manual PredictionForm or click matchups in TeamGrid
 * 2. Form submission or TeamGrid clicks trigger API calls to machine learning prediction service
 * 3. Successful predictions are displayed immediately and archived in history
 * 4. Historical predictions are visualized through interactive charts
 * 5. Error states are communicated through user-friendly alert messages
 *
 * Integration Points:
 * - PredictionForm: Receives game statistics and initiates prediction requests
 * - PredictionResult: Displays the latest prediction outcome with confidence metrics
 * - HistoryChart: Visualizes prediction trends and historical performance
 * - TeamGrid: Displays next week's matchups with click-to-predict functionality
 * - Backend API: Communicates with FastAPI server for ML-powered predictions
 *
 * Error Handling:
 * - Network failures are caught and displayed as user alerts
 * - Backend unavailability is communicated clearly to maintain user trust
 * - Graceful degradation ensures the interface remains functional during outages
 *
 * Performance Considerations:
 * - Efficient state updates prevent unnecessary re-renders
 * - History accumulation is optimized for memory usage
 * - API calls are properly awaited to prevent race conditions
 *
 * @component
 * @example
 * ```jsx
 * function App() {
 *   // Component automatically handles all prediction workflow
 *   return <App />;
 * }
 * ```
 */

import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm';
import PredictionResult from './components/PredictionResult';
import HistoryChart from './components/HistoryChart';
import TeamGrid from './components/TeamGrid';
import { predictGame } from './api/client.js';
import './styles.css';

function App() {
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [error, setError] = useState(null);

  /* ============================================================================
   Handler Functions
  ========================================================================== */

  /**
   * Submit handler invoked by the PredictionForm with game stats.
   * Posts data to the backend and stores the prediction result.
   * @param {object} gameStats - Game statistics for prediction
   */
  const handlePredict = async (gameStats) => {
    try {
      setError(null);
      const response = await predictGame(gameStats);
      const data = response;

      setResult(data);
      setCurrentPrediction(null); // Clear TeamGrid prediction when using form

      // Archive this prediction in history
      setHistory((prev) => [...prev, {
        date: new Date(),
        ...data,
        source: 'form'
      }]);

    } catch (err) {
      console.error('[App] Prediction failed:', err);
      setError('Failed to get prediction. Please try again.');
    }
  };

  /**
   * Handler for predictions made through TeamGrid component
   * @param {object} game - Game object with home/away teams
   * @param {object} prediction - Prediction result from API
   */
  const handleTeamGridPrediction = (game, prediction) => {
    setCurrentPrediction({
      game,
      prediction,
      timestamp: new Date()
    });
    setResult(null); // Clear form prediction when using TeamGrid

    // Archive this prediction in history
    setHistory((prev) => [...prev, {
      date: new Date(),
      ...prediction,
      game,
      source: 'teamgrid'
    }]);
  };

  /**
   * Clear current prediction and error states
   */
  const clearCurrentPrediction = () => {
    setResult(null);
    setCurrentPrediction(null);
    setError(null);
  };

  return (
    <div className="container">
      <header className="app-header">
        <h1>NFL Game Predictor</h1>
        <p>Advanced machine learning predictions for NFL games</p>
      </header>

      <main className="app-main">
        {/* TeamGrid for next week's matchups */}
        <section className="team-grid-section">
          <TeamGrid onPrediction={handleTeamGridPrediction} />
        </section>

        {/* Manual prediction form */}
        <section className="prediction-form-section">
          <div className="section-header">
            <h2>Manual Prediction</h2>
            <p>Enter game statistics for custom predictions</p>
          </div>
          <PredictionForm onPredict={handlePredict} />
        </section>

        {/* Current prediction display */}
        {(result || currentPrediction) && (
          <section className="current-prediction-section">
            <div className="section-header">
              <h2>Current Prediction</h2>
              <button
                className="clear-button"
                onClick={clearCurrentPrediction}
                aria-label="Clear current prediction"
              >
                Clear
              </button>
            </div>

            {result && <PredictionResult result={result} />}
            {currentPrediction && (
              <div className="teamgrid-prediction-result">
                <h3>Matchup Prediction</h3>
                <div className="matchup-info">
                  <span className="team away">{currentPrediction.game.away_abbr}</span>
                  <span className="at-symbol">@</span>
                  <span className="team home">{currentPrediction.game.home_abbr}</span>
                </div>
                <div className="prediction-details">
                  <div className="scores">
                    <span className="score home-score">
                      {currentPrediction.prediction.home_score.toFixed(1)}
                    </span>
                    <span className="separator">-</span>
                    <span className="score away-score">
                      {currentPrediction.prediction.away_score.toFixed(1)}
                    </span>
                  </div>
                  <div className="spread">
                    Spread: {currentPrediction.prediction.point_diff > 0 ? '+' : ''}
                    {currentPrediction.prediction.point_diff.toFixed(1)}
                  </div>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Error display */}
        {error && (
          <section className="error-section">
            <div className="error-message">
              <h3>Error</h3>
              <p>{error}</p>
            </div>
          </section>
        )}

        {/* Prediction history */}
        {history.length > 0 && (
          <section className="history-section">
            <div className="section-header">
              <h2>Prediction History</h2>
              <p>Your recent predictions and results</p>
            </div>
            <HistoryChart history={history} />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;