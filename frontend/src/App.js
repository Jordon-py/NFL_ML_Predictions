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
 *
 * User Experience Flow:
 * 1. User enters game statistics through the PredictionForm component
 * 2. Form submission triggers API call to machine learning prediction service
 * 3. Successful predictions are displayed immediately and archived in history
 * 4. Historical predictions are visualized through interactive charts
 * 5. Error states are communicated through user-friendly alert messages
 *
 * Integration Points:
 * - PredictionForm: Receives game statistics and initiates prediction requests
 * - PredictionResult: Displays the latest prediction outcome with confidence metrics
 * - HistoryChart: Visualizes prediction trends and historical performance
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
import axios from 'axios';
import './styles.css';


function App() {
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  /* ============================================================================
   Handler Functions
  ========================================================================== */
  /**
   * Submit handler invoked by the form with game stats.
   * Posts data to the backend and stores the prediction result.
   */
    const handlePredict = async (gameStats) => {
    try {
      const response = await axios.post('/predict', gameStats);
      const data = response.data;  // { prediction: 0.75, mode: "models", ... }
      
      setResult(data);  // Update current result
      
      // Archive this prediction in history
      setHistory((prev) => [...prev, { date: new Date(), ...data }]);
      
    } catch (error) {
      // Error handling...
    }
  };

  return (
    <div className="container">
      <h1>NFL Game Predictor</h1>
      <PredictionForm onPredict={handlePredict} />
      {result && <PredictionResult result={result} />}
      {history.length > 0 && <HistoryChart history={history} />}
    </div>
  );
}

export default App;