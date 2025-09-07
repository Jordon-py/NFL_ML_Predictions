import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm';
import PredictionResult from './components/PredictionResult';
import HistoryChart from './components/HistoryChart';
import axios from 'axios';

/**
 * Root component for the NFL prediction dashboard.
 *
 * This component maintains the current prediction result and a history of
 * predictions. It passes handlers to the PredictionForm for submitting
 * game stats to the backend and renders the PredictionResult and
 * HistoryChart accordingly.
 */
function App() {
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  /**
   * Submit handler invoked by the form with game stats.
   * Posts data to the backend and stores the prediction result.
   */
  const handlePredict = async (gameStats) => {
    try {
      const response = await axios.post('/predict', gameStats);
      const data = response.data;
      setResult(data);
      setHistory((prev) => [...prev, { date: new Date(), ...data }]);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Failed to get prediction. Ensure the backend is running.');
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