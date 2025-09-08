import React from 'react';

/**
 * Component for displaying prediction results.
 *
 * Receives a `result` prop which contains the neural network, gradient
 * boosting and ensemble probabilities. Displays them in a clean card.
 */
function PredictionResult({ result }) {
  const { neural_network_proba, gradient_boosting_proba, ensemble_proba } = result;
  const toPercent = (p) => (p * 100).toFixed(1) + '%';
  return (
    <div className="result-card">
      <h2>Prediction Result</h2>
      <p><strong>Neural Network:</strong> {toPercent(neural_network_proba)}</p>
      <p><strong>Gradient Boosting:</strong> {toPercent(gradient_boosting_proba)}</p>
      <p><strong>Ensemble (Final):</strong> {toPercent(ensemble_proba)}</p>
    </div>
  );
}

export default PredictionResult;