import React from "react";

/**
 * Card container component.
 *
 * Purpose:
 *   - Displays matchup information and, if available, prediction results for a single game.
 * Props:
 *   - (Expected) matchup: object containing game details (teams, time, etc.)
 *   - (Optional) prediction: object containing prediction results for the matchup.
 * State:
 *   - hasPrediction: boolean indicating if prediction data is present.
 */

function Card({ matchup, prediction }) {
  // Determine if prediction data is present
  const hasPrediction = !!prediction;

  // Render the card with matchup info and prediction (if available)
  return (
    <div className="card-container">
      <h3>Matchup Info</h3>
      {/* Render matchup details here */}
      {hasPrediction && (
        <div className="prediction-section">
          <h4>Prediction Result</h4>
          {/* Render prediction result details here */}
        </div>
      )}
    </div>
  );
}

export default Card;
  );
}

export default Card;