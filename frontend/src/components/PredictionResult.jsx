// ==========================================
// File: frontend/src/components/PredictionResult.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ../api/client, ./PredictionResult.css
// Notes: Presentation-focused component.
// ==========================================

/**
 * FILE: frontend/src/components/PredictionResult.jsx
 * PURPOSE: Renders the prediction outcome for a single game.
 * INPUT PROPS:
 *   - entry: { home_score, away_score, home_win_probability, away_win_probability, ... }
 * OUTPUT / SIDE EFFECTS: Renders score blocks and win probability bars.
 * KEY FUNCTIONS:
 *   - TeamBlock(team, score, range): Sub-component for layout.
 * DEPENDENCIES: React, PredictionResult.css
 */

import React, { useState } from 'react';
import { explainPrediction } from '../api/client';
import './PredictionResult.css';

const TeamBlock = ({ team, score, range }) => {
  const scoreNum = Number(score);
  const displayScore = Number.isFinite(scoreNum) ? Math.round(scoreNum) : "--";
  return (
  <div className={`team-block ${team.toLowerCase()}`}>
    <span className="team-name">{team}</span>
    <span className="score-main">{displayScore}</span>
    {range && (
      <div className="range-box">
        <span className="range-label">Expected Range</span>
        <span className="range-val">{Math.max(0, range[0])}-{range[1]}</span>
      </div>
    )}
  </div>
  );
};

export default function PredictionResult({ entry }) {
  const [explaining, setExplaining] = useState(false);
  const [explanation, setExplanation] = useState(null);
  const [error, setError] = useState(null);

  if (!entry) {
    return (
      <div className="prediction-result" aria-live="polite">
        <h3>Prediction</h3>
        <p>No prediction selected yet.</p>
      </div>
    );
  }

  const sim = entry.simulation_metrics;
  const homeProb = Number(entry.home_win_probability);
  const awayProb = Number(entry.away_win_probability);
  const homePct = Number.isFinite(homeProb) ? Math.round(homeProb * 100) : 0;
  const awayPct = Number.isFinite(awayProb) ? Math.round(awayProb * 100) : 0;
  const homeLabel = entry.home_name || entry.home_team || "HOME";
  const awayLabel = entry.away_name || entry.away_team || "AWAY";
  const seasonLabel = entry.season;
  const weekLabel = entry.week;

  const handleExplain = async () => {
    if (explaining || explanation) return;
    setExplaining(true);
    setError(null);
    try {
      const result = await explainPrediction({
        prediction: entry,
        home_team: entry.home_team,
        away_team: entry.away_team,
        season: seasonLabel,
        week: weekLabel,
      });
      if (!result?.explanation) {
        setExplanation(null);
        setError(result?.error || "Explanation is unavailable.");
        return;
      }
      setExplanation({
        explanation: result.explanation,
        bullets: Array.isArray(result?.bullets) ? result.bullets : [],
        caveats: Array.isArray(result?.caveats) ? result.caveats : [],
      });
    } catch (err) {
      setError(err?.message || "Failed to generate explanation.");
    } finally {
      setExplaining(false);
    }
  };

  const getRange = (val, sd) => (val != null && sd != null) ? 
    [Math.round(val - 1.28 * sd), Math.round(val + 1.28 * sd)] : null;

  return (
    <div className={`prediction-result-container ${sim ? 'expert-mode' : ''}`} aria-live="polite">
      <header className="prediction-header">
        <div className="header-text">
          <h3>{sim ? 'Ensemble Mixture Analysis' : 'Prediction'}</h3>
          <span className="meta-text">
            Week {weekLabel ?? "?"} - {seasonLabel ?? "?"}
          </span>
        </div>
        {sim && <div className="expert-badge">ML + MC Ensemble</div>}
      </header>

      <div className="expert-content">
        <div className="team-row">
          <TeamBlock
            team={homeLabel}
            score={entry.home_score}
            range={getRange(sim?.sim_home_score, sim?.sim_home_sd)}
          />
          <div className="vs-divider"><div className="vs-circle">VS</div></div>
          <TeamBlock
            team={awayLabel}
            score={entry.away_score}
            range={getRange(sim?.sim_away_score, sim?.sim_away_sd)}
          />
        </div>

        <div className="win-probability-expert">
          <div className="prob-header">
            <span>Win Probability</span>
            {sim && <span className="sim-meta">{sim.sim_n} trials</span>}
          </div>
          <div className="prob-bar-base">
            <div className="prob-fill home" style={{ width: `${homePct}%` }}>{homePct}%</div>
            <div className="prob-fill away" style={{ width: `${awayPct}%` }}>{awayPct}%</div>
          </div>
        </div>

        <div className="explanation-section">
          {!explanation && !explaining && (
            <button className="explain-btn" onClick={handleExplain}>Explain This Prediction</button>
          )}
          {explaining && <div className="explaining-loader">Analyzing...</div>}
          {error && <div className="explanation-error">{error}</div>}
          {explanation && (
            <div className="explanation-box animate-in">
              <h4>Model Strategy Insights</h4>
              <p>{explanation.explanation}</p>
              {explanation.bullets?.length > 0 && (
                <ul className="explanation-bullets">
                  {explanation.bullets.map((line, idx) => <li key={idx}>{line}</li>)}
                </ul>
              )}
              {explanation.caveats?.length > 0 && (
                <div className="explanation-caveats">
                  <h5>Caveats</h5>
                  <ul>
                    {explanation.caveats.map((line, idx) => <li key={idx}>{line}</li>)}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
