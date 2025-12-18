// ==========================================
// File: frontend/src/components/TeamGrid.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ./TeamGrid.css
// Notes: Presentation-focused component.
// ==========================================

/**
 * FILE: frontend/src/components/TeamGrid.jsx
 * PURPOSE: Render upcoming NFL matchups as interactive cards.
 * INPUT PROPS:
 *   - schedule: List of ScheduleGame objects.
 *   - predictions: Map keyed by game_id with flat prediction payloads.
 *   - onPredict: Callback triggered on card click.
 * OUTPUT / SIDE EFFECTS: Renders a grid of matchup cards.
 * DEPENDENCIES: React, TeamGrid.css
 */

import React from 'react';
import './TeamGrid.css';

const TeamCard = ({ game, prediction, onPredict }) => {
  const isWinner = (side) => {
    const homeProb = prediction ? prediction.home_win_probability : prediction.away_win_probability;
    const awayProb = prediction ? prediction.away_win_probability : prediction.home_win_probability;
    return side === 'home' ? homeProb >= awayProb : awayProb > homeProb;
  };

  return (
    <div 
      className={`matchup-card ${prediction ? 'has-prediction' : ''}`}
      onClick={() => onPredict(game)}
      role="button"
      tabIndex={0}
    >
      <header className="matchup-head">
        <div className="teams-row">
          <div className={`team-info ${isWinner('away') ? 'winner' : ''}`}>
            <strong>{game.away_abbr}</strong>
          </div>
          <span className="at-symbol">@</span>
          <div className={`team-info ${isWinner('home') ? 'winner' : ''}`}>
            <strong>{game.home_abbr}</strong>
          </div>
        </div>
        <span className="kickoff">{game.kickoff || 'TBD'}</span>
      </header>

      {prediction ? (
        <div className="prediction">
          <div className="scores">
            {(prediction.home_score ?? 0).toFixed(0)} - {(prediction.away_score ?? 0).toFixed(0)}
          </div>
          <div className="prob">
            Win Prob: {Math.round((prediction.home_win_probability ?? 0) * 100)}%
          </div>
        </div>
      ) : (
        <div className="cta">Click to predict</div>
      )}
    </div>
  );
};

export default function TeamGrid({ schedule, predictions, onPredict }) {
  if (!schedule?.length) return <div className="team-grid-empty">No games found.</div>;

  return (
    <div className="team-grid-section">
      <div className="team-grid-cards">
        {schedule.map(game => {
          const key = game.game_id || `${game.season}-${game.week}-${game.home_abbr}-${game.away_abbr}`;
          return (
          <TeamCard 
            key={key}
            game={game}
            prediction={predictions[key]}
            onPredict={onPredict}
          />
          );
        })}
      </div>
    </div>
  );
}
