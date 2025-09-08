// components/TeamGrid.jsx
import React from 'react';

/**
 * TeamGrid
 * - Displays next-week matchups
 * - On click, calls onPick({ home, away }) which should POST to /predict_raw
 */
const GAMES_NEXT_WEEK = [
  { home: '49ers', away: 'Seahawks', homeLogo: '/logos/49ers.png', awayLogo: '/logos/SEA.png' },
  { home: 'Chiefs', away: 'Chargers', homeLogo: '/logos/KC.png', awayLogo: '/logos/LAC.png' },
  // TODO: hydrate from backend schedule
];

export default function TeamGrid({ onPick }) {
  return (
    <div>
      <h2>Next Week</h2>
      <div className="grid teams">
        {GAMES_NEXT_WEEK.map((g, i) => (
          <button key={i} className="team-tile" onClick={() => onPick({ home: g.home, away: g.away })}>
            <img src={g.homeLogo} alt={g.home} />
            <div>@</div>
            <img src={g.awayLogo} alt={g.away} />
            <div>{g.away} at {g.home}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
