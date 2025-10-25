/**
 * DashBoard.jsx
 * -------------
 * Component Purpose:
 *   Compose the primary dashboard layout: it renders the grid of matchups,
 *   the latest prediction, and historical trend in one place.
 *
 * Core Logic Overview:
 *   - Reads shared state from `usePredictions()` (context provider).
 *   - Delegates user interactions to child components; this component stays
 *     focused on layout and accessibility semantics.
 *
 * Modification Guide:
 *   - To inject new sections (e.g. filters, leaderboards), add `<section>`
 *     blocks so screen readers understand the layout.
 *   - Keep data transformations in the context/provider layer—children should
 *     receive ready-to-render props.
 */
import {usePredictions} from '../PredictionContext.jsx';

import TeamGrid from './TeamGrid.jsx';
import PredictionResult from './PredictionResult.jsx';
import HistoryChart from './HistoryChart.jsx';
import NavBar from './NavBar/NavBar.jsx';
import './TeamGrid.css';

export default function DashBoard() {
  // `state` exposes { current, history } for the entire app.
  const {state} = usePredictions();

  return (
    <>
      <NavBar state={state} />
      <main className="dashboard">
        <header>
          <div className="team-grid-header">
            <h2 className="nfl-matchups">Next Week's NFL Matchups</h2>
            <p>Click any matchup to see predicted scores</p>
          </div>
        </header>
        <section>
          <TeamGrid state={state} />
        </section>

        <section aria-live="polite">
          {/* Pass the current prediction entry; component handles the empty state. */}
          <PredictionResult entry={state.current} />
        </section>

        <section>
          {/* Historical predictions show trend data to the user. */}
          <HistoryChart history={state.history} />
        </section>
      </main>
    </>
  );
}
