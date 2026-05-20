// ==========================================
// File: frontend/src/components/HistoryPage.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ./NavBar/NavBar.jsx, ./HistoryChart.jsx, ./D_BUTTON.jsx
// Notes: Presentation-focused component.
// ==========================================

/**
 * HistoryPage.jsx
 * ----------------
 * Purpose:
 *   Standalone route that renders the HistoryChart using props
 *   supplied by the top-level App state.
 *
 * Contract:
 *   - Receives prediction state via props.
 *   - Supplies a safe `history` array to <HistoryChart/>.
 *
 * Notes:
 *   - Chart render cost is roughly O(n) over `history.length`.
 *   - Page re-renders when `history` changes in App state.
 */
import React from 'react';
import HistoryChart from './HistoryChart.jsx';
import NavBar from './NavBar/NavBar.jsx';
import D_BUTTON from './D_BUTTON.jsx';

export default function HistoryPage({
  authSession,
  onSignOut,
  history = [],
  historySummary = null,
  health,
  onClearHistory,
  historyCount = 0,
}) {
  const safeHistory = Array.isArray(history) ? history : [];
  const safeCount = Number.isFinite(Number(historyCount))
    ? Number(historyCount)
    : safeHistory.length;

  return (
    <>
      <NavBar
        authSession={authSession}
        onSignOut={onSignOut}
        state={{
          health,
          title: 'Prediction History',
          heroSubtitle: 'Review saved forecasts, filter resolved results, and compare your calls to final scores.',
          subtitle: `${safeCount} saved prediction${safeCount === 1 ? '' : 's'}`,
          healthLabel:
            health?.status === 'healthy'
              ? 'Service: Live'
              : `Service: ${health?.status ?? 'unknown'}`,
        }}
      />

      <section className="history-controls">
        <D_BUTTON onClear={onClearHistory} count={safeCount} />
      </section>

      <HistoryChart history={safeHistory} summary={historySummary} />
    </>
  );
}
