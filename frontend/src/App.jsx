// ==========================================
// File: frontend/src/App.jsx
// Role: Frontend module.
// Input Data: Module inputs.
// Output Data: Exports for UI usage.
// Dependencies: react, react-router-dom, ./components/ErrorBoundary, ./hooks/usePredictionState
// Notes: Shared application code.
// ==========================================

/**
 * App.jsx
 * 
 * Root React application component for the NFL prediction UI.
 * 
 * - Holds shared prediction state in App and passes it via props.
 * - Wraps the UI in a shared ErrorBoundary for global error handling.
 * - Defines top-level routes using React Router (Dashboard, History, Stats, Settings, 404).
 * - All prediction logic and state live in child components (e.g., TeamGrid, HistoryPage).
 * 
 * Architecture notes:
 *   - React Router handles page-level navigation.
 *   - Global layout/styling is managed via TeamGrid.css as the main stylesheet entrypoint.
 * 
 * Change Log:
 *   2025-11-11:
 *     - Replaced placeholder content with a working App component.
 *     - Fixed Dashboard import path and component name mismatch.
 *     - Removed unused imports (Link, useState, HistoryChart) to reduce noise.
 */

import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import { usePredictionState } from './hooks/usePredictionState';

// Lazy load components for better performance
const Dashboard = lazy(() => import('./components/DashBoard/Dashboard'));
const HistoryPage = lazy(() => import('./components/HistoryPage'));
const StatsPage = lazy(() => import('./pages/StatsPage'));

// Inline fallback SettingsPage to avoid missing module errors
function SettingsPage() {
  return (
    <div className="settings-page">
      <h1>Settings</h1>
      <p>Settings page coming soon.</p>
    </div>
  );
}

// Simple 404 page component for unknown routes
function NotFoundPage() {
  return (
    <div className="not-found-page">
      <h1>404 - Page Not Found</h1>
      <p>The page you're looking for doesn't exist.</p>
    </div>
  );
}

// Main App Component with shared state and error handling
function App() {
  const predictionState = usePredictionState();

  const {
    schedule,
    week,
    predictions,
    loading,
    errors,
    current,
    history,
    health,
    setPrediction,
    setLoading,
    setError,
    pushHistory,
    resetHistory,
    count,
  } = predictionState;

  return (
    <ErrorBoundary>
      <Router>
        <div className="app-container">
          <Suspense fallback={<div role="status" aria-live="polite">Loading...</div>}>
            <Routes>
              <Route
                path="/"
                element={(
                  <Dashboard
                    schedule={schedule}
                    week={week}
                    predictions={predictions}
                    loading={loading}
                    errors={errors}
                    current={current}
                    history={history}
                    health={health}
                    setPrediction={setPrediction}
                    setLoading={setLoading}
                    setError={setError}
                    pushHistory={pushHistory}
                  />
                )}
              />
              <Route
                path="/history"
                element={(
                  <HistoryPage
                    history={history}
                    health={health}
                    onClearHistory={resetHistory}
                    historyCount={count}
                  />
                )}
              />
              <Route path="/stats" element={<StatsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </Suspense>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
