<<<<<<< HEAD
import { Suspense, lazy, useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NavBar from './components/NavBar/NavBar.jsx';
import { getStatusOverview } from './api/client.js';

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

// Main App Component with centralized context and error handling
function App() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const overview = await getStatusOverview();
        if (!active) return;
        setHealth(overview?.health ?? null);
      } catch {
        if (!active) return;
        setHealth(null);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  return (
    <Router>
      <NavBar health={health} />
      <div className="app-container">
        <Suspense fallback={<div role="status" aria-live="polite">Loading…</div>}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/stats" element={<StatsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Suspense>
      </div>
    </Router>
  );
}

=======
/**
 * App.jsx
 * 
 * Root React application component for the NFL prediction UI.
 * 
 * - Provides prediction context to all children via PredictionProvider.
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
import { PredictionProvider } from './PredictionContext';
import ErrorBoundary from './components/ErrorBoundary';

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

// Main App Component with centralized context and error handling
function App() {
  return (
    <ErrorBoundary>
      <PredictionProvider>
        <Router>
          <div className="app-container">
            <Suspense fallback={<div role="status" aria-live="polite">Loading...</div>}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/stats" element={<StatsPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="*" element={<NotFoundPage />} />
              </Routes>
            </Suspense>
          </div>
        </Router>
      </PredictionProvider>
    </ErrorBoundary>
  );
}

>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
export default App;
