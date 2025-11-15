/*
File: App.jsx
Purpose: Root React component; wraps DashBoard in PredictionProvider and ErrorBoundary for centralized state and error handling.
Functions: App (React component)
Variables: none (stateless wrapper)
Interacts With: PredictionContext (provides state), DashBoard (main UI), ErrorBoundary (error catch)

App.jsx
-------
Purpose:
  Root React component that wires up top-level routes for the NFL
  prediction UI. All prediction logic and state live in children
  (e.g., TeamGrid, HistoryPage), not in this file.

Architecture notes:
  - React Router is responsible for page-level navigation.
  - Global layout/styling is pulled in via TeamGrid.css as the main
    stylesheet entrypoint.

Change Log
  2025-11-11:
    - Replaced placeholder content with a working App component.
 *     - Fixed Dashboard import path and component name mismatch.
 *     - Removed unused imports (Link, useState, HistoryChart) to reduce noise.
 */

/**
 * Enhanced App.jsx with Error Boundaries and better data flow
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { PredictionProvider } from './contexts/PredictionContext';
import ErrorBoundary from './components/ErrorBoundary';
import LoadingFallback from './components/LoadingFallback';

// Lazy load components for better performance
const Dashboard = React.lazy(() => import('./components/Dashboard/Dashboard'));
const HistoryPage = React.lazy(() => import('./components/HistoryPage'));
const StatsPage = React.lazy(() => import('./pages/StatsPage'));
const SettingsPage = React.lazy(() => import('./pages/SettingsPage'));

// Enhanced Error Boundary Component
class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error('App Error Boundary caught an error:', error, errorInfo);

    // Send error to monitoring service
    this.logError(error, errorInfo);
  }

  logError(error, errorInfo) {
    // Implement error logging service integration
    if (window._ENV_?.NODE_ENV === 'production') {
      // Send to error monitoring service
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="app-error-boundary">
          <h1>Something went wrong</h1>
          <details style={{ whiteSpace: 'pre-wrap' }}>
            {this.state.error && this.state.error.toString()}
            <br />
            {this.state.errorInfo.componentStack}
          </details>
          <button onClick={() => window.location.reload()}>
            Reload Application
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Main App Component with enhanced data flow
function App() {
  return (
    <AppErrorBoundary>
      <Router>
        <PredictionProvider>
          <div className="app-container">
            <React.Suspense fallback={<LoadingFallback />}>
              <Routes>
                <Route
                  path="/"
                  element={<Dashboard />}
                  errorElement={<ErrorBoundary />}
                />
                <Route
                  path="/history"
                  element={<HistoryPage />}
                  errorElement={<ErrorBoundary />}
                />
                <Route
                  path="/stats"
                  element={<StatsPage />}
                  errorElement={<ErrorBoundary />}
                />
                <Route
                  path="/settings"
                  element={<SettingsPage />}
                  errorElement={<ErrorBoundary />}
                />
                <Route
                  path="*"
                  element={<NotFoundPage />}
                />
              </Routes>
            </React.Suspense>
          </div>
        </PredictionProvider>
      </Router>
    </AppErrorBoundary>
  );
}

// 404 Page Component
function NotFoundPage() {
  return (
    <div className="not-found-page">
      <h1>404 - Page Not Found</h1>
      <p>The page you're looking for doesn't exist.</p>
    </div>
  );
}

export default App;
    