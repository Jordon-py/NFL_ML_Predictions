// ==========================================
// File: frontend/src/App.jsx
// Role: Frontend module.
// Input Data: Module inputs.
// Output Data: Exports for UI usage.
// Dependencies: react, react-router-dom, ./components/ErrorBoundary, ./hooks/usePredictionState
// Notes: Shared application code.
// ==========================================

import React, { Suspense, lazy } from 'react';
import {
  BrowserRouter as Router,
  Navigate,
  Route,
  Routes,
  useLocation,
} from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import { usePredictionState } from './hooks/usePredictionState';
import { useAuthSession } from './hooks/useAuthSession';
import NavBar from './components/NavBar/NavBar';

const Dashboard = lazy(() => import('./components/DashBoard/Dashboard'));
const HistoryPage = lazy(() => import('./components/HistoryPage'));
const StatsPage = lazy(() => import('./pages/StatsPage'));
const LandingPage = lazy(() => import('./pages/LandingPage'));

function SettingsPage({ authSession, onSignOut }) {
  return (
    <div className="settings-page-shell">
      <NavBar authSession={authSession} onSignOut={onSignOut} />
      <main className="settings-page">
        <div className="settings-page__card">
          <p className="settings-page__eyebrow">Settings</p>
          <h1>Account and application controls are still being built.</h1>
          <p>
            The new landing and access flow are live. Settings can be layered in without
            changing the session model that now protects the app experience.
          </p>
        </div>
      </main>
    </div>
  );
}

function NotFoundPage({ isSignedIn }) {
  return (
    <div className="not-found-page">
      <h1>404</h1>
      <p>The page you requested does not exist in this app shell.</p>
      <a href={isSignedIn ? '/app' : '/'}>{isSignedIn ? 'Return to dashboard' : 'Return to landing'}</a>
    </div>
  );
}

function PredictionAppRoutes({ authSession, onSignOut }) {
  const predictionState = usePredictionState(authSession);
  const {
    schedule,
    week,
    predictions,
    loading,
    errors,
    current,
    history,
    health,
    seasonContext,
    setPrediction,
    setLoading,
    setError,
    pushHistory,
    resetHistory,
    count,
  } = predictionState;

  return (
    <Routes>
      <Route
        path="app"
        element={(
          <Dashboard
            authSession={authSession}
            onSignOut={onSignOut}
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
        path="history"
        element={(
          <HistoryPage
            authSession={authSession}
            onSignOut={onSignOut}
            history={history}
            health={health}
            onClearHistory={resetHistory}
            historyCount={count}
          />
        )}
      />
      <Route
        path="stats"
        element={<StatsPage authSession={authSession} onSignOut={onSignOut} />}
      />
      <Route
        path="settings"
        element={<SettingsPage authSession={authSession} onSignOut={onSignOut} />}
      />
      <Route path="*" element={<NotFoundPage isSignedIn />} />
    </Routes>
  );
}

function ProtectedAppShell({ authSession, onSignOut }) {
  const location = useLocation();

  // Redirect anonymous visitors back to the landing page while preserving the intent.
  if (!authSession.isAuthenticated) {
    return <Navigate to="/" replace state={{ from: location.pathname }} />;
  }

  return <PredictionAppRoutes authSession={authSession} onSignOut={onSignOut} />;
}

function App() {
  const authSession = useAuthSession();

  return (
    <ErrorBoundary>
      <Router>
        <div className="app-container">
          <Suspense fallback={<div role="status" aria-live="polite">Loading...</div>}>
            <Routes>
              <Route
                path="/"
                element={(
                  <LandingPage
                    authSession={authSession}
                    onSignIn={authSession.signIn}
                    onSignOut={authSession.signOut}
                  />
                )}
              />
              <Route
                path="/*"
                element={(
                  <ProtectedAppShell
                    authSession={authSession}
                    onSignOut={authSession.signOut}
                  />
                )}
              />
            </Routes>
          </Suspense>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
