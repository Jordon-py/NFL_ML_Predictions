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

export default App;
