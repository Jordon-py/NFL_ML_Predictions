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

import React from 'react';
import { Routes, Route } from 'react-router-dom';

// Components (relative to src/)
import Dashboard from './components/Dashboard/Dashboard.jsx';
import HistoryPage from './components/HistoryPage.jsx';

// Pages
import StatsPage from './pages/StatsPage.jsx';

// Global styles (root entry point for shared layout + theme)
import './components/Card/TeamGrid.css';

/**
 * App
 * -----
 * Composes top-level pages using React Router:
 *   - "/"        → Dashboard (next-week matchups + predictions)
 *   - "/history" → HistoryPage (prediction logs and charts)
 *   - "/stats"   → StatsPage (model / data stats)
 */
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/history" element={<HistoryPage />} />
      <Route path="/stats" element={<StatsPage />} />
    </Routes>
  );
}
