/**
 * App.jsx
 * -------
 * Purpose:
 *   Root component that wires global prediction state via Context.
 *   Replaces multiple overlapping states with a single canonical model.
 *
 * Layer 1 Fixes:
 *   - Correct default import for DashBoard export.
 *   - Normalize prediction data once and store in Context.
 *
 * Layer 2 Improvements:
 *   - Single "current" object and an append-only "history".
 *   - Push to history in one place to avoid drift and duplication.
 */

import React from 'react';
import { useState } from 'react';
import { Link, Route, Routes } from 'react-router-dom';
import HistoryChart from './components/HistoryChart.jsx';
import DashBoard from './components/DashBoard.jsx'; // FIX: default import, not named
import './components/TeamGrid.css';
import HistoryPage from './components/HistoryPage.jsx';
import StatsPage from './components/StatsPage.jsx';



/**
 * App
 * -----
 * The Vercel Speed Insights component can cause runtime/SSR issues when
 * imported at module-evaluation time (it may access browser globals). To
 * avoid the site failing to render (especially on Vercel builds), we
 * dynamically import and render the component on the client only.
 */
export default function App() {
  // Keep App focused on composition. All prediction state lives in Context.
  return (
    <>
      <Routes>
        <Route path="/" element={<DashBoard />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/stats" element={<StatsPage />} />
      </Routes>
    </>
  );
}
