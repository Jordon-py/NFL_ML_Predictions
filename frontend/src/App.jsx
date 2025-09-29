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
import DashBoard from './components/DashBoard.jsx'; // FIX: default import, not named
import ErrorBoundary from './components/ErrorBoundary.jsx';
import { PredictionProvider } from './PredictionContext.js';
import { SpeedInsights } from "@vercel/speed-insights/react"

export default function App() {
  // Keep App focused on composition. All prediction state lives in Context.
  return (
    <SpeedInsights>
      <ErrorBoundary>
        <PredictionProvider>
          <DashBoard />
        </PredictionProvider>
      </ErrorBoundary>
    </SpeedInsights>
  );
}
