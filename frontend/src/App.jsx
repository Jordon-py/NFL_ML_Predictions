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

import React, { useState, useEffect } from 'react';
import DashBoard from './components/DashBoard.jsx'; // FIX: default import, not named
import ErrorBoundary from './components/ErrorBoundary.jsx';
import { PredictionProvider } from './PredictionContext.js';

/**
 * App
 * -----
 * The Vercel Speed Insights component can cause runtime/SSR issues when
 * imported at module-evaluation time (it may access browser globals). To
 * avoid the site failing to render (especially on Vercel builds), we
 * dynamically import and render the component on the client only.
 */
export default function App() {
  const [SpeedInsightsComp, setSpeedInsightsComp] = useState(null);

  useEffect(() => {
    let mounted = true;
    // Dynamically import on the client. If the package isn't installed or
    // the import fails, we silently fall back to rendering children.
    import('@vercel/speed-insights/react')
      .then((mod) => {
        if (!mounted) return;
        // Support both named and default exports
        const Comp = mod.SpeedInsights || mod.default || null;
        setSpeedInsightsComp(() => Comp);
      })
      .catch(() => {
        // Ignore failures — Speed Insights is optional and should not
        // prevent the app from rendering.
      });
    return () => {
      mounted = false;
    };
  }, []);

  const Wrapper = ({ children }) => {
    if (SpeedInsightsComp) {
      const SI = SpeedInsightsComp;
      return <SI>{children}</SI>;
    }
    // Client not ready or module not available: render children directly.
    return <>{children}</>;
  };

  // Keep App focused on composition. All prediction state lives in Context.
  return (
    <Wrapper>
      <ErrorBoundary>
        <PredictionProvider>
          <DashBoard />
        </PredictionProvider>
      </ErrorBoundary>
    </Wrapper>
  );
}
