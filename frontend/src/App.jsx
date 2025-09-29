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
    // Only load Speed Insights when explicitly enabled via env var and when
    // the current host is allowed. This prevents the client script from
    // attempting to POST vitals to /_vercel/speed-insights/vitals on domains
    // where the endpoint is not configured (which causes console errors).
    const enabled = (process.env.REACT_APP_SPEED_INSIGHTS || 'false').toLowerCase() === 'true';
    const host = typeof window !== 'undefined' ? window.location.host : '';
    const allowedHosts = [
      'localhost',
      '127.0.0.1',
      // allow your known preview/dev hostnames if needed
      'nfl-ml-predictions.vercel.app',
    ];
    const hostAllowed = allowedHosts.some((h) => host.includes(h));

    if (!enabled || !hostAllowed) {
      // Do not import — either disabled by env or host not whitelisted.
      return () => {
        mounted = false;
      };
    }

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
