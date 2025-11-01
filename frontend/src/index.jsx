/**
 * index.jsx
 * ---------
 * Component Purpose:
 *   Hydrates the React application into the static `index.html` shell.
 *   We wrap everything in `React.StrictMode` so React can surface double-invocation
 *   warnings in development, and we keep a top-level `ErrorBoundary` here as a
 *   last line of defence in case the boundary in `App` fails earlier.
 *
 * Core Logic Overview:
 *   - Locate the `#root` mount point.
 *   - Create a React root (React 18 API).
 *   - Render the app tree inside a defensive error boundary.
 *
 * Modification Guide:
 *   - Add providers (Routing, Query, Theme, etc.) by nesting them inside
 *     `<React.StrictMode>` but outside `<App />` so the entire tree can access them.
 *   - Preserve `<ErrorBoundary>` wrapper or replace it with your custom boundary
 *     to avoid uncaught errors crashing the whole page.
 */
import App from './App';
import React from 'react';
import ReactDOM from 'react-dom/client';
import ErrorBoundary from './components/ErrorBoundary';
import { PredictionProvider } from './PredictionContext';
import './styles/base.css';        // ← load first
import './styles/theme-grid.css';  // ← load second
// index.jsx
import '@material/web/button/filled-button.js';
import '@material/web/button/outlined-button.js';
import '@material/web/checkbox/checkbox.js';


// Grab the static DOM node that Vite injects for us.
const rootElement = document.getElementById('root');

// React 18's concurrent root API replaces the legacy render function.
const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <PredictionProvider>
        <App />
      </PredictionProvider>
    </ErrorBoundary>
  </React.StrictMode>
);