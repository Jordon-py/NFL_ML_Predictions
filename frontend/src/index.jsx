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

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';

// Grab the static DOM node that Vite injects for us.
const rootElement = document.getElementById('root');

// React 18's concurrent root API replaces the legacy render function.
const root = ReactDOM.createRoot(rootElement);

// Render the application tree. StrictMode intentionally double-renders in dev
// to help find impure render logic. The ErrorBoundary ensures the user still
// sees a helpful message if a descendant throws.
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);