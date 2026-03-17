// ==========================================
// File: frontend/src/components/ErrorBoundary.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ./ErrorBoundary.css
// Notes: Presentation-focused component.
// ==========================================

import { Component } from 'react';
import './ErrorBoundary.css';

/**
 * ErrorBoundary.jsx
 * -----------------
 * Purpose: Catch runtime errors in descendants and show fallback UI.
 * Fixed: Ensured all JSX is properly closed and imports are valid to prevent 500 compilation errors.
 */

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error, errorInfo: null };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ error, errorInfo });
    console.error("ErrorBoundary caught an error", error, errorInfo);

    // Report error to backend for observability
    try {
      const payload = JSON.stringify({
        message: error?.message,
        stack: error?.stack,
        componentStack: errorInfo?.componentStack,
        url: window.location.href,
        userAgent: navigator.userAgent,
        ts: new Date().toISOString(),
      });

      // sendBeacon avoids blocking unload/navigation
      if (navigator.sendBeacon) {
        navigator.sendBeacon('/api/client-errors', new Blob([payload], { type: 'application/json' }));
      }
    } catch (reportError) {
      console.warn('Failed to report error to backend:', reportError);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-container">
          <h2>Something went wrong.</h2>
          {this.state.error && (
            <div className="error-summary">
              <strong>Error:</strong> {this.state.error.toString()}
            </div>
          )}
          {this.state.errorInfo && (
            <details style={{ whiteSpace: 'pre-wrap' }}>
              {this.state.errorInfo.componentStack}
            </details>
          )}
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
