import React, {Component} from 'react';

/**
 * ErrorBoundary.jsx
 * -----------------
 * Component Purpose:
 *   Catch runtime errors in descendant components and render a fallback UI
 *   instead of letting the entire React tree unmount.
 *
 * Core Logic Overview:
 *   - React calls `getDerivedStateFromError` when a descendant throws; we flip
 *     `hasError` there to trigger the fallback.
 *   - `componentDidCatch` records diagnostic info so you can log/report.
 *
 * Modification Guide:
 *   - Replace the fallback markup with a branded error screen, but keep
 *     essential diagnostics for debugging.
 *   - If you introduce error logging (Sentry, etc.), call it inside
 *     `componentDidCatch` where both error and component stack are available.
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = {hasError: false, error: null, errorInfo: null};
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render shows the fallback UI.
    return {hasError: true};
  }

  componentDidCatch(error, errorInfo) {
    // Persist details for troubleshooting or remote logging.
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    console.error("ErrorBoundary caught an error", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{padding: '20px', border: '1px solid red', margin: '20px'}}>
          <h2>Something went wrong.</h2>
          <details style={{whiteSpace: 'pre-wrap'}}>
            {this.state.error && this.state.error.toString()}
            <br />
            {this.state.errorInfo && this.state.errorInfo.componentStack}
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
