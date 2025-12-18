// ==========================================
// File: frontend/src/components/ErrorDisplay.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ./ErrorDisplay.css
// Notes: Presentation-focused component.
// ==========================================

/*
File: ErrorDisplay.jsx
Purpose:
  Small presentational component for displaying recoverable errors
  (e.g., dashboard load failure) with an optional retry action.

Key ideas:
  - Uses role="alert" for accessibility.
  - Keeps error messaging and layout consistent across pages.
  - Styled via ErrorDisplay.css using the shared LCH color palette.
*/

import React from 'react';
import './ErrorDisplay.css';

/**
 * @param {{ error: any, onRetry?: () => void }} props
 */
export default function ErrorDisplay({ error, onRetry }) {
    if (!error) return null;

    const message = typeof error === 'string'
        ? error
        : error?.message || 'Something went wrong while loading the dashboard.';

    return (
        <section className="error-display" role="alert">
            <p className="error-display__message">{message}</p>
            {onRetry && (
                <button
                    type="button"
                    className="error-display__retry"
                    onClick={onRetry}
                >
                    Try again
                </button>
            )}
        </section>
    );
}
