// ==========================================
// File: frontend/src/components/D_BUTTON.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react
// Notes: Presentation-focused component.
// ==========================================

/**
 * D_BUTTON.jsx — Clear History Button
 * -----------------------------------
 * Purpose:
 *   Provide a single, accessible control to clear the prediction history.
 *   Uses a parent-provided handler and relies on App state to persist.
 *
 * Contract:
 *   - No props required.
 *   - When clicked, clears history and announces the action to screen readers.
 *
 * Notes:
 *   - The parent state owns persistence; this button just triggers it.
 */
import React from "react";


export default function D_BUTTON({ onClear, count = 0, isClearing = false }) {
  const handleClear = () => {
    if (typeof onClear === 'function') {
      onClear();
    }
  };

  return (
    <button
      type="button"
      className="clear-history-button"
      onClick={handleClear}
      aria-label="Clear saved prediction history"
      title="Clear saved prediction history"
      disabled={count === 0 || isClearing}
      aria-busy={isClearing ? "true" : "false"}
    >
      {isClearing ? "Clearing history..." : "Clear saved history"}
    </button>
  );
}
