// ==========================================
// File: frontend/src/components/LoadingState.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ./LoadingState.css
// Notes: Presentation-focused component.
// ==========================================

/*
File: LoadingState.jsx
Purpose:
  Reusable loading placeholder section for the dashboard and other views.
  Presents a simple spinner and descriptive message with accessible markup.

Key ideas:
  - Keeps loading UX consistent across pages.
  - Uses semantic <section> with aria-busy to aid assistive tech.
  - Styling is defined in LoadingState.css using the shared LCH palette.
*/

import React from 'react';
import './LoadingState.css';

/**
 * @param {{ message?: string }} props
 */
export default function LoadingState({ message = 'Loading…' }) {
    return (
        <section className="loading-state" aria-busy="true">
            <div className="loading-state__spinner" aria-hidden="true" />
            <p className="loading-state__message">{message}</p>
        </section>
    );
}
