// ==========================================
// File: frontend/src/components/HamburgerMenu.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ./HamburgerMenu.css
// Notes: Presentation-focused component.
// ==========================================


import React, { useState } from 'react';
import './HamburgerMenu.css';

/**
 * HamburgerMenu Component
 * -----------------------
 * A collapsible hamburger menu with toggle functionality.
 * 
 * Features:
 * - Click to toggle open/closed state
 * - CSS-driven animations for smooth transitions
 * - Accessible button with proper ARIA attributes
 * 
 * Usage:
 *   <HamburgerMenu />
 */
export default function HamburgerMenu() {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);

  return (
    <div className="hamburger-menu-container">
      <button
        className={`hamburger-button ${isOpen ? 'open' : ''}`}
        onClick={toggleMenu}
        aria-label="Toggle menu"
        aria-expanded={isOpen}
      >
        <span className="hamburger-icon">
          
        </span>
      </button>

      <nav className={`menu-panel ${isOpen ? 'open' : 'closed'}`}>
        <div className="menu-item">
          <a href="/">Dashboard</a>
        </div>
        <div className="menu-item">
          <a href="/predictions">Predictions</a>
        </div>
        <div className="menu-item">
          <a href="/history">History</a>
        </div>
      </nav>
    </div>
  );
}


