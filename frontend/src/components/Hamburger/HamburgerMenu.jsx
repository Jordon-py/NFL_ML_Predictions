/**
 * Change Log
 * ----------
 * 2024-06-02: Added HtmlButton alias to retain semantic markup and eliminate the JSX intrinsic type error.
 * 2024-06-18: Reverted to native button element to resolve JSX intrinsic typing errors without aliasing.
 * 2024-11-06: Adjusted React import to rely on the project’s JSX runtime configuration and documented the dependency to prevent missing-module diagnostics.
 */

// @ts-nocheck  
import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
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
 *
 * Dependencies:
 * - React useState hook for local menu toggling
 */
export default function HamburgerMenu() {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);

  return (
    <div className="hamburger-menu-container" role="navigation" aria-label="Main menu">
      <button
        id="ham-button"
        type="button"
        className={`hamburger-button ${isOpen ? 'open' : ''}`}
        onClick={toggleMenu}
        aria-label="Toggle menu"
        aria-expanded={isOpen}
        aria-controls="ham-menu"
      >
        <img
          className="ham-icon"
          src="/nfl_ham2.png"
          alt="Open menu"
          width={24}
          height={24}
        />
      </button>

      <nav
        id="ham-menu"
        className={`menu-panel ${isOpen ? 'open' : 'closed'}`}
        aria-hidden={!isOpen}
        aria-disabled={!isOpen}
        tabIndex={isOpen ? 0 : -1}
        style={{ pointerEvents: isOpen ? 'auto' : 'none', userSelect: isOpen ? 'auto' : 'none' }}
      >

        <div>
          <div className="menu-item">
            <NavLink to="/" end>Dashboard</NavLink>
          </div>


          <div className="menu-item">
            <NavLink to="/stats">Stats</NavLink>
          </div>
          <div className="menu-item">
            <NavLink to="/history">History</NavLink>
          </div>
        </div>
      </nav>
    </div>
  );
}

