// ==========================================
// File: frontend/src/components/NavBar/NavBar.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, react-router-dom, ./NavBar.css
// Notes: Presentation-focused component.
// ==========================================

import React, { useEffect, useMemo, useState } from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import './NavBar.css';

const NAV_ITEMS = [
  { to: '/app', label: 'Dashboard', end: true },
  { to: '/history', label: 'History' },
  { to: '/stats', label: 'Stats' },
  { to: '/settings', label: 'Settings' },
];

const ROUTE_LABELS = {
  '/app': 'Prediction Dashboard',
  '/history': 'Historical Trail',
  '/stats': 'System Status',
  '/settings': 'Workspace Settings',
};

function formatTimestamp(value) {
  if (!value) return 'Session ready';
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) return 'Session active';
  return `Signed in ${timestamp.toLocaleDateString()}`;
}

/**
 * NavBar.jsx
 * ----------
 * Purpose:
 *   Shared navigation shell for all authenticated views.
 *
 * Notes:
 *   - A small scroll threshold tightens the glass treatment once the page starts moving.
 *   - The mobile menu is internal to keep route links and sign-out controls in one component.
 *   - `state` is intentionally permissive because older pages only pass health details today.
 */
function NavBar({ state = {}, authSession, onSignOut }) {
  const [isSticking, setIsSticking] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const health = state?.health ?? state;

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;

    const syncScrollState = () => {
      setIsSticking(window.scrollY > 14);
    };

    syncScrollState();
    window.addEventListener('scroll', syncScrollState, { passive: true });

    return () => {
      window.removeEventListener('scroll', syncScrollState);
    };
  }, []);

  useEffect(() => {
    setIsMenuOpen(false);
  }, [location.pathname]);

  const healthTone = health?.status === 'healthy'
    ? 'ok'
    : health?.status === 'unhealthy'
      ? 'error'
      : 'unknown';

  const pageTitle = state?.title || ROUTE_LABELS[location.pathname] || 'Forecast Workspace';
  const pageSubtitle = state?.heroSubtitle || state?.subtitle || 'Protected forecasting surfaces';
  const healthLabel = state?.healthLabel || `Backend: ${health?.status ?? 'unknown'}`;
  const weekLabel = state?.weekLabel || null;
  const signedInLabel = formatTimestamp(authSession?.user?.signedInAt);
  const canSignOut = typeof onSignOut === 'function' && authSession?.isAuthenticated;

  const userInitials = useMemo(() => {
    const name = authSession?.user?.name || authSession?.user?.email || 'NF';
    return name
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part.charAt(0).toUpperCase())
      .join('');
  }, [authSession?.user?.email, authSession?.user?.name]);

  const handleSignOut = () => {
    if (!canSignOut) return;
    onSignOut();
    navigate('/', { replace: true });
  };

  return (
    <header className={`navBar ${isSticking ? 'sticking' : ''}`}>
      <div className="navBar__surface">
        <div className="navBar__brandCluster">
          <NavLink className="navBar__brand" to={authSession?.isAuthenticated ? '/app' : '/'}>
            <span className="navBar__brandMark" aria-hidden="true" />
            <span className="navBar__brandCopy">
              <span className="navBar__eyebrow">NFL ML Predictions</span>
              <span className="navBar__headline">Forecast Studio</span>
            </span>
          </NavLink>

          <div className="navBar__pageSummary">
            <span className="navBar__pageTitle">{pageTitle}</span>
            <span className="navBar__pageSubtitle">{pageSubtitle}</span>
          </div>
        </div>

        <button
          type="button"
          className="navBar__menuButton"
          aria-expanded={isMenuOpen}
          aria-controls="app-primary-menu"
          onClick={() => setIsMenuOpen((current) => !current)}
        >
          {isMenuOpen ? 'Close' : 'Menu'}
        </button>

        <div className={`navBar__menu ${isMenuOpen ? 'is-open' : ''}`} id="app-primary-menu">
          <nav className="navBar__links" aria-label="Primary">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                className={({ isActive }) => `navBar__link ${isActive ? 'is-active' : ''}`}
                end={item.end}
                to={item.to}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="navBar__meta">
            <div className={`navBar__status navBar__status--${healthTone}`} title={health?.reason || healthLabel}>
              <span className="navBar__statusDot" aria-hidden="true" />
              <span>{healthLabel}</span>
            </div>

            {weekLabel ? <span className="navBar__metaPill">{weekLabel}</span> : null}

            {authSession?.isAuthenticated ? (
              <div className="navBar__user">
                <span className="navBar__avatar" aria-hidden="true">{userInitials}</span>
                <span className="navBar__userCopy">
                  <strong>{authSession.user?.name}</strong>
                  <span>{signedInLabel}</span>
                </span>
              </div>
            ) : null}

            {canSignOut ? (
              <button type="button" className="navBar__signOut" onClick={handleSignOut}>
                Sign out
              </button>
            ) : null}
          </div>
        </div>
      </div>
    </header>
  );
}

export default NavBar;
