import React, { useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './LandingPage.css';

const VALUE_PILLARS = [
  {
    title: 'Signal-first weekly models',
    body: 'Schedule health, prediction history, and matchup context are surfaced in one place so the interface feels calm instead of crowded.',
  },
  {
    title: 'Fast research surface',
    body: 'Move from the premium landing experience into the prediction dashboard, stats view, and history trail without a separate auth backend.',
  },
  {
    title: 'Session-aware access flow',
    body: 'A local session gives you a real sign-in and sign-out path today while leaving room for a proper identity layer later.',
  },
];

const HERO_METRICS = [
  { label: 'Prediction Surface', value: '4 Views' },
  { label: 'Model Posture', value: 'Live Ready' },
  { label: 'Access Layer', value: 'Local Session' },
];

export default function LandingPage({ authSession, onSignIn, onSignOut }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState(authSession.user?.email || '');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const destinationLabel = useMemo(() => {
    const from = location.state?.from;
    return from && from !== '/' ? `Continue to ${from}` : 'Enter dashboard';
  }, [location.state]);

  const handleSubmit = (event) => {
    event.preventDefault();
    const result = onSignIn(email, password);
    if (!result.ok) {
      setErrorMessage(result.message);
      return;
    }
    setErrorMessage('');
    navigate('/app');
  };

  const handleContinue = () => navigate('/app');

  return (
    <main className="landing-shell">
      <div className="landing-noise" aria-hidden="true" />

      <header className="landing-topbar">
        <div className="landing-brand">
          <span className="landing-brand__mark" aria-hidden="true" />
          <div>
            <p className="landing-brand__eyebrow">NFL ML Predictions</p>
            <strong className="landing-brand__name">Private Forecast Studio</strong>
          </div>
        </div>

        <nav className="landing-nav" aria-label="Landing sections">
          <a href="#experience">Experience</a>
          <a href="#access">Access</a>
        </nav>

        <div className="landing-topbar__actions">
          {authSession.isAuthenticated ? (
            <>
              <button type="button" className="landing-button landing-button--ghost" onClick={onSignOut}>
                Sign out
              </button>
              <button type="button" className="landing-button landing-button--solid" onClick={handleContinue}>
                Open app
              </button>
            </>
          ) : (
            <a className="landing-button landing-button--ghost" href="#access">
              Sign in
            </a>
          )}
        </div>
      </header>

      <section className="landing-hero" id="experience">
        <div className="landing-copy">
          <p className="landing-kicker">Minimal, premium, and purpose-built</p>
          <h1>Model-driven NFL forecasting in a calmer, sharper front door.</h1>
          <p className="landing-lead">
            A new landing experience now frames the prediction product like a private studio:
            polished entry, cleaner routing, and immediate sign-in access without touching the
            existing prediction engine underneath.
          </p>

          <div className="landing-hero__actions">
            {authSession.isAuthenticated ? (
              <>
                <button type="button" className="landing-button landing-button--solid" onClick={handleContinue}>
                  {destinationLabel}
                </button>
                <button type="button" className="landing-button landing-button--ghost" onClick={onSignOut}>
                  Sign out
                </button>
              </>
            ) : (
              <>
                <a className="landing-button landing-button--solid" href="#access">
                  Sign in to continue
                </a>
                <a className="landing-button landing-button--ghost" href="#access">
                  Preview access flow
                </a>
              </>
            )}
          </div>

          <ul className="landing-metrics" aria-label="Key product metrics">
            {HERO_METRICS.map((item) => (
              <li key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </li>
            ))}
          </ul>
        </div>

        <aside className="landing-panel">
          <div className="landing-panel__glass">
            <p className="landing-panel__label">Access</p>
            {authSession.isAuthenticated ? (
              <div className="landing-session">
                <div>
                  <p className="landing-session__eyebrow">Signed in locally</p>
                  <h2>{authSession.user?.name}</h2>
                  <p>{authSession.user?.email}</p>
                </div>
                <div className="landing-session__meta">
                  <span>Session active</span>
                  <span>Dashboard protected</span>
                </div>
                <button type="button" className="landing-button landing-button--solid" onClick={handleContinue}>
                  Continue to dashboard
                </button>
              </div>
            ) : (
              <form className="landing-form" id="access" onSubmit={handleSubmit}>
                <label htmlFor="landing-email">
                  Email
                  <input
                    id="landing-email"
                    name="email"
                    type="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    placeholder="analyst@gridline.ai"
                    autoComplete="email"
                  />
                </label>
                <label htmlFor="landing-password">
                  Password
                  <input
                    id="landing-password"
                    name="password"
                    type="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="Minimum 6 characters"
                    autoComplete="current-password"
                  />
                </label>

                {errorMessage ? <p className="landing-form__error">{errorMessage}</p> : null}

                <button type="submit" className="landing-button landing-button--solid landing-button--block">
                  Sign in
                </button>
                <p className="landing-form__note">
                  This is a local session layer for the current frontend. It protects the app flow
                  without requiring a backend identity provider yet.
                </p>
              </form>
            )}
          </div>
        </aside>
      </section>

      <section className="landing-grid">
        {VALUE_PILLARS.map((pillar, index) => (
          <article key={pillar.title} className="landing-card">
            <p className="landing-card__eyebrow">{String(index + 1).padStart(2, '0')}</p>
            <h2>{pillar.title}</h2>
            <p>{pillar.body}</p>
          </article>
        ))}
      </section>
    </main>
  );
}
