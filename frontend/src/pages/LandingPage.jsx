import React, { useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './LandingPage.css';

const VALUE_PILLARS = [
  {
    title: 'Sign in once',
    body: 'Open the app with your email so forecasts, history, and matchup context stay tied to this device.',
  },
  {
    title: 'Browse the 2026 slate',
    body: 'Move through the upcoming regular season, archived weeks, and saved predictions from one workspace.',
  },
  {
    title: 'Review every result',
    body: 'History saves each forecast alongside final scores as games resolve through the season.',
  },
];

const HERO_METRICS = [
  { label: '2026 schedule', value: '272 games' },
  { label: 'Pipeline', value: 'Contract checked' },
  { label: 'History', value: 'Saved per user' },
];

export default function LandingPage({ authSession, onSignIn, onSignOut }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState(authSession.user?.email || '');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const destinationLabel = useMemo(() => {
    const from = location.state?.from;
    return from && from !== '/' ? 'Continue where you left off' : 'Open dashboard';
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
            <strong className="landing-brand__name">Game Forecast Center</strong>
          </div>
        </div>

        <nav className="landing-nav" aria-label="Landing sections">
          <a href="#experience">How it works</a>
          <a href="#access">Sign in</a>
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
          <p className="landing-kicker">2026 schedule-ready forecasting</p>
          <h1>Forecast upcoming NFL matchups with a checked model bundle and clean future-game rows.</h1>
          <p className="landing-lead">
            Sign in to capture forecasts, browse the current or past slate, and compare each prediction
            once final scores sync back into your history.
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
                  Open the app
                </a>
                <a className="landing-button landing-button--ghost" href="#experience">
                  See the workflow
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
                  <p className="landing-session__eyebrow">Signed in on this device</p>
                  <h2>{authSession.user?.name}</h2>
                  <p>{authSession.user?.email}</p>
                </div>
                <div className="landing-session__meta">
                  <span>History ready</span>
                  <span>Dashboard available</span>
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
                    placeholder="you@example.com"
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
                  Your session stays on this device, and final scores refresh as completed games are synced.
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
