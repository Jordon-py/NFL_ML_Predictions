import React, { useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './LandingPage.css';

const VALUE_PILLARS = [
  {
    title: '1. Sign in once',
    body: 'Open the app with your email so your work stays cached on this device and you can pick up exactly where you left off.',
  },
  {
    title: '2. Browse any slate',
    body: 'Toggle current or past weeks and seasons, then pick a matchup to generate a polished forecast in a single tap.',
  },
  {
    title: '3. Track predictions vs. final score',
    body: 'History saves every forecast alongside the final score that syncs after Sunday, Monday, and Thursday nights.',
  },
];

const HERO_METRICS = [
  { label: 'Score sync', value: 'Sun · Mon · Thu' },
  { label: 'Season navigation', value: 'Live + archives' },
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
          <p className="landing-kicker">Weekly slates, finalized results</p>
          <h1>Forecast any matchup, then compare how your call stacked up once the final score syncs.</h1>
          <p className="landing-lead">
            Sign in to capture forecasts, browse the current or past slate, and rely on synced final scores
            after Sunday, Monday, and Thursday night games.
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
                  Your session stays on this device and final scores refresh automatically after every Sunday,
                  Monday, and Thursday night slate.
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
