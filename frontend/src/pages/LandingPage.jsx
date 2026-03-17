import React, { useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './LandingPage.css';

const VALUE_PILLARS = [
  {
    title: '1. Sign in and start fast',
    body: 'Use your email to open the app, keep your recent activity on this device, and return without extra setup.',
  },
  {
    title: '2. Choose a matchup',
    body: 'Open the dashboard, select the next game you want to review, and generate a score forecast with win probabilities in one click.',
  },
  {
    title: '3. Review and track your calls',
    body: 'Use the forecast breakdown, saved history, and service overview to stay organized throughout the week.',
  },
];

const HERO_METRICS = [
  { label: 'Workflow', value: '3 Steps' },
  { label: 'Coverage', value: 'Weekly Slate' },
  { label: 'History', value: 'Saved Per User' },
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
          <p className="landing-kicker">Fast weekly forecasts, clearer decisions</p>
          <h1>Pick a matchup, review the outlook, and track your calls in one place.</h1>
          <p className="landing-lead">
            Sign in with your email to open the dashboard, generate matchup forecasts, and keep a
            simple history of the predictions you have already reviewed.
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
                <a className="landing-button landing-button--ghost" href="#access">
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
                  Your session stays on this device so your recent forecasts are ready the next
                  time you return.
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
