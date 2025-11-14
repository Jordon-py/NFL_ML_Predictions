import React from "react";
import styles from "./Card.module.css";

/**
 * Card v2 — Prop-driven & motion-aware
 * Props:
 *  - matchup: { away_team, home_team, kickoff, away_logo, home_logo }
 *  - prediction?: { home_win_probability, away_win_probability, home_score?, away_score?, point_diff? }
 *  - title?: string  (e.g., "AI Node")
 *  - status?: "Active" | "Idle" | "Error" | string
 *  - icon?: ReactNode
 *  - progress?: number (0..100)
 *  - loading?: boolean
 *  - error?: string
 *  - index?: number (stagger)
 *  - onClick?: () => void
 */
export default function Card({
  matchup,
  prediction,
  title,
  status,
  icon,
  progress,
  loading = false,
  error,
  index = 0,
  onClick
}) {
  if (!matchup) return null;

  const { away_team, home_team, kickoff, away_logo, home_logo } = matchup;
  const hasPrediction = !!prediction;

  const pct = (v) =>
    typeof v === "number" && isFinite(v) ? Math.round(v * 100) : null;

  return (
    <article
      className={[
        styles.card,
        hasPrediction ? styles.hasPrediction : "",
        loading ? styles.isLoading : "",
        error ? styles.isError : "",
      ].join(" ")}
      style={{ ["--i"]: index }}
      tabIndex={0}
      role="button"
      aria-pressed={loading ? "true" : "false"}
      onClick={onClick}
      onKeyDown={(e) => (e.key === "Enter" ? onClick?.(e) : null)}
    >
      {/* Top bar: optional icon/title/status */}
      {(title || status || icon) && (
        <div className={styles.topBar}>
          <div className={styles.left}>
            {icon && <span className={styles.icon} aria-hidden>{icon}</span>}
            {title && <strong className={styles.title}>{title}</strong>}
          </div>
          {status && <span className={styles.status}>{status}</span>}
        </div>
      )}

      <header className={styles.head}>
        <div className={styles.teamsRow}>
          <div className={[styles.teamInfo, styles.away].join(" ")}>
            <img className={styles.teamLogo} src={away_logo} alt={`${away_team} logo`} />
            <span className={styles.teamName}>{away_team}</span>
          </div>
          <span className={styles.atSymbol} aria-hidden>@</span>
          <div className={[styles.teamInfo, styles.home].join(" ")}>
            <img className={styles.teamLogo} src={home_logo} alt={`${home_team} logo`} />
            <span className={styles.teamName}>{home_team}</span>
          </div>
        </div>
        <div className={styles.meta}>
          <time className={styles.kickoff} dateTime={kickoff}>
            {kickoff ? new Date(kickoff).toLocaleString() : "TBD"}
          </time>
        </div>
      </header>

      <section className={styles.prediction}>
        {loading ? (
          <p className={styles.cta} aria-live="polite">Predicting…</p>
        ) : error ? (
          <p className={styles.errorMsg} role="status">{error}</p>
        ) : hasPrediction ? (
          <div className={styles.predictionBody}>
            <div className={styles.probRow}><span>Home</span><b>{pct(prediction.home_win_probability)}%</b></div>
            <div className={styles.probRow}><span>Away</span><b>{pct(prediction.away_win_probability)}%</b></div>
            {/* Optional numeric details if present */}
            {(prediction.home_score != null || prediction.away_score != null || prediction.point_diff != null) && (
              <div className={styles.detailRow}>
                <span>Score</span>
                <b>
                  {away_team} {prediction.away_score ?? "—"} – {prediction.home_score ?? "—"} {home_team}
                  {prediction.point_diff != null && <em className={styles.diff}> • Δ {prediction.point_diff.toFixed?.(1) ?? prediction.point_diff}</em>}
                </b>
              </div>
            )}
          </div>
        ) : (
          <p className={styles.cta} aria-live="polite">Click to generate prediction</p>
        )}
      </section>

      {/* Optional progress meter */}
      {typeof progress === "number" && isFinite(progress) && (
        <div className={styles.progressTrack} aria-hidden>
          <div className={styles.progressBar} style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
        </div>
      )}
    </article>
  );
}
