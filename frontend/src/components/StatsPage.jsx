/**
 * StatsPage.jsx
 * --------------
 * Purpose:
 *   Placeholder route for future advanced stats and analytics.
 *   Keeps routing stable today while we build out real content.
 *
 * Contract:
 *   - No props required.
 *   - Renders a minimal, accessible placeholder.
 *
 * UX notes:
 *   - Keep some padding so content does not touch edges.
 *   - Use semantic headings for future SEO/accessibility.
 */
import NavBar from "./NavBar/NavBar";

export default function StatsPage() {
  return (
    <>

      <NavBar />
      <main aria-labelledby="stats-title" style={{ padding: 20 }}>
        <main aria-labelledby="stats-title" style={{ padding: 20 }}>
          <h2 id="stats-title">Team & Model Stats</h2>
          <p>Stats coming soon…</p>
        </main>
      </main>
    </>
  );
}