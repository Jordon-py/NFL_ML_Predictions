/**
 * client.js
 * ---------
 * Purpose:
 *   Typed-ish thin client for backend API.
 *
 * Layer 1 Fix:
 *   - Standardize response key names used across the app:
 *       { home_score, away_score, point_diff,
 *         home_win_probability, away_win_probability, ensemble_probability? }
 *   - Avoid mismatches like home_win_prob vs home_win_probability.
 *
 * Deployment Note:
 *   Set VITE_API_URL in your front-end host to avoid fallbacks.
 */

const BASE_URL =
  import.meta?.env?.VITE_API_URL ||
  process.env?.VITE_API_URL ||
  // Fallback is acceptable for local dev but set Vercel env for prod:
  'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';

async function api(path, opts = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API ${path} failed: ${res.status} ${text}`);
  }
  return res.json();
}

/**
 * Returns:
 *   Array<{ season, week, game_id?, home_abbr, away_abbr, game_time? }>
 */
export async function getNextWeekSchedule() {
  return api('/schedule/next-week');
}

/**
 * Args:
 *   { season:number, week:number, home_abbr:string, away_abbr:string }
 *
 * Returns:
 *   {
 *     home_score:number,
 *     away_score:number,
 *     point_diff:number,
 *     home_win_probability?:number,  // 0..1
 *     away_win_probability?:number,  // 0..1
 *     ensemble_probability?:number   // 0..1
 *   }
 */
export async function predictGame(body) {
  return api('/predict', { method: 'POST', body: JSON.stringify(body) });
}
