/**
 * Production Frontend: https://nfl-ml-predictions.vercel.app
 * Backend: https://nfl-predict-ecf5a5bd34fe.herokuapp.com
 * Local Dev: frontend http://localhost:3000 ↔ backend http://localhost:8000
 */

const LOCAL_BACKEND = 'http://localhost:8000';
const PROD_BACKEND = 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';

// Intelligent environment selection
const BASE_URL =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? LOCAL_BACKEND : PROD_BACKEND);

console.log(`[API Client] Environment: ${import.meta.env.MODE}`);
console.log(`[API Client] Backend: ${BASE_URL}`);

function buildUrl(path) {
  try {
    return new URL(path, BASE_URL).toString();
  } catch (err) {
    const base = BASE_URL.replace(/\/+$/, '');
    const cleanPath = String(path).replace(/^\/+/, '');
    return `${base}/${cleanPath}`;
  }
}

// Safe API wrapper with graceful fallback
async function api(path, opts = {}) {
  const url = buildUrl(path);
  try {
    const res = await fetch(url, {
      headers: {'Content-Type': 'application/json'},
      ...opts,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`API ${path} failed: ${res.status} ${text}`);
    }
    return await res.json();
  } catch (err) {
    console.error(`[API Error] ${path}:`, err.message);
    return {error: true, message: err.message};
  }
}

export async function getNextWeekSchedule() {
  return api('schedule/next-week');
}

export async function predictGame(body) {
  if (import.meta.env.DEV) console.log('[API Client] predictGame:', body);
  return api('predict', {method: 'POST', body: JSON.stringify(body)});
}
