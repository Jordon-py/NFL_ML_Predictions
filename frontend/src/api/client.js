/**
 * client.js
/**
 * client.js
 * ---------
 * Thin client for backend API.
/**
 * client.js
 * ---------
 * Thin client for backend API. Uses environment variables available via
 * `import.meta.env` (Vite) to select the backend base URL.
 * 
 * Environment Variables:
 * - VITE_API_URL: Override API base URL (e.g., "http://localhost:8000")
 * - Falls back to production Heroku URL if not set
 */

const BASE_URL =
  import.meta.env.VITE_API_URL ||
  'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';

// Debug log for development
if (import.meta.env.DEV) {
  console.log('[API Client] Using BASE_URL:', BASE_URL);
  console.log('[API Client] Mode:', import.meta.env.MODE);
}

function buildUrl(path) {
  // Prefer the standard URL constructor which normalizes slashes.
  try {
    return new URL(path, BASE_URL).toString();
  } catch (err) {
    // Fallback: trim trailing/leading slashes and join
    return `${BASE_URL.replace(/\/+$|$/,'').replace(/\/+$|^$/,'')}`.replace(/\/+$/, '') + '/' + String(path).replace(/^\/+/, '');
  }
}

async function api(path, opts = {}) {
  const url = buildUrl(path);
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API ${path} failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function getNextWeekSchedule() {
  return api('schedule/next-week');
}

export async function predictGame(body) {
  return api('predict', { method: 'POST', body: JSON.stringify(body) });
}

