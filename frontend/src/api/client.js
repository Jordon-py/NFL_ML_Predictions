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
 * `process.env` to select the backend base URL so it works with both CRA
 * and Vite-based setups.
 */

const BASE_URL =
  process.env.REACT_APP_API_URL || process.env.VITE_API_URL ||
  'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';

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

