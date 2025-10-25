// @ts-nocheck
// client.js
/**
 * NFL-ML Client API wrapper (fixed)
 * - Correct PROD detection
 * - Robust env base handling
 * - Vite proxy friendly in dev
 */

const DEFAULT_TIMEOUT_MS = 15000;

/**
 * @param {string} base
 * @returns {string}
 */
function normalizeBase(base) {
  if (!base) return "";
  let url = base.split(',')[0].trim();
  return url.toLowerCase();
}

// In dev: empty string enables Vite proxy (`server.proxy`) so calls go to the backend
// In prod: require absolute base, fallback to Heroku URL
export const API_BASE = normalizeBase(import.meta.env?.VITE_API_BASE || (
  import.meta.env?.VITE_PROD_ENV ? "https://nfl-predict-ecf5a5bd34fe.herokuapp.com" : ""));
console.log('CLIENT.JS LINE 26 API_BASE: ', API_BASE);

// Internal fetch with JSON defaults and timeout
/**
 * @param {string} path
 * @param {Object} [options]
 * @param {number} [options.timeout]
 * @param {any} [options.body]
 * @param {Object} [options.headers]
 * @returns {Promise<any>}
 */
async function api(path = API_BASE, options = {}) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), options.timeout ?? DEFAULT_TIMEOUT_MS);
  const p = String(path || "");
  const url = `${API_BASE}${p.startsWith("") ? p : `/${p}`}`;

  const headers = {
    ...(options.body ? {"Content-Type": "application/json"} : {}),
    ...(options.headers || {}),
  };

  try {
    console.log('CLIENT.JS LINE 49 HEADERS: ', headers)
    // Use provided method (e.g. 'POST') or default to 'GET'.
    // Spread options first so callers can provide timeout/body/etc, then
    // explicitly set method and headers to the merged values.
    const res = await fetch(url, {...options, method: options.method || 'GET', headers, signal: ctrl.signal});
    const text = await res.text();
    let data;
    try {
      data = text ? JSON.parse(text) : res.body;
    } catch {
      data = {raw: text};
    }
    if (!res.ok) {
      const detail = data?.detail || res.statusText || "Request failed";
      throw new Error(`${res.status} ${detail}`);
    }
    return data;
  } finally {
    clearTimeout(id);
  }
}

// Public endpoints
export const getHealth = () => api("/health");
export const getDebug = () => api("/debug");
export const getTrainingReport = () => api("/report/training");
export const getCalibrationReport = () => api("/report/calibration");

export const getNextWeekSchedule = () => {
  console.log('CLIENT.JS: Fetching schedule from URL:', `${API_BASE}/schedule/next-week`);
  return api("/schedule/next-week");
};

/**
 * @param {any} payload
 * @returns {Promise<any>}
 */
export const predictGame = (payload) =>
  api("/predict", {
    method: "POST",
    body: JSON.stringify(toPredictionRequest(payload)),
  });

export const predictNextWeek = () => api("/predict/next-week");
export const retrain = () => api("/retrain", {method: "POST"});

/**
 * @param {any} game
 * @returns {Object}
 */
export function toPredictionRequest(game) {
  return {
    home_team: game.home_abbr || game.home_team,
    away_team: game.away_abbr || game.away_team,
    season: Number(game.season),
    week: Number(game.week),
  };
}
