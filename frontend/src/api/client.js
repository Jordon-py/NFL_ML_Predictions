/**
 * NFL-ML Client API wrapper
 *
 * Purpose: Provides a unified interface for interacting with the NFL prediction backend API.
 * Key Logic Flow: Defines base URL, internal fetch helper with timeout and JSON handling, and exports public endpoints for health checks, reports, predictions, and maintenance.
 * Dependencies: Relies on environment variables (VITE_API_BASE or API_BASE) or window.location.origin for API base URL.
 */

const DEFAULT_TIMEOUT_MS = 15000;

// Normalize API base:
// - In dev: use absolute env base if provided (http/https). If not, use Vite proxy ("").
// - In prod: require absolute base, else fallback to Heroku URL.
const isProd = import.meta.env.PROD === true;

// Read raw base from env (supports comma-separated list); may be empty in dev
const rawEnvBase = (
  import.meta.env.VITE_API_BASE ??
  import.meta.env.API_BASE ??
  import.meta.env.VITE_NODE_DEV ??
  ""
).trim();

/** Pick the first absolute http(s) base from a possibly comma-separated env value. */
function pickAbsoluteBase(s) {
  if (!s) return "";
  const parts = s.split(",").map((p) => p.trim()).filter(Boolean);
  for (const p of parts) {
    const abs = normalizeBase(p);
    if (abs) return abs;
  }
  return "";
}

const envBase = pickAbsoluteBase(rawEnvBase);

function normalizeBase(base) {
  if (!base) return "";
  const abs = /^https?:\/\//i.test(base);
  return abs ? base.replace(/\/+$/, "") : "";
}

export const API_BASE = isProd
  ? normalizeBase(envBase) || "https://nfl-predict-ecf5a5bd34fe.herokuapp.com"
  : normalizeBase(envBase) || "http://localhost:3000"; // empty => Vite proxy

/** Internal fetch with JSON defaults and basic timeout */
async function api(path, options = {}) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), options.timeout ?? DEFAULT_TIMEOUT_MS);

  // Join base + path safely
  const p = String(path || "");
  const url = `${API_BASE}${p.startsWith("/") ? p : `/${p}`}`;

  try {
    if (!isProd && !API_BASE) {
      // Using Vite proxy in dev
      console.debug(`[api] dev proxy -> ${url}`);
    }

    const res = await fetch(url, {
      method: "GET",
      headers: {"Content-Type": "application/json", ...(options.headers || {})},
      ...options,
      signal: ctrl.signal,
    });

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

// Schedule + Predict
export const getNextWeekSchedule = () => api("/schedule/next-week");

export const predictGame = (payload) =>
  api("/predict", {
    method: "POST",
    body: JSON.stringify(toPredictionRequest(payload)),
  });

export const predictNextWeek = () => api("/predict/next-week");

// Maintenance
export const retrain = () => api("/retrain", {method: "POST"});

/** Helper: shape payload from schedule row */
export function toPredictionRequest(game) {
  return {
    home_team: game.home_abbr || game.home_team,
    away_team: game.away_abbr || game.away_team,
    season: Number(game.season),
    week: Number(game.week),
  };
}
