/**
 * NFL-ML Client API wrapper
 *
 * Base URL strategy:
 *  - Use window.location.origin by default
 *  - Override with VITE_API_BASE or API_BASE if set
 */

const DEFAULT_TIMEOUT_MS = 15000;

export const API_BASE =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE) ||
  (typeof process !== "undefined" && process.env?.API_BASE) ||
  (typeof window !== "undefined" && window.location?.origin) ||
  "";

/** Internal fetch with JSON defaults and basic timeout */
async function api(path, options = {}) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), options.timeout ?? DEFAULT_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "GET",
      headers: {"Content-Type": "application/json", ...(options.headers || {})},
      ...options,
      signal: ctrl.signal,
    });

    const text = await res.text();
    let data;
    try {
      data = text ? JSON.parse(text) : null;
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
    body: JSON.stringify(payload),
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
