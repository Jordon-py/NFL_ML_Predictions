/**
 * NFL Prediction App — Core API Client (Expert v1.2)
 * ================================================
 * 
 * A robust, high-performance fetch wrapper engineered for the NFL ML Predictions ecosystem.
 * Features:
 *  - Unified error handling with custom HttpError class.
 *  - Request timeout and cancellation via AbortController.
 *  - Environment-aware URL resolution with trailing-slash normalization.
 *  - Defensive JSON parsing resilient to empty or malformed responses.
 *  - Data normalization layers to bridge backend-frontend schema drift.
 */

/**
 * Custom Error class for API-originated failures.
 */
export class HttpError extends Error {
  constructor(message, { status, url, body } = {}) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.url = url;
    this.body = body;

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, HttpError);
    }
  }

  toJSON() {
    return { name: this.name, message: this.message, status: this.status, url: this.url, body: this.body };
  }
}

// ---------------------------------------------------------
// Configuration & Utilities
// ---------------------------------------------------------

const RAW_BASE_URL = import.meta.env?.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
export const API_BASE = RAW_BASE_URL.replace(/\/+$/, "");

const DEFAULT_TIMEOUT = 15000;

/**
 * Resilient JSON reader that handles empty responses and parse errors gracefully.
 */
async function safeReadJson(res) {
  try {
    const text = await res.text();
    if (!text || text.trim().length === 0) return null;
    return JSON.parse(text);
  } catch (err) {
    console.warn(`[API Client] JSON parse failure: ${err.message}`);
    return null;
  }
}

/**
 * The core fetch engine with timeout, headers, and error normalization.
 */
export async function fetchJson(path, options = {}) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${API_BASE}${normalizedPath}`;

  const controller = new AbortController();
  const timeoutMs = options.timeout ?? DEFAULT_TIMEOUT;
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      method: "GET", // default
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      signal: controller.signal,
    });

    const body = await safeReadJson(res);

    if (!res.ok) {
      const detail = body?.detail || body?.message || res.statusText;
      const errorMessage = `API Error: ${res.status} (${detail})`;

      throw new HttpError(errorMessage, {
        status: res.status,
        url,
        body,
      });
    }

    return body;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new HttpError(`Request timed out after ${timeoutMs}ms`, { url, status: 408 });
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

// ---------------------------------------------------------
// Domain Endpoints
// ---------------------------------------------------------

/**
 * Retrieve system health and model readiness.
 */
export async function getHealthStatus() {
  return fetchJson("/health");
}

/**
 * Legacy alias for getHealthStatus.
 */
export const health = getHealthStatus;

/**
 * Fetch the upcoming week's matchup schedule.
 * Normalizes between {games: []} and direct array responses.
 */
export async function getNextWeekSchedule() {
  const data = await fetchJson("/schedule/next-week");
  if (Array.isArray(data)) return data;
  return data?.games ?? data?.ScheduleGame ?? [];
}

/**
 * Submit a game for ML inference.
 * Re-maps camelCase keys to backend snake_case.
 */
export async function predictGame(payload) {
  const home = payload?.homeTeam || payload?.home_team;
  const away = payload?.awayTeam || payload?.away_team;

  if (!home || !away) {
    throw new Error("[API Client] predictGame requires home_team and away_team");
  }

  const body = {
    home_team: String(home).trim().toUpperCase(),
    away_team: String(away).trim().toUpperCase(),
    season: Number(payload.season || new Date().getFullYear()),
    week: Number(payload.week || 1),
  };

  return fetchJson("/predict", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/**
 * Fetch historical prediction accuracy and logs.
 */
export async function getPredictionHistory(limit = 100) {
  const data = await fetchJson(`/history?limit=${limit}`);
  // Return normalized { entries, total } shape
  if (Array.isArray(data)) return { entries: data, total: data.length };
  return {
    entries: data?.entries ?? [],
    total: data?.total ?? data?.entries?.length ?? 0
  };
}

/**
 * Get a high-level overview of system status, metrics, and data scales.
 */
export async function getStatusOverview() {
  const data = await fetchJson("/status/overview");
  return {
    health: data?.health ?? { status: "unknown" },
    dataset: data?.dataset ?? { rows: 0 },
    history: data?.history ?? { metrics: { total_predictions: 0, win_rate: 0 } }
  };
}

/**
 * Utility to generate an AbortController for component cleanup.
 */
export function createAbortController() {
  return new AbortController();
}

/** Default Export Object */
export default {
  fetchJson,
  getHealthStatus,
  health,
  getNextWeekSchedule,
  predictGame,
  getPredictionHistory,
  getStatusOverview,
  createAbortController,
  HttpError,
  API_BASE,
};