// /src/api/client.js
/**
 * NFL-ML API Client (Robust)
 * - Safe base resolution (local dev vs hosted, env-aware)
 * - Typed errors, retries, and timeouts
 * - Clean schema mapping for /predict
 * - Batch-friendly helpers for weekly flows
 */

const DEFAULT_TIMEOUT_MS = 15000;
const RETRY_ATTEMPTS = 2;    // total tries = 1 + RETRY_ATTEMPTS
const RETRY_BASE_MS = 300;   // backoff base
const HEROKU_FALLBACK = "https://nfl-predict-ecf5a5bd34fe.herokuapp.com";

// ---------- URL helpers ----------

function normalizeBase(base) {
  if (!base) return "";
  let b = String(base).trim();
  if (b.includes(",")) b = b.split(",").map(s => s.trim()).find(Boolean) || "";     // allow comma-joined mistakes; keep first non-empty
  return b.replace(/\/+$/, "");       // remove trailing slashes
}

function joinUrl(base, path) {
  const b = normalizeBase(base);
  const p = String(path || "").trim().replace(/^\/+/, "");
  return b ? `${b}/${p}` : `/${p}`;
}

// Base URL resolution:
// - Local dev (localhost/127.*): use relative URLs (Vite proxy handles forwarding)
// - Hosted (Vercel/Netlify/etc.): prefer VITE_API_BASE/VITE_API_URL; else fallback to known Heroku URL.
function resolveApiBase() {
  const fromEnv = normalizeBase(import.meta?.env?.VITE_API_BASE || import.meta?.env?.VITE_API_URL);
  const host = (typeof window !== "undefined" && window.location && window.location.hostname) || "";
  const isLocalHost = /^(localhost|127\.0\.0\.1)$/i.test(host);
  const base = isLocalHost ? "http://127.0.0.1:8000" : (fromEnv || HEROKU_FALLBACK);
  // One-time diagnostic: if hosted and no explicit VITE_API_BASE provided, warn about fallback
  if (!isLocalHost && !fromEnv && typeof window !== "undefined" && !window.__NFL_API_BASE_WARNED__) {
    try {
      // eslint-disable-next-line no-console
      console.warn("[NFL-ML] Using Heroku API. Set VITE_API_BASE or VITE_API_URL to your backend URL to remove this warning.");
      window.__NFL_API_BASE_WARNED__ = true;
    } catch (_) { /* noop */ }
  }
  return base;
}

export const API_BASE = resolveApiBase();

// ---------- Error type ----------

export class ApiError extends Error {
  constructor(status, message, payload, url) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
    this.url = url;
  }
}

export function normalizePredictError(err) {
  if (err instanceof ApiError) {
    if (err.status === 503 && /raw_feature_columns/i.test(String(err.message))) {
      return "Models need retraining: backend metadata is missing feature columns."
    }
    if (err.status === 422) {
      return "Request validation failed. Double-check team codes, season, and week.";
    }
  }
  return err?.message || "Request failed";
}

// ---------- Core fetch with timeout + retry ----------

/**
 * CHANGED: Added the missing `api()` function that `get()` and `postJson()` reference.
 *
 * Why this pattern?
 * - Centralizes all HTTP logic: timeout, retry, error normalization.
 * - Uses AbortController for request timeouts (browser-native, no dependencies).
 * - Implements exponential backoff for transient failures (network hiccups).
 * - Returns parsed JSON directly, or throws ApiError with status code.
 *
 * @param {string} url - Full URL to fetch
 * @param {Object} opts - Fetch options (method, body, headers, etc.)
 * @returns {Promise<any>} Parsed JSON response
 * @throws {ApiError} On HTTP errors or timeout
 */
async function api(url, opts = {}) {
  const { timeout = DEFAULT_TIMEOUT_MS, retries = RETRY_ATTEMPTS, ...fetchOpts } = opts;

  // Default headers for JSON APIs
  const headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    ...fetchOpts.headers,
  };

  let lastError;

  // Retry loop with exponential backoff
  for (let attempt = 0; attempt <= retries; attempt++) {
    // AbortController provides a way to cancel fetch after timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...fetchOpts,
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Parse response body (even for errors, backend may send JSON details)
      let data;
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        data = await response.json();
      } else {
        data = await response.text();
      }

      // HTTP 2xx = success; otherwise throw structured error
      if (response.ok) {
        return data;
      }

      // Non-2xx: wrap in ApiError for downstream handling
      const errorMessage = typeof data === "object" ? (data.detail || data.message || JSON.stringify(data)) : data;
      throw new ApiError(response.status, errorMessage, data, url);

    } catch (err) {
      clearTimeout(timeoutId);

      // Handle AbortController timeout
      if (err.name === "AbortError") {
        lastError = new ApiError(408, `Request timeout after ${timeout}ms`, null, url);
      } else if (err instanceof ApiError) {
        lastError = err;
        // Don't retry client errors (4xx) — only retry server/network errors
        if (err.status >= 400 && err.status < 500) {
          throw err;
        }
      } else {
        // Network error or other fetch failure
        lastError = new ApiError(0, err.message || "Network error", null, url);
      }

      // Exponential backoff before retry: 300ms, 600ms, 1200ms...
      if (attempt < retries) {
        const delay = RETRY_BASE_MS * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  // All retries exhausted
  throw lastError;
}

// Convenience wrappers
const get = (path, opts = {}) => api(path, { ...opts, method: "GET" });
const postJson = (path, body, opts = {}) => api(path, { ...opts, method: "POST", body: JSON.stringify(body) });

// ---------- Schema mappers ----------

/**
 * Normalize UI params → backend PredictionRequest.
 * Accepts either explicit fields or schedule objects with home/away_abbr.
 */
function PredictionRequest({ stadium, homeTeam, awayTeam, season, week, home_abbr, away_abbr, home_team, away_team }) {
  return {
    stadium: String(stadium),
    home_team: String(home_abbr || home_team || homeTeam),
    away_team: String(away_abbr || away_team || awayTeam),
    season: Number(season),
    week: Number(week),
  };
}

// ---------- Public API ----------

/**
 * Creates an API client instance bound to a specific backend URL.
 *
 * Educational: This factory pattern allows:
 * - Testing with mock servers by passing a different base URL
 * - Supporting multiple environments (dev, staging, prod) from the same codebase
 * - Easy switching between local and deployed backends
 *
 * @param {string} base - Backend URL (defaults to auto-resolved API_BASE)
 * @returns {Object} API client with methods for each endpoint
 */
export function createApi(base = API_BASE) {
  // Internal helpers that prefix all paths with the base URL
  const _get = (p, o) => api(joinUrl(base, p), { ...o, method: "GET" });
  const _post = (p, b, o) => api(joinUrl(base, p), { ...o, method: "POST", body: JSON.stringify(b) });

  return {
    // ──────────────────────────────────────────────────────────────
    // Health & Reports
    // ──────────────────────────────────────────────────────────────

    /** Check if the backend is healthy and models are loaded */
    getHealth: async () => await _get("/health"),

    /** Alias for getHealth — used by hooks that poll training/model readiness */
    getHealthStatus: async () => await _get("/health"),

    /** Fetch the full training report (metrics, hyperparameters, etc.) */
    getTrainingReport: async () => await _get("/report/training"),

    /** Fetch calibration metrics for the win probability model */
    getCalibrationReport: async () => await _get("/report/calibration"),

    // ──────────────────────────────────────────────────────────────
    // Schedule & Batch Predictions
    // ──────────────────────────────────────────────────────────────

    /** Get list of games for the upcoming NFL week */
    getNextWeekSchedule: async () => await _get("/schedule/next-week"),

    /** Batch predict all games in the next week */
    predictNextWeek: () => _get("/predict/next-week"),

    // ──────────────────────────────────────────────────────────────
    // Prediction History (NOTE: Backend endpoint may not exist)
    // ──────────────────────────────────────────────────────────────

    /**
     * CHANGED: Added note that /history endpoint may not exist in current backend.
     * This method is kept for forward compatibility when the endpoint is added.
     */
    getPredictionHistory: (limit = 100) => _get(`/history?limit=${Number(limit) || 100}`),

    // ──────────────────────────────────────────────────────────────
    // Model Training (best-effort; backend may not support)
    // ──────────────────────────────────────────────────────────────

    /** Trigger model training — backend may treat as no-op if unsupported */
    startTraining: () => _post("/train", {}),

    // ──────────────────────────────────────────────────────────────
    // Status Overview
    // ──────────────────────────────────────────────────────────────

    /**
     * Get comprehensive status overview.
     * CHANGED: Has graceful fallback to /health if /status/overview doesn't exist.
     */
    getStatusOverview: async () => {
      try {
        return await _get("/status/overview");
      } catch (err) {
        // Graceful fallback: reuse /health payload so UI can still render
        const health = await _get("/health").catch(() => null);
        return { health, dataset: null, history: { metrics: {} } };
      }
    },

    // ──────────────────────────────────────────────────────────────
    // Single Game Prediction
    // ──────────────────────────────────────────────────────────────

    // Single-game prediction
    // CHANGED: Fixed broken code — `JSON.response.body` was invalid JavaScript syntax.
    // The `response` variable already contains the parsed JSON from `api()`.
    predictGame: async (params) => {
      try {
        // _post returns parsed JSON directly (api() handles parsing)
        const response = await _post("/predict", PredictionRequest(params));
        // Educational: No need to parse again — `response` is already the prediction object
        // with fields: home_score, away_score, home_win_probability, etc.
        return response;
      } catch (err) {
        // normalizePredictError converts ApiError to user-friendly messages
        throw new Error(normalizePredictError(err));
      }
    },
  }
}

// For convenience default instance (uses resolved API_BASE)
export const apiClient = createApi();

// Named exports for direct import
export const getNextWeekSchedule = apiClient.getNextWeekSchedule;
export const predictGame = apiClient.predictGame;

export const predictNextWeek = apiClient.predictNextWeek;
export const getHealth = apiClient.getHealth;
export const getHealthStatus = apiClient.getHealthStatus;
export const getPredictionHistory = apiClient.getPredictionHistory;
export const startTraining = apiClient.startTraining;
export const getStatusOverview = apiClient.getStatusOverview;
export const getTrainingReport = apiClient.getTrainingReport;
export const getCalibrationReport = apiClient.getCalibrationReport;
