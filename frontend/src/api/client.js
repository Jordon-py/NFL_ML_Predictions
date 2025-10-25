// /frontend/src/api/client.js
// @ts-nocheck

/**
 * NFL-ML Client API wrapper (hardened)
 * - Correct PROD detection
 * - Safe base handling (no double slashes)
 * - Timeout + retries for flaky networks
 * - JSON helpers and typed errors
 */

const DEFAULT_TIMEOUT_MS = 15000;
const RETRY_ATTEMPTS = 2;   // total tries = 1 + RETRY_ATTEMPTS
const RETRY_BASE_MS = 300;  // backoff base

function normalizeBase(base) {
  if (!base) return "";
  let b = String(base).trim();
  // allow comma-joined .env mistakes; keep first non-empty
  if (b.includes(",")) b = b.split(",").map(s => s.trim()).find(Boolean) || "";
  // remove trailing slashes
  return b.replace(/\/+$/, "");
}

function joinUrl(base, path) {
  const b = normalizeBase(base);
  const p = String(path || "").trim().replace(/^\/+/, "");
  return b ? `${b}/${p}` : `/${p}`;
}

// DEV uses Vite proxy ("" base) → server.proxy
export const API_BASE = import.meta.env?.DEV
  ? ""
  : normalizeBase(import.meta.env?.VITE_API_BASE || "https://nfl-predict-ecf5a5bd34fe.herokuapp.com");

/** Lightweight typed API error */
export class ApiError extends Error {
  constructor(message, { status, url, details } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
    this.details = details;
  }
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

/**
 * Internal fetch with JSON defaults, timeout, and retries.
 * @param {string} path
 * @param {Object} [options]
 * @param {number} [options.timeout]
 * @param {number} [options.retries]
 * @returns {Promise<any>}
 */
async function api(path, options = {}) {
  const {
    method = "GET",
    headers = {},
    body,
    timeout = DEFAULT_TIMEOUT_MS,
    retries = RETRY_ATTEMPTS,
  } = options;

  const url = joinUrl(API_BASE, path);
  const requestId = Math.random().toString(36).slice(2, 10);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  const fetchOnce = async () => {
    const init = {
      method,
      headers,
      body,
      signal: controller.signal,
      credentials: "omit", // play nice with CORS credentials
    };
    // Dev-time trace
    if (import.meta.env?.DEV) {
      console.debug(`[api:${requestId}] ${method} ${url}`, init);
    }
    const res = await fetch(url, init);
    const ctype = res.headers.get("content-type") || "";

    if (!res.ok) {
      let details;
      if (ctype.includes("application/json")) {
        try { details = await res.json(); } catch {}
      } else {
        try { details = await res.text(); } catch {}
      }
      throw new ApiError(`Request failed (${res.status})`, { status: res.status, url, details });
    }

    if (ctype.includes("application/json")) {
      return res.json();
    }
    // allow CSV/text passthrough for reports when needed
    return res.text();
  };

  try {
    let attempt = 0, lastErr;
    while (attempt <= retries) {
      try {
        const out = await fetchOnce();
        clearTimeout(timer);
        return out;
      } catch (err) {
        lastErr = err;
        // only retry for network/timeout/5xx
        const retriable =
          (err.name === "AbortError") ||
          (err instanceof TypeError) ||
          (err.status >= 500);
        if (!retriable || attempt === retries) break;
        await delay(RETRY_BASE_MS * Math.pow(2, attempt)); // simple backoff
        attempt += 1;
      }
    }
    throw lastErr;
  } finally {
    clearTimeout(timer);
  }
}

/** JSON helpers */
function get(path, opts = {}) {
  return api(path, { ...opts, method: "GET" });
}
function postJson(path, payload, opts = {}) {
  return api(path, {
    ...opts,
    method: "POST",
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    body: JSON.stringify(payload ?? {}),
  });
}

/** Shape normalizer for /predict input */
export function toPredictionRequest(game) {
  return {
    home_team: game.home_abbr || game.home_team,
    away_team: game.away_abbr || game.away_team,
    season: Number(game.season),
    week: Number(game.week),
  };
}

/** Public endpoints */
export const getNextWeekSchedule = () => get("/schedule/next-week");
export const predictGame = (game) => postJson("/predict", toPredictionRequest(game));
export const predictNextWeek = () => get("/predict/next-week");
export const getTrainingReport = () => get("/report/training");
export const getCalibrationReport = () => get("/report/calibration");
export const health = () => get("/health");
export const retrain = () => postJson("/retrain", {});
