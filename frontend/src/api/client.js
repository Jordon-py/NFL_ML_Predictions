<<<<<<< HEAD
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
=======
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
  import.meta.env?.VITE_PROD_ENV ? "https://nfl-predict-ecf5a5bd34fe.herokuapp.com/" : ""));
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
  console.log('CLIENT.JS: Fetching schedule from URL:', `${API_BASE}schedule/next-week`);
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
>>>>>>> c6845983cfbfd1be9afb17b5b47b7331808ca550
export function toPredictionRequest(game) {
  return {
    home_team: game.home_abbr || game.home_team,
    away_team: game.away_abbr || game.away_team,
    season: Number(game.season),
    week: Number(game.week),
  };
}
<<<<<<< HEAD

/** Public endpoints */
export const getNextWeekSchedule = () => get("/schedule/next-week");
export const predictGame = (game) => postJson("/predict", toPredictionRequest(game));
export const predictNextWeek = () => get("/predict/next-week");
export const getTrainingReport = () => get("/report/training");
export const getCalibrationReport = () => get("/report/calibration");
export const health = () => get("/health");
export const retrain = () => postJson("/retrain", {});
=======
>>>>>>> c6845983cfbfd1be9afb17b5b47b7331808ca550
