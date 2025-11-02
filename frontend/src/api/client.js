// /src/api/client.js
/**
 * NFL-ML API Client (Robust)
 * - Safe base resolution (local dev vs hosted)
 * - Typed errors, retries, and timeouts
 * - Clean schema mapping for /predict
 * - Batch-friendly helpers for weekly flows
 */

const DEFAULT_TIMEOUT_MS = 15000;
const RETRY_ATTEMPTS = 2;    // total tries = 1 + RETRY_ATTEMPTS
const RETRY_BASE_MS = 300;   // backoff base

// ---------- URL helpers ----------

function normalizeBase(base) {
  if (!base) return "";
  let b = String(base).trim();
  // allow comma-joined mistakes; keep first non-empty
  if (b.includes(",")) b = b.split(",").map(s => s.trim()).find(Boolean) || "";
  // remove trailing slashes
  return b.replace(/\/+$/, "");
}

function joinUrl(base, path) {
  const b = normalizeBase(base);
  const p = String(path || "").trim().replace(/^\/+/, "");
  return b ? `${b}/${p}` : `/${p}`;
}

// Base URL resolution:
// - Local dev (localhost/127.*): use relative URLs (Vite proxy handles forwarding)
// - Hosted (Vercel/Netlify/etc.): prefer VITE_API_BASE; else fallback to known Heroku URL if you have one.
function resolveApiBase() {
  const herokuFallback = "https://nfl-predict-ecf5a5bd34fe.herokuapp.com"; // <- replace if needed
  const fromEnv = normalizeBase(import.meta?.env?.VITE_API_BASE);
  const host = (typeof window !== "undefined" && window.location && window.location.hostname) || "";
  const isLocalHost = /^(localhost|127\.0\.0\.1)$/i.test(host);
  const base = isLocalHost ? "" : (fromEnv || herokuFallback);
  // One-time diagnostic: if hosted and no explicit VITE_API_BASE provided, warn about fallback
  if (!isLocalHost && !fromEnv && typeof window !== "undefined" && !window.__NFL_API_BASE_WARNED__) {
    try {
      // eslint-disable-next-line no-console
      console.warn("[NFL-ML] Using Heroku API fallback. Set Vercel env VITE_API_BASE to your backend URL to remove this warning.");
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

// ---------- Core fetch with timeout + retry ----------

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

async function api(path, init = {}, { timeoutMs = DEFAULT_TIMEOUT_MS, retries = RETRY_ATTEMPTS } = {}) {
  // If caller already provided an absolute URL, use it as-is to avoid double-prefixing API_BASE
  const url = /^https?:\/\//i.test(String(path))
    ? String(path)
    : joinUrl(API_BASE, path);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  const doFetch = async () => {
    const res = await fetch(url, {
      // Do not send cookies; backend is stateless and allows "*" in dev
      credentials: "omit",
      ...init,
      headers: { "Content-Type": "application/json", ...(init.headers || {}) },
      signal: controller.signal,
    });

    const ctype = String(res.headers.get("Content-Type") || "");
    const parseJson = async () => { try { return await res.json(); } catch { return null; } };
    const parseText = async () => { try { return await res.text(); } catch { return null; } };

    if (!res.ok) {
      const payload = ctype.includes("application/json") ? await parseJson() : await parseText();
      const msg = (payload && (payload.detail || payload.message)) || res.statusText || "Request failed";
      throw new ApiError(res.status, msg, payload, url);
    }
    return ctype.includes("application/json") ? await parseJson() : await parseText();
  };

  try {
    let attempt = 0;
    while (true) {
      try {
        const out = await doFetch();
        return out;
      } catch (err) {
        const retriable =
          err?.name === "AbortError" ||
          err instanceof TypeError ||          // network
          (typeof err.status === "number" && err.status >= 500);
        if (!retriable || attempt >= retries) throw err;
        await delay(RETRY_BASE_MS * Math.pow(2, attempt));
        attempt += 1;
      }
    }
  } finally {
    clearTimeout(timer);
  }
}

const get = (path, opts = {}) => api(path, { ...opts, method: "GET" });
const postJson = (path, body, opts = {}) => api(path, { ...opts, method: "POST", body: JSON.stringify(body) });

// ---------- Schema mappers ----------

/**
 * Normalize UI params → backend PredictionRequest.
 * Accepts either explicit fields or schedule objects with home/away_abbr.
 */
function toPredictionRequest({ homeTeam, awayTeam, season, week, home_abbr, away_abbr, home_team, away_team }) {
  return {
    home_team: (home_abbr || home_team || homeTeam),
    away_team: (away_abbr || away_team || awayTeam),
    season: Number(season),
    week: Number(week),
  };
}

// ---------- Public API ----------

export function createApi(base = API_BASE) {
  // You can pass an explicit base to talk to a different instance
  const _get = (p, o) => api(joinUrl(base, p), { ...o, method: "GET" });
  const _post = (p, b, o) => api(joinUrl(base, p), { ...o, method: "POST", body: JSON.stringify(b) });

  return {
    // Health & reports
    getHealth: () => _get("/health"),
    // Alias for hooks/useTrainingStatus.js compatibility
    getHealthStatus: () => _get("/health"),
    getTrainingReport: () => _get("/report/training"),
    getCalibrationReport: () => _get("/report/calibration"),

    // Schedule & batch predictions
    getNextWeekSchedule: () => _get("/schedule/next-week"),
    predictNextWeek: () => _get("/predict/next-week"),

    // Single-game prediction
    predictGame: (params) => _post("/predict", toPredictionRequest(params)),

    // Training control (backend provides lightweight /retrain endpoint)
    startTraining: () => _post("/retrain", {}),
  };
}

// For convenience default instance (uses resolved API_BASE)
export const apiClient = createApi();

// Named exports for direct import
export const getNextWeekSchedule = apiClient.getNextWeekSchedule;
export const predictGame = apiClient.predictGame;
export const predictNextWeek = apiClient.predictNextWeek;
export const getHealth = apiClient.getHealth;
export const getHealthStatus = apiClient.getHealthStatus;
export const getTrainingReport = apiClient.getTrainingReport;
export const getCalibrationReport = apiClient.getCalibrationReport;
export const startTraining = apiClient.startTraining;
