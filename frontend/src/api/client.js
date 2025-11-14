// /src/api/client.js 
/**
 * NFL-ML API Client (Hardened + Teachable v2)
 *
 * WHY THIS FILE EXISTS
 * ---------------------
 * Frontends fail in production for three common reasons:
 * 1) Environment/base-URL confusion (local vs hosted)
 * 2) Flaky networks/timeouts without retries
 * 3) Inconsistent payloads causing 400s
 *
 * This client addresses all three with:
 * - Safe base resolution (localhost uses Vite proxy; hosted uses VITE_API_BASE or a fallback)
 * - Abortable fetch with per-attempt timeout and exponential backoff + jitter
 * - 429/503 Retry-After handling
 * - Strict request normalization for /predict to avoid server 400s
 * - Small, optional in-memory cache for GETs (TTL) to reduce chatter
 * - Rich, typed-ish errors to power helpful UI messages
 *
 * QUICK START
 * -----------
 * import { apiClient, predictGame } from "./api/client";
 * await apiClient.getHealth();
 * await predictGame({ homeTeam: "SF", awayTeam: "SEA", season: 2025, week: 10 });
 *
 * POWER TIPS
 * ----------
 * - Override base per instance: createApi("https://your-backend")
 * - Per-call opts: api(path, init, { timeoutMs: 10000, retries: 1, cacheTtlMs: 5000 })
 */

/** @typedef {{home_team: string, away_team: string, season: number, week: number}} PredictionRequest */
/** @typedef {{status:number,message:string,payload?:any,url?:string,headers?:Headers,retryAfterMs?:number}} ApiErrorLike */

const DEFAULT_TIMEOUT_MS = 15000;
const RETRY_ATTEMPTS = 2;      // total tries = 1 + RETRY_ATTEMPTS
const RETRY_BASE_MS = 300;     // backoff base
const JITTER_FRAC = 0.25;      // up to +25% random jitter

// ---------- Tiny in-memory cache for idempotent GETs ----------
const _cache = new Map(); // key: url, value: { expiresAt:number, data:any }
function readCache(url) {
  const e = _cache.get(url);
  if (!e) return undefined;
  if (Date.now() > e.expiresAt) { _cache.delete(url); return undefined; }
  return e.data;
}
function writeCache(url, data, ttlMs) {
  if (!ttlMs || ttlMs <= 0) return;
  _cache.set(url, { expiresAt: Date.now() + ttlMs, data });
}

// ---------- URL helpers ----------
function normalizeBase(base) {
  if (!base) return "";
  let b = String(base).trim();
  if (b.includes(",")) b = b.split(",").map(s => s.trim()).find(Boolean) || ""; // pick first
  return b.replace(/\/+$/, "");
}
function joinUrl(base, path) {
  const b = normalizeBase(base);
  const p = String(path || "").trim().replace(/^\/+/, "");
  return b ? `${b}/${p}` : `/${p}`;
}

/** Validate absolute URLs to prevent accidental text like "VITE_API_BASE not set" from becoming a base. */
function isValidAbsoluteUrl(maybeUrl) {
  if (!maybeUrl || typeof maybeUrl !== "string") return false;
  const s = maybeUrl.trim();
  if (s.includes(" ")) return false; // no spaces allowed
  try {
    const u = new URL(s);
    return /^https?:$/.test(u.protocol);
  } catch { return false; }
}

// Base URL resolution with strong guards against invalid env strings
function resolveApiBase() {
  const herokuFallback = "https://nfl-predict-ecf5a5bd34fe.herokuapp.com"; // <- replace if needed
  const fromEnvRaw = import.meta?.env?.VITE_API_BASE || "";
  const fromEnv = normalizeBase(fromEnvRaw);
  const host = (typeof window !== "undefined" && window.location && window.location.hostname) || "";
  const isLocalHost = /^(localhost|127\.0\.0\.1)$/i.test(host);

  // DEV: rely on Vite proxy; use relative URLs only
  if (isLocalHost) return "";

  // PROD: only trust env if it looks like a full http(s) URL
  if (fromEnv && isValidAbsoluteUrl(fromEnv)) return fromEnv;

  // If env exists but invalid, log once and fall back
  if (fromEnv && typeof window !== "undefined" && !window.__NFL_API_BASE_INVALID_WARNED__) {
    try {
      console.error(`[NFL-ML] Ignoring invalid VITE_API_BASE=\"${fromEnvRaw}\". It must start with http(s) and contain no spaces.`);
      window.__NFL_API_BASE_INVALID_WARNED__ = true;
    } catch { }
  }

  // Final fallback (Heroku or your hosted backend)
  if (typeof window !== "undefined" && !window.__NFL_API_BASE_FALLBACK_WARNED__) {
    try {
      console.warn("[NFL-ML] Using Heroku API fallback. Set Vercel env VITE_API_BASE to your backend URL.");
      window.__NFL_API_BASE_FALLBACK_WARNED__ = true;
    } catch { }
  }
  return herokuFallback;
}
export const API_BASE = resolveApiBase();

// ---------- Error type ----------
export class ApiError extends Error {
  /** @param {number} status @param {string} message @param {any} payload @param {string} url @param {Headers} headers @param {number=} retryAfterMs */
  constructor(status, message, payload, url, headers, retryAfterMs) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
    this.url = url;
    this.headers = headers;
    this.retryAfterMs = retryAfterMs;
  }
}

// ---------- Retry/timeout fetch ----------
const delay = (ms) => new Promise(r => setTimeout(r, ms));
const withJitter = (ms) => Math.floor(ms * (1 + Math.random() * JITTER_FRAC));

/** Parse Retry-After header into ms. Supports delta-seconds or http-date. */
function parseRetryAfterMs(headers) {
  const ra = headers?.get?.("Retry-After");
  if (!ra) return undefined;
  const s = Number(ra);
  if (Number.isFinite(s)) return Math.max(0, s * 1000);
  const t = Date.parse(ra);
  if (Number.isFinite(t)) return Math.max(0, t - Date.now());
  return undefined;
}

/** Decide if an error is worth retrying */
function isRetriable(err) {
  if (!err) return false;
  if (err?.name === "AbortError") return true;
  if (err instanceof TypeError) return true; // network
  const s = /** @type {any} */(err).status;
  return typeof s === "number" && (s >= 500 || s === 429);
}

/**
 * Core API call with per-attempt AbortController, timeout, optional GET caching, and retries.
 * @param {string} path
 * @param {RequestInit & {cacheTtlMs?: number}} init
 * @param {{ timeoutMs?: number, retries?: number, cacheTtlMs?: number }} opts
 */
async function api(path, init = {}, { timeoutMs = DEFAULT_TIMEOUT_MS, retries = RETRY_ATTEMPTS, cacheTtlMs } = {}) {
  const isAbsolute = /^https?:\/\//i.test(String(path));
  const url = isAbsolute ? String(path) : joinUrl(API_BASE, path);

  // Simple GET cache (idempotent only)
  const method = (init.method || "GET").toUpperCase();
  const effectiveTtl = init.cacheTtlMs ?? cacheTtlMs;
  if (method === "GET" && effectiveTtl) {
    const cached = readCache(url);
    if (cached !== undefined) return cached;
  }

  let attempt = 0;
  while (true) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const hasBody = Object.prototype.hasOwnProperty.call(init, "body") && init.body != null;
      const headers = { ...(init?.headers || {}) };
      // Only set JSON Content-Type when body is a string (postJson uses stringified body)
      if (hasBody && typeof init.body === "string" && !("Content-Type" in headers)) headers["Content-Type"] = "application/json";
      if (!("Accept" in headers)) headers["Accept"] = "application/json, text/plain;q=0.9, */*;q=0.8";

      const res = await fetch(url, { credentials: "omit", ...init, headers, signal: controller.signal });
      const ctype = String(res.headers.get("Content-Type") || "");

      const parseJson = async () => { try { return await res.json(); } catch { return null; } };
      const parseText = async () => { try { return await res.text(); } catch { return null; } };

      if (!res.ok) {
        const payload = ctype.includes("application/json") ? await parseJson() : await parseText();
        const msg = (payload && (payload.detail || payload.message)) || res.statusText || "Request failed";
        const retryAfterMs = (res.status === 429 || res.status === 503) ? parseRetryAfterMs(res.headers) : undefined;
        throw new ApiError(res.status, msg, payload, url, res.headers, retryAfterMs);
      }

      const out = ctype.includes("application/json") ? await parseJson() : await parseText();
      if (method === "GET" && effectiveTtl) writeCache(url, out, effectiveTtl);
      return out;
    } catch (err) {
      const retriable = isRetriable(err);
      if (!retriable || attempt >= retries) throw err;
      const base = withJitter(RETRY_BASE_MS * Math.pow(2, attempt));
      const extra = (/** @type {any} */(err))?.retryAfterMs || 0;
      await delay(Math.max(base, extra));
      attempt += 1;
    } finally {
      clearTimeout(timer);
    }
  }
}

// Convenience wrappers
const get = (path, opts = {}) => api(path, { ...opts, method: "GET" }, opts);
const postJson = (path, body, opts = {}) => api(path, { ...opts, method: "POST", body: JSON.stringify(body) }, opts);

// ---------- Team coercion & validation ----------
const TEAM_ABBR = new Set([
  "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
  "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"
]);
const TEAM_NAME_TO_ABBR = {
  "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
  "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
  "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
  "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
  "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
  "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
  "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
  "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
  // legacy / relocations (best-effort)
  "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS"
};
function coerceTeam(value) {
  if (!value && value !== 0) throw new ApiError(400, "Missing team", null, "coerceTeam");
  const s = String(value).trim();
  if (TEAM_ABBR.has(s.toUpperCase())) return s.toUpperCase();
  if (TEAM_NAME_TO_ABBR[s]) return TEAM_NAME_TO_ABBR[s];
  const title = s.replace(/\s+/g, " ").replace(/\b\w/g, m => m.toUpperCase());
  if (TEAM_NAME_TO_ABBR[title]) return TEAM_NAME_TO_ABBR[title];
  throw new ApiError(400, `Unknown team: ${value}`, null, "coerceTeam");
}
function clampInt(n, lo, hi, label) {
  const x = Number(n);
  if (!Number.isFinite(x) || !Number.isInteger(x)) throw new ApiError(400, `${label} must be an integer`, null, "clampInt");
  if (x < lo || x > hi) throw new ApiError(400, `${label} out of range (${lo}-${hi})`, null, "clampInt");
  return x;
}

// ---------- Schema mapper ----------
/**
 * Normalize UI params → backend PredictionRequest.
 * Accepts either explicit fields or schedule objects with home/away_abbr.
 * Validates to prevent server-side 400s.
 * @param {{homeTeam?:string,awayTeam?:string,season?:number|string,week?:number|string,home_abbr?:string,away_abbr?:string,home_team?:string,away_team?:string}} p
 * @returns {PredictionRequest}
 */
export function toPredictionRequest({ homeTeam, awayTeam, season, week, home_abbr, away_abbr, home_team, away_team }) {
  const home = coerceTeam(home_abbr || home_team || homeTeam);
  const away = coerceTeam(away_abbr || away_team || awayTeam);
  const s = clampInt(season, 2000, 2100, "season");
  const w = clampInt(week, 1, 22, "week");
  return { home_team: home, away_team: away, season: s, week: w };
}

// ---------- Public API ----------
export function createApi(base = API_BASE) {
  const _get = (p, o) => api(joinUrl(base, p), { ...o, method: "GET" }, o);
  const _post = (p, b, o) => api(joinUrl(base, p), { ...o, method: "POST", body: JSON.stringify(b) }, o);

  return {
    /** Health endpoint. Uses a tiny cache (2s) to avoid spamming */
    getHealth: () => _get("/health", { cacheTtlMs: 2000 }),
    getHealthStatus: () => _get("/health", { cacheTtlMs: 2000 }),
    getTrainingReport: () => _get("/report/training"),
    getCalibrationReport: () => _get("/report/calibration"),

    /** Next-week endpoints are stable within a session → short cache helps */
    getNextWeekSchedule: () => _get("/schedule/next-week", { cacheTtlMs: 15000 }),
    predictNextWeek: () => _get("/predict/next-week"),

    /** Single-game prediction (validated) */
    predictGame: (params) => _post("/predict", toPredictionRequest(params)),

    /** History + status endpoints */
    getPredictionHistory: (limit = 50) => _get(`/history?limit=${limit}`, { cacheTtlMs: 5000 }),
    getStatusOverview: (limit = 5) => _get(`/status/overview?limit=${limit}`, { cacheTtlMs: 10000 }),

    /** Training control */
    startTraining: () => _post("/retrain", {}),
  };
}

// Default instance (uses resolved API_BASE)
export const apiClient = createApi();

// Named exports for direct import convenience
export const getNextWeekSchedule = apiClient.getNextWeekSchedule;
export const predictGame = apiClient.predictGame;
export const predictNextWeek = apiClient.predictNextWeek;
export const getHealth = apiClient.getHealth;
export const getHealthStatus = apiClient.getHealthStatus;
export const getTrainingReport = apiClient.getTrainingReport;
export const getCalibrationReport = apiClient.getCalibrationReport;
export const startTraining = apiClient.startTraining;
export const getPredictionHistory = apiClient.getPredictionHistory;
export const getStatusOverview = apiClient.getStatusOverview;

// ---------- Helper to present ApiError nicely in UI ----------
/** @param {unknown} err */
export function explainApiError(err) {
  if (!err) return "Unknown error";
  if (err instanceof ApiError) {
    const p = err.payload;
    const detail = (p && (p.detail || p.message)) || "";
    const at = err.url ? ` @ ${err.url}` : "";
    return `[${err.status}] ${err.message}${detail ? ` — ${detail}` : ""}${at}`;
  }
  if (/** @type {any} */(err)?.name === "AbortError") return "Request timed out";
  if (err instanceof TypeError) return "Network error";
  return String(/** @type {any} */(err)?.message || err);
}
