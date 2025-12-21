/**
* File Metrics:
* - Purpose: One tiny, consistent fetch wrapper for the whole app.
* - Why: Every endpoint gets the same error handling, JSON parsing, and abort support.
*
* Key Concepts:
* - AbortController cancels in-flight requests when components unmount.
* - "HttpError" carries status + body for better debugging.
*
* Learning Checkpoints:
* - You should be able to answer: "Where do I change my API base URL?"
* - You should be able to answer: "Where do errors get normalized?"
*/

export class HttpError extends Error {
  constructor(message, { status, url, body } = {}) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

// If you use Vite proxy, set BASE_URL = "" (empty string) and call "/api/..."
const RAW_BASE_URL =
  import.meta.env.VITE_API_BASE ??
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_URL ??
  "";
const BASE_URL = RAW_BASE_URL.replace(/\/+$/, ""); // "" works great with Vite proxy
export const API_BASE = BASE_URL;

async function safeReadJson(res) {
  try {
    return await res.json();
  } catch {
    return null; // sometimes backends return empty bodies
  }
}

/**
 * fetchJson(path, options)
 * - path: "/api/..." style
 * - options: { method, headers, body, signal }
 */
export async function fetchJson(path, options = {}) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${BASE_URL}${normalizedPath}`;

  // Custom option (not part of the native fetch API):
  // - timeoutMs: abort the request if it takes too long (default: 15s)
  const { timeoutMs: timeoutMsRaw, ...fetchOptions } = options;
  const timeoutMs = Number.isFinite(Number(timeoutMsRaw)) ? Number(timeoutMsRaw) : 15000;

  // If the caller did not pass a signal, create one so we can enforce timeouts.
  const controller = fetchOptions.signal ? null : new AbortController();
  const signal = fetchOptions.signal ?? controller.signal;

  // Auto-encode plain objects as JSON. This makes call sites cleaner:
  // fetchJson("/predict", { method:"POST", body:{...} })
  const rawBody = fetchOptions.body;
  const body =
    rawBody != null &&
    typeof rawBody === "object" &&
    !(rawBody instanceof FormData) &&
    !(rawBody instanceof URLSearchParams)
      ? JSON.stringify(rawBody)
      : rawBody;

  let timeoutId = null;
  if (controller && timeoutMs > 0) {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  }

  try {
    const res = await fetch(url, {
      method: "GET",
      ...fetchOptions,
      body,
      signal,
      headers: {
        "Content-Type": "application/json",
        ...(fetchOptions.headers || {}),
      },
    });

    // Try to parse body (even for errors) so your UI can show useful messages
    const resBody = await safeReadJson(res);

    if (!res.ok) {
      throw new HttpError(`Request failed (${res.status})`, {
        status: res.status,
        url,
        body: resBody,
      });
    }

    return resBody;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

export async function health() {
  return fetchJson("/health");
}


export async function getHealthStatus() {
  return health();
}

export async function getNextWeekSchedule() {
  // Prefer the wrapped response: { games: [...] }
  try {
    const api = await fetchJson("/api/games/next-week");
    if (api && Array.isArray(api.games)) return api.games;
  } catch (err) {
    // If this route doesn't exist on a given deployment, we'll fall back gracefully.
    if (!(err instanceof HttpError && (err.status === 404 || err.status === 405))) {
      console.warn("Next-week API route failed, falling back to /schedule/next-week");
    }
  }

  // Fallback: raw list response
  const res = await fetchJson("/schedule/next-week");
  if (Array.isArray(res)) return res;
  if (res && Array.isArray(res.games)) return res.games;
  if (res && Array.isArray(res.ScheduleGame)) return res.ScheduleGame;
  return [];
}

export async function predictGame(payload, options = {}) {
  // Accept either camelCase (React-friendly) or snake_case (API-native) keys.
  const home = String(payload?.homeTeam ?? payload?.home_team ?? "").trim().toUpperCase();
  const away = String(payload?.awayTeam ?? payload?.away_team ?? "").trim().toUpperCase();
  const season = Number(payload?.season);
  const week = Number(payload?.week);

  if (!home || !away || !Number.isFinite(season) || !Number.isFinite(week)) {
    throw new Error("predictGame requires { homeTeam, awayTeam, season, week } (or snake_case equivalents).");
  }

  // Note: fetchJson will JSON.stringify the body for us.
  return fetchJson("/predict", {
    method: "POST",
    body: { home_team: home, away_team: away, season, week },
    timeoutMs: 20000,
    ...options, // allow callers to pass { signal } for cancellation
  });
}

export async function startTraining() {
  const tryPost = async (path) => {
    try {
      return await fetchJson(path, { method: "POST" });
    } catch (err) {
      if (err instanceof HttpError) {
        if (err.status === 404 || err.status === 405) return null;
        const detail = err.body?.detail ?? err.body;
        const msg =
          typeof detail === "string"
            ? detail
            : `Training request failed (${err.status})`;
        throw new Error(msg);
      }
      throw err;
    }
  };

  const res = (await tryPost("/retrain")) ?? (await tryPost("/train"));
  if (res == null) throw new Error("Training endpoint not available");
  return res;
}

// Missing endpoints referenced by StatsPage.jsx
export async function getPredictionHistory(limit = 100) {
  try {
    const safeLimit = Number.isFinite(Number(limit)) ? Number(limit) : 100;

    // Prefer stable, wrapped response
    try {
      const api = await fetchJson(`/api/history?limit=${safeLimit}`);
      if (api && Array.isArray(api.entries)) {
        return {
          entries: api.entries,
          total: Number.isFinite(Number(api.total)) ? Number(api.total) : api.entries.length,
        };
      }
    } catch (err) {
      // Fall back to legacy route below.
      if (!(err instanceof HttpError && (err.status === 404 || err.status === 405))) {
        console.warn("History API route failed, falling back to /history");
      }
    }

    // Legacy: backend returns a raw list
    const res = await fetchJson(`/history?limit=${safeLimit}`);
    if (Array.isArray(res)) return { entries: res, total: res.length };
    if (res && Array.isArray(res.entries)) {
      return {
        entries: res.entries,
        total: Number.isFinite(Number(res.total)) ? Number(res.total) : res.entries.length,
      };
    }
    return { entries: [], total: 0 };
  } catch (err) {
    console.warn("History endpoint unavailable, falling back to empty");
    return { entries: [], total: 0 };
  }
}

export async function getStatusOverview() {
  try {
    const res = await fetchJson("/status/overview");

    if (res && typeof res === "object") {
      const dataset = res.dataset ?? { rows: 0 };
      return {
        ...res,
        health: res.health ?? { status: "unknown" },
        dataset,
        history:
          res.history ??
          ({
            metrics: { total_predictions: Number(dataset?.rows ?? 0) },
          }),
      };
    }

    return {
      health: { status: "unknown" },
      dataset: { rows: 0 },
      history: { metrics: { total_predictions: 0 } },
    };
  } catch (err) {
    console.warn("Status overview unavailable");
    return {
      health: { status: "unknown" },
      dataset: { rows: 0 },
      history: { metrics: { total_predictions: 0 } }
    };
  }
}
