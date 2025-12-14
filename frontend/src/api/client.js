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

  const res = await fetch(url, {
    method: "GET",
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  // Try to parse body (even for errors) so your UI can show useful messages
  const body = await safeReadJson(res);

  if (!res.ok) {
    throw new HttpError(`Request failed (${res.status})`, {
      status: res.status,
      url,
      body,
    });
  }

  return body;
}

export async function health() {
  return fetchJson("/health");
}

export async function getHealthStatus() {
  return health();
}

export async function getNextWeekSchedule() {
  const res = await fetchJson("/schedule/next-week");
  if (Array.isArray(res)) return res;
  if (res && Array.isArray(res.games)) return res.games;
  if (res && Array.isArray(res.ScheduleGame)) return res.ScheduleGame;
  return [];
}

export async function predictGame(payload) {
  const body = {
    home_team: String(payload?.home_team ?? payload?.homeTeam ?? "").trim().toUpperCase(),
    away_team: String(payload?.away_team ?? payload?.awayTeam ?? "").trim().toUpperCase(),
    season: Number(payload?.season ?? payload?.season_num ?? payload?.seasonNum),
    week: Number(payload?.week ?? payload?.week_num ?? payload?.weekNum),
  };

  if (!body.home_team || !body.away_team || !Number.isFinite(body.season) || !Number.isFinite(body.week)) {
    throw new Error("predictGame requires {home_team, away_team, season, week}");
  }

  return fetchJson("/predict", {
    method: "POST",
    body: JSON.stringify(body),
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
    const res = await fetchJson(`/history?limit=${safeLimit}`);

    // Backend currently returns a raw list; normalize for dashboard callers.
    if (Array.isArray(res)) {
      return { entries: res, total: res.length };
    }

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
