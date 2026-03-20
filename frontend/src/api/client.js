/**
 * File: frontend/src/api/client.js
 *
 * Purpose:
 *   One tiny, consistent fetch wrapper for the whole app (schedule + predict + history).
 *
 * The production gotcha (Vercel):
 *   Vite does NOT ship your local `.env` file to Vercel. You must set env vars
 *   in Vercel Project Settings → Environment Variables.
 *
 * Required env:
 *   - Local dev:    VITE_API_BASE_URL=http://127.0.0.1:8000
 *   - Vercel prod:  VITE_API_BASE_URL=https://<your-heroku-app>.herokuapp.com
 *
 * Notes:
 *   - Trailing slashes are stripped so URL joins are predictable.
 *   - Errors are thrown as HttpError(status, url, body) so UI can display useful info.
 */

import { buildPredictPayload as buildGamePredictPayload } from "../utils/gameUtils.js";

export class HttpError extends Error {
  constructor(message, { status, url, body } = {}) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

/**
 * Resolve the API base URL.
 *
 * Why this matters:
 *   - Locally you can hit FastAPI directly.
 *   - On Vercel you must call your Heroku domain (or you'll accidentally call localhost / a relative path).
 *
 * Optional:
 *   - If you intentionally use a Vite proxy in DEV, you can set VITE_API_BASE_URL=""
 *     (empty string) and call "/api/..." paths.
 */
const RAW_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_BASE ??
  import.meta.env.VITE_API_URL;

// If not set, default to localhost in DEV, but require it in PROD.
const BASE_URL = (
  RAW_BASE_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:8000" : "")
).replace(/\/+$/, "");

export const API_BASE = BASE_URL;

async function safeReadJson(res) {
  try {
    return await res.json();
  } catch {
    return null; // some endpoints (or errors) can return empty bodies
  }
}

function extractArrayPayload(payload, keys = []) {
  if (Array.isArray(payload)) return payload;

  for (const key of keys) {
    if (Array.isArray(payload?.[key])) {
      return payload[key];
    }
  }

  return [];
}

function extractObjectPayload(payload, keys = []) {
  for (const key of keys) {
    if (payload && typeof payload === "object" && !Array.isArray(payload?.[key])) {
      const candidate = payload[key];
      if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
        return candidate;
      }
    }
  }

  if (payload && typeof payload === "object" && !Array.isArray(payload)) return payload;

  return {};
}

function normalizeTeamCode(value) {
  return (value ?? "").toString().trim().toUpperCase();
}

function toNumberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function normalizeScheduleRows(rows) {
  if (!Array.isArray(rows)) return [];
  return rows.map((game) => {
    const home = normalizeTeamCode(game?.home_abbr || game?.home_team);
    const away = normalizeTeamCode(game?.away_abbr || game?.away_team);
    const season = toNumberOrNull(game?.season);
    const week = toNumberOrNull(game?.week);
    return {
      ...game,
      season: season ?? game?.season,
      week: week ?? game?.week,
      home_abbr: home || game?.home_abbr,
      away_abbr: away || game?.away_abbr,
      home_team: game?.home_team || home,
      away_team: game?.away_team || away,
    };
  });
}

function buildGameKey(game) {
  const season = toNumberOrNull(game?.season);
  const week = toNumberOrNull(game?.week);
  const home = normalizeTeamCode(game?.home_abbr || game?.home_team);
  const away = normalizeTeamCode(game?.away_abbr || game?.away_team);
  if (season == null || week == null || !home || !away) return "";
  return `${season}-${week}-${home}-${away}`;
}

function dedupeScheduleRows(rows) {
  const seen = new Set();
  const out = [];
  for (const row of Array.isArray(rows) ? rows : []) {
    const key = buildGameKey(row) || String(row?.game_id || "");
    if (key && seen.has(key)) continue;
    if (key) seen.add(key);
    out.push(row);
  }
  return out;
}

function deriveSeasonContext(scheduleRows, statusOverview = null) {
  const normalized = normalizeScheduleRows(scheduleRows);
  const firstGame = normalized[0] || null;
  const season = toNumberOrNull(firstGame?.season);
  const week = toNumberOrNull(firstGame?.week);
  const kickoff = firstGame?.kickoff || firstGame?.game_date || null;
  const gamesInNextWindow = normalized.length;
  const currentSeason = season ?? new Date().getFullYear();
  const phase = !normalized.length
    ? "offseason"
    : week != null && week >= 19
      ? "postseason"
      : "in_season";
  const label = phase === "offseason" ? "Offseason" : phase === "postseason" ? "Postseason" : `Week ${week ?? "?"}`;
  const message =
    phase === "offseason"
      ? "No live weekly slate is available right now."
      : phase === "postseason"
        ? "The next slate is in the playoffs."
        : "Upcoming games are ready for forecasting.";

  return {
    phase,
    label,
    message,
    current_season: currentSeason,
    display_week: week,
    games_in_next_window: Number.isFinite(gamesInNextWindow) ? gamesInNextWindow : 0,
    next_kickoff: kickoff,
    generated_at: new Date().toISOString(),
    dataset_rows: Number.isFinite(Number(statusOverview?.dataset?.rows))
      ? Number(statusOverview.dataset.rows)
      : 0,
  };
}

function createStatusOverviewFallback(totalPredictions = 0) {
  return {
    health: { status: "unknown" },
    dataset: { rows: 0 },
    history: { metrics: { total_predictions: totalPredictions } },
  };
}

function normalizeHistoryResponse(payload) {
  if (Array.isArray(payload)) {
    return { entries: payload, total: payload.length };
  }

  if (payload && Array.isArray(payload.entries)) {
    return {
      entries: payload.entries,
      total: Number.isFinite(Number(payload.total))
        ? Number(payload.total)
        : payload.entries.length,
    };
  }

  return { entries: [], total: 0 };
}

function buildUserHeaders(userId) {
  return userId ? { "X-User-Id": String(userId) } : undefined;
}

/**
 * fetchJson(path, options)
 * - path: "/health" | "/predict" | "/schedule/next-week" ...
 * - options: { method, headers, body, signal }
 */
export async function fetchJson(path, options = {}) {
  // Fail fast in production if the base URL wasn't configured on Vercel.
  if (import.meta.env.PROD && !BASE_URL) {
    throw new Error(
      "Missing VITE_API_BASE_URL. Set it in Vercel → Project Settings → Environment Variables " +
        "(example: https://<your-heroku-app>.herokuapp.com)."
    );
  }

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${BASE_URL}${normalizedPath}`;

  const res = await fetch(url, {
    method: "GET",
    ...options,
    credentials: "omit",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  // Parse body even for errors (helps UI show backend detail)
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

// -------------------------
// Health / Debug
// -------------------------

export async function getStatusOverview(userId = null) {
  // This endpoint is optional. If it fails, return a safe fallback object.
  try {
    const headers = buildUserHeaders(userId);
    const res = await fetchJson("/status/overview", headers ? { headers } : {});

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

    return createStatusOverviewFallback();
  } catch {
    console.warn("[client] Status overview unavailable; using fallback");
    return createStatusOverviewFallback();
  }
}

export async function getHealthStatus() {
  try {
    return await fetchJson("/health");
  } catch {
    console.warn("[client] Health endpoint unavailable; using fallback");
    return { status: "unknown", reason: "unavailable" };
  }
}

// Legacy alias kept for older wrappers that still import `health`.
export const health = getHealthStatus;

// -------------------------
// Context endpoints (cheap, cacheable)
// -------------------------

export async function getNextWeekSchedule() {
  // Backend may return:
  // - { games: [...] } (recommended)
  // - [...] (older)
  const res = await fetchJson("/schedule/next-week");
  return extractArrayPayload(res, ["games", "ScheduleGame"]);
}

export async function getScheduleForWeek(season, week, { fallbackRows = [] } = {}) {
  const requestedSeason = toNumberOrNull(season);
  const requestedWeek = toNumberOrNull(week);
  const rows = dedupeScheduleRows(normalizeScheduleRows(await getNextWeekSchedule()));

  if (requestedSeason == null && requestedWeek == null) {
    return rows;
  }

  const matches = rows.filter((game) => {
    const gameSeason = toNumberOrNull(game?.season);
    const gameWeek = toNumberOrNull(game?.week);
    const seasonOk = requestedSeason == null || gameSeason === requestedSeason;
    const weekOk = requestedWeek == null || gameWeek === requestedWeek;
    return seasonOk && weekOk;
  });

  if (matches.length > 0) {
    return matches;
  }

  if (Array.isArray(fallbackRows) && fallbackRows.length > 0) {
    const normalizedFallback = dedupeScheduleRows(normalizeScheduleRows(fallbackRows));
    const fallbackMatches = normalizedFallback.filter((game) => {
      const gameSeason = toNumberOrNull(game?.season);
      const gameWeek = toNumberOrNull(game?.week);
      const seasonOk = requestedSeason == null || gameSeason === requestedSeason;
      const weekOk = requestedWeek == null || gameWeek === requestedWeek;
      return seasonOk && weekOk;
    });
    return fallbackMatches.length > 0 ? fallbackMatches : normalizedFallback;
  }

  return [];
}

export async function getTeamLogos() {
  try {
    const res = await fetchJson("/teams/logos");
    const payload = extractObjectPayload(res, ["teams", "logos", "data"]);
    const out = {};

    for (const [code, meta] of Object.entries(payload)) {
      const normalizedCode = normalizeTeamCode(code);
      if (!normalizedCode || !meta || typeof meta !== "object") continue;
      out[normalizedCode] = {
        name: meta.name || meta.team_name || normalizedCode,
        logoUrl: meta.logoUrl || meta.logo_url || meta.team_logo_espn || "",
        primaryColor: meta.primaryColor || meta.primary_color || null,
        secondaryColor: meta.secondaryColor || meta.secondary_color || null,
        wordmark: meta.wordmark || meta.word_mark || null,
      };
    }

    return out;
  } catch {
    console.warn("[client] Team logos endpoint unavailable; using empty map");
    return {};
  }
}

export async function getSeasonContext(scheduleRows = null, statusOverview = null) {
  try {
    const rows = Array.isArray(scheduleRows) ? scheduleRows : await getNextWeekSchedule();
    return deriveSeasonContext(rows, statusOverview);
  } catch {
    return deriveSeasonContext([]);
  }
}

// -------------------------
// Cognitive endpoints (compute)
// -------------------------

export async function predictGame(payload, userId = null) {
  // Reuse the same normalization rules that the dashboard uses before it stores keys.
  const body = buildGamePredictPayload(payload);

  // Simple contract check: better to fail here than send junk to the API.
  if (
    !body.home_team ||
    !body.away_team ||
    !Number.isFinite(body.season) ||
    !Number.isFinite(body.week)
  ) {
    throw new Error("predictGame requires {home_team, away_team, season, week}");
  }

  return fetchJson("/predict", {
    method: "POST",
    body: JSON.stringify(body),
    ...(buildUserHeaders(userId) ? { headers: buildUserHeaders(userId) } : {}),
  });
}

export async function getPredictionHistory(limit = 100, userId = null) {
  try {
    const safeLimit = Number.isFinite(Number(limit)) ? Number(limit) : 100;
    const headers = buildUserHeaders(userId);
    const res = await fetchJson(`/history?limit=${safeLimit}`, headers ? { headers } : {});
    return normalizeHistoryResponse(res);
  } catch {
    console.warn("[client] History endpoint unavailable; using empty list");
    return normalizeHistoryResponse(null);
  }
}
