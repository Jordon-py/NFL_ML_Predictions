/**
 * File: frontend/src/api/client.js
 *
 * Purpose:
 *   One tiny, consistent fetch wrapper for the whole app (schedule + predict + history).
 *
 * Data shapes:
 *   Schedule calls return arrays of normalized game rows. History calls return
 *   `{ entries, total }` plus summary metrics. Prediction calls return the flat
 *   backend PredictionResponse contract.
 *
 * The production gotcha (Vercel):
 *   Vite does NOT ship your local `.env` file to Vercel. You must set env vars
 *   in Vercel Project Settings → Environment Variables.
 *
 * Required env:
 *   - Local dev:    VITE_API_DEV=http://127.0.0.1:8000
 *   - Vercel prod:  VITE_API_BASE_URL=https://<your-heroku-app>.herokuapp.com
 *
 * Notes:
 *   - Trailing slashes are stripped so URL joins are predictable.
 *   - Errors are thrown as HttpError(status, url, body) so UI can display useful info.
 *
 * Important functions (line numbers last refreshed 2026-04-30):
 *   - fetchJson: around line 585
 *   - getOffseasonStatus: around line 676
 *   - getScheduleForWeek: around line 716
 *   - predictGame: around line 792
 *
 * Possible bugs:
 *   - Local CSV fallbacks can drift from the backend if schedule assets are not
 *     updated together.
 *
 * Enhancement ideas:
 *   - Extract schedule normalization into a separate tested module.
 *   - Add an endpoint capability probe object to make fallback behavior visible.
 */

import { buildPredictPayload as buildGamePredictPayload } from "../utils/gameUtils.js";
import Papa from "papaparse";

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
 *   - If you intentionally use a Vite proxy in DEV, you can set VITE_API_DEV=""
 *     (empty string) and call "/api/..." paths.
 */
const DEV_BASE_URL =
  import.meta.env.VITE_API_DEV ??
  import.meta.env.VITE_DEV_ENV ??
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_BASE ??
  import.meta.env.VITE_API_URL;

const PROD_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_BASE ??
  import.meta.env.VITE_API_URL;

const RAW_BASE_URL = import.meta.env.DEV ? DEV_BASE_URL : PROD_BASE_URL;

// If not set, default to localhost in DEV, but require it in PROD.
const BASE_URL = (
  RAW_BASE_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:8000" : "")
).replace(/\/+$/, "");

export const API_BASE = BASE_URL;
const APP_BASE_PATH = import.meta.env.BASE_URL || "/";
const LOCAL_SCHEDULE_CACHE = new Map();
// Cache one-time endpoint capability checks so the app does not keep retrying
// known-missing routes against older deployments on every render.
const ENDPOINT_SUPPORT = {
  historySummary: null,
  scheduleQuery: null,
  nextWeekSchedule: null,
};

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

function isHttpNotFound(error) {
  return error instanceof HttpError && error.status === 404;
}

function buildPublicAssetPath(relativePath) {
  const base = APP_BASE_PATH.endsWith("/") ? APP_BASE_PATH : `${APP_BASE_PATH}/`;
  return `${base}${String(relativePath ?? "").replace(/^\/+/, "")}`;
}

function toIsoStringOrNull(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

function pickLatestIso(currentValue, candidateValue) {
  const candidateIso = toIsoStringOrNull(candidateValue);
  if (!candidateIso) return currentValue;
  if (!currentValue) return candidateIso;
  return new Date(candidateIso) > new Date(currentValue) ? candidateIso : currentValue;
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

/**
 * Build summary metrics from raw history entries.
 *
 * This is the compatibility path for older backends that expose `/history`
 * but not `/history/summary`. Keeping the logic here lets newer UI surfaces
 * depend on one normalized summary shape without caring where it came from.
 */
function buildHistorySummaryFromEntries(entries = [], totalOverride = null) {
  const rows = Array.isArray(entries) ? entries : [];
  let resolvedGames = 0;
  let correctPredictions = 0;
  let spreadErrorTotal = 0;
  let spreadErrorCount = 0;
  let confidenceTotal = 0;
  let confidenceCount = 0;
  let latestPredictionAt = null;
  let lastScoreSyncAt = null;

  for (const entry of rows) {
    if (!entry || typeof entry !== "object") continue;

    latestPredictionAt = pickLatestIso(
      latestPredictionAt,
      entry.ts || entry.timestamp || entry.created_at || entry.predicted_at || null
    );

    const homeProb = toNumberOrNull(entry.home_win_probability);
    const awayProb = toNumberOrNull(entry.away_win_probability);
    if (homeProb != null || awayProb != null) {
      confidenceTotal += Math.max(homeProb ?? 0, awayProb ?? 0);
      confidenceCount += 1;
    }

    const actualHome = toNumberOrNull(entry.final_home_score ?? entry.actual_home_score);
    const actualAway = toNumberOrNull(entry.final_away_score ?? entry.actual_away_score);
    if (actualHome == null || actualAway == null) continue;

    resolvedGames += 1;
    lastScoreSyncAt = pickLatestIso(
      lastScoreSyncAt,
      entry.score_updated_at || entry.last_score_sync_at || entry.updated_at || null
    );

    const predictedHome = toNumberOrNull(entry.home_score);
    const predictedAway = toNumberOrNull(entry.away_score);
    const predictedDiff =
      predictedHome != null && predictedAway != null
        ? predictedHome - predictedAway
        : toNumberOrNull(entry.point_diff);
    const actualDiff = actualHome - actualAway;

    if (predictedDiff != null) {
      spreadErrorTotal += Math.abs(predictedDiff - actualDiff);
      spreadErrorCount += 1;
    }

    let predictedHomeWins = null;
    if (predictedDiff != null) {
      predictedHomeWins = predictedDiff >= 0;
    } else if (homeProb != null || awayProb != null) {
      predictedHomeWins = (homeProb ?? 0) >= (awayProb ?? 0);
    }

    if (predictedHomeWins != null) {
      const actualHomeWins = actualDiff >= 0;
      if (predictedHomeWins === actualHomeWins) correctPredictions += 1;
    }
  }

  return normalizeHistorySummaryResponse({
    total_predictions: Number.isFinite(Number(totalOverride)) ? Number(totalOverride) : rows.length,
    resolved_games: resolvedGames,
    win_rate: resolvedGames > 0 ? correctPredictions / resolvedGames : null,
    avg_abs_spread_error: spreadErrorCount > 0 ? spreadErrorTotal / spreadErrorCount : null,
    avg_confidence: confidenceCount > 0 ? confidenceTotal / confidenceCount : null,
    latest_prediction_at: latestPredictionAt,
    last_score_sync_at: lastScoreSyncAt,
  });
}

function buildKickoffFromScheduleRow(row) {
  const gameday = String(row?.gameday || "").trim();
  const gametime = String(row?.gametime || "").trim();
  if (!gameday) return null;
  return gametime ? `${gameday}T${gametime}:00` : gameday;
}

function getKickoffTimestamp(row) {
  const kickoffCandidate =
    row?.kickoff ||
    row?.game_date ||
    row?.scheduled ||
    buildKickoffFromScheduleRow(row);
  if (!kickoffCandidate) return null;
  const date = new Date(kickoffCandidate);
  const timestamp = date.getTime();
  return Number.isNaN(timestamp) ? null : timestamp;
}

/**
 * Hide already-completed slates from the dashboard's default "next slate"
 * experience. Explicit season/week requests still opt into past games.
 */
function filterUpcomingScheduleRows(rows, now = Date.now()) {
  return dedupeScheduleRows(normalizeScheduleRows(rows)).filter((row) => {
    const kickoffTimestamp = getKickoffTimestamp(row);
    if (kickoffTimestamp == null) return true;
    return kickoffTimestamp > now;
  });
}

function normalizeLocalScheduleRows(rows) {
  if (!Array.isArray(rows)) return [];
  return dedupeScheduleRows(
    normalizeScheduleRows(
      rows.map((row) => {
        const season = toNumberOrNull(row?.season);
        const week = toNumberOrNull(row?.week);
        const home = normalizeTeamCode(row?.home_team || row?.home_abbr);
        const away = normalizeTeamCode(row?.away_team || row?.away_abbr);
        return {
          game_id:
            row?.game_id ||
            (season != null && week != null && home && away
              ? `${season}-${week}-${home}-${away}`
              : ""),
          season,
          week,
          home_team: home,
          away_team: away,
          home_abbr: home,
          away_abbr: away,
          kickoff: buildKickoffFromScheduleRow(row),
          stadium: row?.stadium || null,
          game_type: row?.game_type || null,
          gameday: row?.gameday || null,
          gametime: row?.gametime || null,
        };
      })
    )
  );
}

/**
 * Load a bundled public schedule CSV for one season.
 *
 * These CSVs are a frontend-only fallback used when the deployed backend is
 * older than the current client or when a local backend is unavailable.
 */
async function loadLocalScheduleSeason(season) {
  const normalizedSeason = toNumberOrNull(season);
  if (normalizedSeason == null) return [];
  if (LOCAL_SCHEDULE_CACHE.has(normalizedSeason)) {
    return LOCAL_SCHEDULE_CACHE.get(normalizedSeason) ?? [];
  }

  const url = buildPublicAssetPath(`schedules/Nfl_schedule_${normalizedSeason}.csv`);
  try {
    const response = await fetch(url, { method: "GET", credentials: "omit" });
    if (!response.ok) {
      LOCAL_SCHEDULE_CACHE.set(normalizedSeason, []);
      return [];
    }
    const csvText = await response.text();
    const parsed = Papa.parse(csvText, {
      header: true,
      skipEmptyLines: true,
    });
    const normalizedRows = normalizeLocalScheduleRows(parsed.data);
    LOCAL_SCHEDULE_CACHE.set(normalizedSeason, normalizedRows);
    return normalizedRows;
  } catch {
    LOCAL_SCHEDULE_CACHE.set(normalizedSeason, []);
    return [];
  }
}

async function loadLocalScheduleForWeek(season, week) {
  const normalizedSeason = toNumberOrNull(season);
  const normalizedWeek = toNumberOrNull(week);
  if (normalizedSeason == null || normalizedWeek == null) return [];
  const rows = await loadLocalScheduleSeason(normalizedSeason);
  return rows.filter(
    (game) =>
      toNumberOrNull(game?.season) === normalizedSeason &&
      toNumberOrNull(game?.week) === normalizedWeek
  );
}

async function loadLocalNextWeekSchedule() {
  const candidateSeasons = Array.from(
    new Set([new Date().getFullYear() + 1, new Date().getFullYear(), 2025, 2024, new Date().getFullYear() - 1])
  )
    .map((value) => toNumberOrNull(value))
    .filter((value) => value != null);

  const seasons = [];
  for (const season of candidateSeasons) {
    if (!seasons.includes(season)) seasons.push(season);
  }

  const rowsBySeason = await Promise.all(seasons.map((season) => loadLocalScheduleSeason(season)));
  const futureRows = filterUpcomingScheduleRows(rowsBySeason.flat());
  if (futureRows.length === 0) return [];

  const sorted = [...futureRows].sort((a, b) => {
    const kickoffA = getKickoffTimestamp(a) ?? Number.MAX_SAFE_INTEGER;
    const kickoffB = getKickoffTimestamp(b) ?? Number.MAX_SAFE_INTEGER;
    if (kickoffA !== kickoffB) return kickoffA - kickoffB;
    const seasonDelta = Number(toNumberOrNull(a?.season) ?? 0) - Number(toNumberOrNull(b?.season) ?? 0);
    if (seasonDelta !== 0) return seasonDelta;
    return Number(toNumberOrNull(a?.week) ?? 0) - Number(toNumberOrNull(b?.week) ?? 0);
  });

  const firstGame = sorted[0];
  const targetSeason = toNumberOrNull(firstGame?.season);
  const targetWeek = toNumberOrNull(firstGame?.week);
  if (targetSeason == null || targetWeek == null) return [];

  return sorted.filter(
    (game) =>
      toNumberOrNull(game?.season) === targetSeason &&
      toNumberOrNull(game?.week) === targetWeek
  );
}

/**
 * Find the newest archived slate available in the bundled CSVs.
 *
 * This keeps the dashboard useful during the offseason without pretending
 * those games are still upcoming.
 */
export async function getLatestArchivedSchedule() {
  const candidateSeasons = Array.from(
    new Set([
      new Date().getFullYear() + 1,
      new Date().getFullYear(),
      2025,
      2024,
      new Date().getFullYear() - 1,
      new Date().getFullYear() - 2,
    ])
  )
    .map((value) => toNumberOrNull(value))
    .filter((value) => value != null);

  const seasons = [];
  for (const season of candidateSeasons) {
    if (!seasons.includes(season)) seasons.push(season);
  }

  const rows = (await Promise.all(seasons.map((season) => loadLocalScheduleSeason(season)))).flat();
  if (rows.length === 0) return [];

  const sorted = dedupeScheduleRows(normalizeScheduleRows(rows)).sort((a, b) => {
    const seasonDelta = Number(toNumberOrNull(b?.season) ?? 0) - Number(toNumberOrNull(a?.season) ?? 0);
    if (seasonDelta !== 0) return seasonDelta;
    const weekDelta = Number(toNumberOrNull(b?.week) ?? 0) - Number(toNumberOrNull(a?.week) ?? 0);
    if (weekDelta !== 0) return weekDelta;
    const kickoffA = getKickoffTimestamp(a) ?? 0;
    const kickoffB = getKickoffTimestamp(b) ?? 0;
    return kickoffB - kickoffA;
  });

  const latestGame = sorted[0];
  const targetSeason = toNumberOrNull(latestGame?.season);
  const targetWeek = toNumberOrNull(latestGame?.week);
  if (targetSeason == null || targetWeek == null) return [];

  return sorted.filter(
    (game) =>
      toNumberOrNull(game?.season) === targetSeason &&
      toNumberOrNull(game?.week) === targetWeek
  );
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
    return { entries: payload, total: payload.length, summary: null };
  }

  if (payload && Array.isArray(payload.entries)) {
    return {
      entries: payload.entries,
      total: Number.isFinite(Number(payload.total))
        ? Number(payload.total)
        : payload.entries.length,
      summary:
        payload.summary && typeof payload.summary === "object" && !Array.isArray(payload.summary)
          ? payload.summary
          : null,
    };
  }

  return { entries: [], total: 0, summary: null };
}

/**
 * Normalize backend and fallback summary payloads into one stable frontend
 * contract so dashboard, history, and stats pages can stay in sync.
 */
function normalizeHistorySummaryResponse(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return {
      total_predictions: 0,
      resolved_games: 0,
      win_rate: null,
      avg_abs_spread_error: null,
      avg_confidence: null,
      latest_prediction_at: null,
      last_score_sync_at: null,
    };
  }

  return {
    total_predictions: Number.isFinite(Number(payload.total_predictions))
      ? Number(payload.total_predictions)
      : 0,
    resolved_games: Number.isFinite(Number(payload.resolved_games))
      ? Number(payload.resolved_games)
      : 0,
    win_rate:
      typeof payload.win_rate === "number" && Number.isFinite(payload.win_rate)
        ? payload.win_rate
        : null,
    avg_abs_spread_error:
      typeof payload.avg_abs_spread_error === "number" && Number.isFinite(payload.avg_abs_spread_error)
        ? payload.avg_abs_spread_error
        : null,
    avg_confidence:
      typeof payload.avg_confidence === "number" && Number.isFinite(payload.avg_confidence)
        ? payload.avg_confidence
        : null,
    latest_prediction_at: payload.latest_prediction_at || null,
    last_score_sync_at: payload.last_score_sync_at || null,
  };
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
  // This endpoint is optional. Some older deployments only expose `/health`
  // and `/history`, so the UI degrades to a safe, low-detail snapshot.
  try {
    const headers = buildUserHeaders(userId);
    const res = await fetchJson("/status/overview", headers ? { headers } : {});

    if (res && typeof res === "object") {
      const dataset = res.dataset ?? { rows: 0 };
      const historyMetrics =
        res.history?.metrics && typeof res.history.metrics === "object"
          ? normalizeHistorySummaryResponse(res.history.metrics)
          : normalizeHistorySummaryResponse({
              total_predictions: res.history?.total_predictions,
              resolved_games: res.history?.resolved_games,
              win_rate: res.history?.win_rate,
              avg_abs_spread_error: res.history?.avg_abs_spread_error,
              avg_confidence: res.history?.avg_confidence,
              latest_prediction_at: res.history?.latest_prediction_at,
              last_score_sync_at: res.history?.last_score_sync_at,
            });
      return {
        ...res,
        health: res.health ?? { status: "unknown" },
        dataset,
        history: {
          ...(res.history && typeof res.history === "object" ? res.history : {}),
          metrics: historyMetrics,
        },
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

export async function getOffseasonStatus() {
  try {
    return await fetchJson("/offseason/status");
  } catch {
    console.warn("[client] Offseason status unavailable; using fallback");
    return {
      offseason_mode: false,
      current_season: null,
      current_week: null,
      next_known_schedule_date: null,
      days_until_next_game: null,
      data_freshness_seconds: null,
      dataset_hash: null,
      last_trained_at: null,
    };
  }
}

// -------------------------
// Context endpoints (cheap, cacheable)
// -------------------------

export async function getNextWeekSchedule() {
  // Backend may return:
  // - { games: [...] } (recommended)
  // - [...] (older)
  if (ENDPOINT_SUPPORT.nextWeekSchedule !== false) {
    try {
      const res = await fetchJson("/schedule/next-week");
      ENDPOINT_SUPPORT.nextWeekSchedule = true;
      return filterUpcomingScheduleRows(extractArrayPayload(res, ["games", "ScheduleGame"]));
    } catch (error) {
      if (!isHttpNotFound(error)) throw error;
      ENDPOINT_SUPPORT.nextWeekSchedule = false;
    }
  }

  return loadLocalNextWeekSchedule();
}

export async function getScheduleForWeek(season, week, { fallbackRows = [] } = {}) {
  const requestedSeason = toNumberOrNull(season);
  const requestedWeek = toNumberOrNull(week);
  if (requestedSeason != null && requestedWeek != null) {
    if (ENDPOINT_SUPPORT.scheduleQuery !== false) {
      try {
        const res = await fetchJson(`/schedule?season=${requestedSeason}&week=${requestedWeek}`);
        ENDPOINT_SUPPORT.scheduleQuery = true;
        return dedupeScheduleRows(normalizeScheduleRows(extractArrayPayload(res, ["games", "ScheduleGame"])));
      } catch (error) {
        if (!isHttpNotFound(error)) throw error;
        ENDPOINT_SUPPORT.scheduleQuery = false;
      }
    }

    const localRows = await loadLocalScheduleForWeek(requestedSeason, requestedWeek);
    if (localRows.length > 0) return localRows;
  }

  if (requestedSeason == null && requestedWeek == null) {
    return dedupeScheduleRows(normalizeScheduleRows(await getNextWeekSchedule()));
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

export async function getHistorySummary(userId = null) {
  if (ENDPOINT_SUPPORT.historySummary !== false) {
    try {
      const headers = buildUserHeaders(userId);
      const res = await fetchJson("/history/summary", headers ? { headers } : {});
      ENDPOINT_SUPPORT.historySummary = true;
      return normalizeHistorySummaryResponse(res);
    } catch (error) {
      if (!isHttpNotFound(error)) {
        console.warn("[client] History summary endpoint unavailable; using empty summary");
        return normalizeHistorySummaryResponse(null);
      }
      ENDPOINT_SUPPORT.historySummary = false;
    }
  }

  try {
    // Compatibility path: compute the summary client-side from `/history`
    // when the backend has not been redeployed with `/history/summary` yet.
    const history = await getPredictionHistory(250, userId);
    return buildHistorySummaryFromEntries(history.entries, history.total);
  } catch {
    console.warn("[client] History summary endpoint unavailable; using empty summary");
    return normalizeHistorySummaryResponse(null);
  }
}
