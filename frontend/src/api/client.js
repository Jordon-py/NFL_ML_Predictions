/**
 * NFL Prediction App - Core API Client (Expert v1.2)
 * ================================================
 *
 * A robust, high-performance fetch wrapper engineered for the NFL ML Predictions ecosystem.
 * Features:
 *  - Unified error handling in fetchJson.
 *  - Request timeout and cancellation via AbortController.
 *  - Environment-aware URL resolution with trailing-slash normalization.
 *  - Defensive JSON parsing resilient to empty or malformed responses.
 *  - Data normalization layers to bridge backend-frontend schema drift.
 */
/**
 * Retrieve system health and model readiness.
 */
// client.js (minimal edits)

import { fetchJson } from "./fetch";
import { dedupeGamesByKey } from "../utils/predictionContextUtils.js";

const POSTSEASON_WEEK_BY_ROUND = {
  "Wild Card": 19,
  "Divisional": 20,
  "Conference Championship": 21,
  "Super Bowl": 22,
};

const toNumberOrNull = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const DEFAULT_SEASON_CONTEXT = {
  phase: "offseason",
  label: "Offseason",
  message: "No live weekly slate is available right now.",
  current_season: new Date().getFullYear(),
  display_week: null,
  games_in_next_window: 0,
  next_kickoff: null,
  generated_at: new Date().toISOString(),
};

/**
 * Robust CSV parser for schedule data.
 */
function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = [];
    let cur = "";
    let inQuotes = false;
    for (let char of line) {
      if (char === '"') inQuotes = !inQuotes;
      else if (char === "," && !inQuotes) {
        values.push(cur.trim());
        cur = "";
      } else cur += char;
    }
    values.push(cur.trim());
    const row = {};
    headers.forEach((h, i) => (row[h.trim()] = values[i] || ""));
    return row;
  });
}

const normalizeTeamCode = (value) =>
  (value ?? "").toString().trim().toUpperCase();

const withUserContext = (userId, options = {}) =>
  userId ? { ...options, userId } : options;

const extractScheduleRows = (data) => {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.games)) return data.games;
  if (Array.isArray(data?.schedule)) return data.schedule;
  return [];
};

const resolvePublicAssetUrl = (assetPath) => {
  const base = import.meta.env.BASE_URL || "/";
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  const normalizedAsset = assetPath.startsWith("/") ? assetPath.slice(1) : assetPath;
  return `${normalizedBase}${normalizedAsset}`;
};

const hasFutureKickoff = (rows) => {
  if (!Array.isArray(rows) || rows.length === 0) return false;
  const now = Date.now();
  return rows.some((game) => {
    if (!game?.kickoff) return false;
    const ts = new Date(game.kickoff).getTime();
    return Number.isFinite(ts) && ts >= now;
  });
};

const normalizePostseasonSchedule = (payload) => {
  if (!payload || typeof payload !== "object") return [];
  if (Array.isArray(payload.games)) return payload.games;

  const season = toNumberOrNull(payload.season);
  const rounds = Array.isArray(payload?.postseason?.rounds)
    ? payload.postseason.rounds
    : [];
  if (!rounds.length) return [];

  const games = [];
  rounds.forEach((round, index) => {
    const roundName = round?.name;
    const week =
      toNumberOrNull(POSTSEASON_WEEK_BY_ROUND[roundName]) ?? 19 + index;
    const roundGames = Array.isArray(round?.games) ? round.games : [];

    roundGames.forEach((game) => {
      const home = game?.home || null;
      const away = game?.away || null;
      const homeAbbr = normalizeTeamCode(home?.abbr || home?.team || home);
      const awayAbbr = normalizeTeamCode(away?.abbr || away?.team || away);
      if (!homeAbbr || !awayAbbr) return;

      games.push({
        season,
        week,
        kickoff: game?.kickoff_local || game?.kickoff || null,
        home_team: homeAbbr,
        away_team: awayAbbr,
        home_abbr: homeAbbr,
        away_abbr: awayAbbr,
        home_name: home?.name || homeAbbr,
        away_name: away?.name || awayAbbr,
      });
    });
  });

  return games;
};

const fetchPostseasonSchedule = async () => {
  try {
    const url = resolvePublicAssetUrl("Nfl_schedule_2025.csv");
    const response = await fetch(url);
    if (!response.ok) {
      console.warn("Failed to fetch schedule CSV:", url, response.status);
      return [];
    }
    const text = await response.text();
    const allRows = parseCsv(text);
    console.log(`Parsed ${allRows.length} rows from CSV`);

    // Filter for postseason games (or everything if REG fails)
    // We prefer games with game_type !== 'REG' for postseason logic
    const postRows = allRows.filter(r => r.game_type !== "REG");

    return postRows.map(r => ({
      game_id: r.game_id,
      season: parseInt(r.season) || 2025,
      week: parseInt(r.week) || 0,
      game_type: r.game_type,
      kickoff: r.gametime ? `${r.gameday} ${r.gametime}` : r.gameday,
      home_team: r.home_team,
      away_team: r.away_team,
      home_abbr: r.home_team,
      away_abbr: r.away_team,
      stadium: r.stadium
    }));
  } catch (error) {
    console.error("Error fetching/parsing postseason schedule CSV:", error);
    return [];
  }
};

/**
 * Check if the backend is healthy and models are loaded.
 * @returns {Promise<import('./types').HealthResponse>}
 */
export async function getHealthStatus() {
  return fetchJson("/api/health");
}

/**
 * Fetch schedule-aware season context (in-season/postseason/offseason).
 * @returns {Promise<Object>} Normalized season context.
 */
export async function getSeasonContext() {
  try {
    const data = await fetchJson("/api/season/context");
    if (!data || typeof data !== "object") return DEFAULT_SEASON_CONTEXT;
    return {
      ...DEFAULT_SEASON_CONTEXT,
      ...data,
      current_season: toNumberOrNull(data.current_season) ?? DEFAULT_SEASON_CONTEXT.current_season,
      display_week: toNumberOrNull(data.display_week),
      games_in_next_window: toNumberOrNull(data.games_in_next_window) ?? 0,
    };
  } catch {
    return DEFAULT_SEASON_CONTEXT;
  }
}

/**
 * Retrieve debug information about the backend state, including dataset stats.
 * @returns {Promise<Object>} Debug info object.
 */
export async function getDebugInfo() {
  return fetchJson("/api/debug");
}

/**
 * Fetch the schedule for the upcoming week.
 *
 * @param {number|null} [season=null] - Optional season year. Defaults to current if null.
 * @returns {Promise<Array<Object>>} Array of upcoming games (ScheduleEntry-like rows).
 */
export async function getNextWeekSchedule(season = null) {
  let scheduleRows = [];
  try {
    const url = season ? `/api/schedule/next-week?season=${season}` : "/api/schedule/next-week";
    const data = await fetchJson(url);
    scheduleRows = extractScheduleRows(data);
  } catch {
    scheduleRows = [];
  }

  const weekValue = toNumberOrNull(scheduleRows?.[0]?.week);
  const week18Over = weekValue === 18 && !hasFutureKickoff(scheduleRows);
  const shouldUsePostseason =
    scheduleRows.length === 0 ||
    (weekValue != null && weekValue > 18) ||
    week18Over;

  if (!shouldUsePostseason) {
    return dedupeGamesByKey(scheduleRows);
  }

  const postseasonRows = await fetchPostseasonSchedule();
  return dedupeGamesByKey(postseasonRows.length ? postseasonRows : scheduleRows);
}

/**
 * Fetch team metadata (logos, colors, names).
 * @returns {Promise<Object>} Dictionary of team metadata keyed by abbreviation.
 */
export async function getTeamLogos() {
  const data = await fetchJson("/api/teams/logos");
  if (data && typeof data === "object" && data.teams && typeof data.teams === "object") {
    return data.teams;
  }
  return {};
}

/**
 * Send a prediction request for a specific game.
 *
 * @param {string} home - Home team code (e.g. "KC")
 * @param {string} away - Away team code (e.g. "BUF")
 * @param {number} season - Season year
 * @param {number} week - Week number
 * @returns {Promise<import('./types').UnifiedPredictionResponse>} The prediction result.
 */
export async function predictGame(home, away, season, week, userId = null) {
  const payload = {
    home_team: home,
    away_team: away,
    season: parseInt(season, 10),
    week: parseInt(week, 10)
  };
  return fetchJson("/predict", {
    ...withUserContext(userId),
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * simple chat interface for context-aware LLM interactions.
 * @param {Object} payload - { messages: Array, prediction: Object }
 * @returns {Promise<Object>} LLM response.
 */
export async function chatLLM(payload, userId = null) {
  return fetchJson("/llm/chat", {
    ...withUserContext(userId),
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Request an explanation for a specific prediction.
 * @param {Object} payload - { home_team, away_team, season, week, ... }
 * @returns {Promise<Object>} Explanation result.
 */
export async function explainPrediction(payload, userId = null) {
  return fetchJson("/predict/explain", {
    ...withUserContext(userId),
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Get recent prediction history.
 * @param {number} [limit=100] - Max number of entries.
 * @returns {Promise<{entries: Array, total: number}>} History data.
 */
export async function getPredictionHistory(limit = 100, userId = null) {
  const data = await fetchJson(`/history?limit=${limit}`, withUserContext(userId));
  if (Array.isArray(data)) {
    return { entries: data, total: data.length };
  }
  if (data && Array.isArray(data.entries)) {
    return { entries: data.entries, total: data.total ?? data.entries.length };
  }
  return { entries: [], total: 0 };
}

/**
 * Get a high-level overview of system status (health, history stats, dataset info).
 * @returns {Promise<import('./types').StatusOverviewResponse>} Status overview.
 */
export async function getStatusOverview(userId = null) {
  const data = await fetchJson("/status/overview", withUserContext(userId));
  if (!data) {
    return {
      health: { status: "unknown", mode: "unknown", reason: "no data" },
      dataset: { rows: 0, features: 0 },
      history: { total_predictions: 0, win_rate: null, note: "no data" },
    };
  }
  return data;
}

export async function getModelsStatus() {
  return fetchJson("/api/status/models");
}

export async function reloadSystem() {
  return fetchJson("/api/admin/reload", {
    method: "POST",
  });
}

export async function retrainModel(config = {}) {
  return fetchJson("/api/admin/retrain", {
    method: "POST",
    body: JSON.stringify(config),
  });
}
