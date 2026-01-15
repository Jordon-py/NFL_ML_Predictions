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

const normalizeTeamCode = (value) =>
  (value ?? "").toString().trim().toUpperCase();

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
    const response = await fetch(resolvePublicAssetUrl("post_schedule.json"));
    if (!response.ok) return [];
    const payload = await response.json();
    return normalizePostseasonSchedule(payload);
  } catch {
    return [];
  }
};

/**
 * Check if the backend is healthy and models are loaded.
 * @returns {Promise<import('./types').HealthResponse>}
 */
export async function getHealthStatus() {
  return fetchJson("/health");
}

/**
 * Retrieve debug information about the backend state, including dataset stats.
 * @returns {Promise<Object>} Debug info object.
 */
export async function getDebugInfo() {
  return fetchJson("/debug");
}

/**
 * Fetch the schedule for the upcoming week.
 * 
 * @param {number|null} [season=null] - Optional season year. Defaults to current if null.
 * @returns {Promise<import('./types').ScheduleResponse>} The schedule data.
 */
export async function getNextWeekSchedule(season = null) {
  let scheduleRows = [];
  try {
    const url = season ? `/schedule/next-week?season=${season}` : "/schedule/next-week";
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
    return scheduleRows;
  }

  const postseasonRows = await fetchPostseasonSchedule();
  return postseasonRows.length ? postseasonRows : scheduleRows;
}

/**
 * Fetch team metadata (logos, colors, names).
 * @returns {Promise<Object>} Dictionary of team metadata keyed by abbreviation.
 */
export async function getTeamLogos() {
  const data = await fetchJson("/teams/logos");
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
export async function predictGame(home, away, season, week) {
  const payload = {
    home_team: home,
    away_team: away,
    season: parseInt(season, 10),
    week: parseInt(week, 10)
  };
  return fetchJson("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * simple chat interface for context-aware LLM interactions.
 * @param {Object} payload - { messages: Array, prediction: Object }
 * @returns {Promise<Object>} LLM response.
 */
export async function chatLLM(payload) {
  return fetchJson("/llm/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Request an explanation for a specific prediction.
 * @param {Object} payload - { home_team, away_team, season, week, ... }
 * @returns {Promise<Object>} Explanation result.
 */
export async function explainPrediction(payload) {
  return fetchJson("/predict/explain", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Get recent prediction history.
 * @param {number} [limit=100] - Max number of entries.
 * @returns {Promise<{entries: Array, total: number}>} History data.
 */
export async function getPredictionHistory(limit = 100) {
  const data = await fetchJson(`/history?limit=${limit}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
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
export async function getStatusOverview() {
  const data = await fetchJson("/status/overview", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
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
  return fetchJson("/status/models");
}

export async function reloadSystem() {
  return fetchJson("/admin/reload", {
    method: "POST",
  });
}

export async function retrainModel(config = {}) {
  return fetchJson("/admin/retrain", {
    method: "POST",
    body: JSON.stringify(config),
  });
}
