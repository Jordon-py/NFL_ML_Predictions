// Shared helpers for PredictionContext

export const PREDICTION_HISTORY_KEY = "prediction_history";
export const MAX_HISTORY_ENTRIES = 100;

/**
 * Safely access import.meta.env without tripping TS definitions.
 * @returns {Record<string, any> | undefined}
 */
export function getMetaEnv() {
  const meta = typeof import.meta !== "undefined" ? /** @type {any} */ (import.meta) : undefined;
  return meta?.env;
}

/**
 * Build a consistent game key from either schedule rows or prediction entries.
 * @param {any} gameLike
 * @returns {string}
 */
export function buildGameKey(gameLike) {
  if (!gameLike) return "";
  if (typeof gameLike.game_id === "string" && gameLike.game_id.trim()) {
    return gameLike.game_id;
  }
  const parts = [
    gameLike.season,
    gameLike.week,
    gameLike.home_abbr || gameLike.home_team,
    gameLike.away_abbr || gameLike.away_team,
  ].filter(Boolean);
  return parts.join("-");
}

/**
 * Safe hydration from localStorage.
 * @param {string} storageKey
 * @returns {any[]}
 */
export function loadPredictionHistoryFromStorage(storageKey = PREDICTION_HISTORY_KEY) {
  try {
    const rawHistoryData = localStorage.getItem(storageKey);
    if (!rawHistoryData) return [];
    const parsedHistory = JSON.parse(rawHistoryData);
    return Array.isArray(parsedHistory) ? parsedHistory : [];
  } catch {
    return [];
  }
}

// Lightweight CSV parser for public/data/myteamdescriptions.csv
// Format: team_name,abbr,logo_url
/**
 * @param {string} text
 * @returns {Record<string, {name: string, logoUrl: string}>}
 */
export function parseTeamsCsv(text) {
  if (!text) return {};
  const lines = text.trim().split(/\r?\n/);
  const out = {};
  for (let i = 1; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) continue;
    const parts = line.split(",");
    if (parts.length < 3) continue;
    const [teamName, abbr, logoUrl] = parts;
    const code = (abbr || "").trim().toUpperCase();
    if (!code) continue;
    out[code] = {
      name: (teamName || code).trim(),
      logoUrl: (logoUrl || "").trim(),
    };
  }
  return out;
}
