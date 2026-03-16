// ==========================================
// File: frontend/src/utils/predictionContextUtils.js
// Role: Frontend utility helpers.
// Input Data: Function parameters.
// Output Data: Transformed values.
// Dependencies: None
// Notes: Shared across UI modules.
// ==========================================

// Shared helpers for prediction state and UI normalization.

export const PREDICTION_HISTORY_KEY = "prediction_history";
export const MAX_HISTORY_ENTRIES = 100;

const normalizeToken = (value) =>
  value == null ? "" : String(value).trim().toUpperCase();

function normalizeGameId(rawId) {
  const trimmed = typeof rawId === "string" ? rawId.trim() : "";
  if (!trimmed) return "";
  const parts = trimmed.split(/[-_]/).filter(Boolean);
  if (parts.length >= 4) {
    const [season, week, home, away] = parts;
    return `${season}-${week}-${normalizeToken(home)}-${normalizeToken(away)}`;
  }
  return trimmed;
}

function buildCompositeKey(gameLike) {
  const parts = [
    gameLike?.season,
    gameLike?.week,
    normalizeToken(gameLike?.home_abbr || gameLike?.home_team),
    normalizeToken(gameLike?.away_abbr || gameLike?.away_team),
  ].filter(Boolean);
  return parts.join("-");
}

/**
 * Build a consistent game key from either schedule rows or prediction entries.
 * @param {any} gameLike
 * @returns {string}
 */
export function buildGameKey(gameLike) {
  if (!gameLike) return "";
  const compositeKey = buildCompositeKey(gameLike);
  if (compositeKey) {
    return compositeKey;
  }
  if (typeof gameLike.game_id === "string" && gameLike.game_id.trim()) {
    return normalizeGameId(gameLike.game_id);
  }
  return "";
}

/**
 * Remove duplicate games while preserving the first occurrence order.
 * This protects the UI against noisy schedule feeds without changing card rendering logic.
 * @param {any[]} rows
 * @returns {any[]}
 */
export function dedupeGamesByKey(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return [];

  const seen = new Set();
  const deduped = [];
  for (const row of rows) {
    const key = buildGameKey(row);
    const fallbackKey = key || JSON.stringify([
      row?.season ?? "",
      row?.week ?? "",
      row?.kickoff ?? "",
      normalizeToken(row?.home_team ?? row?.home_abbr),
      normalizeToken(row?.away_team ?? row?.away_abbr),
    ]);
    if (seen.has(fallbackKey)) continue;
    seen.add(fallbackKey);
    deduped.push(row);
  }
  return deduped;
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
  // Frontend aliases to match backend normalization
  const ALIASES = {
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV"
  };

  // 1. Build canonical map
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

  // 2. Populate aliases
  Object.entries(ALIASES).forEach(([alias, canonical]) => {
    if (out[canonical] && !out[alias]) {
      out[alias] = out[canonical];
    }
  });

  return out;
}
