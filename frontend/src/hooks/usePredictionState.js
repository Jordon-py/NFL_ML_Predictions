// ==========================================
// File: frontend/src/hooks/usePredictionState.js
// Role: React hook for UI state management.
// Input Data: Hook params and state.
// Output Data: State values and actions.
// Dependencies: react
// Notes: Consumed by components.
// ==========================================

/**
 * FILE: frontend/src/hooks/usePredictionState.js
 * PURPOSE: Centralized state for NFL predictions, polling, and history.
 * INPUTS / DATA SHAPES:
 *   - Fetches from: getNextWeekSchedule, getHealthStatus, getPredictionHistory.
 *   - State: { schedule, week, predictions, history, health, loading, errors, current }.
 * OUTPUT / SIDE EFFECTS: Polling for health; localStorage sync for history.
 * KEY FUNCTIONS:
 *   - usePredictionState(): Returns unified state object.
 * DEPENDENCIES: React, client.js
 */

import { useEffect, useState, useCallback } from "react";
import {
  getNextWeekSchedule,
  getHealthStatus as fetchHealth,
  getPredictionHistory,
  getTeamLogos,
  getSeasonContext,
} from "../api/client.js";
import {
  buildGameKey,
  loadPredictionHistoryFromStorage,
  MAX_HISTORY_ENTRIES,
  PREDICTION_HISTORY_KEY,
} from "../utils/predictionContextUtils.js";

const INITIAL_HEALTH = { status: "loading", mode: "none" };
const HEALTH_POLL_MS = 25000; // Poll every 25 seconds - balanced load reduction
const INITIAL_SEASON_CONTEXT = {
  phase: "offseason",
  label: "Offseason",
  message: "No live weekly slate is available right now.",
  current_season: new Date().getFullYear(),
  display_week: null,
  games_in_next_window: 0,
  next_kickoff: null,
  generated_at: new Date().toISOString(),
};

const toNumberOrNull = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const normalizeTeamCode = (value) =>
  (value ?? "").toString().trim().toUpperCase();

/**
 * Normalize schedule rows to a consistent shape so downstream components
 * can avoid defensive checks on every render.
 */
function normalizeSchedule(rows) {
  if (!Array.isArray(rows)) return [];
  return rows.map((game) => {
    const home = normalizeTeamCode(game?.home_abbr || game?.home_team);
    const away = normalizeTeamCode(game?.away_abbr || game?.away_team);
    const season = toNumberOrNull(game?.season);
    const week = toNumberOrNull(game?.week);
    const gameId = buildGameKey({
      ...game,
      season,
      week,
      home_abbr: home,
      away_abbr: away,
    });

    return {
      ...game,
      season: season ?? game?.season,
      week: week ?? game?.week,
      home_abbr: home || game?.home_abbr,
      away_abbr: away || game?.away_abbr,
      home_team: game?.home_team || home,
      away_team: game?.away_team || away,
      home_name: game?.home_name || game?.home_team || home,
      away_name: game?.away_name || game?.away_team || away,
      game_id: gameId,
    };
  });
}

function applyTeamMeta(rows, teamMeta) {
  if (!Array.isArray(rows)) return [];
  if (!teamMeta || typeof teamMeta !== "object") return rows;

  return rows.map((game) => {
    if (!game) return game;

    const homeCode = normalizeTeamCode(game?.home_abbr || game?.home_team);
    const awayCode = normalizeTeamCode(game?.away_abbr || game?.away_team);
    const homeMeta = homeCode ? teamMeta[homeCode] : null;
    const awayMeta = awayCode ? teamMeta[awayCode] : null;

    if (!homeMeta && !awayMeta) return game;

    const next = { ...game };

    const applyMeta = (side, meta, code) => {
      if (!meta) return;
      if (!next[`${side}_logo`] && meta.logoUrl) next[`${side}_logo`] = meta.logoUrl;
      if (
        meta.name &&
        (!next[`${side}_name`] || next[`${side}_name`] === next[`${side}_team`] || next[`${side}_name`] === code)
      ) {
        next[`${side}_name`] = meta.name;
      }
      if (!next[`${side}_color`] && meta.primaryColor) next[`${side}_color`] = meta.primaryColor;
      if (!next[`${side}_color2`] && meta.secondaryColor) next[`${side}_color2`] = meta.secondaryColor;
      if (!next[`${side}_wordmark`] && meta.wordmark) next[`${side}_wordmark`] = meta.wordmark;
    };

    applyMeta("home", homeMeta, homeCode);
    applyMeta("away", awayMeta, awayCode);

    return next;
  });
}

/**
 * Normalize history entries to a flat shape with all required fields.
 * Handles multiple backend response formats gracefully.
 */
function ensureHistoryEntry(entry) {
  if (!entry || typeof entry !== "object") return entry;

  // Extract nested prediction if present
  const pred = entry.prediction && typeof entry.prediction === "object" ? entry.prediction : entry;
  const game = entry.game || entry.request || {};

  // Helper to pick first defined value
  const pick = (...vals) => vals.find((v) => v != null);

  // Core fields with fallback chain
  const base = {
    ...pred,
    ts: pick(entry.ts, pred.ts, entry.timestamp) || new Date().toISOString(),
    season: toNumberOrNull(pick(pred.season, game.season, entry.season)),
    week: toNumberOrNull(pick(pred.week, game.week, entry.week)),
    home_team: normalizeTeamCode(pick(pred.home_team, game.home_team, game.home_abbr)),
    away_team: normalizeTeamCode(pick(pred.away_team, game.away_team, game.away_abbr)),
    home_score: pick(pred.home_score, pred.scores?.home_score, pred.metrics?.home_score),
    away_score: pick(pred.away_score, pred.scores?.away_score, pred.metrics?.away_score),
    point_diff: pick(pred.point_diff, pred.metrics?.point_diff),
    home_win_probability: pick(pred.home_win_probability, pred.winner?.proba_home, pred.probs?.home),
    away_win_probability: pick(pred.away_win_probability, pred.winner?.proba_away, pred.probs?.away),
  };

  // Build game_id if missing
  const canBuildKey = base.home_team && base.away_team && base.season != null && base.week != null;
  base.game_id = pred.game_id || (canBuildKey ? buildGameKey(base) : "");

  return base;
}

export function usePredictionState() {
  const [schedule, setSchedule] = useState([]);
  const [week, setWeek] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [history, setHistory] = useState(() => {
    const stored = loadPredictionHistoryFromStorage(PREDICTION_HISTORY_KEY);
    return Array.isArray(stored) ? stored.map(ensureHistoryEntry) : [];
  });
  const [current, setCurrent] = useState(null);
  const [currentKey, setCurrentKey] = useState("");
  const [health, setHealth] = useState(INITIAL_HEALTH);
  const [seasonContext, setSeasonContext] = useState(INITIAL_SEASON_CONTEXT);
  const [loadingByKey, setLoadingByKey] = useState({});
  const [errorsByKey, setErrorsByKey] = useState({});

  // 1. Initial Load: Schedule & History
  useEffect(() => {
    let active = true;

    const init = async () => {
      const [scheduleRes, historyRes, logosRes, seasonContextRes] = await Promise.allSettled([
        getNextWeekSchedule(),
        getPredictionHistory(MAX_HISTORY_ENTRIES),
        getTeamLogos(),
        getSeasonContext(),
      ]);

      if (!active) return;

      const scheduleRows = scheduleRes.status === "fulfilled" ? scheduleRes.value : [];
      const normalized = normalizeSchedule(scheduleRows);
      const teamMeta =
        logosRes.status === "fulfilled" && logosRes.value && typeof logosRes.value === "object"
          ? logosRes.value
          : {};
      const enriched = applyTeamMeta(normalized, teamMeta);
      setSchedule(enriched);
      const nextSeasonContext =
        seasonContextRes.status === "fulfilled" && seasonContextRes.value
          ? seasonContextRes.value
          : INITIAL_SEASON_CONTEXT;
      setSeasonContext(nextSeasonContext);
      setWeek(toNumberOrNull(enriched?.[0]?.week) ?? toNumberOrNull(nextSeasonContext?.display_week));

      if (historyRes.status === "fulfilled") {
        const entries = Array.isArray(historyRes.value?.entries)
          ? historyRes.value.entries
          : [];
        setHistory(entries.map(ensureHistoryEntry));
      }
    };

    init();
    return () => {
      active = false;
    };
  }, []);

  // 2. Health Polling
  useEffect(() => {
    const poll = async () => {
      try {
        const h = await fetchHealth();
        setHealth(h);
      } catch (err) {
        const message = err?.message || "fetch failed";
        setHealth({ status: "error", reason: message });
      }
    };
    poll();
    const id = setInterval(poll, HEALTH_POLL_MS);
    return () => clearInterval(id);
  }, []);

  // 3. History persistence (best-effort)
  useEffect(() => {
    try {
      const trimmed = history.slice(0, MAX_HISTORY_ENTRIES);
      localStorage.setItem(PREDICTION_HISTORY_KEY, JSON.stringify(trimmed));
    } catch (err) {
      console.warn("History persistence failed", err);
    }
  }, [history]);

  const setLoading = useCallback((key, value) => {
    if (!key) return;
    setLoadingByKey((prev) => ({ ...prev, [key]: Boolean(value) }));
  }, []);

  const setError = useCallback((key, message) => {
    if (!key) return;
    setErrorsByKey((prev) => {
      const next = { ...prev };
      if (message == null || message === "") {
        delete next[key];
      } else {
        next[key] = message;
      }
      return next;
    });
  }, []);

  const setPrediction = useCallback(
    (key, entry) => {
      if (!key) return;
      setPredictions((prev) => {
        const next = { ...prev };
        if (entry == null) {
          delete next[key];
        } else {
          next[key] = entry;
        }
        return next;
      });
      if (entry != null) {
        setCurrent(entry);
        setCurrentKey(key);
      } else if (currentKey === key) {
        setCurrent(null);
        setCurrentKey("");
      }
    },
    [currentKey]
  );

  const pushHistory = useCallback((entry) => {
    if (!entry) return;
    const normalized = ensureHistoryEntry(entry);
    const entryKey = entry?.game_id || buildGameKey(entry?.game || entry);
    setHistory((prev) => [normalized, ...prev].slice(0, MAX_HISTORY_ENTRIES));
    setCurrent(entry);
    if (entryKey) setCurrentKey(entryKey);
  }, []);

  const resetHistory = useCallback(() => {
    setHistory([]);
    setCurrent(null);
    setCurrentKey("");
  }, []);

  return {
    schedule,
    week,
    predictions,
    loading: loadingByKey,
    errors: errorsByKey,
    current,
    history,
    health,
    seasonContext,
    setPrediction,
    setLoading,
    setError,
    pushHistory,
    resetHistory,
    count: history.length,
  };
}
