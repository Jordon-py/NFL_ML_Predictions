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
} from "../api/client.js";
import {
  buildGameKey,
  loadPredictionHistoryFromStorage,
  MAX_HISTORY_ENTRIES,
  PREDICTION_HISTORY_KEY,
} from "../utils/predictionContextUtils.js";

const INITIAL_HEALTH = { status: "loading", mode: "none" };
const HEALTH_POLL_MS = 15000;

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

/**
 * Ensure history entries always include a timestamp field and a flat shape.
 */
function ensureHistoryEntry(entry) {
  if (!entry || typeof entry !== "object") return entry;
  let base =
    entry.prediction && typeof entry.prediction === "object"
      ? { ...entry.prediction, ts: entry.ts || entry.prediction.ts }
      : { ...entry };
  const fallbackGame = entry.game || entry.request || {};

  if (base.scores && (base.home_score == null || base.away_score == null)) {
    base = {
      ...base,
      home_score: base.home_score ?? base.scores?.home_score,
      away_score: base.away_score ?? base.scores?.away_score,
    };
  }

  if (base.metrics && (base.home_score == null || base.away_score == null)) {
    base = {
      ...base,
      home_score: base.home_score ?? base.metrics?.home_score,
      away_score: base.away_score ?? base.metrics?.away_score,
      point_diff: base.point_diff ?? base.metrics?.point_diff,
    };
  }

  if (base.winner && (base.home_win_probability == null || base.away_win_probability == null)) {
    base = {
      ...base,
      home_win_probability: base.home_win_probability ?? base.winner?.proba_home,
      away_win_probability: base.away_win_probability ?? base.winner?.proba_away,
    };
  }

  if (base.probs && (base.home_win_probability == null || base.away_win_probability == null)) {
    base = {
      ...base,
      home_win_probability: base.home_win_probability ?? base.probs?.home ?? base.probs?.ensemble,
      away_win_probability: base.away_win_probability ?? base.probs?.away,
    };
  }

  // Backfill core game identity fields so downstream UI only reads the flat shape.
  const season = toNumberOrNull(base.season ?? fallbackGame.season);
  const week = toNumberOrNull(base.week ?? fallbackGame.week);
  const home = normalizeTeamCode(base.home_team ?? fallbackGame.home_team ?? fallbackGame.home_abbr);
  const away = normalizeTeamCode(base.away_team ?? fallbackGame.away_team ?? fallbackGame.away_abbr);
  const canBuildKey = Boolean(home && away && season != null && week != null);
  const gameId = base.game_id || (canBuildKey ? buildGameKey({ season, week, home_team: home, away_team: away }) : "");
  base = {
    ...base,
    season: season ?? base.season,
    week: week ?? base.week,
    home_team: home || base.home_team,
    away_team: away || base.away_team,
    game_id: gameId || base.game_id,
  };

  if (base.ts || base.timestamp || base.time) return base;
  return { ...base, ts: new Date().toISOString() };
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
  const [loadingByKey, setLoadingByKey] = useState({});
  const [errorsByKey, setErrorsByKey] = useState({});

  // 1. Initial Load: Schedule & History
  useEffect(() => {
    let active = true;

    const init = async () => {
      const [scheduleRes, historyRes] = await Promise.allSettled([
        getNextWeekSchedule(),
        getPredictionHistory(MAX_HISTORY_ENTRIES),
      ]);

      if (!active) return;

      if (scheduleRes.status === "fulfilled") {
        const normalized = normalizeSchedule(scheduleRes.value);
        setSchedule(normalized);
        const derivedWeek = toNumberOrNull(normalized?.[0]?.week);
        setWeek(derivedWeek);
      } else {
        setSchedule([]);
        setWeek(null);
      }

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
      } catch {
        setHealth({ status: "error", reason: "fetch failed" });
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
    setPrediction,
    setLoading,
    setError,
    pushHistory,
    resetHistory,
    count: history.length,
  };
}
