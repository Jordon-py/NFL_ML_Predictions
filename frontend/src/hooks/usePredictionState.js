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

import { useCallback, useEffect, useState } from "react";
import {
  getHealthStatus as fetchHealth,
  getNextWeekSchedule,
  getPredictionHistory,
  getTeamLogos,
  predictGame,
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

const normalizeConference = (value) => {
  const conf = (value ?? "").toString().trim().toUpperCase();
  return conf === "AFC" || conf === "NFC" ? conf : null;
};

const isPlaceholderTeam = (value) => {
  const token = normalizeTeamCode(value);
  return !token || token === "TBD" || token === "AFC" || token === "NFC";
};

const pickWinnerFromPrediction = (prediction) => {
  const homeProb = prediction?.home_win_probability;
  const awayProb = prediction?.away_win_probability;
  const home = normalizeTeamCode(prediction?.home_team);
  const away = normalizeTeamCode(prediction?.away_team);

  if (!home || !away) return null;
  if (typeof homeProb !== "number" || typeof awayProb !== "number") return null;
  return homeProb >= awayProb ? home : away;
};

/**
 * Normalize schedule rows to a consistent shape so downstream components
 * can avoid defensive checks on every render.
 */
function normalizeSchedule(rows) {
  if (!Array.isArray(rows)) return [];
  const normalized = rows
    .filter(g => g && (g.home_team || g.home_abbr) && (g.away_team || g.away_abbr))
    .map((game) => {
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

  // Guard: schedule CSVs sometimes contain duplicate placeholder rows (e.g. TBD vs TBD).
  const seen = new Map();
  return normalized.map((game) => {
    const baseKey = buildGameKey(game);
    if (!baseKey) return game;
    const nextCount = (seen.get(baseKey) || 0) + 1;
    seen.set(baseKey, nextCount);
    if (nextCount === 1) return game;
    return { ...game, game_id: `${baseKey}#${nextCount}` };
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

function sortScheduleChronologically(rows) {
  if (!Array.isArray(rows)) return [];
  const kickoffTs = (game) => {
    if (!game?.kickoff) return Number.POSITIVE_INFINITY;
    const ts = new Date(game.kickoff).getTime();
    return Number.isFinite(ts) ? ts : Number.POSITIVE_INFINITY;
  };

  return [...rows].sort((a, b) => {
    const weekA = toNumberOrNull(a?.week) ?? Number.POSITIVE_INFINITY;
    const weekB = toNumberOrNull(b?.week) ?? Number.POSITIVE_INFINITY;
    if (weekA !== weekB) return weekA - weekB;
    return kickoffTs(a) - kickoffTs(b);
  });
}

async function projectPostseasonSchedule(rows, teamMeta) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return { schedule: [], predictions: {} };
  }

  const hasPostseason = rows.some((g) => g?.game_type && g.game_type !== "REG");
  if (!hasPostseason) return { schedule: rows, predictions: {} };

  const divisionalGames = rows.filter((g) => g?.game_type === "DIV");
  if (divisionalGames.length === 0) return { schedule: rows, predictions: {} };

  const predictionsByKey = {};

  // 1) Predict divisional winners.
  const divisionalResults = await Promise.allSettled(
    divisionalGames.map(async (game) => {
      const season = toNumberOrNull(game?.season);
      const week = toNumberOrNull(game?.week);
      const home = normalizeTeamCode(game?.home_abbr || game?.home_team);
      const away = normalizeTeamCode(game?.away_abbr || game?.away_team);
      if (season == null || week == null || !home || !away) return null;
      if (isPlaceholderTeam(home) || isPlaceholderTeam(away)) return null;

      const pred = await predictGame(home, away, season, week, { record: false });
      const key = buildGameKey(game);
      if (key) predictionsByKey[key] = pred;
      return { game, prediction: pred };
    })
  );

  const winners = [];
  divisionalResults.forEach((result) => {
    if (result.status !== "fulfilled" || !result.value) return;
    const winner = pickWinnerFromPrediction(result.value.prediction);
    if (!winner) return;
    const conf = normalizeConference(teamMeta?.[winner]?.conference);
    if (!conf) return;
    winners.push({ team: winner, conference: conf });
  });

  const afcWinners = winners.filter((w) => w.conference === "AFC").map((w) => w.team);
  const nfcWinners = winners.filter((w) => w.conference === "NFC").map((w) => w.team);

  if (afcWinners.length !== 2 || nfcWinners.length !== 2) {
    // Not enough data to safely project the bracket.
    return { schedule: rows, predictions: predictionsByKey };
  }

  const confTemplates = rows.filter((g) => g?.game_type === "CON");
  const sbTemplate = rows.find((g) => g?.game_type === "SB") || null;

  const season = toNumberOrNull(divisionalGames[0]?.season) ?? toNumberOrNull(sbTemplate?.season);
  const confWeek = toNumberOrNull(confTemplates[0]?.week) ?? (toNumberOrNull(divisionalGames[0]?.week) ?? 20) + 1;
  const sbWeek = toNumberOrNull(sbTemplate?.week) ?? confWeek + 1;

  const makeProjectedGame = ({ template, week, home, away, gameType }) => {
    const base = template && typeof template === "object" ? template : {};
    const kickoff = base.kickoff || null;
    const next = {
      ...base,
      season,
      week,
      game_type: gameType,
      kickoff,
      home_team: home,
      away_team: away,
      home_abbr: home,
      away_abbr: away,
      game_id: `${season}-${week}-${home}-${away}`,
    };
    return next;
  };

  // 2) Build conference championship matchups from predicted winners.
  const sortedAfc = [...afcWinners].sort();
  const sortedNfc = [...nfcWinners].sort();

  const afcGame = makeProjectedGame({
    template: confTemplates[0] || sbTemplate,
    week: confWeek,
    home: sortedAfc[1],
    away: sortedAfc[0],
    gameType: "CON",
  });
  const nfcGame = makeProjectedGame({
    template: confTemplates[1] || confTemplates[0] || sbTemplate,
    week: confWeek,
    home: sortedNfc[1],
    away: sortedNfc[0],
    gameType: "CON",
  });

  // 3) Predict conference winners, then plug into Super Bowl.
  const confPredResults = await Promise.allSettled(
    [afcGame, nfcGame].map(async (game) => {
      const pred = await predictGame(game.home_team, game.away_team, season, confWeek, { record: false });
      const key = buildGameKey(game);
      if (key) predictionsByKey[key] = pred;
      return { game, prediction: pred };
    })
  );

  const confWinners = confPredResults
    .filter((r) => r.status === "fulfilled" && r.value)
    .map((r) => pickWinnerFromPrediction(r.value.prediction))
    .filter(Boolean);

  const afcWinner = confWinners.find((team) => normalizeConference(teamMeta?.[team]?.conference) === "AFC") || sortedAfc[1];
  const nfcWinner = confWinners.find((team) => normalizeConference(teamMeta?.[team]?.conference) === "NFC") || sortedNfc[1];

  const superBowlGame = makeProjectedGame({
    template: sbTemplate || confTemplates[0] || rows[0],
    week: sbWeek,
    home: nfcWinner,
    away: afcWinner,
    gameType: "SB",
  });

  try {
    const sbPred = await predictGame(superBowlGame.home_team, superBowlGame.away_team, season, sbWeek, { record: false });
    const key = buildGameKey(superBowlGame);
    if (key) predictionsByKey[key] = sbPred;
  } catch {
    // best-effort
  }

  const stripped = rows.filter((g) => g?.game_type !== "CON" && g?.game_type !== "SB");
  const merged = sortScheduleChronologically([...stripped, afcGame, nfcGame, superBowlGame]);
  return { schedule: merged, predictions: predictionsByKey };
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
  const [loadingByKey, setLoadingByKey] = useState({});
  const [errorsByKey, setErrorsByKey] = useState({});

  // 1. Initial Load: Schedule & History
  useEffect(() => {
    let active = true;

    const init = async () => {
      const [scheduleRes, historyRes, logosRes] = await Promise.allSettled([
        getNextWeekSchedule(),
        getPredictionHistory(MAX_HISTORY_ENTRIES),
        getTeamLogos(),
      ]);

      if (!active) return;

      const scheduleRows = scheduleRes.status === "fulfilled" ? scheduleRes.value : [];
      const normalized = normalizeSchedule(scheduleRows);
      const teamMeta =
        logosRes.status === "fulfilled" && logosRes.value && typeof logosRes.value === "object"
          ? logosRes.value
          : {};

      let enriched = applyTeamMeta(normalized, teamMeta);
      let prefilled = {};
      try {
        const projected = await projectPostseasonSchedule(enriched, teamMeta);
        enriched = applyTeamMeta(projected.schedule, teamMeta);
        prefilled = projected.predictions || {};
      } catch (err) {
        console.warn("Postseason projection skipped:", err);
      }

      setSchedule(enriched);
      setWeek(toNumberOrNull(enriched?.[0]?.week));
      if (prefilled && Object.keys(prefilled).length) {
        setPredictions(prefilled);
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
    setPrediction,
    setLoading,
    setError,
    pushHistory,
    resetHistory,
    count: history.length,
  };
}
