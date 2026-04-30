/**
 * Central route-level state for the protected app shell.
 *
 * The hook owns the shared schedule, prediction maps, history, summary, logos,
 * health, and offseason slate context used by Dashboard and HistoryPage so
 * those screens cannot drift apart by maintaining duplicate copies.
 *
 * Data shapes:
 * - Schedule state is an array of normalized game rows keyed by season/week/team.
 * - History state is an array of flattened prediction records plus summary metrics.
 *
 * Important functions (line numbers last refreshed 2026-04-30):
 * - usePredictionState: around line 224
 * - refreshHistory: around line 258
 * - loadScheduleForWeek: around line 361
 *
 * Possible bugs:
 * - If /offseason/status points at a season/week with no rows, the dashboard can
 *   show a valid context with an empty slate.
 *
 * Enhancement ideas:
 * - Move season-context derivation into a small tested adapter.
 * - Let StatsPage consume this same hook instead of fetching separately.
 */

import { useEffect, useState, useCallback } from "react";
import {
  getHistorySummary,
  getOffseasonStatus,
  getNextWeekSchedule,
  getHealthStatus as fetchHealth,
  getPredictionHistory,
  getScheduleForWeek,
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
const INITIAL_HISTORY_SUMMARY = {
  total_predictions: 0,
  resolved_games: 0,
  win_rate: null,
  avg_abs_spread_error: null,
  avg_confidence: null,
  latest_prediction_at: null,
  last_score_sync_at: null,
};

const toNumberOrNull = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const normalizeTeamCode = (value) =>
  (value ?? "").toString().trim().toUpperCase();

function dedupeGamesByKey(rows) {
  if (!Array.isArray(rows)) return [];
  const seen = new Set();
  const out = [];
  for (const row of rows) {
    const key = buildGameKey(row);
    if (key && seen.has(key)) continue;
    if (key) seen.add(key);
    out.push(row);
  }
  return out;
}

/**
 * Normalize schedule rows to a consistent shape so downstream components
 * can avoid defensive checks on every render.
 */
function normalizeSchedule(rows) {
  if (!Array.isArray(rows)) return [];
  const normalizedRows = rows.map((game) => {
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
  return dedupeGamesByKey(normalizedRows);
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
    final_home_score: toNumberOrNull(
      pick(entry.final_home_score, pred.final_home_score, entry.actual_home_score, pred.actual_home_score)
    ),
    final_away_score: toNumberOrNull(
      pick(entry.final_away_score, pred.final_away_score, entry.actual_away_score, pred.actual_away_score)
    ),
    actual_home_score: toNumberOrNull(
      pick(entry.actual_home_score, pred.actual_home_score, entry.final_home_score, pred.final_home_score)
    ),
    actual_away_score: toNumberOrNull(
      pick(entry.actual_away_score, pred.actual_away_score, entry.final_away_score, pred.final_away_score)
    ),
    game_status: pick(entry.game_status, pred.game_status, entry.status, pred.status) || null,
    score_updated_at: pick(
      entry.score_updated_at,
      pred.score_updated_at,
      entry.last_score_sync_at,
      pred.last_score_sync_at
    ) || null,
  };

  // Build game_id if missing
  const canBuildKey = base.home_team && base.away_team && base.season != null && base.week != null;
  base.game_id = pred.game_id || (canBuildKey ? buildGameKey(base) : "");

  return base;
}

function buildOffseasonStatusContext(status = null) {
  const currentSeason = toNumberOrNull(status?.current_season) ?? new Date().getFullYear();
  const currentWeek = toNumberOrNull(status?.current_week) ?? 1;

  return {
    phase: "offseason",
    label: "Offseason",
    message: "No live weekly slate is available right now. Showing the next season when it is available.",
    current_season: currentSeason,
    display_week: currentWeek,
    games_in_next_window: 0,
    next_kickoff: status?.next_known_schedule_date || null,
    generated_at: status?.generated_at || new Date().toISOString(),
    offseason_mode: true,
  };
}

export function usePredictionState(authSession = null) {
  const userId = authSession?.userId || "anonymous";
  const historyStorageKey = `${PREDICTION_HISTORY_KEY}:${userId}`;
  const [schedule, setSchedule] = useState([]);
  const [week, setWeek] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [history, setHistory] = useState(() => {
    const stored = loadPredictionHistoryFromStorage(historyStorageKey);
    return Array.isArray(stored) ? stored.map(ensureHistoryEntry) : [];
  });
  const [historySummary, setHistorySummary] = useState(INITIAL_HISTORY_SUMMARY);
  const [current, setCurrent] = useState(null);
  const [currentKey, setCurrentKey] = useState("");
  const [health, setHealth] = useState(INITIAL_HEALTH);
  const [seasonContext, setSeasonContext] = useState(INITIAL_SEASON_CONTEXT);
  const [scheduleLoading, setScheduleLoading] = useState(true);
  const [scheduleError, setScheduleError] = useState(null);
  const [loadingByKey, setLoadingByKey] = useState({});
  const [errorsByKey, setErrorsByKey] = useState({});
  const [teamMeta, setTeamMeta] = useState({});

  useEffect(() => {
    const stored = loadPredictionHistoryFromStorage(historyStorageKey);
    setHistory(Array.isArray(stored) ? stored.map(ensureHistoryEntry) : []);
    setPredictions({});
    setErrorsByKey({});
    setLoadingByKey({});
    setCurrent(null);
    setCurrentKey("");
    setHistorySummary(INITIAL_HISTORY_SUMMARY);
    setScheduleError(null);
  }, [historyStorageKey]);

  const refreshHistory = useCallback(
    async (limit = MAX_HISTORY_ENTRIES) => {
      const [historyRes, summaryRes] = await Promise.all([
        getPredictionHistory(limit, userId),
        getHistorySummary(userId),
      ]);

      const entries = Array.isArray(historyRes?.entries) ? historyRes.entries : [];
      setHistory(entries.map(ensureHistoryEntry));
      setHistorySummary(summaryRes || INITIAL_HISTORY_SUMMARY);
      return entries;
    },
    [userId]
  );

  // 1. Initial Load: Schedule & History
  useEffect(() => {
    let active = true;

    const init = async () => {
      setScheduleLoading(true);
      setScheduleError(null);
      const offseasonRes = await getOffseasonStatus();
      const offseasonMode = Boolean(offseasonRes?.offseason_mode);
      const offseasonSeason = toNumberOrNull(offseasonRes?.current_season) ?? new Date().getFullYear();
      const offseasonWeek = toNumberOrNull(offseasonRes?.current_week) ?? 1;

      const [scheduleRes, historyRes, logosRes, summaryRes] = await Promise.allSettled([
        offseasonMode ? getScheduleForWeek(offseasonSeason, offseasonWeek) : getNextWeekSchedule(),
        getPredictionHistory(MAX_HISTORY_ENTRIES, userId),
        getTeamLogos(),
        getHistorySummary(userId),
      ]);

      if (!active) return;

      let scheduleRows = scheduleRes.status === "fulfilled" ? scheduleRes.value : [];
      let nextSeasonContext = offseasonMode
        ? buildOffseasonStatusContext(offseasonRes)
        : await getSeasonContext(scheduleRows);

      const normalized = normalizeSchedule(scheduleRows);
      const teamMeta =
        logosRes.status === "fulfilled" && logosRes.value && typeof logosRes.value === "object"
          ? logosRes.value
          : {};
      const enriched = applyTeamMeta(normalized, teamMeta);
      setSchedule(enriched);
      setSeasonContext(nextSeasonContext);
      setWeek(toNumberOrNull(enriched?.[0]?.week) ?? toNumberOrNull(nextSeasonContext?.display_week));
      setTeamMeta(teamMeta);
      setScheduleError(
        scheduleRes.status === "rejected" ? scheduleRes.reason?.message ?? "Failed to load schedule" : null
      );

      if (historyRes.status === "fulfilled") {
        const entries = Array.isArray(historyRes.value?.entries)
          ? historyRes.value.entries
          : [];
        setHistory(entries.map(ensureHistoryEntry));
      }
      if (summaryRes.status === "fulfilled") {
        setHistorySummary(summaryRes.value || INITIAL_HISTORY_SUMMARY);
      }
      setScheduleLoading(false);
    };

    init();
    return () => {
      active = false;
    };
  }, [userId]);

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
      localStorage.setItem(historyStorageKey, JSON.stringify(trimmed));
    } catch (err) {
      console.warn("History persistence failed", err);
    }
  }, [history, historyStorageKey]);

  const loadScheduleForWeek = useCallback(
    async (seasonOverride, weekOverride) => {
      setScheduleLoading(true);
      setScheduleError(null);
      try {
        const isDefaultSlateRequest = seasonOverride == null && weekOverride == null;
        const offseasonSeason = toNumberOrNull(seasonContext?.current_season);
        const offseasonWeek = toNumberOrNull(seasonContext?.display_week) ?? 1;
        let rows = await getScheduleForWeek(
          isDefaultSlateRequest && seasonContext?.phase === "offseason" && offseasonSeason != null
            ? offseasonSeason
            : seasonOverride,
          isDefaultSlateRequest && seasonContext?.phase === "offseason" && offseasonSeason != null
            ? offseasonWeek
            : weekOverride
        );
        let nextSeasonContext = null;

        if (isDefaultSlateRequest && seasonContext?.phase !== "offseason") {
          nextSeasonContext = await getSeasonContext(rows);
        } else if (isDefaultSlateRequest && seasonContext?.phase === "offseason") {
          nextSeasonContext = buildOffseasonStatusContext(seasonContext);
        }

        const normalized = normalizeSchedule(rows);
        const enriched = applyTeamMeta(normalized, teamMeta);
        const derivedWeek = toNumberOrNull(
          weekOverride ?? normalized?.[0]?.week ?? nextSeasonContext?.display_week
        );
        const derivedSeason =
          toNumberOrNull(seasonOverride ?? normalized?.[0]?.season ?? nextSeasonContext?.current_season) ||
          seasonOverride ||
          seasonContext?.current_season;
        setSchedule(enriched);
        setWeek(derivedWeek);
        if (nextSeasonContext) {
          setSeasonContext(nextSeasonContext);
        } else {
          setSeasonContext((prev) => ({
            ...prev,
            current_season: derivedSeason || prev.current_season,
            display_week: derivedWeek ?? prev.display_week,
          }));
        }
        return enriched;
      } catch (error) {
        setScheduleError(error?.message ?? "Failed to load schedule");
        return Array.isArray(schedule) ? schedule : [];
      } finally {
        setScheduleLoading(false);
      }
    },
    [schedule, seasonContext?.current_season, seasonContext?.display_week, seasonContext?.phase, teamMeta]
  );

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
    const entryKey = normalized?.game_id || buildGameKey(normalized?.game || normalized);
    setHistory((prev) => [normalized, ...prev].slice(0, MAX_HISTORY_ENTRIES));
    setHistorySummary((prev) => ({
      ...prev,
      total_predictions: Number(prev?.total_predictions || 0) + 1,
      latest_prediction_at: normalized?.ts || prev?.latest_prediction_at || null,
    }));
    setCurrent(normalized);
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
    historySummary,
    health,
    seasonContext,
    scheduleLoading,
    scheduleError,
    setPrediction,
    setLoading,
    setError,
    pushHistory,
    refreshHistory,
    resetHistory,
    count: history.length,
    loadScheduleForWeek,
  };
}
