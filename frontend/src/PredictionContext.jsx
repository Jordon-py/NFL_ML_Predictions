/*
File: PredictionContext.jsx
Purpose: Centralized React context for NFL prediction state; manages schedule fetch, prediction requests, loading/error states, and team metadata.
Functions: PredictionProvider (React component), usePredictions (hook), fetchSchedule (API call), makePrediction (action), reducer (state updates), getKey (game identifier)
Variables: schedule (games array), week (current week number), teams (team metadata), predictions (keyed by game), loading (keyed by game), errors (keyed by game), current (latest prediction), history (prediction array)
Interacts With: api/client.js (fetch wrappers), backend /predict and /schedule endpoints, DashBoard/TeamGrid (consumers)

PredictionContext.jsx
--------------------
Component Purpose:
  Provide a shared prediction store (current result + historical list + schedule)
  using React Context + Reducer so all views stay in sync.

Core Logic Overview:
  - `initialState` tracks schedule, predictions, loading/error states, and prediction history.
  - `reducer` responds to explicit action types for schedule loading, predictions, and history.
  - Action creators are memoized callbacks exposed through context.
  - Fetches schedule on mount and manages prediction state per-game.
 *
 * Modification Guide:
 *   - Add new action types inside the reducer, then expose a matching
 *     callback in the provider so components never call `dispatch` directly.
 *   - Extend `history` trimming/deduping here instead of inside components.
 */
import React, {
  createContext, useContext, useMemo,
  useReducer, useCallback, useEffect
} from 'react';
import { getNextWeekSchedule, predictGame, getHealthStatus, getPredictionHistory } from './api/client';

const PREDICTION_HISTORY_KEY = "prediction_history";
const MAX_HISTORY_ENTRIES = 100;

// Generate unique key for a game
function generateGameKey(game) {
  return game?.game_id ?? [game?.season, game?.week, game?.home_abbr, game?.away_abbr].filter(Boolean).join("-");
}

// Action types
const SET_CURRENT = 'SET_CURRENT';
const PUSH_HISTORY = 'PUSH_HISTORY';
const RESET_HISTORY = 'RESET_HISTORY';
const SET_SCHEDULE = 'SET_SCHEDULE';
const SET_PREDICTION = 'SET_PREDICTION';
const SET_LOADING = 'SET_LOADING';
const SET_ERROR = 'SET_ERROR';
const SET_HEALTH = 'SET_HEALTH';
const SET_HISTORY = 'SET_HISTORY';
const SET_TEAMS = 'SET_TEAMS';

const initialState = {
  current: null,
  history: [],
  schedule: [],
  week: 11,
  teams: {},
  predictions: {},
  loading: {},
  errors: {},
  health: { status: 'unknown', mode: 'none', reason: 'init' }
};

function reducer(state, action) {
  switch (action.type) {
    case SET_CURRENT:
      return { ...state, current: action.payload };
    case PUSH_HISTORY:
      return { ...state, history: [action.payload, ...state.history].slice(0, MAX_HISTORY_ENTRIES) };
    case RESET_HISTORY:
      return { ...state, history: [] };
    case SET_SCHEDULE:
      return { ...state, schedule: action.payload.schedule, week: action.payload.week };
    case SET_PREDICTION: {
      const { key, prediction } = action.payload;
      return {
        ...state,
        predictions: { ...state.predictions, [key]: prediction },
        current: prediction
      };
    }
    case SET_LOADING: {
      const { key, loading } = action.payload;
      return { ...state, loading: { ...state.loading, [key]: loading } };
    }
    case SET_ERROR: {
      const { key, error } = action.payload;
      return { ...state, errors: { ...state.errors, [key]: error } };
    }
    case SET_HEALTH: {
      return { ...state, health: action.payload };
    }
    case SET_HISTORY: {
      const incoming = Array.isArray(action.payload) ? action.payload : [];
      return { ...state, history: incoming.slice(0, MAX_HISTORY_ENTRIES) };
    }
    case SET_TEAMS: {
      const next = action.payload && typeof action.payload === 'object' ? action.payload : {};
      return { ...state, teams: { ...state.teams, ...next } };
    }
    default:
      return state;
  }
}

// Safe hydration from localStorage
function loadPredictionHistoryFromStorage() {
  try {
    const rawHistoryData = localStorage.getItem(PREDICTION_HISTORY_KEY);
    const parsedHistory = JSON.parse(rawHistoryData);
    return Array.isArray(parsedHistory) ? parsedHistory : [];
  } catch {
    return [];
  }
}

// Lightweight CSV parser for public/data/myteamdescriptions.csv
// Format: team_name,abbr,logo_url
function parseTeamsCsv(text) {
  if (!text) return {};
  const lines = text.trim().split(/\r?\n/);
  const out = {};
  for (let i = 1; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) continue;
    const parts = line.split(",");
    if (parts.length < 3) continue;
    const [teamName, abbr, logoUrl] = parts;
    const code = (abbr || "").trim();
    if (!code) continue;
    out[code] = {
      name: (teamName || code).trim(),
      logoUrl: (logoUrl || "").trim(),
    };
  }
  return out;
}

const Ctx = createContext(null);

export function PredictionProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState, (s) => ({
    ...s, history: loadPredictionHistoryFromStorage()
  }));

  // Actions
  const setCurrent = useCallback((e) => dispatch({ type: SET_CURRENT, payload: e }), []);
  const pushHistory = useCallback((e) => dispatch({ type: PUSH_HISTORY, payload: e }), []);
  const resetHistory = useCallback(() => dispatch({ type: RESET_HISTORY }), []);

  const setSchedule = useCallback((schedule, week) =>
    dispatch({ type: SET_SCHEDULE, payload: { schedule, week } }), []);

  const setPrediction = useCallback((key, prediction) =>
    dispatch({ type: SET_PREDICTION, payload: { key, prediction } }), []);

  const setLoading = useCallback((key, loading) =>
    dispatch({ type: SET_LOADING, payload: { key, loading } }), []);

  const setError = useCallback((key, error) =>
    dispatch({ type: SET_ERROR, payload: { key, error } }), []);

  const setHealth = useCallback((h) => dispatch({ type: SET_HEALTH, payload: h }), []);
  const setHistoryState = useCallback((entries) => dispatch({ type: SET_HISTORY, payload: entries }), []);
  const setTeams = useCallback((teams) => dispatch({ type: SET_TEAMS, payload: teams }), []);

  // Fetch schedule on mount
  useEffect(() => {
    let mounted = true;
    const fetchSchedule = async () => {
      try {
        const scheduleData = await getNextWeekSchedule();
        if (!mounted || !Array.isArray(scheduleData) || scheduleData.length === 0) return;

        // Extract week from first game
        const week = scheduleData[0]?.week || 11;
        setSchedule(scheduleData, week);

        console.log(`[PredictionContext] Loaded ${scheduleData.length} games for Week ${week}`);
      } catch (err) {
        console.error('[PredictionContext] Failed to fetch schedule:', err);
      }
    };
    fetchSchedule();
    return () => { mounted = false; };
  }, [setSchedule]);

  // Poll health (lightweight) so UI can gate prediction attempts until backend ready
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const h = await getHealthStatus();
        if (active && h && h.status) setHealth(h);
      } catch (e) {
        if (active) setHealth({ status: 'unhealthy', mode: 'none', reason: 'health fetch failed' });
      }
    };
    poll();
    const id = setInterval(poll, 15000); // 15s cadence
    return () => { active = false; clearInterval(id); };
  }, [setHealth]);

  // Hydrate history from backend (falls back to localStorage seed when API unavailable)
  useEffect(() => {
    let active = true;
    const loadHistoryFromBackend = async () => {
      try {
        const payload = await getPredictionHistory(MAX_HISTORY_ENTRIES);
        if (!active || !payload) return;
        const entries = Array.isArray(payload.entries) ? payload.entries : [];
        setHistoryState(entries);
      } catch (err) {
        console.warn('[PredictionContext] History fetch failed, using local cache.', err);
      }
    };
    loadHistoryFromBackend();
    const id = setInterval(loadHistoryFromBackend, 60000);
    return () => { active = false; clearInterval(id); };
  }, [setHistoryState]);

  // Load team metadata (names + logo URLs) from public CSV once on mount.
  useEffect(() => {
    let active = true;
    const loadTeams = async () => {
      try {
        const res = await fetch("/data/myteamdescriptions.csv");
        if (!res.ok) return;
        const text = await res.text();
        if (!active) return;
        const teamsMap = parseTeamsCsv(text);
        if (teamsMap && Object.keys(teamsMap).length) {
          setTeams(teamsMap);
          if (import.meta?.env?.DEV) {
            console.debug("[PredictionContext] Loaded team metadata for", Object.keys(teamsMap).length, "teams");
          }
        }
      } catch (err) {
        console.warn("[PredictionContext] Failed to load team descriptions; logos may be missing.", err);
      }
    };
    loadTeams();
    return () => { active = false; };
  }, [setTeams]);

  // Make a prediction for a game
  const healthStatus = state.health?.status ?? 'unknown';
  const healthReason = state.health?.reason ?? 'health not confirmed';

  const makePrediction = useCallback(async (game) => {
    // Only short-circuit when the backend has explicitly reported an unhealthy state;
    // if health is still `unknown` we attempt the request and let per-call errors surface.
    if (healthStatus === 'unhealthy') {
      console.warn(`[PredictionContext] Backend unhealthy (${healthReason}); skipping prediction request.`);
      return;
    }
    if (healthStatus !== 'healthy') {
      console.info('[PredictionContext] Backend health pending; attempting prediction anyway.');
    }
    const gameKey = generateGameKey(game);
    setLoading(gameKey, true);
    setError(gameKey, null);

    try {
      const prediction = await predictGame({
        homeTeam: game.home_abbr || game.home_team,
        awayTeam: game.away_abbr || game.away_team,
        season: game.season,
        week: game.week
      });

      setPrediction(gameKey, prediction);
      pushHistory({ ...prediction, timestamp: new Date().toISOString(), game });
      console.log(`[PredictionContext] Prediction for ${game.away_abbr}@${game.home_abbr}:`, prediction);
    } catch (err) {
      const errorMsg = err?.message || String(err);
      setError(gameKey, errorMsg);
      console.error(`[PredictionContext] Prediction failed for ${game.away_abbr}@${game.home_abbr}:`, err);
    } finally {
      setLoading(gameKey, false);
    }
  }, [healthStatus, healthReason, setLoading, setError, setPrediction, pushHistory]);

  // Persist history to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(PREDICTION_HISTORY_KEY, JSON.stringify(state.history));
    } catch { }
  }, [state.history]);

  // Tiny dev logger
  useEffect(() => {
    if (typeof window !== "undefined" && import.meta && import.meta.env && import.meta.env.DEV) {
      console.debug("[PredictionContext] state:", state);
    }
  }, [state]);


  // Selectors
  const count = state.history.length;
  const latest = state.history[0] ?? null;

  const value = useMemo(() => ({
    // State
    ...state,
    // Actions
    setCurrent,
    pushHistory,
    resetHistory,
    makePrediction,
    // Direct health setter (rarely needed externally)
    setHealth,
    // Selectors
    count,
    latest,
  }), [state, setCurrent, pushHistory, resetHistory, makePrediction, count, latest]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const usePredictions = () => {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("usePredictions must be used within PredictionProvider");
  return ctx;
};
