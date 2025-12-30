// Centralized prediction state for the app (no React context).
// Owns schedule/health polling, local history persistence, and team metadata.

import { useEffect, useMemo, useState } from "react";
import { getNextWeekSchedule, getStatusOverview, getHealthStatus as fetchHealth } from "../api/client.js";
import {
  MAX_HISTORY_ENTRIES,
  PREDICTION_HISTORY_KEY,
  buildGameKey,
  loadPredictionHistoryFromStorage,
  parseTeamsCsv,
} from "../utils/predictionContextUtils.js";

const DEFAULT_HEALTH = { status: "unknown", mode: "none", reason: "init" };

export function usePredictionState() {
  const initialHistory = useMemo(() => loadPredictionHistoryFromStorage(), []);

  const [current, setCurrent] = useState(null);
  const [history, setHistory] = useState(initialHistory);
  const [schedule, setSchedule] = useState([]);
  const [week, setWeek] = useState(1);
  const [teams, setTeams] = useState({});
  const [predictions, setPredictions] = useState({});
  const [loading, setLoadingMap] = useState({});
  const [errors, setErrors] = useState({});
  const [healthState, setHealth] = useState(DEFAULT_HEALTH);

  const setScheduleState = (nextSchedule, nextWeek) => {
    setSchedule(Array.isArray(nextSchedule) ? nextSchedule : []);
    setWeek(Number.isFinite(Number(nextWeek)) ? Number(nextWeek) : 1);
  };

  const setPrediction = (key, prediction) => {
    if (!key) return;
    setPredictions((prev) => ({ ...prev, [key]: prediction }));
    setCurrent(prediction ?? null);
  };

  const setLoading = (key, isLoading) => {
    if (!key) return;
    setLoadingMap((prev) => ({ ...prev, [key]: Boolean(isLoading) }));
  };

  const setError = (key, error) => {
    if (!key) return;
    setErrors((prev) => ({ ...prev, [key]: error }));
  };

  const pushHistory = (entry) => {
    if (!entry) return;
    setHistory((prev) => [entry, ...(prev || [])].slice(0, MAX_HISTORY_ENTRIES));
  };

  const resetHistory = () => {
    setHistory([]);
  };

  // Fetch schedule on mount.
  useEffect(() => {
    let mounted = true;

    const fetchSchedule = async () => {
      try {
        const response = await getNextWeekSchedule();
        if (!mounted) return;

        const scheduleData = Array.isArray(response) ? response : [];
        if (!scheduleData.length) return;

        const firstGame = scheduleData[0];
        const rawWeek = firstGame?.week ?? firstGame?.week_num ?? firstGame?.weekNum;
        const nextWeek = Number.isFinite(Number(rawWeek)) ? Number(rawWeek) : 1;

        setScheduleState(scheduleData, nextWeek);
      } catch {
        if (mounted) setScheduleState([], 1);
      }
    };

    fetchSchedule();
    return () => {
      mounted = false;
    };
  }, []);

  // Poll health so UI can gate prediction attempts until backend ready.
  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const h = await fetchHealth();
        if (active && h && h.status) setHealth(h);
      } catch {
        if (active) setHealth({ status: "unhealthy", mode: "none", reason: "health fetch failed" });
      }
    };

    poll();
    const id = setInterval(poll, 15000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  // Seed predictions from stored history once.
  useEffect(() => {
    if (!initialHistory || initialHistory.length === 0) return;

    const trimmed = initialHistory.slice(0, MAX_HISTORY_ENTRIES);
    setHistory(trimmed);

    const seeded = {};
    trimmed.forEach((entry) => {
      const key = buildGameKey(entry);
      if (key) seeded[key] = entry;
    });
    if (Object.keys(seeded).length) {
      setPredictions((prev) => ({ ...prev, ...seeded }));
    }
  }, [initialHistory]);

  // Load team metadata (names + logo URLs) from public CSV once on mount.
  useEffect(() => {
    let active = true;

    const loadTeams = async () => {
      try {
        const res = await fetch("myteamdescriptions.csv");
        if (!res.ok) return;
        const text = await res.text();
        if (!active) return;
        const teamsMap = parseTeamsCsv(text);
        if (teamsMap && Object.keys(teamsMap).length) {
          setTeams(teamsMap);
        }
      } catch {
        // Silent: logos are a nice-to-have.
      }
    };

    loadTeams();
    return () => {
      active = false;
    };
  }, []);

  // Persist history to localStorage.
  useEffect(() => {
    try {
      localStorage.setItem(PREDICTION_HISTORY_KEY, JSON.stringify(history));
    } catch {}
  }, [history]);

  const count = history.length;
  const latest = history[0] ?? null;

  return {
    current,
    history,
    schedule,
    week,
    teams,
    predictions,
    loading,
    errors,
    health: healthState,
    setCurrent,
    pushHistory,
    resetHistory,
    setPrediction,
    setLoading,
    setError,
    setHealth,
    count,
    latest,
  };
}
