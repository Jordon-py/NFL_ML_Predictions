/**
 * File: src/hooks/useNextWeekSchedule.js
 *
 * File Metrics:
 * - Purpose: Load *context* data (schedule + team logos) with plain React state.
 * - Inputs: none
 * - Outputs: { games, teams, isLoading, error, refresh }
 *
 * Key Concepts:
 * - useEffect runs after render: it’s your “run this once on mount” tool.
 * - useState stores the *latest* results so your UI can render from memory.
 * - We use an `alive` flag to avoid setting state after unmount.
 *
 * Mental Model 🧠
 * - Schedule = “what games exist?” (context feed)
 * - Logos    = “how do we render them nicely?” (context feed)
 * - Prediction = “what will happen?” (cognitive compute) -> call predictGame() on click
 *
 * Tips & Next Steps:
 * - Keep this hook “read-only” (no predictions here). Predictions should be user-triggered.
 */

import { useCallback, useEffect, useState } from "react";
import { getNextWeekSchedule, getTeamLogos } from "../api/client";

export function useNextWeekSchedule() {
  const [games, setGames] = useState([]);
  const [teams, setTeams] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Run both requests in parallel (faster than sequential)
      const [g, t] = await Promise.all([getNextWeekSchedule(), getTeamLogos()]);
      setGames(Array.isArray(g) ? g : []);
      setTeams(t && typeof t === "object" ? t : {});
    } catch (err) {
      setError(err?.message ?? "Failed to load schedule/logos");
      setGames([]);
      setTeams({});
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let alive = true;

    (async () => {
      // we call load(), but only commit state if component is still mounted
      try {
        setIsLoading(true);
        setError(null);

        const [g, t] = await Promise.all([getNextWeekSchedule(), getTeamLogos()]);
        if (!alive) return;

        setGames(Array.isArray(g) ? g : []);
        setTeams(t && typeof t === "object" ? t : {});
      } catch (err) {
        if (!alive) return;
        setError(err?.message ?? "Failed to load schedule/logos");
        setGames([]);
        setTeams({});
      } finally {
        if (!alive) return;
        setIsLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, []);

  return { games, teams, isLoading, error, refresh: load };
}
