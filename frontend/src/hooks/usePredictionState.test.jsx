import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const clientMocks = vi.hoisted(() => ({
  mockGetLatestArchivedSchedule: vi.fn(async () => ([])),
  mockGetNextWeekSchedule: vi.fn(async () => ([
    { season: 2025, week: 1, home_abbr: "buf", away_abbr: "kc", home_team: "BUF", away_team: "KC" },
    { season: 2025, week: 1, home_abbr: "BUF", away_abbr: "KC", home_team: "BUF", away_team: "KC" },
  ])),
  mockGetPredictionHistory: vi.fn(async () => ({
    entries: [
      {
        ts: "2026-03-20T00:00:00.000Z",
        season: 2025,
        week: 1,
        home_team: "BUF",
        away_team: "KC",
        home_score: 24,
        away_score: 21,
        home_win_probability: 0.64,
        final_home_score: 27,
        final_away_score: 20,
        game_status: "final",
        score_updated_at: "2026-03-21T00:00:00.000Z",
      },
    ],
    total: 1,
  })),
  mockGetHistorySummary: vi.fn(async () => ({
    total_predictions: 1,
    resolved_games: 1,
    win_rate: 1,
    avg_abs_spread_error: 3,
    avg_confidence: 0.64,
    latest_prediction_at: "2026-03-20T00:00:00.000Z",
    last_score_sync_at: "2026-03-21T00:00:00.000Z",
  })),
  mockGetScheduleForWeek: vi.fn(async (_season, _week, { fallbackRows } = {}) => fallbackRows ?? []),
  mockGetTeamLogos: vi.fn(async () => ({
    BUF: { name: "Buffalo Bills", logoUrl: "https://example.com/buf.png", primaryColor: "#00338D" },
    KC: { name: "Kansas City Chiefs", logoUrl: "https://example.com/kc.png", primaryColor: "#E31837" },
  })),
  mockGetSeasonContext: vi.fn(async () => ({
    phase: "in_season",
    label: "Week 1",
    message: "Upcoming games are ready for forecasting.",
    current_season: 2025,
    display_week: 1,
    games_in_next_window: 1,
    next_kickoff: null,
    generated_at: "2026-03-20T00:00:00.000Z",
  })),
  mockGetHealthStatus: vi.fn(async () => ({ status: "ok", mode: "ready" })),
}));

vi.mock("../api/client.js", () => ({
  getNextWeekSchedule: clientMocks.mockGetNextWeekSchedule,
  getHistorySummary: clientMocks.mockGetHistorySummary,
  getLatestArchivedSchedule: clientMocks.mockGetLatestArchivedSchedule,
  getHealthStatus: clientMocks.mockGetHealthStatus,
  getPredictionHistory: clientMocks.mockGetPredictionHistory,
  getScheduleForWeek: clientMocks.mockGetScheduleForWeek,
  getTeamLogos: clientMocks.mockGetTeamLogos,
  getSeasonContext: clientMocks.mockGetSeasonContext,
}));

import { usePredictionState } from "./usePredictionState.js";

describe("usePredictionState", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("loads deduped schedule rows, team metadata, history, and week-specific reloads", async () => {
    const { result } = renderHook(() => usePredictionState({ userId: "tester" }));

    await waitFor(() => expect(result.current.schedule).toHaveLength(1));

    expect(result.current.schedule[0].home_logo).toBe("https://example.com/buf.png");
    expect(result.current.schedule[0].away_logo).toBe("https://example.com/kc.png");
    expect(result.current.week).toBe(1);
    expect(result.current.seasonContext.current_season).toBe(2025);
    expect(result.current.history).toHaveLength(1);
    expect(result.current.historySummary.total_predictions).toBe(1);
    expect(result.current.history[0].game_id).toBe("2025-1-BUF-KC");
    expect(result.current.history[0].final_home_score).toBe(27);
    expect(result.current.history[0].final_away_score).toBe(20);
    expect(result.current.history[0].game_status).toBe("final");
    expect(result.current.history[0].score_updated_at).toBe("2026-03-21T00:00:00.000Z");

    await act(async () => {
      await result.current.loadScheduleForWeek(2025, 1);
    });

    expect(clientMocks.mockGetScheduleForWeek).toHaveBeenCalledWith(
      2025,
      1,
      expect.objectContaining({
        fallbackRows: expect.any(Array),
      })
    );
  });

  it("falls back to the latest archived slate when no live next slate is available", async () => {
    clientMocks.mockGetNextWeekSchedule.mockResolvedValueOnce([]);
    clientMocks.mockGetSeasonContext.mockResolvedValueOnce({
      phase: "offseason",
      label: "Offseason",
      message: "No live weekly slate is available right now.",
      current_season: 2026,
      display_week: null,
      games_in_next_window: 0,
      next_kickoff: null,
      generated_at: "2026-03-20T00:00:00.000Z",
    });
    clientMocks.mockGetLatestArchivedSchedule.mockResolvedValueOnce([
      {
        season: 2025,
        week: 22,
        home_abbr: "NE",
        away_abbr: "SEA",
        home_team: "NE",
        away_team: "SEA",
        kickoff: "2025-02-08T18:30:00Z",
      },
    ]);

    const { result } = renderHook(() => usePredictionState({ userId: "tester" }));

    await waitFor(() => expect(result.current.schedule).toHaveLength(1));

    expect(result.current.week).toBe(22);
    expect(result.current.schedule[0].home_team).toBe("NE");
    expect(result.current.seasonContext.archive_fallback).toBe(true);
    expect(result.current.seasonContext.current_season).toBe(2025);
    expect(result.current.seasonContext.display_week).toBe(22);
  });
});
