import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const clientMocks = vi.hoisted(() => ({
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
      },
    ],
    total: 1,
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
    expect(result.current.history[0].game_id).toBe("2025-1-BUF-KC");

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
});
