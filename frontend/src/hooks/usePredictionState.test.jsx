/**
 * File: frontend/src/hooks/usePredictionState.test.jsx
 *
 * What it does:
 *   Verifies the shared prediction-state hook hydrates schedule, health, history,
 *   and offseason schedule routing without duplicating dashboard state.
 *
 * Data shapes:
 *   Mocked client calls return schedule row arrays, history `{ entries, total }`
 *   objects, and backend-shaped summary/status payloads.
 *
 * Syntax notes:
 *   Vitest hoisted mocks replace `../api/client.js` before the hook import.
 *
 * Important tests (line numbers last refreshed 2026-04-30):
 *   - initializes shared state: around line 87
 *   - loadScheduleForWeek: around line 114
 *   - offseason explicit schedule: around line 124
 *
 * Possible bugs:
 *   Empty offseason schedule responses can still look like valid season context.
 *
 * Enhancement ideas:
 *   Add a regression for archived-slate fallback context, and move fixture rows
 *   into a small test data builder.
 */

import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const localStorageMock = (() => {
  const store = new Map();
  return {
    getItem: vi.fn((key) => (store.has(String(key)) ? store.get(String(key)) : null)),
    setItem: vi.fn((key, value) => {
      store.set(String(key), String(value));
    }),
    removeItem: vi.fn((key) => {
      store.delete(String(key));
    }),
    clear: vi.fn(() => {
      store.clear();
    }),
  };
})();

vi.stubGlobal("localStorage", localStorageMock);

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
  mockGetOffseasonStatus: vi.fn(async () => ({
    offseason_mode: false,
    current_season: 2025,
    current_week: 1,
    next_known_schedule_date: null,
    days_until_next_game: null,
    data_freshness_seconds: null,
    dataset_hash: null,
    last_trained_at: null,
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
  getHealthStatus: clientMocks.mockGetHealthStatus,
  getOffseasonStatus: clientMocks.mockGetOffseasonStatus,
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

    expect(clientMocks.mockGetScheduleForWeek).toHaveBeenCalledWith(2025, 1);
  });

  it("requests the next season explicitly when offseason mode is active", async () => {
    clientMocks.mockGetOffseasonStatus.mockResolvedValueOnce({
      offseason_mode: true,
      current_season: 2026,
      current_week: 1,
      next_known_schedule_date: null,
      days_until_next_game: null,
      data_freshness_seconds: null,
      dataset_hash: null,
      last_trained_at: null,
    });

    clientMocks.mockGetScheduleForWeek.mockResolvedValueOnce([
      {
        season: 2026,
        week: 1,
        home_abbr: "BUF",
        away_abbr: "MIA",
        home_team: "BUF",
        away_team: "MIA",
      },
    ]);

    const { result } = renderHook(() => usePredictionState({ userId: "tester" }));

    await waitFor(() => expect(result.current.schedule).toHaveLength(1));

    expect(result.current.schedule).toHaveLength(1);
    expect(result.current.seasonContext.phase).toBe("offseason");
    expect(result.current.seasonContext.current_season).toBe(2026);
    expect(result.current.seasonContext.display_week).toBe(1);
    expect(clientMocks.mockGetScheduleForWeek).toHaveBeenCalledWith(2026, 1);
  });
});
