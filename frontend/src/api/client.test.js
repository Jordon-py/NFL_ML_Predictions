import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function jsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

function textResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      throw new Error("not json");
    },
    text: async () => body,
  };
}

describe("client compatibility fallbacks", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("builds history summary from history entries when /history/summary is unavailable", async () => {
    const fetchMock = vi.fn(async (url) => {
      const target = String(url);
      if (target.includes("/history/summary")) {
        return jsonResponse({ detail: "not found" }, 404);
      }
      if (target.includes("/history?limit=250")) {
        return jsonResponse({
          entries: [
            {
              ts: "2026-03-20T00:00:00.000Z",
              season: 2025,
              week: 1,
              home_team: "BUF",
              away_team: "KC",
              home_score: 24,
              away_score: 20,
              home_win_probability: 0.68,
              away_win_probability: 0.32,
              final_home_score: 27,
              final_away_score: 20,
            },
          ],
          total: 1,
        });
      }
      throw new Error(`Unexpected fetch: ${target}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    const { getHistorySummary } = await import("./client.js");

    const summary = await getHistorySummary("tester");
    const summaryAgain = await getHistorySummary("tester");

    expect(summary.total_predictions).toBe(1);
    expect(summary.resolved_games).toBe(1);
    expect(summary.win_rate).toBe(1);
    expect(summary.avg_abs_spread_error).toBe(3);
    expect(summary.avg_confidence).toBe(0.68);
    expect(summaryAgain.total_predictions).toBe(1);

    const urls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(urls.filter((url) => url.includes("/history/summary"))).toHaveLength(1);
    expect(urls.filter((url) => url.includes("/history?limit=250"))).toHaveLength(2);
  });

  it("loads a requested slate from bundled schedule CSV when queried /schedule is unavailable", async () => {
    const fetchMock = vi.fn(async (url) => {
      const target = String(url);
      if (target.includes("/schedule?season=2024&week=15")) {
        return jsonResponse({ detail: "not found" }, 404);
      }
      if (target.includes("/schedules/Nfl_schedule_2024.csv")) {
        return textResponse([
          "game_id,season,week,gameday,gametime,away_team,home_team,stadium",
          "2024_15_KC_BUF,2024,15,2024-12-15,13:00,KC,BUF,Highmark Stadium",
          "2024_16_PHI_DAL,2024,16,2024-12-22,16:25,PHI,DAL,AT&T Stadium",
        ].join("\n"));
      }
      throw new Error(`Unexpected fetch: ${target}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    const { getScheduleForWeek } = await import("./client.js");

    const rows = await getScheduleForWeek(2024, 15);
    const rowsAgain = await getScheduleForWeek(2024, 15);

    expect(rows).toHaveLength(1);
    expect(rows[0].season).toBe(2024);
    expect(rows[0].week).toBe(15);
    expect(rows[0].away_team).toBe("KC");
    expect(rows[0].home_team).toBe("BUF");
    expect(rows[0].stadium).toBe("Highmark Stadium");
    expect(rows[0].kickoff).toBe("2024-12-15T13:00:00");
    expect(rowsAgain).toHaveLength(1);

    const urls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(urls.filter((url) => url.includes("/schedule?season=2024&week=15"))).toHaveLength(1);
    expect(urls.filter((url) => url.includes("/schedules/Nfl_schedule_2024.csv"))).toHaveLength(1);
  });

  it("hides already-played rows from the default next-slate response", async () => {
    const fetchMock = vi.fn(async (url) => {
      const target = String(url);
      if (target.includes("/schedule/next-week")) {
        return jsonResponse({
          games: [
            {
              season: 2025,
              week: 22,
              home_abbr: "BUF",
              away_abbr: "KC",
              kickoff: "2000-01-01T13:00:00Z",
            },
            {
              season: 2025,
              week: 22,
              home_abbr: "DAL",
              away_abbr: "PHI",
              kickoff: "2000-01-01T16:25:00Z",
            },
          ],
        });
      }
      throw new Error(`Unexpected fetch: ${target}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    const { getNextWeekSchedule } = await import("./client.js");

    const rows = await getNextWeekSchedule();

    expect(rows).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
