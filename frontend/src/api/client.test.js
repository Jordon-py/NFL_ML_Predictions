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

function neverResolvingFetch() {
  return vi.fn((url, init = {}) =>
    new Promise((resolve, reject) => {
      init.signal?.addEventListener(
        "abort",
        () => reject(init.signal.reason || new DOMException("signal is aborted without reason", "AbortError")),
        { once: true },
      );
    }),
  );
}

describe("API base URL resolution", () => {
  it("uses the local backend when a production preview runs on localhost", async () => {
    const { resolveApiBaseUrl } = await import("./client.js");

    expect(
      resolveApiBaseUrl(
        {
          DEV: false,
          VITE_API_DEV: "http://127.0.0.1:8000/",
          VITE_API_BASE_URL: "https://prod.example.com/",
        },
        "localhost",
      ),
    ).toBe("http://127.0.0.1:8000");
  });

  it("can force localhost preview to use the deployed backend", async () => {
    const { resolveApiBaseUrl } = await import("./client.js");

    expect(
      resolveApiBaseUrl(
        {
          DEV: false,
          VITE_API_DEV: "http://127.0.0.1:8000",
          VITE_API_BASE_URL: "https://prod.example.com/",
          VITE_FORCE_PROD_API: "true",
        },
        "localhost",
      ),
    ).toBe("https://prod.example.com");
  });

  it("uses the deployed backend away from localhost", async () => {
    const { resolveApiBaseUrl } = await import("./client.js");

    expect(
      resolveApiBaseUrl(
        {
          DEV: false,
          VITE_API_DEV: "http://127.0.0.1:8000",
          VITE_API_BASE_URL: "https://prod.example.com/",
        },
        "new-nfl-predict.vercel.app",
      ),
    ).toBe("https://prod.example.com");
  });
});

describe("client compatibility fallbacks", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
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

it("retries transient server failures and eventually succeeds", async () => {
  vi.resetModules();

  let calls = 0;
  const fetchMock = vi.fn(async () => {
    calls += 1;
    if (calls === 1) {
      return jsonResponse({ detail: "temporary" }, 503);
    }
    return jsonResponse({ status: "ok" }, 200);
  });

  vi.stubGlobal("fetch", fetchMock);

  const { fetchJson } = await import("./client.js");
  const result = await fetchJson("/health");

  expect(result.status).toBe("ok");
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it("passes fetch options through without leaking client-only controls", async () => {
  vi.resetModules();

  const fetchMock = vi.fn(async () => jsonResponse({ ok: true }, 200));
  vi.stubGlobal("fetch", fetchMock);

  try {
    const { fetchJson } = await import("./client.js");
    const result = await fetchJson("/predict", {
      method: "POST",
      headers: { "X-User-Id": "tester" },
      body: JSON.stringify({ home_team: "KC", away_team: "BUF", season: 2026, week: 1 }),
      retryAttempts: 0,
      timeoutMs: 5000,
    });

    expect(result.ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(init.credentials).toBe("omit");
    expect(init.body).toContain('"home_team":"KC"');
    expect(init.retryAttempts).toBeUndefined();
    expect(init.timeoutMs).toBeUndefined();

    const headers = new Headers(init.headers);
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("x-user-id")).toBe("tester");
  } finally {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  }
});

it("converts request timeout aborts into readable HTTP errors", async () => {
  vi.resetModules();
  vi.useFakeTimers();
  vi.stubGlobal("fetch", neverResolvingFetch());

  try {
    const { fetchJson } = await import("./client.js");
    const request = fetchJson("/health", { timeoutMs: 1000, retryAttempts: 0 });
    const assertion = expect(request).rejects.toMatchObject({
      name: "HttpError",
      status: 408,
      message: "Request timed out after 1s",
    });

    await vi.advanceTimersByTimeAsync(1000);
    await assertion;
  } finally {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  }
});

it("keeps premium explain requests open past the default API timeout", async () => {
  vi.resetModules();
  vi.useFakeTimers();
  const fetchMock = neverResolvingFetch();
  vi.stubGlobal("fetch", fetchMock);

  try {
    const { getPremiumExplanation } = await import("./client.js");
    const request = getPremiumExplanation({
      home_team: "KC",
      away_team: "BUF",
      season: 2026,
      week: 1,
    });
    const settled = vi.fn();
    request.then(settled, settled);

    await vi.advanceTimersByTimeAsync(12000);
    await Promise.resolve();

    expect(settled).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(168000);

    await expect(request).rejects.toMatchObject({
      name: "HttpError",
      status: 408,
      message: "Request timed out after 180s",
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  } finally {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  }
});
