/**
 * File: frontend/src/api/client.js
 *
 * Purpose:
 *   One tiny, consistent fetch wrapper for the whole app (schedule + predict + history).
 *
 * The production gotcha (Vercel):
 *   Vite does NOT ship your local `.env` file to Vercel. You must set env vars
 *   in Vercel Project Settings → Environment Variables.
 *
 * Required env:
 *   - Local dev:    VITE_API_BASE_URL=http://127.0.0.1:8000
 *   - Vercel prod:  VITE_API_BASE_URL=https://<your-heroku-app>.herokuapp.com
 *
 * Notes:
 *   - Trailing slashes are stripped so URL joins are predictable.
 *   - Errors are thrown as HttpError(status, url, body) so UI can display useful info.
 */

import { buildPredictPayload as buildGamePredictPayload } from "../utils/gameUtils.js";

export class HttpError extends Error {
  constructor(message, { status, url, body } = {}) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

/**
 * Resolve the API base URL.
 *
 * Why this matters:
 *   - Locally you can hit FastAPI directly.
 *   - On Vercel you must call your Heroku domain (or you'll accidentally call localhost / a relative path).
 *
 * Optional:
 *   - If you intentionally use a Vite proxy in DEV, you can set VITE_API_BASE_URL=""
 *     (empty string) and call "/api/..." paths.
 */
const RAW_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  import.meta.env.VITE_API_BASE ??
  import.meta.env.VITE_API_URL;

// If not set, default to localhost in DEV, but require it in PROD.
const BASE_URL = (
  RAW_BASE_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:8000" : "")
).replace(/\/+$/, "");

export const API_BASE = BASE_URL;

async function safeReadJson(res) {
  try {
    return await res.json();
  } catch {
    return null; // some endpoints (or errors) can return empty bodies
  }
}

function extractArrayPayload(payload, keys = []) {
  if (Array.isArray(payload)) return payload;

  for (const key of keys) {
    if (Array.isArray(payload?.[key])) {
      return payload[key];
    }
  }

  return [];
}

function createStatusOverviewFallback(totalPredictions = 0) {
  return {
    health: { status: "unknown" },
    dataset: { rows: 0 },
    history: { metrics: { total_predictions: totalPredictions } },
  };
}

function normalizeHistoryResponse(payload) {
  if (Array.isArray(payload)) {
    return { entries: payload, total: payload.length };
  }

  if (payload && Array.isArray(payload.entries)) {
    return {
      entries: payload.entries,
      total: Number.isFinite(Number(payload.total))
        ? Number(payload.total)
        : payload.entries.length,
    };
  }

  return { entries: [], total: 0 };
}

/**
 * fetchJson(path, options)
 * - path: "/health" | "/predict" | "/schedule/next-week" ...
 * - options: { method, headers, body, signal }
 */
export async function fetchJson(path, options = {}) {
  // Fail fast in production if the base URL wasn't configured on Vercel.
  if (import.meta.env.PROD && !BASE_URL) {
    throw new Error(
      "Missing VITE_API_BASE_URL. Set it in Vercel → Project Settings → Environment Variables " +
        "(example: https://<your-heroku-app>.herokuapp.com)."
    );
  }

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${BASE_URL}${normalizedPath}`;

  const res = await fetch(url, {
    method: "GET",
    ...options,
    credentials: "omit",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  // Parse body even for errors (helps UI show backend detail)
  const body = await safeReadJson(res);

  if (!res.ok) {
    throw new HttpError(`Request failed (${res.status})`, {
      status: res.status,
      url,
      body,
    });
  }

  return body;
}

// -------------------------
// Health / Debug
// -------------------------

export async function getStatusOverview() {
  // This endpoint is optional. If it fails, return a safe fallback object.
  try {
    const res = await fetchJson("/status/overview");

    if (res && typeof res === "object") {
      const dataset = res.dataset ?? { rows: 0 };
      return {
        ...res,
        health: res.health ?? { status: "unknown" },
        dataset,
        history:
          res.history ??
          ({
            metrics: { total_predictions: Number(dataset?.rows ?? 0) },
          }),
      };
    }

    return createStatusOverviewFallback();
  } catch {
    console.warn("[client] Status overview unavailable; using fallback");
    return createStatusOverviewFallback();
  }
}

export async function getHealthStatus() {
  try {
    return await fetchJson("/health");
  } catch {
    console.warn("[client] Health endpoint unavailable; using fallback");
    return { status: "unknown", reason: "unavailable" };
  }
}

// Legacy alias kept for older wrappers that still import `health`.
export const health = getHealthStatus;

// -------------------------
// Context endpoints (cheap, cacheable)
// -------------------------

export async function getNextWeekSchedule() {
  // Backend may return:
  // - { games: [...] } (recommended)
  // - [...] (older)
  const res = await fetchJson("/schedule/next-week");
  return extractArrayPayload(res, ["games", "ScheduleGame"]);
}

// -------------------------
// Cognitive endpoints (compute)
// -------------------------

export async function predictGame(payload) {
  // Reuse the same normalization rules that the dashboard uses before it stores keys.
  const body = buildGamePredictPayload(payload);

  // Simple contract check: better to fail here than send junk to the API.
  if (
    !body.home_team ||
    !body.away_team ||
    !Number.isFinite(body.season) ||
    !Number.isFinite(body.week)
  ) {
    throw new Error("predictGame requires {home_team, away_team, season, week}");
  }

  return fetchJson("/predict", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getPredictionHistory(limit = 100) {
  try {
    const safeLimit = Number.isFinite(Number(limit)) ? Number(limit) : 100;
    const res = await fetchJson(`/history?limit=${safeLimit}`);
    return normalizeHistoryResponse(res);
  } catch {
    console.warn("[client] History endpoint unavailable; using empty list");
    return normalizeHistoryResponse(null);
  }
}
