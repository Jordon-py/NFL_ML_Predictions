/**
 * NFL Prediction App - Core API Client (Expert v1.2)
 * ================================================
 *
 * A robust, high-performance fetch wrapper engineered for the NFL ML Predictions ecosystem.
 * Features:
 *  - Unified error handling in fetchJson.
 *  - Request timeout and cancellation via AbortController.
 *  - Environment-aware URL resolution with trailing-slash normalization.
 *  - Defensive JSON parsing resilient to empty or malformed responses.
 *  - Data normalization layers to bridge backend-frontend schema drift.
 */
/**
 * Retrieve system health and model readiness.
 */
// client.js (minimal edits)

import { fetchJson } from "./fetch";

export async function getHealthStatus() {
  return fetchJson("/health");
}

export async function getDebugInfo() {
  return fetchJson("/debug");
}

export async function getNextWeekSchedule() {
  const data = await fetchJson("/schedule/next-week");
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.games)) return data.games;
  if (Array.isArray(data?.schedule)) return data.schedule;
  return [];
}

export async function getTeamLogos() {
  const data = await fetchJson("/teams/logos");
  if (data && typeof data === "object" && data.teams && typeof data.teams === "object") {
    return data.teams;
  }
  return {};
}

export async function predictGame(payload) {
  return fetchJson("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function chatLLM(payload) {
  return fetchJson("/llm/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function explainPrediction(payload) {
  return fetchJson("/predict/explain", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getPredictionHistory(limit = 100) {
  const data = await fetchJson(`/history?limit=${limit}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (Array.isArray(data)) {
    return { entries: data, total: data.length };
  }
  if (data && Array.isArray(data.entries)) {
    return { entries: data.entries, total: data.total ?? data.entries.length };
  }
  return { entries: [], total: 0 };
}

export async function getStatusOverview() {
  const data = await fetchJson("/status/overview", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!data) {
    return {
      health: { status: "unknown", mode: "unknown", reason: "no data" },
      dataset: { rows: 0, features: 0 },
      history: { total_predictions: 0, win_rate: null, note: "no data" },
    };
  }
  return data;
}

export async function getModelsStatus() {
  return fetchJson("/status/models");
}

export async function reloadSystem() {
  return fetchJson("/admin/reload", {
    method: "POST",
  });
}

export async function retrainModel(config = {}) {
  return fetchJson("/admin/retrain", {
    method: "POST",
    body: JSON.stringify(config),
  });
}
