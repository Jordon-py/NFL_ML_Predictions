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
  const res = await fetchJson("/health");
  console.log('getHealthStatus', res);

  return res;
}

export async function getDebugInfo() {
  const res = await fetchJson("/debug");
  console.log("getDebugInfo", res);
  return res;
}

export async function getNextWeekSchedule() {
  const data = await fetchJson("/schedule/next-week", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  return Array.isArray(data?.games) ? data.games : [];
}

export async function predictGame(payload) {
  const res = await fetchJson("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  console.log('predictGame', res);

  return res;
}

export async function chatLLM(payload) {
  const res = await fetchJson("/llm/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  console.log('chatLLM', res);

  return res;
}

export async function explainPrediction(payload) {
  const res = await fetchJson("/predict/explain", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  console.log('explainPrediction', res);

  return res;
}

export async function getPredictionHistory(limit = 100) {
  const data = await fetchJson(`/history?limit=${limit}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  console.log('getPredictionHistory', data);
  if (Array.isArray(data)) return { entries: data, total: data.length };
  return { entries: data?.entries ?? [], total: data?.total ?? data?.entries?.length ?? 0 };
}

export async function getStatusOverview() {
  const data = await fetchJson("/status/overview", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  console.log('getStatusOverview', data);
  return {
    health: data ? data.health : { status: "unknown" },
    dataset: data ? data.dataset : { rows: 0 },
    history: data ? data.history : { metrics: { total_predictions: 0, win_rate: 0 } },
  };
}
