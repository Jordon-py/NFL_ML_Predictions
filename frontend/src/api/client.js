/**
* File Metrics:
* - Purpose: One tiny, consistent fetch wrapper for the whole app.
* - Why: Every endpoint gets the same error handling, JSON parsing, and abort support.
*
* Key Concepts:
* - AbortController cancels in-flight requests when components unmount.
* - "HttpError" carries status + body for better debugging.
*
* Learning Checkpoints:
* - You should be able to answer: "Where do I change my API base URL?"
* - You should be able to answer: "Where do errors get normalized?"
*/

export class HttpError extends Error {
  constructor(message, { status, url, body } = {}) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

// If you use Vite proxy, set BASE_URL = "" (empty string) and call "/api/..."
const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const BASE_URL = RAW_BASE_URL.replace(/\/+$/, ""); // "" works great with Vite proxy

async function safeReadJson(res) {
  try {
    return await res.json();
  } catch {
    return null; // sometimes backends return empty bodies
  }
}

/**
 * fetchJson(path, options)
 * - path: "/api/..." style
 * - options: { method, headers, body, signal }
 */
export async function fetchJson(path, options = {}) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${BASE_URL}${normalizedPath}`;

  const res = await fetch(url, {
    method: "GET",
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  // Try to parse body (even for errors) so your UI can show useful messages
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

// Re-export specific domain helpers so components like StatsPage.jsx
// don't break when importing from "./client".
// Ideally, components should import from "./nfl" for domain logic, but
// we support the existing pattern here.
import { getNextWeekSchedule, health } from "./nfl";

export { getNextWeekSchedule, health };

// Missing endpoints referenced by StatsPage.jsx
export async function getPredictionHistory(limit = 100) {
  // Gracefully handle missing endpoint until implemented on backend
  try {
    return await fetchJson(`/history?limit=${limit}`);
  } catch (err) {
    console.warn("History endpoint unavailable, falling back to empty");
    return { entries: [], total: 0 };
  }
}

export async function getStatusOverview() {
  // Gracefully handle missing endpoint
  try {
    return await fetchJson("/status/overview");
  } catch (err) {
    console.warn("Status overview unavailable");
    return {
      health: { status: "unknown" },
      dataset: { rows: 0 },
      history: { metrics: { total_predictions: 0 } }
    };
  }
}
