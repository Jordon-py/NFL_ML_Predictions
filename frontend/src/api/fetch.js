// ==========================================
// File: frontend/src/api/fetch.js
// Role: Fetch wrapper for API requests.
// Input Data: URL path and fetch options.
// Output Data: Parsed JSON or errors.
// Dependencies: None
// Notes: Centralizes timeouts and error handling.
// ==========================================

/**
 * Core fetch wrapper with timeout, error handling, and environment-aware URL resolution.
 */

function resolveApiBase() {
  const devBase = import.meta.env.VITE_API_BASE_DEV ? import.meta.env.VITE_API_BASE_DEV : "http://127.0.0.1:8000/";
  const prodBase = import.meta.env.VITE_API_BASE_URL ? import.meta.env.VITE_API_BASE_URL : "https://nfl-predict-ecf5a5bd34fe.herokuapp.com/";
  return import.meta.env.DEV ? devBase : prodBase;
}

function normalizeApiBase(rawBase) {
  const base = (rawBase || "").trim().replace(/\/+$/, "");
  if (!base) return "";
  if (base.startsWith("/")) return base;
  if (/^https?:\/\//i.test(base)) return base;
  // Support values like "127.0.0.1:8000" (missing protocol)
  return `http://${base}`;
}

// Normalize: trim, remove trailing slashes, and ensure protocol when needed.
export const API_BASE = normalizeApiBase(resolveApiBase());
export const API_BASE_CONFIGURED = Boolean(API_BASE);

const DEFAULT_TIMEOUT = 25000;



export class HttpError extends Error {
  constructor(message, { status, url, body } = {}) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

// Read body ONCE, then decide what it is.
async function readBody(response) {
  const text = await response.text(); // ✅ single read
  if (!text) return { data: null, rawText: "" };

  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");

  if (isJson) {
    try {
      return { data: JSON.parse(text), rawText: text };
    } catch {
      // JSON header lied or backend returned HTML with json content-type
      return { data: null, rawText: text };
    }
  }

  // Non-JSON (often HTML error page / Vercel index.html)
  return { data: text, rawText: text };
}

export async function fetchJson(path, options = {}) {
  if (!API_BASE_CONFIGURED && !import.meta.env.DEV) {
    throw new HttpError(
      "Missing API base URL. Set VITE_API_BASE_URL (prod) and optionally VITE_API_BASE_DEV (dev) in `frontend/.env` (or Vercel env vars).",
      { status: 0, url: path }
    );
  }
  const url = `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), options.timeout ?? DEFAULT_TIMEOUT);

  // ✅ If caller passed a signal (React cleanup), abort our controller too
  if (options.signal) {
    options.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    const method = (options.method || "GET").toUpperCase();

    const headers = {
      Accept: "application/json",
      ...(options.headers),
    };

    // ✅ Only send Content-Type when we actually send a JSON body
    if (options.body && typeof options.body === "string" && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }

    const response = await fetch(url, {
      method,
      ...options,
      headers,
      signal: controller.signal,
    });

    const { data, rawText } = await readBody(response);
    const contentType = response.headers.get("content-type") || "";
    // Optional: dev-only debug
    if (import.meta.env.DEV) console.log("[fetchJson]", { url, status: response.status, data });

    // If we got routed to a static site (HTML) or some proxy, fail loudly so the UI can show a clear error.
    if (response.ok && !contentType.includes("application/json")) {
      throw new HttpError(`Expected JSON but got ${contentType || "unknown content-type"}`, {
        status: response.status,
        url,
        body: data ?? rawText,
      });
    }

    // ✅ If backend sent JSON error {detail: ...}, keep it. If HTML, show preview.
    if (!response.ok) {
      const detail =
        data && typeof data === "object"
          ? data.detail || data.message
          : typeof data === "string"
            ? data.slice(0, 200)
            : response.statusText;

      if (import.meta.env.DEV) console.log("[fetchJson] Error detail:", detail);

      throw new HttpError(`API Error: ${response.status} (${detail})`, {
        status: response.status,
        url,
        body: data ?? rawText,
      });
    }

    // ✅ If content wasn’t JSON, this returns string; you can enforce JSON here if you prefer.
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new HttpError("Request timed out", { url, status: 408 });
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}
