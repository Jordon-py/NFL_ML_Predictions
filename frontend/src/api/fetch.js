// ==========================================
// File: frontend/src/api/fetch.js
// Role: Fetch wrapper for API requests.
// Input Data: URL path and fetch options.
// Output Data: Parsed JSON or errors.
// Dependencies: None
// Notes: Centralizes timeouts and error handling.
// ==========================================

// fetch.js
/**
 * Custom fetch function with timeout and error handling
 * Environment-based API configuration
 * Safely parse JSON, return null for empty/invalid responses
 * fetch function with educational comments
 */

// Environment-based API configuration
// fetch.js (minimal-but-strong)

const isLocalhost =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
const DEV_BASE =
  import.meta.env.VITE_API_DEV ||
  import.meta.env.VITE_DEV_ENV ||
  import.meta.env.VITE_API_BASE_DEV;
const PROD_BASE = import.meta.env.VITE_API_BASE_URL;
const RAW_BASE = isLocalhost ? (DEV_BASE || PROD_BASE) : PROD_BASE;

// ✅ Fail fast instead of silently becoming "undefined/"
if (!RAW_BASE) {
  throw new Error(
    "[fetch.js] Missing VITE_API_BASE_URL (prod) or VITE_API_DEV/VITE_DEV_ENV (dev). " +
    "On Vercel, set VITE_API_BASE_URL in Project > Settings > Environment Variables."
  );
}

// ✅ Normalize: store WITHOUT trailing slash
export const API_BASE = String(RAW_BASE);
const DEFAULT_TIMEOUT = 15000;

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
    // Optional: dev-only debug
    if (import.meta.env.DEV) console.log("[fetchJson]", { url, status: response.status, data });

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
