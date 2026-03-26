/**
 * Legacy fetch helper kept for older experiments.
 *
 * The active app shell uses `frontend/src/api/client.js`. This module remains
 * useful as a small standalone wrapper, but changes to production API behavior
 * should normally happen in `client.js` first.
 */

function resolveApiBase() {
  // In local dev, prefer the explicit dev API target instead of production.
  const isDev = Boolean(import.meta.env.DEV);
  const devBase =
    import.meta.env.VITE_API_DEV ||
    import.meta.env.VITE_API_BASE_DEV ||
    import.meta.env.VITE_DEV_ENV ||
    "";
  const prodBase =
    import.meta.env.VITE_API_BASE_URL ||
    "https://nfl-predict-ecf5a5bd34fe.herokuapp.com/";
  const selected = isDev && devBase ? devBase : prodBase;
  console.log(`Using API base URL: ${selected}`);
  return selected;
}

function stripSurroundingQuotes(value) {
  const s = (value ?? "").toString().trim();
  if (s.length >= 2 && s[0] === s[s.length - 1] && (s[0] === "'" || s[0] === '"')) {
    return s.slice(1, -1).trim();
  }
  return s;
}

function normalizeApiBase(rawBase) {
  let base = stripSurroundingQuotes(rawBase);
  if (!base) return "";

  // Normalize: trim and remove trailing slashes only (do NOT touch protocol slashes)
  base = base.replace(/\/+$/, "");
  if (!base) return "";

  // Relative base (same-origin reverse proxy), e.g. "/api"
  if (base.startsWith("/")) return base;

  // Absolute URL or protocol-relative URL.
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(base) || base.startsWith("//")) {
    return base;
  }

  // Host[:port] with missing scheme -> use current page protocol to avoid mixed-content.
  const protocol =
    typeof window !== "undefined" && window.location && window.location.protocol
      ? window.location.protocol
      : "http:";
  return `${protocol}//${base}`;
}

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

async function readBody(response) {
  let rawText = "";
  try {
    rawText = await response.text();
  } catch {
    return { data: null, rawText: "" };
  }

  if (!rawText) {
    return { data: null, rawText: "" };
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      return { data: JSON.parse(rawText), rawText };
    } catch {
      return { data: rawText, rawText };
    }
  }

  const trimmed = rawText.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      return { data: JSON.parse(trimmed), rawText };
    } catch {
      return { data: rawText, rawText };
    }
  }

  return { data: rawText, rawText };
}

export async function fetchJson(path, options = {}) {
  const { timeout, userId, ...fetchOptions } = options;
  if (!API_BASE_CONFIGURED) {
    throw new HttpError(
      "Missing API base URL. Set VITE_API_BASE_URL (prod) and VITE_API_DEV (dev) in `frontend/.env` or Vercel env vars (example: http://127.0.0.1:8000).",
      { status: 0, url: path }
    );
  }
  const url = `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout ?? DEFAULT_TIMEOUT);

  // ✅ If caller passed a signal (React cleanup), abort our controller too
  if (fetchOptions.signal) {
    fetchOptions.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    const method = (fetchOptions.method || "GET").toUpperCase();

    const headers = {
      Accept: "application/json",
      ...(fetchOptions.headers),
    };

    // Signed-in sessions send a lightweight user identifier so backend history
    // can be stored per user without a full auth provider yet.
    if (userId) {
      headers["X-User-Id"] = userId;
    }

    // ✅ Only send Content-Type when we actually send a JSON body
    if (fetchOptions.body && typeof fetchOptions.body === "string" && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }

    const response = await fetch(url, {
      method,
      ...fetchOptions,
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
    // Normalize fetch() network failures into HttpError for consistent UI handling.
    if (error instanceof HttpError) throw error;
    throw new HttpError(error?.message || "Network error", { url, status: 0, body: null });
  } finally {
    clearTimeout(timeoutId);
  }
}
