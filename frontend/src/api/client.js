// File: frontend/src/api/client.js
// Purpose: Fetch client for the FastAPI backend with simple retry / timeout / (optional) cache.
// Exports: apiClient (default), getHealthStatus, getNextWeekSchedule, getStatusOverview,
//          getPredictionHistory, predictGame.
// Interacts With: Backend endpoints /health, /schedule/next-week, /status/overview,
//                 /history, /predict. Used by PredictionContext, Dashboard, StatsPage.

/**
 * Minimal, explicit API client for the NFL-ML backend.
 * - Resolves a sensible base URL for dev vs production.
 * - Adds timeout + retry behavior around fetch.
 * - Supports a small in-memory cache for idempotent GET requests.
 */
class TypedApiClient
{
  constructor ( config = {} )
  {
    this.config = {
      timeoutMs: 15000,
      retries: 2,
      cacheTtlMs: 0,
      baseUrl: this.resolveBaseUrl( config.baseUrl ),
      ...config,
    };

    /** @type {Map<string, { expiresAt: number, value: any }>} */
    this.cache = new Map();
  }

  /**
   * Decide which base URL to hit.
   * - If VITE_API_BASE is set, always use it.
   * - Otherwise, use localhost:8000 when running on localhost,
   *   and the deployed Heroku URL in production.
   */
  resolveBaseUrl( overrideUrl )
  {
    if ( overrideUrl ) return overrideUrl;

    try {
      // Vite-style env var
      if ( import.meta && import.meta.env && import.meta.env.VITE_API_BASE ) {
        return import.meta.env.VITE_API_BASE;
      }
    } catch ( e ) {
      // ignore; import.meta may not exist in some runtimes (tests, SSR, etc.)
    }

    if ( typeof window !== "undefined" ) {
      const host = window.location.hostname;
      if ( host === "localhost" || host === "127.0.0.1" ) {
        return "http://127.0.0.1:8000";
      }
    }

    // Default production backend
    return "https://nfl-predict-ecf5a5bd34fe.herokuapp.com";
  }

  /**
   * Build an absolute URL from a path or a full URL.
   */
  buildUrl( path )
  {
    if ( !path ) throw new Error( "client.buildUrl: path is required" );

    // Allow callers to pass a full URL
    if ( /^https?:\/\//i.test( path ) ) {
      return path;
    }

    const base = ( this.config.baseUrl || "" ).replace( /\/+$/, "" );
    const suffix = path.startsWith( "/" ) ? path : `/${path}`;
    return `${base}${suffix}`;
  }

  /**
   * Internal helper to read from (and optionally write to) the simple cache.
   */
  getCached( key, ttlMs )
  {
    if ( !ttlMs ) return null;
    const now = Date.now();
    const entry = this.cache.get( key );
    if ( !entry ) return null;
    if ( entry.expiresAt <= now ) {
      this.cache.delete( key );
      return null;
    }
    return entry.value;
  }

  setCached( key, value, ttlMs )
  {
    if ( !ttlMs ) return;
    const expiresAt = Date.now() + ttlMs;
    this.cache.set( key, { expiresAt, value } );
  }

  /**
   * Core request helper with:
   * - automatic base URL resolution
   * - timeout via AbortController
   * - basic retry loop on network/5xx errors
   * - optional caching for GETs
   */
  async request( path, init = {}, options = {} )
  {
    const merged = { ...this.config, ...options };
    const { timeoutMs, retries, cacheTtlMs } = merged;
    const url = this.buildUrl( path );

    const method = ( init.method || "GET" ).toUpperCase();
    const isCacheableGet = method === "GET" && cacheTtlMs && cacheTtlMs > 0;
    const cacheKey = isCacheableGet ? `${method}:${url}` : null;

    // Cache hit for GET
    if ( isCacheableGet && cacheKey ) {
      const cached = this.getCached( cacheKey, cacheTtlMs );
      if ( cached !== null ) return cached;
    }

    let lastError = null;

    for ( let attempt = 0; attempt <= retries; attempt++ ) {
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        timeoutMs || 15000
      );

      try {
        const response = await fetch( url, {
          ...init,
          signal: controller.signal,
          headers: {
            "Content-Type": "application/json",
            ...( init.headers || {} ),
          },
        } );

        clearTimeout( timeoutId );

        const text = await response.text();
        const body = text ? JSON.parse( text ) : null;

        if ( !response.ok ) {
          // Do not retry on 4xx; surface the error.
          if ( response.status >= 400 && response.status < 500 ) {
            const message =
              ( body && body.detail ) ||
              body?.message ||
              `Request failed with status ${response.status}`;
            throw new Error( message );
          }

          // For 5xx errors, fall through to retry logic below.
          lastError = new Error(
            `Server error ${response.status}: ${response.statusText || "Unknown error"
            }`
          );
        } else {
          // Success path
          if ( isCacheableGet && cacheKey ) {
            this.setCached( cacheKey, body, cacheTtlMs );
          }
          return body;
        }
      } catch ( err ) {
        clearTimeout( timeoutId );

        lastError = err;
        const isAbort = err && err.name === "AbortError";
        const isLastAttempt = attempt === retries;

        if ( isLastAttempt || isAbort ) {
          break;
        }
      }
    }

    // If we exhausted retries, throw the last error we observed.
    if ( lastError instanceof Error ) {
      throw lastError;
    }
    throw new Error( "Unknown network error while calling API" );
  }
}

// Shared instance used by the React app.
const apiClient = new TypedApiClient();

/* -------------------------------------------------------------------------- */
/*  Exported functions used by PredictionContext / Dashboard / StatsPage      */
/* -------------------------------------------------------------------------- */

/**
 * Lightweight health check for the backend.
 * Wraps GET /health and enables a short cache to avoid spamming the server.
 */
export function getHealthStatus( options = {} )
{
  return apiClient.request(
    "/health",
    { method: "GET" },
    { cacheTtlMs: 60000, ...options }
  );
}

/**
 * Fetch the next-week schedule from the backend.
 * GET /schedule/next-week
 */
export function getNextWeekSchedule( options = {} )
{
  return apiClient.request( "/schedule/next-week", { method: "GET" }, options );
}

/**
 * Fetch a compact status/overview object for stats pages.
 * GET /status/overview
 * If the endpoint is missing or fails, returns null instead of throwing.
 */
export async function getStatusOverview( options = {} )
{
  try {
    return await apiClient.request(
      "/status/overview",
      { method: "GET" },
      options
    );
  } catch ( err ) {
    console.warn(
      "[api/client] Status overview endpoint failed; returning null.",
      err
    );
    return null;
  }
}

/**
 * Fetch prediction history for charts and audit UI.
 * GET /history?limit={limit}
 * Returns a normalized shape: { entries: Array, total: number }
 */
export async function getPredictionHistory( limit = 100, options = {} )
{
  const res = await apiClient.request(
    `/history?limit=${encodeURIComponent( limit )}`,
    { method: "GET" },
    options
  );

  const entries = Array.isArray( res )
    ? res
    : Array.isArray( res?.entries )
      ? res.entries
      : [];

  const total =
    typeof res?.total === "number" ? res.total : entries.length;

  return { entries, total };
}

/**
 * Call the /predict endpoint for a single game.
 *
 * Accepts a flexible input shape:
 *  - { home_team, away_team, season, week }
 *  - { homeTeam, awayTeam, seasonNum, weekNum }
 *  - a full schedule/game row with home_abbr/away_abbr + season/week
 *
 * Always sends the backend the canonical snake_case payload:
 *  { home_team, away_team, season, week }
 */
export async function predictGame( params, options = {} )
{
  if ( !params ) {
    throw new Error( "predictGame: params are required" );
  }

  // Normalize camelCase and various key names into the canonical ones.
  const homeVal =
    params.home_team ||
    params.homeTeam ||
    params.home_abbr ||
    params.homeAbbr ||
    "";
  const awayVal =
    params.away_team ||
    params.awayTeam ||
    params.away_abbr ||
    params.awayAbbr ||
    "";
  const seasonVal =
    params.season ||
    params.season_num ||
    params.seasonNum ||
    params.seasonNumber;
  const weekVal =
    params.week ||
    params.week_num ||
    params.weekNum ||
    params.weekNumber;

  const season = Number( seasonVal );
  const week = Number( weekVal );

  if (
    !homeVal ||
    !awayVal ||
    !Number.isFinite( season ) ||
    !Number.isFinite( week )
  ) {
    throw new Error(
      `predictGame: invalid input. Got home="${homeVal}", away="${awayVal}", season="${seasonVal}", week="${weekVal}".`
    );
  }

  const payload = {
    home_team: String( homeVal ).trim().toUpperCase(),
    away_team: String( awayVal ).trim().toUpperCase(),
    season,
    week,
  };

  // POST to /predict and return the parsed JSON response.
  // Backend contract (from README):
  // Returns a PredictionResponse including:
  //  - home_score, away_score
  //  - home_win_probability, point_diff
  //  - mode, prediction_source, confidence_score, etc.
  const res = await apiClient.request(
    "/predict",
    { method: "POST", body: JSON.stringify( payload ) },
    options
  );

  return res;
}

export default apiClient;
