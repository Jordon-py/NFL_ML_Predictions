// /src/api/client.js 
/**
 * NFL-ML API Client (Hardened + Teachable v2)
 *
 * WHY THIS FILE EXISTS
 * ---------------------
 * Frontends fail in production for three common reasons:
 * 1) Environment/base-URL confusion (local vs hosted)
 * 2) Flaky networks/timeouts without retries
 * 3) Inconsistent payloads causing 400s
 *
 * This client addresses all three with:
 * - Safe base resolution (localhost uses Vite proxy; hosted uses VITE_API_BASE or a fallback)
 * - Abortable fetch with per-attempt timeout and exponential backoff + jitter
 * - 429/503 Retry-After handling
 * - Strict request normalization for /predict to avoid server 400s
 * - Small, optional in-memory cache for GETs (TTL) to reduce chatter
 * - Rich, typed-ish errors to power helpful UI messages
 *
 * QUICK START
 * -----------
 * import { apiClient, predictGame } from "./api/client";
 * await apiClient.getHealth();
 * await predictGame({ homeTeam: "SF", awayTeam: "SEA", season: 2025, week: 10 });
 *
 * POWER TIPS
 * ----------
 * - Override base per instance: createApi("https://your-backend")
 * - Per-call opts: api(path, init, { timeoutMs: 10000, retries: 1, cacheTtlMs: 5000 })
 */

/**
 * Enhanced NFL-ML API Client with Type Safety
 */

// Type Definitions
/**
 * @typedef {Object} PredictionRequest
 * @property {string} home_team
 * @property {string} away_team  
 * @property {number} season
 * @property {number} week
 */

/**
 * @typedef {Object} PredictionResponse
 * @property {number} home_score
 * @property {number} away_score
 * @property {number} home_win_probability
 * @property {number} away_win_probability
 * @property {number} point_diff
 * @property {string} mode
 * @property {string} prediction_source
 * @property {boolean} win_classifier_used
 * @property {string} win_probability_source
 * @property {number} [win_threshold_used]
 * @property {number} [confidence_score]
 */

/**
 * @typedef {Object} ApiConfig
 * @property {number} [timeoutMs]
 * @property {number} [retries]
 * @property {number} [cacheTtlMs]
 * @property {string} [baseUrl]
 */

class TypedApiClient {
  /**
   * @param {ApiConfig} config
   */
  constructor(config = {}) {
    this.config = {
      timeoutMs: config.timeoutMs || 15000,
      retries: config.retries || 2,
      cacheTtlMs: config.cacheTtlMs || 0,
      baseUrl: config.baseUrl || this.resolveBaseUrl(),
      ...config
    };

    this.cache = new Map();
    this.requestQueue = new Map();
  }

  // Enhanced URL resolution with validation
  resolveBaseUrl() {
    // Guard against non-Vite environments where `import.meta.env` may not
    // be declared in type definitions.
    const meta = typeof import.meta !== 'undefined' ? /** @type {any} */ (import.meta) : {};
    const envUrl = meta.env && typeof meta.env.VITE_API_BASE === 'string'
      ? meta.env.VITE_API_BASE
      : undefined;
    const isLocalhost = window.location.hostname.includes('localhost');

    if (envUrl && this.isValidUrl(envUrl)) {
      return envUrl;
    }

    if (isLocalhost) {
      return ''; // Use relative URLs for local development
    }

    return 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';
  }

  /**
   * @param {string} url
   * @returns {boolean}
   */
  isValidUrl(url) {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * @param {string} path
   * @param {RequestInit} init
   * @param {ApiConfig} options
   * @returns {Promise<any>}
   */
  async request(path, init = {}, options = {}) {
    const mergedOptions = { ...this.config, ...options };
    const url = this.buildUrl(path, mergedOptions.baseUrl);
    const cacheKey = init.method === 'GET' ? url : null;

    // Check cache for GET requests
    if (cacheKey && mergedOptions.cacheTtlMs > 0) {
      const cached = this.getCache(cacheKey);
      if (cached !== undefined) {
        return cached;
      }
    }

    // Prevent duplicate requests
    if (this.requestQueue.has(url)) {
      return this.requestQueue.get(url);
    }

    const requestPromise = this.executeRequest(url, init, mergedOptions);
    this.requestQueue.set(url, requestPromise);

    try {
      const result = await requestPromise;

      // Cache successful GET responses
      if (cacheKey && mergedOptions.cacheTtlMs > 0) {
        this.setCache(cacheKey, result, mergedOptions.cacheTtlMs);
      }

      return result;
    } finally {
      this.requestQueue.delete(url);
    }
  }

  /**
   * @param {string} url
   * @param {RequestInit} init
   * @param {ApiConfig} options
   * @returns {Promise<any>}
   */
  async executeRequest(url, init, options) {
    const maxRetries = typeof options.retries === 'number' ? options.retries : 0;
    const timeoutMs = typeof options.timeoutMs === 'number' ? options.timeoutMs : this.config.timeoutMs;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      try {
        const response = await fetch(url, {
          ...init,
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            ...init.headers
          }
        });

        if (!response.ok) {
          throw await this.createApiError(response, url);
        }

        const contentType = response.headers.get('content-type');
        const data = contentType?.includes('application/json')
          ? await response.json()
          : await response.text();

        return this.validateResponse(data, init.method || 'GET', url);

      } catch (/** @type {any} */ error) {
        if (attempt === maxRetries || !this.isRetriable(error)) {
          throw error;
        }

        await this.delay(this.calculateBackoff(attempt, error));
      } finally {
        clearTimeout(timeoutId);
      }
    }
  }

  /**
   * @param {Response} response
   * @param {string} url
   * @returns {Promise<ApiError>}
   */
  async createApiError(response, url) {
    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = await response.text();
    }

    return new ApiError(
      response.status,
      payload?.detail || payload?.message || response.statusText,
      payload,
      url,
      response.headers
    );
  }

  /**
   * @param {any} data
   * @param {string} method
   * @param {string} url
   * @returns {any}
   */
  validateResponse(data, method, url) {
    // Always validate prediction responses regardless of HTTP verb so
    // both POST /predict and any future GET-based prediction endpoints
    // share the same contract checks.
    if (url.includes('/predict') && data) {
      return this.validatePredictionResponse(data);
    }
    return data;
  }

  /**
   * @param {any} data
   * @returns {PredictionResponse}
   */
  validatePredictionResponse(data) {
    const required = ['home_score', 'away_score', 'home_win_probability', 'point_diff'];
    const missing = required.filter(field => data[field] === undefined);

    if (missing.length > 0) {
      throw new ApiError(500, `Invalid prediction response: missing ${missing.join(', ')}`);
    }

    return data;
  }

  // Prediction-specific methods with enhanced validation
  /**
   * @param {PredictionRequest} request
   * @param {ApiConfig} options
   * @returns {Promise<PredictionResponse>}
   */
  async predictGame(request, options = {}) {
    this.validatePredictionRequest(request);

    return this.request('/predict', {
      method: 'POST',
      body: JSON.stringify(request)
    }, options);
  }

  /**
   * @param {PredictionRequest} request
   */
  validatePredictionRequest(request) {
    if (!request.home_team || !request.away_team) {
      throw new ApiError(400, 'Home and away team are required');
    }

    if (request.season < 2000 || request.season > 2100) {
      throw new ApiError(400, 'Season must be between 2000 and 2100');
    }

    if (request.week < 1 || request.week > 22) {
      throw new ApiError(400, 'Week must be between 1 and 22');
    }

    if (request.home_team === request.away_team) {
      throw new ApiError(400, 'Home and away teams cannot be the same');
    }
  }

  // Cache management
  /**
   * @param {string} key
   * @returns {any}
   */
  getCache(key) {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }

    return entry.data;
  }

  /**
   * @param {string} key
   * @param {any} data
   * @param {number} ttlMs
   */
  setCache(key, data, ttlMs) {
    this.cache.set(key, {
      expiresAt: Date.now() + ttlMs,
      data
    });
  }

  // Utility methods
  /**
   * @param {number} ms
   * @returns {Promise<void>}
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * @param {number} attempt
    * @param {Error|ApiError} error
   * @returns {number}
   */
  calculateBackoff(attempt, error) {
    const baseMs = 300;
    const jitter = 0.25;
    const retryAfter = error instanceof ApiError && error.retryAfterMs ? error.retryAfterMs : 0;

    return Math.max(
      baseMs * Math.pow(2, attempt) * (1 + Math.random() * jitter),
      retryAfter
    );
  }

  /**
   * @param {Error|ApiError} error
   * @returns {boolean}
   */
  isRetriable(error) {
    if (error.name === 'AbortError') return true;
    if (error instanceof TypeError) return true; // Network errors
    if (error instanceof ApiError) {
      return error.status >= 500 || error.status === 429;
    }
    return false;
  }

  /**
   * @param {string} path
   * @param {string} baseUrl
   * @returns {string}
   */
  buildUrl(path, baseUrl) {
    if (path.startsWith('http')) return path;

    const normalizedBase = baseUrl.replace(/\/+$/, '');
    const normalizedPath = path.replace(/^\/+/, '');

    return normalizedBase ? `${normalizedBase}/${normalizedPath}` : `/${normalizedPath}`;
  }
}

// Enhanced Error Class
class ApiError extends Error {
  /**
   * @param {number} status
   * @param {string} message
   * @param {any} [payload]
   * @param {string} [url]
   * @param {Headers} [headers]
   */
  constructor(status, message, payload, url, headers) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
    this.url = url;
    this.headers = headers;
    this.retryAfterMs = this.parseRetryAfter(headers);
  }

  /**
    * @param {Headers|undefined} headers
    * @returns {number|undefined}
    */
  parseRetryAfter(headers) {
    const value = headers?.get('Retry-After');
    if (!value) return undefined;

    const seconds = parseInt(value, 10);
    if (!isNaN(seconds)) return seconds * 1000;

    const date = Date.parse(value);
    if (!isNaN(date)) return Math.max(0, date - Date.now());

    return undefined;
  }
}

// Create default instance
const defaultClient = new TypedApiClient();

/**
 * Factory helper: create a new API client instance.
 *
 * @param {ApiConfig|string} [configOrBaseUrl] Either a config object or a base URL string.
 * @returns {TypedApiClient}
 */
export function createApi(configOrBaseUrl = {}) {
  if (typeof configOrBaseUrl === 'string') {
    return new TypedApiClient({ baseUrl: configOrBaseUrl });
  }
  return new TypedApiClient(configOrBaseUrl || {});
}

// Named alias for the default instance used across the app.
export const apiClient = defaultClient;

/**
 * Fetch lightweight backend health status.
 * Mirrors FastAPI `/health` endpoint.
 *
 * @param {ApiConfig} [options]
 * @returns {Promise<any>}
 */
export function getHealthStatus(options = {}) {
  // Small cache TTL keeps the navbar and dashboard responsive while
  // avoiding excessive polling in production.
  return defaultClient.request('/health', { method: 'GET' }, {
    cacheTtlMs: 5000,
    ...options,
  });
}

/**
 * Fetch the upcoming NFL week schedule from `/schedule/next-week`.
 *
 * @param {ApiConfig} [options]
 * @returns {Promise<any[]>}
 */
export async function getNextWeekSchedule(options = {}) {
  const data = await defaultClient.request('/schedule/next-week', { method: 'GET' }, options);

  // Backend may return an array of games or a wrapped object; normalize to an array.
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.games)) return data.games;
  return [];
}

/**
 * Fetch recent prediction history entries.
 * The backend may expose this as `/history` or `/history/predictions`; we
 * start with the simpler `/history` contract and let the caller handle
 * missing endpoints via try/catch (see PredictionContext).
 *
 * @param {number} [limit]
 * @param {ApiConfig} [options]
 * @returns {Promise<any>}
 */
export function getPredictionHistory(limit = 100, options = {}) {
  const safeLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : 100;
  const path = `/history?limit=${encodeURIComponent(String(safeLimit))}`;
  return defaultClient.request(path, { method: 'GET' }, options);
}

/**
 * High-level prediction helper used by PredictionContext.
 * Accepts camelCase keys and normalises them to the backend's
 * snake_case request payload.
 *
 * @param {{ homeTeam?: string; awayTeam?: string; home_team?: string; away_team?: string; season: number; week: number }} params
 * @param {ApiConfig} [options]
 * @returns {Promise<PredictionResponse>}
 */
export function predictGame(params, options = {}) {
  if (!params) {
    throw new ApiError(400, 'Prediction parameters are required');
  }

  const homeRaw = params.homeTeam ?? params.home_team;
  const awayRaw = params.awayTeam ?? params.away_team;

  if (!homeRaw || !awayRaw) {
    throw new ApiError(400, 'Home and away team are required for prediction');
  }

  const payload = {
    home_team: String(homeRaw).trim().toUpperCase(),
    away_team: String(awayRaw).trim().toUpperCase(),
    season: Number(params.season),
    week: Number(params.week),
  };

  return defaultClient.predictGame(payload, options);
}

// Export both class and default instance for advanced use cases.
export { TypedApiClient, ApiError };
export default defaultClient;
