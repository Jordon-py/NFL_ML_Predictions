/**
 * API client for NFL prediction system
 * Centralizes all API calls with error handling and base URL management
 * Uses relative URLs since the backend is proxied through the development server
 */

const DEFAULT_PROD_API = 'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';

function resolveApiBase() {
  if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  if (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  if (typeof window !== 'undefined' && window.__API_BASE_URL__) {
    return window.__API_BASE_URL__;
  }
  const hostname = typeof window !== 'undefined' ? window.location.hostname : '';
  const isProdHost = hostname && hostname !== 'localhost' && hostname !== '127.0.0.1';
  if (isProdHost) {
    console.warn('[API] No env variable detected in production; defaulting base URL to Heroku backend.');
    return DEFAULT_PROD_API;
  }
  return '';
}

// Prefer Vite env (Vercel) and fall back to CRA env for compatibility
const API = resolveApiBase();

// Basic environment detection (window hostname is more reliable in browser bundle)
const isProd = typeof window !== 'undefined'
  ? !['localhost', '127.0.0.1'].includes(window.location.hostname)
  : (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'production');

if (!API) {
  // eslint-disable-next-line no-console
  console.warn('[API] Base URL is not set. Set VITE_API_URL (Vite/Vercel) or REACT_APP_API_URL (CRA) to your Heroku backend URL.');
}

function buildUrl(endpoint) {
  const ep = String(endpoint || '');
  const normalized = ep.startsWith('/') ? ep : `/${ep}`;
  return `${API}${normalized}`;
}

/**
 * Generic fetch wrapper with JSON handling and error management
 * @param {string} endpoint - API endpoint (without base URL)
 * @param {object} options - Fetch options (method, body, etc.)
 * @returns {Promise<object>} Parsed JSON response
 * @throws {Error} If request fails or returns non-2xx status
 */
async function apiRequest(endpoint, options = {}) {
  if (!API && isProd) {
    throw new Error('[API] Base URL is empty in production. Set VITE_API_URL (Vercel) or REACT_APP_API_URL (CRA) to your Heroku API, e.g. https://<your-app>.herokuapp.com');
  }

  const url = buildUrl(endpoint);

  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });
    /** Log API request details */
    console.log(`[API] ${options.method || 'GET'} ${endpoint} - Status: ${response.status}`);

    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');

    if (!response.ok) {
      if (isJson) {
        const errorData = await response.json().catch(() => ({}));
        const detail = (errorData && (errorData.detail || errorData.message)) || response.statusText;
        throw new Error(`API Error: ${response.status} - ${detail}`);
      }
      const text = await response.text();
      const preview = text ? text.slice(0, 200).replace(/\s+/g, ' ').trim() : '(empty response)';
      throw new Error(`API Error: ${response.status} (non-JSON: ${contentType || 'unknown'}) - Preview: ${preview}`);
    }

    if (!isJson) {
      const text = await response.text();
      const preview = text ? text.slice(0, 200).replace(/\s+/g, ' ').trim() : '(empty response)';
      throw new Error(`API Response not JSON (got ${contentType || 'unknown'}). Preview: ${preview}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`[API] ${options.method || 'GET'} ${endpoint} failed: error`, error);
    throw error;
  }
}

/**
 * Fetch next week's NFL schedule
 * @returns {Promise<Array<{season: number, week: number, kickoff_iso: string, home_abbr: string, away_abbr: string}>>}
 */
export async function getNextWeekSchedule() {
  /** Log API request details */
  console.log("Fetching next week's schedule");
  return apiRequest('/schedule/next-week');
}

/**
 * Predict game outcome using team abbreviations
 * @param {object} payload - Prediction payload
 * @param {string} payload.home_team - Home team abbreviation
 * @param {string} payload.away_team - Away team abbreviation
 * @param {number} payload.season - Season year
 * @param {number} payload.week - Week number
 * @returns {Promise<{home_score: number, away_score: number, point_diff: number, home_win_prob: number, away_win_prob: number}>}
 */
export async function predictGame(payload) {
  /** Log API request details */
  console.log(`[API] POST /predict - Payload: ${JSON.stringify(payload)}`);
    return await apiRequest('/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

   

/**
 * Start model retraining process
 * @returns {Promise<{status: 'started' | 'queued' | 'done'}>}
 */
export async function startTraining() {
  return apiRequest('/train', {
    method: 'POST',
  });
}

/**
 * Health check for API connectivity
 * @returns {Promise<{status: string, mode: string, reason?: string}>}
 */
export async function getHealthStatus() {
  return apiRequest('/health');
}
