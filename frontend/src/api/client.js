/**
 * API client for NFL prediction system
 * Centralizes all API calls with error handling and base URL management
 * Uses relative URLs since the backend is proxied through the development server
 */

const API_BASE = '';

/**
 * Generic fetch wrapper with JSON handling and error management
 * @param {string} endpoint - API endpoint (without base URL)
 * @param {object} options - Fetch options (method, body, etc.)
 * @returns {Promise<object>} Parsed JSON response
 * @throws {Error} If request fails or returns non-2xx status
 */
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

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

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`API Error: ${response.status} - ${errorData.detail || response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`[API] ${options.method || 'GET'} ${endpoint} failed:`, error);
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
 * @returns {Promise<{home_score: number, away_score: number, point_diff: number}>}
 */
export async function predictGame(payload) {
  /** Log API request details */
  console.log(`[API] POST /predict - Payload: ${JSON.stringify(payload)}`);
  return apiRequest('/predict', {
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
