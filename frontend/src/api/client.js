/**
 * client.js
 * ---------
 * Component Purpose:
 *   Offer tiny wrapper functions around `fetch` so components stay declarative.
 *
 * Core Logic Overview:
 *   - Detect the API base URL from `import.meta.env.VITE_API_URL` (Vite convention)
 *     and fall back to the deployed Heroku URL for production builds.
 *   - Provide a generic `api` helper that sets JSON headers and raises an Error
 *     when the network request fails, keeping calling code simple.
 *   - Export specific domain functions (`getNextWeekSchedule`, `predictGame`)
 *     that components/hooks can call without worrying about HTTP details.
 *
 * Modification Guide:
 *   - Add new endpoints by building small wrappers that call `api(path, opts)`.
 *   - Keep data transformation (e.g. mapping to view models) outside this file
 *     so the client stays focused on transport concerns.
 *   - When authentication is introduced, inject headers inside the `api`
 *     helper so all downstream requests pick up the token automatically.
 */

const BASE_URL =
  import.meta.env.VITE_API_URL ||
  'https://nfl-predict-ecf5a5bd34fe.herokuapp.com';

// Debug log for development
if (import.meta.env.DEV) {
  console.log('[API Client] Using BASE_URL:', BASE_URL);
  console.log('[API Client] Mode:', import.meta.env.MODE);
}

function buildUrl(path) {
  // Prefer the standard URL constructor which normalizes slashes.
  try {
    return new URL(path, BASE_URL).toString();
  } catch (err) {
    // Fallback: trim trailing/leading slashes and join.
    const base = BASE_URL.replace(/\/+$|$/, '');
    const cleanPath = String(path).replace(/^\/+/, '');
    return `${base}/${cleanPath}`;
  }
}

async function api(path, opts = {}) {
  const url = buildUrl(path);
  const res = await fetch(url, {
    headers: {'Content-Type': 'application/json'},
    ...opts,
  });
  if (!res.ok) {
    // Bubble up a useful error so calling components can show friendly UI.
    const text = await res.text().catch(() => '');
    throw new Error(`API ${path} failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function getNextWeekSchedule() {
  return api('schedule/next-week');
}

export async function predictGame(body) {
  return api('predict', {method: 'POST', body: JSON.stringify(body)});
}

