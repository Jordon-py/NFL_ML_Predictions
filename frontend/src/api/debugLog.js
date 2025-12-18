// /frontend/src/api/debugLog.js
// ----------------------------------------------------------------------------
// Lightweight client-side debug log for API errors and notable events.
// Stores a bounded log (default 50 entries) in localStorage under the key
// "api_debug_log". This is intentionally tiny to avoid pulling in a logging
// library and to keep user data local.
// ----------------------------------------------------------------------------

const STORAGE_KEY = 'api_debug_log';
const MAX_ENTRIES = 50;

function safeNowISO() {
  try {
    return new Date().toISOString();
  } catch {
    return String(Date.now());
  }
}

export function addLog(entry) {
  try {
    const base = {
      ts: safeNowISO(),
      level: 'error',
      source: 'frontend',
      ...entry,
    };
    const raw = localStorage.getItem(STORAGE_KEY);
    const arr = raw ? (JSON.parse(raw) || []) : [];
    arr.push(base);
    const bounded = arr.slice(-MAX_ENTRIES);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(bounded));
  } catch {
    // Non-fatal: localStorage unavailable or quota exceeded.
  }
}

export function getLogs() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

export function clearLogs() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
