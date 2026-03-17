// ==========================================
// File: frontend/src/hooks/useAuthSession.js
// Role: React hook for UI state management.
// Input Data: Hook params and state.
// Output Data: State values and actions.
// Dependencies: react
// Notes: Consumed by App routing and landing/auth UI.
// ==========================================

import { useCallback, useMemo, useState } from 'react';

const STORAGE_KEY = 'nfl_predict_local_session_v1';

function readStoredSession() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;

    const parsed = JSON.parse(raw);
    if (!parsed?.email || !parsed?.name) return null;
    return parsed;
  } catch {
    return null;
  }
}

function buildDisplayName(email) {
  const prefix = (email || '').split('@')[0] || 'Analyst';
  const formatted = prefix.replace(/[._-]+/g, ' ').trim();
  return formatted
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

export function useAuthSession() {
  const [session, setSession] = useState(() => readStoredSession());

  const signIn = useCallback((email, password) => {
    const normalizedEmail = (email || '').trim().toLowerCase();
    const normalizedPassword = (password || '').trim();

    if (!normalizedEmail || !normalizedEmail.includes('@')) {
      return { ok: false, message: 'Enter a valid email address.' };
    }
    if (normalizedPassword.length < 6) {
      return { ok: false, message: 'Enter a password with at least 6 characters.' };
    }

    // This is intentionally local-only session state until a real auth backend exists.
    const nextSession = {
      email: normalizedEmail,
      name: buildDisplayName(normalizedEmail),
      signedInAt: new Date().toISOString(),
    };

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(nextSession));
    } catch {
      return { ok: false, message: 'Unable to save your session on this device.' };
    }

    setSession(nextSession);
    return { ok: true };
  }, []);

  const signOut = useCallback(() => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Local sign-out should still proceed even if storage cleanup fails.
    }
    setSession(null);
  }, []);

  return useMemo(
    () => ({
      user: session,
      userId: session?.email || null,
      isAuthenticated: Boolean(session),
      signIn,
      signOut,
    }),
    [session, signIn, signOut],
  );
}
