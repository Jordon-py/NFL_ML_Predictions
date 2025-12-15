import { createContext, useContext, useState } from "react";

// Minimal PredictionContext used as a lightweight fallback for pages/components
// that expect a global prediction state. This keeps the app buildable and working
// offline while still allowing consumers to override via a real provider.

const defaultState = {
  history: [],
  health: { status: "unknown" },
  // other shape hints that components may access
  loading: false,
  lastUpdated: null,
};

const PredictionContext = createContext(defaultState);

export function PredictionProvider({ children, initialState }) {
  const [state] = useState({ ...defaultState, ...(initialState || {}) });
  return (
    <PredictionContext.Provider value={state}>
      {children}
    </PredictionContext.Provider>
  );
}

export function usePredictions() {
  return useContext(PredictionContext) ?? defaultState;
}

export function usePredictionHistory() {
  const ctx = useContext(PredictionContext) ?? defaultState;
  return Array.isArray(ctx.history) ? ctx.history : [];
}

export default PredictionContext;
