// File: frontend/src/PredictionContext.jsx
// Purpose: React context managing schedule fetch, prediction state, health polling, and history caching.
// Functions: reducer(235), loadPredictionHistoryFromStorage(282), parseTeamsCsv(300), PredictionProvider(329), usePredictions(517)
// Variables: PREDICTION_HISTORY_KEY(97), MAX_HISTORY_ENTRIES(98), initialState(218)
// Interacts With: api/client endpoints, Dashboard/StatsPage consumers, localStorage for history cache.
/*
File: PredictionContext.jsx
Purpose: Centralized React context for NFL prediction state; manages schedule fetch, prediction requests, loading/error states, and team metadata.
Functions: PredictionProvider (React component), usePredictions (hook), fetchSchedule (API call), reducer (state updates), getKey (game identifier)
Variables: schedule (games array), week (current week number), teams (team metadata), predictions (keyed by game), loading (keyed by game), errors (keyed by game), current (latest prediction), history (prediction array)
Interacts With: api/client.js (fetch wrappers), backend /predict and /schedule endpoints, DashBoard/TeamGrid (consumers)

PredictionContext.jsx
--------------------
Component Purpose:
  Provide a shared prediction store (current result + historical list + schedule)
  using React Context + Reducer so all views stay in sync.

Core Logic Overview:
  - `initialState` tracks schedule, predictions, loading/error states, and prediction history.
  - `reducer` responds to explicit action types for schedule loading, predictions, and history.
  - Action creators are memoized callbacks exposed through context.
  - Fetches schedule on mount and manages prediction state per-game.
 *
 * Modification Guide:
 *   - Add new action types inside the reducer, then expose a matching
 *     callback in the provider so components never call `dispatch` directly.
 *   - Extend `history` trimming/deduping here instead of inside components.
 */
import React, {
  createContext, useContext, useMemo,
  useReducer, useCallback, useEffect
} from 'react';
import { getNextWeekSchedule, getHealthStatus, getPredictionHistory } from './api/client';
import {
  MAX_HISTORY_ENTRIES,
  PREDICTION_HISTORY_KEY,
  buildGameKey,
  getMetaEnv,
  loadPredictionHistoryFromStorage,
  parseTeamsCsv,
} from './utils/predictionContextUtils';

/**
 * @typedef {*} PredictionResult
 */

/**
 * @typedef {Object} Game
 * @property {string} [game_id]
 * @property {number} [season]
 * @property {number} [week]
 * @property {string} [home_abbr]
 * @property {string} [away_abbr]
 * @property {string} [home_team]
 * @property {string} [away_team]
 */

/**
 * @typedef {Object} TeamMeta
 * @property {string} name
 * @property {string} logoUrl
 */

/** @typedef {Record<string, TeamMeta>} TeamsMap */
/** @typedef {Record<string, boolean>} LoadingMap */
/** @typedef {Record<string, string | null | undefined>} ErrorMap */

/**
 * @typedef {Object} HealthState
 * @property {"unknown" | "healthy" | "unhealthy"} status
 * @property {string} mode
 * @property {string} reason
 */

/** @typedef {PredictionResult & { timestamp: string, game: Game }} PredictionHistoryEntry */

/**
 * @typedef {Object} PredictionState
 * @property {PredictionResult | null} current
 * @property {PredictionHistoryEntry[]} history
 * @property {Game[]} schedule
 * @property {number} week
 * @property {TeamsMap} teams
 * @property {Record<string, PredictionResult>} predictions
 * @property {LoadingMap} loading
 * @property {ErrorMap} errors
 * @property {HealthState} health
 */

/**
 * @typedef {PredictionState & {
 *   setCurrent: (prediction: PredictionResult | null) => void,
 *   pushHistory: (entry: PredictionHistoryEntry) => void,
 *   resetHistory: () => void,
 *   setPrediction: (key: string, prediction: PredictionResult) => void,
 *   setLoading: (key: string, loading: boolean) => void,
 *   setError: (key: string, error: string | null | undefined) => void,
 *   setHealth: (health: HealthState) => void,
 *   count: number,
 *   latest: PredictionHistoryEntry | null,
 * }} PredictionContextValue
 */

// Action types
const SET_CURRENT = 'SET_CURRENT';
const PUSH_HISTORY = 'PUSH_HISTORY';
const RESET_HISTORY = 'RESET_HISTORY';
const SET_SCHEDULE = 'SET_SCHEDULE';
const SET_PREDICTION = 'SET_PREDICTION';
const SET_LOADING = 'SET_LOADING';
const SET_ERROR = 'SET_ERROR';
const SET_HEALTH = 'SET_HEALTH';
const SET_HISTORY = 'SET_HISTORY';
const SET_TEAMS = 'SET_TEAMS';

/**
 * @typedef {Object} SetCurrentAction
 * @property {typeof SET_CURRENT} type
 * @property {PredictionResult | null} payload
 */

/**
 * @typedef {Object} PushHistoryAction
 * @property {typeof PUSH_HISTORY} type
 * @property {PredictionHistoryEntry} payload
 */

/**
 * @typedef {Object} ResetHistoryAction
 * @property {typeof RESET_HISTORY} type
 */

/**
 * @typedef {Object} SetScheduleAction
 * @property {typeof SET_SCHEDULE} type
 * @property {{ schedule: Game[], week: number }} payload
 */

/**
 * @typedef {Object} SetPredictionAction
 * @property {typeof SET_PREDICTION} type
 * @property {{ key: string, prediction: PredictionResult }} payload
 */

/**
 * @typedef {Object} SetLoadingAction
 * @property {typeof SET_LOADING} type
 * @property {{ key: string, loading: boolean }} payload
 */

/**
 * @typedef {Object} SetErrorAction
 * @property {typeof SET_ERROR} type
 * @property {{ key: string, error: string | null | undefined }} payload
 */

/**
 * @typedef {Object} SetHealthAction
 * @property {typeof SET_HEALTH} type
 * @property {HealthState} payload
 */

/**
 * @typedef {Object} SetHistoryAction
 * @property {typeof SET_HISTORY} type
 * @property {PredictionHistoryEntry[]} payload
 */

/**
 * @typedef {Object} SetTeamsAction
 * @property {typeof SET_TEAMS} type
 * @property {TeamsMap} payload
 */

/**
 * @typedef {
 *   | SetCurrentAction
 *   | PushHistoryAction
 *   | ResetHistoryAction
 *   | SetScheduleAction
 *   | SetPredictionAction
 *   | SetLoadingAction
 *   | SetErrorAction
 *   | SetHealthAction
 *   | SetHistoryAction
 *   | SetTeamsAction
 * } PredictionAction
 */

/** @type {PredictionState} */
const initialState = {
  current: null,
  history: [],
  schedule: [],
  week: 11,
  teams: {},
  predictions: {},
  loading: {},
  errors: {},
  health: { status: 'unknown', mode: 'none', reason: 'init' }
};

/**
 * @param {PredictionState} state
 * @param {PredictionAction} action
 * @returns {PredictionState}
 */
function reducer( state, action )
{
  switch ( action.type ) {
    case SET_CURRENT:
      return { ...state, current: action.payload };
    case PUSH_HISTORY:
      return { ...state, history: [ action.payload, ...state.history ].slice( 0, MAX_HISTORY_ENTRIES ) };
    case RESET_HISTORY:
      return { ...state, history: [] };
    case SET_SCHEDULE:
      return { ...state, schedule: action.payload.schedule, week: action.payload.week };
    case SET_PREDICTION: {
      const { key, prediction } = action.payload;
      return {
        ...state,
        predictions: { ...state.predictions, [ key ]: prediction },
        current: prediction
      };
    }
    case SET_LOADING: {
      const { key, loading } = action.payload;
      return { ...state, loading: { ...state.loading, [ key ]: loading } };
    }
    case SET_ERROR: {
      const { key, error } = action.payload;
      return { ...state, errors: { ...state.errors, [ key ]: error } };
    }
    case SET_HEALTH: {
      return { ...state, health: action.payload };
    }
    case SET_HISTORY: {
      const incoming = Array.isArray( action.payload ) ? action.payload : [];
      return { ...state, history: incoming.slice( 0, MAX_HISTORY_ENTRIES ) };
    }
    case SET_TEAMS: {
      const next = action.payload && typeof action.payload === 'object' ? action.payload : {};
      return { ...state, teams: { ...state.teams, ...next } };
    }
    default:
      return state;
  }
}

/** @type {React.Context<PredictionContextValue | null>} */
const Ctx = createContext(/** @type {PredictionContextValue | null} */( null ) );

/**
 * @param {{ children: React.ReactNode }} props
 * @returns {React.ReactElement}
 */
export function PredictionProvider( { children } )
{
  const [ state, dispatch ] = useReducer( reducer, initialState, ( s ) => ( {
    ...s, history: loadPredictionHistoryFromStorage()
  } ) );

  // Actions
  /** @type {(prediction: PredictionResult | null) => void} */
  const setCurrent = useCallback( ( e ) => dispatch( { type: SET_CURRENT, payload: e } ), [] );
  /** @type {(entry: PredictionHistoryEntry) => void} */
  const pushHistory = useCallback( ( e ) => dispatch( { type: PUSH_HISTORY, payload: e } ), [] );
  const resetHistory = useCallback( () => dispatch( { type: RESET_HISTORY } ), [] );

  /** @type {(schedule: Game[], week: number) => void} */
  const setSchedule = useCallback(
    ( schedule, week ) => dispatch( { type: SET_SCHEDULE, payload: { schedule, week } } ),
    []
  );

  /** @type {(key: string, prediction: PredictionResult) => void} */
  const setPrediction = useCallback( ( key, prediction ) =>
    dispatch( { type: SET_PREDICTION, payload: { key, prediction } } ), [] );

  /** @type {(key: string, loading: boolean) => void} */
  const setLoading = useCallback( ( key, loading ) =>
    dispatch( { type: SET_LOADING, payload: { key, loading } } ), [] );

  /** @type {(key: string, error: string | null | undefined) => void} */
  const setError = useCallback( ( key, error ) =>
    dispatch( { type: SET_ERROR, payload: { key, error } } ), [] );

  /** @type {(health: HealthState) => void} */
  const setHealth = useCallback( ( h ) => dispatch( { type: SET_HEALTH, payload: h } ), [] );
  /** @type {(entries: PredictionHistoryEntry[]) => void} */
  const setHistoryState = useCallback( ( entries ) => dispatch( { type: SET_HISTORY, payload: entries } ), [] );
  /** @type {(teams: TeamsMap) => void} */
  const setTeams = useCallback( ( teams ) => dispatch( { type: SET_TEAMS, payload: teams } ), [] );

  // Fetch schedule on mount
  useEffect( () =>
  {
    let mounted = true;
    const fetchSchedule = async () =>
    {
      try {
        const scheduleData = await getNextWeekSchedule();

        if ( !mounted || !Array.isArray( scheduleData ) ) return;

        console.info( `[scheduleData] Fetched ${scheduleData.length} games from backend schedule API.` );
        // Extract week from first game and coerce to number. Accept several
        // possible field names to be resilient against backend shape changes.
        const rawWeek = scheduleData[ 0 ]?.week ?? scheduleData[ 0 ]?.week_num ?? scheduleData[ 0 ]?.weekNum;
        const week = Number.isFinite( Number( rawWeek ) ) ? Number( rawWeek ) : 1;
        setSchedule( scheduleData, week );

        console.log( `[PredictionContext] Loaded ${scheduleData.length} games for Week ${week}` );
      } catch ( err ) {
        console.error( '[PredictionContext] Failed to fetch schedule:', err );
      }
    };
    fetchSchedule();
    return () => { mounted = false; };
  }, [ setSchedule ] );

  // Poll health (lightweight) so UI can gate prediction attempts until backend ready
  useEffect( () =>
  {
    let active = true;
    const poll = async () =>
    {
      try {
        const h = await getHealthStatus();
        if ( active && h && h.status ) setHealth( h );
      } catch ( e ) {
        if ( active ) setHealth( { status: 'unhealthy', mode: 'none', reason: 'health fetch failed' } );
      }
    };
    poll();
    const id = setInterval( poll, 15000 ); // 15s cadence
    return () => { active = false; clearInterval( id ); };
  }, [ setHealth ] );

  // Hydrate history from backend (falls back to localStorage seed when API unavailable)
  useEffect( () =>
  {
    let active = true;
    const loadHistoryFromBackend = async () =>
    {
      try {
        const payload = await getPredictionHistory( MAX_HISTORY_ENTRIES );
        if ( !active || !payload ) return;
        const entries = Array.isArray( payload.entries ) ? payload.entries : [];
        setHistoryState( entries );

        // Seed predictions map so schedule grid can show prior outcomes.
        entries.forEach( ( entry ) => {
          const key = buildGameKey( entry );
          if ( key ) {
            setPrediction( key, entry );
          }
        } );
      } catch ( err ) {
        console.warn( '[PredictionContext] History fetch failed, using local cache.', err );
      }
    };
    loadHistoryFromBackend();
    const id = setInterval( loadHistoryFromBackend, 60000 );
    return () => { active = false; clearInterval( id ); };
  }, [ setHistoryState, setPrediction ] );

  // Load team metadata (names + logo URLs) from public CSV once on mount.
  useEffect( () =>
  {
    let active = true;
    const loadTeams = async () =>
    {
      try {
        const res = await fetch( "myteamdescriptions.csv" );
        if ( !res.ok ) return;
        const text = await res.text();
        if ( !active ) return;
        const teamsMap = parseTeamsCsv( text );
        if ( teamsMap && Object.keys( teamsMap ).length ) {
          setTeams( teamsMap );
          const env = getMetaEnv();
          if ( env?.DEV ) {
            console.debug( "[PredictionContext] Loaded team metadata for", Object.keys( teamsMap ).length, "teams" );
          }
        }
      } catch ( err ) {
        console.warn( "[PredictionContext] Failed to load team descriptions; logos may be missing.", err );
      }
    };
    loadTeams();
    return () => { active = false; };
  }, [ setTeams ] );

  // Make a prediction for a game
  // Note: prediction requests are performed by UI components directly
  // (e.g., Dashboard -> predictGame) and then the context is updated via
  // setPrediction / pushHistory / setLoading / setError. This keeps the
  // prediction logic outside of this context and centralizes network calls
  // to the caller that initiates the request.

  useEffect( () =>
  {
    try {
      localStorage.setItem( PREDICTION_HISTORY_KEY, JSON.stringify( state.history ) );
    } catch { }
  }, [ state.history ] );

  // Tiny dev logger
  useEffect( () =>
  {
    const env = getMetaEnv();
    if ( typeof window !== "undefined" && env?.DEV ) {
      console.debug( "[PredictionContext] state:", state );
    }
  }, [ state ] );


  // Selectors
  const count = state.history.length;
  const latest = state.history[ 0 ] ?? null;

  const value = useMemo( () => ( {
    // State
    ...state,
    // Actions
    setCurrent,
    pushHistory,
    resetHistory,
    // Expose setters for external callers to attach prediction results
    setPrediction,
    setLoading,
    setError,
    // Direct health setter (rarely needed externally)
    setHealth,
    // Selectors
    count,
    latest,
  } ), [ state, setCurrent, pushHistory, resetHistory, setPrediction, setLoading, setError, setHealth, count, latest ] );

  return <Ctx.Provider value={ value }>{ children }</Ctx.Provider>;
}

/**
 * @returns {PredictionContextValue}
 */
export const usePredictions = () =>
{
  const ctx = useContext( Ctx );
  if ( !ctx ) throw new Error( "usePredictions must be used within PredictionProvider" );
  return ctx;
};
