// ==========================================
// File: frontend/src/components/DashBoard/Dashboard.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ../../api/client.js, ../Card/TeamGrid, ../PredictionResult
// Notes: Presentation-focused component.
// ==========================================

/**
 * NFL Prediction Dashboard (Simplified)
 * -------------------------------------
 * Straightforward container for predictions, schedule display, history, and LLM chat.
 */

import { useState, useMemo, useEffect, useCallback } from "react";
import { getGameScores, predictGame } from "../../api/client.js";
import TeamGrid from "../Card/TeamGrid";
import PredictionResult from "../PredictionResult";
import HistoryChart from "../HistoryChart";
import NavBar from "../NavBar/NavBar";
import ErrorDisplay from "../ErrorDisplay";
import LLMChat from "../LLMChat/LLMChat";
import { buildGameKey } from "../../utils/predictionContextUtils";
import { toEntry } from "../../utils/predictionHelpers";
import "./Dashboard.css";

export default function Dashboard({
  authSession,
  onSignOut,
  schedule,
  week,
  predictions,
  loading,
  errors,
  current,
  history,
  health,
  seasonContext,
  loadScheduleForWeek,
  setPrediction,
  setLoading,
  setError,
  pushHistory,
}) {
  const [showcase, setShowcase] = useState(null);
  const [showcaseLoading, setShowcaseLoading] = useState(false);
  const [showcaseError, setShowcaseError] = useState("");
  const [selectedSeason, setSelectedSeason] = useState(
    () => seasonContext?.current_season ?? new Date().getFullYear()
  );
  const [selectedWeek, setSelectedWeek] = useState(() =>
    Number.isFinite(Number(week))
      ? Number(week)
      : seasonContext?.display_week ?? null
  );
  const [scheduleOverrideLoading, setScheduleOverrideLoading] = useState(false);
  const [scheduleOverrideError, setScheduleOverrideError] = useState("");
  const [actualScores, setActualScores] = useState({});
  const [scoreLoading, setScoreLoading] = useState(false);
  const [scoreError, setScoreError] = useState("");
  const SCORE_SYNC_LABEL = "Sun · Mon · Thu nights (UTC)";
  const activeSeason =
    selectedSeason ?? seasonContext?.current_season ?? new Date().getFullYear();
  const activeWeek =
    Number.isFinite(Number(selectedWeek)) && selectedWeek !== null
      ? selectedWeek
      : Number.isFinite(Number(week))
        ? Number(week)
        : seasonContext?.display_week ?? null;
  const finalScoreCount = Object.keys(actualScores).length;
  const scoreboardSummaryHeadline = finalScoreCount
    ? `${finalScoreCount} final result${finalScoreCount === 1 ? "" : "s"} synced`
    : "Final scores appear after the next sync.";
  const scoreboardSummaryDetail = `Latest refresh: ${SCORE_SYNC_LABEL}.`;
  const gridWeek = activeWeek ?? week;

  const TEAM_POOL = useMemo(
    () => [
      "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
      "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
      "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
      "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    ],
    []
  );

  useEffect(() => {
    if (seasonContext?.current_season) {
      setSelectedSeason(seasonContext.current_season);
    }
  }, [seasonContext?.current_season]);

  useEffect(() => {
    const weekValue = Number.isFinite(Number(week))
      ? Number(week)
      : seasonContext?.display_week ?? null;
    setSelectedWeek(weekValue);
  }, [week, seasonContext?.display_week]);

  const handlePredict = async (game) => {
    if (!game) return;

    const key = buildGameKey(game);
    if (!key) return;

    setLoading(key, true);
    setError(key, null);

    try {
      const home = game.home_team || game.home_abbr;
      const away = game.away_team || game.away_abbr;
      const season = game.season;
      const week = game.week;

      const rawPrediction = await predictGame(home, away, season, week, authSession?.userId);
      const entry = toEntry({ prediction: rawPrediction, game, source: "teamgrid" });
      const predictionKey = buildGameKey(entry) || key;
      const normalizedEntry = {
        ...entry,
        game_id: key || entry.game_id,
      };

      setPrediction(key, normalizedEntry);
      pushHistory(normalizedEntry);
    } catch (error) {
      const detail = error?.body?.detail || error?.message || "Prediction request failed";
      setError(key, detail);
    } finally {
      setLoading(key, false);
    }
  };

  const handleReset = (game) => {
    if (!game) return;
    const key = buildGameKey(game);
    if (!key) return;
    setPrediction(key, null);
    setError(key, null);
    setLoading(key, false);
  };

  const fetchScoresForSelection = useCallback(
    async (seasonValue, weekValue) => {
      if (!seasonValue || !Number.isFinite(Number(weekValue))) {
        setActualScores({});
        return;
      }
      setScoreError("");
      setScoreLoading(true);
      try {
        const rows = await getGameScores(seasonValue, weekValue);
        const map = rows.reduce((acc, entry) => {
          if (entry?.game_id) {
            acc[entry.game_id] = entry;
          }
          return acc;
        }, {});
        setActualScores(map);
      } catch (err) {
        setActualScores({});
        setScoreError("Unable to load final scores for that slate.");
      } finally {
        setScoreLoading(false);
      }
    },
    []
  );

  const applyScheduleOverride = useCallback(
    async (seasonValue, weekValue) => {
      if (!Number.isFinite(Number(weekValue))) return;
      setScheduleOverrideLoading(true);
      setScheduleOverrideError("");
      try {
        await loadScheduleForWeek(seasonValue, weekValue);
        setSelectedSeason(seasonValue);
        setSelectedWeek(weekValue);
      } catch (error) {
        setScheduleOverrideError(
          "Could not load that week. Try another week or reset to live slate."
        );
      } finally {
        setScheduleOverrideLoading(false);
      }
    },
    [loadScheduleForWeek]
  );

  const handleWeekChange = (delta) => {
    const baseWeek = Number.isFinite(Number(selectedWeek))
      ? selectedWeek
      : Number.isFinite(Number(week))
        ? Number(week)
        : seasonContext?.display_week ?? 1;
    const targetWeek = Math.max(1, baseWeek + delta);
    const targetSeason =
      selectedSeason ?? seasonContext?.current_season ?? new Date().getFullYear();
    applyScheduleOverride(targetSeason, targetWeek);
  };

  const handleSeasonChange = (delta) => {
    const baseSeason = selectedSeason ?? seasonContext?.current_season ?? new Date().getFullYear();
    const targetSeason = Math.max(2000, baseSeason + delta);
    const targetWeek = Number.isFinite(Number(selectedWeek))
      ? selectedWeek
      : Number.isFinite(Number(week))
        ? Number(week)
        : seasonContext?.display_week ?? 1;
    applyScheduleOverride(targetSeason, targetWeek);
  };

  const handleResetSchedule = () => {
    const seasonValue = seasonContext?.current_season ?? new Date().getFullYear();
    const weekValue =
      Number.isFinite(Number(week)) ? Number(week) : seasonContext?.display_week ?? 1;
    applyScheduleOverride(seasonValue, weekValue);
  };

  useEffect(() => {
    if (selectedSeason && Number.isFinite(Number(selectedWeek))) {
      fetchScoresForSelection(selectedSeason, selectedWeek);
    } else {
      setActualScores({});
    }
  }, [selectedSeason, selectedWeek, fetchScoresForSelection]);

  const runOffseasonShowcase = async () => {
    if (showcaseLoading) return;
    setShowcaseError("");
    setShowcaseLoading(true);
    try {
      const homeIdx = Math.floor(Math.random() * TEAM_POOL.length);
      let awayIdx = Math.floor(Math.random() * TEAM_POOL.length);
      if (awayIdx === homeIdx) awayIdx = (awayIdx + 7) % TEAM_POOL.length;
      const home = TEAM_POOL[homeIdx];
      const away = TEAM_POOL[awayIdx];
      const season = Number(seasonContext?.current_season) || new Date().getFullYear();
      const kickoffWeek = Number(seasonContext?.display_week) || 1;

      const rawPrediction = await predictGame(home, away, season, kickoffWeek);
      const syntheticGame = {
        home_team: home,
        away_team: away,
        home_abbr: home,
        away_abbr: away,
        season,
        week: kickoffWeek,
      };
      const gameKey = buildGameKey(syntheticGame);
      const entry = toEntry({
        prediction: rawPrediction,
        game: syntheticGame,
        source: "offseason-showcase",
      });
      const normalizedEntry = {
        ...entry,
        game_id: gameKey || entry.game_id,
      };
      setShowcase(normalizedEntry);
      if (gameKey) {
        setPrediction(gameKey, normalizedEntry);
      }
      pushHistory(normalizedEntry);
    } catch (error) {
      const detail = error?.body?.detail || error?.message || "Could not generate offseason showcase matchup.";
      setShowcaseError(detail);
    } finally {
      setShowcaseLoading(false);
    }
  };

  const healthStatus = health?.status;
  const backendHealthy = healthStatus === "healthy";
  const scheduleEmpty = !schedule || schedule.length === 0;
  const seasonPhase = seasonContext?.phase || "offseason";
  const seasonLabel = seasonContext?.label || "Offseason";
  const seasonMessage = seasonContext?.message || "No live slate currently available.";
  const nextKickoffLabel = seasonContext?.next_kickoff
    ? new Date(seasonContext.next_kickoff).toLocaleString()
    : "TBD";
  const isOffseasonMode = seasonPhase === "offseason";
  const shouldShowBackendError =
    scheduleEmpty && Boolean(healthStatus) && healthStatus !== "loading" && !backendHealthy;

  if (shouldShowBackendError) {
    return (
      <ErrorDisplay
        error={new Error(health?.reason || health?.mode || `Service status: ${healthStatus}`)}
        onRetry={() => window.location.reload()}
      />
    );
  }

  return (
    <div className="dashboard-layout advanced">
        <NavBar
          authSession={authSession}
          onSignOut={onSignOut}
          state={{
            health,
            title: "Dashboard",
            heroSubtitle: "Load a slate, run a forecast, and compare once the final score lands.",
            subtitle:
              "Use the week + season controls to browse live or archived slates; final scores sync Sun/Mon/Thu nights.",
            weekLabel:
              typeof activeWeek === "number" && Number.isFinite(activeWeek)
                ? `Week ${activeWeek}`
                : null,
            healthLabel:
              health?.status === "healthy"
                ? "Service: Live"
              : `Service: ${health?.status ?? "unknown"}`,
        }}
      />

      <main className="dashboard-main advanced">
        <header className="dashboard-header advanced">
          <div className="dashboard-header-content">
            <h1 className="dashboard-title">Upcoming matchups</h1>
            <p className="dashboard-subtitle">
              Select any game to generate a forecast, then review the full breakdown below.
            </p>
            <div className={`season-context-ribbon phase-${seasonPhase}`}>
              <strong className="season-context-pill">{seasonLabel}</strong>
              <span className="season-context-message">{seasonMessage}</span>
              <span className="season-context-kickoff">Next kickoff: {nextKickoffLabel}</span>
            </div>
          </div>
        </header>

        <section className="dashboard-content advanced">
          <div className="content-grid advanced">
            {isOffseasonMode ? (
              <section className="offseason-mode-panel" aria-live="polite">
                <h2>No live slate right now</h2>
                <p>
                  Generate a sample matchup to preview the forecast experience between official NFL
                  slates.
                </p>
                <div className="offseason-mode-actions">
                  <button
                    type="button"
                    onClick={runOffseasonShowcase}
                    disabled={showcaseLoading}
                    className="offseason-mode-button"
                  >
                    {showcaseLoading ? "Generating sample matchup..." : "Generate sample matchup"}
                  </button>
                </div>
                {showcaseError ? <p className="offseason-mode-error">{showcaseError}</p> : null}
                <PredictionResult entry={showcase || current} />
              </section>
            ) : (
              <>
                <div className="schedule-selector">
                  <div className="schedule-selector__row">
                    <button
                      type="button"
                      onClick={() => handleWeekChange(-1)}
                      disabled={scheduleOverrideLoading}
                    >
                      Previous Week
                    </button>
                    <span className="schedule-selector__label">
                      Week {activeWeek ?? "—"} · Season {activeSeason}
                    </span>
                    <button
                      type="button"
                      onClick={() => handleWeekChange(1)}
                      disabled={scheduleOverrideLoading}
                    >
                      Next Week
                    </button>
                  </div>
                  <div className="schedule-selector__row">
                    <button
                      type="button"
                      onClick={() => handleSeasonChange(-1)}
                      disabled={scheduleOverrideLoading}
                    >
                      Previous Season
                    </button>
                    <button
                      type="button"
                      onClick={() => handleSeasonChange(1)}
                      disabled={scheduleOverrideLoading}
                    >
                      Next Season
                    </button>
                    <button
                      type="button"
                      onClick={handleResetSchedule}
                      disabled={scheduleOverrideLoading}
                    >
                      Live slate
                    </button>
                  </div>
                  <p className="schedule-selector__feedback">
                    {scheduleOverrideError ||
                      scoreError ||
                      "Use the controls to load a different week or season, then pick any matchup to forecast."}
                  </p>
                  <div
                    className={`scoreboard-summary ${
                      finalScoreCount ? "scoreboard-summary--ready" : "scoreboard-summary--waiting"
                    }`}
                  >
                    <div className="scoreboard-summary__body">
                      <strong>{scoreboardSummaryHeadline}</strong>
                      <span>{scoreboardSummaryDetail}</span>
                    </div>
                    <span className="scoreboard-summary__cta">
                      {scoreLoading
                        ? "Refreshing final results…"
                        : finalScoreCount
                          ? "Matchups now show the synced final score."
                          : "Run a forecast to compare it to the scoreboard."}
                    </span>
                  </div>
                </div>
                {(!schedule || schedule.length <= 1) && (
                  <div className="schedule-hint">
                    <p>
                      Only {schedule?.length ?? 0} matchup{schedule?.length === 1 ? "" : "s"} are available for Week{" "}
                      {activeWeek ?? "—"} of Season {activeSeason}. Use the week and season controls above to
                      explore another slate or pull last season's archive.
                    </p>
                    <p className="schedule-hint__note">
                      Final scores will show up here after they sync on Sunday, Monday, and Thursday nights.
                    </p>
                  </div>
                )}
                <div className="team-grid-section enhanced">
                <TeamGrid
                  week={gridWeek}
                  season={activeSeason}
                  games={schedule}
                  predictions={predictions}
                  loading={loading}
                  errors={errors}
                  onPredict={handlePredict}
                  onReset={handleReset}
                  actualScores={actualScores}
                />
              </div>
              </>
            )}

            <div className="history-section enhanced">
              <HistoryChart history={history} />
            </div>
          </div>
        </section>

        <section className="prediction-results-section advanced" aria-live="polite">
          <PredictionResult entry={current} userId={authSession?.userId} />
        </section>

        <section className="llm-chat-section">
          <LLMChat prediction={current} userId={authSession?.userId} />
        </section>
      </main>
    </div>
  );
}
