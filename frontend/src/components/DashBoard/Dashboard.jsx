import { useEffect, useMemo, useState } from "react";
import TeamGrid from "../Card/TeamGrid.jsx";
import NavBar from "../NavBar/NavBar.jsx";
import { predictGame, premiumChat } from "../../api/client.js";
import { buildMatchupKey, buildPredictPayload, getGameWeek } from "../../utils/gameUtils.js";
import "./Dashboard.css";

const MAX_WEEK = 22;
const PREMIUM_WELCOME_MESSAGE =
  "Welcome to Premium NFL AI Analyst!\n\n" +
  "I can help you with:\n" +
  "- Which team is favored to win\n" +
  "- Expected point margins\n" +
  "- Team records and performance\n" +
  "- Historical trends and patterns\n" +
  "- Anything else about the upcoming slate\n\n" +
  "How can I help you with this week's matchups?";

function extractErrorMessage(error, fallback = "Premium AI Coach is unavailable.") {
  const detail = error?.body?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  if (error?.body?.error?.message) return error.body.error.message;
  return error?.message || fallback;
}

function formatKickoff(value) {
  if (!value) return "TBD";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "TBD";
  return date.toLocaleString([], {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function DashboardStat({ label, value, detail, tone = "default" }) {
  return (
    <article className={`dashboard__stat dashboard__stat--${tone}`}>
      <span className="dashboard__statLabel">{label}</span>
      <strong className="dashboard__statValue">{value}</strong>
      <span className="dashboard__statDetail">{detail}</span>
    </article>
  );
}

function buildSeasonOptions(currentSeason, selectedSeason) {
  const baseSeason = Number.isFinite(Number(currentSeason))
    ? Number(currentSeason)
    : new Date().getFullYear();
  const seasons = new Set([selectedSeason, baseSeason, baseSeason - 1, baseSeason - 2, baseSeason + 1]);
  return Array.from(seasons)
    .filter((value) => Number.isFinite(Number(value)))
    .sort((a, b) => Number(b) - Number(a));
}

export default function Dashboard({
  authSession = null,
  onSignOut,
  schedule = [],
  week = null,
  predictions = {},
  loading = {},
  errors = {},
  history = [],
  pushHistory,
  refreshHistory,
  health = { status: "unknown", reason: null },
  seasonContext = null,
  scheduleLoading = false,
  scheduleError = null,
  loadScheduleForWeek,
  setPrediction,
  setLoading,
  setError,
}) {
  const [selectedSeason, setSelectedSeason] = useState(
    Number(seasonContext?.current_season || new Date().getFullYear())
  );
  const [selectedWeek, setSelectedWeek] = useState(
    Number(week ?? seasonContext?.display_week ?? 1)
  );
  const [isBulkLoading, setIsBulkLoading] = useState(false);
  const safeGames = Array.isArray(schedule) ? schedule : [];
  const safePredictions = predictions && typeof predictions === "object" ? predictions : {};
  const safeLoading = loading && typeof loading === "object" ? loading : {};
  const safeErrors = errors && typeof errors === "object" ? errors : {};
  const safeHistory = Array.isArray(history) ? history : [];
  const userId = authSession?.userId || null;

  const [chatMessages, setChatMessages] = useState(() => {
    if (authSession?.userId) {
      return [{ role: "assistant", text: PREMIUM_WELCOME_MESSAGE }];
    }
    return [];
  });
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);

  useEffect(() => {
    if (!userId) {
      setChatMessages([]);
      return;
    }
    setChatMessages((current) =>
      current.length ? current : [{ role: "assistant", text: PREMIUM_WELCOME_MESSAGE }]
    );
  }, [userId]);

  useEffect(() => {
    if (Number.isFinite(Number(seasonContext?.current_season))) {
      setSelectedSeason(Number(seasonContext.current_season));
    }
  }, [seasonContext?.current_season]);

  useEffect(() => {
    const nextWeek = Number(week ?? seasonContext?.display_week);
    if (Number.isFinite(nextWeek)) {
      setSelectedWeek(nextWeek);
    }
  }, [seasonContext?.display_week, week]);

  const weekValue = getGameWeek(safeGames[0]) ?? seasonContext?.display_week ?? week ?? null;
  const weekLabel = weekValue != null ? `Week ${weekValue}` : seasonContext?.label || "Next Slate";
  const nextKickoff = safeGames[0]?.kickoff || seasonContext?.next_kickoff || null;
  const healthyService = health?.status === "healthy";
  const predictionCount = Object.keys(safePredictions).length;
  const seasonOptions = useMemo(
    () => buildSeasonOptions(seasonContext?.current_season, selectedSeason),
    [seasonContext?.current_season, selectedSeason]
  );
  const selectedGames = safeGames.length;
  const resolvedHistoryCount = safeHistory.filter(
    (entry) => entry?.final_home_score != null && entry?.final_away_score != null
  ).length;

  const loadRequestedSlate = async () => {
    if (typeof loadScheduleForWeek !== "function") return;
    await loadScheduleForWeek(selectedSeason, selectedWeek);
  };

  const loadNextSlate = async () => {
    if (typeof loadScheduleForWeek !== "function") return;
    await loadScheduleForWeek(null, null);
  };

  const onPredict = async (game) => {
    const key = buildMatchupKey(game);
    if (!key || safeLoading[key]) return;

    setError?.(key, null);
    setLoading?.(key, true);

    try {
      const payload = buildPredictPayload(game);
      const prediction = await predictGame(payload, userId);
      const predictionKey = buildMatchupKey(prediction);
      setPrediction?.(key, prediction);
      if (predictionKey && predictionKey !== key) {
        setPrediction?.(predictionKey, prediction);
      }
      let refreshedFromServer = false;
      if (typeof refreshHistory === "function") {
        try {
          await refreshHistory();
          refreshedFromServer = true;
        } catch (refreshError) {
          console.warn("History refresh failed after prediction", refreshError);
        }
      }
      if (!refreshedFromServer && typeof pushHistory === "function") {
        pushHistory(prediction);
      }
    } catch (error) {
      const detail =
        error?.body?.detail?.message ||
        error?.body?.error?.message ||
        error?.body?.detail ||
        error?.message ||
        "Prediction failed";
      setError?.(key, detail);
    } finally {
      setLoading?.(key, false);
    }
  };

  async function runWithLimit(items, limit, worker) {
    const queue = [...items];
    const workers = Array.from({ length: Math.max(1, limit) }, async () => {
      while (queue.length) {
        const item = queue.shift();
        if (item == null) return;
        await worker(item);
      }
    });
    await Promise.all(workers);
  }

  const onPredictAll = async (targetGames = null) => {
    if (isBulkLoading) return;
    const sourceGames = Array.isArray(targetGames) ? targetGames : safeGames;
    setIsBulkLoading(true);
    try {
      const targets = sourceGames.filter((game) => {
        const key = buildMatchupKey(game);
        return key && !safePredictions[key] && !safeLoading[key];
      });
      await runWithLimit(targets, 4, onPredict);
    } finally {
      setIsBulkLoading(false);
    }
  };

  const onReset = (gameOrMatchup) => {
    const key = buildMatchupKey(gameOrMatchup);
    if (!key) return;
    setPrediction?.(key, null);
    setError?.(key, null);
    setLoading?.(key, false);
  };

  const handleSendMessage = async (presetMessage = "") => {
    const message = String(presetMessage || chatInput).trim();
    if (!message || chatLoading) return;

    setChatMessages((current) => [...current, { role: "user", text: message }]);
    if (!presetMessage) {
      setChatInput("");
    }
    setChatLoading(true);

    try {
      const response = await premiumChat(message, selectedSeason, selectedWeek, userId);
      setChatMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: response?.reply || "I could not generate a Premium AI response for that question.",
        },
      ]);
    } catch (error) {
      console.error("[Dashboard] premium chat failed", error);
      setChatMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: `I could not reach the Premium AI Coach. ${extractErrorMessage(error)}`,
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <>
      <NavBar
        authSession={authSession}
        onSignOut={onSignOut}
        state={{
          health,
          title: "Prediction Dashboard",
          heroSubtitle: seasonContext?.message || "Forecast the current NFL slate and compare every matchup.",
          subtitle: `${safeGames.length} matchup${safeGames.length === 1 ? "" : "s"} ready to forecast`,
          healthLabel: healthyService ? "Service: Live" : `Service: ${health?.status ?? "unknown"}`,
          weekLabel,
        }}
      />

      <main className="dashboard" aria-label="NFL Predict Dashboard">
        <section className="dashboard__hero">
          <div className="dashboard__titleWrap">
            <p className="dashboard__eyebrow">Forecast workspace</p>
            <h1 className="dashboard__title">Forecast any slate with one shared prediction flow.</h1>
            <p className="dashboard__subtitle">
              {seasonContext?.message || "Browse a specific week, run predictions, and keep history aligned with the backend."}
            </p>
          </div>

          <div className="dashboard__actions">
            <div className="dashboard__controlGroup">
              <label className="dashboard__control">
                <span>Season</span>
                <select
                  value={selectedSeason}
                  onChange={(event) => setSelectedSeason(Number(event.target.value))}
                >
                  {seasonOptions.map((seasonOption) => (
                    <option key={seasonOption} value={seasonOption}>
                      {seasonOption}
                    </option>
                  ))}
                </select>
              </label>

              <label className="dashboard__control">
                <span>Week</span>
                <select
                  value={selectedWeek}
                  onChange={(event) => setSelectedWeek(Number(event.target.value))}
                >
                  {Array.from({ length: MAX_WEEK }, (_, index) => index + 1).map((weekOption) => (
                    <option key={weekOption} value={weekOption}>
                      Week {weekOption}
                    </option>
                  ))}
                </select>
              </label>

              <button
                type="button"
                className="dashboard__btn"
                onClick={loadRequestedSlate}
                disabled={scheduleLoading}
                aria-busy={scheduleLoading ? "true" : "false"}
              >
                {scheduleLoading ? "Loading..." : "Load Slate"}
              </button>

              <button
                type="button"
                className="dashboard__btn dashboard__btn--ghost"
                onClick={loadNextSlate}
                disabled={scheduleLoading}
              >
                Next Slate
              </button>
            </div>
          </div>
        </section>

        <section className="dashboard__summaryGrid" aria-label="Slate summary">
          <DashboardStat
            label="Active slate"
            value={weekLabel}
            detail={
              selectedSeason
                ? `Season ${selectedSeason}`
                : seasonContext?.phase === "offseason"
                  ? "Offseason mode"
                  : "Upcoming matchups ready"
            }
            tone="accent"
          />
          <DashboardStat
            label="Games loaded"
            value={selectedGames}
            detail={selectedGames ? "Cards ready for prediction" : "No games returned for this filter"}
          />
          <DashboardStat
            label="Next kickoff"
            value={formatKickoff(nextKickoff)}
            detail="Local browser time"
          />
          <DashboardStat
            label="History resolved"
            value={resolvedHistoryCount}
            detail={`${predictionCount} prediction${predictionCount === 1 ? "" : "s"} in this session`}
            tone={healthyService ? "success" : "warning"}
          />
        </section>

        {scheduleError ? (
          <section className="dashboard__notice dashboard__notice--error" role="alert">
            <p>
              <strong>Schedule load failed:</strong> {scheduleError}
            </p>
            <button type="button" className="dashboard__btn" onClick={loadRequestedSlate}>
              Try again
            </button>
          </section>
        ) : null}

        {!healthyService ? (
          <section className="dashboard__notice" role="status">
            <p>
              <strong>Prediction service is degraded.</strong> Schedule and history remain available, but
              model blockers may prevent new forecasts until backend readiness is restored.
            </p>
          </section>
        ) : null}

        <TeamGrid
          week={weekValue ?? selectedWeek}
          games={safeGames}
          isLoading={Boolean(scheduleLoading)}
          predictions={safePredictions}
          loading={safeLoading}
          errors={safeErrors}
          onPredict={onPredict}
          onReset={onReset}
          onPredictAll={onPredictAll}
          isBulkLoading={isBulkLoading}
        />

        {/* Premium AI Chat Coach Assistant Widget */}
        <div className="premium-chat-container">
          {!isChatOpen ? (
            <button
              type="button"
              className="premium-chat-trigger-btn"
              onClick={() => setIsChatOpen(true)}
              aria-label="Open Premium AI Coach Chat"
            >
              <span className="premium-chat-trigger-icon">✨</span>
              <span className="premium-chat-trigger-text">Premium AI Coach</span>
              <span className="premium-chat-trigger-badge">Live</span>
            </button>
          ) : (
            <div className="premium-chat-window">
              <header className="premium-chat-header">
                <div className="premium-chat-header-title">
                  <span className="premium-chat-sparkle">✨</span>
                  <h3>Premium NFL AI Analyst</h3>
                </div>
                <button
                  type="button"
                  className="premium-chat-close-btn"
                  onClick={() => setIsChatOpen(false)}
                  aria-label="Close Chat"
                >
                  ✕
                </button>
              </header>

              <div className="premium-chat-messages">
                {chatMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`premium-chat-message-bubble premium-chat-message-bubble--${msg.role}`}
                  >
                    <div className="premium-chat-message-avatar">
                      {msg.role === "assistant" ? "🤖" : "🏈"}
                    </div>
                    <div className="premium-chat-message-content">
                      {String(msg.text || "").split("\n").map((line, lIdx) => {
                        const trimmed = line.trim();
                        if (trimmed.startsWith("###")) {
                          return <h4 key={lIdx}>{trimmed.replace("###", "")}</h4>;
                        } else if (trimmed.startsWith("##")) {
                          return <h3 key={lIdx}>{trimmed.replace("##", "")}</h3>;
                        } else if (trimmed.startsWith("*") || trimmed.startsWith("-")) {
                          return <li key={lIdx}>{trimmed.substring(1).trim()}</li>;
                        } else if (trimmed) {
                          return <p key={lIdx}>{trimmed}</p>;
                        }
                        return <div key={lIdx} style={{ height: "6px" }} />;
                      })}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="premium-chat-message-bubble premium-chat-message-bubble--assistant premium-chat-loading-bubble">
                    <div className="premium-chat-message-avatar">🤖</div>
                    <div className="premium-chat-message-content">
                      <span className="premium-chat-loading-dots">
                        <span>.</span><span>.</span><span>.</span>
                      </span>
                    </div>
                  </div>
                )}
              </div>

              <div className="premium-chat-suggestions">
                <button
                  type="button"
                  onClick={() => handleSendMessage("Who is favored to win the upcoming matchups?")}
                  disabled={chatLoading}
                >
                  💡 Next Slate Favorites?
                </button>
                <button
                  type="button"
                  onClick={() => handleSendMessage("Which team has the highest point differential historically?")}
                  disabled={chatLoading}
                >
                  📊 Highest Point Diff?
                </button>
              </div>

              <footer className="premium-chat-footer">
                <input
                  type="text"
                  placeholder="Ask about team records, win margins..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
                  disabled={chatLoading}
                />
                <button
                  type="button"
                  onClick={() => handleSendMessage()}
                  disabled={chatLoading || !chatInput.trim()}
                >
                  Send
                </button>
              </footer>
            </div>
          )}
        </div>
      </main>
    </>
  );
}
