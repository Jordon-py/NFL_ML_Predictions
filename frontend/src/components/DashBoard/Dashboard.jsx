import { useEffect, useState } from "react";
import TeamGrid from "../Card/TeamGrid.jsx";
import NavBar from "../NavBar/NavBar.jsx";
import { getNextWeekSchedule, predictGame } from "../../api/client.js";
import {
  buildMatchupKey,
  buildPredictPayload,
  getGameWeek,
} from "../../utils/gameUtils.js";
import "./Dashboard.css";

function removeKey(map, key) {
  if (!key || !Object.prototype.hasOwnProperty.call(map, key)) return map;
  const next = { ...map };
  delete next[key];
  return next;
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

export default function Dashboard({
  authSession = null,
  onSignOut,
  pushHistory,
  health = { status: "unknown", reason: null },
  seasonContext = null,
}) {
  const [games, setGames] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [loadingMap, setLoadingMap] = useState({});
  const [errorsMap, setErrorsMap] = useState({});
  const [isBulkLoading, setIsBulkLoading] = useState(false);

  const safeGames = Array.isArray(games) ? games : [];
  const userId = authSession?.userId || null;

  const loadSchedule = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const schedule = await getNextWeekSchedule();
      setGames(Array.isArray(schedule) ? schedule : []);
    } catch (e) {
      setGames([]);
      setError(e?.message ?? "Failed to load schedule");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadSchedule();
  }, []);

  const onPredict = async (game) => {
    const key = buildMatchupKey(game);
    if (!key || loadingMap[key]) return;

    setErrorsMap((prev) => removeKey(prev, key));
    setLoadingMap((prev) => ({ ...prev, [key]: true }));

    try {
      const payload = buildPredictPayload(game);
      const prediction = await predictGame(payload, userId);
      const predictionKey = buildMatchupKey(prediction);

      setPredictions((prev) => ({
        ...prev,
        [key]: prediction,
        ...(predictionKey && predictionKey !== key ? { [predictionKey]: prediction } : {}),
      }));

      if (typeof pushHistory === "function") {
        pushHistory(prediction);
      }
    } catch (e) {
      setErrorsMap((prev) => ({
        ...prev,
        [key]: e?.message ?? "Prediction failed",
      }));
    } finally {
      setLoadingMap((prev) => removeKey(prev, key));
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

  const onPredictAll = async () => {
    if (isBulkLoading) return;
    setIsBulkLoading(true);
    try {
      const targets = safeGames.filter((g) => {
        const key = buildMatchupKey(g);
        return key && !predictions[key] && !loadingMap[key];
      });
      await runWithLimit(targets, 4, onPredict);
    } finally {
      setIsBulkLoading(false);
    }
  };

  const onReset = (gameOrMatchup) => {
    const key = buildMatchupKey(gameOrMatchup);
    if (!key) return;

    setPredictions((prev) => removeKey(prev, key));
    setErrorsMap((prev) => removeKey(prev, key));
    setLoadingMap((prev) => removeKey(prev, key));
  };

  const weekValue = getGameWeek(safeGames[0]) ?? seasonContext?.display_week ?? null;
  const weekLabel = weekValue != null ? `Week ${weekValue}` : seasonContext?.label || "Next Slate";
  const nextKickoff = safeGames[0]?.kickoff || seasonContext?.next_kickoff || null;
  const healthyService = health?.status === "healthy";
  const predictionCount = Object.keys(predictions).length;

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
            <h1 className="dashboard__title">Forecast the slate with cleaner context and faster feedback.</h1>
            <p className="dashboard__subtitle">
              {seasonContext?.message || "Run predictions one game at a time or fan out across the full board."}
            </p>
          </div>

          <div className="dashboard__actions">
            <button
              type="button"
              className="dashboard__btn"
              onClick={loadSchedule}
              disabled={isLoading}
              aria-busy={isLoading ? "true" : "false"}
            >
              {isLoading ? "Refreshing..." : "Refresh Schedule"}
            </button>
          </div>
        </section>

        <section className="dashboard__summaryGrid" aria-label="Slate summary">
          <DashboardStat
            label="Active slate"
            value={weekLabel}
            detail={seasonContext?.phase === "offseason" ? "Offseason mode" : "Upcoming matchups ready"}
            tone="accent"
          />
          <DashboardStat
            label="Games loaded"
            value={safeGames.length}
            detail={safeGames.length ? "Cards ready for prediction" : "No matchups returned yet"}
          />
          <DashboardStat
            label="Next kickoff"
            value={formatKickoff(nextKickoff)}
            detail="Local browser time"
          />
          <DashboardStat
            label="Predictions this session"
            value={predictionCount}
            detail={healthyService ? "Saved to your signed-in history" : "Backend status needs attention"}
            tone={healthyService ? "success" : "warning"}
          />
        </section>

        {error && (
          <section className="dashboard__notice dashboard__notice--error" role="alert">
            <p>
              <strong>Schedule load failed:</strong> {error}
            </p>
            <button type="button" className="dashboard__btn" onClick={loadSchedule}>
              Try again
            </button>
          </section>
        )}

        <TeamGrid
          week={weekValue ?? undefined}
          games={safeGames}
          isLoading={Boolean(isLoading)}
          predictions={predictions}
          loading={loadingMap}
          errors={errorsMap}
          onPredict={onPredict}
          onReset={onReset}
          onPredictAll={onPredictAll}
          isBulkLoading={isBulkLoading}
        />
      </main>
    </>
  );
}
