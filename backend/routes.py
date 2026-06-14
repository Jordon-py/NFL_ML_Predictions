# ==========================================
# File: backend/routes.py
# Role: Legacy / Compatibility Routes
# Input Data: HTTP requests.
# Output Data: JSON responses (Legacy shapes).
# Dependencies: backend.main_helpers, backend.services.prediction_service
# Notes: This module is not mounted by backend.main and is kept only as a
# compatibility reference while the canonical API surface lives in backend.main.
# ==========================================

from __future__ import annotations
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.utils.main_helpers import (
    get_schedule,
    select_next_week_rows,
    _pick_col,
    parse_kickoff,
    _HOME_COLS,
    _AWAY_COLS,
    _STADIUM_COLS,
)

from backend.prediction_store import (
    append_prediction_record,
    build_prediction_user_context,
    get_prediction_history,
    get_prediction_history_count,
)
from backend.pipeline_models import StoredPredictionRequest

router = APIRouter()
log = logging.getLogger(__name__)


def _resolve_user_context(request: Request):
    return build_prediction_user_context(request.headers.get("X-User-Id"))


# -------------------------
# Legacy Schemas
# -------------------------
class HealthResponse(BaseModel):
    status: str
    reason: str
    started_at: Optional[str] = None

class DebugResponse(BaseModel):
    started_at: Optional[str] = None
    models_loaded: bool
    models_dir: Optional[str] = None
    dataset_rows: int
    dataset_cols: int
    dataset_sample_cols: List[str]

class PredictionRequest(BaseModel):
    home_team: str = Field(..., examples=["KC"])
    away_team: str = Field(..., examples=["BUF"])
    season: int = Field(..., examples=[2025])
    week: int = Field(..., examples=[15])

class PredictionResponse(BaseModel):
    home_score: float
    away_score: float
    point_diff: float
    home_win_probability: float
    away_win_probability: float
    win_classifier_used: bool
    prediction_source: str

class ScheduleGame(BaseModel):
    season: int
    week: int
    kickoff: Optional[datetime] = None
    home_team: str
    away_team: str
    game_id: Optional[str] = None
    stadium: Optional[str] = None

class NextWeekGamesResponse(BaseModel):
    games: List[ScheduleGame]

class TeamMeta(BaseModel):
    logoUrl: str
    name: Optional[str] = None
    primaryColor: Optional[str] = None
    secondaryColor: Optional[str] = None
    wordmark: Optional[str] = None

class TeamLogosResponse(BaseModel):
    teams: Dict[str, TeamMeta]

class PredictedGame(BaseModel):
    season: int
    week: int
    kickoff: Optional[datetime] = None
    home_team: str
    away_team: str
    game_id: Optional[str] = None
    prediction: PredictionResponse

class NextWeekPredictionsResponse(BaseModel):
    games: List[PredictedGame]


class HistoryIntelligenceResponse(BaseModel):
    total_predictions: int
    average_confidence: float
    average_margin: float
    confidence_calibration_gap: float
    strongest_pick: Optional[Dict[str, Any]] = None
    closest_matchup: Optional[Dict[str, Any]] = None
    most_frequent_team: Optional[str] = None
    generated_at: datetime

# -------------------------
# Routes
# -------------------------

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    # Use service availability as health check
    service = getattr(request.app.state, "service", None)
    started_at = getattr(request.app.state, "started_at", None)
    if not service:
        return HealthResponse(status="unhealthy", reason="prediction_service_not_loaded", started_at=started_at)
    return HealthResponse(status="healthy", reason="ok", started_at=started_at)

@router.get("/debug", response_model=DebugResponse)
async def debug(request: Request) -> DebugResponse:
    models = getattr(request.app.state, "models", None)
    ds: pd.DataFrame = getattr(request.app.state, "dataset", pd.DataFrame())
    started_at = getattr(request.app.state, "started_at", None)
    return DebugResponse(
        started_at=started_at,
        models_loaded=bool(models),
        models_dir=(models.get("models_dir") if models else None),
        dataset_rows=int(0 if ds is None else len(ds)),
        dataset_cols=int(0 if ds is None else ds.shape[1]),
        dataset_sample_cols=list(ds.columns[:30]) if ds is not None and not ds.empty else [],
    )

@router.get("/schedule/next-week", response_model=NextWeekGamesResponse)
async def schedule_next_week(request: Request, season: int = 2025) -> NextWeekGamesResponse:
    df = await asyncio.to_thread(get_schedule, season=season)
    df_next, use_season, use_week = select_next_week_rows(df)

    home_col = _pick_col(df_next, _HOME_COLS)
    away_col = _pick_col(df_next, _AWAY_COLS)
    stadium_col = _pick_col(df_next, _STADIUM_COLS)

    games: List[ScheduleGame] = []
    for _, r in df_next.iterrows():
        home = str(r.get(home_col, "") if home_col else r.get("home", "")).strip().upper()
        away = str(r.get(away_col, "") if away_col else r.get("away", "")).strip().upper()
        if not home or not away:
            continue
        stadium = str(r.get(stadium_col, "") if stadium_col else r.get("stadium", "")).strip()
        game_id = f"{int(use_season)}-{int(use_week)}-{home}-{away}"

        games.append(
            ScheduleGame(
                season=int(use_season),
                week=int(use_week),
                kickoff=parse_kickoff(r),
                home_team=home,
                away_team=away,
                game_id=game_id,
                stadium=stadium,
            )
        )

    return NextWeekGamesResponse(games=games)

@router.get("/teams/logos", response_model=TeamLogosResponse)
async def team_logos(request: Request) -> TeamLogosResponse:
    cached = getattr(request.app.state, "team_logos", None)
    if isinstance(cached, dict) and cached:
        return TeamLogosResponse(teams=cached)

    # Use main_helpers logic, essentially re-run but with cache intent
    # Note: main.py usually hydrates app.state.team_logos on startup
    # If not found, we try to load it

    # We can try to reuse the global logic if we knew the path, but let's just return empty if not cached
    # to encourage main.py being the source of truth.
    return TeamLogosResponse(teams={})

@router.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, request: Request) -> PredictionResponse:
    """Legacy compatibility endpoint; the UI uses api_runtime.predict_game."""
    service: Optional[Any] = getattr(request.app.state, "service", None)
    if not service:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")

    try:
        UnifiedRequest = PredictionRequest

        # Convert local legacy req to unified req
        unified_req = UnifiedRequest(
            home_team=req.home_team,
            away_team=req.away_team,
            season=req.season,
            week=req.week
        )

        res = service.predict(unified_req)

        # Flatten
        flat_pred = PredictionResponse(
            home_score=res.scores.home_score,
            away_score=res.scores.away_score,
            point_diff=res.scores.home_score - res.scores.away_score,
            home_win_probability=res.winner.proba_home,
            away_win_probability=res.winner.proba_away,
            win_classifier_used=res.win_classifier_used,
            prediction_source=res.prediction_source,
        )

        # best-effort history append
        try:
            append_prediction_record(
                _resolve_user_context(request),
                StoredPredictionRequest(**req.model_dump()),
                flat_pred.model_dump(),
            )
        except Exception as exc:
            log.warning("Legacy prediction history append failed: %s", exc)

        return flat_pred

    except Exception as e:
        log.error(f"Legacy predict failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@router.get("/predict/next-week", response_model=NextWeekPredictionsResponse)
async def predict_next_week(request: Request, season: int = 2025) -> NextWeekPredictionsResponse:
    # Re-use local schedule fetch
    sched = await schedule_next_week(request, season=season)
    out: List[PredictedGame] = []

    # We can just loop and call our local predict for convenience/consistency
    for g in sched.games:
        try:
            p_req = PredictionRequest(
                home_team=g.home_team,
                away_team=g.away_team,
                season=g.season,
                week=g.week
            )
            pred = await predict(p_req, request)
            out.append(PredictedGame(
                season=g.season,
                week=g.week,
                kickoff=g.kickoff,
                home_team=g.home_team,
                away_team=g.away_team,
                game_id=g.game_id,
                prediction=pred
            ))
        except Exception as e:
            log.warning(f"Skipping game {g.home_team} vs {g.away_team}: {e}")
            continue

    return NextWeekPredictionsResponse(games=out)

@router.get("/status/overview")
async def status_overview(request: Request):
    service = getattr(request.app.state, "service", None)
    ds = getattr(request.app.state, "dataset", None)

    ready = bool(service)
    return {
        "health": {
            "status": "healthy" if ready else "unhealthy",
            "reason": "ok" if ready else "prediction_service_not_loaded",
        },
        "dataset": {"rows": len(ds) if ds is not None else 0},
        "history": {
            "total_predictions": get_prediction_history_count(_resolve_user_context(request)),
            "win_rate": None,
            "note": "win_rate requires actual outcomes",
        },
    }

@router.get("/history")
async def history(request: Request, limit: int = 100):
    return await asyncio.to_thread(get_prediction_history, _resolve_user_context(request), limit=limit)


@router.get("/history/intelligence", response_model=HistoryIntelligenceResponse)
async def history_intelligence(request: Request, limit: int = 250) -> HistoryIntelligenceResponse:
    records = await asyncio.to_thread(get_prediction_history, _resolve_user_context(request), limit=limit)
    if not records:
        return HistoryIntelligenceResponse(
            total_predictions=0,
            average_confidence=0.0,
            average_margin=0.0,
            confidence_calibration_gap=0.0,
            generated_at=datetime.utcnow(),
        )

    total = len(records)
    confidence_values: List[float] = []
    margin_values: List[float] = []
    team_counts: Dict[str, int] = {}
    strongest_pick: Optional[Dict[str, Any]] = None
    closest_matchup: Optional[Dict[str, Any]] = None

    for item in records:
        home = str(item.get("home_team", "")).upper()
        away = str(item.get("away_team", "")).upper()
        if home:
            team_counts[home] = team_counts.get(home, 0) + 1
        if away:
            team_counts[away] = team_counts.get(away, 0) + 1

        home_win_probability = float(item.get("home_win_probability", 0.0) or 0.0)
        away_win_probability = float(item.get("away_win_probability", 0.0) or 0.0)
        point_diff = float(item.get("point_diff", 0.0) or 0.0)
        confidence = max(home_win_probability, away_win_probability)
        margin = abs(point_diff)

        confidence_values.append(confidence)
        margin_values.append(margin)

        if strongest_pick is None or confidence > strongest_pick["confidence"]:
            strongest_pick = {
                "season": item.get("season"),
                "week": item.get("week"),
                "home_team": home,
                "away_team": away,
                "predicted_winner": home if point_diff >= 0 else away,
                "confidence": round(confidence, 4),
                "predicted_margin": round(margin, 2),
            }

        if closest_matchup is None or margin < closest_matchup["predicted_margin"]:
            closest_matchup = {
                "season": item.get("season"),
                "week": item.get("week"),
                "home_team": home,
                "away_team": away,
                "predicted_margin": round(margin, 2),
                "confidence": round(confidence, 4),
            }

    avg_confidence = sum(confidence_values) / total
    avg_margin = sum(margin_values) / total
    calibration_gap = abs(avg_confidence - 0.5) - min(avg_margin / 20.0, 0.5)
    most_frequent_team = max(team_counts.items(), key=lambda pair: pair[1])[0] if team_counts else None

    return HistoryIntelligenceResponse(
        total_predictions=total,
        average_confidence=round(avg_confidence, 4),
        average_margin=round(avg_margin, 2),
        confidence_calibration_gap=round(calibration_gap, 4),
        strongest_pick=strongest_pick,
        closest_matchup=closest_matchup,
        most_frequent_team=most_frequent_team,
        generated_at=datetime.utcnow(),
    )

@router.post("/train")
@router.post("/retrain")
def train_stub() -> Dict[str, str]:
    return {"status": "not_implemented", "message": "Use backend CLI or admin tools"}
