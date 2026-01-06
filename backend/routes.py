# ==========================================
# File: backend/routes.py
# Role: Legacy / Compatibility Routes
# Input Data: HTTP requests.
# Output Data: JSON responses (Legacy shapes).
# Dependencies: backend.main_helpers, backend.services.prediction_service
# Notes: Delegates to request.app.state.service where possible.
# ==========================================

from __future__ import annotations
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.main_helpers import (
    get_schedule,
    _select_next_week_rows,
    _pick_col,
    _parse_kickoff,
    _load_team_logos_map,
    _append_prediction_history_to_disk,
    _HOME_COLS,
    _AWAY_COLS,
    _GAME_ID_COLS,
    _STADIUM_COLS,
)
from backend.services.prediction_service import PredictionService

router = APIRouter()
log = logging.getLogger(__name__)

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

# -------------------------
# Routes
# -------------------------

@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    # Use service availability as health check
    service = getattr(request.app.state, "service", None)
    started_at = getattr(request.app.state, "started_at", None)
    if not service:
        return HealthResponse(status="unhealthy", reason="prediction_service_not_loaded", started_at=started_at)
    return HealthResponse(status="healthy", reason="ok", started_at=started_at)

@router.get("/debug", response_model=DebugResponse)
def debug(request: Request) -> DebugResponse:
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
def schedule_next_week(request: Request, season: int = 2025) -> NextWeekGamesResponse:
    df = get_schedule(season=season)
    df_next, use_season, use_week = _select_next_week_rows(df)

    home_col = _pick_col(df_next, _HOME_COLS)
    away_col = _pick_col(df_next, _AWAY_COLS)
    game_id_col = _pick_col(df_next, _GAME_ID_COLS)
    stadium_col = _pick_col(df_next, _STADIUM_COLS)

    games: List[ScheduleGame] = []
    for _, r in df_next.iterrows():
        home = str(r.get(home_col, "") if home_col else r.get("home", "")).strip().upper()
        away = str(r.get(away_col, "") if away_col else r.get("away", "")).strip().upper()
        if not home or not away:
            continue
        stadium = str(r.get(stadium_col, "") if stadium_col else r.get("stadium", "")).strip()
        game_id = None
        if game_id_col:
            raw_id = r.get(game_id_col)
            if pd.notna(raw_id):
                game_id = str(raw_id).strip()
        if not game_id:
            game_id = f"{use_season}-{use_week}-{home}-{away}"

        games.append(
            ScheduleGame(
                season=int(use_season),
                week=int(use_week),
                kickoff=_parse_kickoff(r),
                home_team=home,
                away_team=away,
                game_id=game_id,
                stadium=stadium,
            )
        )

    return NextWeekGamesResponse(games=games)

@router.get("/teams/logos", response_model=TeamLogosResponse)
def team_logos(request: Request) -> TeamLogosResponse:
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
def predict(req: PredictionRequest, request: Request) -> PredictionResponse:
    service: Optional[PredictionService] = getattr(request.app.state, "service", None)
    if not service:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")

    try:
        # Service returns a nested backend.schemas.PredictionResponse
        # We need to flatten it to our local PredictionResponse
        from backend.schemas import PredictionRequest as UnifiedRequest
        
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
             _append_prediction_history_to_disk(req.model_dump(), flat_pred.model_dump())
        except Exception:
            pass

        return flat_pred

    except Exception as e:
        log.error(f"Legacy predict failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@router.get("/predict/next-week", response_model=NextWeekPredictionsResponse)
def predict_next_week(request: Request, season: int = 2025) -> NextWeekPredictionsResponse:
    # Re-use local schedule fetch
    sched = schedule_next_week(request, season=season)
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
            pred = predict(p_req, request)
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
def status_overview(request: Request):
    service = getattr(request.app.state, "service", None)
    ds = getattr(request.app.state, "dataset", None)
    hist = getattr(request.app.state, "prediction_history", [])
    
    return {
        "health": {"status": "healthy" if service else "unhealthy"},
        "dataset": {"rows": len(ds) if ds is not None else 0},
        "history": {"metrics": {"total_predictions": len(hist) if isinstance(hist, list) else 0}}
    }

@router.get("/history")
def history(request: Request, limit: int = 100):
    hist = getattr(request.app.state, "prediction_history", []) or []
    if isinstance(hist, list):
        return hist[:limit]
    return []

@router.post("/train")
@router.post("/retrain")
def train_stub() -> Dict[str, str]:
    return {"status": "not_implemented", "message": "Use backend CLI or admin tools"}
