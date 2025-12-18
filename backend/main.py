# ==========================================
# File: backend/main.py
# Role: FastAPI app entrypoint and routing.
# Input Data: HTTP requests (JSON bodies, query params).
# Output Data: JSON responses for API endpoints.
# Dependencies: __future__, datetime, logging, os, pathlib, pandas
# Notes: Loads models/datasets at startup, validates schema, and mounts legacy routes under /legacy.
# ==========================================

"""
FILE: backend/main.py
PURPOSE: FastAPI application for NFL ML Predictions.
DATA SHAPES:
  - PredictionRequest: { home_team: str, away_team: str, season: int, week: int }
  - UnifiedPredictionResponse: { home_score, away_score, point_diff, probabilities, ... }
KEY FUNCTIONS/CLASSES:
  - lifespan: Preloads models and datasets on startup.
  - predict: Unified inference endpoint delegating to PredictionService.
  - legacy routes: APIRouter mounted under /legacy for backward compatibility.
SIDE EFFECTS / I/O: Loads ML artifacts from MODELS_DIR, reads dataset from DATA_DIR, reads team logos CSV.
ERROR HANDLING: 404 for missing games, 503 for uninitialized models.
DEPENDENCIES: FastAPI, Pydantic, PredictionService, InferenceBundle.
"""
# -------------------------------------
# IMPORTS
# -------------------------------------
# backend/main.py

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    UnifiedPredictionResponse,
    HealthResponse,
    StatusOverviewResponse,
    HistoryResponse,
)
from backend.services.prediction_service import PredictionService
from backend.services.inference_row import build_model_input_row
from backend.config import DATA_DIR as CFG_DATA_DIR, MODELS_DIR as CFG_MODELS_DIR, resolve_cors, TRUTHY
from backend.main_helpers import (
    load_inference_bundle,
    load_dataset_df,
    _append_prediction_history_to_disk,
    prediction_history_entries,
    _prediction_history_lock,
)
from backend.ollama.llm_ollama import explain_prediction as llm_explain_prediction, chat_messages as llm_chat_messages
from backend.routes import (
    _infer_next_week as _infer_next_week_routes,
    _load_schedule_df as _load_schedule_df_routes,
    _load_team_logos_map as _load_team_logos_map_routes,
    _parse_kickoff as _parse_kickoff_routes,
    router as legacy_router,
)
if __name__ == "__main__" and __package__ is None:
    # Allow running as a script by ensuring repo root is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -------------------------------------
# GLOBALS
# ------------------------------------- 
# Load environment variables
load_dotenv(dotenv_path=".env")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global state
state: Dict[str, Any] = {
    "bundle": None,
    "dataset": None,
    "service": None,
    "team_logos": None,
}

def _build_game_id(season: int, week: int, home_team: str, away_team: str) -> str:
    parts = [season, week, home_team, away_team]
    return "-".join(str(p) for p in parts if p is not None and str(p).strip())

def _get_team_meta_map() -> Dict[str, Dict[str, str]]:
    """Load team metadata once and cache it for schedule/prediction enrichment."""
    cached = state.get("team_logos")
    if isinstance(cached, dict):
        return cached

    backend_dir = Path(__file__).resolve().parent
    repo_root = backend_dir.parent
    candidates = [
        backend_dir / "team_logo.csv",
        backend_dir / "team_logos.csv",
        backend_dir / "data" / "team_logo.csv",
        backend_dir / "data" / "team_logos.csv",
        repo_root / "team_logo.csv",
        repo_root / "team_logos.csv",
        repo_root / "data" / "team_logo.csv",
        repo_root / "data" / "team_logos.csv",
    ]
    team_map: Dict[str, Dict[str, str]] = {}
    for csv_path in candidates:
        if csv_path.exists():
            team_map = _load_team_logos_map_routes(csv_path)
            break

    state["team_logos"] = team_map or {}
    return state["team_logos"]

# -------------------------------------
# FUNCTIONS ----- 
# -------------------------------------
def _build_prediction_payload(req: PredictionRequest, res: PredictionResponse) -> Dict[str, Any]:
    """Flatten model output into the unified API response shape."""
    home_code = str(req.home_team).strip().upper()
    away_code = str(req.away_team).strip().upper()
    home_score = float(res.scores.home_score)
    away_score = float(res.scores.away_score)
    team_meta = _get_team_meta_map()
    home_meta = team_meta.get(home_code, {})
    away_meta = team_meta.get(away_code, {})

    payload = {
        "home_score": home_score,
        "away_score": away_score,
        "point_diff": home_score - away_score,
        "home_win_probability": float(res.winner.proba_home),
        "away_win_probability": float(res.winner.proba_away),
        "prediction_source": res.prediction_source,
        "win_classifier_used": res.win_classifier_used,
        "simulation_metrics": (
            res.simulation_metrics.model_dump() if res.simulation_metrics is not None else None
        ),
        "game_id": _build_game_id(req.season, req.week, home_code, away_code),
        "season": req.season,
        "week": req.week,
        "home_team": home_code,
        "away_team": away_code,
        "home_name": home_meta.get("name") or home_code,
        "away_name": away_meta.get("name") or away_code,
    }
    return payload

def _resolve_expected_features(bundle: Any) -> list[str]:
    pre = getattr(bundle, "preprocessor", None)
    features_in = getattr(pre, "feature_names_in_", None)
    if features_in is not None:
        expected = list(features_in)
        if expected:
            return expected
    return list(getattr(bundle, "raw_feature_columns", None) or [])

def _validate_feature_schema(bundle: Any, dataset: pd.DataFrame) -> None:
    expected = _resolve_expected_features(bundle)
    if not expected:
        raise RuntimeError("Model feature list missing; cannot validate schema.")
    missing = [c for c in expected if c not in dataset.columns]
    if missing:
        sample = ", ".join(missing[:25])
        raise RuntimeError(
            f"Dataset missing {len(missing)} model features. Sample: {sample}"
        )

def _extract_prediction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Support explain payloads with either {prediction:{...}} or flat fields."""
    pred = payload.get("prediction")
    if isinstance(pred, dict):
        return pred
    if any(k in payload for k in ("home_score", "away_score", "home_win_probability")):
        return payload
    return {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models and dataset
    try:
        log.info("Starting up: Loading model bundle and dataset...")
        state["bundle"] = load_inference_bundle(CFG_MODELS_DIR)
        state["dataset"] = load_dataset_df(CFG_DATA_DIR)
        _validate_feature_schema(state["bundle"], state["dataset"])
        state["service"] = PredictionService(state["bundle"], state["dataset"])
        _get_team_meta_map()
        log.info("Startup complete: Models and dataset ready.")
    except Exception as e:
        log.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield

app = FastAPI(title="NFL ML Predictions API", lifespan=lifespan)

# CORS Middleware
cors_origins, cors_origin_regex = resolve_cors()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount legacy router with a safe prefix to avoid clobbering unified endpoints.
app.include_router(legacy_router, prefix="/legacy")

def _require_ready() -> PredictionService:
    if state["service"] is None:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized.")
    return state["service"]

@app.get("/health", response_model=HealthResponse)
async def health():
    if state["service"]:
        return HealthResponse(status="ok", mode="ml-inference", reason="Models loaded and ready")
    return HealthResponse(status="initializing", mode="none", reason="Startup in progress or failed")

@app.get("/schedule/next-week")
async def get_next_week_schedule(season: int = 2025) -> Dict[str, Any]:
    df = _load_schedule_df_routes(season)
    if df is None or df.empty:
        return {"games": []}

    use_season, nxt_week = _infer_next_week_routes(df)
    team_meta = _get_team_meta_map()
    if "week" in df.columns:
        dfw = df[pd.to_numeric(df["week"], errors="coerce") == nxt_week].copy()
    else:
        dfw = df.copy()

    games = []
    for _, row in dfw.iterrows():
        home = str(row.get("home_team", row.get("home", ""))).strip().upper()
        away = str(row.get("away_team", row.get("away", ""))).strip().upper()
        if not home or not away:
            continue
        home_info = team_meta.get(home, {})
        away_info = team_meta.get(away, {})
        game_id = str(row.get("game_id", "")).strip() or _build_game_id(use_season, nxt_week, home, away)
        games.append({
            "season": int(use_season),
            "week": int(nxt_week),
            "kickoff": _parse_kickoff_routes(row),
            "home_team": home,
            "away_team": away,
            "game_id": game_id,
            "home_abbr": home,
            "away_abbr": away,
            "home_logo": home_info.get("logoUrl"),
            "away_logo": away_info.get("logoUrl"),
            "home_name": home_info.get("name") or home,
            "away_name": away_info.get("name") or away,
        })

    return {"games": games}

@app.get("/debug")
async def debug() -> Dict[str, Any]:
    origins, origin_regex = resolve_cors()
    dataset = state["dataset"]
    rows = int(len(dataset)) if dataset is not None else 0
    cols = int(dataset.shape[1]) if dataset is not None else 0
    return {
        "status": "ok" if state["service"] else "initializing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "models_dir": str(CFG_MODELS_DIR),
            "data_dir": str(CFG_DATA_DIR),
            "offline_mode": os.getenv("OFFLINE_MODE", "false"),
        },
        "dataset_info": {
            "rows": rows,
            "cols": cols,
            "shape": [rows, cols],
            "sample_cols": list(dataset.columns[:25]) if dataset is not None else [],
        },
        "cors_origins": origins,
        "cors_origin_regex": origin_regex,
        "restrict_cors": os.getenv("RESTRICT_CORS", "false").strip().lower() in TRUTHY,
    }

@app.post("/debug/predict-input")
async def debug_predict_input(req: PredictionRequest) -> Dict[str, Any]:
    service = _require_ready()
    bundle = state.get("bundle")
    dataset = state.get("dataset")
    if bundle is None or dataset is None:
        raise HTTPException(status_code=503, detail="Models or dataset not loaded.")

    schedule_df = None
    if hasattr(service, "_get_schedule_df"):
        try:
            schedule_df = service._get_schedule_df(req.season)
        except Exception:
            schedule_df = None

    row_df, source, debug_info = build_model_input_row(
        dataset_df=dataset,
        preprocessor=getattr(bundle, "preprocessor", None),
        season=req.season,
        week=req.week,
        home_team=req.home_team,
        away_team=req.away_team,
        schedule_df=schedule_df,
        raw_feature_columns=getattr(bundle, "raw_feature_columns", None),
        team_history_cache=getattr(service, "_team_history_cache", None),
        debug=True,
    )

    log.info(
        "Debug input %s@%s W%s: source=%s missing_after=%s missing_home_prior=%s missing_away_prior=%s",
        req.away_team,
        req.home_team,
        req.week,
        source,
        debug_info.get("missing_after_impute"),
        debug_info.get("missing_home_prior_count"),
        debug_info.get("missing_away_prior_count"),
    )

    return {
        "models_dir": str(CFG_MODELS_DIR),
        "prediction_source": source,
        "debug": debug_info,
    }

@app.post("/predict", response_model=UnifiedPredictionResponse)
async def predict(req: PredictionRequest, request: Request):
    service = _require_ready()
    res = service.predict(req)
    payload = _build_prediction_payload(req, res)
    _append_prediction_history_to_disk(req.model_dump(), payload)
    return payload

@app.post("/predict/explain")
async def explain(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    pred = _extract_prediction_payload(payload)
    home_team = payload.get("home_team") or pred.get("home_team")
    away_team = payload.get("away_team") or pred.get("away_team")
    season_raw = payload.get("season") or pred.get("season")
    week_raw = payload.get("week") or pred.get("week")

    try:
        season = int(season_raw) if season_raw is not None else None
    except (TypeError, ValueError):
        season = None
    try:
        week = int(week_raw) if week_raw is not None else None
    except (TypeError, ValueError):
        week = None

    needs_prediction = (
        not pred
        or pred.get("home_score") is None
        or pred.get("away_score") is None
        or pred.get("home_win_probability") is None
    )
    if needs_prediction:
        if not (home_team and away_team and season is not None and week is not None):
            raise HTTPException(status_code=400, detail="prediction or full game context required")
        req = PredictionRequest(home_team=home_team, away_team=away_team, season=season, week=week)
        service = _require_ready()
        pred = _build_prediction_payload(req, service.predict(req))

    if home_team:
        pred["home_team"] = home_team
    if away_team:
        pred["away_team"] = away_team
    if season is not None:
        pred["season"] = season
    if week is not None:
        pred["week"] = week

    game_id = pred.get("game_id")
    if not game_id and season is not None and week is not None and home_team and away_team:
        game_id = _build_game_id(season, week, home_team, away_team)

    llm_result = await llm_explain_prediction(pred)
    return {
        "game_id": game_id,
        "used_llm": bool(llm_result.get("used_llm")),
        "llm_model": llm_result.get("model"),
        "explanation": llm_result.get("explanation", ""),
        "bullets": llm_result.get("bullets", []) or [],
        "caveats": llm_result.get("caveats", []) or [],
        "error": llm_result.get("error"),
    }

@app.post("/llm/chat")
async def llm_chat(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    messages = payload.get("messages")
    prediction = payload.get("prediction")
    system_prompt = None

    if isinstance(prediction, dict) and prediction:
        home = prediction.get("home_team") or prediction.get("home_abbr")
        away = prediction.get("away_team") or prediction.get("away_abbr")
        season = prediction.get("season")
        week = prediction.get("week")
        system_prompt = (
            "You are an NFL prediction assistant. "
            f"Context: {home} vs {away}, season {season}, week {week}. "
            f"Prediction snapshot: {prediction}."
        )

    result = await llm_chat_messages(messages if isinstance(messages, list) else [], system_prompt=system_prompt)
    reply = result.get("reply") or ""
    if not reply and result.get("error"):
        reply = f"Error: {result.get('error')}"
    return {
        "reply": reply,
        "used_llm": bool(result.get("used_llm")),
        "llm_model": result.get("model"),
        "error": result.get("error"),
    }

@app.get("/history", response_model=HistoryResponse)
async def get_history(limit: int = 100):
    with _prediction_history_lock:
        data = prediction_history_entries[:limit]
        return HistoryResponse(entries=data, total=len(prediction_history_entries))

@app.get("/status/overview", response_model=StatusOverviewResponse)
async def get_status_overview():
    h = await health()
    dataset_info = {
        "rows": len(state["dataset"]) if state["dataset"] is not None else 0,
        "features": len(state["bundle"].raw_feature_columns) if state["bundle"] else 0,
    }
    with _prediction_history_lock:
        history_metrics = {
            "total_predictions": len(prediction_history_entries),
            "win_rate": None,
            "note": "win_rate requires actual outcomes",
        }
    return StatusOverviewResponse(health=h, dataset=dataset_info, history=history_metrics)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
