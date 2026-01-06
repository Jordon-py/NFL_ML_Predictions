# ==========================================
# File: /main.py
# Role: FastAPI app entrypoint and routing.
# Input Data: HTTP requests (JSON bodies, query params).
# Output Data: JSON responses for API endpoints.
# Dependencies: __future__, datetime, logging, os, pathlib, pandas
# Notes: Loads models/datasets at startup, validates schema, and mounts legacy routes under /legacy.
# ==========================================

"""
FILE: /main.py
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
# /main.py

from __future__ import annotations

import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    UnifiedPredictionResponse,
    HealthResponse,
    StatusOverviewResponse,
    HistoryResponse,
    ScheduleResponse,
    ScheduleEntry,
)
from .services.prediction_service import PredictionService
from .services.inference_row import build_model_input_row
from .config import DATA_DIR as CFG_DATA_DIR, MODELS_DIR as CFG_MODELS_DIR, resolve_cors, TRUTHY
from .main_helpers import (
    load_inference_bundle,
    load_dataset_df,
    _append_prediction_history_to_disk,
    prediction_history_entries,
    _prediction_history_lock,
    # New imports from refactor
    get_schedule as _load_schedule_df_routes,
    _select_next_week_rows as _select_next_week_rows_routes,
    _pick_col as _pick_col_routes,
    _load_team_logos_map as _load_team_logos_map_routes,
    _parse_kickoff as _parse_kickoff_routes,
    _HOME_COLS as _HOME_COLS_ROUTES,
    _AWAY_COLS as _AWAY_COLS_ROUTES,
    _GAME_ID_COLS as _GAME_ID_COLS_ROUTES,
    _STADIUM_COLS as _STADIUM_COLS_ROUTES,
)
from .ollama.llm_ollama import explain_prediction as llm_explain_prediction, chat_messages as llm_chat_messages
from .routes import (
    TeamLogosResponse,
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
    "model_metadata": None,
    "model_metadata_path": None,
    "dataset_path": None,
    "team_logos": None,
}

ADMIN_ENABLED = os.getenv("ENABLE_ADMIN", "false").strip().lower() in TRUTHY


def _build_game_id(season: int, week: int, home_team: str, away_team: str) -> str:
    parts = [season, week, home_team, away_team]
    return "-".join(str(p) for p in parts if p is not None and str(p).strip())

def _normalize_team_code(value: str) -> str:
    """Normalize team abbreviation to uppercase."""
    return str(value or "").strip().upper()


def _get_team_meta_map() -> Dict[str, Dict[str, str]]:
    """Load team metadata once and cache it for schedule/prediction enrichment."""
    cached = state.get("team_logos")
    if isinstance(cached, dict):
        return cached

    _dir = Path(__file__).resolve().parent
    repo_root = _dir.parent
    candidates = [
        _dir / "team_logo.csv",
        _dir / "team_logos.csv",
        _dir / "data" / "team_logo.csv",
        _dir / "data" / "team_logos.csv",
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
    home_code = _normalize_team_code(req.home_team)
    away_code = _normalize_team_code(req.away_team)
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

def _flatten_raw_feature_columns(raw: Any) -> list[str]:
    """Normalize 'raw_feature_columns' shapes into a flat list of column names.

    Supports:
      - list[str]
      - {"numeric":[...], "categorical":[...]}
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        nums = raw.get("numeric") or []
        cats = raw.get("categorical") or []
        out: list[str] = []
        out.extend([str(c) for c in nums if c is not None])
        out.extend([str(c) for c in cats if c is not None])
        return out
    if isinstance(raw, (list, tuple, set)):
        return [str(c) for c in raw if c is not None]
    return []

def _find_latest_metadata_json(models_dir: Path) -> Path | None:
    """Find the most recently modified metadata.json under a models directory."""
    try:
        root = Path(models_dir)
    except Exception:
        return None
    if root.is_file():
        return None
    candidates: list[Path] = []
    direct = root / "metadata.json"
    if direct.exists():
        candidates.append(direct)
    # Common patterns: MODELS_DIR/YYYYMMDD/metadata.json or MODELS_DIR/models/metadata.json
    candidates.extend([p for p in root.glob("**/metadata.json") if p.is_file()])
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _load_model_metadata(models_dir: Path) -> tuple[Path | None, Dict[str, Any] | None]:
    md_path = _find_latest_metadata_json(models_dir)
    if md_path is None:
        return None, None
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            return md_path, json.load(f)
    except Exception as e:
        log.warning("Could not read metadata.json at %s: %s", md_path, e)
        return md_path, None

def _find_latest_dataset_csv(data_dir: Path) -> Path | None:
    """Find the newest game_features_*.csv (or any .csv) under DATA_DIR."""
    try:
        root = Path(data_dir)
    except Exception:
        return None
    if root.is_file():
        return root if root.suffix.lower() == ".csv" else None

    patterns = ("game_features_*.csv", "*.csv")
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend([p for p in root.glob(pat) if p.is_file()])
    # allow nested datasets/YYYYMMDD/game_features_YYYYMMDD.csv layouts
    for pat in patterns:
        candidates.extend([p for p in root.glob(f"**/{pat}") if p.is_file()])

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _resolve_expected_features(bundle: Any, metadata: Dict[str, Any] | None = None) -> list[str]:
    """Resolve the *raw* input feature columns expected by the preprocessor/model."""
    pre = getattr(bundle, "preprocessor", None)
    features_in = getattr(pre, "feature_names_in_", None)
    if features_in is not None:
        expected = [str(x) for x in list(features_in)]
        if expected:
            return expected

    # Prefer explicit lists from training metadata (stable across sklearn versions)
    for cand in (
        (metadata or {}).get("feature_names"),
        getattr(bundle, "feature_names", None),
    ):
        if isinstance(cand, (list, tuple)) and len(cand) > 0:
            return [str(x) for x in cand if x is not None]

    # Fall back to 'raw_feature_columns' (either list or {"numeric","categorical"})
    raw = getattr(bundle, "raw_feature_columns", None)
    if metadata and "raw_feature_columns" in metadata:
        raw = metadata.get("raw_feature_columns")
    return _flatten_raw_feature_columns(raw)

def _validate_feature_schema(bundle: Any, dataset: pd.DataFrame, metadata: Dict[str, Any] | None = None) -> None:
    expected = _resolve_expected_features(bundle, metadata=metadata)
    if not expected:
        raise RuntimeError("Model feature list missing; cannot validate schema.")
    missing = [c for c in expected if c not in dataset.columns]
    if missing:
        sample = ", ".join(missing[:25])
        raise RuntimeError(f"Dataset missing {len(missing)} model features. Sample: {sample}")
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
        state["model_metadata_path"], state["model_metadata"] = _load_model_metadata(CFG_MODELS_DIR)
        expected_features = _resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))
        # Dataset: prefer helper loader; fall back to newest CSV under DATA_DIR
        try:
            state["dataset"] = load_dataset_df(CFG_DATA_DIR, expected_features=expected_features)
        except Exception:
            ds_path = _find_latest_dataset_csv(CFG_DATA_DIR)
            if ds_path is None:
                raise
            state["dataset"] = pd.read_csv(ds_path)
        state["dataset_path"] = str(_find_latest_dataset_csv(CFG_DATA_DIR) or "")
        _validate_feature_schema(state["bundle"], state["dataset"], metadata=state.get("model_metadata"))
        state["service"] = PredictionService(state["bundle"], state["dataset"])
        _get_team_meta_map()
        app.state.dataset = state["dataset"]
        app.state.model_metadata = state.get("model_metadata") or {}
        app.state.model_metadata_path = str(state.get("model_metadata_path") or "")
        app.state.dataset_path = state.get("dataset_path") or ""
        bundle = state["bundle"]
        app.state.models = {
            "preprocessor": getattr(bundle, "preprocessor", None),
            "home_model": getattr(bundle, "home_model", None),
            "away_model": getattr(bundle, "away_model", None),
            "hist_win_clf": getattr(bundle, "hist_win_clf", None),
            "models_dir": str(CFG_MODELS_DIR),
        }
        app.state.service = state["service"]  # Expose service for routes.py
        app.state.team_logos = state.get("team_logos") or {}
        app.state.started_at = datetime.now(timezone.utc).isoformat()
        log.info("Startup complete: Models and dataset ready.")
    except Exception as e:
        log.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield

app = FastAPI(title="NFL ML Predictions API", lifespan=lifespan)

# CORS Middleware
cors_origins, cors_origin_regex = resolve_cors()
cors_restrict = os.getenv("RESTRICT_CORS", "false").strip().lower() in TRUTHY
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=False,
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
    ready = state.get("service") is not None
    status = "healthy" if ready else "unhealthy"
    mode = "ml-inference" if ready else "initializing"
    reason = "models_loaded" if ready else "prediction engine not initialized"
    return HealthResponse(status=status, mode=mode, reason=reason)



@app.get("/schedule/next-week", response_model=ScheduleResponse)
async def get_next_week_schedule(season: int | None = None) -> ScheduleResponse:
    df = _load_schedule_df_routes(season=season)
    if not isinstance(df, pd.DataFrame) or df.empty:
        fallback_path = Path(__file__).resolve().parent / "post_schedule.json"
        if fallback_path.exists():
            try:
                raw = json.loads(fallback_path.read_text(encoding="utf-8"))
                games = raw.get("games") if isinstance(raw, dict) else None
                if isinstance(games, list) and games:
                    df = pd.DataFrame(games)
            except Exception as exc:
                log.warning("Schedule fallback read failed for %s: %s", fallback_path, exc)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
    df_next, use_season, use_week = _select_next_week_rows_routes(df)
    team_meta = _get_team_meta_map()

    home_col = _pick_col_routes(df_next, _HOME_COLS_ROUTES)
    away_col = _pick_col_routes(df_next, _AWAY_COLS_ROUTES)
    game_id_col = _pick_col_routes(df_next, _GAME_ID_COLS_ROUTES)
    stadium_col = _pick_col_routes(df_next, _STADIUM_COLS_ROUTES)

    games: list[ScheduleEntry] = []
    for _, row in df_next.iterrows():
        home = _normalize_team_code(row.get(home_col, "") if home_col else row.get("home", ""))
        away = _normalize_team_code(row.get(away_col, "") if away_col else row.get("away", ""))
        if not home or not away:
            continue

        home_info = team_meta.get(home, {})
        away_info = team_meta.get(away, {})

        game_id = ""
        if game_id_col:
            raw_id = row.get(game_id_col)
            if pd.notna(raw_id):
                game_id = str(raw_id).strip()
        if not game_id:
            game_id = _build_game_id(use_season, use_week, home, away)

        stadium = row.get(stadium_col, "") if stadium_col else row.get("stadium", "")

        games.append(
            ScheduleEntry(
                season=int(use_season),
                week=int(use_week),
                kickoff=_parse_kickoff_routes(row),
                home_team=home,
                away_team=away,
                game_id=game_id,
                home_abbr=home,
                away_abbr=away,
                home_logo=home_info.get("logoUrl"),
                away_logo=away_info.get("logoUrl"),
                home_name=home_info.get("name") or home,
                away_name=away_info.get("name") or away,
                stadium=stadium,
            )
        )

    return ScheduleResponse(games=games)

@app.get("/teams/logos", response_model=TeamLogosResponse)
async def get_team_logos() -> TeamLogosResponse:
    return TeamLogosResponse(teams=_get_team_meta_map())

@app.get("/debug")
async def debug() -> Dict[str, Any]:
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
        "cors_origins": cors_origins,
        "cors_origin_regex": cors_origin_regex,
        "restrict_cors": cors_restrict,
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

@app.get("/predict/next-week")
async def predict_next_week() -> Dict[str, Any]:
    service = _require_ready()
    schedule = await get_next_week_schedule()
    games: list[Dict[str, Any]] = []

    for game in schedule.games:
        req = PredictionRequest(
            home_team=game.home_team,
            away_team=game.away_team,
            season=game.season,
            week=game.week,
        )
        prediction = _build_prediction_payload(req, service.predict(req))
        item = game.model_dump()
        item["prediction"] = prediction
        games.append(item)

    return {"games": games}

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


@app.get("/status/models")
async def get_status_models() -> Dict[str, Any]:
    """Return model + dataset provenance (from training metadata), plus expected feature schema."""
    bundle = state.get("bundle")
    md = state.get("model_metadata") or {}
    expected = _resolve_expected_features(bundle, metadata=md) if bundle is not None else []
    return {
        "health": "ok" if state.get("service") else "initializing",
        "models_dir": str(CFG_MODELS_DIR),
        "metadata_path": str(state.get("model_metadata_path") or ""),
        "dataset_path": state.get("dataset_path") or "",
        "expected_features_count": len(expected),
        "expected_features_sample": expected[:25],
        "metadata": md,
    }

@app.post("/admin/reload")
async def admin_reload() -> Dict[str, Any]:
    """Reload model bundle + dataset without restarting the server (local/dev only)."""
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    log.info("Admin reload: reloading models + dataset...")
    state["bundle"] = load_inference_bundle(CFG_MODELS_DIR)
    state["model_metadata_path"], state["model_metadata"] = _load_model_metadata(CFG_MODELS_DIR)
    expected_features = _resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))

    # Dataset reload
    try:
        state["dataset"] = load_dataset_df(CFG_DATA_DIR, expected_features=expected_features)
    except Exception:
        ds_path = _find_latest_dataset_csv(CFG_DATA_DIR)
        if ds_path is None:
            raise
        state["dataset"] = pd.read_csv(ds_path)

    state["dataset_path"] = str(_find_latest_dataset_csv(CFG_DATA_DIR) or "")

    _validate_feature_schema(state["bundle"], state["dataset"], metadata=state.get("model_metadata"))
    state["service"] = PredictionService(state["bundle"], state["dataset"])

    return {
        "reloaded": True,
        "models_dir": str(CFG_MODELS_DIR),
        "metadata_path": str(state.get("model_metadata_path") or ""),
        "dataset_path": state.get("dataset_path") or "",
    }

@app.post("/admin/retrain")
async def admin_retrain(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Train models on the newest dataset and hot-reload them (local/dev only).

    NOTE: training can take minutes and will block this request while it runs.
    """
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    dataset_path = payload.get("dataset_path") or (state.get("dataset_path") or "")
    if not dataset_path:
        ds = _find_latest_dataset_csv(CFG_DATA_DIR)
        dataset_path = str(ds) if ds else ""

    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=400, detail=f"dataset_path not found: {dataset_path}")

    out_dir = payload.get("out_dir") or str(CFG_MODELS_DIR)
    log.info("Admin retrain: dataset=%s out_dir=%s", dataset_path, out_dir)

    # Import lazily to avoid import cycles during normal startup
    try:
        from .train_models import main as train_main
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not import training script: {e}")

    train_main(data_path=str(dataset_path), out_dir=str(out_dir))

    # Reload freshly trained artifacts
    await admin_reload()

    md = state.get("model_metadata") or {}
    return {
        "trained": True,
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "metadata_path": str(state.get("model_metadata_path") or ""),
        "dataset_hash": md.get("dataset_hash"),
        "training_timestamp_utc": md.get("training_timestamp_utc"),
        "production_ready": md.get("production_ready"),
    }

@app.get("/status/overview", response_model=StatusOverviewResponse)
async def get_status_overview():
    h = await health()
    dataset_info = {
        "rows": len(state["dataset"]) if state["dataset"] is not None else 0,
        "features": (len(_resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))) if state["bundle"] else 0),
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
