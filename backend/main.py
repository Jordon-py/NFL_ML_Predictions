from __future__ import annotations
"""
NFL ML Predictions API — Backend Server
=====================================

FastAPI backend serving ML predictions for NFL game outcomes.

Endpoints implemented in THIS file:
  GET  /health
  GET  /debug
  GET  /schedule/next-week
  POST /predict
  POST /predict/explain
  GET  /history
  GET  /status/overview

Key environment variables:
  MODELS_DIR                 Path to model artifacts directory (contains metadata.json)
  DATA_DIR                   Path to directory containing game_features_*.csv
  DATASET                    Optional: direct path to a specific engineered features CSV
  ALLOWED_ORIGINS            Optional: comma-separated allowed origins for CORS
  ALLOW_ORIGIN_REGEX         Optional: regex for dynamic CORS (default: vercel.app)
  PREDICTION_HISTORY_MAX     Max number of history entries to keep (default 1000)
  ALLOW_FALLBACK_PREDICTIONS If false, missing game rows will raise instead of roll-forward (default true)
  SCHEDULE_SEASON            Season year used by nflreadpy schedule fetch (default current year)
  MC_SIMS                    Monte Carlo simulation count (default 2000)

Design principles:
- Prefer clarity over cleverness.
- One “readiness” gate for all endpoints.
- Keep optional analysis features deterministic and dependency-light.

Run locally:

uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

Test endpoints:
curl http://127.0.0.1:8000/teams/$body = @{ home_team="KC"; away_team="BUF"; season=2025; week=15 } | ConvertTo-Json -Depth 10; Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body

curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/debug
curl http://127.0.0.1:8000/schedule/next-week
curl http://127.0.0.1:8000/history?limit=5
curl http://127.0.0.1:8000/status/overview

$body = @{ home_team="KC"; away_team="BUF"; season=2025; week=15 } | ConvertTo-Json -Depth 10; Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body


curl -X POST http://127.0.0.1:8000/predict/explain \\
  -H "Content-Type: application/json" \\
  -d "{\\"home_team\\":\\"KC\\",\\"away_team\\}":{\\"BUF\\",\\"season\\":2025,\\"week\\":1}"


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



import logging
import sys
import os
import json
import uuid
import time
import math
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
try:
    from pythonjsonlogger import jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    UnifiedPredictionResponse,
    HealthResponse,
    StatusOverviewResponse,
    HistoryResponse,
    ScheduleResponse,
    ScheduleEntry,
    TeamLogosResponse,
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
    get_schedule,
    select_next_week_rows,
    get_team_meta,
    parse_kickoff,
    _pick_col,
    _HOME_COLS,
    _AWAY_COLS,
    _GAME_ID_COLS,
    _STADIUM_COLS,
)
from .ollama.llm_ollama import explain_prediction as llm_explain_prediction, chat_messages as llm_chat_messages
from .team_assets import (
    normalize_abbr,
    load_team_assets_map,
    TeamAsset
)
try:
    from .routes.auth import router as auth_router
except ImportError:
    auth_router = None

try:
    from .routes.upload import router as upload_router
except ImportError:
    upload_router = None

if __name__ == "__main__" and __package__ is None:
    # Allow running as a script by ensuring repo root is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# -------------------------------------
# GLOBALS
# -------------------------------------
# Load environment variables
load_dotenv(dotenv_path=".env")

# Setup logging with JSON format for production observability
def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    if HAS_JSON_LOGGER and os.getenv("LOG_FORMAT", "text").lower() == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    handler.setFormatter(formatter)
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

setup_logging()
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

# -------------------------------------
# Helper Functions
# -------------------------------------

def get_logos(home_team, away_team):
    team_logos = _get_team_meta_map()
    home_logo = team_logos.get(home_team)
    away_logo = team_logos.get(away_team)
    return home_logo, away_logo

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
            team_map = get_team_meta(csv_path)
            break

    state["team_logos"] = team_map or {}
    return state["team_logos"]

def _clean_s(val: Any) -> Optional[str]:
    """Convert nan or empty values to None for Pydantic Optional[str]."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip() == "":
        return None
    return str(val).strip()


def _derive_season_phase(df_next: pd.DataFrame) -> tuple[str, str]:
    """
    Infer broad NFL season phase from next-slate rows.
    Returns (phase, human_label) where phase is one of:
    - in_season
    - postseason
    - offseason
    """
    if isinstance(df_next, pd.DataFrame) and not df_next.empty:
        game_type_col = _pick_col(df_next, ["game_type", "season_type", "type"])
        if game_type_col:
            game_types = (
                df_next[game_type_col]
                .dropna()
                .astype(str)
                .str.upper()
                .str.strip()
                .unique()
                .tolist()
            )
            has_post = any(gt not in {"REG", "R"} for gt in game_types)
            if has_post:
                return ("postseason", "Postseason")
        return ("in_season", "Regular Season")

    month = datetime.now(timezone.utc).month
    # Typical NFL offseason window: Feb-Jul (inclusive)
    if 2 <= month <= 7:
        return ("offseason", "Offseason")
    # Aug with no schedule is effectively preseason prep for users.
    if month == 8:
        return ("offseason", "Preseason Build-Up")
    return ("offseason", "Offseason")


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

def _filter_expected_features(features: list[str]) -> list[str]:
    """Drop empty/duplicate names and pandas index placeholders (e.g., 'Unnamed: 0')."""
    if not features:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for name in features:
        s = str(name).strip()
        if not s:
            continue
        if s.lower().startswith("unnamed:"):
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned

def _find_latest_metadata_json(models_dir: Path) -> Path | None:
    """Find the most recently modified metadata.json under a models directory."""
    try:
        root = Path(models_dir)
    except Exception:
        return None
    if root.is_file():
        return root if root.name == "metadata.json" else None

    candidates: list[Path] = []
    direct = root / "metadata.json"
    if direct.exists():
        candidates.append(direct)
    # Common patterns: MODELS_DIR/YYYYMMDD/metadata.json or MODELS_DIR/models/metadata.json
    candidates.extend([p for p in root.glob("**/metadata.json") if p.is_file()])
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)




# ---------------------------
# LLM helper (best-effort)
# ---------------------------

def _fallback_explain(pred: Dict[str, Any]) -> Dict[str, Any]:
    home = str(pred.get("home_team", "")).upper()
    away = str(pred.get("away_team", "")).upper()
    hs = pred.get("home_score")
    as_ = pred.get("away_score")
    p_home = pred.get("home_win_probability")
    pdiff = pred.get("point_diff")

    bullets: List[str] = []
    if isinstance(pdiff, (int, float)):
        fav = home if float(pdiff) >= 0 else away
        bullets.append(f"{fav} is favored by about {abs(float(pdiff)):.1f} points (model estimate).")
    if isinstance(p_home, (int, float)):
        bullets.append(f"Home win probability is ~{100.0 * float(p_home):.0f}% (calibrated/ensemble output).")
    if isinstance(hs, (int, float)) and isinstance(as_, (int, float)):
        bullets.append(f"Projected score: {home} {hs}  •  {away} {as_}.")

    caveats = [
        "This is a pre-game estimate. Late injuries, weather, and market moves can shift reality.",
        "If the game row was missing, features may be rolled-forward/mean-filled (see prediction_source).",
    ]

    favored = home if (isinstance(pdiff, (int, float)) and float(pdiff) >= 0) else away
    explanation = f"{home} vs {away}: model leans {favored} based on learned pre-game feature patterns."
    return {"explanation": explanation, "bullets": bullets, "caveats": caveats}


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

def _find_latest_dataset_csv(data_dir: Path=os.getenv("DATA_DIR", Path("./data/dataset"))) -> Path | None:
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
















# ---------------------------
# Lifespan
# ---------------------------

def _resolve_expected_features(bundle: Any, metadata: Dict[str, Any] | None = None) -> list[str]:
    """Resolve the *raw* input feature columns expected by the preprocessor/model."""
    pre = getattr(bundle, "preprocessor", None)
    features_in = getattr(pre, "feature_names_in_", None)
    if features_in is not None:
        expected = _filter_expected_features([str(x) for x in list(features_in)])
        if expected:
            return expected

    # Prefer explicit lists from training metadata (stable across sklearn versions)
    for cand in (
        (metadata or {}).get("feature_names"),
        getattr(bundle, "feature_names", None),
    ):
        if isinstance(cand, (list, tuple)) and len(cand) > 0:
            expected = _filter_expected_features([str(x) for x in cand if x is not None])
            if expected:
                return expected

    # Fall back to 'raw_feature_columns' (either list or {"numeric","categorical"})
    raw = getattr(bundle, "raw_feature_columns", None)
    if metadata and "raw_feature_columns" in metadata:
        raw = metadata.get("raw_feature_columns")
    return _filter_expected_features(_flatten_raw_feature_columns(raw))

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


def _predict_home_win_prob(bundle: InferenceBundle, X_raw: pd.DataFrame, point_diff: float) -> Tuple[float, bool]:
    """
    Returns (probability, used_fallback).
    Since we use full pipelines (preprocessor included), we pass raw data directly.
    """
    clf = bundle.win_pipe or bundle.hist_win_clf  # prefer win_pipe (pipeline) logic

    if hasattr(clf, "predict_proba"):
        try:
            # Direct prediction using the pipeline
            proba = clf.predict_proba(X_raw)
            idx = _pick_positive_class_index(clf)
            val = float(proba[0][idx])
            return float(np.clip(val, 0.0, 1.0)), False
        except Exception as e:
            log.warning("[Predict] Pipeline predict_proba failed: %s", e)

    # logistic fallback
    if pd.isna(point_diff):
        return 0.5, True
    p = 1.0 / (1.0 + math.exp(-0.25 * float(point_diff)))
    return float(np.clip(p, 0.0, 1.0)), True

async def build_and_reload_dataset():
    """
    Scheduled task:
      1. Run build_csv_datasets_v3.py as a subprocess to fetch new data.
      2. If successful, reload the dataset into memory.
    """
    log.info("Starting scheduled dataset build...")
    try:
        # Run the build script as a module from the project root
        # We need to add 'backend' to PYTHONPATH so 'utils' can be imported directly
        # because build_csv_datasets_v3.py uses 'from utils import ...'
        repo_root = Path(__file__).resolve().parent.parent
        backend_dir = repo_root / "backend"
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "backend.build_csv_datasets_v3",
            "--out-dir", str(CFG_DATA_DIR),
            "--legacy-root-copy",
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": str(backend_dir)}
        )

        stdout, stderr = await process.communicate()

        if stdout:
            log.info(f"Build Subprocess STDOUT:\n{stdout.decode().strip()}")
        if stderr:
            log.error(f"Build Subprocess STDERR:\n{stderr.decode().strip()}")

        if process.returncode == 0:
            log.info("Dataset build successful.")
            log.info("Reloading dataset...")
            new_csv = _find_latest_dataset_csv(CFG_DATA_DIR)
            if new_csv:
                expected_features = _resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))
                state["dataset"] = load_dataset_df(CFG_DATA_DIR, expected_features=expected_features)
                state["dataset_path"] = str(new_csv)

                # Update app state
                app.state.dataset = state["dataset"]

                # Re-initialize service with new dataset
                state["service"] = PredictionService(state["bundle"], state["dataset"])

                # Update status overview metrics in real-time?
                # (metrics are pulled from state["dataset"] so they should auto-update)

                log.info(f"Dataset reloaded from {new_csv}")
            else:
                log.warning("Build finished but no CSV found to reload.")
        else:
            log.error(f"Dataset build failed with return code {process.returncode}")
            if stderr:
                log.error(f"Build Error: {stderr.decode()}")
            if stdout:
                log.info(f"Build Output: {stdout.decode()}")

    except Exception as e:
        log.error(f"Error during scheduled dataset build: {e}", exc_info=True)

# Startup / Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models and dataset
    try:
        log.info(f"Starting up: Loading model bundle from {CFG_MODELS_DIR}...")
        ensure_artifacts()
        load_prediction_history()

        # 1. Load Bundle (Models)
        state["bundle"] = load_inference_bundle(CFG_MODELS_DIR)

        # 2. Load Metadata (if separate)
        # Assuming metadata.json is always in MODELS_DIR as enforced by config
        meta_path = CFG_MODELS_DIR / "metadata.json"
        if meta_path.exists():
            state["model_metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
            state["model_metadata_path"] = str(meta_path)

        expected_features = _resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))

        # 3. Load Dataset
        # Config enforces DATASET_PATH if specific file is needed.
        log.info(f"Loading dataset from {CFG_DATA_DIR}...")
        state["dataset"] = load_dataset_df(CFG_DATA_DIR, expected_features=expected_features)

        # Validate
        _validate_feature_schema(state["bundle"], state["dataset"], metadata=state.get("model_metadata"))

        # Init Service
        state["service"] = PredictionService(state["bundle"], state["dataset"])
        _get_team_meta_map()

        # Expose state
        app.state.dataset = state["dataset"]
        app.state.models = {"models_dir": str(CFG_MODELS_DIR)}
        app.state.team_logos = state.get("team_logos") or {}
        app.state.started_at = datetime.now(timezone.utc).isoformat()

        app.state.started_at = datetime.now(timezone.utc).isoformat()

        # 4. Start Scheduler
        scheduler = AsyncIOScheduler()
        # Schedule to run every day at 3:00 AM (server time/UTC depending on env)
        scheduler.add_job(build_and_reload_dataset, 'cron', hour=3, minute=0)
        scheduler.start()
        log.info("Scheduler started: Auto-build set for 03:00 daily.")

        log.info("Startup complete: Models and dataset ready.")
    except Exception as e:
        log.error(f"Startup failed: {e}", exc_info=True)
        # We raise here because the user wants to fix the traceback, meaning we should fail if invalid.
        raise

    yield

app = FastAPI(title="NFL ML Predictions API", lifespan=lifespan)

# Optional auth and file upload routes.
if auth_router is not None:
    app.include_router(auth_router, prefix="/auth")
if upload_router is not None:
    app.include_router(upload_router, prefix="/upload")
# CORS Middleware
cors_origins, cors_origin_regex = resolve_cors()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*", "X-Request-ID"],
)

# Request-ID and timing middleware for observability
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start_time = time.perf_counter()
        
        # Attach to request state for use in handlers
        request.state.request_id = request_id
        
        response = await call_next(request)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        
        # Structured log entry
        log.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            }
        )
        return response

app.add_middleware(RequestContextMiddleware)

# Prometheus metrics instrumentation
if HAS_PROMETHEUS:
    Instrumentator().instrument(app).expose(app, endpoint="/api/metrics")

# ---------------------------
# Routes
# ---------------------------


@app.get("/api/teams/{team_abbr}", response_model=TeamAsset)
def teams_get(team_abbr: str) -> TeamAsset:
    """
    Get a team's branding assets (preferred non-square logo included).

    Example:
      GET /teams/LAR
    """
    response_model = get_team_asset(team_abbr)
    if response_model is None:
        raise HTTPException(status_code=404, detail=f"Team not found: {team_abbr}")
    return response_model

def _require_ready() -> PredictionService:
    if state["service"] is None:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized.")
    return state["service"]

# ---------------------------
# API ROUTES
# ---------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """System health check (fast)."""
    ready = state.get("service") is not None
    status = "healthy" if ready else "unhealthy"
    mode = "ml-inference" if ready else "initializing"
    reason = "models_loaded" if ready else "prediction engine not initialized"
    return HealthResponse(status=status, mode=mode, reason=reason)

@app.get("/health", response_model=HealthResponse)
async def health_legacy():
    """Legacy system health check alias for compatibility."""
    return await health()

@app.get("/api/health/deep")
async def health_deep() -> Dict[str, Any]:
    """Deep health check: verifies models, dataset, and dependencies."""
    checks = {}
    overall_healthy = True
    
    # Check model bundle
    bundle = state.get("bundle")
    if bundle is not None:
        checks["model_bundle"] = {"status": "ok", "loaded": True}
    else:
        checks["model_bundle"] = {"status": "error", "loaded": False}
        overall_healthy = False
    
    # Check dataset
    dataset = state.get("dataset")
    if dataset is not None and len(dataset) > 0:
        checks["dataset"] = {"status": "ok", "rows": len(dataset)}
    else:
        checks["dataset"] = {"status": "error", "rows": 0}
        overall_healthy = False
    
    # Check prediction service
    service = state.get("service")
    if service is not None:
        checks["prediction_service"] = {"status": "ok", "initialized": True}
    else:
        checks["prediction_service"] = {"status": "error", "initialized": False}
        overall_healthy = False
    
    # Check metadata
    metadata = state.get("model_metadata")
    checks["metadata"] = {
        "status": "ok" if metadata else "warning",
        "loaded": metadata is not None,
    }
    
    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/api/client-errors", status_code=204)
async def receive_client_error(request: Request):
    """Receive and log frontend error reports."""
    try:
        body = await request.body()
        error_data = json.loads(body.decode("utf-8")) if body else {}
        log.warning(
            "client_error_reported",
            extra={
                "client_error": error_data.get("message"),
                "url": error_data.get("url"),
                "user_agent": error_data.get("userAgent"),
                "timestamp": error_data.get("ts"),
                "stack": error_data.get("stack", "")[:500],  # Truncate stack
            }
        )
    except Exception as e:
        log.warning(f"Failed to parse client error: {e}")
    return Response(status_code=204)

@app.get("/api/status/overview", response_model=StatusOverviewResponse)
async def get_status_overview():
    """High-level system overview."""
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


@app.get("/api/status/models")
async def get_status_models() -> Dict[str, Any]:
    """Detailed model and dataset metadata."""
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

@app.get("/api/season/context", response_model=SeasonContextResponse)
async def get_season_context(season: int | None = None) -> SeasonContextResponse:
    """
    Return schedule-aware season context so clients can render
    in-season/postseason/offseason UX without guessing.
    """
    df = get_schedule(season=season)
    df_next, use_season, use_week = select_next_week_rows(df)
    phase, label = _derive_season_phase(df_next)

    next_kickoff: Optional[datetime] = None
    if isinstance(df_next, pd.DataFrame) and not df_next.empty:
        kickoff_candidates = [parse_kickoff(row) for _, row in df_next.iterrows()]
        kickoff_candidates = [dt for dt in kickoff_candidates if isinstance(dt, datetime)]
        if kickoff_candidates:
            next_kickoff = min(kickoff_candidates)

    now_utc = datetime.now(timezone.utc)
    if phase != "offseason":
        kickoff_utc = None
        if isinstance(next_kickoff, datetime):
            kickoff_utc = (
                next_kickoff.astimezone(timezone.utc)
                if next_kickoff.tzinfo is not None
                else next_kickoff.replace(tzinfo=timezone.utc)
            )
        # If no future kickoff is available during typical offseason months,
        # force offseason mode to keep client UX stable.
        if (kickoff_utc is None or kickoff_utc < now_utc) and 2 <= now_utc.month <= 8:
            phase, label = "offseason", "Offseason"

    games_count = int(len(df_next)) if isinstance(df_next, pd.DataFrame) else 0
    if phase == "offseason":
        message = (
            "No live weekly slate is available right now. "
            "Use Offseason Mode to explore projected matchups and model health."
        )
    elif phase == "postseason":
        message = "Postseason slate is active."
    else:
        message = "Regular season slate is active."

    return SeasonContextResponse(
        phase=phase,
        label=label,
        message=message,
        current_season=int(use_season),
        display_week=int(use_week) if use_week is not None else None,
        games_in_next_window=games_count,
        next_kickoff=next_kickoff,
        generated_at=datetime.now(timezone.utc),
    )

@app.get("/api/debug")
async def debug() -> Dict[str, Any]:
    """In-depth debugging information."""
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
             "path": state.get("dataset_path"),
        }
    }

@app.post("/api/debug/trigger-build")
async def trigger_build_manually():
    """Manually trigger the day's dataset build (async)."""
    if not ADMIN_ENABLED:
         # Optional: secure this endpoint
         pass

    asyncio.create_task(build_and_reload_dataset())
    return {"status": "Build triggered in background. Check server logs."}


# --- Core Prediction Routes ---

@app.post("/api/predict", response_model=UnifiedPredictionResponse)
async def predict(req: PredictionRequest):
    """Generate a prediction for a single NFL matchup."""
    home_norm = norm_team(req.home_team)
    away_norm = norm_team(req.away_team)
    if not home_norm or not away_norm or home_norm == away_norm:
        raise HTTPException(status_code=422, detail="home_team and away_team must be different valid teams")
    if req.week < 1 or req.week > 22:
        raise HTTPException(status_code=422, detail="week must be between 1 and 22")

    req = PredictionRequest(
        home_team=home_norm,
        away_team=away_norm,
        season=req.season,
        week=req.week,
    )

    service = _require_ready()
    try:
        res = service.predict(req)
        payload = _build_prediction_payload(req, res)
        _append_prediction_history_to_disk(req.model_dump(), payload)
        return payload
    except Exception as e:
        log.error(f"[Predict] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/explain")
async def explain(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Generate a deterministic natural language explanation for a prediction."""
    pred = _extract_prediction_payload(payload)
    home_team = payload.get("home_team") or pred.get("home_team")
    away_team = payload.get("away_team") or pred.get("away_team")
    season_raw = payload.get("season") or pred.get("season")
    week_raw = payload.get("week") or pred.get("week")

    try:
        season = int(season_raw) if season_raw is not None else None
        week = int(week_raw) if week_raw is not None else None
    except (TypeError, ValueError):
        season = None
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

    for k, v in [("home_team", home_team), ("away_team", away_team), ("season", season), ("week", week)]:
        if v is not None:
             pred[k] = v

    game_id = pred.get("game_id")
    if not game_id and season is not None and week is not None and home_team and away_team:
        game_id = _build_game_id(season, week, home_team, away_team)

    explanation_payload = _fallback_explain(pred)
    return {
        "game_id": game_id,
        "used_llm": False,
        "llm_model": None,
        "explanation": explanation_payload.get("explanation", ""),
        "bullets": explanation_payload.get("bullets", []) or [],
        "caveats": explanation_payload.get("caveats", []) or [],
        "error": None,
    }

@app.get("/api/schedule/next-week", response_model=ScheduleResponse)
async def get_next_week_schedule(season: int | None = None) -> ScheduleResponse:
    """Fetch the schedule for the next upcoming games."""
    df = get_schedule(season=season)
    if not isinstance(df, pd.DataFrame) or df.empty:
        fallback_path = Path(__file__).resolve().parent / "post_schedule.json"
        if fallback_path.exists():
            try:
                raw = json.loads(fallback_path.read_text(encoding="utf-8"))
                games_list = raw.get("games")
                if isinstance(games_list, list) and games_list:
                    df = pd.DataFrame(games_list)
            except Exception as exc:
                log.warning("Schedule fallback failed: %s", exc)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()

    df_next, use_season, use_week = select_next_week_rows(df)
    team_meta = _get_team_meta_map()

    home_col = _pick_col(df_next, _HOME_COLS)
    away_col = _pick_col(df_next, _AWAY_COLS)
    game_id_col = _pick_col(df_next, _GAME_ID_COLS)
    stadium_col = _pick_col(df_next, _STADIUM_COLS)

    games: list[ScheduleEntry] = []
    for _, row in df_next.iterrows():
        home = str(row.get(home_col, "") if home_col else row.get("home", "")).strip().upper()
        away = str(row.get(away_col, "") if away_col else row.get("away", "")).strip().upper()
        if not home or not away:
            continue

        home_info = team_meta.get(home, {})
        away_info = team_meta.get(away, {})

        game_id = _clean_s(row.get(game_id_col)) if game_id_col else _build_game_id(use_season, use_week, home, away)
        stadium = _clean_s(row.get(stadium_col))

        games.append(
            ScheduleEntry(
                season=int(use_season),
                week=int(use_week),
                kickoff=parse_kickoff(row),
                home_team=home,
                away_team=away,
                game_id=game_id,
                home_abbr=home,
                away_abbr=away,
                home_logo=_clean_s(home_info.get("logoUrl")),
                away_logo=_clean_s(away_info.get("logoUrl")),
                home_name=_clean_s(home_info.get("name")) or home,
                away_name=_clean_s(away_info.get("name")) or away,
                stadium=stadium,
            )
        )
    return ScheduleResponse(games=games)

@app.get("/api/teams/logos", response_model=TeamLogosResponse)
async def get_team_logos() -> TeamLogosResponse:
    """Fetch current team metadata dictionary."""
    return TeamLogosResponse(teams=_get_team_meta_map())

@app.get("/api/history", response_model=HistoryResponse)
async def get_history(limit: int = 100):
    """Get recent prediction history from disk."""
    with _prediction_history_lock:
        data = prediction_history_entries[:limit]
        return HistoryResponse(entries=data, total=len(prediction_history_entries))

# --- Admin & Debug Routes ---

@app.post("/api/admin/reload")
async def admin_reload() -> Dict[str, Any]:
    """Hot-reload models and datasets."""
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=403, detail="Admin disabled")

    log.info("Admin reload requested.")
    state["bundle"] = load_inference_bundle(CFG_MODELS_DIR)
    state["model_metadata_path"], state["model_metadata"] = _load_model_metadata(CFG_MODELS_DIR)
    expected_features = _resolve_expected_features(state["bundle"], metadata=state.get("model_metadata"))
    state["dataset"] = load_dataset_df(CFG_DATA_DIR, expected_features=expected_features)
    state["service"] = PredictionService(state["bundle"], state["dataset"])

    return {"reloaded": True, "models_dir": str(CFG_MODELS_DIR)}

@app.post("/api/admin/retrain")
async def admin_retrain(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Trigger model retraining."""
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=403, detail="Admin disabled")

    dataset_path = payload.get("dataset_path") or state.get("dataset_path") or ""
    if not dataset_path:
         ds = _find_latest_dataset_csv(CFG_DATA_DIR)
         dataset_path = str(ds) if ds else ""

    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=400, detail="dataset path not found")

    try:
        from backend.train_models import main as train_main
    except ImportError:
        from train_models import main as train_main
    train_main(data_path=str(dataset_path), out_dir=str(CFG_MODELS_DIR))
    await admin_reload()
    return {"trained": True}

@app.post("/api/debug/predict-input")
async def debug_predict_input(req: PredictionRequest) -> Dict[str, Any]:
    """Expose the raw feature vector for debugging."""
    service = _require_ready()
    bundle = state.get("bundle")
    dataset = state.get("dataset")
    row_df, source, debug_info = build_model_input_row(
        dataset_df=dataset,
        preprocessor=getattr(bundle, "preprocessor", None),
        season=req.season,
        week=req.week,
        home_team=req.home_team,
        away_team=req.away_team,
        raw_feature_columns=getattr(bundle, "raw_feature_columns", None),
        debug=True,
    )
    return {"prediction_source": source, "debug": debug_info}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
