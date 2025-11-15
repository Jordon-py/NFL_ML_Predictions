# -*- coding: utf-8 -*-
"""
File: main.py

Purpose:
    FastAPI (v0.110+) entry point for the NFL prediction API. Loads ML model artifacts, exposes prediction and schedule endpoints, manages CORS, and performs startup health checks.

Key Functions:
    - get_current_nfl_context
    - health
    - predict
    - schedule endpoints (next_week, season, current_week)
    - training_status
    - lifespan (async context manager)
    - reload_models (manual reload of model pipelines)

Note:
    Pipelines are loaded at module scope and will not automatically reload if the models directory changes after startup. Use the `/reload-models` endpoint to manually reload pipelines without restarting the server.

Key Variables:
    - app (FastAPI)
    - models (dict)
    - preprocessor
    - metadata
    - CORS_ORIGINS (from environment)
    - MODEL_DIR
    - logger

Critical Environment Variables:
    - MODELS_DIR: Path to directory containing model artifacts (overrides auto-detection if set)
    - DATASET_PATH: Path to the CSV file with game features (overrides default if set)
    - SCHEDULE_PATH: Path to the NFL schedule CSV (overrides default if set)
    - ALLOWED_ORIGINS: Comma-separated list of allowed CORS origins
    - SERVE_FRONTEND: If "true", serves the frontend build from /frontend/build or /frontend/dist

Interacts With:
    - backend/models/*.joblib (model artifacts)
    - backend/data/team_abbr_map.json (team abbreviation mapping)
    - frontend API client (PredictionContext)
    - CORS environment configuration

This file is the main entry point for backend deployment and local development.
"""
from __future__ import annotations

import json
import logging
import logging.config
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, AsyncGenerator, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# sklearn optional bits
try:
    from sklearn.utils.validation import check_is_fitted
    SKLEARN_CHECK_AVAILABLE = True
except Exception:
    check_is_fitted = None  # type: ignore
    SKLEARN_CHECK_AVAILABLE = False

try:
    from sklearn.exceptions import NotFittedError  # type: ignore
    SKLEARN_NOTFITTED_AVAILABLE = True
except Exception:
    NotFittedError = None  # type: ignore
    SKLEARN_NOTFITTED_AVAILABLE = False

# ---------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------
backend_dir = Path(__file__).parent
ENV = backend_dir / ".env"
repo_root = backend_dir.parent
dotenv_loaded = load_dotenv(dotenv_path=ENV)
if not dotenv_loaded:
    load_dotenv(dotenv_path=repo_root / ".env")

# -----------------------
# Paths and constants
# -----------------------
THIS_FILE = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
BASE_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
HISTORY_FILE = DATA_DIR / "prediction_history.json"
MAX_HISTORY_ROWS = int(os.getenv("PREDICTION_HISTORY_LIMIT", "200"))
LOG_DIR = BACKEND_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
# CHANGE-LOG: Pre-create the log directory so file handlers never fail during startup.

# ---------------------------------------------------------------------
# Models directory resolution
#
# On startup, we attempt to locate the most appropriate model artifact
# directory. By default, the API expects to find pre-trained pipelines
# (home_pipe.joblib, away_pipe.joblib, win_pipe.joblib) under a
# `models/` folder in the backend directory. However, the enhanced
# training pipeline writes its artifacts into date-stamped subfolders
# (e.g. `20251110/models/`). To support seamless upgrades, we resolve
# the models directory in the following order:
#   1. If the environment variable `MODELS_DIR` is set, that path is
#      used unconditionally.
#   2. If a `models` directory exists in the backend directory, use it.
#   3. Otherwise, scan for the most recent `<YYYYMMDD>/models` folder
#      within the backend directory and use that.

def _resolve_models_dir() -> Path:
    """Resolve the directory containing model artifacts.

    Returns a Path pointing to the directory with pre-trained pipelines.
    The resolution order is described above. If no directory can be
    resolved, falls back to `<backend>/models`.
    """
    # 1) Environment override
    env_path = os.getenv("MODELS_DIR", "").strip()
    if env_path:
        p = Path(env_path)
        if p.is_dir():
            return p
        # Use a local logger to avoid relying on global `log` before it is defined
        logging.getLogger("api").warning("MODELS_DIR=%s is set but does not exist", p)
    # 2) Legacy location: backend/models
    legacy = BACKEND_DIR / "models"
    if legacy.is_dir():
        return legacy
    # 3) Latest date-stamped run directory
    # Look for directories matching YYYYMMDD under the backend dir
    candidates: List[Path] = []
    for child in BACKEND_DIR.iterdir():
        try:
            # Expect directory names like '20251110' that are all digits
            if child.is_dir() and child.name.isdigit() and len(child.name) == 8:
                models_sub = child / "models"
                if models_sub.is_dir():
                    candidates.append(models_sub)
        except Exception:
            continue
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = candidates[0]
        logging.getLogger("api").info("Resolved latest models directory: %s", latest)
        return latest
    fallback = BACKEND_DIR / "models"
    fallback.mkdir(parents=True, exist_ok=True)
    logging.getLogger("api").warning("Falling back to default models directory: %s", fallback)
    return fallback

# -----------------------
# Logging configuration
# -----------------------
# Logging
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"d": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
        "handlers": {
            "console": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "d"},
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "d",
                "filename": str(LOG_DIR / "api.log"),
                "encoding": "utf-8",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)
log = logging.getLogger("api")



MODELS_DIR = _resolve_models_dir()
log.info("Initial MODELS_DIR resolved to %s", MODELS_DIR)
# CHANGE-LOG: Emit the resolved models path for quick operational diagnostics.
DEFAULT_DATASET = DATA_DIR / "game_features_20251111.csv"
DEFAULT_SCHEDULE = DATA_DIR / "Nfl_schedule_2025_2026.csv"

FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_BUILD = FRONTEND_DIR / "build"
FRONTEND_DIST = FRONTEND_DIR / "dist"

TRUTHY = {"true", "t", "1", "yes", "y"}
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "false").strip().lower() in TRUTHY

# Logging
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"d": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
        "handlers": {
            "console": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "d"},
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "d",
                "filename": str(LOG_DIR / "api.log"),
                "encoding": "utf-8",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)
log = logging.getLogger("api")

# Globals
model_objects: Optional[Dict[str, Any]] = None
dataset_df: Optional[pd.DataFrame] = None
home_pipe: Optional[Any] = None
away_pipe: Optional[Any] = None
win_pipe: Optional[Any] = None
prediction_history: List[Dict[str, Any]] = []
history_lock = Lock()
ACTIVE_DATASET_PATH: Optional[str] = None
DATASET_LAST_LOADED: Optional[datetime] = None

# -----------------------
# CORS configuration
# -----------------------
DEFAULT_ALLOWED_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://nfl-ml-predictions.vercel.app",
    "https://nfl-ml-predictions-pr5uahmqx-christopher-jordons-projects.vercel.app",
    "https://nfl-predict-6fghcp7sx-christopher-jordons-projects.vercel.app",
    "https://new-nfl-predict.vercel.app",
    "http://www.nfl-predict.vercel.app",
]

def _parse_cors_origins() -> List[str]:
    env_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
    if env_origins:
        origins = [o.strip() for o in env_origins.split(",") if o.strip()]
        if origins:
            log.info("CORS origins from env: %s", origins)
            return origins
    log.info("CORS origins (default): %s", DEFAULT_ALLOWED_ORIGINS)
    return DEFAULT_ALLOWED_ORIGINS

if os.getenv("RESTRICT_CORS", "false").strip().lower() in TRUTHY:
    ALLOWED_ORIGINS = _parse_cors_origins()
    log.info("CORS restricted mode enabled")
else:
    ALLOWED_ORIGINS = ["*"]
    log.info("CORS configured to allow all origins")

ALLOW_ORIGIN_REGEX = os.getenv("ALLOW_ORIGIN_REGEX", "").strip() or None

# -----------------------------------------------------------
# Required Pipelines
# -----------------------------------------------------------

# FIX: Added the missing 'load_pipelines' function
def load_pipelines() -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
    global MODELS_DIR
    def _attempt_load(dir_path: Path) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
        try:
            hp = joblib.load(dir_path / "home_pipe.joblib")
            ap = joblib.load(dir_path / "away_pipe.joblib")
            wp = joblib.load(dir_path / "win_pipe.joblib")
            log.info("✓ Pipelines loaded successfully from %s", dir_path)
            return hp, ap, wp
        except Exception as exc:
            log.error(
                "✗ Failed to load pipelines from %s: %s", dir_path, exc, exc_info=True
            )
            return None, None, None

    # First attempt using the currently resolved MODELS_DIR
    hp, ap, wp = _attempt_load(MODELS_DIR)
    if all((hp, ap, wp)):
        return hp, ap, wp
    # Try to re-resolve in case a new models directory was created after startup
    new_dir = _resolve_models_dir()
    if new_dir != MODELS_DIR:
        MODELS_DIR = new_dir
        log.info("Retrying pipeline load after resolving new MODELS_DIR=%s", MODELS_DIR)
        hp, ap, wp = _attempt_load(MODELS_DIR)
    return hp, ap, wp

home_pipe, away_pipe, win_pipe = load_pipelines()



def reload_pipelines() -> Dict[str, str]:
    """Reload model pipelines from the resolved ``MODELS_DIR``.

    This updates the module-level ``home_pipe``, ``away_pipe``, and
    ``win_pipe`` variables in-place and returns a small status payload
    suitable for the ``/reload-models`` endpoint.

    Returns
    -------
    dict
        A JSON-serializable status dictionary with keys ``status`` and
        ``detail`` indicating whether reload succeeded.
    """
    global home_pipe, away_pipe, win_pipe

    hp, ap, wp = load_pipelines()
    home_pipe, away_pipe, win_pipe = hp, ap, wp

    if all((home_pipe, away_pipe, win_pipe)):
        log.info("Pipelines reloaded successfully via /reload-models endpoint.")
        return {"status": "success", "detail": "Pipelines reloaded successfully."}

    log.error("Failed to reload pipelines via /reload-models endpoint.")
    return {"status": "error", "detail": "Failed to reload pipelines."}


# ---------------------------------------------------------------
# Team maps
# ---------------------------------------------------------------
TEAM_ABBREVIATIONS = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}
TEAM_CODE_FIX = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS"}
VALID_ABBRS = set(TEAM_ABBREVIATIONS.keys()) | set(TEAM_CODE_FIX.keys()) | set(TEAM_ABBREVIATIONS.values())

def to_team_abbr(name: str) -> str:
    n = str(name).strip()
    if n in TEAM_CODE_FIX:
        return TEAM_CODE_FIX[n]
    if n in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[n]
    if n in VALID_ABBRS:
        return TEAM_CODE_FIX.get(n, n)
    raise ValueError(f"Unknown team: {name}")

# ---------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------

def _glob_latest(dir_path: Path, pattern: str) -> Optional[Path]:
    try:
        matches = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None
    except Exception:
        return None

def _normalize_feature_cols(cols: Dict[str, List[str]] | List[str]) -> List[str]:
    if isinstance(cols, dict):
        return cols.get("numeric", []) + cols.get("categorical", [])
    if isinstance(cols, list):
        return cols
    return []

# FIX: Added back the required helper 'nan_safe_df'
def _nan_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, safe NaN/inf sanitizer for inference."""
    out = df.copy()
    # Replace +/- inf → NaN
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Replace NaN → 0.0 (safe fallback that keeps model running)
    out.fillna(0.0, inplace=True)
    return out

def _ensure_home_away(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    if {"home_team", "away_team"}.issubset(cols):
        return df
    if {"team", "opponent_team", "is_home"}.issubset(cols):
        is_home = df["is_home"].astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})
        return df.assign(
            is_home=is_home,
            home_team=np.where(is_home, df["team"], df["opponent_team"]),
            away_team=np.where(is_home, df["opponent_team"], df["team"]),
        )
    log.warning("Dataset missing home/away columns and team/opponent fallback")
    return df

def _validate_dataset_schema(df: pd.DataFrame, model_objects: Dict[str, Any]) -> None:
    expected = _normalize_feature_cols(model_objects.get("raw_feature_columns", {}))
    if not expected:
        log.warning("No expected feature list available; skipping dataset schema check")
        return
    miss = [c for c in expected if c not in df.columns]
    if miss:
        log.error("Dataset schema mismatch: %d missing engineered features. Sample: %s", len(miss), miss[:10])
        raise RuntimeError(f"Dataset missing required engineered features: {miss[:5]}")

# ---------------------------------------------------------------
# Prediction history helpers
# ---------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _load_history_from_disk() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            normalized: List[Dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    normalized.append(_normalize_history_entry(item))
                except Exception as exc:
                    log.warning("Skipping malformed history entry: %s", exc)
            return normalized[:MAX_HISTORY_ROWS]
    except Exception as exc:
        log.warning("Failed to parse %s: %s", HISTORY_FILE, exc)
    return []


def _persist_history(entries: List[Dict[str, Any]]) -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Unable to persist prediction history: %s", exc)


def _history_snapshot(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with history_lock:
        snapshot = list(prediction_history)
    if limit is None:
        return snapshot
    limit = max(1, min(limit, MAX_HISTORY_ROWS))
    return snapshot[:limit]


def _normalize_history_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a legacy prediction log row into the canonical /history schema."""
    required = {"timestamp", "game_id", "season", "week", "home_score_pred"}
    if required.issubset(raw.keys()):
        return raw

    timestamp_val = raw.get("timestamp") or raw.get("ts") or raw.get("time") or raw.get("when")
    try:
        ts_iso = pd.to_datetime(timestamp_val).to_pydatetime().astimezone(timezone.utc).isoformat()
    except Exception:
        ts_iso = datetime.now(timezone.utc).isoformat()

    home_team = str(raw.get("home_team") or raw.get("home") or "HOME")
    away_team = str(raw.get("away_team") or raw.get("away") or "AWAY")

    def _score(key_options: List[str]) -> Optional[float]:
        for key in key_options:
            if key in raw:
                val = _safe_float(raw.get(key))
                if val is not None:
                    return val
        return None

    home_pred = _score(["home_score_pred", "pred_home", "home_score"])
    away_pred = _score(["away_score_pred", "pred_away", "away_score"])
    home_prob = _score(["home_win_probability", "prob_home"])
    if home_prob is None:
        home_prob = _safe_float(raw.get("probs", {}).get("home") if isinstance(raw.get("probs"), dict) else None)
    away_prob = _score(["away_win_probability", "prob_away"])
    if away_prob is None:
        away_prob = 1 - home_prob if home_prob is not None else None
    if away_prob is None:
        away_prob = 1 - home_prob if home_prob is not None else None
        probs_home_val = raw["probs"].get("home")
    home_prob = _score(["home_win_probability", "prob_home"]) or _safe_float(probs_home_val)
    away_prob = _score(["away_win_probability", "prob_away"]) or (1 - home_prob if home_prob is not None else None)
    point_diff = _safe_float(raw.get("point_diff"))
    if point_diff is None and home_pred is not None and away_pred is not None:
        point_diff = round(home_pred - away_pred, 2)

    actual_home = _score(["actual_home_score", "espn_home", "final_home"])
    actual_away = _score(["actual_away_score", "espn_away", "final_away"])
    predicted_winner = raw.get("predicted_winner")
    if not predicted_winner:
        if home_prob is not None and away_prob is not None:
            predicted_winner = home_team if home_prob >= away_prob else away_team
        elif home_pred is not None and away_pred is not None:
            predicted_winner = home_team if home_pred >= away_pred else away_team
        else:
            predicted_winner = home_team

    actual_winner = raw.get("actual_winner")
    if not actual_winner and actual_home is not None and actual_away is not None:
        actual_winner = home_team if actual_home >= actual_away else away_team

    game_id = raw.get("game_id") or f"legacy-{away_team}@{home_team}"
    season_val = int(_safe_float(raw.get("season")) or datetime.now().year)
    week_val = int(_safe_float(raw.get("week")) or 0)

    return {
        # Use explicit None checks to avoid masking valid 0.0 probabilities
        "away_win_probability": away_prob if away_prob is not None else (1 - home_prob if home_prob is not None else 0.0),
        "game_id": game_id,
        "season": season_val,
        "week": week_val,
        "home_team": home_team,
        "away_team": away_team,
        "home_score_pred": home_pred or 0.0,
        "away_score_pred": away_pred or 0.0,
        "home_win_probability": home_prob or 0.0,
        "away_win_probability": away_prob or (1 - (home_prob or 0.0)),
        "point_diff": point_diff or 0.0,
        "mode": raw.get("mode", "legacy"),
        "prediction_source": raw.get("prediction_source", "legacy-log"),
        "win_threshold_used": raw.get("win_threshold_used"),
        "predicted_winner": predicted_winner,
        "kickoff": raw.get("kickoff") or raw.get("when"),
        "actual_home_score": actual_home,
        "actual_away_score": actual_away,
        "actual_winner": actual_winner,
    }


def _build_history_entry(
    req: "PredictionRequest",
    prediction: "PredictionResponse",
    game_row: Optional[pd.Series]
) -> Dict[str, Any]:
    actual_home = _safe_float(game_row.get("home_points_for")) if game_row is not None else None
    actual_away = _safe_float(game_row.get("away_points_for")) if game_row is not None else None
    kickoff_val = None
    if game_row is not None:
        kickoff_val = game_row.get("kickoff_ts_utc") or game_row.get("kickoff") or game_row.get("gameday")
        if hasattr(kickoff_val, "isoformat"):
            kickoff_val = kickoff_val.isoformat()
        elif kickoff_val is not None:
            try:
                kickoff_val = pd.to_datetime(kickoff_val).isoformat()
            except Exception:
                kickoff_val = str(kickoff_val)

    predicted_winner = req.home_team if prediction.home_win >= prediction.away_win else req.away_team
    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "game_id": f"{req.season}W{req.week}-{req.away_team}@{req.home_team}",
        "season": req.season,
        "week": req.week,
        "home_team": req.home_team,
        "away_team": req.away_team,
        "home_score_pred": prediction.home_score,
        "away_score_pred": prediction.away_score,
        "home_win_probability": prediction.home_win_probability,
        "away_win_probability": prediction.away_win_probability,
        "point_diff": prediction.point_diff,
        "mode": prediction.mode,
        "prediction_source": prediction.prediction_source,
        "win_threshold_used": prediction.win_threshold_used,
        "predicted_winner": predicted_winner,
        "kickoff": kickoff_val,
    }

    if actual_home is not None and actual_away is not None:
        entry.update(
            {
                "actual_home_score": actual_home,
                "actual_away_score": actual_away,
                "actual_winner": req.home_team if actual_home >= actual_away else req.away_team,
            }
        )

    return entry


def _record_prediction_history(
    req: "PredictionRequest",
    prediction: "PredictionResponse",
    game_row: Optional[pd.Series]
) -> Dict[str, Any]:
    entry = _build_history_entry(req, prediction, game_row)
    with history_lock:
        prediction_history.insert(0, entry)
        if len(prediction_history) > MAX_HISTORY_ROWS:
            del prediction_history[MAX_HISTORY_ROWS:]
        _persist_history(prediction_history)
    return entry


def _compute_history_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {"total_predictions": 0, "avg_point_diff": None, "win_rate": None}

    diffs = [abs(float(e.get("point_diff", 0.0))) for e in entries if isinstance(e.get("point_diff"), (int, float))]
    actuals = [e for e in entries if e.get("actual_winner") and e.get("predicted_winner")]

    win_rate = None
    if actuals:
        correct = sum(1 for e in actuals if e["actual_winner"] == e["predicted_winner"])
        win_rate = round(correct / len(actuals), 3)

    avg_diff = round(float(np.mean(diffs)), 2) if diffs else None
    return {
        "total_predictions": len(entries),
        "avg_point_diff": avg_diff,
        "win_rate": win_rate,
    }


def _dataset_status() -> Dict[str, Any]:
    rows = int(dataset_df.shape[0]) if dataset_df is not None else 0
    cols = int(dataset_df.shape[1]) if dataset_df is not None else 0
    return {
        "path": ACTIVE_DATASET_PATH,
        "rows": rows,
        "columns": cols,
        "loaded": bool(dataset_df is not None and not dataset_df.empty),
        "last_loaded": DATASET_LAST_LOADED.isoformat() if DATASET_LAST_LOADED else None,
    }

# ---------------------------------------------------------------
# Lifespan (Startup/Shutdown)
# ---------------------------------------------------------------

# [KEPT: This is the correct, refactored sanity check]
def _sanity_predict(df: pd.DataFrame) -> None:
    global home_pipe, away_pipe, win_pipe
    if df is None or df.empty:
        log.warning("Sanity predict: skipping, no dataset loaded")
        return
        
    # Check if pipelines were loaded
    if home_pipe is None or away_pipe is None or win_pipe is None:
        log.info("Pipelines not loaded; skipping sanity predict")
        return

    # Get a sample row
    sample = df.sample(1, random_state=42).copy()
    
    # We must drop target columns that might exist in the CSV
    # but are not part of the features.
    X = sample.drop(columns=["home_points_for", "away_points_for", "home_win"], errors="ignore")
    failures = []
    
    def _try(pipe, label, method):
        try:
            if method == "predict":
                pipe.predict(X)
            elif method == "predict_proba":
                pipe.predict_proba(X)
        except Exception as e:
            msg = str(e).lower()
            is_not_fitted = (
                SKLEARN_NOTFITTED_AVAILABLE
                and NotFittedError is not None
                and isinstance(e, NotFittedError)
            )
            if is_not_fitted or ("not fitted" in msg):
                failures.append(f"{label} is not fitted")
            else:
                failures.append(f"{label}.{method} failed: {e}")

    _try(home_pipe, "home_pipe", "predict")
    _try(away_pipe, "away_pipe", "predict")
    _try(win_pipe, "win_pipe", "predict_proba")

    if failures:
        raise RuntimeError(f"Sanity prediction failed: {'; '.join(failures)}")

# [KEPT: This is the correct, refactored lifespan]
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model_objects, dataset_df, home_pipe, away_pipe, win_pipe, ACTIVE_DATASET_PATH, DATASET_LAST_LOADED 
    log.info("=" * 60)
    log.info("STARTUP: NFL Prediction API v2.1.1 (Refactored)")
    log.info("=" * 60)
    
    model_objects = {} # Initialize global
    
    try:
        # Load metadata to get feature lists and thresholds
        meta_path = MODELS_DIR / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")
        meta_content = meta_path.read_text(encoding="utf-8")
        meta = json.loads(meta_content)

        # Determine feature metadata file path. If the training pipeline wrote
        # "feature_metadata.json" into the same directory, use it; else fall
        # back to a top-level feature metadata file. This covers both the
        # legacy and enhanced pipeline formats.
        feat_meta_path = MODELS_DIR / "feature_metadata.json"
        if not feat_meta_path.exists():
            # Try parent directory as fallback
            parent_fm = MODELS_DIR.parent / "feature_metadata.json"
            if parent_fm.exists():
                feat_meta_path = parent_fm
            else:
                raise FileNotFoundError(
                    f"Missing feature metadata file at {feat_meta_path} or {parent_fm}"
                )

        feat_meta_content = feat_meta_path.read_text(encoding="utf-8")

        # Parse feature metadata: support both list-of-dicts and dict formats
        if feat_meta_content.strip().startswith("["):
            feat_meta_list = json.loads(feat_meta_content)
            model_objects["raw_feature_columns"] = {
                "numeric": [f["feature"] for f in feat_meta_list if f.get("dtype") == "float64"],
                "categorical": [f["feature"] for f in feat_meta_list if f.get("dtype") != "float64"],
            }
        elif feat_meta_content.strip().startswith("{"):
            feat_meta_dict = json.loads(feat_meta_content)
            model_objects["raw_feature_columns"] = {
                "numeric": list(feat_meta_dict.get("numeric", [])),
                "categorical": list(feat_meta_dict.get("categorical", [])),
            }
        else:
            raise ValueError(
                "Unknown format for feature_metadata.json: expected JSON list or dict"
            )

        # Store only the metadata needed by other routes
        # Mode: fallback to 'production' if unspecified
        model_objects["mode"] = str(meta.get("mode", meta.get("training_mode", "production")))

        # Determine optimal win threshold. The enhanced pipeline may not include
        # an explicit optimal threshold in the metadata; fallback to 0.5.
        win_summary = meta.get("validation_summary", {}).get("win", {})
        # Support alternative naming in enhanced metadata (e.g. 'win' -> 'win' metrics)
        # Attempt to fetch any of the known keys; default to 0.5
        threshold = win_summary.get("optimal_threshold") or win_summary.get("optimal_threshold_f1") or win_summary.get("optimal_threshold_acc")
        if threshold is None:
            threshold = 0.5
        model_objects["win_threshold_optimal"] = float(threshold)

        log.info("✓ Pipelines loaded (from module level)")
        log.info("✓ Metadata and feature lists loaded from %s and %s", meta_path, feat_meta_path)
        log.info("Model registry keys: %s", list(model_objects.keys()))

    except Exception as e:
        log.error("✗ Failed to load metadata: %s", e, exc_info=True)
        log.warning(
            "Continuing without models/metadata; /health will report unhealthy"
        )

    # --- Dataset Loading ---
    ds_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    log.info("Dataset path: %s", ds_path)
    if not ds_path.exists():
        # FIX: Removed hardcoded Windows path
        alternates = (DATA_DIR / "game_features.csv", DATA_DIR / "merged_game_features.csv")
        ds_alt = next((a for a in alternates if a.exists()), None)
        if ds_alt is not None:
            ds_path = ds_alt
            log.info("Using alternate dataset: %s", ds_path)
    ACTIVE_DATASET_PATH = str(ds_path)

    try:
        if ds_path.exists():
            df = pd.read_csv(ds_path)
            if not df.empty:
                df.columns = [c.strip() for c in df.columns]
                df = _ensure_home_away(df)
                try:
                    if model_objects:
                        _validate_dataset_schema(df, model_objects)
                except Exception as e:
                    log.warning("Dataset schema validation failed: %s", e)
                dataset_df = df
                DATASET_LAST_LOADED = datetime.now(timezone.utc)
                try:
                    # Pass df to the refactored sanity check
                    _sanity_predict(df) 
                    log.info("✓ Sanity prediction passed")
                except Exception as e:
                    log.warning("Sanity prediction failed: %s; continuing", e)
                log.info("✓ Dataset loaded: %d rows, %d columns", len(df), df.shape[1])
            else:
                dataset_df = pd.DataFrame()
                log.warning("Dataset CSV is empty at %s", ds_path)
        else:
            dataset_df = pd.DataFrame()
            log.warning("Dataset file does not exist: %s", ds_path)
    except Exception as e:
        dataset_df = pd.DataFrame()
        log.error("Failed to load dataset: %s", e, exc_info=True)

    log.info("=" * 60)
    log.info("STARTUP COMPLETE")
    log.info("Pipelines: %s", "✓ Loaded" if home_pipe else "✗ Missing")
    log.info("Metadata: %s", "✓ Loaded" if model_objects else "✗ Missing")
    log.info("Dataset: %s", "✓ Loaded" if dataset_df is not None and not dataset_df.empty else "✗ Missing")
    log.info("=" * 60)
    try:
        history_seed = _load_history_from_disk()
        with history_lock:
            prediction_history.clear()
            prediction_history.extend(history_seed)
        log.info("Prediction history entries loaded: %d", len(prediction_history))
    except Exception as exc:
        log.warning("Failed loading prediction history: %s", exc)
    try:
        yield
    finally:
        log.info("SHUTDOWN: Cleaning up resources")

# FastAPI app
app = FastAPI(title="NFL ML Predictions API", version="2.1.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["https://nfl-ml-predictions.vercel.app"],
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

if SERVE_FRONTEND:
    for candidate in (FRONTEND_BUILD, FRONTEND_DIST):
        if candidate.exists():
            app.mount("/", StaticFiles(directory=str(candidate), html=True), name="frontend")
            log.info("Serving frontend from %s", candidate)
            break
    else:
        log.warning("SERVE_FRONTEND=true but no frontend build found.")

# -----------------------
# Schemas
# -----------------------
class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    season: int
    week: int

class PredictionResponse(BaseModel):
    home_score: float
    away_score: float
    home_win_probability: float
    away_win_probability: float
    home_win: float
    away_win: float
    point_diff: float
    mode: str
    prediction_source: str
    win_classifier_used: bool
    win_probability_source: str  # 'classifier' | 'classifier-margin' | 'legacy-sigmoid'
    win_threshold_used: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    mode: str
    reason: str


class PredictionHistoryEntry(BaseModel):
    timestamp: datetime
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_score_pred: float
    away_score_pred: float
    home_win_probability: float
    away_win_probability: float
    point_diff: float
    mode: str
    prediction_source: str
    win_threshold_used: Optional[float] = None
    predicted_winner: str
    kickoff: Optional[str] = None
    actual_home_score: Optional[float] = None
    actual_away_score: Optional[float] = None
    actual_winner: Optional[str] = None


class PredictionHistoryResponse(BaseModel):
    total: int
    limit: int
    entries: List[PredictionHistoryEntry]


class StatusOverview(BaseModel):
    health: HealthResponse
    dataset: Dict[str, Any]
    models: Dict[str, Any]
    history: Dict[str, Any]

class ScheduleGame(BaseModel):
    season: int
    week: int
    kickoff: datetime
    home_team: str
    home_abbr: str
    away_team: str
    away_abbr: str
    predicted_home_score: Optional[float] = None
    predicted_away_score: Optional[float] = None
    home_win_probability: Optional[float] = None
    away_win_probability: Optional[float] = None

# -----------------------
# Helpers
# -----------------------
def _resolve_schedule_path() -> Path:
    env_val = os.getenv("SCHEDULE_PATH", "").strip()
    env_path = Path(env_val) if env_val else None
    if env_path and env_path.exists():
        log.info("Using schedule from SCHEDULE_PATH=%s", env_path)
        return env_path
    if DEFAULT_SCHEDULE.exists():
        log.info("Using default schedule at %s", DEFAULT_SCHEDULE)
        return DEFAULT_SCHEDULE
    latest = _glob_latest(DATA_DIR, "Nfl_schedule_*.csv")
    if latest and latest.exists():
        log.info("Using latest schedule candidate at %s", latest)
        return latest
    log.warning("No schedule file found; returning DEFAULT_SCHEDULE path (may not exist): %s", DEFAULT_SCHEDULE)
    return DEFAULT_SCHEDULE

# Simple in-memory cache for the schedule CSV to avoid re-reading the same
# file on every request. The cache is invalidated automatically when the
# underlying file's modification time changes.
SCHEDULE_CACHE_PATH: Optional[Path] = None
SCHEDULE_CACHE_MTIME: Optional[float] = None
SCHEDULE_CACHE_DF: Optional[pd.DataFrame] = None


def _load_schedule_df(spath: Path) -> pd.DataFrame:
    """Load the schedule CSV with a lightweight mtime-aware cache.

    Parameters
    ----------
    spath : pathlib.Path
        Filesystem path to the schedule CSV.

    Returns
    -------
    pandas.DataFrame
        A *copy* of the cached schedule dataframe. Mutations to the
        returned dataframe will not affect the cache.

    Raises
    ------
    FileNotFoundError
        If the path does not exist. Callers handle this uniformly so the
        API can respond with a 503/500 as appropriate.
    """
    global SCHEDULE_CACHE_PATH, SCHEDULE_CACHE_MTIME, SCHEDULE_CACHE_DF

    try:
        stat = spath.stat()
    except FileNotFoundError:
        # Let callers handle the missing-file case consistently.
        raise

    mtime = stat.st_mtime
    if (
        SCHEDULE_CACHE_PATH == spath
        and SCHEDULE_CACHE_MTIME == mtime
        and SCHEDULE_CACHE_DF is not None
    ):
        # Return a copy so route handlers can safely mutate (filter, sort)
        # without polluting the cache.
        return SCHEDULE_CACHE_DF.copy()

    df = pd.read_csv(spath)
    SCHEDULE_CACHE_PATH = spath
    SCHEDULE_CACHE_MTIME = mtime
    SCHEDULE_CACHE_DF = df
    return df.copy()


def get_current_nfl_context() -> Dict[str, Any]:
    """Infer the current NFL season/week context from the dataset.

    The function uses the loaded feature dataset (if available) to
    determine the last completed game week and the *next* week to
    generate predictions for. If no dataset is present, it falls back
    to a conservative preseason-style default.

    Returns
    -------
    dict
        Keys include:
        - ``current_season`` : int
        - ``last_completed_season`` : int
        - ``last_completed_week`` : int
        - ``next_prediction_season`` : int
        - ``next_prediction_week`` : int
        - ``status`` : {"nfl_season_active", "offseason", "preseason_or_early"}
    """
    now = datetime.now()
    cur_season = now.year if now.month >= 8 else now.year - 1
    
    # Default fallback values
    default_context = {
        "current_season": cur_season,
        "last_completed_season": cur_season,
        "last_completed_week": 1,
        "next_prediction_season": cur_season,
        "next_prediction_week": 2,
        "status": "preseason_or_early"
    }
    
    # Try to get more accurate context from dataset if available
    if dataset_df is not None and not dataset_df.empty:
        try:
            if "season" in dataset_df.columns and "week" in dataset_df.columns:
                max_season = int(dataset_df["season"].max())
                max_week_in_season = int(dataset_df[dataset_df["season"] == max_season]["week"].max())
                
                default_context["last_completed_season"] = max_season
                default_context["last_completed_week"] = max_week_in_season
                default_context["next_prediction_season"] = max_season
                default_context["next_prediction_week"] = min(max_week_in_season + 1, 18)
                default_context["status"] = "nfl_season_active"
        except Exception as e:
            log.warning(f"Failed to extract context from dataset: {e}")
    
    return default_context

def build_game_mask(df: pd.DataFrame, season: int, week: int,
                    home_abbr: str, away_abbr: str) -> pd.Series:
    """Build a boolean mask for a single game in the feature dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature dataset containing game-level rows.
    season : int
        Target season.
    week : int
        Target week number.
    home_abbr : str
        Home team abbreviation (e.g. ``"NE"``).
    away_abbr : str
        Away team abbreviation (e.g. ``"NYJ"``).

    Returns
    -------
    pandas.Series
        Boolean mask aligned with ``df.index`` where True marks the
        target game row. If required columns are missing, returns an
        all-False mask.
    """
    season_mask = (df.get("season") == season) if "season" in df.columns else pd.Series(False, index=df.index)
    week_mask = (df.get("week") == week) if "week" in df.columns else pd.Series(False, index=df.index)
    home_col = df.get("home_team")
    away_col = df.get("away_team")
    if home_col is None or away_col is None:
        return pd.Series(False, index=df.index)
    mask = season_mask & week_mask & (home_col == home_abbr) & (away_col == away_abbr)
    return mask

# -----------------------
# Routes
# -----------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Simple liveness + readiness probe for the API.

    Returns healthy only when:
      * all three pipelines are loaded,
      * model metadata is available, and
      * the feature dataset is loaded and non-empty.
    """
    if all((home_pipe, away_pipe, win_pipe, model_objects, dataset_df is not None and not dataset_df.empty)):
        mode = model_objects.get("mode", "production") if isinstance(model_objects, dict) else "production"
        return HealthResponse(status="healthy", mode=str(mode), reason="models and dataset loaded")

    reasons = []
    if not all((home_pipe, away_pipe, win_pipe)):
        reasons.append("models not loaded")
    if not model_objects:
        reasons.append("metadata not loaded")
    if dataset_df is None or dataset_df.empty:
        reasons.append("dataset not loaded")

    return HealthResponse(status="unhealthy", mode="none", reason="; ".join(reasons))


@app.get("/history", response_model=PredictionHistoryResponse)
def get_prediction_history(limit: int = 50) -> PredictionHistoryResponse:
    limit = max(1, min(limit, MAX_HISTORY_ROWS))
    with history_lock:
        total = len(prediction_history)
        entries = [dict(item) for item in prediction_history[:limit]]
    return PredictionHistoryResponse(total=total, limit=len(entries), entries=entries)


@app.get("/status/overview", response_model=StatusOverview)
def status_overview(limit: int = 5) -> StatusOverview:
    limit = max(1, min(limit, MAX_HISTORY_ROWS))
    with history_lock:
        total = len(prediction_history)
        recent = [dict(item) for item in prediction_history[:limit]]
        metrics_source = list(prediction_history)

    metrics = _compute_history_metrics(metrics_source)
    dataset_status = _dataset_status()
    models_status = {
        "mode": model_objects.get("mode", "production") if isinstance(model_objects, dict) else "unknown",
        "win_threshold": model_objects.get("win_threshold_optimal") if isinstance(model_objects, dict) else None,
        "pipelines_loaded": all((home_pipe, away_pipe, win_pipe)),
    }

    history_payload = {
        "total": total,
        "recent": recent,
        "metrics": metrics,
    }

    return StatusOverview(
        health=health(),
        dataset=dataset_status,
        models=models_status,
        history=history_payload,
    )

@app.get("/debug")
def debug_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "active",
        "ALLOWED_ORIGINS": ALLOWED_ORIGINS,
    }
    try:
        mpath = MODELS_DIR / "metadata.json"
        if mpath.is_file():
            out["metadata"] = json.loads(mpath.read_text(encoding="utf-8"))
        tr = _glob_latest(MODELS_DIR, "training_report*.json")
        out["training_report_present"] = tr is not None
        if tr is not None:
            out["training_report_path"] = tr.name
        if model_objects and isinstance(model_objects, dict):
            out["preprocessor_n_features_in_"] = "N/A (in pipeline)"
            out["raw_feature_columns_counts"] = {
                "numeric": len(model_objects.get("raw_feature_columns", {}).get("numeric", [])),
                "categorical": len(model_objects.get("raw_feature_columns", {}).get("categorical", [])),
            }
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out

@app.get("/report/training")
def report_training() -> Dict[str, Any]:
    tr = _glob_latest(MODELS_DIR, "training_report*.json") or (MODELS_DIR / "training_report.json" if (MODELS_DIR / "training_report.json").exists() else None)
    if tr is None or not tr.exists():
        raise HTTPException(404, "training report not found")
    return json.loads(tr.read_text(encoding="utf-8"))

@app.get("/report/calibration")
def report_calibration() -> Dict[str, Any]:
    tr = _glob_latest(MODELS_DIR, "training_report*.json") or (MODELS_DIR / "training_report.json" if (MODELS_DIR / "training_report.json").exists() else None)
    if tr is None or not tr.exists():
        raise HTTPException(404, "training report not found")
    j = json.loads(tr.read_text(encoding="utf-8"))
    win = j.get("models", {}).get("win_clf", {})
    return {
        "reliability_bins": win.get("reliability_bins", []),
        "auc_val": win.get("auc_val"),
        "brier_val": win.get("brier_val"),
        "logloss_val": win.get("logloss_val"),
        "optimal_threshold": win.get("optimal_threshold"),
        "optimal_threshold_f1": win.get("optimal_threshold_f1"),
        "optimal_threshold_acc": win.get("optimal_threshold_acc"),
    }

@app.get("/schedule/next-week", response_model=List[ScheduleGame])
def get_next_week_schedule() -> List[ScheduleGame]:
    """Return the schedule for the *next* upcoming NFL week.

    The function:
    1. Resolves the active schedule CSV path.
    2. Uses a cached dataframe for performance.
    3. Normalizes team names/abbreviations.
    4. Derives a UTC kickoff timestamp from ``gameday``/``gametime``.
    5. Selects the week of the next future game (or last week if all
       games are in the past).

    Returns
    -------
    list of ScheduleGame
        Pydantic model instances ready to be serialized as JSON.
    """
    spath = _resolve_schedule_path()
    try:
        df = _load_schedule_df(spath)
    except FileNotFoundError:
        log.error(f"Schedule file not found: {spath}")
        raise HTTPException(503, "Schedule data unavailable")
    except Exception as exc:
        log.error(f"Failed to load schedule from {spath}: {exc}")
        raise HTTPException(500, "Failed to load schedule data from server")

    # Parse kickoff timestamps
    if "kickoff_ts_utc" not in df.columns:
        df["kickoff_ts_utc"] = pd.to_datetime(
            df["gameday"].astype(str) + " " + df["gametime"].fillna("00:00").astype(str),
            errors="coerce",
            utc=True
        )

    now = pd.Timestamp.now(tz="UTC")
    future = df[df["kickoff_ts_utc"].notna() & (df["kickoff_ts_utc"] >= now)]
    current_week = int(future["week"].min()) if not future.empty else int(df["week"].max())

    # Games for the identified week
    week_games = df[df["week"] == current_week]
    
    results = []
    for _, r in week_games.iterrows():
        try:
            h = to_team_abbr(r["home_team"])
            a = to_team_abbr(r["away_team"])
            kickoff_val = r.get("kickoff_ts_utc")

            # Ensure we always return a concrete datetime, even if the
            # raw schedule data is incomplete.
            if kickoff_val is not None and hasattr(kickoff_val, "to_pydatetime") and not pd.isna(kickoff_val):
                kickoff_val = kickoff_val.to_pydatetime()
            elif pd.isna(kickoff_val) or kickoff_val is None:
                kickoff_val = datetime.now(timezone.utc)
                log.warning(f"Invalid kickoff time for {a}@{h}, using current time")
            
            results.append(
                ScheduleGame(
                    game_id=r.get("game_id", f"{r['season']}_{r['week']:02d}_{a}_{h}"),
                    season=int(r["season"]),
                    week=int(r["week"]),
                    home_team=r["home_team"],
                    away_team=r["away_team"],
                    home_abbr=h,
                    away_abbr=a,
                    kickoff=kickoff_val,
                )
            )
        except Exception as e:
            log.warning(f"Skipping game row due to error: {e}")
            continue
    
    return results


# -----------------------------------------------------------
# Prediction Route (FIXED)
# -----------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict_game(req: PredictionRequest) -> PredictionResponse:
    """Run a single-game prediction using precomputed features and pipelines.

    This endpoint *does not* recompute features. Instead, it:
      1. Locates the corresponding row in the pre-built feature dataset.
      2. Selects the exact feature columns the pipelines were trained on.
      3. Performs basic NaN/inf sanitization.
      4. Runs home/away score regressors and win classifier.
      5. Applies the optimal win threshold from training metadata.

    Parameters
    ----------
    req : PredictionRequest
        JSON body specifying ``home_team``, ``away_team``, ``season``,
        and ``week``.

    Returns
    -------
    PredictionResponse
        Model predictions for scores, probabilities, and winner flags.

    Raises
    ------
    fastapi.HTTPException
        - 400 for invalid team names.
        - 404 when the game is missing from the feature dataset.
        - 503 when models/metadata/dataset are not loaded.
        - 500 for unexpected model inference errors.
    """
    # 0. Check if models and data are loaded (using the global variables)
    if not all((home_pipe, away_pipe, win_pipe)):
        log.error("Prediction failed: Pipelines are not loaded")
        raise HTTPException(503, "Models are not loaded. Server is unhealthy.")

    if model_objects is None:
        log.error("Prediction failed: model_objects is None")
        raise HTTPException(503, "Model metadata is not loaded. Server is unhealthy.")

    if dataset_df is None or dataset_df.empty:
        log.error("Prediction failed: dataset_df is None or empty")
        raise HTTPException(503, "Feature dataset is not loaded.")

    # 1. Normalize teams to canonical abbreviations (handles legacy codes)
    try:
        home = to_team_abbr(req.home_team)
        away = to_team_abbr(req.away_team)
    except ValueError as e:
        raise HTTPException(400, f"Invalid team name: {e}")

    # 2. Locate the corresponding game row in the feature dataset
    df = dataset_df  # type: ignore[assignment]
    assert df is not None  # for type-checkers

    mask = build_game_mask(df, req.season, req.week, home, away)
    games = df[mask]

    if games.empty:
        log.warning(
            "No feature row found for game %s@%s S=%s W=%s",
            away,
            home,
            req.season,
            req.week,
        )
        raise HTTPException(404, "Game not found in feature dataset.")

    if len(games) > 1:
        log.warning(
            "Multiple feature rows found for game %s@%s S=%s W=%s; using first row.",
            away,
            home,
            req.season,
            req.week,
        )

    game_row = games.iloc[0]

    # 3. Get expected features from loaded metadata
    raw_cols = model_objects.get("raw_feature_columns")
    if not raw_cols:
        log.error("Server configuration error: raw_feature_columns not in model_objects")
        raise HTTPException(500, "Server configuration error: Missing feature list")

    expected_features = _normalize_feature_cols(raw_cols)
    if not expected_features:
        log.error("Server configuration error: expected_features resolved to an empty list")
        raise HTTPException(500, "Server configuration error: Empty feature list")

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        log.error("Dataset missing required features: %s", missing[:10])
        raise HTTPException(500, "Feature dataset missing required engineered features.")

    # 4. Build a single-row feature matrix X
    X = pd.DataFrame([game_row])
    # Drop known target columns if present
    X = X.drop(columns=["home_points_for", "away_points_for", "home_win"], errors="ignore")
    # Reorder/limit to the expected feature set
    X = X[expected_features]

    # Final NaN/inf clean, then inference
    X = _nan_safe_df(df=X)

    # 5. Score prediction (home/away regressors)
    try:
        home_score = float(home_pipe.predict(X)[0])  # type: ignore[call-arg]
        away_score = float(away_pipe.predict(X)[0])  # type: ignore[call-arg]
    except Exception as e:
        log.error("Score prediction failed: %s", e, exc_info=True)
        raise HTTPException(500, "Score prediction failed.")

    # 6. Win probability using classifier when available
    home_win_prob: Optional[float] = None
    away_win_prob: Optional[float] = None
    win_classifier_used = False
    win_probability_source = "none"
    win_threshold_used: Optional[float] = None

    try:
        if SKLEARN_CHECK_AVAILABLE and check_is_fitted is not None:
            try:
                check_is_fitted(win_pipe)  # type: ignore[arg-type]
            except Exception:
                # Not fitted or check failed; we'll fall back below.
                pass
            else:
                probs = win_pipe.predict_proba(X)[0]  # type: ignore[call-arg]
                # Determine index of the "home win" class in a robust way
                idx_home = 0
                classes = getattr(win_pipe, "classes_", None)
                if classes is not None:
                    try:
                        cls_list = list(classes)
                        if 1 in cls_list:
                            idx_home = cls_list.index(1)
                        elif "home_win" in cls_list:
                            idx_home = cls_list.index("home_win")
                    except Exception:
                        idx_home = 0

                home_win_prob = float(probs[idx_home])
                away_win_prob = float(1.0 - home_win_prob)
                win_classifier_used = True
                win_probability_source = "classifier"
                win_threshold_used = float(model_objects.get("win_threshold_optimal", 0.5))
    except Exception as e:
        log.warning("Win classifier failed; falling back to score-margin heuristic: %s", e, exc_info=True)

    # 7. Fallback: map score margin to a probability via logistic
    if home_win_prob is None or away_win_prob is None:
        margin = home_score - away_score
        home_win_prob = float(1.0 / (1.0 + np.exp(-margin)))
        away_win_prob = float(1.0 - home_win_prob)
        win_probability_source = "legacy-sigmoid"
        if win_threshold_used is None:
            win_threshold_used = 0.5

    # 8. Derive binary win flags from probabilities
    threshold = win_threshold_used if win_threshold_used is not None else 0.5
    home_win_flag = float(home_win_prob >= threshold)
    away_win_flag = float(1.0 - home_win_flag)

    point_diff = float(home_score - away_score)

    resp = PredictionResponse(
        home_score=home_score,
        away_score=away_score,
        home_win_probability=home_win_prob,
        away_win_probability=away_win_prob,
        home_win=home_win_flag,
        away_win=away_win_flag,
        point_diff=point_diff,
        mode=str(model_objects.get("mode", "production")),
        prediction_source="api-v2",
        win_classifier_used=win_classifier_used,
        win_probability_source=win_probability_source,
        win_threshold_used=win_threshold_used,
    )

    # 9. Record prediction history (best-effort; failures do not affect API response)
    try:
        _record_prediction_history(req, resp, game_row)
    except Exception as exc:
        log.warning("Failed to record prediction history: %s", exc)

    return resp


# -----------------------
# Batch predict next week
# -----------------------
@app.get("/predict/next-week")
def predict_next_week() -> Dict[str, Any]:
    """Generate predictions for all games in the next NFL week.

    The function:
    1. Uses :func:`get_current_nfl_context` to determine the next
       prediction season/week.
    2. Loads the schedule CSV and filters games for that week.
    3. For each scheduled game, calls :func:`predict_game` with the
       appropriate teams/season/week.
    4. Returns a JSON payload containing context plus per-game results.

    Games that exist in the schedule but are missing from the feature
    dataset are reported with an ``error`` key instead of a prediction.
    """
    if not all((home_pipe, away_pipe, win_pipe)):
        raise HTTPException(503, "Models not loaded.")
    
    # Get the context to determine which week to predict
    context = get_current_nfl_context()
    pred_season = context["next_prediction_season"]
    pred_week = context["next_prediction_week"]
    
    # Load the schedule
    spath = _resolve_schedule_path()
    try:
        schedule_df = _load_schedule_df(spath)
    except FileNotFoundError:
        log.error(f"Schedule file not found: {spath}")
        raise HTTPException(503, "Schedule data unavailable for next-week predictions")
    except Exception as exc:
        log.error(f"Failed to load schedule from {spath}: {exc}")
        raise HTTPException(500, "Failed to load schedule data for next-week predictions")
    
    # Filter to the target week
    week_games = schedule_df[
        (schedule_df["season"] == pred_season) & (schedule_df["week"] == pred_week)
    ]
    
    results = []
    for _, row in week_games.iterrows():
        try:
            home_abbr = to_team_abbr(row["home_team"])
            away_abbr = to_team_abbr(row["away_team"])
            
            # Call the single-game prediction endpoint
            req = PredictionRequest(
                home_team=home_abbr,
                away_team=away_abbr,
                season=pred_season,
                week=pred_week
            )
            prediction = predict_game(req)
            
            results.append({
                "game": {
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_abbr": home_abbr,
                    "away_abbr": away_abbr,
                    "season": pred_season,
                    "week": pred_week
                },
                "prediction": prediction.model_dump()
            })
        except HTTPException as http_exc:
            # Propagate HTTP exceptions
            if http_exc.status_code >= 500:
                raise
            # For 400/404, record as an error in the results
            results.append({
                "game": {
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "season": pred_season,
                    "week": pred_week
                },
                "error": http_exc.detail
            })
        except Exception as exc:
            log.warning(f"Failed to predict {row.get('away_team')}@{row.get('home_team')}: {exc}")
            results.append({
                "game": {
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "season": pred_season,
                    "week": pred_week
                },
                "error": str(exc)
            })
    
    return {
        "context": context,
        "season": pred_season,
        "week": pred_week,
        "predictions": results,
        "total_games": len(results)
    }


# -----------------------
# Reload models endpoint
# -----------------------
@app.post("/reload-models")
def reload_models():
    """
    Reload the model pipelines from the current MODELS_DIR.
    Use this endpoint after updating or replacing model artifacts.
    """
    result = reload_pipelines()
    return result

# -----------------------
# Retrain stub
# -----------------------
@app.post("/retrain")
def retrain(background: BackgroundTasks):
    def job():
        log.info("Retraining job started...")
        # Example: C:\Python311\python.exe pipeline_enhanced.py --data ...
        pass
    background.add_task(job)
    return {"status": "accepted", "detail": "Retraining started"}
# CHANGE-LOG: Stub stays lean while highlighting where to hook long-running training.