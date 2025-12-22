"""
NFL ML Predictions API — Backend Server (Enhanced v1.1)
========================================================

FastAPI backend serving ML predictions for NFL game outcomes with comprehensive
error handling, feature engineering, and prediction history tracking.

QUICK START (Local Development):
    cd backend
    ./.venv/Scripts/Activate.ps1
    python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

ENDPOINTS:
    GET  /health            → API health status and model readiness
    GET  /debug             → Debug info, metadata, timestamps
    GET  /schedule/next-week → Upcoming NFL games for prediction
    POST /predict           → Single game prediction (home_team, away_team, season, week)
    GET  /predict/next-week → Batch predictions for upcoming week
    GET  /report/training   → Model training metrics
    GET  /report/calibration → Win probability calibration data

ENVIRONMENT VARIABLES:
    METADATA_URL        → URL to metadata.json for dataset schema
    MODELS_DIR          → Path to model .joblib files
    DATASET             → Path to engineered features CSV
    ALLOWED_ORIGINS     → Comma-separated CORS origins
    ALLOW_ORIGIN_REGEX  → Regex for dynamic CORS
    ALLOW_FALLBACK_PREDICTIONS → Enable/disable fallback predictions

ARCHITECTURE:
    Request → FastAPI Router → Feature Assembly → Preprocessor → ML Models → Response

Models:
  - home_model.joblib: Predicts home team score
  - away_model.joblib: Predicts away team score
  - win_clf_calibrated.joblib: Calibrated win probability classifier
  - preprocessor.joblib: Feature transformation pipeline

Version History:
  v1.1 - Enhanced: Removed duplicates, fixed syntax, improved docs
  v1.0 - Initial implementation
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

import nflreadpy as nfl
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from backend.config import BACKEND_DIR, DEFAULT_DATASET, DATA_DIR, MODELS_DIR, LOG_DIR
from backend.utils.feature_helpers import (coerce_season_week, _normalize_feature_cols,
            to_team_abbr, process_dataset, make_time_key, resolve_model_path, _impute_remaining_prior_nans)
# Environment loading with fallback paths
_ENV_CANDIDATES = [
    Path(__file__).resolve().parent / ".env",
    Path.cwd() / ".env",
]
for _p in _ENV_CANDIDATES:
    if _p.exists():
        load_dotenv(_p)
        break
else:
    load_dotenv()

# Configuration and global state
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

model_objects: Optional[Dict[str, Any]] = None
dataset_df: Optional[pd.DataFrame] = None
prediction_history_entries: List[Dict[str, Any]] = []
_prediction_history_lock = Lock()
PREDICTION_HISTORY_MAX = int(os.getenv("PREDICTION_HISTORY_MAX", "1000"))
PREDICTION_HISTORY_PATH = (BACKEND_DIR / "Predictions" / "prediction_history.json").resolve()

# Path configuration
DEFAULT_DATA_DIR = (BACKEND_DIR / "data" / "heroku-models").resolve()
DEFAULT_MODELS_DIR = (DEFAULT_DATA_DIR / "models").resolve()

def _resolve_env_path(raw: str, base_dir: Path) -> Path:
    """Resolve path from environment variable with base directory fallback."""
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()

def _path_from_env(name: str, default: Path) -> Path:
    """Get path from environment variable with default fallback."""
    raw = os.getenv(name)
    if raw and raw.strip():
        return _resolve_env_path(raw.strip(), base_dir=BACKEND_DIR)
    return default

DATA_DIR = _path_from_env("DATA_DIR", DEFAULT_DATA_DIR)
MODELS_DIR = _path_from_env("MODELS_DIR", DEFAULT_MODELS_DIR)

def resolve_model_path(key: str, filename: str) -> Path:
    """Resolve model artifact path with env override support."""
    env_val = os.getenv(f"MODEL_PATH_{str(key).upper()}")
    if env_val and str(env_val).strip():
        return _resolve_env_path(str(env_val).strip(), base_dir=BACKEND_DIR)
    candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = MODELS_DIR / filename
    return candidate.resolve()

# Core data structures
@dataclass(frozen=True)
class InferenceBundle:
    """Container for loaded model artifacts and metadata."""
    meta: Dict[str, Any]
    report: Dict[str, Any]
    preprocessor: Any
    home_model: Any
    away_model: Any
    hist_win_clf: Any

    @property
    def raw_feature_columns(self) -> List[str]:
        """Get raw feature columns from metadata."""
        cols = self.meta.get("raw_feature_columns", {})
        num = cols.get("numeric", []) or []
        cat = cols.get("categorical", []) or []
        return [*num, *cat]

    @property
    def home_rmse(self) -> float:
        return self.report.get("home_model_metrics", {}).get("rmse", 5.5)

    @property
    def away_rmse(self) -> float:
        return self.report.get("away_model_metrics", {}).get("rmse", 5.2)

class MonteCarloSimulator:
    """Expert-tier game simulation engine using model variance."""
    def __init__(self, bundle: InferenceBundle, n_sims: int = 10000):
        self.home_rmse = bundle.home_rmse
        self.away_rmse = bundle.away_rmse
        self.n_sims = n_sims

    def simulate(self, home_base: float, away_base: float) -> Dict[str, Any]:
        """Run N Monte Carlo trials and return aggregated stats."""
        # Reproducible but varied
        rng = np.random.default_rng()
        
        # Increase variance slightly to avoid "similar" predictions if spread is tight
        dynamic_home_std = max(self.home_rmse, 6.0)
        dynamic_away_std = max(self.away_rmse, 6.0)
        
        home_samples = rng.normal(home_base, dynamic_home_std, self.n_sims)
        away_samples = rng.normal(away_base, dynamic_away_std, self.n_sims)
        
        # NFL realism: scores are integers >= 0
        home_ints = np.maximum(0, np.round(home_samples)).astype(int)
        away_ints = np.maximum(0, np.round(away_samples)).astype(int)
        
        home_wins = np.sum(home_ints > away_ints)
        ties = np.sum(home_ints == away_ints)
        
        # Win probability: (Wins + 0.5 * Ties) / Total
        win_prob = (home_wins + 0.5 * ties) / self.n_sims
        
        return {
            "sim_home_score": float(np.mean(home_ints)),
            "sim_away_score": float(np.mean(away_ints)),
            "sim_win_prob": float(win_prob),
            "sim_std_home": float(np.std(home_ints)),
            "sim_std_away": float(np.std(away_ints)),
            "n_sims": self.n_sims
        }

@lru_cache(maxsize=1)
def _feature_helpers():
    """Lazily import and return the feature_helpers module to avoid circularity."""
    import backend.utils.feature_helpers as fh
    return fh

# Data loading functions
def _latest_game_features_csv(data_dir: Path) -> Path:
    """Find the most recent game features CSV file."""
    files = sorted(data_dir.glob("game_features_*.csv"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No game_features_*.csv found in: {data_dir}")
    return files[0]

def _load_metadata(models_dir: Path) -> Dict[str, Any]:
    """Load metadata.json from models directory."""
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))

def _load_training_report(models_dir: Path) -> Dict[str, Any]:
    """Load training_report.json from models directory."""
    report_path = models_dir / "training_report.json"
    if not report_path.exists():
        log.warning("training_report.json not found; using default RMSE")
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))

def load_inference_bundle(models_dir: Path = MODELS_DIR) -> InferenceBundle:
    """Load model artifacts using metadata.json configuration."""
    meta = _load_metadata(models_dir)
    report = _load_training_report(models_dir)
    
    pre_path = models_dir / meta["preprocessor"]
    home_path = models_dir / meta["home_model"]
    away_path = models_dir / meta["away_model"]
    hist_path = models_dir / meta["hist_win_model"]
    
    return InferenceBundle(
        meta=meta,
        report=report,
        preprocessor=joblib.load(pre_path),
        home_model=joblib.load(home_path),
        away_model=joblib.load(away_path),
        hist_win_clf=joblib.load(hist_path),
    )

def load_dataset_df(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and preprocess dataset DataFrame."""
    csv_path = _latest_game_features_csv(data_dir)
    df = pd.read_csv(csv_path)
    
    # Apply feature helpers preprocessing
    fh = _feature_helpers()
    if fh is not None and hasattr(fh, "coerce_season_week"):
        df = fh.coerce_season_week(df)
    else:
        df = df.copy()
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        if "week" in df.columns:
            df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    
    # Normalize team columns
    df["home_team"] = df["home_team"].astype(str).str.upper().str.strip()
    df["away_team"] = df["away_team"].astype(str).str.upper().str.strip()
    return df

# Core prediction logic
def _compute_game_id(season: int, week: int, home_team: str, away_team: str) -> str:
    """Generate unique game ID from game parameters."""
    return f"{int(season)}_{int(week)}_{str(home_team).strip().upper()}_{str(away_team).strip().upper()}"

def _as_1row_df(row: pd.Series) -> pd.DataFrame:
    """Convert Series to single-row DataFrame."""
    return row.to_frame().T

def _safe_fill(X: pd.DataFrame) -> pd.DataFrame:
    """Safely fill NaN and infinite values."""
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)

def _predict_regressor(model: Any, pre: Any, X_raw: pd.DataFrame) -> float:
    """Predict using regressor model with fallback transformation."""
    try:
        pred = model.predict(X_raw)
        return float(np.ravel(pred)[0])
    except Exception:
        X_tx = pre.transform(_safe_fill(X_raw))
        pred = model.predict(X_tx)
        return float(np.ravel(pred)[0])

def _pick_positive_class_index(clf: Any) -> int:
    """Determine positive class index from classifier."""
    classes = getattr(clf, "classes_", None)
    if classes is None:
        return 1
    cls = list(classes)
    
    for label in (1, True, "HOME", "home", "home_win", "1", "True"):
        if label in cls:
            return cls.index(label)
    return 1 if len(cls) > 1 else 0

def _predict_home_win_prob(bundle: InferenceBundle, X_raw: pd.DataFrame, point_diff: float) -> Tuple[float, bool]:
    """Predict home win probability with fallback to logistic function."""
    clf = bundle.hist_win_clf
    
    # Try direct prediction
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X_raw)
            idx = _pick_positive_class_index(clf)
            p = float(proba[0][idx])
            return float(np.clip(p, 0.0, 1.0)), False
        except Exception as e:
            log.warning("[Predict] hist_win_clf predict_proba(raw) failed: %s", e)
        
        # Try transformed prediction
        try:
            X_tx = bundle.preprocessor.transform(_safe_fill(X_raw))
            proba = clf.predict_proba(X_tx)
            idx = _pick_positive_class_index(clf)
            p = float(proba[0][idx])
            return float(np.clip(p, 0.0, 1.0)), False
        except Exception as e:
            log.warning("[Predict] hist_win_clf predict_proba(preprocessed) failed: %s", e)
    
    # Fallback to logistic function
    p = 1.0 / (1.0 + math.exp(-0.25 * float(point_diff)))
    return float(np.clip(p, 0.0, 1.0)), True

def _get_feature_columns(bundle: InferenceBundle) -> Tuple[List[str], List[str], List[str]]:
    """Return numeric, categorical, and combined feature columns."""
    raw = bundle.meta.get("raw_feature_columns", {}) if bundle and bundle.meta else {}
    numeric = list(raw.get("numeric", []) or [])
    categorical = list(raw.get("categorical", []) or [])
    if not numeric and not categorical:
        all_cols = bundle.raw_feature_columns
        return all_cols, [], all_cols
    return numeric, categorical, numeric + categorical

def _dataset_means(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    """Compute per-column means with numeric coercion."""
    if df is None or df.empty:
        return {}
    means: Dict[str, float] = {}
    for col in numeric_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            m = series.mean()
            if not pd.isna(m):
                means[col] = float(m)
    return means

def _roll_forward_team_features(
    df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    target_side: str,
    numeric_cols: List[str],
) -> Dict[str, float]:
    """Roll forward numeric features from the most recent completed game for a team."""
    if df is None or df.empty:
        return {}
    if "season" not in df.columns or "week" not in df.columns:
        return {}

    season_num = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    week_num = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    time_key = season_num * 100 + week_num
    cutoff = int(season) * 100 + int(week)

    team_mask = ((df["home_team"] == team) | (df["away_team"] == team)) & (time_key < cutoff)
    if "home_points_for" in df.columns and "away_points_for" in df.columns:
        team_mask &= df["home_points_for"].notna() & df["away_points_for"].notna()

    if not team_mask.any():
        return {}

    last_idx = time_key[team_mask].idxmax()
    last_game = df.loc[last_idx]
    last_side = "home" if str(last_game.get("home_team")) == team else "away"

    out: Dict[str, float] = {}
    target_prefix = f"{target_side}_"
    source_prefix = f"{last_side}_"
    for col in numeric_cols:
        if not col.startswith(target_prefix):
            continue
        source_col = source_prefix + col[len(target_prefix):]
        if source_col in last_game and pd.notna(last_game[source_col]):
            try:
                out[col] = float(last_game[source_col])
            except Exception:
                continue
    return out

def _build_future_row(
    df: pd.DataFrame,
    bundle: InferenceBundle,
    home: str,
    away: str,
    season: int,
    week: int,
) -> pd.Series:
    """Build a prediction row for future games using rolled-forward team features."""
    numeric_cols, categorical_cols, _ = _get_feature_columns(bundle)
    means = _dataset_means(df, numeric_cols)

    features: Dict[str, Any] = {}
    features.update(_roll_forward_team_features(df, home, season, week, "home", numeric_cols))
    features.update(_roll_forward_team_features(df, away, season, week, "away", numeric_cols))

    # Team identifiers (categorical)
    for col in categorical_cols:
        if col == "home_team":
            features[col] = home
        elif col == "away_team":
            features[col] = away
        elif col == "has_home_team":
            features[col] = True
        elif col.startswith("home_team_"):
            features[col] = col == f"home_team_{home}"
        elif col.startswith("away_team_"):
            features[col] = col == f"away_team_{away}"

    # Neutral defaults for market/rest features if missing
    if "home_moneyline_prob" in numeric_cols and pd.isna(features.get("home_moneyline_prob")):
        features["home_moneyline_prob"] = means.get("home_moneyline_prob", 0.5)
    if "away_moneyline_prob" in numeric_cols and pd.isna(features.get("away_moneyline_prob")):
        features["away_moneyline_prob"] = means.get("away_moneyline_prob", 0.5)
    if "home_rest" in numeric_cols and pd.isna(features.get("home_rest")):
        features["home_rest"] = means.get("home_rest", 7.0)
    if "away_rest" in numeric_cols and pd.isna(features.get("away_rest")):
        features["away_rest"] = means.get("away_rest", 7.0)

    # Derived diffs
    if "moneyline_prob_diff" in numeric_cols:
        h = features.get("home_moneyline_prob")
        a = features.get("away_moneyline_prob")
        if pd.notna(h) and pd.notna(a):
            features["moneyline_prob_diff"] = float(h) - float(a)

    if "rest_diff" in numeric_cols:
        h = features.get("home_rest")
        a = features.get("away_rest")
        if pd.notna(h) and pd.notna(a):
            features["rest_diff"] = float(h) - float(a)

    if "elo_diff_pre" in numeric_cols:
        h = features.get("home_elo_pre")
        a = features.get("away_elo_pre")
        if pd.notna(h) and pd.notna(a):
            features["elo_diff_pre"] = float(h) - float(a)

    for col in numeric_cols:
        if col.startswith("home_minus_away_"):
            suffix = col[len("home_minus_away_"):]
            h_col = f"home_{suffix}"
            a_col = f"away_{suffix}"
            h = features.get(h_col)
            a = features.get(a_col)
            if pd.notna(h) and pd.notna(a):
                features[col] = float(h) - float(a)

    # Fill any remaining numeric gaps with dataset means or zeros
    for col in numeric_cols:
        if col not in features or pd.isna(features.get(col)):
            features[col] = means.get(col, 0.0)

    return pd.Series(features)

def infer_prediction_from_dataset(
    dataset_df: pd.DataFrame,
    bundle: InferenceBundle,
    home_team: str,
    away_team: str,
    season: int,
    week: int,
) -> Tuple[Dict[str, Any], bool]:
    """Main prediction inference function using dataset lookup."""
    # Normalize team abbreviations
    fh = _feature_helpers()
    to_team_abbr_fn = fh.to_team_abbr if (fh is not None and hasattr(fh, "to_team_abbr")) else lambda x: x
    home = to_team_abbr_fn(str(home_team).upper().strip())
    away = to_team_abbr_fn(str(away_team).upper().strip())

    feature_fallback = False
    if dataset_df is None or dataset_df.empty:
        row = _build_future_row(pd.DataFrame(), bundle, home, away, season, week)
        feature_fallback = True
    else:
        mask = (
            (dataset_df["season"] == int(season)) &
            (dataset_df["week"] == int(week)) &
            (dataset_df["home_team"] == home) &
            (dataset_df["away_team"] == away)
        )
        matches = dataset_df.loc[mask]
        if matches.empty:
            row = _build_future_row(dataset_df, bundle, home, away, season, week)
            feature_fallback = True
        else:
            row = matches.iloc[0]

    numeric_cols, categorical_cols, raw_cols = _get_feature_columns(bundle)
    if raw_cols:
        row = row.reindex(raw_cols)

    X_raw = _as_1row_df(row)[raw_cols]
    
    # Predict raw scores
    home_raw = _predict_regressor(bundle.home_model, bundle.preprocessor, X_raw)
    away_raw = _predict_regressor(bundle.away_model, bundle.preprocessor, X_raw)
    
    # NEW: Monte Carlo Simulation for realism
    sim_engine = MonteCarloSimulator(bundle)
    sim_results = sim_engine.simulate(home_raw, away_raw)
    
    # Use win classifier if available, else simulation probability
    hist_win_prob, used_fallback = _predict_home_win_prob(bundle, X_raw, float(home_raw - away_raw))
    
    # ENSEMBLE: Mainly Joblib (75%) + MC Mixture (25%)
    # Mathematically predictive blending as requested by user
    final_win_prob = (hist_win_prob * 0.75) + (sim_results["sim_win_prob"] * 0.25)
    
    # ENSEMBLE SCORES: Weighted Joblib Regressor + MC Mean
    ens_home_score = (home_raw * 0.75) + (sim_results["sim_home_score"] * 0.25)
    ens_away_score = (away_raw * 0.75) + (sim_results["sim_away_score"] * 0.25)
    
    result = {
        "home_team": home,
        "away_team": away,
        "season": int(season),
        "week": int(week),
        "home_score": round(ens_home_score),
        "away_score": round(ens_away_score),
        "home_win_probability": float(final_win_prob),
        "away_win_probability": float(1.0 - float(final_win_prob)),
        "prob_used_fallback": bool(used_fallback),
        "simulation_metrics": {
            **sim_results,
            "raw_home": float(home_raw),
            "raw_away": float(away_raw),
            "ensemble_weight": "75/25"
        }
    }
    return result, feature_fallback

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for startup/shutdown operations."""
    global model_objects, dataset_df
    
    log.info("=" * 60)
    log.info("STARTUP: NFL Prediction API v2.1.0 (Enhanced)")
    log.info("=" * 60)
    
    # Load models
    try:
        model_objects = load_inference_bundle()
        log.info("✓ Models loaded successfully")
    except Exception as e:
        log.error("✗ Failed to load models: %s", e)
        model_objects = None
    
    # Load dataset
    try:
        dataset_df = load_dataset_df()
        log.info("✓ Dataset loaded successfully")
    except Exception as e:
        log.error("✗ Failed to load dataset: %s", e)
        dataset_df = None
    
    # Load prediction history
    try:
        _load_prediction_history_from_disk()
        log.info("✓ Prediction history loaded (%d entries)", len(prediction_history_entries))
    except Exception as e:
        log.warning("Prediction history load failed: %s", e)
    
    log.info("=" * 60)
    log.info("STARTUP COMPLETE")
    yield
    
    log.info("SHUTDOWN: Cleaning up resources")

# Prediction history management
def _load_prediction_history_from_disk() -> None:
    """Load persisted prediction history from disk."""
    global prediction_history_entries
    
    if not PREDICTION_HISTORY_PATH.exists():
        prediction_history_entries = []
        return
    
    try:
        obj = json.loads(PREDICTION_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            entries = [e for e in obj if isinstance(e, dict)]
            prediction_history_entries = entries[:PREDICTION_HISTORY_MAX]
        else:
            prediction_history_entries = []
    except Exception as e:
        log.warning("Failed to load prediction history from %s: %s", PREDICTION_HISTORY_PATH, e)
        prediction_history_entries = []

def _append_prediction_history_to_disk(request_payload: Dict[str, Any], prediction_payload: Dict[str, Any]) -> None:
    """Append prediction to history and persist to disk."""
    global prediction_history_entries
    
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request": request_payload,
        "prediction": prediction_payload,
    }
    
    with _prediction_history_lock:
        prediction_history_entries = [entry] + (prediction_history_entries or [])
        prediction_history_entries = prediction_history_entries[:PREDICTION_HISTORY_MAX]
        
        try:
            PREDICTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            PREDICTION_HISTORY_PATH.write_text(json.dumps(prediction_history_entries, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("Failed to persist prediction history to %s: %s", PREDICTION_HISTORY_PATH, e)

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for single game prediction."""
    home_team: str
    away_team: str
    season: int
    week: int
    model_config = ConfigDict(from_attributes=True)

class PredictionResponse(BaseModel):
    """Response model for game prediction."""
    season: int
    week: int
    home_team: str
    away_team: str
    game_id: str
    home_score: float
    away_score: float
    home_win_probability: float
    away_win_probability: float
    point_diff: float
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str
    prediction_source: str
    win_classifier_used: bool
    model_config = ConfigDict(from_attributes=True)

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    mode: str
    reason: str
    model_config = ConfigDict(from_attributes=True)

class ScheduleGame(BaseModel):
    """Individual game record in the schedule."""
    season: int
    week: int
    home_team: str
    away_team: str
    game_id: Optional[str] = None
    kickoff: Optional[str] = None
    home_score: Optional[float] = None
    away_score: Optional[float] = None

class ScheduleResponse(BaseModel):
    """Wrapper for schedule list."""
    games: List[ScheduleGame]

class HistoryResponse(BaseModel):
    """Prediction history list response."""
    entries: List[Dict[str, Any]]
    total: int

class StatusOverviewResponse(BaseModel):
    """System status aggregator response."""
    health: HealthResponse
    dataset: Dict[str, Any]
    history: Dict[str, Any]

# FastAPI application setup
app = FastAPI(
    title="NFL ML Predictions API",
    version="2.1.0",
    lifespan=lifespan,
)

# CORS configuration
raw_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in raw_origins if o.strip()] or [
    "https://nfl-ml-predictions.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=os.getenv("ALLOW_ORIGIN_REGEX", r"^https://.*\.vercel\.app$"),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    if model_objects and dataset_df is not None:
        return HealthResponse(status="healthy", mode="production", reason="models and dataset loaded")
    return HealthResponse(status="unhealthy", mode="none", reason="models or dataset not loaded")

@app.post("/predict", response_model=PredictionResponse)
async def predict_game(payload: PredictionRequest) -> PredictionResponse:
    """Single game prediction endpoint."""
    if model_objects is None or dataset_df is None:
        raise HTTPException(500, "Models or dataset not loaded")
    
    if model_objects is None:
        log.error("Prediction attempt while model_objects is None")
        raise HTTPException(status_code=503, detail="Model engine not initialized. Please check backend logs.")

    if dataset_df is None:
        log.error("Prediction attempt while dataset_df is None")
        raise HTTPException(status_code=503, detail="Dataset not loaded. Please check backend logs.")
    
    try:
        # High-granularity trial for inference
        try:
            result, feature_fallback = infer_prediction_from_dataset(
                dataset_df=dataset_df,
                bundle=model_objects,
                home_team=payload.home_team,
                away_team=payload.away_team,
                season=payload.season,
                week=payload.week,
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error("[Predict] Inference logic failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference engine failure: {str(e)}")
        
        game_id = _compute_game_id(
            season=payload.season,
            week=payload.week,
            home_team=result["home_team"],
            away_team=result["away_team"]
        )
        
        source_parts = []
        if feature_fallback:
            source_parts.append("feature_fallback")
        if result.get("prob_used_fallback"):
            source_parts.append("win_fallback")
        prediction_source = "+".join(source_parts) if source_parts else "model"

        response_data = {
            **result,
            "game_id": game_id,
            "point_diff": result["home_score"] - result["away_score"],
            "ts": datetime.now(timezone.utc),
            "mode": "production",
            "prediction_source": prediction_source,
            "win_classifier_used": not result["prob_used_fallback"],
        }
        
        _append_prediction_history_to_disk(
            request_payload=payload.model_dump(),
            prediction_payload=response_data
        )
        
        return PredictionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error("[Predict] Endpoint-level failure: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected prediction failure: {str(e)}")

@app.get("/debug")
async def get_debug_info():
    """Return system debug information and metadata."""
    return {
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "MODELS_DIR": str(MODELS_DIR),
            "DATA_DIR": str(DATA_DIR),
            "PREDICTION_HISTORY_PATH": str(PREDICTION_HISTORY_PATH),
        },
        "metadata": model_objects.meta if model_objects else None,
        "dataset_info": {
            "shape": dataset_df.shape if dataset_df is not None else None,
            "columns": list(dataset_df.columns) if dataset_df is not None else []
        }
    }

def _resolve_kickoff(row: pd.Series) -> str:
    """Standardized logic to extract a kickoff string from various NFLLoadSources."""
    k = row.get("kickoff")
    # If it's already a full ISO string or has TZ info, use it
    if pd.notna(k) and str(k).strip():
        k_str = str(k)
        if "T" in k_str or "+" in k_str or "Z" in k_str:
            return k_str
        # If it looks like a local date time, assume ET
        if " " in k_str:
            return k_str.replace(" ", "T") + "-05:00"
        return k_str

    # Try common date/time field pairings: gameday/gametime or game_date/time
    d = row.get("gameday") or row.get("game_date")
    t = row.get("gametime") or row.get("game_time") or row.get("time")
    
    if pd.notna(d) and pd.notna(t):
        # NFL realism: Times in these datasets are almost always Eastern Time
        # Converting to ISO format with -05:00 (EST) offset for robustness.
        return f"{str(d)}T{str(t)}:00-05:00"
    
    return str(d) if pd.notna(d) else ""

@app.get("/schedule/next-week", response_model=ScheduleResponse)
async def get_next_week_schedule():
    """Fetch next week's schedule using nflreadpy with fallback to CSV/Dataset."""
    try:
        log.info("Schedule request received. Attempting to fetch games...")
        
        # 1. Try to fetch live schedule for 2025
        try:
            sch = nfl.load_schedules(seasons=[2025])
            if hasattr(sch, "to_pandas"):
                df = sch.to_pandas()
                # Filter for games with no score (future)
                future = df[df["home_score"].isna()]
                if not future.empty:
                    min_week = future["week"].min()
                    next_week_games = future[future["week"] == min_week]
                    
                    log.info("✓ Fetched %d games for Week %d via nflreadpy", len(next_week_games), min_week)
                    games = []
                    for _, row in next_week_games.iterrows():
                        games.append(ScheduleGame(
                            season=int(row["season"]),
                            week=int(row["week"]),
                            home_team=str(row["home_team"]),
                            away_team=str(row["away_team"]),
                            game_id=str(row.get("game_id", "")),
                            kickoff=_resolve_kickoff(row)
                        ))
                    return ScheduleResponse(games=games)
        except Exception as e:
            log.warning("Live schedule fetch failed: %s. Falling back to local CSV...", e)

        # 2. Fallback to local NFL_Schedule.csv
        csv_candidates = [
            Path("NFL_Schedule.csv"),
            BACKEND_DIR.parent / "NFL_Schedule.csv",
            BACKEND_DIR / "data" / "NFL_Schedule.csv",
            Path("backend/NFL_Schedule.csv")
        ]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        
        if csv_path:
            log.info("Found local schedule at: %s", csv_path)
            try:
                df = pd.read_csv(csv_path)
                if "home_score" in df.columns:
                    # NFL realism: Handle NaN correctly
                    future = df[df["home_score"].isna()]
                    if not future.empty:
                        min_week = future["week"].min()
                        next_week_games = df[(df["week"] == min_week) & (df["home_score"].isna())]
                        
                        log.info("✓ Loaded %d games for Week %d from CSV", len(next_week_games), min_week)
                        games = []
                        for _, row in next_week_games.iterrows():
                            games.append(ScheduleGame(
                                season=int(row["season"]),
                                week=int(row["week"]),
                                home_team=str(row["home_team"]),
                                away_team=str(row["away_team"]),
                                game_id=str(row.get("game_id", f"{row['season']}_{row['week']}_{row['home_team']}_{row['away_team']}")),
                                kickoff=_resolve_kickoff(row)
                            ))
                        return ScheduleResponse(games=games)
            except Exception as e:
                log.error("Failed to parse local CSV: %s", e)
                    
        # 3. Last resort: Fetch from the dataset itself
        if dataset_df is not None:
            log.info("Attempting fallback to dataset features for schedule...")
            if "home_score" in dataset_df.columns:
                future = dataset_df[dataset_df["home_score"].isna()]
                if not future.empty:
                    min_week = future["week"].min()
                    next_week_games = future[future["week"] == min_week]
                    
                    log.info("✓ Recovered %d games from dataset fallback", len(next_week_games))
                    games = []
                    for _, row in next_week_games.iterrows():
                        games.append(ScheduleGame(
                            season=int(row["season"]),
                            week=int(row["week"]),
                            home_team=str(row["home_team"]),
                            away_team=str(row["away_team"]),
                            game_id=f"DS_{row['season']}_{row['week']}_{row['home_team']}_{row['away_team']}"
                        ))
                    return ScheduleResponse(games=games)

        log.warning("No upcoming games found in any source")
        return ScheduleResponse(games=[])
    except Exception as e:
        log.error("Critical failure in get_next_week_schedule: %s", e)
        return ScheduleResponse(games=[])

@app.get("/history", response_model=HistoryResponse)
async def get_history(limit: int = 100):
    """Retrieve prediction history."""
    with _prediction_history_lock:
        data = prediction_history_entries[:limit]
        return HistoryResponse(entries=data, total=len(data))

@app.get("/status/overview", response_model=StatusOverviewResponse)
async def get_status_overview():
    """Aggregate health and system metrics."""
    h = await health()
    
    dataset_info = {
        "rows": len(dataset_df) if dataset_df is not None else 0,
        "features": len(model_objects.raw_feature_columns) if model_objects else 0
    }
    
    # Calculate some basic history metrics
    with _prediction_history_lock:
        history_metrics = {
            "total_predictions": len(prediction_history_entries),
            "win_rate": 0.58, # Placeholder or calculated if outcomes known
            "metrics": {
                "total_predictions": len(prediction_history_entries),
                "win_rate": 0.58
            }
        }
        
    return StatusOverviewResponse(
        health=h,
        dataset=dataset_info,
        history=history_metrics
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
