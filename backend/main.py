"""
NFL Game Prediction API — FastAPI Entrypoint

Architectural Role:
    - Main backend entrypoint for NFL game predictions and reporting.
    - Loads ML models, engineered datasets, and serves as the API gateway.
    - Integrates with FastAPI, pandas, joblib, and environment configuration (.env).

Key Endpoints:
    - /health: Service and model status.
    - /debug: Metadata and config diagnostics.
    - /report/training: Model training report.
    - /report/calibration: Classifier calibration metrics.
    - /schedule/next-week: Upcoming NFL games (from schedule CSV).
    - /predict: Predict scores and win probabilities for a given matchup.
    - /predict/next-week: Batch predictions for next scheduled week.

Dependencies:
    - DATASET_PATH, SCHEDULE_PATH,ALLOWED_ORIGINS, ALLOWED_ORIGINS, SERVE_FRONTEND (from .env)
    - Models and metadata in backend/models/
    - Engineered features in backend/data/game_features.csv

    Run:
        uvicorn backend.main:app --reload --port 8000

    Maintainer Notes:
        - All endpoints return JSON; errors use HTTPException.
        - Models are loaded once at startup for performance.
        - Frontend static files can be served if configured.

---------------------------------------------------------------------
    # File: backend/main.py
    # Purpose: FastAPI backend for NFL game predictions, serving ML models and API endpoints.
    
    # Functions: 
    # load_objects, _validate_dataset_schema,_validate_features_present,  
    #  _sanity_predict,_coerce_bool, _ensure_home_away, get_current_nfl_context,                  
    #  _build_future_row, _normalize_feature_cols, health, debug_info, 
    #  report_training, report_calibration, build_game_mask,
    #  get_next_week_schedule, predict_game, predict_next_week
    
    # Variables: 
    # model_objects, dataset_df, DEFAULT_ALLOWALLOWED_ORIGINS,ALLOWED_ORIGINS, ALLOWED_ORIGINS, TEAM_ABBREVIATIONS, TEAM_CODE_FIX, VALID_ABBRS, THIS_FILE, BACKEND_DIR, BASE_DIR, DATA_DIR, MODELS_DIR, LOG_DIR, DEFAULT_DATASET, DEFAULT_SCHEDULE, FRONTEND_DIR, FRONTEND_BUILD, FRONTEND_DIST, TRUTHY, SERVE_FRONTEND
    
    # Interacts With: backend/models/ (joblib models), backend/data/ (CSV datasets), frontend/ (static files if served), .env (config)
"""


from __future__ import annotations

import json
import logging
import logging.config
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env
# Load .env from backend directory or repository root for consistent environment variable loading
backend_dir = Path(__file__).parent
repo_root = backend_dir.parent
dotenv_loaded = load_dotenv(backend_dir / ".env")
if not dotenv_loaded:
    load_dotenv(repo_root / ".env")

# -----------------------
# Paths and constants
# -----------------------
THIS_FILE = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
BASE_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
LOG_DIR = BACKEND_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Use game_features.csv which has the engineered features (prior stats, differentials, betting data)
# merged_game_features.csv only has raw stats and won't work with trained models
# ---------------------------------------------------------------
DEFAULT_DATASET = DATA_DIR / "game_features.csv"
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
        "formatters": {
            "d": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "d",
            },
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

# backend/main.py
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import os, re

def _origins_from_env():
    raw = os.getenv("ALLOWED_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]

ALLOWED_ORIGINS = _origins_from_env() or [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://nfl-ml-predictions.vercel.app",
    "https://nfl-ml-predictions-pr5uahmqx-christopher-jordons-projects.vercel.app",
    "https://nfl-predict-6fghcp7sx-christopher-jordons-projects.vercel.app",
    "https://new-nfl-predict.vercel.app",
    "https://www.nfl-predict.vercel.app",
]

# ⚠️ Add ANY custom middlewares BEFORE this line (auth/logging/sentry/etc)


log.info("ADD_LOCALHOST_ORIGINS enabled: appended local dev origins to ALLOWED_ORIGINS")

log.debug("ALLOWED_ORIGINS=%s", ALLOWED_ORIGINS)

# Teams
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
def _normalize_feature_cols(cols: Dict[str, List[str]]) -> List[str]:
    """Normalize feature columns from metadata dict to flat list."""
    return cols.get("numeric", []) + cols.get("categorical", [])


def get_abbr(name: str) -> str:
    """
    Normalize a team name or abbreviation to its canonical 2- or 3-letter code.

    Logic:
        - Accepts full team names, legacy codes, or abbreviations.
        - Applies TEAM_CODE_FIX for legacy/relocated teams.
        - Maps official names via TEAM_ABBREVIATIONS.
        - Raises ValueError if input is not recognized.

    Args:
        name (str): Team name, abbreviation, or legacy code.

    Returns:
        str: Canonical team abbreviation.

    Raises:
        ValueError: If the team name/code is not recognized.
    """
    n = str(name).strip()
    if n in VALID_ABBRS:
        return TEAM_CODE_FIX.get(n, n)
    if n in TEAM_CODE_FIX:
        return TEAM_CODE_FIX[n]
    if n in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[n]
    raise ValueError(f"Unknown team: {name}")


# -----------------------
def load_objects() -> Dict[str, Any]:
    """
    Load model metadata and instantiate reusable predictors for the API.

    Loads the following models from disk:
        - preprocessor: Feature engineering pipeline (joblib)
        - home_model: Home team score regressor (joblib or dict with 'hgbr', 'ridge', 'weight')
        - away_model: Away team score regressor (joblib or dict with 'hgbr', 'ridge', 'weight')
        - win_model: Calibrated win probability classifier (joblib, optional)

    Returns:
        dict with keys:
            - mode: str, operational mode (e.g., "production")
            - preprocessor: sklearn pipeline or transformer
            - home_model: regressor or ensemble dict
            - away_model: regressor or ensemble dict
            - win_model: classifier or None
            - raw_feature_columns: dict of feature columns
            - win_threshold_optimal: float, optimal win threshold
    """
    meta_path = MODELS_DIR / "metadata.json"
    log.debug("Loading model metadata from %s", meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def resolve_model_path(meta_key: str, fallback: str) -> Path:
        candidate = Path(meta.get(meta_key, fallback))
        return candidate if candidate.is_absolute() else MODELS_DIR / candidate

    preprocessor = joblib.load(resolve_model_path("preprocessor", "preprocessor.joblib"))
    home_model = joblib.load(resolve_model_path("home_model", "home_model.joblib"))
    away_model = joblib.load(resolve_model_path("away_model", "away_model.joblib"))
    win_model_path = resolve_model_path("win_model", "win_clf_calibrated.joblib")
    win_model = joblib.load(win_model_path) if win_model_path.exists() else None

    return {
        "mode": meta.get("mode", "production"),
        "preprocessor": preprocessor,
        "home_model": home_model,
        "away_model": away_model,
        "win_model": win_model,
        "raw_feature_columns": meta.get("raw_feature_columns", {}),
        "win_threshold_optimal": meta.get("win_threshold_optimal", 0.5),
    }


def _validate_dataset_schema(df: pd.DataFrame, model_objects: Dict[str, Any]) -> None:
    """Fail-fast check that dataset contains required engineered features.

    Reads expected feature names from model_objects['raw_feature_columns'] and
    ensures those columns exist in the dataframe. Raises RuntimeError with
    actionable message if mismatch detected.
    """
    expected = _normalize_feature_cols(model_objects.get("raw_feature_columns", {}))
    missing = [c for c in expected if c not in df.columns]
    if missing:
        # Log a concise message and raise to prevent serving incompatible data
        log.error("Dataset schema mismatch: %d missing engineered features. Sample: %s", len(missing), missing[:10])
        raise RuntimeError(
            f"Dataset missing engineered features required by models: {missing[:20]}. "
            "Run the feature engineering pipeline or point DATASET_PATH to the correct file."
        )


def _sanity_predict(model_objects: Dict[str, Any], df: pd.DataFrame) -> None:
    """
    Perform a tiny sanity prediction to exercise model deserialization and pipeline
    behavior at startup. This builds a minimal synthetic feature row using the first
    row in `df` (or constructs defaults) and runs home/away predictions and the
    win probability model if present.

    This function logs warnings but does not raise exceptions; failures are
    surfaced in logs to avoid bringing down the service after a non-fatal
    prediction error. The primary purpose is to detect deserialization errors
    or unexpected model interface changes early.
    """
    # Build a representative feature row: prefer a real engineered row if available
    failures: List[str] = []
    if not df.empty:
        sample = df.iloc[0]
        # Ensure the sample contains numeric features expected by preprocessor
        features = {}
        raw_cols = model_objects.get("raw_feature_columns", {})
        # raw_cols may be a dict with numeric/categorical lists
        cols = []
        if isinstance(raw_cols, dict):
            cols = raw_cols.get("numeric", []) + raw_cols.get("categorical", [])
        elif isinstance(raw_cols, list):
            cols = raw_cols
        for c in cols:
            features[c] = sample.get(c, 0)
    else:
        features = {c: 0 for c in model_objects.get("raw_feature_columns", {}).get("numeric", [])}

    x = pd.DataFrame([features])

    pre = model_objects.get("preprocessor")
    home_m = model_objects.get("home_model")
    away_m = model_objects.get("away_model")
    win_m = model_objects.get("win_model")

    # Attempt to transform and predict; collect failures and raise at end to fail-fast
    transformed = None
    if pre is not None:
        try:
            # Check if preprocessor is fitted before trying to transform
            if hasattr(pre, '_is_fitted') and pre._is_fitted:
                transformed = pre.transform(x)
            else:
                log.warning("Preprocessor not fitted, skipping transform in sanity check")
        except Exception as e:
            failures.append(f"preprocessor.transform failed: {type(e).__name__}: {e}")
            log.debug("Sanity predict: preprocessor.transform failed during startup check", exc_info=True)

    def try_predict(m, label: str):
        try:
            inp = x if transformed is None else transformed
            if hasattr(m, "predict"):
                _ = m.predict(inp)
            else:
                failures.append(f"{label} missing predict method")
        except Exception as e:
            failures.append(f"{label} predict failed: {type(e).__name__}: {e}")
            log.debug("Sanity predict: %s predict failed", label, exc_info=True)

    if home_m is not None:
        try_predict(home_m, "home_model")
    else:
        failures.append("home_model not present")
    if away_m is not None:
        try_predict(away_m, "away_model")
    else:
        failures.append("away_model not present")
    if win_m is not None and hasattr(win_m, "predict_proba"):
        try:
            inp = x if transformed is None else transformed
            _ = win_m.predict_proba(inp)
        except Exception as e:
            failures.append(f"win_model.predict_proba failed: {type(e).__name__}: {e}")
            log.debug("Sanity predict: win_model.predict_proba failed", exc_info=True)

    if failures:
        # Raise RuntimeError to make startup fail-fast when sanity check not satisfied
        msg = "; ".join(failures[:10])
        log.error("Startup sanity-predict failed: %s", msg)
        raise RuntimeError(f"Startup sanity-predict failed: {msg}")


def _coerce_bool(s: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series to boolean values for dataset ingestion.

    Intended Use:
        - Converts various representations of boolean-like values (e.g., "True", "yes", "1", etc.) to actual bools.
        - Handles edge cases where the input Series is not of boolean dtype (e.g., object, string, int).
        - Used for standardizing 'is_home', 'is_away', or similar columns in NFL datasets.

    Args:
        s (pd.Series): Input Series with possible boolean or boolean-like values.

    Returns:
        pd.Series: Series of bools suitable for downstream feature engineering.
    """
    truthy = {"true", "t", "1", "yes", "y"}
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    return s.astype(str).str.strip().str.lower().isin(truthy)


def _ensure_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame contains 'home_team' and 'away_team' columns.

    Logic:
        - If 'home_team' and 'away_team' exist, return as-is.
        - If 'team', 'opponent_team', and 'is_home' exist, derive home/away columns.
        - If neither set is present, log a warning and return the DataFrame unchanged.
          This means downstream features relying on home/away context may be unavailable,
          and only synthetic/statistical features can be used for predictions.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with ensured 'home_team' and 'away_team' columns if possible.
    """
    cols = set(df.columns)
    if {"home_team", "away_team"}.issubset(cols):
        # Already has required columns
        return df
    if {"team", "opponent_team", "is_home"}.issubset(cols):
        # Derive home/away columns from fallback structure
        is_home = _coerce_bool(df["is_home"])
        return df.assign(
            is_home=is_home,
            home_team=np.where(is_home, df["team"], df["opponent_team"]),
            away_team=np.where(is_home, df["opponent_team"], df["team"]),
        )
    # Missing both canonical and fallback columns; log and return unchanged
    log.warning(
        "Dataset missing home/away columns and team/opponent fallback; synthetic features only. "
        "Predictions may be limited or inaccurate due to lack of home/away context."
    )
    return df


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model_objects, dataset_df
    log.info("Startup: loading models and dataset")
    model_objects = load_objects()

    ds_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    if not ds_path.exists():
        raise RuntimeError(f"Dataset not found: {ds_path}")

    df = pd.read_csv(ds_path)
    if df.empty:
        raise RuntimeError("Dataset CSV is empty")

    df.columns = [c.strip() for c in df.columns]
    df = _ensure_home_away(df)
    # Validate dataset schema against metadata to fail fast if features mismatch
    try:
        _validate_dataset_schema(df, model_objects)
    except RuntimeError:
        # Re-raise so startup fails visibly
        raise
    dataset_df = df
    # Run a lightweight startup sanity check to exercise model deserialization and pipeline integrity
    _sanity_predict(model_objects, df)

    log.info("Loaded dataset rows=%d cols=%d", len(df), df.shape[1])
    try:
        yield
    finally:
        log.info("Shutdown complete")


# -----------------------
# FastAPI app + CORS + static
# -----------------------
app = FastAPI(title="NFL Game Prediction API", version="2.1.0", lifespan=lifespan)

# If the regex is an empty string / None, pass None to the middleware so that
# only explicit origins (or '*' in the list) are used.
_allow_origin_regex = r"https://.*\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=_allow_origin_regex,
    allow_methods=["*"],
    allow_headers=["*"],
)

if SERVE_FRONTEND:
    for candidate in (FRONTEND_BUILD, FRONTEND_DIST):
        if candidate.exists():
            app.mount(
                "/", StaticFiles(directory=str(candidate), html=True), name="frontend"
            )
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
    point_diff: float
    mode: str


class HealthResponse(BaseModel):
    """
    HealthResponse describes the API health payload.

    Fields:
        status: 'healthy' or 'unhealthy'
        mode: operational mode from loaded models (e.g. 'production') or 'none'
        reason: human-readable reason for current health
    """
    status: str
    mode: str
    reason: str


class ScheduleGame(BaseModel):
    """
    Represents a scheduled NFL game, including basic details and optional ML predictions.
    
    Used in the /schedule/next-week endpoint to return game info with injected predictions.
    """
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


def get_current_nfl_context() -> Dict[str, Any]:
    """
    Determine the current NFL season context for prediction and reporting.

    Logic:
        - Uses the current date to infer the active NFL season (season starts in August).
        - If the dataset is loaded and contains completed games, finds the last completed week and season.
        - Calculates the next prediction week and season, rolling over to the next season if week > 22.
        - Handles edge cases:
            * If no completed games are found, assumes preseason or early season.
            * If the next prediction season matches the current season, status is 'nfl_season_active'.
            * Otherwise, status is 'offseason'.
        - Returns a dictionary with current season, last completed season/week, next prediction season/week, and status.

    Edge Cases:
        - If dataset is missing or lacks required columns, defaults to preseason context.
        - If all games are completed, rolls over to next season week 1.
    """
    now = datetime.now()
    cur_season = now.year if now.month >= 8 else now.year - 1
    if dataset_df is not None and {
        "season",
        "week",
        "home_points_for",
        "away_points_for",
    }.issubset(dataset_df.columns):
        done = dataset_df[
            dataset_df["home_points_for"].notna()
            & dataset_df["away_points_for"].notna()
        ]
        if not done.empty:
            last = done.sort_values(by=["season", "week"]).iloc[-1]
            last_s, last_w = int(last["season"]), int(last["week"])
            nxt_s, nxt_w = last_s, last_w + 1
            if nxt_w > 22:
                nxt_s, nxt_w = last_s + 1, 1
            return {
                "current_season": cur_season,
                "last_completed_season": last_s,
                "last_completed_week": last_w,
                "next_prediction_season": nxt_s,
                "next_prediction_week": nxt_w,
                "status": "nfl_season_active" if nxt_s == cur_season else "offseason",
            }
    return {
        "current_season": cur_season,
        "last_completed_season": cur_season,
        "last_completed_week": 0,
        "next_prediction_season": cur_season,
        "next_prediction_week": 1,
        "status": "preseason_or_early",
    }


def _validate_features_present(feature_names: List[str], row: pd.Series) -> List[str]:
    """
    Quick helper used during development to find which expected features are missing
    from a candidate row (either from dataset or dynamically built).

    Returns a list of missing feature names (empty if none).
    """
    missing = [c for c in feature_names if c not in row.index or pd.isna(row.get(c))]
    return missing


def _build_future_row(
    df: pd.DataFrame, home: str, away: str, season: int, week: int
) -> pd.Series:
    """
    Build engineered features for a future game using historical data.

    Assumptions:
        - Input DataFrame `df` must contain columns for 'season', 'week', 'home_team', 'away_team', 'home_points_for', 'away_points_for', and engineered prior stats.
        - Teams `home` and `away` are canonical abbreviations matching those in the dataset.
        - Returns a pandas Series with all required model features for prediction, including rolling averages, advanced stats, and neutral betting/rest features.

    Returns:
        pd.Series: Feature vector for the specified future matchup, suitable for model input.
    """
    local = df.copy()
    local["time_key"] = local["season"].astype(int) * 100 + local["week"].astype(int)
    cutoff = season * 100 + week

    def compute_team_features(team: str, prefix: str) -> Dict[str, Any]:
        """Compute prior features for a team using their last N completed games."""
        # Find all games where this team played
        team_mask = (local["home_team"] == team) | (local["away_team"] == team)
        # Only use completed games before the target game
        completed_mask = (
            local["home_points_for"].notna() & 
            local["away_points_for"].notna() & 
            (local["time_key"] < cutoff)
        )
        history = local[team_mask & completed_mask].sort_values("time_key")
        
        if history.empty:
            raise ValueError(f"No prior data for {team} before {season} Week {week}")
        
        features = {}
        
        # Get last 5 games for 5-game averages, last 3 for 3-game averages
        last_5 = history.tail(5)
        last_3 = history.tail(3)
        
        # Helper to extract team's stats from a game row
        def get_team_stats(row, team_abbr):
            is_home = row["home_team"] == team_abbr
            if is_home:
                return {
                    "pf": row.get("home_points_for", np.nan),
                    "pa": row.get("away_points_for", np.nan),
                    "win": 1 if row.get("winner") == team_abbr else 0,
                }
            else:
                return {
                    "pf": row.get("away_points_for", np.nan),
                    "pa": row.get("home_points_for", np.nan),
                    "win": 1 if row.get("winner") == team_abbr else 0,
                }
        
        # Compute 3-game averages
        if len(last_3) >= 1:
            stats_3 = [get_team_stats(row, team) for _, row in last_3.iterrows()]
            features[f"{prefix}prior_pf_avg_3"] = np.mean([s["pf"] for s in stats_3 if not pd.isna(s["pf"])])
            features[f"{prefix}prior_pa_avg_3"] = np.mean([s["pa"] for s in stats_3 if not pd.isna(s["pa"])])
            features[f"{prefix}prior_win_pct_3"] = np.mean([s["win"] for s in stats_3])
        
        # Compute 5-game averages
        if len(last_5) >= 1:
            stats_5 = [get_team_stats(row, team) for _, row in last_5.iterrows()]
            features[f"{prefix}prior_pf_avg_5"] = np.mean([s["pf"] for s in stats_5 if not pd.isna(s["pf"])])
            features[f"{prefix}prior_pa_avg_5"] = np.mean([s["pa"] for s in stats_5 if not pd.isna(s["pa"])])
            features[f"{prefix}prior_win_pct_5"] = np.mean([s["win"] for s in stats_5])
        
        # For advanced stats, try to use the most recent values from the dataset
        # (these are pre-computed in game_features.csv)
        last_game = history.iloc[-1]
        was_home_last = last_game["home_team"] == team
        source_prefix = "home_" if was_home_last else "away_"
        
        # Copy advanced prior stats from last game
        for stat_name in [
            "off_epa_per_play", "off_success_rate", "off_explosive_rate",
            "off_third_down_pct", "off_pass_over_expected",
            "def_success_rate_allowed", "def_explosive_rate_allowed",
            "def_epa_per_play", "def_takeaway_rate", "off_turnover_rate"
        ]:
            for window in ["3", "5"]:
                col_name = f"{source_prefix}prior_{stat_name}_{window}"
                if col_name in last_game.index and pd.notna(last_game[col_name]):
                    features[f"{prefix}prior_{stat_name}_{window}"] = last_game[col_name]
        
        return features
    
    # Get features for both teams
    home_features = compute_team_features(home, "home_")
    away_features = compute_team_features(away, "away_")
    
    # Merge all features
    feature_row = {**home_features, **away_features}
    
    # Compute differential features (home - away)
    for stat_suffix in [
        "pf_avg_3", "pa_avg_3", "win_pct_3",
        "off_epa_per_play_3", "off_success_rate_3", "off_explosive_rate_3",
        "off_third_down_pct_3", "off_pass_over_expected_3",
        "def_success_rate_allowed_3", "def_explosive_rate_allowed_3",
        "def_epa_per_play_3", "def_takeaway_rate_3", "off_turnover_rate_3",
        "pf_avg_5", "pa_avg_5", "win_pct_5",
        "off_epa_per_play_5", "off_success_rate_5", "off_explosive_rate_5",
        "off_third_down_pct_5", "off_pass_over_expected_5",
        "def_success_rate_allowed_5", "def_explosive_rate_allowed_5",
        "def_epa_per_play_5", "def_takeaway_rate_5", "off_turnover_rate_5",
    ]:
        home_key = f"home_prior_{stat_suffix}"
        away_key = f"away_prior_{stat_suffix}"
        if home_key in feature_row and away_key in feature_row:
            h_val = feature_row[home_key]
            a_val = feature_row[away_key]
            if not pd.isna(h_val) and not pd.isna(a_val):
                feature_row[f"home_minus_away_{stat_suffix}"] = h_val - a_val
    
    # Add betting/rest features with neutral defaults
    feature_row["home_moneyline_prob"] = 0.5  # Neutral betting line
    feature_row["away_moneyline_prob"] = 0.5
    feature_row["moneyline_prob_diff"] = 0.0
    feature_row["spread_line"] = 0.0  # Pick'em
    feature_row["total_line"] = 45.0  # Average NFL total
    feature_row["home_rest"] = 7  # Standard week rest
    feature_row["away_rest"] = 7
    feature_row["rest_diff"] = 0
    feature_row["home_game_date"] = f"{season}-W{week:02d}"  # Categorical feature
    
    log.debug("Built future row for %s vs %s: %d features", home, away, len(feature_row))
    return pd.Series(feature_row)


# -----------------------
# Routes
# -----------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health endpoint: returns service status and model mode.

    Defensive access used because `model_objects` is a dict loaded at startup.
    Returns healthy when models are present and a sensible default for mode.
    """
    if model_objects:
        # model_objects is a dict - prefer .get for safe access
        if isinstance(model_objects, dict):
            mode = model_objects.get("mode", "production")
        else:
            # fallback to attribute access for backward compatibility
            mode = getattr(model_objects, "mode", "production")
        return HealthResponse(status="healthy", mode=mode, reason="models loaded")
    # Not ready yet
    return HealthResponse(status="unhealthy", mode="none", reason="models not loaded")




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
        tr = MODELS_DIR / "training_report.json"
        out["training_report_present"] = tr.is_file()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@app.get("/report/training")
def report_training() -> Dict[str, Any]:
    tr = MODELS_DIR / "training_report.json"
    if not tr.exists():
        raise HTTPException(404, "training_report.json not found")
    return json.loads(tr.read_text(encoding="utf-8"))


@app.get("/report/calibration")
def report_calibration() -> Dict[str, Any]:
    tr = MODELS_DIR / "training_report.json"
    if not tr.exists():
        raise HTTPException(404, "training_report.json not found")
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


def build_game_mask(df: pd.DataFrame, season: int, week: int, home_abbr: str, away_abbr: str) -> pd.Series:
    """
    Helper to build a boolean mask for selecting a specific game from the dataset.
    """
    mask = (
        (df["season"] == season)
        & (df["week"] == week)
        & (df["home_team"] == home_abbr)
        & (df["away_team"] == away_abbr)
    )
    if "is_home" in df.columns:
        mask &= df["is_home"].astype(bool)
    return mask

@app.get("/schedule/next-week", response_model=List[ScheduleGame])
def get_next_week_schedule() -> List[ScheduleGame]:
    """
    Retrieve the list of scheduled NFL games for the upcoming week.
    
    This endpoint filters the schedule CSV based on current NFL context (season/week),
    normalizes team abbreviations, and formats kickoff times. It supports frontend
    rendering of matchups and prediction requests. Depends on: get_current_nfl_context(),
    SCHEDULE_PATH env var, and team_abbr_map.json for normalization.
    """
    spath = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
    if not spath.exists():
        raise HTTPException(status_code=404, detail=f"Schedule not found: {spath}")
    df = pd.read_csv(spath)

    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace(TEAM_CODE_FIX)

    kickoff = pd.to_datetime(
        (
            df["gameday"].astype(str).str.strip()
            + " "
            + df["gametime"].astype(str).str.strip()
        ),
        errors="coerce",
        utc=True,
    )
    date_only = pd.to_datetime(df["gameday"], errors="coerce", utc=True)
    df["kickoff_ts_utc"] = kickoff.where(kickoff.notna(), date_only)

    now = pd.Timestamp.now(tz="UTC")
    future = df[df["kickoff_ts_utc"].notna() & (df["kickoff_ts_utc"] >= now)]
    current_week = (
        int(future["week"].min()) if not future.empty else int(df["week"].max())
    )
    week_games = df[df["week"] == current_week].copy()

    games: List[ScheduleGame] = []
    for _, r in week_games.iterrows():
        h = get_abbr(r["home_team"])
        a = get_abbr(r["away_team"])
        games.append(
            ScheduleGame(
                season=int(r["season"]),
                week=int(r["week"]),
                home_team=h,
                home_abbr=h,
                away_team=a,
                away_abbr=a,
                # Add predicted scores and win probabilities
                predicted_home_score=None,
                predicted_away_score=None,
                home_win_probability=None,
                away_win_probability=None,
                kickoff=r["kickoff_ts_utc"],
            )
        )
    log.info("Schedule week %s games=%d", current_week, len(games))
    return games


@app.post("/predict", response_model=PredictionResponse)
def predict_game(payload: PredictionRequest) -> PredictionResponse:
    """
    Predict endpoint: Accepts home/away teams, season, and week; 
    extracts features from dataset or builds them for future games, 
    runs ML models for score and win probability, and handles errors 
    for missing data, completed games, or prediction failures.
    """
    if model_objects is None or dataset_df is None:
        raise HTTPException(500, "Models or dataset not loaded. Please ensure the backend is properly initialized.")

    try:
        h = get_abbr(payload.home_team)
        a = get_abbr(payload.away_team)
        log.debug(f"predict_game: home_team={h}, away_team={a}")
        season, week = int(payload.season), int(payload.week)
        mask = build_game_mask(dataset_df, season, week, h, a)
        rows = dataset_df.loc[mask]
        
        # Try to get existing row from dataset
        if not rows.empty:
            row = rows.iloc[0]
            # Check if game is already completed
            if pd.notna(row.get("home_points_for")) and pd.notna(row.get("away_points_for")):
                logging.info(
                    "Game completed; no prediction needed: %s vs %s (%d Week %d)", h, a, season, week)
        else:
            # Game not in dataset - build features dynamically for future game
            log.info("Building features for future game: %s vs %s (%d Week %d)", h, a, season, week)
            try:
                row = _build_future_row(dataset_df, h, a, season, week)
            except ValueError as e:
                raise HTTPException(
                    400,
                    f"Cannot predict {h} vs {a} ({season} Week {week}): {e}"
                )
        
       # --- Build feature vector using the TRAINING CONTRACT (metadata.raw_feature_columns) ---
        raw_cols = model_objects.get("raw_feature_columns", {})
        exp_num = list(raw_cols.get("numeric", []))
        exp_cat = list(raw_cols.get("categorical", []))
        exp_all = exp_num + exp_cat

        # Safety: we'll source from the found row (dataset or _build_future_row),
        # but we also guarantee required categoricals exist.
        def _get_or_default(col: str):
            # Prefer the dynamic/lookup row value when available
            if col in row.index:
                v = row[col]
                # Keep NaN for numeric; imputer will handle it
                return v if (not pd.isna(v)) else np.nan

            # Provide smart defaults for known categoricals the model expects
            if col == "home_team":
                return h
            if col == "away_team":
                return a
            if col == "home_game_date":
                # Keep a stable categorical token even when we don't have a true date
                return f"{season}-W{week:02d}"

            # Rest features: okay to be NaN; imputer (median) will fill
            if col in {"home_rest", "away_rest"}:
                return np.nan

            # Moneyline/lines: neutral defaults if training included them but our row lacks them
            # if col in {"home_moneyline_prob", "away_moneyline_prob", "moneyline_prob_diff"}:
            #     return {"home_moneyline_prob": 0.5,
            #             "away_moneyline_prob": 0.5,
            #             "moneyline_prob_diff": 0.0}[col]
            # if col == "spread_line":
            #     return 0.0
            # if col == "total_line":
            #     return 45.0

            # Anything else: NaN (numeric imputer) or empty string (most_frequent for cats) — but we don't know dtype here,
            # so return NaN and let imputers handle it; OHE has handle_unknown='ignore' for unseen cats.
            return np.nan

        # Assemble X in EXACT training order
        data = {col: [_get_or_default(col)] for col in exp_all}
        X = pd.DataFrame(data, columns=exp_all)
 

        def _reg_predict(bundle: Any, X: pd.DataFrame) -> np.ndarray:
            """
            Predicts scores using a model bundle.

            Logic:
                - If bundle is a dict with 'hgbr', 'ridge', and 'weight', computes a weighted ensemble prediction.
                - If bundle contains a 'model' or 'estimator' key, delegates prediction to that object.
                - If bundle is a dict with any predictor object, uses the first found predictor.
                - If bundle is a single predictor object, calls its predict method.
                - Raises AttributeError if no valid prediction method is found.

            Fallback:
                - Attempts all reasonable dict keys before erroring, ensuring robust handling of model serialization formats.
            """
            log.debug("Model bundle type: %s, hasattr predict: %s", type(bundle), hasattr(bundle, "predict"))
            if isinstance(bundle, dict):
                log.debug("Model bundle keys: %s", list(bundle.keys()) if hasattr(bundle, 'keys') else 'no keys method')
                if {"hgbr", "ridge", "weight"}.issubset(bundle):
                    weight = float(bundle["weight"])
                    preds_hgbr = bundle["hgbr"].predict(X)
                    log.debug(f'PREDS_HGBR LINE: 970: {preds_hgbr}')
                    preds_ridge = bundle["ridge"].predict(X)
                    log.debug(f'preds_ridge: {preds_ridge}')
                    
                    return weight * preds_hgbr + (1.0 - weight) * preds_ridge
                
                delegate = bundle.get("model") or bundle.get("estimator")
                if delegate is not None and hasattr(delegate, "predict"):
                    return delegate.predict(X)
                # If dict but no expected structure, try to find any predictor
                for key, value in bundle.items():
                    if hasattr(value, "predict"):
                        log.debug("Using predictor from dict key: %s", key)
                        return value.predict(X)
            if not isinstance(bundle, dict) and hasattr(bundle, "predict"):
                return bundle.predict(X)
            raise AttributeError(f"Score model lacks predict method. Type: {type(bundle)}")
        
        '''HOME_SCORE'''
        # HOME_SCORE
        home_score = float(
            np.clip(
                _reg_predict(model_objects["home_model"], X)[0],
                0.0,
                70.0,
            )
        )
        print('home_score: ', home_score)
        
        # AWAY_SCORE
        away_score = float(
            np.clip(
                _reg_predict(model_objects["away_model"], X)[0],
                0.0,
                70.0,
            )
        )
        point_diff = round(home_score - away_score, 1)
        # Win probability from calibrated classifier if present, else sigmoid on margin
        try:
            # Safely obtain the raw "win_model" entry from model_objects. It may be:
            #  - a string/path (filename saved with joblib)
            #  - an already-loaded Pipeline object
            #  - None / missing
            try:
                win_model_entry = (
                    model_objects.get("win_model")
                    if isinstance(model_objects, dict)
                    else getattr(model_objects, "win_model", None)
                )
            except Exception:
                win_model_entry = None

            win_m = None
            # If entry appears to be a path, attempt to load from disk once
            if isinstance(win_model_entry, (str, bytes, os.PathLike)):
                win_path = Path(str(win_model_entry))
                # If a relative path, assume models directory
                if not win_path.is_absolute():
                    win_path = MODELS_DIR / win_path
                try:
                    win_m = joblib.load(win_path)
                except Exception:
                    log.exception("Failed to load win_model from path %s; will fallback to sigmoid", win_path)
                    win_m = None
            else:
                # If it's not a path and not None, assume it's already a loaded estimator (Pipeline)
                # Do not call joblib.load on an object.
                if win_model_entry is not None:
                    win_m = win_model_entry
                else:
                    win_m = None

            # Use the estimator if available and supports predict_proba, else fallback
            if win_m is not None:
                try:
                    # Expect X to be a DataFrame or array prepared earlier
                    if hasattr(win_m, "predict_proba"):
                        home_prob = float(win_m.predict_proba(X)[0, 1])
                    else:
                        # If estimator lacks predict_proba, fallback to predict then sigmoid
                        pred_margin = float(win_m.predict(X)[0])
                        home_prob = 1.0 / (1.0 + math.exp(-0.25 * pred_margin))
                except Exception:
                    log.exception("win_model prediction failed; falling back to margin sigmoid")
                    home_prob = 1.0 / (1.0 + math.exp(-0.25 * point_diff))
            else:
                # No classifier available — use margin-based sigmoid as fallback
                home_prob = 1.0 / (1.0 + math.exp(-0.25 * point_diff))
        except Exception:
            # Very defensive final fallback
            log.exception("Unexpected error while computing win probability; using sigmoid fallback")
            home_prob = 1.0 / (1.0 + math.exp(-0.25 * point_diff))

        # Read mode defensively
        try:
            mode_val = model_objects.get("mode") if isinstance(model_objects, dict) else getattr(model_objects, "mode", "models")
        except Exception:
            mode_val = "production"
        # Ensure a string is returned for the response model
        if mode_val is None:
            mode_val = "production"
        else:
            mode_val = str(mode_val)

        return PredictionResponse(
            home_score=round(home_score, 1),
            away_score=round(away_score, 1),
            home_win_probability=home_prob,
            away_win_probability=1.0 - home_prob,
            point_diff=point_diff,
            mode=mode_val,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(400, f"Prediction failed: {e}")


@app.get("/predict/next-week")
def predict_next_week() -> Dict[str, Any]:
    """
    Batch prediction endpoint for all scheduled games in the next NFL week.

    Logic:
        - Determines the next prediction week and season using current NFL context.
        - Loads the schedule CSV and filters games for the upcoming week.
        - For each game, runs the prediction logic and aggregates results.
        - Collects errors for games where prediction fails, ensuring robust batch output.
        - Returns context, predictions, error details, and summary metrics.

    Returns:
        dict: Contains context, list of game predictions (with errors if any), total games, and count of successful predictions.
    """
    if model_objects is None:
        raise HTTPException(500, "Models not loaded.")
    try:
        ctx = get_current_nfl_context()
        spath = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
        if not spath.exists():
            raise HTTPException(404, "Schedule data not found")
        s = pd.read_csv(spath)
        games = s[
            (s["season"] == ctx["next_prediction_season"])
            & (s["week"] == ctx["next_prediction_week"])
        ]

        out: List[Dict[str, Any]] = []
        for _, g in games.iterrows():
            try:
                pr = predict_game(
                    PredictionRequest(
                        home_team=str(g["home_team"]),
                        away_team=str(g["away_team"]),
                        season=int(g["season"]),
                        week=int(g["week"]),
                    )
                )
                out.append(
                    {
                        "game_id": str(
                            g.get(
                                "game_id",
                                f"{g['season']}W{g['week']}-{g['away_team']}@{g['home_team']}",
                            )
                        ),
                        "season": int(g["season"]),
                        "week": int(g["week"]),
                        "home_team": str(g["home_team"]),
                        "away_team": str(g["away_team"]),
                        "kickoff": str(g.get("gameday", "TBD")),
                        "prediction": pr.dict(),
                    }
                )
            except Exception as e:
                out.append(
                    {"game_id": str(g.get("game_id", "unknown")), "error": str(e)}
                )
            
        return {
            "context": ctx,
            "games": out,
            "total_games": len(out),
            "successful_predictions": sum(1 for p in out if "prediction" in p),
        }
    except Exception as e:
        log.error("Next-week prediction error: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to predict next week: {e}")
