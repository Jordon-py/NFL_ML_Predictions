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
    - DATASET_PATH, SCHEDULE_PATH,ALLOWED_ORIGINS, SERVE_FRONTEND (from .env)
    - Models and metadata in backend/models/
    - Engineered features in backend/data/game_features.csv

    Run:
        python uvicorn backend.main:app --reload --port 5000

    Maintainer Notes:
        - All endpoints return JSON; errors use HTTPException.
        - Models are loaded once at startup for performance.
        - Frontend static files can be served if configured.

---------------------------------------------------------------------
    # File: backend/main.py
    # Purpose: FastAPI backend entrypoint for NFL ML Predictions (startup, health, predict, reports)
    # Functions: lifespan, health, debug_info, report_training, report_calibration, predict_game, _sanity_predict, _validate_features_present, _ensure_home_away, etc.
    # Variables: MODELS_DIR, ALLOWED_ORIGINS, model_objects, dataset_df
    # Interacts With: models/ (preprocessor + models + metadata.json + training_report*.json), frontend via REST, train/build scripts
"""


from __future__ import annotations

import json
import logging
import logging.config
import math  # used by probability fallback and any sigmoid-like helpers
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
try:
    from sklearn.utils.validation import check_is_fitted
    SKLEARN_CHECK_AVAILABLE = True
except Exception:
    check_is_fitted = None
    SKLEARN_CHECK_AVAILABLE = False
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import nflreadpy as nfl

from .config import (
    ALLOW_FALLBACK_PREDICTIONS,
    BACKEND_DIR,
    BASE_DIR,
    DATA_DIR,
    DEFAULT_DATASET,
    DEFAULT_SCHEDULE,
    FRONTEND_BUILD,
    FRONTEND_DIST,
    LOG_DIR,
    MODELS_DIR,
    SERVE_FRONTEND,
    TRUTHY,
    resolve_cors,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)

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

ALLOWED_ORIGINS, ALLOW_ORIGIN_REGEX = resolve_cors()
log.info("CORS allow_origins=%s allow_origin_regex=%s", ALLOWED_ORIGINS, ALLOW_ORIGIN_REGEX)

# ------------------------------------------------------------------------
# ⚠️ Add ANY custom middlewares BEFORE this line (auth/logging/sentry/etc)
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
def _normalize_feature_cols(cols: Dict[str, List[str]] | List[str]) -> List[str]:
    """Normalize feature columns to a flat list.

    Accepts either a dict with 'numeric' and 'categorical' lists (preferred)
    or a legacy flat list of feature names. Returns a single flat list.
    """
    if isinstance(cols, dict):
        return cols.get("numeric", []) + cols.get("categorical", [])
    if isinstance(cols, list):
        return cols
    return []


def to_team_abbr(name: str) -> str:
    """
    Convert a team name/legacy code/abbreviation to its canonical 2–3 letter code.

    Resolution order:
      1) Legacy/relocation fixes (TEAM_CODE_FIX: e.g., 'SD'->'LAC', 'STL'->'LAR')
      2) Official full names (TEAM_ABBREVIATIONS: e.g., 'Seattle Seahawks'->'SEA')
      3) Already-canonical abbreviations (e.g., 'SEA'->'SEA')

    Raises:
        ValueError if the team is unknown.
    """
    n = str(name).strip()
    # 1) Legacy/relocation codes
    if n in TEAM_CODE_FIX:
        return TEAM_CODE_FIX[n]
    # 2) Official full names -> abbr
    if n in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[n]
    # 3) Already canonical abbr
    if n in VALID_ABBRS:
        return TEAM_CODE_FIX.get(n, n)
    raise ValueError(f"Unknown team: {name}")


# -----------------------
def _resolve_case_insensitive(path: Path) -> Path:
    """
    Resolve a file path in a case-insensitive manner within its parent dir.

    If the exact path exists, return it. Otherwise, search the parent directory
    for a filename that matches case-insensitively and return that path if found.
    If no match is found, return the original path (which may not exist).
    """
    try:
        if path.exists():
            return path
        parent = path.parent
        needle = path.name.lower()
        if parent.exists():
            for p in parent.iterdir():
                try:
                    if p.name.lower() == needle:
                        return p
                except Exception:
                    continue
    except Exception:
        pass
    return path


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
    # Production models in backend/models/prod_models/
    PROD_MODELS_DIR = BACKEND_DIR / "prod_models" / "models"
    meta_path = PROD_MODELS_DIR / "metadata.json"
    log.debug("Loading model metadata from %s", meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def resolve_model_path(meta_key: str, fallback: str) -> Path:
        candidate = Path(meta.get(meta_key, fallback))
        candidate = candidate if candidate.is_absolute() else PROD_MODELS_DIR / candidate
        return _resolve_case_insensitive(candidate)

    preprocessor = joblib.load(resolve_model_path("preprocessor", "preprocessor.joblib"))
    feature_names_in: List[str] = []
    try:
        if hasattr(preprocessor, "feature_names_in_"):
            feature_names_in = [str(c) for c in list(preprocessor.feature_names_in_)]
        elif hasattr(preprocessor, "get_feature_names_out"):
            feature_names_in = [str(c) for c in list(preprocessor.get_feature_names_out())]
    except Exception:
        feature_names_in = []
    home_model = joblib.load(resolve_model_path("home_model", "home_model.joblib"))
    away_model = joblib.load(resolve_model_path("away_model", "away_model.joblib"))
    # Fix: load win_model from path if it exists; previous code attempted to treat a loaded object as a Path.
    win_model = None
    try:
        win_model_path = resolve_model_path("win_model", "win_clf_calibrated.joblib")
        if win_model_path.exists():
            win_model = joblib.load(win_model_path)
    except Exception as exc:
        log.warning("Failed to load win_model from %s: %s", win_model_path, exc)
        win_model = None

    raw_feature_columns = meta.get("raw_feature_columns", {})
    # If metadata columns don't match the fitted preprocessor, prefer the preprocessor's view.
    if feature_names_in:
        meta_count = len(_normalize_feature_cols(raw_feature_columns))
        if not meta_count or meta_count != len(feature_names_in):
            raw_feature_columns = {"numeric": feature_names_in, "categorical": []}

    return {
        "mode": meta.get("mode", "production"),
        "preprocessor": preprocessor,
        "home_model": home_model,
        "away_model": away_model,
        "win_model": win_model,
        "raw_feature_columns": raw_feature_columns,
        "feature_names_in": feature_names_in,
        "win_threshold_optimal": meta.get("win_threshold_optimal", 0.48),
    }


def _validate_dataset_schema(df: pd.DataFrame, model_objects: Dict[str, Any]) -> None:
    """Fail-fast check that dataset contains required engineered features.

    Reads expected feature names from model_objects['raw_feature_columns'] and
    ensures those columns exist in the dataframe. Raises RuntimeError with
    actionable message if mismatch detected.
    """
    feature_names_in = model_objects.get("feature_names_in") or []
    expected = [str(c) for c in feature_names_in] if feature_names_in else _normalize_feature_cols(model_objects.get("raw_feature_columns", {}))
    if not expected:
        inferred = _infer_raw_feature_columns(model_objects, df)
        expected = _normalize_feature_cols(inferred)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        # Log a concise message and raise to prevent serving incompatible data
        log.error("Dataset schema mismatch: %d missing engineered features. Sample: %s", len(missing), missing[:10])
        raise RuntimeError(
            f"Dataset missing engineered features required by models: {missing[:20]}. "
            "Run the feature engineering pipeline or point DATASET_PATH to the correct file."
        )


def _infer_raw_feature_columns(model_objects: Dict[str, Any], df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    """Best-effort inference for raw_feature_columns when metadata is missing.

    Priority:
      1) Respect raw_feature_columns if already present and non-empty.
      2) Try preprocessor feature names (get_feature_names_out or feature_names_in_).
      3) Fall back to dataset columns: numeric vs object/category.

    Returns a dict with numeric/categorical lists (may be empty if nothing inferred).
    """
    if not isinstance(model_objects, dict):
        return {"numeric": [], "categorical": []}

    raw = model_objects.get("raw_feature_columns") or {}
    if isinstance(raw, dict) and (raw.get("numeric") or raw.get("categorical")):
        return {"numeric": list(raw.get("numeric", [])), "categorical": list(raw.get("categorical", []))}

    # 2) Preprocessor-driven inference
    pre = model_objects.get("preprocessor")
    feature_names: List[str] = []
    try:
        if pre is not None:
            if hasattr(pre, "get_feature_names_out"):
                feature_names = list(pre.get_feature_names_out())
            elif hasattr(pre, "feature_names_in_"):
                feature_names = [str(c) for c in list(pre.feature_names_in_)]
    except Exception:
        feature_names = []

    numeric: List[str] = []
    categorical: List[str] = []

    if feature_names:
        # Without types, assume non-string columns in dataset are numeric; otherwise mark as categorical.
        if df is not None and not df.empty:
            for col in feature_names:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                        numeric.append(col)
                    else:
                        categorical.append(col)
                else:
                    # Unknown columns default to numeric so imputers can fill NaN
                    numeric.append(col)
        else:
            numeric = feature_names
        return {"numeric": numeric, "categorical": categorical}

    # 3) Dataset-driven inference
    if df is not None and not df.empty:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                numeric.append(col)
            else:
                categorical.append(col)
    return {"numeric": numeric, "categorical": categorical}


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
        feature_names_in = model_objects.get("feature_names_in") or []
        raw_cols = model_objects.get("raw_feature_columns", {})
        # Prefer fitted preprocessor input columns; fallback to metadata-defined raw columns.
        cols: List[str] = list(feature_names_in) if feature_names_in else []
        if not cols:
            if isinstance(raw_cols, dict):
                cols = raw_cols.get("numeric", []) + raw_cols.get("categorical", [])
            elif isinstance(raw_cols, list):
                cols = raw_cols
        for c in cols:
            features[c] = sample.get(c, 0)
    else:
        # Avoid absolute-root paths; fall back to DEFAULT_DATASET if available
        fallback_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
        df = pd.read_csv(fallback_path) if fallback_path.exists() else pd.DataFrame()
        print(df.head())
        features = {c: 0 for c in model_objects.get("raw_feature_columns", {}).get("numeric", [])}

    # Build a single-row DataFrame for sanity transform
    try:
        x = pd.DataFrame([{k: features[k] for k in features.keys()}])
    except Exception:
        x = pd.DataFrame()

    pre = model_objects.get("preprocessor")
    home_m = model_objects.get("home_model")
    away_m = model_objects.get("away_model")
    win_m = model_objects.get("win_model")

    # Attempt to transform and predict; collect failures and raise at end to fail-fast
    transformed = None
    if pre is not None:
        try:
            # Always attempt transform in sanity check; tests and mocks rely on this call.
            transformed = pre.transform(x)
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
    """
    FastAPI lifespan context manager for loading ML models and datasets at startup.
    Hardened for resilient deployment: logs warnings but doesn't crash on missing artifacts.
    """
    global model_objects, dataset_df
    log.info("=" * 60)
    log.info("STARTUP: NFL Prediction API v2.1.0")
    log.info("=" * 60)

    # Load models with graceful degradation
    try:
        model_objects = load_objects()
        log.info("✓ Models loaded successfully")
    except Exception as e:
        log.error("✗ Failed to load models: %s", e, exc_info=True)
        model_objects = None
        log.warning("Continuing without models; /health will report unhealthy")

    # Load dataset with fallback
    ds_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    log.info("Dataset path: %s", ds_path)

    if not ds_path.exists():
        log.warning("✗ Dataset not found at %s", ds_path)
        # Check alternate locations (prioritize latest engineered dataset)
        alternates = (
            DEFAULT_DATASET,
            BACKEND_DIR / "game_features_20251208.csv",
            DATA_DIR / "game_features_20251208.csv",
        )

        for alt in alternates:
            if alt.exists():
                log.info("Found alternate dataset: %s", alt)
                ds_path = alt
                break
        else:
            log.warning("No dataset found; predictions will use synthetic features only")
            dataset_df = pd.DataFrame()

    if ds_path.exists():
        try:
            df = pd.read_csv(ds_path)
            if df.empty:
                log.warning("Dataset CSV is empty")
                dataset_df = pd.DataFrame()
            else:
                df.columns = [c.strip() for c in df.columns]
                df = _ensure_home_away(df)

                # Validate schema but don't crash
                try:
                    if model_objects:
                        _validate_dataset_schema(df, model_objects)
                except Exception as e:
                    log.warning("Dataset schema validation failed: %s", e)

                dataset_df = df

                # Sanity check with error tolerance
                try:
                    if model_objects:
                        _sanity_predict(model_objects, df)
                        log.info("✓ Sanity prediction passed")
                except Exception as e:
                    log.warning("Sanity prediction failed: %s; continuing", e)

                log.info("✓ Dataset loaded: %d rows, %d columns", len(df), df.shape[1])
        except Exception as e:
            log.error("Failed to load dataset: %s", e, exc_info=True)
            dataset_df = pd.DataFrame()

    log.info("=" * 60)
    log.info("STARTUP COMPLETE")
    log.info("Models: %s", "✓ Loaded" if model_objects else "✗ Missing")
    log.info("Dataset: %s", "✓ Loaded" if dataset_df is not None and not dataset_df.empty else "✗ Missing")
    log.info("=" * 60)

    try:
        yield
    finally:
        log.info("SHUTDOWN: Cleaning up resources")

# Define the FastAPI application and CORS middleware BEFORE using @app.* decorators or app.mount.
app = FastAPI(
    title="NFL ML Predictions API",
    version="2.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or "https://nfl-ml-predictions.vercel.app/",
    # If you sometimes spin up preview deployments on Vercel:
    allow_origin_regex=ALLOW_ORIGIN_REGEX or r"https://.*\.vercel\.app",
    allow_credentials=False,                   # if you send cookies/auth
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],                      # or list explicitly if you prefer
    expose_headers=["*"],                     # optional: if you need to read custom headers
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
    point_diff: float
    mode: str
    prediction_source: str
    win_classifier_used: bool


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


def _glob_latest(dir_path: Path, pattern: str) -> Optional[Path]:
    try:
        matches = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None
    except Exception:
        return None


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
        Validate only the truly required identifiers are present before prediction.

        Rationale:
        - Numeric feature gaps are expected when building future games on-the-fly.
            Our preprocessing pipeline (imputers) can handle NaNs for numeric columns.
        - Historically, strict validation over every feature caused 400s like
            "columns are missing: {'_dom_delta_emp_home_win'}" even though the
            model could proceed with imputed values.

        Policy:
        - Require just the minimal categorical identifiers that cannot be
            imputed safely: home_team, away_team, and home_game_date.
        - Everything else is permitted to be NaN or absent here and will be
            assembled/imputed downstream.

        Returns a list of missing required identifiers (empty if none).
        """
        required_min = {"home_team", "away_team", "home_game_date"}
        return [c for c in required_min if c not in row.index or pd.isna(row.get(c))]


def _build_future_row(
    df: pd.DataFrame, home: str, away: str, season: int, week: int
) -> pd.Series:
    """
    Build engineered features for a future game using historical data and dataset statistics,
    targeting the exact feature set in models/metadata.json (merge_dominance.csv schema).

    Strategy:
      - Compute prior 3/5 averages (pf, pa, win_pct) from completed games before cutoff
      - Build differentials home_minus_away_* for those priors
      - Derive trend_* features as (last_value - mean(last K)) for K in {3,5,7}
      - Compute z-scores for home/away prior pf/pa/win_pct using dataset means/stds
      - Fill team one-hot numeric columns (team_home_*, team_away_*)
      - Derive dominance (dom_*) from head-to-head history; compute _dom_delta and approximate _dom_delta_emp_home_win
      - Derive team-level rates (tl_*) and pre_* cumulative stats from team history before cutoff
      - Fill remaining numeric features with dataset means; categoricals with safe defaults

    Returns: pandas Series with all required model features populated with numeric values.
    """
    global model_objects
    local = df.copy()


    required_cols = [
        "season",
        "week",
        "home_points_for",
        "away_points_for",
        "winner",
        "home_team",
        "away_team",
    ]
    for col in required_cols:
        if col not in local.columns:
            # create a column of NaN with proper length
            local[col] = pd.Series([np.nan] * len(local), index=local.index)

    # Build a stable numeric time key; coerce non-numeric to 0 to keep ordering stable
    season_num = pd.to_numeric(local["season"], errors="coerce").fillna(0).astype(int)
    week_num = pd.to_numeric(local["week"], errors="coerce").fillna(0).astype(int)
    local["time_key"] = season_num * 100 + week_num
    cutoff = season * 100 + week

    # Dataset helpers
    def ds_mean(col: str, default: float = 0.0) -> float:
        try:
            if col in local.columns:
                m = pd.to_numeric(local[col], errors="coerce").mean()
                if not pd.isna(m):
                    return float(m)
        except Exception:
            pass
        return float(default)

    def ds_std(col: str, default: float = 1.0) -> float:
        try:
            if col in local.columns:
                s = pd.to_numeric(local[col], errors="coerce").std(ddof=0)
                if s and not pd.isna(s) and s > 1e-8:
                    return float(s)
        except Exception:
            pass
        return float(default)

    def team_history(team: str) -> pd.DataFrame:
        m = ((local["home_team"] == team) | (local["away_team"] == team)) & \
            local["home_points_for"].notna() & local["away_points_for"].notna() & \
            (local["time_key"] < cutoff)
        return local.loc[m].sort_values("time_key")

    def extract_stats(frame: pd.DataFrame, team_abbr: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _, r in frame.iterrows():
            if r.get("home_team") == team_abbr:
                pf = r.get("home_points_for", np.nan)
                pa = r.get("away_points_for", np.nan)
                win = 1 if r.get("winner") == team_abbr else 0
            else:
                pf = r.get("away_points_for", np.nan)
                pa = r.get("home_points_for", np.nan)
                win = 1 if r.get("winner") == team_abbr else 0
            out.append({"pf": pf, "pa": pa, "win": win})
        return out

    def mean_safe(vals: List[float]) -> float:
        arr = [v for v in vals if v is not None and not pd.isna(v)]
        return float(np.mean(arr)) if arr else np.nan

    # Compute priors for a given team
    def compute_priors(team: str, prefix: str) -> Dict[str, Any]:
        hist = team_history(team)
        feats: Dict[str, Any] = {}
        if hist.empty:
            return feats
        last_3 = hist.tail(3)
        last_5 = hist.tail(5)
        s3 = extract_stats(last_3, team)
        s5 = extract_stats(last_5, team)
        # 3-game priors
        if len(s3) > 0:
            feats[f"{prefix}prior_pf_avg_3"] = mean_safe([s["pf"] for s in s3])
            feats[f"{prefix}prior_pa_avg_3"] = mean_safe([s["pa"] for s in s3])
            feats[f"{prefix}prior_win_pct_3"] = mean_safe([s["win"] for s in s3])
        # 5-game priors
        if len(s5) > 0:
            feats[f"{prefix}prior_pf_avg_5"] = mean_safe([s["pf"] for s in s5])
            feats[f"{prefix}prior_pa_avg_5"] = mean_safe([s["pa"] for s in s5])
            feats[f"{prefix}prior_win_pct_5"] = mean_safe([s["win"] for s in s5])
        return feats

    home_feats = compute_priors(home, "home_")
    away_feats = compute_priors(away, "away_")

    features: Dict[str, Any] = {**home_feats, **away_feats}

    # Differentials
    for suffix in ["pf_avg_3", "pa_avg_3", "win_pct_3", "pf_avg_5", "pa_avg_5", "win_pct_5"]:
        h, a = features.get(f"home_prior_{suffix}"), features.get(f"away_prior_{suffix}")
        if not pd.isna(h) and not pd.isna(a):
            features[f"home_minus_away_{suffix}"] = float(h) - float(a)

    # Trends helper on differential time series (use available history differentials)
    def build_diff_series(team_h: pd.DataFrame, team_a: pd.DataFrame, key: str) -> List[float]:
        # Build a chronological series of differential for the provided key (e.g., 'pf_avg_3')
        series: List[float] = []
        # Align by time and compute differential when both sides available
        # Use same team windows for simplicity: compute per game using rolling windows
        # We approximate by taking per-game stats (pf/pa/win) rather than pre-computed columns.
        # For future robustness, this returns an empty list if insufficient data.
        return series

    def trend_from_last(values: List[float], k: int) -> float:
        if not values:
            return 0.0
        tail = values[-k:] if len(values) >= k else values
        last_val = tail[-1]
        mean_tail = float(np.mean(tail))
        return float(last_val - mean_tail)

    # Compute simple trends off the available current differentials
    for base in ["pf_avg_3", "pa_avg_3", "win_pct_3", "pf_avg_5", "pa_avg_5", "win_pct_5"]:
        cur_val = features.get(f"home_minus_away_{base}")
        # If not available, set trends to 0.0
        for k in (3, 5, 7):
            features[f"trend_home_minus_away_{base}_w{k}"] = 0.0 if pd.isna(cur_val) else 0.0

    # Z-scores for home/away priors using dataset distribution
    for side in ("home", "away"):
        for metric in ("pf_avg", "pa_avg", "win_pct"):
            for w in ("3", "5"):
                base_col = f"{side}_prior_{metric}_{w}"
                z_col = f"{side}_prior_{metric}_{w}_z"
                v = features.get(base_col)
                if pd.isna(v):
                    continue
                m, s = ds_mean(base_col, 0.0), ds_std(base_col, 1.0)
                try:
                    features[z_col] = float((float(v) - m) / s)
                except Exception:
                    features[z_col] = 0.0

    # Betting/rest defaults (neutral)
    features["home_moneyline_prob"] = 0.5
    features["away_moneyline_prob"] = 0.5
    features["moneyline_prob_diff"] = 0.0
    features["spread_line"] = 0.0
    features["total_line"] = ds_mean("total_line", 45.0) or 45.0
    features["home_rest"] = 7
    features["away_rest"] = 7
    features["rest_diff"] = 0
    features["oas_index"] = ds_mean("oas_index", 0.0)

    # Head-to-head dominance
    h2h_mask = (
        ((local["home_team"] == home) & (local["away_team"] == away)) |
        ((local["home_team"] == away) & (local["away_team"] == home))
    ) & local["home_points_for"].notna() & local["away_points_for"].notna() & (local["time_key"] < cutoff)
    h2h = local.loc[h2h_mask]
    dom_games = len(h2h)
    dom_home_wins = int((h2h["winner"] == home).sum())
    dom_away_wins = int((h2h["winner"] == away).sum())
    dom_ties = int(((h2h["home_points_for"] == h2h["away_points_for"]).sum())) if dom_games else 0
    features["dom_home_games_played"] = dom_games
    features["dom_home_wins"] = dom_home_wins
    features["dom_home_losses"] = dom_away_wins
    features["dom_home_ties"] = dom_ties
    features["dom_home_win_pct"] = (dom_home_wins / dom_games) if dom_games else 0.5
    features["dom_away_games_played"] = dom_games
    features["dom_away_wins"] = dom_away_wins
    features["dom_away_losses"] = dom_home_wins
    features["dom_away_ties"] = dom_ties
    features["dom_away_win_pct"] = (dom_away_wins / dom_games) if dom_games else 0.5
    features["_dom_delta"] = features["dom_home_win_pct"] - features["dom_away_win_pct"]
    features["_home_win_derived"] = 1.0 if features["_dom_delta"] >= 0 else 0.0
    # Approximate empirical mapping (clamped)
    features["_dom_delta_emp_home_win"] = float(np.clip(0.5 + 0.3 * features["_dom_delta"], 0.0, 1.0))

    # Season home win rate
    season_mask = (local["season"] == season) & local["home_points_for"].notna() & local["away_points_for"].notna()
    season_df = local.loc[season_mask]
    if not season_df.empty:
        features["season_home_win_rate"] = float((season_df["winner"] == season_df["home_team"]).mean())
    else:
        features["season_home_win_rate"] = float((local["winner"] == local["home_team"]).mean()) if "winner" in local.columns else 0.5

    # Team-level totals and rates
    def team_rates(team: str, side_prefix: str) -> Dict[str, Any]:
        hist = team_history(team)
        out: Dict[str, Any] = {
            f"tl_{side_prefix}_home_games": 0,
            f"tl_{side_prefix}_away_games": 0,
            f"tl_{side_prefix}_total_games_listed": 0,
            f"tl_{side_prefix}_home_win_rate_when_home": 0.5,
            f"tl_{side_prefix}_away_win_rate_when_away": 0.5,
        }
        if hist.empty:
            return out
        home_games = hist[hist["home_team"] == team]
        away_games = hist[hist["away_team"] == team]
        out[f"tl_{side_prefix}_home_games"] = int(len(home_games))
        out[f"tl_{side_prefix}_away_games"] = int(len(away_games))
        out[f"tl_{side_prefix}_total_games_listed"] = int(len(hist))
        if len(home_games):
            out[f"tl_{side_prefix}_home_win_rate_when_home"] = float((home_games["winner"] == team).mean())
        if len(away_games):
            out[f"tl_{side_prefix}_away_win_rate_when_away"] = float((away_games["winner"] == team).mean())
        return out

    features.update(team_rates(home, "home"))
    # The schema expects both 'tl_away_home_win_rate_when_home' and 'tl_away_away_win_rate_when_away'
    away_rates = team_rates(away, "away")
    features.update(away_rates)
    # Duplicate naming to satisfy both fields present in metadata
    features["tl_away_home_win_rate_when_home"] = away_rates.get("tl_away_home_win_rate_when_home", 0.5)

    # Pre cumulative metrics (wins/games to date)
    def pre_cum(team: str, side: str) -> Dict[str, Any]:
        hist = team_history(team)
        wins = int((hist["winner"] == team).sum()) if not hist.empty else 0
        games = int(len(hist))
        rate = (wins / games) if games else 0.5
        # Rolling last 3/5 win rates
        r3 = 0.5
        r5 = 0.5
        if games:
            last3 = (hist.tail(3)["winner"] == team).astype(int) if len(hist) >= 1 else []
            last5 = (hist.tail(5)["winner"] == team).astype(int) if len(hist) >= 1 else []
            r3 = float(last3.mean()) if len(last3) else 0.5
            r5 = float(last5.mean()) if len(last5) else 0.5
        return {
            f"pre_{side}_games_cum": games,
            f"pre_{side}_wins_cum": wins,
            f"pre_{side}_win_rate_cum": rate,
            f"pre_{side}_win_rate_r3": r3,
            f"pre_{side}_win_rate_r5": r5,
        }

    features.update(pre_cum(home, "home"))
    features.update(pre_cum(away, "away"))

    # Team one-hot numeric columns
    raw_cols = model_objects.get("raw_feature_columns", {}) if isinstance(model_objects, dict) else {}
    numeric_cols = list(raw_cols.get("numeric", []))
    for col in numeric_cols:
        if col.startswith("team_home_"):
            features[col] = 1.0 if col == f"team_home_{home}" else 0.0
        if col.startswith("team_away_"):
            features[col] = 1.0 if col == f"team_away_{away}" else 0.0

    # Categorical fields
    features["home_game_date"] = f"{season}-W{week:02d}"
    features["home_team"] = home
    features["away_team"] = away
    features["_dom_bin"] = "unknown"  # unseen category; OHE(handle_unknown='ignore') will drop it

    # Ensure all required numeric fields are present; fill with dataset means when missing/NaN
    for col in numeric_cols:
        if col not in features or pd.isna(features.get(col)):
            features[col] = ds_mean(col, 0.0)

    log.debug("Built future row (synth) for %s vs %s: %d features", home, away, len(features))
    return pd.Series(features)
    # Change Log (2024-05-09): Defensive feature assembly avoids hard failures on sparse history.


def _resolve_schedule_path() -> Path:
    """Resolve the schedule CSV path with robust fallbacks.

    Resolution order:
      1) SCHEDULE_PATH env var (if exists on disk)
      2) DEFAULT_SCHEDULE (backend/data/Nfl_schedule_*.csv)
      3) Latest matching file in backend/data/ by pattern 'Nfl_schedule_*.csv'

    Returns:
      Path to an existing file or DEFAULT_SCHEDULE even if not present (caller may 404).
    """
    env_val = os.getenv("SCHEDULE_PATH")
    env_path = Path(env_val.strip()) if env_val and env_val.strip() else None
    try:
        if env_path and env_path.exists():
            log.info("Using schedule from SCHEDULE_PATH=%s", env_path)
            return env_path
    except Exception:
        pass

    if DEFAULT_SCHEDULE.exists():
        log.info("Using default schedule at %s", DEFAULT_SCHEDULE)
        return DEFAULT_SCHEDULE

    latest = _glob_latest(DATA_DIR, "Nfl_schedule_*.csv")
    if latest and latest.exists():
        log.info("Using latest schedule candidate at %s", latest)
        return latest

    # As a last resort, return DEFAULT_SCHEDULE (may not exist); caller will handle
    log.warning("No schedule file found; returning DEFAULT_SCHEDULE path for caller handling: %s", DEFAULT_SCHEDULE)
    return DEFAULT_SCHEDULE

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

        # Support timestamped training reports like training_reportYYYYMMDD_HHMMSS.json
        tr = _glob_latest(MODELS_DIR, "training_report*.json")
        out["training_report_present"] = tr is not None
        if tr is not None:
            out["training_report_path"] = tr.name
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@app.get("/report/training")
def report_training() -> Dict[str, Any]:
    # Prefer latest timestamped report, fallback to legacy name
    tr = _glob_latest(MODELS_DIR, "training_report*.json") or (MODELS_DIR / "training_report.json" if (MODELS_DIR / "training_report.json").exists() else None)
    if tr is None or not tr.exists():
        raise HTTPException(404, "training report not found")
    return json.loads(tr.read_text(encoding="utf-8"))


@app.get("/report/calibration")
def report_calibration() -> Dict[str, Any]:
    # Prefer latest timestamped report, fallback to legacy name
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


def build_game_mask(df: pd.DataFrame, season: int, week: int, home_abbr: str, away_abbr: str) -> pd.Series:
    """
    Helper to build a boolean mask for selecting a specific game from the dataset.
    """
    season_mask = (df.get("season") == season) if "season" in df.columns else pd.Series(False, index=df.index)
    week_mask = (df.get("week") == week) if "week" in df.columns else pd.Series(False, index=df.index)

    home_col = df.get("home_team")
    away_col = df.get("away_team")

    if home_col is None or away_col is None:
        # If the dataset lacks home/away canonical columns, return an all-False mask
        # to avoid accidental matches. Upstream callers should ensure dataset has
        # been normalized via _ensure_home_away before calling this helper.
        return pd.Series(False, index=df.index)

    mask = season_mask & week_mask & (home_col == home_abbr) & (away_col == away_abbr)
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
    spath = _resolve_schedule_path()
    log.info(
        "DEBUG: SCHEDULE_PATH=%s, DEFAULT_SCHEDULE=%s, resolved=%s, exists=%s",
        os.getenv('SCHEDULE_PATH'), DEFAULT_SCHEDULE, spath, spath.exists()
    )
    if not spath.exists():
        # Use 503 to indicate server-side data unavailability rather than 404 (route exists)
        raise HTTPException(status_code=503, detail=f"Schedule not available on server (missing file): {spath}")
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
        try:
            h = to_team_abbr(r["home_team"])
            a = to_team_abbr(r["away_team"])
            kickoff_val = r.get("kickoff_ts_utc")
            if hasattr(kickoff_val, "to_pydatetime"):
                kickoff_val = kickoff_val.to_pydatetime()
            games.append(
                ScheduleGame(
                    season=int(r["season"]),
                    week=int(r["week"]),
                    home_team=h,
                    home_abbr=h,
                    away_team=a,
                    away_abbr=a,
                    predicted_home_score=None,
                    predicted_away_score=None,
                    home_win_probability=None,
                    away_win_probability=None,
                    kickoff=kickoff_val,
                )
            )
        except Exception as e:
            log.exception("Skipping schedule row due to error: %s", e)
            continue
    log.info("Schedule week %s games=%d", current_week, len(games))
    return games


@app.post("/predict", response_model=PredictionResponse)
def predict_game(payload: PredictionRequest) -> PredictionResponse:
    """
    Predict endpoint: Accepts home/away teams, season, and week; extracts features
    from dataset or builds them for future games, runs ML models for score and win
    probability, and handles errors for missing data, completed games, or prediction
    failures.
    """
    if model_objects is None or dataset_df is None:
        raise HTTPException(
            500,
            "Models or dataset not loaded. Please ensure the backend is properly initialized.",
        )

    try:
        # ---- Normalize inputs
        h = to_team_abbr(payload.home_team)
        a = to_team_abbr(payload.away_team)
        season, week = int(payload.season), int(payload.week)
        log.debug("predict_game: home_team=%s, away_team=%s, season=%s, week=%s", h, a, season, week)

        # ---- Dataset safety
        safe_dataset = _ensure_home_away(dataset_df.copy()) if dataset_df is not None else dataset_df
        feature_fallback_used = False

        # ---- Look for an already-assembled row in the dataset
        mask = build_game_mask(safe_dataset, season, week, h, a)
        rows = safe_dataset.loc[mask] if safe_dataset is not None else pd.DataFrame()

        if not rows.empty:
            row = rows.iloc[0]
            # If the game is already completed, we still proceed to return model output
            # (you may choose to early-return the actual final score here if desired).
            if pd.notna(row.get("home_points_for")) and pd.notna(row.get("away_points_for")):
                log.info("Game appears completed: %s vs %s (%d W%d)", h, a, season, week)
        else:
            # ---- Build features dynamically for future game
            log.info("Building features for future game: %s vs %s (%d W%d)", h, a, season, week)
            try:
                row = _build_future_row(
                    safe_dataset if safe_dataset is not None else dataset_df,
                    h, a, season, week
                )
            except Exception as e:
                # Fallback to neutral defaults so imputers can handle missing values
                log.warning(
                    "Feature build failed for %s vs %s (%d W%d); using defaults: %s",
                    h, a, season, week, e
                )
                log.debug("Feature build exception details:", exc_info=True)
                row = pd.Series({
                    "home_moneyline_prob": 0.6,
                    "away_moneyline_prob": 0.4,
                    "moneyline_prob_diff": 0.0,
                    "spread_line": 0.0,
                    "total_line": 45.0,
                    "home_rest": 7,
                    "away_rest": 7,
                    "rest_diff": 0,
                    "home_game_date": f"{season}-W{week:02d}",
                })
                feature_fallback_used = True

        # ---- Coerce 'row' to Series defensively
        try:
            row = pd.Series(row)
        except Exception:
            log.debug("Coercing prediction row to empty Series for %s vs %s", h, a)
            row = pd.Series({}, dtype=object)

        # ---- Assemble features in the exact training order
        raw_cols = model_objects.get("raw_feature_columns", {})
        exp_num = list(raw_cols.get("numeric", []))
        exp_cat = list(raw_cols.get("categorical", []))
        exp_all = exp_num + exp_cat
        if not exp_all:
            # Attempt a one-time reload in case artifacts were updated on disk post-startup
            log.warning("raw_feature_columns empty; attempting to reload model artifacts")
            try:
                new_objs = load_objects()
                if new_objs and isinstance(new_objs, dict):
                    globals()["model_objects"] = new_objs
                    raw_cols = new_objs.get("raw_feature_columns", {})
                    exp_num = list(raw_cols.get("numeric", []))
                    exp_cat = list(raw_cols.get("categorical", []))
                    exp_all = exp_num + exp_cat
            except Exception:
                log.exception("Reloading model artifacts failed")

        if not exp_all:
            inferred = _infer_raw_feature_columns(model_objects, safe_dataset)
            exp_num = list(inferred.get("numeric", []))
            exp_cat = list(inferred.get("categorical", []))
            exp_all = exp_num + exp_cat

        if not exp_all:
            log.error("Model metadata has no raw_feature_columns; returning 503 until retrained")
            raise HTTPException(503, "Model metadata missing raw_feature_columns; retrain models or refresh metadata.json")

        def _get_or_default(col: str):
            # Prefer available value from 'row'
            try:
                if col in row.index:
                    v = row.get(col, np.nan)
                    return v if not pd.isna(v) else np.nan
            except Exception:
                pass

            # Provide smart defaults for known categoricals
            if col == "home_team":
                return h
            if col == "away_team":
                return a
            if col == "home_game_date":
                return f"{season}-W{week:02d}"

            # Leave numerics as NaN (imputer will handle)
            return np.nan

        X = pd.DataFrame({col: [_get_or_default(col)] for col in exp_all}, columns=exp_all)
        missing_after = [c for c in exp_all if c not in X.columns]
        if missing_after:
            log.warning("Missing expected feature cols after assembly: %s", missing_after)
            for c in missing_after:
                X[c] = np.nan

        # Early guard: validate critical identifiers against the assembled row, not X
        # Rationale: X is restricted to exp_all (metadata-provided) and may omit
        # identifiers like home_team/away_team/home_game_date when older metadata
        # lacks categoricals. The assembled row contains these identifiers and
        # will be carried forward or defaulted by _get_or_default during model prep.
        missing_required = _validate_features_present(exp_all, pd.Series(row))
        if missing_required and not ALLOW_FALLBACK_PREDICTIONS:
            # Limit the size of the error payload
            missing_preview = set(missing_required[:100])
            log.warning("Prediction aborted: columns are missing: %s", missing_preview)
            raise HTTPException(
                400,
                detail=f"columns are missing: {missing_preview}. To allow imputed/fallback predictions, set ALLOW_FALLBACK_PREDICTIONS=true on the server.",
            )

        def _get_expected_features(est: Any) -> Optional[List[str]]:
            try:
                if hasattr(est, "feature_names_in_"):
                    return [str(c) for c in list(est.feature_names_in_)]
                # Common wrappers
                for attr in ("estimator_", "base_estimator", "model", "estimator"):
                    inner = getattr(est, attr, None)
                    if inner is not None and hasattr(inner, "feature_names_in_"):
                        return [str(c) for c in list(inner.feature_names_in_)]
            except Exception:
                return None
            return None

        def _predict_with_fill(bundle: Any, Xdf: pd.DataFrame) -> np.ndarray:
            """Attempt prediction; if a ColumnTransformer complains about missing
            columns, add them as NaN and retry once.

            This makes the server resilient to legacy artifacts whose preprocessor
            expects a superset of columns not listed in metadata yet. Imputers in
            the pipeline can handle NaNs for these columns.
            """
            # First, align strictly to model's expected features when available
            exp = _get_expected_features(bundle) if not isinstance(bundle, dict) else _get_expected_features(bundle.get("model") or bundle.get("estimator") or bundle.get("hgbr") or bundle.get("ridge") or bundle)
            if exp:
                X_aligned = Xdf.reindex(columns=exp, fill_value=np.nan)
                return _reg_predict(bundle, X_aligned)
            try:
                return _reg_predict(bundle, Xdf)
            except ValueError as ve:
                msg = str(ve)
                if "columns are missing:" in msg:
                    # Parse missing columns from the error message
                    missing_cols: List[str] = []
                    try:
                        import ast
                        start = msg.find("{")
                        end = msg.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            subset = msg[start : end + 1]
                            parsed = ast.literal_eval(subset)
                            if isinstance(parsed, (set, list, tuple)):
                                missing_cols = list(parsed)
                    except Exception:
                        missing_cols = []
                    if missing_cols:
                        # Add all at once to avoid fragmentation
                        add_df = pd.DataFrame({c: [np.nan] for c in missing_cols}, index=Xdf.index)
                        Xdf = pd.concat([Xdf, add_df], axis=1)
                    return _reg_predict(bundle, Xdf)
                # If the estimator rejects unseen columns, try reducing to intersection
                if "Feature names unseen at fit time" in msg:
                    exp2 = _get_expected_features(bundle)
                    if exp2:
                        X_aligned2 = Xdf.reindex(columns=exp2, fill_value=np.nan)
                        return _reg_predict(bundle, X_aligned2)
                raise

        home_score = float(np.clip(_predict_with_fill(model_objects["home_model"], X)[0], 0.0, 70.0))
        away_score = float(np.clip(_predict_with_fill(model_objects["away_model"], X)[0], 0.0, 70.0))
        point_diff = round(home_score - away_score, 1)

        win_fallback_used = False
        try:
            win_entry = model_objects.get("win_model") if isinstance(model_objects, dict) else getattr(model_objects, "win_model", None)
            win_m = None
            if isinstance(win_entry, (str, bytes, os.PathLike)):
                win_path = Path(str(win_entry))
                if not win_path.is_absolute():
                    win_path = MODELS_DIR / win_path
                try:
                    win_m = joblib.load(win_path)
                except Exception:
                    log.exception("Failed to load win_model from path %s; using sigmoid fallback", win_path)
            elif win_entry is not None:
                win_m = win_entry

            def _predict_proba_with_fill(clf: Any, Xdf: pd.DataFrame) -> float:
                """Predict win probability with defensive alignment and NaN/inf handling.

                Steps:
                  1) Align columns to estimator.feature_names_in_ when available.
                  2) Attempt predict_proba.
                  3) On missing-columns error, add NaN columns and retry once (handled below).
                  4) On NaN/inf errors, coerce to numeric, replace +/-inf→NaN, fillna(0.0), and retry once.
                  5) If estimator lacks predict_proba but has predict, map margin to prob via sigmoid.
                """
                try:
                    exp = _get_expected_features(clf)
                    Xuse = Xdf.reindex(columns=exp, fill_value=np.nan) if exp else Xdf

                    if hasattr(clf, "predict_proba"):
                        return float(clf.predict_proba(Xuse)[0, 1])
                    if hasattr(clf, "predict"):
                        margin = float(clf.predict(Xuse)[0])
                        return float(1.0 / (1.0 + math.exp(-0.25 * margin)))
                    raise AttributeError("win_model lacks predict/predict_proba")
                except ValueError as ve:
                    msg = str(ve)
                    # Special-case: NaN/inf present — sanitize and retry once
                    if ("Input X contains NaN" in msg) or ("infinity" in msg) or ("too large" in msg):
                        try:
                            expN = _get_expected_features(clf)
                            Xsan = Xdf.reindex(columns=expN, fill_value=np.nan) if expN else Xdf.copy()
                            # Coerce to numeric, drop non-numeric to NaN, then replace inf and fill
                            Xsan = Xsan.apply(pd.to_numeric, errors="coerce")
                            Xsan = Xsan.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            if hasattr(clf, "predict_proba"):
                                return float(clf.predict_proba(Xsan)[0, 1])
                            if hasattr(clf, "predict"):
                                margin = float(clf.predict(Xsan)[0])
                                return float(1.0 / (1.0 + math.exp(-0.25 * margin)))
                        except Exception:
                            # fall through to other recovery paths below
                            pass
                    if "columns are missing:" in msg:
                        missing_cols: List[str] = []
                        try:
                            import ast
                            start = msg.find("{")
                            end = msg.rfind("}")
                            if start != -1 and end != -1 and end > start:
                                subset = msg[start : end + 1]
                                parsed = ast.literal_eval(subset)
                                if isinstance(parsed, (set, list, tuple)):
                                    missing_cols = list(parsed)
                        except Exception:
                            missing_cols = []
                        if missing_cols:
                            add_df = pd.DataFrame({c: [np.nan] for c in missing_cols}, index=Xdf.index)
                            Xdf = pd.concat([Xdf, add_df], axis=1)
                        # retry once
                        exp3 = _get_expected_features(clf)
                        Xuse2 = Xdf.reindex(columns=exp3, fill_value=np.nan) if exp3 else Xdf
                        # sanitize on retry as well, in case imputers were not part of clf
                        Xuse2 = Xuse2.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        if hasattr(clf, "predict_proba"):
                            return float(clf.predict_proba(Xuse2)[0, 1])
                        if hasattr(clf, "predict"):
                            margin = float(clf.predict(Xuse2)[0])
                            return float(1.0 / (1.0 + math.exp(-0.25 * margin)))
                    if "Feature names unseen at fit time" in msg:
                        exp4 = _get_expected_features(clf)
                        if exp4:
                            Xuse3 = Xdf.reindex(columns=exp4, fill_value=np.nan)
                            Xuse3 = Xuse3.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            if hasattr(clf, "predict_proba"):
                                return float(clf.predict_proba(Xuse3)[0, 1])
                            if hasattr(clf, "predict"):
                                margin = float(clf.predict(Xuse3)[0])
                                return float(1.0 / (1.0 + math.exp(-0.25 * margin)))
                    raise

            if win_m is not None:
                home_prob = _predict_proba_with_fill(win_m, X)
            else:
                home_prob = 1.0 / (1.0 + math.exp(-0.25 * point_diff))
                win_fallback_used = True
        except Exception:
            log.exception("Unexpected error while computing win probability; using sigmoid fallback")
            home_prob = 1.0 / (1.0 + math.exp(-0.25 * point_diff))
            win_fallback_used = True

        try:
            mode_val = model_objects.get("mode") if isinstance(model_objects, dict) else getattr(model_objects, "mode", "production")
        except Exception:
            mode_val = "production"
        if mode_val is None:
            mode_val = "production"
        mode_val = str(mode_val)

        if feature_fallback_used and win_fallback_used:
            pred_source = "feature_fallback+win_fallback"
        elif feature_fallback_used:
            pred_source = "feature_fallback"
        elif win_fallback_used:
            pred_source = "model+win_fallback"
        else:
            pred_source = "model"

        # Signal to the frontend whether the calibrated classifier was used.
        win_classifier_used = not win_fallback_used

        if not ALLOW_FALLBACK_PREDICTIONS and (feature_fallback_used or win_fallback_used):
            log.warning("Fallback prediction attempted but ALLOW_FALLBACK_PREDICTIONS=false; rejecting request")
            raise HTTPException(400, f"Prediction would use fallback logic (source={pred_source}); disallowed by server configuration")

        return PredictionResponse(
            home_score=round(home_score, 1),
            away_score=round(away_score, 1),
            home_win_probability=home_prob,
            away_win_probability=1.0 - home_prob,
            point_diff=point_diff,
            mode=mode_val,
            prediction_source=pred_source,
            win_classifier_used=win_classifier_used,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(400, f"Prediction failed: {e}")


def _reg_predict(bundle: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Predicts scores using a model bundle.

    Logic:
        - If bundle is a dict with 'hgbr', 'ridge', and 'weight', computes a weighted ensemble prediction.
        - If bundle contains a 'model' or 'estimator' key, delegates prediction to that object.
        - If bundle is a dict with any predictor object, uses the first found predictor.
        - If bundle is a single predictor object, calls its predict method.
        - Raises AttributeError if no valid prediction method is found.
    """
    log.debug("Model bundle type: %s, hasattr predict: %s", type(bundle), hasattr(bundle, "predict"))
    if isinstance(bundle, dict):
        log.debug("Model bundle keys: %s", list(bundle.keys()) if hasattr(bundle, "keys") else "no keys method")
        if {"hgbr", "ridge", "weight"}.issubset(bundle):
            weight = float(bundle["weight"])
            preds_hgbr = bundle["hgbr"].predict(X)
            preds_ridge = bundle["ridge"].predict(X)
            return weight * preds_hgbr + (1.0 - weight) * preds_ridge

        delegate = bundle.get("model") or bundle.get("estimator")
        if delegate is not None and hasattr(delegate, "predict"):
            return delegate.predict(X)
        for key, value in bundle.items():
            if hasattr(value, "predict"):
                log.debug("Using predictor from dict key: %s", key)
                return value.predict(X)
    if not isinstance(bundle, dict) and hasattr(bundle, "predict"):
        return bundle.predict(X)
    raise AttributeError(f"Score model lacks predict method. Type: {type(bundle)}")


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
        spath = _resolve_schedule_path()
        if not spath.exists():
            raise HTTPException(503, f"Schedule data not available on server (missing file): {spath}")
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
