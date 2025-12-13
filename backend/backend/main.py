"""
NFL ML Predictions API — Backend Server
========================================

FastAPI backend serving ML predictions for NFL game outcomes.

FILE METRICS:
    - Purpose: Serve NFL game predictions (scores + win probabilities) via a small, stable HTTP API.
    - Primary consumers: frontend/src/api/client.js and any CLI/test scripts.
    - Change philosophy: minimal, backwards-compatible changes; prefer adding new /api/* routes over breaking old ones.

KEY CONCEPTS:
    - "Raw features" vs "transformed features": datasets should match preprocessor.feature_names_in_ (raw columns).
    - CORS: ALLOWED_ORIGINS and ALLOW_ORIGIN_REGEX control which frontends can call this API.
    - Model bundles: artifacts may be pipelines (preprocess+model) or bare estimators depending on metadata.json.

LEARNING CHECKPOINTS:
    - You can explain how a /predict request becomes a 1-row DataFrame and then a model prediction.
    - You know where to change the dataset path and how schema validation works.

TIPS & NEXT STEPS:
    - For dashboards, prefer stable wrappers: /api/history and /api/games/next-week.
    - If predictions fail in production, check /health then confirm model artifacts exist under MODELS_DIR.


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
    METADATA_URL    → URL to metadata.json for dataset schema
    MODELS_DIR          → Path to model .joblib files (default: backend/data/prod-models/models)
    DATASET        → Path to engineered features CSV
    ALLOWED_ORIGINS     → Comma-separated CORS origins
    ALLOW_ORIGIN_REGEX  → Regex for dynamic CORS (e.g., https://.*/.vercel/.app)

ARCHITECTURE:
    Request → FastAPI Router → Feature Assembly → Preprocessor → ML Models → Response

    Models:
      - home_model.joblib: Predicts home team score
      - away_model.joblib: Predicts away team score
      - win_clf_calibrated.joblib: Calibrated win probability classifier
      - preprocessor.joblib: Feature transformation pipeline
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import nflreadpy as nfl
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from backend.utils import (coerce_season_week, normalize_team_columns, get_next_week_schedule, predict_game_outcome, load_dataset, load_model_objects, validate_dataset_schema, infer_raw_feature_columns, calculate_win_probability, find_fitted_column_transformer, numeric_input_columns_from_preprocessor, sanity_predict)  


# Load .env from backend/ (preferred) or project root (fallback).
# This matters because you often run uvicorn from the project root, so ".env" is not in CWD.
_ENV_CANDIDATES = [
    Path(__file__).resolve().parent / ".env",
    Path.cwd() / ".env",
]
for _p in _ENV_CANDIDATES:
    if _p.exists():
        load_dotenv(_p)
        break
else:
    # No .env found (common in production). Environment variables should already be set.
    load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Tip: Override paths via environment variables for different deployment targets

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global state containers — initialized in lifespan() at startup
model_objects: Optional[Dict[str, Any]] = None
dataset_df: Optional[pd.DataFrame] = pd.DataFrame()

# Path resolution
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"

# Model directory: where .joblib files live
# Educational: Using env var allows same code to work locally and on Heroku
# Production models are in backend/prod-models/models
# Production models are typically under backend/data/heroku-models/models (this repo's layout).
# We also support the legacy location backend/prod-models/models.
_default_models_dir = DATA_DIR / "heroku-models" / "models"
if not _default_models_dir.exists():
    _default_models_dir = BACKEND_DIR / "heroku-models" / "models"

MODELS_DIR = Path(os.getenv("MODELS_DIR", str(_default_models_dir))).resolve()
# Dataset path: engineered features CSV for predictions
DEFAULT_DATASET = BACKEND_DIR / "data/heroku-models/game_features_20251218.csv"

# Default local schedule CSV fallback (used when nflreadpy is unavailable/offline).
DEFAULT_SCHEDULE_PATH = DATA_DIR / "Nfl_schedule_2025.csv"

# Metadata path: model metadata JSON
PROD_MODELS_PATH = Path(
    os.getenv(
        "PROD_MODELS_PATH",
        str(MODELS_DIR / "metadata.json"),
    )
)

def _parse_allowed_origins(raw: str) -> List[str]:
    """Parse env string into a clean allow list with safe defaults."""
    items = []
    for part in raw.split(","):
        origin = part.strip().rstrip("/")
        if origin:
            items.append(origin)
    if not items:
        items = [
            "https://nfl-ml-predictions.vercel.app",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    return items


ALLOWED_ORIGINS = _parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))
ALLOW_ORIGIN_REGEX = os.getenv("ALLOW_ORIGIN_REGEX", r"https://.*/.vercel/.app")
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "false").lower() == "true"
FRONTEND_BUILD = BACKEND_DIR.parent / "frontend" / "build"
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"
ALLOW_FALLBACK_PREDICTIONS = os.getenv("ALLOW_FALLBACK_PREDICTIONS", "true").lower() == "true"

TEAM_CODE_FIX = {
    "WSH": "WAS",
    "HST": "HOU",
    "CLV": "CLE",
    "BLT": "BAL",
    "ARZ": "ARI",
}

def to_team_abbr(t: str) -> str:
    """Normalize a team identifier into the abbreviation used by our models.

    Accepts:
        - Common abbreviations (e.g., "PHI")
        - Mixed case / whitespace (e.g., " phi ")
        - A few legacy codes (e.g., "CLV" → "CLE")

    Returns:
        str: Uppercased, cleaned team abbreviation.
    """
    if t is None:
        return ""
    cleaned = str(t).strip().upper()
    return TEAM_CODE_FIX.get(cleaned, cleaned)

def resolve_model_path(key: str, filename: str) -> Path:
    env_val = os.getenv(f"MODEL_PATH_{key.upper()}")
    if env_val:
        return Path(env_val)
    return MODELS_DIR / filename

def _normalize_feature_cols(raw: Any) -> List[str]:
    """Normalize raw_feature_columns metadata into a flat list of feature names.

    Supports:
      - dict form: {"numeric": [...], "categorical": [...]}
      - sequence / pandas Index of column names
      - single value fallback

    Returns:
        List[str]: feature names as strings.
    """
    if raw is None:
        return []

    # Dict form from metadata.json
    if isinstance(raw, dict):
        cols: List[str] = []
        for key in ("numeric", "categorical"):
            vals = raw.get(key)
            if isinstance(vals, (list, tuple, set, np.ndarray, pd.Index)):
                cols.extend([str(c) for c in vals])
        return cols

    # Sequence / Index form
    if isinstance(raw, (list, tuple, set, np.ndarray, pd.Index)):
        return [str(c) for c in raw]

    # Fallback: treat as single column name
    return [str(raw)]

def load_objects() -> Dict[str, Any]:
    """Load model artifacts + metadata.

    Contract:
      - MODELS_DIR contains metadata.json and joblib artifacts.
      - We attach *raw input* column expectations to model_objects so startup validation
        can compare against the dataset CSV without confusing transformed names like
        'num__...' produced by ColumnTransformer.

    Returns:
        Dict[str, Any]: model_objects with loaded artifacts.
    """
    models_dir = MODELS_DIR
    meta_path = models_dir / "metadata.json"
    if not meta_path.is_file():
        contents = [p.name for p in models_dir.glob("*")] if models_dir.exists() else []
        raise FileNotFoundError(
            f"Metadata file not found. Expected: {meta_path}\n"
            f"Folder contents: {contents}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    artifacts = meta.get("artifacts", {}) or {}

    # ✅ Load required files (these variables MUST exist before returning)
    preprocessor = joblib.load(models_dir / artifacts.get("preprocessor", "preprocessor.joblib"))
    home_model   = joblib.load(models_dir / artifacts.get("home_model", "home_model.joblib"))
    away_model   = joblib.load(models_dir / artifacts.get("away_model", "away_model.joblib"))
    win_clf      = joblib.load(models_dir / artifacts.get("win_clf", "win_clf_calibrated.joblib"))

    # Optional
    hist_win_path = models_dir / artifacts.get("hist_win_clf", "hist_win_clf_calibrated.joblib")
    hist_win_clf = joblib.load(hist_win_path) if hist_win_path.is_file() else None

    # Raw input schema: prefer metadata, then preprocessor.feature_names_in_
    raw_feature_columns = (
        meta.get("raw_feature_columns")
        or meta.get("raw_columns")
        or meta.get("feature_columns")  # tolerate older metadata key
        or {}
    )

    feature_names_in = []
    try:
        if hasattr(preprocessor, "feature_names_in_"):
            feature_names_in = list(preprocessor.feature_names_in_)
        elif hasattr(preprocessor, "named_steps"):
            for step in preprocessor.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    feature_names_in = list(step.feature_names_in_)
                    break
    except Exception:
        feature_names_in = []

    model_objects = {
        "models_dir": str(models_dir),
        "meta": meta,
        "artifacts": artifacts,
        "raw_feature_columns": raw_feature_columns,
        "feature_names_in": feature_names_in,
        "preprocessor": preprocessor,
        "home_model": home_model,
        "away_model": away_model,
        "win_clf": win_clf,
        "hist_win_clf": hist_win_clf,
    }
    return model_objects
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
        # Calculate overlap percentage to determine severity
        overlap_pct = (len(expected) - len(missing)) / len(expected) * 100 if expected else 0
        log.warning(
            "Dataset schema mismatch: %d missing required raw input features (%.0f%% overlap). Sample: %s",
            len(missing), overlap_pct, missing[:10]
        )
        # Only raise if overlap is critically low (< 50%); otherwise warn and continue
        if overlap_pct < 50:
            raise RuntimeError(
                f"Dataset missing required raw input features for models: {missing[:20]}. "
                "Run the feature engineering pipeline or point DATASET_PATH to the correct file."
            )


def _infer_raw_feature_columns(model_objects: Dict[str, Any], df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    """Best-effort inference for *raw input* feature columns.

    Why this exists:
      - Some older metadata.json files don't include raw_feature_columns.
      - ColumnTransformer.get_feature_names_out() returns *transformed* names like
        'num__...' which do NOT exist in the raw CSV. Using those causes false schema
        mismatches and broken sanity checks.

    Priority:
      1) raw_feature_columns (metadata) if present
      2) preprocessor.feature_names_in_ (raw input columns)
      3) dataset-driven inference (numeric vs categorical)

    Returns:
        {"numeric": [...], "categorical": [...]} (lists may be empty)
    """
    if not isinstance(model_objects, dict):
        return {"numeric": [], "categorical": []}

    raw = model_objects.get("raw_feature_columns") or {}
    if isinstance(raw, dict) and (raw.get("numeric") or raw.get("categorical")):
        return {
            "numeric": [str(c) for c in raw.get("numeric", [])],
            "categorical": [str(c) for c in raw.get("categorical", [])],
        }

    # Legacy form: some runs saved a flat list of columns.
    if isinstance(raw, (list, tuple, set, np.ndarray, pd.Index)) and raw:
        cols = [str(c) for c in raw]
        if df is None or df.empty:
            return {"numeric": cols, "categorical": []}
        numeric: List[str] = []
        categorical: List[str] = []
        for col in cols:
            if (
                col in df.columns
                and pd.api.types.is_numeric_dtype(df[col])
                and not pd.api.types.is_object_dtype(df[col])
            ):
                numeric.append(col)
            else:
                categorical.append(col)
        return {"numeric": numeric, "categorical": categorical}

    # Prefer fitted raw input columns from the preprocessor.
    pre = model_objects.get("preprocessor")
    try:
        cols: Optional[List[str]] = None
        if pre is not None:
            if hasattr(pre, "feature_names_in_"):
                cols = [str(c) for c in getattr(pre, "feature_names_in_")]
            elif hasattr(pre, "named_steps"):
                for step in pre.named_steps.values():
                    if hasattr(step, "feature_names_in_"):
                        cols = [str(c) for c in getattr(step, "feature_names_in_")]
                        break

        if cols:
            if df is None or df.empty:
                return {"numeric": cols, "categorical": []}
            numeric = []
            categorical = []
            for col in cols:
                if (
                    col in df.columns
                    and pd.api.types.is_numeric_dtype(df[col])
                    and not pd.api.types.is_object_dtype(df[col])
                ):
                    numeric.append(col)
                else:
                    categorical.append(col)
            return {"numeric": numeric, "categorical": categorical}
    except Exception:
        pass

    # Dataset-driven inference (last resort).
    if df is None or df.empty:
        return {"numeric": [], "categorical": []}

    numeric = []
    categorical = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
            numeric.append(str(col))
        else:
            categorical.append(str(col))
    return {"numeric": numeric, "categorical": categorical}


def _calculate_win_probability(
    win_model: Any,
    X: pd.DataFrame,
    h_score: float,
    a_score: float,
) -> Tuple[float, bool]:
    """
    Safely compute home-team win probability.

    If the classifier is available and supports predict_proba, use it.
    Otherwise fall back to a logistic transform of the point differential.
    """
    clf_used = False

    if hasattr(win_model, "predict_proba"):
        try:
            win_prob = float(win_model.predict_proba(X)[0][1])
            clf_used = True
            return win_prob, clf_used
        except Exception as clf_err:  # pragma: no cover - defensive path
            logging.warning(
                "[Predict] win_clf predict_proba failed, falling back: %s", clf_err
            )
            try:
                pre_X = load_objects[MODELS_DIR]["preprocessor"].transform(X)
                win_prob = float(win_model.predict_proba(pre_X)[0][1])
                clf_used = True
                return win_prob, clf_used
            except Exception as pre_err:  # pragma: no cover - defensive path
                logging.warning(
                    "[Predict] win_clf predict_proba failed after preprocessing, falling back: %s", pre_err
                )

    diff = h_score - a_score
    win_prob = float(1.0 / (1.0 + np.exp(-0.3 * diff)))
    return win_prob, clf_used


# -------------------------------------------------------------------
# App State
# -------------------------------------------------------------------


class AppState:
    """
    In-memory state container for:
      - dataset : feature DataFrame used for prediction
      - models  : dict of trained models (home / away / win)
      - history : list of prediction results returned to clients
    """

    def __init__(self) -> None:
        self.dataset: Optional[pd.DataFrame] = None
        self.models: Dict[str, Any] = {}
        # Optional shared preprocessor artifact (may be saved separately from models)
        self.preprocessor: Optional[Any] = None
        self.history: List[Dict[str, Any]] = []

    # -------------------------
    # Startup Loader
    # -------------------------
    def load(self) -> None:
        """Load dataset + models at startup with defensive logging."""
        self._load_dataset()
        self._load_models()

    def _load_dataset(self) -> None:
        """Load the most recent game_features*.csv into memory."""
        try:
            path: Path
            # Fallback default filename under data dir
            path = DATA_DIR / "heroku-models/game_features_20251218.csv"

            # If file missing and a DATA_URL is configured, attempt to download it
            if not path.exists():
                meta_url = os.environ.get("METADATA_URL")
                if meta_url:
                    logging.info("[Dataset] Attempting to find metadata.json: %s", meta_url)
                    try:
                        meta = pd.read_json(meta_url)
                        meta_feature_columns = _normalize_feature_cols(meta["feature_columns"])
                        logging.info("[Dataset] Downloaded metadata.json from %s to %s", meta_url, path)
                        if meta_feature_columns:
                            DATASET = os.environ.get("DATASET", DEFAULT_DATASET)
                            df = pd.read_csv(DATASET)
                            self.dataset = df
                            return self.dataset

                    except Exception as e:
                        logging.warning("[Dataset] Failed to download or parse metadata.json from %s: %s", meta_url, e)
                else:
                    logging.warning("[Dataset] No dataset found at: %s and no METADATA_URL configured", path)
                    self.dataset = pd.read_csv(DEFAULT_DATASET)
                    return

            logging.info("[Dataset] Loading dataset from: %s", path)
            df = pd.read_csv(DATASET)

            # Normalize key columns for consistent lookups
            df = _coerce_season_week(df)
            df = _normalize_team_columns(df)

            self.dataset = df
            logging.info("[Dataset] Loaded %d rows from %s", len(df), path)
        except Exception as e:  # pragma: no cover - defensive
            logging.exception("[Dataset] Error while loading dataset: %s", e)
            self.dataset = None

    def _load_models(self) -> None:
        """Load each required model independently."""
        self.models = {}
        # Default model filenames (non-pipeline artifacts)
        model_files: Dict[str, str] = {
            "home": "home_model.joblib",
            "away": "away_model.joblib",
            "win": "win_clf_calibrated.joblib",
        }

        for name, filename in model_files.items():
            # Prefer a full pipeline artifact (e.g. home_pipe.joblib) if present.
            pipe_filename = f"{name}_pipe.joblib" if name != "win" else "win_pipe.joblib"
            pipe_path = MODELS_DIR / pipe_filename
            model_path = MODELS_DIR / filename

            # Choose the path to load
            if pipe_path.exists():
                path = pipe_path
                logging.info("[Model] Found pipeline artifact for '%s' at %s; preferring it", name, path)
            else:
                path = model_path
                logging.info("[Model] Loading model '%s' from: %s", name, path)

            if not path.exists():
                logging.warning("[Model] Missing model file for '%s': %s", name, path)
                continue

            try:
                loaded = joblib.load(path)
                self.models[name] = loaded
                logging.info("[Model] Loaded '%s' successfully from %s", name, path)
                # If this happens to be a sklearn Pipeline, log its steps for visibility
                try:
                    from sklearn.pipeline import Pipeline

                    if isinstance(loaded, Pipeline):
                        steps = list(getattr(loaded, "named_steps", {}).keys())
                        logging.info("[Model] Pipeline '%s' steps: %s", name, steps)
                except Exception:
                    # sklearn may not be available in some analysis contexts; ignore
                    pass
            except Exception as e:  # pragma: no cover - defensive
                logging.exception("[Model] Error loading '%s' from %s: %s", name, path, e)

        logging.info("[Model] Loaded model keys: %s", list(self.models.keys()))
        # Attempt to load a standalone preprocessor artifact if present. Some
        # training runs save the fitted ColumnTransformer / preprocessor as
        # `preprocessor.joblib` under the same models directory. If available,
        # load it so the serving code can use it to impute / transform raw
        # DataFrame rows prior to calling models that expect preprocessed arrays.
        prep_path = MODELS_DIR / "preprocessor.joblib"
        if prep_path.exists():
            try:
                self.preprocessor = joblib.load(prep_path)
                logging.info("[Model] Loaded standalone preprocessor from %s", prep_path)
            except Exception as e:  # pragma: no cover - defensive
                logging.exception("[Model] Failed to load preprocessor %s: %s", prep_path, e)
                self.preprocessor = None
        else:
            logging.info("[Model] No standalone preprocessor found at %s", prep_path)
        # If we loaded a standalone preprocessor and some models are Pipelines
        # that contain an unfitted preprocessor step (ColumnTransformer), try
        # to patch the pipeline to use the standalone preprocessor. This can
        # resolve situations where the serialized pipeline referenced an
        # unfitted transformer in CI/CD packaging.
        if self.preprocessor is not None:
            try:
                # Import sklearn classes lazily so packaging without sklearn
                # won't fail at module import time.
                from sklearn.pipeline import Pipeline
                from sklearn.compose import ColumnTransformer
            except Exception:
                Pipeline = None  # type: ignore
                ColumnTransformer = None  # type: ignore

            if Pipeline is not None:
                for mname, model in list(self.models.items()):
                    try:
                        if isinstance(model, Pipeline):
                            steps = list(model.steps)
                            replaced = False
                            for i, (step_name, step_est) in enumerate(steps):
                                lname = (step_name or "").lower()
                                if lname in ("pre", "prep", "preprocessor", "preprocess"):
                                    # If the pipeline's preprocessor is a ColumnTransformer
                                    # but not fitted (no 'transformers_' attribute), replace
                                    # it with the standalone preprocessor we loaded.
                                    if (
                                        ColumnTransformer is not None
                                        and isinstance(step_est, ColumnTransformer)
                                        and not hasattr(step_est, "transformers_")
                                    ):
                                        steps[i] = (step_name, self.preprocessor)
                                        replaced = True
                                        logging.info(
                                            "[Model] Replacing unfitted preprocessor step '%s' in pipeline '%s' with standalone preprocessor",
                                            step_name,
                                            mname,
                                        )
                                        break
                            if replaced:
                                try:
                                    from sklearn.pipeline import Pipeline as SKPipeline

                                    new_pipe = SKPipeline(steps)
                                    self.models[mname] = new_pipe
                                    logging.info("[Model] Patched pipeline model '%s' successfully", mname)
                                except Exception as e:
                                    logging.exception("[Model] Failed to rebuild pipeline for %s: %s", mname, e)
                    except Exception:
                        # Be defensive: do not fail model loading due to pipeline patching
                        logging.exception("[Model] Error while attempting to inspect/patch model %s", mname)

def _find_fitted_column_transformer(pre: Any) -> Optional[Any]:
    """Return the fitted ColumnTransformer-like object inside `pre`, if present.

    We avoid importing sklearn types here; we just look for the fitted attribute `transformers_`,
    which is present after fitting a ColumnTransformer.
    """
    if pre is None:
        return None

    # Common case: preprocessor IS the fitted ColumnTransformer
    if hasattr(pre, "transformers_"):
        return pre

    # Common case: preprocessor is a Pipeline with a fitted ColumnTransformer step
    if hasattr(pre, "named_steps"):
        for step in pre.named_steps.values():
            if hasattr(step, "transformers_"):
                return step

    return None


def _numeric_input_columns_from_preprocessor(pre: Any) -> List[str]:
    """Best-effort extraction of *raw input* numeric columns from a fitted preprocessor.

    Why this exists:
      - Your numeric pipeline uses a SimpleImputer(strategy='median'), which requires numeric dtype.
      - CSVs sometimes contain whitespace strings ("   ") in numeric fields, causing transform() to crash.

    We find the numeric column list from the fitted ColumnTransformer by:
      - transformer name heuristics (num/numeric),
      - or detecting an imputer step with strategy in {median, mean}.

    Returns:
        Sorted unique list of column names.
    """
    ct = _find_fitted_column_transformer(pre)
    if ct is None:
        return []

    cols: List[str] = []
    for name, transformer, colspec in getattr(ct, "transformers_", []) or []:
        if colspec is None or colspec == "drop":
            continue

        # Only handle explicit column lists (names); slices/callables are ignored safely.
        if isinstance(colspec, slice) or callable(colspec):
            continue

        try:
            col_list = list(colspec)
        except TypeError:
            continue

        lname = str(name).lower()
        is_numeric = ("num" in lname) or ("numeric" in lname)

        # If name isn't clear, inspect a Pipeline for an imputer strategy
        if not is_numeric and hasattr(transformer, "named_steps"):
            imp = transformer.named_steps.get("imputer")
            if imp is not None and getattr(imp, "strategy", None) in ("median", "mean"):
                is_numeric = True

        if is_numeric:
            cols.extend([str(c) for c in col_list])

    # de-dupe + stable order
    return sorted(set(cols))


def _sanity_predict(model_objects: Dict[str, Any], df: pd.DataFrame) -> None:
    """Run a tiny prediction at startup to prove the serving stack is wired correctly.

    Goal: catch issues early (bad deserialization, schema drift, transform/predict mismatch).

    Important detail:
      - The *preprocessor* was fit on RAW columns (feature_names_in_).
      - The regressors/classifier usually expect the transformed matrix (output of preprocessor.transform).

    We purposely build a 1-row DataFrame with the exact raw columns the preprocessor expects,
    then transform + predict. If anything fails, we raise a RuntimeError with a compact summary.
    lifespan() catches it and logs (so the server can still boot in a degraded state if you want).
    """
    failures: List[str] = []

    pre = model_objects.get("preprocessor")
    home_m = model_objects.get("home_model")
    away_m = model_objects.get("away_model")
    win_m = model_objects.get("win_clf") or model_objects.get("win_model")  # tolerate legacy key

    if pre is None or home_m is None or away_m is None:
        raise RuntimeError("Sanity-predict aborted: missing preprocessor/home_model/away_model")

    if df is None or df.empty:
        raise RuntimeError("Sanity-predict aborted: dataset is empty (cannot build a sample row)")

    # 1) Determine expected RAW columns
    raw_cols = model_objects.get("feature_names_in") or []
    if not raw_cols:
        inferred = _infer_raw_feature_columns(model_objects, df)
        raw_cols = _normalize_feature_cols(inferred)

    if not raw_cols:
        raise RuntimeError("Sanity-predict aborted: could not infer raw input columns for preprocessor")

    sample = df.iloc[0].to_dict()

    # 2) Build a 1-row raw frame (missing values become NaN; preprocessor should handle this)
    X_raw = pd.DataFrame([{c: sample.get(c, np.nan) for c in raw_cols}], columns=raw_cols)

    # 3) Transform (unless the model is a full pipeline)
    def _is_pipeline(obj: Any) -> bool:
        return hasattr(obj, "named_steps")

    transformed = None
    if not _is_pipeline(home_m) or not _is_pipeline(away_m) or (win_m and not _is_pipeline(win_m)):
        try:
            transformed = pre.transform(X_raw)
        except Exception as e:
            failures.append(f"preprocessor.transform failed: {type(e).__name__}: {e}")
            # Try a safer variant
            try:
                X_safe = X_raw.fillna(0).replace([np.inf, -np.inf], 0)
                transformed = pre.transform(X_safe)
            except Exception as e2:
                failures.append(f"preprocessor.transform retry failed: {type(e2).__name__}: {e2}")

    # 4) Predict (support both raw and transformed depending on artifact type)
    def _predict(model: Any, X_raw_df: pd.DataFrame, X_tx: Any, label: str) -> Optional[float]:
        try:
            X_in = X_raw_df if _is_pipeline(model) else X_tx
            if X_in is None:
                raise TypeError("transformed features missing")
            return float(model.predict(X_in)[0])
        except Exception as e:
            failures.append(f"{label} predict failed: {type(e).__name__}: {e}")
            return None

    _ = _predict(home_m, X_raw, transformed, "home_model")
    _ = _predict(away_m, X_raw, transformed, "away_model")

    # Win probability is optional (some builds only ship regressors)
    if win_m is not None:
        try:
            X_in = X_raw if _is_pipeline(win_m) else transformed
            if X_in is None:
                raise TypeError("transformed features missing")
            if hasattr(win_m, "predict_proba"):
                _ = float(win_m.predict_proba(X_in)[0, 1])
            else:
                _ = float(win_m.predict(X_in)[0])
        except Exception as e:
            failures.append(f"win_clf predict failed: {type(e).__name__}: {e}")

    if failures:
        raise RuntimeError("Startup sanity-predict failed: " + "; ".join(failures))
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

    # Load dataset
    dataset_df = _load_and_validate_dataset(model_objects)

    log.info("=" * 60)
    log.info("STARTUP COMPLETE")
    log.info("Models: %s", "✓ Loaded" if model_objects else "✗ Missing")
    log.info("Dataset: %s", "✓ Loaded" if dataset_df is not None and not dataset_df.empty else "✗ Missing")
    log.info("=" * 60)

    try:
        yield
    finally:
        log.info("SHUTDOWN: Cleaning up resources")

def _load_and_validate_dataset(models: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Centralized logic to load, clean, and validate the dataset.
    """
    ds_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    log.info("Dataset path: %s", ds_path)

    if not ds_path.exists():
        log.warning("✗ Dataset not found at %s", ds_path)
        # Check alternate locations
        alternates = (
            DATA_DIR / "production_inference.csv",  # <--- NEW: High priority
            DEFAULT_DATASET,
            BACKEND_DIR / "data" / "prod-models" / "game_features_20251210.csv",
            DATA_DIR / "game_features_20251210.csv",
        )
        for alt in alternates:
            if alt.exists():
                log.info("Found alternate dataset: %s", alt)
                ds_path = alt
                break
        else:
            log.warning("No dataset found; predictions will use synthetic features only")
            return pd.DataFrame()

    try:
        df = pd.read_csv(ds_path)
        if df.empty:
            log.warning("Dataset CSV is empty")
            return pd.DataFrame()

        df.columns = [c.strip() for c in df.columns]
        df = _ensure_home_away(df)

        # 1) Treat whitespace-only strings as missing values
        df = df.replace(r"^\s*$", np.nan, regex=True)

        # 2) Ensure ALL raw columns exist (add as NaN if missing)
        if models:
            required_raw = models.get("feature_names_in") or []
            for c in required_raw:
                if c not in df.columns:
                    df[c] = np.nan

            # 3) Coerce numeric raw inputs
            pre = models.get("preprocessor")
            num_cols = _numeric_input_columns_from_preprocessor(pre)
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

        # Validate schema
        try:
            if models:
                _validate_dataset_schema(df, models)
        except Exception as e:
            log.warning("Dataset schema validation failed: %s", e)

        # Sanity check
        try:
            if models:
                _sanity_predict(models, df)
                log.info("✓ Sanity prediction passed")
        except Exception as e:
            log.warning("Sanity prediction failed: %s; continuing", e)

        log.info("✓ Dataset loaded: %d rows, %d columns", len(df), df.shape[1])
        return df

    except Exception as e:
        log.error("Failed to load dataset: %s", e, exc_info=True)
        return pd.DataFrame()

# Define the FastAPI application and CORS middleware BEFORE using @app.* decorators or app.mount.
app = FastAPI(
    title="NFL ML Predictions API",
    version="2.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # If you sometimes spin up preview deployments on Vercel:
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=False,                   # if you send cookies/auth
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],                      # or list explicitly if you prefer
    expose_headers=["*"],                     # optional: if you need to read custom headers
)


@app.options("/{rest_of_path:path}")
async def preflight_ok(rest_of_path: str) -> Response:
    """Return 200 for any OPTIONS route so clients never see a 400 preflight."""
    return Response(status_code=200)


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
    """Request payload for `/predict`.

    The backend is intentionally flexible: it accepts both snake_case and
    camelCase keys so the frontend can stay ergonomic.

    Accepted JSON shapes:
        1) { "home_team": "PHI", "away_team": "DAL", "season": 2025, "week": 15 }
        2) { "homeTeam": "PHI", "awayTeam": "DAL", "season": 2025, "week": 15 }
    """

    home_team: str = Field(..., alias="homeTeam", description="Home team abbreviation (e.g., PHI)")
    away_team: str = Field(..., alias="awayTeam", description="Away team abbreviation (e.g., DAL)")
    season: int
    week: int

    class Config:
        allow_population_by_field_name = True
        anystr_strip_whitespace = True


class PredictionResponse(BaseModel):
    home_score: float
    away_score: float
    home_win_probability: float
    away_win_probability: float
    point_diff: float
    mode: str
    prediction_source: str
    win_classifier_used: bool


class HistoryEntry(BaseModel):
    """A single row for the history API.

    Note: History values come from the dataset CSV, not from live inference.
    """
    season: int
    week: int
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    winner: Optional[str] = None


class HistoryResponse(BaseModel):
    """Frontend-friendly wrapper: { entries: [...], total, limit }."""
    entries: List[HistoryEntry]
    total: int
    limit: int


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

class FullSchedule(BaseModel):
    full_schedule: str  # JSON string of the DataFrame
    ScheduleGame: List[ScheduleGame]

class NextWeekGamesResponse(BaseModel):
    """Frontend-friendly wrapper: { games: [...] }.

    This keeps older clients working (they often expect `payload.games`).
    """
    games: List[ScheduleGame]




def _glob_latest(dir_path: Path, pattern: str) -> Optional[Path]:
    try:
        matches = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None
    except Exception:
        return None


def _get_schedule_df(season: int = 2025) -> pd.DataFrame:
    """
    Centralized schedule loader.

    Strategy:
    1. Try nflreadpy (external API) for the freshest data.
    2. Fallback to local CSV if API fails.
    3. Return empty DataFrame if both fail.

    Educational:
    - Failing gracefully (fallback) is critical for high-availability systems.
    - Centralizing this logic triggers cleaner code in multiple endpoints.
    """
    # 1. Try external API
    try:
        # Check if we should skip API calls in dev/offline mode (optional optimization)
        if os.getenv("OFFLINE_MODE", "false").lower() == "true":
             pass # Skip to fallback
        else:
            log.info("Fetching schedule from nflreadpy for season %d...", season)
            df = nfl.load_schedules(season)
            df = df.to_pandas()
            if not df.empty:
                return df
    except Exception as e:
        log.warning("nflreadpy fetch failed or blocked: %s. Attempting local fallback.", e)

    # 2. Try local fallback
    path = _resolve_schedule_path()
    if path and path.exists():
        log.info("Loading schedule from local fallback: %s", path)
        try:
            return pd.read_csv(path)
        except Exception as e:
            log.error("Failed to read local schedule CSV: %s", e)

    # 3. Give up
    log.error("Could not load schedule from any source.")
    return pd.DataFrame()


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
    now = datetime.now(timezone.utc)
    log.info("Current datetime: %s", now)

    # Current season/week (best effort)
    try:
        current_season = int(nfl.get_current_season(False))
    except Exception:
        current_season = now.year if now.month >= 8 else now.year - 1

    try:
        current_week = int(nfl.get_current_week(False))
    except Exception:
        current_week = 1

    # Prefer dataset-driven "last completed" (it best reflects what we trained/served on).
    last_completed_season = current_season
    last_completed_week = max(current_week - 1, 0)

    try:
        if dataset_df is not None and not dataset_df.empty and {"season", "week"}.issubset(dataset_df.columns):
            completed_mask = pd.Series(True, index=dataset_df.index)

            # If score columns exist, require both to be present.
            score_cols = []
            if "home_points_for" in dataset_df.columns:
                score_cols.append("home_points_for")
            if "away_points_for" in dataset_df.columns:
                score_cols.append("away_points_for")
            for c in score_cols:
                completed_mask &= pd.to_numeric(dataset_df[c], errors="coerce").notna()

            # Winner is often present for completed games; if available, use it.
            if "winner" in dataset_df.columns:
                completed_mask &= dataset_df["winner"].notna()

            completed = dataset_df.loc[completed_mask]
            if not completed.empty:
                last_row = completed.sort_values(by=["season", "week"]).iloc[-1]
                last_completed_season = int(last_row["season"])
                last_completed_week = int(last_row["week"])
    except Exception as e:
        log.debug("Dataset-driven context inference failed: %s", e)

    # Fallback: infer last completed from schedule scores, if dataset is unavailable.
    if dataset_df is None or dataset_df.empty:
        try:
            schedule_df = _get_schedule_df(current_season)
            if not schedule_df.empty and {"season", "week"}.issubset(schedule_df.columns):
                completed_mask = pd.Series(True, index=schedule_df.index)
                if "home_score" in schedule_df.columns:
                    completed_mask &= pd.to_numeric(schedule_df["home_score"], errors="coerce").notna()
                if "away_score" in schedule_df.columns:
                    completed_mask &= pd.to_numeric(schedule_df["away_score"], errors="coerce").notna()

                completed = schedule_df.loc[completed_mask]
                if not completed.empty:
                    last_row = completed.sort_values(by=["season", "week"]).iloc[-1]
                    last_completed_season = int(last_row["season"])
                    last_completed_week = int(last_row["week"])
        except Exception as e:
            log.debug("Schedule-driven context inference failed: %s", e)

    nxt_s, nxt_w = last_completed_season, last_completed_week + 1
    if nxt_w > 22:
        nxt_s, nxt_w = last_completed_season + 1, 1

    status = "preseason_or_early" if last_completed_week < 1 else ("nfl_season_active" if nxt_s == current_season else "offseason")
    return {
        "current_season": current_season,
        "last_completed_season": last_completed_season,
        "last_completed_week": last_completed_week,
        "next_prediction_season": nxt_s,
        "next_prediction_week": nxt_w,
        "status": status,
    }




def _roll_forward_last_game_stats(df: pd.DataFrame, team: str, season: int, week: int, side: str = "home") -> Dict[str, Any]:
    """
    Roll forward rolling/prior stats from the most recent completed game for a specific team.

    This function dynamically fills in stats for future/unplayed games by copying the last
    known values from the team's most recent game. Used when predicting future games where
    rolling stats are 0 or NaN because the game hasn't been played yet.

    Args:
        df: Dataset with historical game data
        team: Team abbreviation (e.g., 'KC', 'BUF')
        season: Target prediction season
        week: Target prediction week
        side: Either 'home' or 'away' to determine which column prefixes to use

    Returns:
        Dictionary of rolled-forward stats with column names as keys

    Example:
        When predicting KC vs LAC for Week 15, and KC's last game was Week 14:
        - Takes Week 14 rolling averages (pf, pa, win_pct for windows 3, 5, 10)
        - Returns them for use in Week 15 prediction
        - These values are NOT saved to the dataset, only used for this prediction
    """
    stats = {}

    try:
        # Find all games where this team played (either home or away) before the target week
        team_mask = ((df["home_team"] == team) | (df["away_team"] == team)) & \
                    ((df["season"] < season) | ((df["season"] == season) & (df["week"] < week)))

        team_history = df.loc[team_mask].sort_values(by=["season", "week"], ascending=False)

        if team_history.empty:
            log.debug(f"No history found for {team} before {season} W{week}")
            return stats

        # Get the most recent game
        last_game = team_history.iloc[0]

        # Determine which columns to pull based on whether team was home or away in that game
        was_home_in_last = (last_game["home_team"] == team)
        last_side = "home" if was_home_in_last else "away"

        # Define the rolling stats we want to roll forward
        rolling_cols = [
            f"{last_side}_rolling_pf_3",
            f"{last_side}_rolling_pa_3",
            f"{last_side}_rolling_win_pct_3",
            f"{last_side}_rolling_pf_5",
            f"{last_side}_rolling_pa_5",
            f"{last_side}_rolling_win_pct_5",
            f"{last_side}_rolling_pf_10",
            f"{last_side}_rolling_pa_10",
            f"{last_side}_rolling_win_pct_10",
            f"{last_side}_prior_pf_avg_3",
            f"{last_side}_prior_pa_avg_3",
            f"{last_side}_prior_win_pct_3",
            f"{last_side}_prior_pf_avg_5",
            f"{last_side}_prior_pa_avg_5",
            f"{last_side}_prior_win_pct_5",
        ]

        # Roll forward the stats, mapping from last game's side to current prediction side
        for col in rolling_cols:
            if col in last_game.index and not pd.isna(last_game[col]):
                # Map to the current side (home/away for this prediction)
                target_col = col.replace(last_side, side)
                stats[target_col] = float(last_game[col])
                log.debug(f"Rolled forward {col}={last_game[col]:.2f} -> {target_col} for {team}")

        log.info(f"✓ Rolled forward {len(stats)} stats for {team} from {int(last_game['season'])} W{int(last_game['week'])}")

    except Exception as e:
        log.warning(f"Failed to roll forward stats for {team}: {e}")

    return stats


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
        m = (
    ((local["home_team"] == team) | (local["away_team"] == team))
    & (local["season"] == season)  # example next condition
)

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

    # NEW: Roll forward stats from last game if we don't have recent data
    # This handles future/unplayed games where rolling stats would be 0/NaN
    home_rolled = _roll_forward_last_game_stats(local, home, season, week, "home")
    away_rolled = _roll_forward_last_game_stats(local, away, season, week, "away")

    # Merge rolled-forward stats, but don't overwrite computed priors if they exist
    for k, v in home_rolled.items():
        if k not in home_feats or pd.isna(home_feats.get(k)):
            home_feats[k] = v

    for k, v in away_rolled.items():
        if k not in away_feats or pd.isna(away_feats.get(k)):
            away_feats[k] = v

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
      2) DEFAULT_SCHEDULE_PATH (backend/data/Nfl_schedule_*.csv)
      3) Latest matching file in backend/data/ by pattern 'Nfl_schedule_*.csv'

    Returns:
      Path to an existing file or DEFAULT_SCHEDULE_PATH even if not present (caller may 404).
    """
    env_val = os.getenv("SCHEDULE_PATH")
    env_path = Path(env_val.strip()) if env_val and env_val.strip() else None
    try:
        if env_path and env_path.exists():
            log.info("Using schedule from SCHEDULE_PATH=%s", env_path)
            return env_path
    except Exception:
        pass

    if DEFAULT_SCHEDULE_PATH.exists():
        log.info("Using default schedule at %s", DEFAULT_SCHEDULE_PATH)
        return DEFAULT_SCHEDULE_PATH

    latest = _glob_latest(DATA_DIR, "Nfl_schedule_*.csv")
    if latest and latest.exists():
        log.info("Using latest schedule candidate at %s", latest)
        return latest

    # As a last resort, return DEFAULT_SCHEDULE_PATH (may not exist); caller will handle
    log.warning(
        "No schedule file found; returning DEFAULT_SCHEDULE_PATH for caller handling: %s",
        DEFAULT_SCHEDULE_PATH,
    )
    return DEFAULT_SCHEDULE_PATH

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

        # Dataset diagnostics
        out["dataset_info"] = {
            "path": str(DEFAULT_DATASET),
            "exists": DEFAULT_DATASET.exists(),
            "rows": len(dataset_df) if dataset_df is not None else 0,
            "columns": list(dataset_df.columns)[:10] if dataset_df is not None and not dataset_df.empty else [],
        }

        # Test a specific game lookup
        if dataset_df is not None and not dataset_df.empty:
            test_mask = (
                (dataset_df["season"] == 2025) &
                (dataset_df["week"] == 15) &
                (dataset_df["home_team"] == "TB") &
                (dataset_df["away_team"] == "ATL")
            )
            out["test_lookup"] = {
                "query": "TB vs ATL 2025 W15",
                "matches": int(test_mask.sum()),
                "home_prior_pf_avg_3": float(dataset_df.loc[test_mask, "home_prior_pf_avg_3"].iloc[0]) if test_mask.sum() > 0 else None,
            }
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

@app.get(path="/schedule/next-week", response_model=List[ScheduleGame])
def get_next_week_schedule() -> List[ScheduleGame]:
    """
    Retrieve the list of scheduled NFL games for the upcoming week.

    This endpoint filters the schedule CSV based on current NFL context (season/week),
    normalizes team abbreviations, and formats kickoff times. It supports frontend
    rendering of matchups and prediction requests. Depends on: get_current_nfl_context(),
    SCHEDULE_PATH env var, and team_abbr_map.json for normalization.
    """
    global TEAM_CODE_FIX

    # Educational: We use the centralized helper to get the dataframe, ensuring consistent data source.
    df = _get_schedule_df(2025)

    # ─────────────────────────────────────────────────────────────────────────────
    # Data Normalization
    # ─────────────────────────────────────────────────────────────────────────────
    # Standardize team abbreviations to match our internal model codes (e.g., PHI, BAL).
    try:
        for col in ("home_team", "away_team"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace(TEAM_CODE_FIX)
    except Exception as e:
        log.warning("Error normalizing team codes in schedule: %s", e)

    # Convert Kickoff Times
    # We construct a full datetime from 'gameday' (YYYY-MM-DD) and 'gametime' (HH:MM).
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


@app.get(path="/api/games/next-week", response_model=NextWeekGamesResponse)
def get_next_week_games_api() -> NextWeekGamesResponse:
    """Compatibility route for older frontends.

    Old clients often call `/api/games/next-week` and expect `{ games: [...] }`.
    We internally reuse the canonical `/schedule/next-week` logic.
    """
    games = get_next_week_schedule()
    return NextWeekGamesResponse(games=games)

# -----------------------
# Prediction Helpers
# -----------------------

def _get_prediction_features(df: Optional[pd.DataFrame], home: str, away: str, season: int, week: int):
    """
    Retrieve or build feature row for a game.
    Returns: (row_series, fallback_used_bool)
    """
    safe_dataset = _ensure_home_away(df.copy()) if df is not None else df
    fallback = False

    # Check for existing row
    if safe_dataset is not None:
        mask = build_game_mask(safe_dataset, season, week, home, away)
        existing = safe_dataset.loc[mask]
        if not existing.empty:
            return existing.iloc[0], False

    # Build future row
    try:
        row = _build_future_row(safe_dataset if safe_dataset is not None else pd.DataFrame(), home, away, season, week)
        return row, False
    except Exception as e:
        log.warning("Feature build failed for %s vs %s: %s", home, away, e)
        # minimal fallback
        row = pd.Series({
            "home_moneyline_prob": 0.5, "away_moneyline_prob": 0.5,
            "total_line": 45.0, "home_rest": 7, "away_rest": 7,
            "home_team": home, "away_team": away,
            "home_game_date": f"{season}-W{week:02d}"
        })
        return row, True

def _prepare_model_input(row: pd.Series, models: Dict[str, Any]):
    """
    Align feature row to model requirements.
    Returns: (X_dataframe, missing_columns_list)
    """
    raw_cols = models.get("raw_feature_columns", {})
    exp_num = list(raw_cols.get("numeric", []))
    exp_cat = list(raw_cols.get("categorical", []))
    exp_all = exp_num + exp_cat

    # Just-in-time inference if metadata missing
    if not exp_all:
        inferred = _infer_raw_feature_columns(models, None) # checking global inference
        exp_all = inferred.get("numeric", []) + inferred.get("categorical", [])

    if not exp_all:
         raise HTTPException(503, "Model feature metadata missing.")

    data = {}
    for col in exp_all:
        if col in row.index:
             val = row[col]
             data[col] = val if not pd.isna(val) else np.nan
        else:
             # Basic defaults for required categoricals
             if col == "home_team": data[col] = row.get("home_team")
             elif col == "away_team": data[col] = row.get("away_team")
             elif col == "home_game_date": data[col] = row.get("home_game_date")
             else: data[col] = np.nan

    X = pd.DataFrame([data], columns=exp_all)

    # Check for missing required identifiers (non-imputable)
    req = _validate_features_present(exp_all, row)
    return X, req

def _predict_scores(models: Dict[str, Any], X: pd.DataFrame) -> tuple[float, float]:
    """
    Run home/away regressions.
    Returns: (home_score, away_score)
    """
    def run(m_key):
        m = models.get(m_key)
        if not m: return 20.0 # safe default
        try:
             # Use _reg_predict logic (simplified inline or call existing)
             return float(_reg_predict(m, X)[0])
        except Exception as e:
             # Try with NaN filling if it failed due to missing/NaN
             X_safe = X.fillna(0)
             try:
                 return float(_reg_predict(m, X_safe)[0])
             except:
                 return 20.0

    h_score = np.clip(run("home_model"), 0, 70)
    a_score = np.clip(run("away_model"), 0, 70)
    return float(h_score), float(a_score)

def _predict_win_prob(models: Dict[str, Any], X: pd.DataFrame, point_diff: float) -> tuple[float, bool]:
    """
    Run win classifier.
    Returns: (home_win_prob, fallback_used)
    """
    win_m = models.get("win_clf") or models.get("win_model") or models.get("classifier") or models.get("clf")
    fallback_prob = 1.0 / (1.0 + math.exp(-0.25 * point_diff))

    if not win_m:
         return fallback_prob, True

    try:
        p = None
        if hasattr(win_m, "predict_proba"):
            p = float(win_m.predict_proba(X)[0, 1])
        elif hasattr(win_m, "predict"): # non-probabilistic
            p = float(win_m.predict(X)[0])

        if p is not None and not math.isnan(p):
            return p, False

    except Exception as e:
         # Attempt sanitize
         try:
             X_safe = X.fillna(0).replace([np.inf, -np.inf], 0)
             if hasattr(win_m, "predict_proba"):
                 p = float(win_m.predict_proba(X_safe)[0, 1])
                 if not math.isnan(p):
                     return p, False
         except:
             pass

    return fallback_prob, True



@app.post("/predict", response_model=PredictionResponse)
def predict_game(payload: PredictionRequest) -> PredictionResponse:
    """
    Predict endpoint: Orchestrates feature assembly and model inference.
    Simplifies complex logic by delegating to specialized helpers.
    """
    if model_objects is None or dataset_df is None:
        raise HTTPException(500, "Backend not initialized.")

    h, a = to_team_abbr(payload.home_team), to_team_abbr(payload.away_team)
    season, week = int(payload.season), int(payload.week)
    log.info("Prediction Request: %s vs %s (%s W%s)", h, a, season, week)

    # 1. Feature Assembly
    row, feature_fallback = _get_prediction_features(dataset_df, h, a, season, week)

    # 2. Prepare Model Input
    X, missing_cols = _prepare_model_input(row, model_objects)

    if missing_cols and not ALLOW_FALLBACK_PREDICTIONS:
         log.warning("Prediction Aborted: Missing columns %s", missing_cols)
         raise HTTPException(400, f"Missing non-imputable columns: {missing_cols}")

    # 3. Inference
    home_score, away_score = _predict_scores(model_objects, X)
    point_diff = round(home_score - away_score, 1)

    win_prob, win_fallback = _predict_win_prob(model_objects, X, point_diff)

    # 4. Response Construction
    mode = model_objects.get("mode", "production") if isinstance(model_objects, dict) else "production"

    # Determine source label
    source_parts = []
    if feature_fallback: source_parts.append("feature_fallback")
    if win_fallback: source_parts.append("win_fallback")
    source = "+".join(source_parts) if source_parts else "model"

    if source != "model" and not ALLOW_FALLBACK_PREDICTIONS:
         raise HTTPException(400, "Prediction fell back to heuristics, but strict mode is on.")

    return PredictionResponse(
        home_score=round(home_score, 1),
        away_score=round(away_score, 1),
        home_win_probability=win_prob,
        away_win_probability=1.0 - win_prob,
        point_diff=point_diff,
        mode=str(mode),
        prediction_source=source,
        win_classifier_used=not win_fallback
    )



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


@app.get("/history")
def history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent prediction history from the loaded dataset.
    Returns: List of game records sorted by recentness.
    """
    if dataset_df is None or dataset_df.empty:
        return []
    try:
        # Sort by season and week descending
        df = dataset_df.sort_values(["season", "week"], ascending=False).head(limit)
        out = []
        for _, row in df.iterrows():
            out.append({
                "season": int(row.get("season", 0)),
                "week": int(row.get("week", 0)),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "home_score": row.get("home_points_for"),
                "away_score": row.get("away_points_for"),
                "winner": row.get("winner")
            })
        return out
    except Exception as e:
        log.error("History fetch failed: %s", e)
        return []


@app.get("/api/history", response_model=HistoryResponse)
def history_api(limit: int = 100) -> HistoryResponse:
    """Stable history endpoint for dashboards.

    Why this exists:
        - `/history` historically returned a raw list.
        - Dashboards prefer a predictable envelope with totals.

    Returns:
        HistoryResponse: { entries: [...], total: int, limit: int }
    """
    entries = history(limit=limit)
    # Pydantic will coerce dicts into HistoryEntry objects.
    return HistoryResponse(entries=entries, total=len(entries), limit=limit)


@app.post("/train")
def train_model():
    """
    Trigger model retraining.
    currently not implemented via API for security/resource reasons.
    """
    raise HTTPException(501, "Training via API is not currently supported. Use the CLI: python backend/train_models.py")


@app.get("/status/overview")
def status_overview() -> Dict[str, Any]:
    """
    Aggregates system health, dataset stats, and model info for dashboards.
    """
    h_resp = health()

    # Small, frontend-friendly history summary.
    total_rows = len(dataset_df) if dataset_df is not None else 0
    latest = None
    try:
        if dataset_df is not None and not dataset_df.empty and {"season", "week"}.issubset(set(dataset_df.columns)):
            last = dataset_df.sort_values(["season", "week"], ascending=False).iloc[0]
            latest = {"season": int(last.get("season", 0)), "week": int(last.get("week", 0))}
    except Exception:
        latest = None

    return {
        "health": h_resp,
        "dataset": {
            "rows": len(dataset_df) if dataset_df is not None else 0,
            "columns": list(dataset_df.columns) if dataset_df is not None and not dataset_df.empty else [],
            "source": str(DEFAULT_DATASET)
        },
        "history": {
            "metrics": {
                "total_predictions": total_rows,
                "latest": latest,
            }
        },
        "model_info": {
             "mode": h_resp.mode,
             "last_loaded": datetime.now().isoformat()
        }
    }


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

        # Determine current context (week/season)
        ctx = get_current_nfl_context()

        # Load schedule (using same consistent source)
        s = _get_schedule_df(2025)

        if s.empty:
             raise HTTPException(503, "Schedule data not available.")

        # Filter for the specific target week
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
