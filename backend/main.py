# -*- coding: utf-8 -*-
"""
File: backend/main.py

Purpose:
    FastAPI backend for the NFL prediction dashboard.

    Exposes endpoints for:
    - /health            : service + components health
    - /status/overview   : lightweight dashboard summary
    - /schedule/next-week: upcoming week schedule
    - /history           : recent prediction history
    - /predict           : single-game prediction (scores + win probabilities)

Key design points:
    - Loads latest game_features*.csv dataset + trained models at startup.
    - Normalizes team codes (home/away, abbr/name) to uppercase for stable matching.
    - Uses Pydantic models for request/response typing where it matters
      (health + prediction).
    - Keeps in-memory prediction history bounded for the /history endpoint.

Notes:
    - The prediction contract is aligned with the current React client:
        - Request body: { home_team, away_team, season, week }
        - Response: PredictionResponse with home_score, away_score,
          home_win_probability, away_win_probability, point_diff, etc.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple, Literal
from urllib.parse import urlparse
import nflreadpy as nfl
from dotenv import load_dotenv
import numpy as np
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.utils.functions_for_main import (
    _add_kickoff_utc_datetime,
    _coerce_season_week,
    _normalize_team_columns,
    _get_game_row,
    TEAM_ABBR_MAP,
    _prepare_inputs,
    _is_pipeline,
    _align_numeric_df_for_model,
    _normalize_team_code,
    _predict_score,
    _clamp_score,
    _smooth_win_probability,
    _roll_forward_missing_player_stats
)

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


# -------------------------------------------------------------------
# Config & Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()

# Load backend/.env no matter where uvicorn is launched from (project root vs backend/)
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"



# Allow overriding the schedule CSV via env; default to backend/data
schedule_env = os.environ.get("SCHEDULE_PATH")
SCHEDULE_PATH = Path(schedule_env) if schedule_env else (DATA_DIR / "Nfl_schedule_2025.csv")

# Required model keys for /predict to be "ready"
REQUIRED_MODELS: Tuple[str, ...] = ("home", "away", "win")
# Models directory is resolved at runtime by _find_models_dir().
# Override in production with: MODELS_DIR=/absolute/or/repo-relative/path


# Ensure expected folders exist (safe on repeated calls)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: MODELS_DIR is resolved via _find_models_dir() (defined below)
PREDICTION_STORAGE = BASE_DIR / "Predictions"
PREDICTION_STORAGE.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------





def _find_schedule_path() -> Optional[Path]:
    """
    Locate a schedule CSV file.

    Priority:
      1. Explicit SCHEDULE_PATH (env override or default backend/data/Nfl_schedule_2025.csv)
      2. Any CSV in backend/data/ that looks like a schedule
      3. Frontend public copy at ../frontend/public/nflSchedule.csv (local dev)
    """

    # 1) explicit path
    if SCHEDULE_PATH.exists():
        return SCHEDULE_PATH

    # 2) search backend/data for schedule-like CSVs
    candidates: List[Path] = []
    for p in DATA_DIR.glob("*.csv"):
        name = p.name.lower()
        if "schedule" in name or name.startswith("nfl"):
            candidates.append(p)
    if candidates:
        # Prefer most recently modified
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    # 3) local dev fallback: frontend/public
    frontend_sched = BASE_DIR.parent / "frontend" / "public" / "nflSchedule.csv"
    if frontend_sched.exists():
        return frontend_sched

    return None


def _models_dir_has_required_artifacts(models_dir: Path) -> bool:
    """Return True if a directory looks like a complete model bundle.

    We treat a bundle as "complete" if it contains home+away regressors and a win classifier,
    either as full pipelines (*_pipe.joblib) or as raw estimators (*_model.joblib / win_clf*.joblib).
    """
    try:
        if not models_dir.exists() or not models_dir.is_dir():
            return False

        has_home = (models_dir / "home_pipe.joblib").exists() or (models_dir / "home_model.joblib").exists()
        has_away = (models_dir / "away_pipe.joblib").exists() or (models_dir / "away_model.joblib").exists()
        has_win = (models_dir / "win_pipe.joblib").exists() or (models_dir / "win_clf_calibrated.joblib").exists()
        return bool(has_home and has_away and has_win)
    except Exception:
        return False


def _find_models_dir() -> Path:
    """Locate the best models directory.

    Priority order:
      1) Env override (recommended for Heroku): MODELS_DIR / MODELS_PATH / MODEL_DIR
      2) A complete bundle in common repo locations (prod-models/models, dated runs, etc.)
      3) Fallback: backend/models (even if incomplete, so errors are visible in logs)

    Tip:
      - On Heroku, always set MODELS_DIR to a path that exists *in the slug*.
    """
    env = (
        os.environ.get("MODELS_DIR")
        or os.environ.get("MODELS_PATH")
        or os.environ.get("MODEL_DIR")
    )
    if env:
        p = Path(env).expanduser()
        if _models_dir_has_required_artifacts(p):
            return p
        if p.exists():
            # Exists but doesn't look complete; still return so logs show what is missing.
            return p

    candidates: List[Path] = []

    # Common packaged pattern: backend/data/prod-models/models
    direct = BASE_DIR / "data" / "prod-models" / "models"
    if _models_dir_has_required_artifacts(direct):
        candidates.append(direct)

    # Common local pattern: backend/models
    local_default = BASE_DIR / "models"
    if _models_dir_has_required_artifacts(local_default):
        candidates.append(local_default)

    # Date-stamped training runs: backend/20251215/models (most recent wins)
    for p in BASE_DIR.glob("20*/models"):
        if _models_dir_has_required_artifacts(p):
            candidates.append(p)

    # Any nested prod-models/models in the repo
    for p in BASE_DIR.glob("**/prod-models/models"):
        if _models_dir_has_required_artifacts(p):
            candidates.append(p)

    if candidates:
        # Prefer the most recently modified bundle
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return local_default


# Resolve models directory once at import time so serving code can rely on MODELS_DIR.
MODELS_DIR: Path = _find_models_dir()


def _load_team_logo_map() -> Dict[str, str]:
    """Load a {TEAM_ABBR: logo_url} mapping if available.

    Why:
      - Your schedule CSV typically contains team abbreviations (PHI, DAL, ...),
        but does NOT include logo URLs.
      - The frontend can only render logos if we attach a URL per game.

    Expected shapes:
      - CSV (recommended): `team_logos.csv` with columns `team_abbr` + `team_logo_espn`
      - JSON: {"KC": "https://...", ...}
    """
    def _resolve_path(p: Path) -> Optional[Path]:
        if p.exists():
            return p
        if not p.is_absolute():
            # Allow relative paths from either repo root or backend/ dir
            for base in (BASE_DIR.parent, BASE_DIR):
                candidate = (base / p).resolve()
                if candidate.exists():
                    return candidate
        return None

    candidates: List[Path] = []
    env = os.environ.get("TEAM_LOGOS_PATH") or os.environ.get("TEAM_LOGO_PATH")
    if env:
        candidates.append(Path(env).expanduser())

    # Prefer the nflverse CSV under backend/data/
    candidates.extend(
        [
            DATA_DIR / "team_logos.csv",
            DATA_DIR / "team_logo.csv",
            DATA_DIR / "team_logo_abbr.json",
        ]
    )

    for raw_path in candidates:
        p = _resolve_path(raw_path)
        if not p:
            continue

        try:
            # JSON map support: {"KC": "https://..."}
            if p.suffix.lower() == ".json":
                with open(p, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                if not isinstance(obj, dict):
                    continue
                out: Dict[str, str] = {}
                for k, v in obj.items():
                    key = str(k).strip().upper()
                    val = str(v).strip()
                    if not key or not val or val.lower() == "nan":
                        continue
                    out[key] = val
                    # Also store canonical alias (e.g., LA->LAR) for lookups.
                    if TEAM_ABBR_MAP and key in TEAM_ABBR_MAP:
                        out.setdefault(TEAM_ABBR_MAP[key], val)
                if out:
                    logging.info("[Logos] Loaded %d team logos from %s", len(out), p)
                    return out
                continue

            # CSV support (flexible schema)
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}

            key_col = None
            for k in ("abbr", "team", "team_abbr", "team_code"):
                if k in cols:
                    key_col = cols[k]
                    break

            # Prefer ESPN logos, per request.
            val_col = None
            for v in (
                "team_logo_espn",
                # fallbacks
                "team_logo_squared",
                "team_logo_wikipedia",
                "team_wordmark",
                "logo_url",
                "logo",
                "url",
                "image_url",
                "image",
            ):
                if v in cols:
                    val_col = cols[v]
                    break

            if not key_col or not val_col:
                continue

            out: Dict[str, str] = {}
            for _, r in df.iterrows():
                key = str(r.get(key_col, "")).strip().upper()
                val = str(r.get(val_col, "")).strip()
                if not key or not val or val.lower() == "nan":
                    continue
                out[key] = val
                if TEAM_ABBR_MAP and key in TEAM_ABBR_MAP:
                    out.setdefault(TEAM_ABBR_MAP[key], val)

            if out:
                logging.info("[Logos] Loaded %d team logos from %s (col=%s)", len(out), p, val_col)
                return out
        except Exception as e:
            logging.warning("[Logos] Failed reading %s: %s", raw_path, e)

    logging.info("[Logos] No team logo map found; schedule will return null logos.")
    return {}


def _calculate_win_probability(
    win_model: Any,
    full_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    h_score: float,
    a_score: float,
    preprocessor: Optional[Any] = None,
) -> Tuple[float, bool]:
    """Compute home-team win probability with sensible fallbacks.

    Preferred order:
      1) Use a fitted sklearn Pipeline (win_pipe.joblib) on the *raw* DataFrame.
      2) Use a calibrated classifier (win_clf_calibrated.joblib) on *preprocessed* features
         via the standalone preprocessor.joblib, if present.
      3) As a last resort, use a simple logistic transform of the predicted point diff.

    Returns:
      (home_win_probability, win_classifier_used)
    """
    if win_model is not None and hasattr(win_model, "predict_proba"):
        is_pipeline = bool(getattr(win_model, "steps", None) is not None or getattr(win_model, "named_steps", None) is not None)

        # A) Pipeline case: pass raw DataFrame
        if is_pipeline:
            try:
                win_prob = float(win_model.predict_proba(full_df)[0][1])
                return win_prob, True
            except Exception as e:
                logging.warning("[Predict] win_pipe predict_proba failed; falling back to logistic: %s", e)

        # B) Classifier-only case: transform then predict_proba
        if (not is_pipeline) and (preprocessor is not None):
            try:
                X_proc = preprocessor.transform(full_df)
                win_prob = float(win_model.predict_proba(X_proc)[0][1])
                return win_prob, True
            except Exception as e:
                logging.warning("[Predict] win_clf predict_proba failed after preprocessor.transform; falling back: %s", e)

        # C) Last attempt: numeric-only (may work if the model was trained on raw numeric columns)
        if not is_pipeline:
            try:
                if numeric_df is not None and not numeric_df.empty:
                    win_prob = float(win_model.predict_proba(numeric_df)[0][1])
                    return win_prob, True
            except Exception as e:
                logging.warning("[Predict] win_clf predict_proba failed on numeric_df; falling back: %s", e)

    # D) Fallback: logistic on predicted point differential
    diff = float(h_score - a_score)
    win_prob = float(1.0 / (1.0 + np.exp(-0.3 * diff)))
    return win_prob, False


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

        # Cached numeric medians from the loaded dataset (used for stable imputation).
        # Set during _load_dataset().
        self.numeric_medians: Optional[pd.Series] = None


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
            candidates = sorted(
                DATA_DIR.glob("game_features*.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            path = candidates[0] if candidates else (DATA_DIR / "game_features.csv")

            if not path.exists():
                logging.warning("[Dataset] No dataset found at: %s", path)
                self.dataset = None
                return

            logging.info("[Dataset] Loading dataset from: %s", path)
            df = pd.read_csv(path)

            # Normalize key columns for consistent lookups
            df = _coerce_season_week(df)
            df = _normalize_team_columns(df, cols=["home_team", "away_team"])

            self.dataset = df


            # Cache numeric medians once so prediction-time imputations are stable


            # even when the matched row contains many missing values (future games).


            try:


                self.numeric_medians = df.select_dtypes(include=[np.number]).median(numeric_only=True)


            except Exception:


                self.numeric_medians = None

            logging.info("[Dataset] Loaded %d rows", len(df))
        except Exception as e:  # pragma: no cover - defensive
            logging.exception("[Dataset] Error while loading dataset: %s", e)
            self.dataset = None

    def _load_models(self) -> None:
        """Load each required model independently."""
        self.models = {}
        logging.info("[Model] Using models directory: %s", MODELS_DIR)
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


state = AppState()

# -------------------------------------------------------------------
# FastAPI App + Lifespan
# -------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan hook.

    - On startup: load dataset + models into global state.
    - On shutdown: log a simple message (no persistent cleanup necessary).
    """
    logging.info("[App] Starting up; loading dataset and models...")
    state.load()
    yield
    logging.info("[App] Shutdown complete.")


app = FastAPI(lifespan=lifespan)

def _normalize_origin(raw: str) -> Optional[str]:
    origin = (raw or "").strip().strip('"').strip("'")
    if not origin:
        return None

    # Browsers send Origin without a trailing slash; normalize to match.
    origin = origin.rstrip("/")

    # If given a hostname without scheme, assume https.
    if "://" not in origin and "." in origin:
        origin = f"https://{origin}"

    try:
        parsed = urlparse(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None
    except Exception:
        return None

    return origin


def _parse_origins(env_value: Optional[str]) -> List[str]:
    if not env_value:
        return []
    parts = [p for p in (env_value or "").split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        o = _normalize_origin(p)
        if o and o not in out:
            out.append(o)
    return out


# CORS configuration
# ------------------
# The browser will send an `Origin` header that looks like:
#   https://nfl-ml-predictions.vercel.app
#
# On Heroku we control CORS via config vars (recommended):
#   - RESTRICT_CORS     : "true" | "false" (default: true)
#   - ALLOWED_ORIGINS   : comma-separated list of exact origins (scheme + host)
#                         Example:
#                           https://nfl-ml-predictions.vercel.app,http://localhost:5173
#                         (We also accept bare hostnames and normalize them to https://...)
#   - ALLOW_ORIGIN_REGEX: regex for dynamic preview origins (e.g., Vercel preview URLs)
#                         Example (recommended):
#                           ^https://.*\.vercel\.app$
#
# Important:
#   - FastAPI's allow_origin_regex expects a REAL regex (not a glob).
#     If you accidentally set "*.vercel.app", we convert it to a safe regex.

def _env_flag(name: str, default: str = "true") -> bool:
    """Parse boolean-ish env vars safely."""
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _maybe_glob_to_regex(raw: Optional[str]) -> Optional[str]:
    """Convert common glob patterns (like *.vercel.app) to a regex FastAPI understands.

    FastAPI / Starlette want a regex that matches the *full origin string*, including scheme.
    """
    if not raw:
        return None

    p = str(raw).strip().strip('"').strip("'").rstrip("/")
    if not p:
        return None

    # If it already looks like a regex, assume the user knows what they're doing.
    if p.startswith("^") or p.endswith("$"):
        return p

    # Common Heroku mistake: ALLOW_ORIGIN_REGEX="*.vercel.app"
    if "vercel.app" in p and "*" in p:
        return r"^https://.*\.vercel\.app$"

    # If it contains a wildcard but isn't a regex, treat it as a glob and convert.
    if "*" in p:
        escaped = re.escape(p).replace(r"\*", ".*")
        if "://" in p:
            return rf"^{escaped}$"
        return rf"^https?://{escaped}$"

    # Hostname only (no scheme): match http/https for that exact host.
    if "://" not in p and "." in p:
        return rf"^https?://{re.escape(p)}$"

    return p


# Read origins from env (Heroku), then fall back to a small safe default list.
_allowed_raw = (
    os.environ.get("ALLOWED_ORIGINS")
    or os.environ.get("CORS_ORIGINS")
    or os.environ.get("CORS_DEPLOYED")  # legacy/diagnostic var some deploy scripts used
)

ALLOWED_ORIGINS: List[str] = _parse_origins(_allowed_raw) or [
    # Production (your chosen canonical frontend)
    "https://nfl-ml-predictions.vercel.app",
    # Local dev (Vite)
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Regex origins: Vercel previews change per deploy, so a regex is the cleanest solution.
_allow_regex_raw = os.environ.get("ALLOW_ORIGIN_REGEX") or os.environ.get("CORS_ORIGINS_REGEX")
ALLOW_ORIGIN_REGEX = _maybe_glob_to_regex(_allow_regex_raw)

# Optional convenience switch: allow previews even if regex isn't explicitly configured.
if not ALLOW_ORIGIN_REGEX and _env_flag("ALLOW_VERCEL_PREVIEWS", "true"):
    ALLOW_ORIGIN_REGEX = r"^https://.*\.vercel\.app$"

# If RESTRICT_CORS is false, we intentionally open it up (useful for quick debugging).
if not _env_flag("RESTRICT_CORS", "true"):
    ALLOWED_ORIGINS = ["*"]
    ALLOW_ORIGIN_REGEX = None

logging.info("[App] CORS allowed origins: %s", ALLOWED_ORIGINS)
if ALLOW_ORIGIN_REGEX:
    logging.info("[App] CORS allow_origin_regex: %s", ALLOW_ORIGIN_REGEX)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,          # ✅ list (string here breaks CORS matching)
    allow_origin_regex=ALLOW_ORIGIN_REGEX,  # ✅ supports Vercel preview deployments
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------


class HealthComponents(BaseModel):
    """Components section for the /health response."""
    dataset: bool
    models: bool
    loaded_models: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """
    Canonical health response model.

    Matches README contract while being friendly to the current frontend:
      - status   : "healthy" | "unhealthy"
      - timestamp: auto-filled UTC timestamp
      - mode     : e.g. "production"
      - reason   : optional human-readable message
      - components: dataset/models readiness
    """
    status: Literal["healthy", "unhealthy"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str = "production"
    reason: Optional[str] = None
    components: HealthComponents


class PredictRequest(BaseModel):
    """
    Request body for /predict.

    The frontend usually sends team abbreviations (e.g. "BUF", "HOU"), but
    the backend will uppercase whatever is provided and match against the
    normalized dataset.
    """
    home_team: str
    away_team: str
    season: int
    week: int


class PredictionResponse(BaseModel):
    """
    Canonical prediction response for /predict.

    Fields are aligned with the React client (Card/TeamGrid + PredictionContext):
      - home_score, away_score           : predicted scores
      - home_win_probability, away_win_probability : probabilities in [0, 1]
      - point_diff                       : home_score - away_score
      - game_id                          : stable key used across the app
      - generated_at                     : UTC timestamp
      - mode                             : e.g., "production"
      - win_classifier_used              : whether the calibrated classifier was used
    """
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
    generated_at: datetime
    mode: str = Field(..., description="Mode of prediction, e.g., 'production'")
    win_classifier_used: bool = Field(
        ..., description="Whether the win probability classifier was used"
    )


# -------------------------------------------------------------------
# Health + Status
# -------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Lightweight health check used by the frontend and deployment checks.

    Returns component readiness + a timestamp and human-friendly reason.
    """
    has_dataset = state.dataset is not None
    models_ok = all(m in state.models for m in REQUIRED_MODELS)

    status: Literal["healthy", "unhealthy"]
    status = "healthy" if has_dataset and models_ok else "unhealthy"

    reasons: List[str] = []
    if not has_dataset:
        reasons.append("dataset not loaded")
    if not models_ok:
        missing = [m for m in REQUIRED_MODELS if m not in state.models]
        reasons.append(f"missing models: {', '.join(missing)}")

    reason_str = ", ".join(reasons) if reasons else None

    return HealthResponse(
        status=status,
        reason=reason_str,
        components=HealthComponents(
            dataset=has_dataset,
            models=models_ok,
            loaded_models=list(state.models.keys()),
        ),
    )


@app.get("/status/overview")
def status_overview() -> Dict[str, Any]:
    """
    Summary endpoint used by StatsPage.jsx.

    Returns:
      - current health (via /health)
      - dataset info (row count)
      - basic history metrics (prediction count placeholder)
    """
    if state.dataset is not None:
        dataset_stats = {
            "rows": len(state.dataset),
            "path": "game_features.csv",  # Label only; not exact filename
        }
    else:
        dataset_stats = {"rows": 0, "path": "none"}

    return {
        "health": health(),  # reuse typed health response
        "dataset": dataset_stats,
        "history": {
            "metrics": {
                "total_predictions": len(state.history),
                "win_rate": 0.0,  # Placeholder until outcomes are tracked
            }
        },
    }


# -------------------------------------------------------------------
# Schedule: Next Week
# -------------------------------------------------------------------


@app.get("/schedule/next-week")
def get_schedule() -> List[Dict[str, Any]]:
    """
    Return the schedule for the "next" NFL week based on the schedule CSV.

    Logic:
      - Resolve schedule path via _find_schedule_path().
      - Normalize season/week + team abbreviations.
      - If 'gameday' is present, interpret as kickoff datetime (UTC-aware).
      - Determine the next slate using the earliest future game; fall back
        to the latest season/week in the file if all games are in the past.
    """
    df = nfl.load_schedules(seasons=2025)
    df = df.to_pandas()
    print(f'List of columns from get schedule in main.py  : {df.columns.tolist()}')
    if df is None or df.empty:
        logging.warning("[Schedule] No schedule data available; returning empty list.")
        return []

    df = _coerce_season_week(df)
    df = df.infer_objects()
    df = _normalize_team_columns(
        df, cols=("home_abbr", "away_abbr", "home_team", "away_team")
    )
    # Attach logos if we have a mapping file. This is optional; missing logo
    logo_map = _load_team_logo_map()
    df = _add_kickoff_utc_datetime(df)  # uses 'gameday' column if present

    # Decide which (season, week) is "next"
    now_utc = pd.Timestamp.now(tz="UTC")
    future = df[df["dt"].notna() & (df["dt"] > now_utc)].sort_values(by=["dt", "season", "week"])

    if not future.empty:
        next_row = future.iloc[0]
        target_s = int(next_row.get("season_num", next_row.get("season", 2024)))
        target_w = int(next_row.get("week_num", next_row.get("week", 1)))
    else:
        # Fallback: last season/week in file
        season_series = df.get("season", df.get("season_num"))
        week_series = df.get("week", df.get("week_num"))
        target_s = int(season_series.max()) if season_series is not None else 2025
        target_w = int(week_series.max()) if week_series is not None else 1

    s_col = "season_num" if "season_num" in df.columns else "season"
    w_col = "week_num" if "week_num" in df.columns else "week"

    week_df = df[(df[s_col] == target_s) & (df[w_col] == target_w)]

    results: List[Dict[str, Any]] = []
    for _, row in week_df.iterrows():
        home_team = logo_map.get("team_name") if "team_name" == row.get("home_team") else row.get("home_team")
        away_team = logo_map.get("team_name") if "team_name" == row.get("away_team") else row.get("away_team")
        home_abbr = row.get("home_abbr", home_team)
        away_abbr = row.get("away_abbr", away_team)

        # Logos: prefer explicit schedule columns, else use logo_map keyed by team abbr
        home_logo = (
            row.get("home_logo")
            or row.get("home_logo_url")
            or logo_map.get(str(home_abbr).upper())
            or logo_map.get(str(home_team).upper())
        )
        away_logo = (
            row.get("away_logo")
            or row.get("away_logo_url")
            or logo_map.get(str(away_abbr).upper())
            or logo_map.get(str(away_team).upper())
        )

        kickoff_val: Optional[str] = None
        if ("dt" in row) and pd.notna(row["dt"]):
            try:
                kickoff_val = row["dt"].isoformat()
            except Exception:
                kickoff_val = None

        results.append(
            {
                "game_id": f"{target_s}_{target_w}_{home_team}_{away_team}",
                "season": int(target_s),
                "week": int(target_w),
                "home_team": home_team,
                "away_team": away_team,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "home_logo": home_logo,
                "away_logo": away_logo,
                "kickoff": kickoff_val,
            }
        )

    logging.info(
        "[Schedule] Returning %d games for season=%s week=%s",
        len(results),
        target_s,
        target_w,
    )
    return results


# -------------------------------------------------------------------
# Prediction History
# -------------------------------------------------------------------


@app.get("/history", response_model=List[PredictionResponse])
def get_prediction_history(
    limit: int = Query(100, ge=1, le=1000)
) -> List[Dict[str, Any]]:
    """
    Return the last N prediction results recorded in memory.

    Used by StatsPage/PredictionContext as a history source.
    """
    if limit <= 0:
        return []
    return state.history[-limit:]


# -------------------------------------------------------------------
# /predict (final enhanced)
# -------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest) -> Dict[str, Any]:
    """
    Predict home/away score and win probability for a single game.

    Enhancements:
      - Cleaner structure (helpers instead of nested functions)
      - Robust row matching (team code OR abbr columns)
      - Stable imputation using dataset medians (computed once)
      - Feature alignment for non-pipeline estimators (feature_names_in_)
      - Output smoothing for probability + sanity clamping for scores
    """
    # ----- Readiness -----
    models_ok = all(m in state.models for m in REQUIRED_MODELS)
    if state.dataset is None or not models_ok:
        missing = [m for m in REQUIRED_MODELS if m not in state.models]
        raise HTTPException(status_code=503, detail=f"Not ready: missing models {missing}")

    # ----- Validate request -----
    season = int(request.season)
    week = int(request.week)

    if week < 1 or week > 30:
        raise HTTPException(status_code=400, detail="Invalid week. Expected 1..30.")
    if season < 1990 or season > 2100:
        raise HTTPException(status_code=400, detail="Invalid season. Expected a realistic year.")
    if not request.home_team or not request.away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team are required.")

    home_team = _normalize_team_code(request.home_team)
    away_team = _normalize_team_code(request.away_team)
    if home_team == away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must be different.")

    df = state.dataset
    assert df is not None  # guarded above

    # ----- Find feature row -----
    row = _get_game_row(df, season, week, home_team, away_team)

    # ----- Fill missing player stats (safe) -----
    row = _roll_forward_missing_player_stats(
        df=df,
        row_df=row,
        home_team=home_team,
        away_team=away_team,
        season=season,
        week=week,
    )

    # ----- Prepare model inputs -----
    full_df, numeric_df = _prepare_inputs(row)

    # ----- Predict scores -----
    home_model = state.models["home"]
    away_model = state.models["away"]
    win_model = state.models.get("win")

    try:
        h_score = _predict_score(
            home_model,
            full_df,
            numeric_df,
            preprocessor=state.preprocessor,
            numeric_medians=getattr(state, "numeric_medians", None),
            model_name="home_model",
        )
        a_score = _predict_score(
            away_model,
            full_df,
            numeric_df,
            preprocessor=state.preprocessor,
            numeric_medians=getattr(state, "numeric_medians", None),
            model_name="away_model",
        )
    except Exception as model_err:
        msg = str(model_err) or "unknown error"
        # Keep messages concise for clients, verbose details stay in logs.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {msg.splitlines()[0]}") from model_err

    # Clamp to sane ranges
    h_score = _clamp_score(h_score)
    a_score = _clamp_score(a_score)

    # ----- Predict win probability -----
    try:
        win_prob_raw, clf_used = _calculate_win_probability(
            win_model,
            full_df,
            numeric_df,
            h_score,
            a_score,
            preprocessor=state.preprocessor,
        )
    except Exception as win_err:
        logging.warning("[Predict] Win probability calc failed; defaulting to 0.5: %s", str(win_err).splitlines()[0])
        win_prob_raw, clf_used = 0.5, False

    point_diff = float(h_score - a_score)

    # Smooth probability (no retraining)
    win_prob = _smooth_win_probability(win_prob_raw, point_diff, clf_used=clf_used)

    # ----- Build response -----
    game_id = f"{season}_{week}_{home_team}_{away_team}"
    generated_at = datetime.now(timezone.utc)

    result: Dict[str, Any] = {
        "season": season,
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
        "game_id": game_id,
        "home_score": float(h_score),
        "away_score": float(a_score),
        "home_win_probability": float(win_prob),
        "away_win_probability": float(1.0 - win_prob),
        "point_diff": float(point_diff),
        "generated_at": generated_at,
        "mode": "production",
        "win_classifier_used": bool(clf_used),
    }

    # ----- History (bounded) -----
    state.history.append(result)
    state.history = state.history[-500:]

    logging.info(
        "[Predict] %s vs %s (season=%s week=%s) -> home=%.1f away=%.1f win_p=%.3f (clf_used=%s)",
        home_team, away_team, season, week, h_score, a_score, win_prob, clf_used
    )

    return result


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
