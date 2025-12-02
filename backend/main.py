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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple, Literal

from dotenv import load_dotenv
import uvicorn
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import nflreadpy as nfl

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

load_dotenv("./.env")

# -------------------------------------------------------------------
# Config & Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Allow overriding the schedule CSV via env; default to backend/data
schedule_env = os.environ.get("SCHEDULE_PATH")
SCHEDULE_PATH = Path(schedule_env) if schedule_env else (DATA_DIR / "Nfl_schedule_2025.csv")

# Required model keys for /predict to be "ready"
REQUIRED_MODELS: Tuple[str, ...] = ("home", "away", "win")

# Ensure expected folders exist (safe on repeated calls)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_STORAGE = BASE_DIR / "Predictions"
PREDICTION_STORAGE.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------


def _coerce_season_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce season/week columns to integers when present.

    Handles both 'season'/'week' and 'season_num'/'week_num' variants.
    """
    for col in ("season", "season_num"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ("week", "week_num"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _normalize_team_columns(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Uppercase and strip team-related columns for stable matching.

    Default columns include both name and abbreviation variants.
    """
    if cols is None:
        cols = ("home_team", "away_team", "home_abbr", "away_abbr")

    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    return df


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
        
            path = 'NFL_ML_Predictions/backend/data/game_features_20251201.csv'
                # Fallback default filename under data dir
            path = DATA_DIR / "game_features_20251201.csv"

            # If file missing and a DATA_URL is configured, attempt to download it
            if not path.exists():
                data_url = os.environ.get("DATA_URL")
                if data_url:
                    logging.info("[Dataset] No local dataset found. Attempting download from DATA_URL: %s", data_url)
                    try:
                        from urllib.request import urlopen

                        resp = urlopen(data_url, timeout=30)
                        if getattr(resp, 'status', None) and resp.status != 200:  # pragma: no cover - network edge
                            logging.warning("[Dataset] DATA_URL returned status %s", resp.status)
                        # Determine target filename
                        fname = Path(data_url).name or "game_features.csv"
                        dest = DATA_DIR / fname
                        DATA_DIR.mkdir(parents=True, exist_ok=True)
                        with open(dest, "wb") as fh:
                            fh.write(resp.read())
                        path = dest
                        logging.info("[Dataset] Downloaded dataset to %s", dest)
                    except Exception as de:  # pragma: no cover - network errors
                        logging.exception("[Dataset] Failed to download from DATA_URL: %s", de)
                        self.dataset = None
                        return
                else:
                    logging.warning("[Dataset] No dataset found at: %s and no DATA_URL configured", path)
                    self.dataset = None
                    return

            logging.info("[Dataset] Loading dataset from: %s", path)
            df = pd.read_csv(path)

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

# CORS configuration: prefer env list; otherwise default to known origins
_allowed = os.environ.get("ALLOWED_ORIGINS")
if _allowed:
    ALLOWED_ORIGINS: List[str] = [o.strip() for o in _allowed.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = [
        "https://nfl-ml-predictions.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "nfl-predict-christopher-jordons-projects.vercel.app",
        "https://nfl-predict-git-main-christopher-jordons-projects.vercel.app"

    ]

logging.info("[App] CORS allowed origins: %s", ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
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
    Return a normalized list of upcoming games for the next 7-day window.

    Behavior:
      - Prefer using `nflreadpy.load_schedules()`. If that's not available,
        fall back to reading a CSV discovered by `_find_schedule_path()`.
      - Parse kickoff datetimes to timezone-aware UTC `gameday` values.
      - Attach `home_team_logo` and `away_team_logo` when a `team_logo.csv`
        mapping is available in the `backend` data folder.
      - If no games are in the next 7 days, try to return the next future
        slate (earliest future season/week). If none, return the latest
        season/week available in the file.
    """
    # Load schedule dataframe (try nflreadpy first, then CSV fallback)
    try:
        sched = nfl.load_schedules()
        df = sched.to_pandas()
    except Exception:
        sched_path = _find_schedule_path()
        if sched_path is None:
            logging.warning("[Schedule] No schedule source found")
            return []
        try:
            df = pd.read_csv(sched_path)
        except Exception as e:
            logging.exception("[Schedule] Failed to read schedule CSV %s: %s", sched_path, e)
            return []

    if df is None or df.empty:
        return []

    # Normalize and parse
    df = df.copy()
    # Preserve original gameday string (useful for 'TBD' values)
    if "gameday" in df.columns:
        df["_gameday_orig"] = df["gameday"]
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce", utc=True)
    else:
        # try alternate names
        for alt in ("kickoff", "game_date", "date"):
            if alt in df.columns:
                df["gameday"] = pd.to_datetime(df[alt], errors="coerce", utc=True)
                break

    # Ensure season/week numeric
    df = _coerce_season_week(df)

    # Normalize team abbreviations into `home_team` / `away_team`
    if "home_team" not in df.columns and "home" in df.columns:
        df["home_team"] = df["home"]
    if "away_team" not in df.columns and "away" in df.columns:
        df["away_team"] = df["away"]

    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].astype(str).str.strip().str.upper()
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].astype(str).str.strip().str.upper()

    now = datetime.now(timezone.utc)
    window_end = now + timedelta(days=7)

    # Pick games in next 7 days. Use a 1-day grace backwards window to
    # account for timezone mismatches so late-night/early-morning Monday
    # games are not dropped.
    mask = False
    if "gameday" in df.columns:
        mask = (df["gameday"] >= (now - timedelta(days=1))) & (df["gameday"] < window_end)
        upcoming = df[mask].copy()
    else:
        upcoming = pd.DataFrame()

    # If none in the next 7 days, try to find the next future slate (earliest gameday >= now)
    if upcoming.empty:
        future_mask = False
        if "gameday" in df.columns:
            future_mask = df["gameday"] >= now
            if future_mask.any():
                earliest = df.loc[future_mask, "gameday"].min()
                sel = df["gameday"] == earliest
                upcoming = df[sel].copy()

    # If still empty, return the latest season/week in the file
    if upcoming.empty:
        try:
            max_season = int(df["season"].max()) if "season" in df.columns else None
            max_week = int(df["week"].max()) if "week" in df.columns else None
            if max_season is not None and max_week is not None:
                upcoming = df[(df["season"] == max_season) & (df["week"] == max_week)].copy()
            else:
                upcoming = df.head(10).copy()
        except Exception:
            upcoming = df.head(10).copy()

    # Load team logos mapping if available
    logos_path_candidates = [DATA_DIR / "team_logo.csv", BASE_DIR / "team_logo.csv", BASE_DIR / "team_logos.csv"]
    logos_df = None
    for p in logos_path_candidates:
        if p.exists():
            try:
                logos_df = pd.read_csv(p)
                break
            except Exception:
                logos_df = None

    logo_map = {}
    if logos_df is not None:
        # normalize columns (abbr, logo_url) flexible mapping
        cols = {c.lower(): c for c in logos_df.columns}
        abbr_col = cols.get("abbr") or cols.get("team") or cols.get("team_name")
        url_col = cols.get("logo_url") or cols.get("logo") or cols.get("logo_url")
        if abbr_col and url_col and abbr_col in logos_df.columns and url_col in logos_df.columns:
            for _, r in logos_df.iterrows():
                try:
                    logo_map[str(r[abbr_col]).strip().upper()] = r[url_col]
                except Exception:
                    continue

    results: List[Dict[str, Any]] = []
    for _, row in upcoming.iterrows():
        try:
            home = str(row.get("home_team") or row.get("home") or "").strip().upper()
            away = str(row.get("away_team") or row.get("away") or "").strip().upper()
            season_v = int(row.get("season")) if row.get("season") is not None else None
            week_v = int(row.get("week")) if row.get("week") is not None else None
            gameday_val = row.get("gameday")
            # Prefer parsed, timezone-aware gameday; if missing (NaT) try to
            # fall back to the original string value so values like 'TBD'
            # are preserved for the frontend.
            kickoff_iso = None
            try:
                parsed = pd.to_datetime(gameday_val, errors="coerce", utc=True)
                if pd.notnull(parsed):
                    kickoff_iso = parsed.isoformat()
                else:
                    orig = row.get("_gameday_orig") or row.get("kickoff") or row.get("date") or row.get("game_date")
                    if isinstance(orig, str) and orig.strip():
                        kickoff_iso = orig.strip()
                    elif orig is not None and not isinstance(orig, float):
                        kickoff_iso = str(orig)
                    else:
                        kickoff_iso = None
            except Exception:
                kickoff_iso = None

            game_id = f"{season_v}_{week_v}_{home}_{away}" if season_v is not None and week_v is not None else f"{home}_{away}_{_}"

            results.append(
                {
                    "game_day": kickoff_iso,
                    "game_id": game_id,
                    "season": season_v,
                    "week": week_v,
                    "home_team": home,
                    "away_team": away,
                    "home_team_logo": logo_map.get(home),
                    "away_team_logo": logo_map.get(away),
                    # Backwards-compatible aliases expected by the frontend
                    "home_logo": logo_map.get(home),
                    "away_logo": logo_map.get(away),
                }
            )
        except Exception:
            logging.exception("[Schedule] Error while processing schedule row: %s", row)
            continue

    logging.info("[Schedule] Returning %d upcoming games", len(results))
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
# Prediction Endpoint
# -------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest) -> Dict[str, Any]:
    """
    Predict home/away score and win probability for a single game.

    Workflow:
      - Validate that dataset and models are loaded.
      - Normalize request identifiers (team abbreviations, season/week).
      - Lookup the corresponding dataset row.
      - Run the regression models to get scores.
      - Run the classifier (if present) to get win probability; otherwise
        fall back to a logistic transform of the score differential.
      - Record result in in-memory history and return it.
    """
    # Ensure backend is "ready"
    models_ok = all(m in state.models for m in REQUIRED_MODELS)
    if state.dataset is None or not models_ok:
        missing = [m for m in REQUIRED_MODELS if m not in state.models]
        raise HTTPException(
            status_code=503,
            detail=f"Not ready: missing models {missing}",
        )

    # Normalize identifiers for matching against dataset
    home_team = request.home_team.strip().upper()
    away_team = request.away_team.strip().upper()
    season = int(request.season)
    week = int(request.week)

    df = state.dataset
    assert df is not None  # guarded above

    # Attempt to locate the specific game row
    try:
        row = df[
            (df["season"] == season)
            & (df["week"] == week)
            & (df["home_team"] == home_team)
            & (df["away_team"] == away_team)
        ]
        logging.debug(
            "[Predict] Matched rows: %d for season=%s week=%s %s vs %s",
            len(row),
            season,
            week,
            home_team,
            away_team,
        )
    except Exception as e:
        logging.exception("[Predict] Dataset lookup failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Dataset is missing required columns for prediction.",
        ) from e

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail="Game data not found for given season/week/teams",
        )

    # Run models
    try:
        # Prepare two views of the input row:
        #  - full_df: full row with all columns (used by preprocessors)
        #  - numeric_df: numeric-only DataFrame used for direct regressor.predict
        full_df = row.drop(columns=["game_id"]) if "game_id" in row.columns else row
        numeric_df = full_df.select_dtypes(include=[np.number])

        home_model = state.models["home"]
        away_model = state.models["away"]
        win_model = state.models.get("win")

        def _safe_predict(model, full_df, numeric_df, model_name="model"):
            """Predict with graceful fallbacks for NaNs / unfitted preprocessors.

            Strategy:
              1) Try model.predict on the full DataFrame (preserves feature names).
              2) If that fails with a NaN / NotFitted error, try a standalone
                 preprocessor.transform(full_df) (if available) and predict on
                 the resulting array.
              3) If no preprocessor is available or it fails, perform a simple
                 median imputation on numeric columns and retry predict.
            """
            try:
                # Primary path depends on whether the model is a Pipeline
                # (which may include a preprocessor expecting a DataFrame)
                if hasattr(model, 'named_steps') or hasattr(model, 'steps'):
                    # Pipeline: pass full DataFrame so named transformers can run
                    return float(model.predict(full_df)[0])
                else:
                    # Plain estimator (e.g., GradientBoostingRegressor): pass numeric-only array
                    return float(model.predict(numeric_df)[0])
            except Exception as err:
                msg = str(err) or ""
                logging.warning("[Predict] %s.predict failed on full_df: %s", model_name, msg)

                is_nan_err = "nan" in msg.lower() or "missing value" in msg.lower() or "contains nan" in msg.lower()
                is_notfitted = "not fitted" in msg.lower() or "notfittederror" in msg.lower()

                # Only attempt fallbacks for NaN / NotFitted situations — otherwise rethrow
                if not (is_nan_err or is_notfitted):
                    raise

                # Fallback A: use standalone preprocessor (if available) to transform the full DataFrame
                if state.preprocessor is not None:
                    try:
                        X_proc = state.preprocessor.transform(full_df)
                        return float(model.predict(X_proc)[0])
                    except Exception as prep_err:
                        logging.exception("[Predict] standalone preprocessor transform failed for %s: %s", model_name, prep_err)

                # Fallback B: simple median imputation for numeric columns
                try:
                    from sklearn.impute import SimpleImputer

                    # If numeric_df is empty, coerce numeric conversion and try again
                    if numeric_df.empty:
                        try:
                            numeric_df_candidate = full_df.apply(lambda c: pd.to_numeric(c, errors="coerce"))
                        except Exception:
                            numeric_df_candidate = full_df.select_dtypes(include=[np.number])
                    else:
                        numeric_df_candidate = numeric_df

                    if numeric_df_candidate is not None and not numeric_df_candidate.empty:
                        imp = SimpleImputer(strategy="median")
                        X_imp = imp.fit_transform(numeric_df_candidate)
                        return float(model.predict(X_imp)[0])
                except Exception as imp_err:
                    logging.exception("[Predict] numeric imputation fallback failed for %s: %s", model_name, imp_err)

                # No viable fallback — re-raise the original exception
                raise

        h_score = _safe_predict(home_model, full_df, numeric_df, "home_model")
        a_score = _safe_predict(away_model, full_df, numeric_df, "away_model")

        # Calculate win probability (the helper will attempt predict_proba and
        # otherwise fall back to a logistic on point diff)
        try:
            win_prob, clf_used = _calculate_win_probability(win_model, full_df, h_score, a_score)
        except Exception as win_err:
            logging.exception("[Predict] Win-probability calculation failed: %s", win_err)
            # If classifier fails, still return scores with a null probability
            win_prob, clf_used = 0.5, False
    except Exception as model_err:
        logging.exception("[Predict] Model execution failed: %s", model_err)
        msg = str(model_err)
        # Provide friendlier error messages for common sklearn issues
        if isinstance(model_err, ValueError) and "columns are missing" in msg:
            raise HTTPException(
                status_code=400,
                detail=f"Model input mismatch: {msg}",
            ) from model_err

        # Handle NaN / missing-value errors with a concise guidance message
        if "contains nan" in msg.lower() or "input x contains nan" in msg.lower() or "missing value" in msg.lower():
            friendly = (
                "Prediction failed due to missing (NaN) values in feature inputs. "
                "The server attempted automatic fallbacks (preprocessor or median imputation) but they failed. "
                "Ensure the trained preprocessor artifact (backend/models/preprocessor.joblib) is present in deployment or retrain models with an imputer."
            )
            raise HTTPException(status_code=500, detail=friendly) from model_err

        # Default: return a concise error without dumping the sklearn stack trace
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {msg.splitlines()[0] if msg else 'unknown error'}",
        ) from model_err

    game_id = f"{season}_{week}_{home_team}_{away_team}"
    generated_at = datetime.now(timezone.utc)

    result: Dict[str, Any] = {
        "season": season,
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
        "game_id": game_id,
        "home_score": h_score,
        "away_score": a_score,
        "home_win_probability": win_prob,
        "away_win_probability": 1.0 - win_prob,
        "point_diff": h_score - a_score,
        "generated_at": generated_at,
        "mode": "production",
        "win_classifier_used": clf_used,
    }

    # Append to in-memory history (cap size to avoid unbounded growth)
    state.history.append(result)
    state.history = state.history[-500:]

    logging.info(
        "[Predict] %s vs %s (season=%s week=%s) -> home=%.1f away=%.1f win_p=%.3f (clf_used=%s)",
        home_team,
        away_team,
        season,
        week,
        h_score,
        a_score,
        win_prob,
        clf_used,
    )

    return result


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
