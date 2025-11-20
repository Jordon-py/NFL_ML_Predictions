# -*- coding: utf-8 -*-
"""
File: main.py

Purpose:
    FastAPI (v0.110+) entry point for the NFL prediction API. Loads trained
    models, exposes prediction and schedule endpoints, manages CORS, and
    performs startup health checks.

Key Functions:
    - get_current_nfl_context
    - health
    - predict
    - schedule endpoints (next_week, season, current_week)
    - training_status
    - lifespan (async context manager)
    - reload_models (manual reload of model pipelines)

Notes:
    - This version focuses on clearer structure and reduced duplication while
      preserving all public API contracts and behaviours.
    - Endpoint paths, request/response models, and general I/O semantics are
      intentionally unchanged.
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Prefer `nflreadpy` if present in the environment; otherwise fall back to the
# more commonly available `nfl_data_py` package. This keeps the build
# compatible with pydantic v1 (which is pinned by default) while allowing
# deployments that already depend on `nflreadpy` to benefit from its richer
# schedule helpers.
try:  # pragma: no cover - optional dependency
    import nflreadpy as nfl  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        nfl = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration and environment helpers
# ---------------------------------------------------------------------------


class Config:
    """Centralised configuration for paths and environment-derived settings."""

    def __init__(self) -> None:
        # Load environment variables from .env if present.
        load_dotenv()

        self.repo_root = self._resolve_repo_root()
        self.data_dir = self._resolve_data_dir()
        self.models_dir = self._resolve_models_dir()
        self.schedule_file = self._resolve_schedule_file()
        self.cors_origins = self._load_cors_origins()
        self.log_dir = self._resolve_log_dir()

    # ------------------------------------------------------------------ #
    # Path resolution helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_repo_root() -> Path:
        """Resolve repository root assuming this file lives in backend/."""
        here = Path(__file__).resolve()
        return here.parent.parent  # repo_root/backend/main.py -> repo_root

    def _resolve_data_dir(self) -> Path:
        """Resolve the directory that holds primary data artifacts."""
        # Environment override first
        env_dir = os.getenv("DATA_DIR")
        if env_dir:
            p = Path(env_dir).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p

        # Default: backend/data inside the repo
        default = self.repo_root / "backend" / "data"
        default.mkdir(parents=True, exist_ok=True)
        return default

    def _resolve_models_dir(self) -> Path:
        """Resolve the directory containing trained model artifacts."""
        env_dir = os.getenv("MODELS_DIR")
        if env_dir:
            p = Path(env_dir).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p

        # Fallback: backend/models
        fallback = self.repo_root / "backend" / "models"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _resolve_schedule_file(self) -> Path:
        """Resolve primary schedule CSV file.

        Preference order:
            1. Environment variable SCHEDULE_FILE if set.
            2. backend/data/Nfl_schedule_2025.csv
            3. backend/Nfl_schedule_2025.csv
        """
        env_path = os.getenv("SCHEDULE_FILE")
        if env_path:
            return Path(env_path).expanduser().resolve()

        candidates = [
            self.data_dir / "Nfl_schedule_2025.csv",
            self.repo_root / "backend" / "Nfl_schedule_2025.csv",
        ]
        for c in candidates:
            if c.exists():
                return c

        # If nothing exists yet, default to backend/data and allow creation.
        return self.data_dir / "Nfl_schedule_2025.csv"

    def _resolve_log_dir(self) -> Path:
        """Resolve logging directory."""
        env_dir = os.getenv("LOG_DIR")
        if env_dir:
            p = Path(env_dir).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p

        default = self.repo_root / "backend" / "logs"
        default.mkdir(parents=True, exist_ok=True)
        return default

    # ------------------------------------------------------------------ #
    # CORS / environment helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_cors_origins() -> List[str]:
        """Load allowed CORS origins from environment.

        Expected formats:
            - Single origin string
            - Comma-separated list of origins
        """
        raw = os.getenv("CORS_ORIGINS", "")
        if not raw:
            return []

        # Split on comma and strip whitespace.
        return [o.strip() for o in raw.split(",") if o.strip()]

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def resolve_schedule_path(self) -> Path:
        """Public wrapper for the schedule file path."""
        return self.schedule_file


# ---------------------------------------------------------------------------
# Data loading and feature management
# ---------------------------------------------------------------------------


class DataManager:
    """Responsible for loading, validating, and serving the engineered dataset."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._dataset_lock = Lock()
        self._dataset: Optional[pd.DataFrame] = None

    @property
    def dataset(self) -> Optional[pd.DataFrame]:
        """Return a *copy* of the current dataset to avoid accidental mutation."""
        with self._dataset_lock:
            if self._dataset is None:
                return None
            return self._dataset.copy()

    def load_dataset(self) -> Dict[str, Any]:
        """Load the engineered dataset from disk.

        Returns a small status dict for diagnostic use.
        """
        dataset_path = self.config.data_dir / "game_features.csv"
        if not dataset_path.exists():
            return {"status": "error", "reason": f"dataset not found at {dataset_path}"}

        try:
            df = pd.read_csv(dataset_path)
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("data").exception("Failed to read dataset CSV")
            return {"status": "error", "reason": str(exc)}

        df = self._validate_and_clean_dataset(df)

        with self._dataset_lock:
            self._dataset = df

        logging.getLogger("data").info(
            "Loaded dataset from %s with %d rows and %d columns",
            dataset_path,
            df.shape[0],
            df.shape[1],
        )
        return {"status": "ok", "rows": int(df.shape[0]), "cols": int(df.shape[1])}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _validate_and_clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lightweight validation and standardisation for the dataset."""
        required_cols = [
            "season",
            "week",
            "home_team",
            "away_team"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        # Ensure keys are of consistent type.
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

        # Normalise team codes to upper-case strings.
        df["home_team"] = df["home_team"].astype(str).str.upper()
        df["away_team"] = df["away_team"].astype(str).str.upper()


        return df

    # Convenience accessors -------------------------------------------------

    def _ensure_home_away_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure 'home_team'/'away_team' columns exist, derived from codes if needed."""
        if "home_team" not in df.columns and "home_abbr" in df.columns:
            df["home_team"] = df["home_abbr"]
        if "away_team" not in df.columns and "away_team_code" in df.columns:
            df["away_team"] = df["away_team_code"]
        return df


# ---------------------------------------------------------------------------
# Model loading and prediction helpers
# ---------------------------------------------------------------------------


class ModelManager:
    """Load and serve scikit-learn pipelines for home/away scores and win probs."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._models_lock = Lock()
        self.home_pipe: Optional[Any] = None
        self.away_pipe: Optional[Any] = None
        self.win_pipe: Optional[Any] = None
        self.preprocess: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}

    def load_models(self) -> Dict[str, Any]:
        """Load all model pipelines and metadata from disk."""
        models_dir = self.config.models_dir
        def _load_joblib(candidates: List[str] | str) -> Any:
            """Attempt to load a joblib artifact from a ordered list of candidates.

            Candidate resolution tries a few common locations to support different
            packaging patterns (e.g., model artifacts sometimes placed directly
            in ``backend/models`` or nested under ``backend/models/models``).
            """
            if isinstance(candidates, str):
                candidates = [candidates]

            # Search directories in order of likelihood
            search_bases = [models_dir, models_dir / "models"]

            for name in candidates:
                for base in search_bases:
                    path = base / name
                    if path.exists():
                        logging.getLogger("models").debug("Loading model artifact from %s", path)
                        return joblib.load(path)

            # No candidate found — raise a helpful error with attempted locations
            tried = []
            for name in candidates:
                for base in search_bases:
                    tried.append(str(base / name))
            raise FileNotFoundError(f"Model artifact not found; tried: {', '.join(tried)}")

        with self._models_lock:
            try:
                self.home_pipe = _load_joblib(["home_model.joblib", "home_pipe.joblib", "home_model.joblib"])
                self.away_pipe = _load_joblib(["away_model.joblib", "away_pipe.joblib", "away_model.joblib"])
                self.win_pipe = _load_joblib(
                    [
                        "win_clf_calibrated.joblib",
                        "win_model.joblib",
                        "win_clf.joblib",
                        "win_clf_calibrated.joblib",
                    ]
                )

                # Preprocessor/pipeline name has historically changed; try common names
                self.preprocess = _load_joblib([
                    "preprocessor.joblib",
                    "preprocessing_pipeline.joblib",
                    "preprocessing.joblib",
                ])

                meta_path = models_dir / "metadata.json"
                if meta_path.exists():
                    with meta_path.open("r", encoding="utf-8") as f:
                        data: Dict[str, Any] = json.load(f)
                    if "mode" not in data:
                        data["mode"] = os.getenv("APP_MODE", "production")
                    self.metadata = data
                else:
                    self.metadata = {
                        "mode": os.getenv("APP_MODE", "production"),
                        "note": "metadata.json missing; using defaults",
                    }

                logging.getLogger("models").info(
                    "Loaded models from %s", models_dir
                )
                return {"status": "ok", "models_dir": str(models_dir)}
            except Exception as exc:  # pragma: no cover - defensive
                logging.getLogger("models").exception("Failed to load models")
                self.home_pipe = None
                self.away_pipe = None
                self.win_pipe = None
                self.preprocess = None
                return {"status": "error", "reason": str(exc)}

    def predict_scores_and_win_prob(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the three model heads on the given feature frame."""
        if self.preprocess is not None:
            X_proc = self.preprocess.transform(X)
        else:
            X_proc = X

        if self.home_pipe is None or self.away_pipe is None or self.win_pipe is None:
            raise RuntimeError("Models not loaded")

        home_scores = self.home_pipe.predict(X_proc)
        away_scores = self.away_pipe.predict(X_proc)

        # win_pipe is a calibrated classifier; we take the probability for 'home win'.
        win_proba = self.win_pipe.predict_proba(X_proc)[:, 1]

        return home_scores, away_scores, win_proba


# ---------------------------------------------------------------------------
# Pydantic models for API contracts (unchanged)
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    season: int = Field(..., ge=2000, le=2100)
    week: int = Field(..., ge=1, le=22)
    home_team: str
    away_team: str
    win_threshold: float = Field(0.7, ge=0.5, le=1.0)


class PredictionResponse(BaseModel):
    home_score: float
    away_score: float
    home_win_probability: float
    away_win_probability: float
    point_diff: float
    mode: str
    prediction_source: str
    win_classifier_used: str
    win_probability_source: str
    win_threshold_used: float
    confidence_score: float


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
    win_threshold_used: float
    predicted_winner: str
    kickoff: Optional[str] = None
    actual_home_score: Optional[float] = None
    actual_away_score: Optional[float] = None
    actual_winner: Optional[str] = None
    confidence_score: float


# ---------------------------------------------------------------------------
# Utility: current NFL context helper
# ---------------------------------------------------------------------------


def get_current_nfl_context() -> Dict[str, Any]:
    """Return a coarse view of the current NFL season/week context.

    This is best-effort and only used for health/debug endpoints; prediction
    endpoints rely entirely on the engineered dataset and schedule CSV.
    """
    now = datetime.now(timezone.utc)

    # Season typically runs Aug/Sept (season label is the year it starts)
    cur_season = now.year if now.month >= 8 else now.year - 1

    # If nfl backend is available, prefer its helper if present
    if nfl is not None:
        for fn in ("get_current_season", "get_current_week"):
            if not hasattr(nfl, fn):
                break
        else:
            try:
                season_loc = int(nfl.get_current_season())
                week_loc = int(nfl.get_current_week())
                return {
                    "season": season_loc,
                    "approx_week": week_loc,
                    "phase": "calendar",
                }
            except Exception:  # pragma: no cover - defensive
                pass

    # Fallback approximation only
    month = now.month
    if month < 2:
        phase = "playoffs"
        approx_week = 19
    elif month < 8:
        phase = "offseason"
        approx_week = 0
    else:
        phase = "regular"
        approx_week = min(18, 1 + (now - datetime(now.year, 9, 1, tzinfo=timezone.utc)).days // 7)

    return {
        "season": cur_season,
        "approx_week": approx_week,
        "phase": phase,
    }


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------


def _select_schedule_scope(df: pd.DataFrame):
    """Choose which season/week to expose via `/schedule/next-week`.

    Preference order:
        1. Earliest kickoff time that has not yet started (within a small grace window).
        2. Calendar context (current season + nearest upcoming week).
        3. Latest season/week present in the dataset.

    Returns (season, week, metadata).
    """
    now = datetime.now(timezone.utc)
    # Log the selection time once for observability without extra temp variables.
    logging.getLogger("schedule").info(
        "select_schedule_scope now=%s", now.isoformat()
    )

    if df is None or df.empty:
        # Fallback: read the default schedule file when no data was provided.
        df = pd.read_csv("./Nfl_schedule_2025.csv")

    working = df.copy()

    selection: Dict[str, Any] = {"strategy": "unknown"}

    def _as_int(value: Any) -> Optional[int]:
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    # 1) Use kickoff timestamps when available so "next week" truly means upcoming games.
    if "gameday" in working.columns:
        try:
            working["gameday_dt"] = pd.to_datetime(
                working["gameday"], errors="coerce", utc=True
            )
        except Exception:  # pragma: no cover - defensive
            working["gameday_dt"] = pd.NaT

        mask_future = working["gameday_dt"] >= (now - timedelta(hours=3))
        future_games = working.loc[mask_future].copy()
        if not future_games.empty:
            earliest = future_games["gameday_dt"].min()
            scope_rows = future_games.loc[future_games["gameday_dt"] == earliest].copy()
            if not scope_rows.empty:
                # Prefer numeric season/week columns from the dataset when available.
                try:
                    season_val = scope_rows["season"].iloc[0]
                    week_val = scope_rows["week"].iloc[0]
                    season = _as_int(season_val)
                    week = _as_int(week_val)
                except Exception:  # pragma: no cover - defensive
                    season = None
                    week = None

                if season is not None and week is not None:
                    selection.update(
                        {
                            "strategy": "kickoff",
                            "gameday_dt": earliest.isoformat() if pd.notna(earliest) else None,
                        }
                    )
                    return season, week, selection

    # 2) Fall back to a calendar-aware view using current NFL context.
    context = get_current_nfl_context()
    cur_season = context["season"]
    approx_week = context["approx_week"]

    # Prefer numeric season/week columns from the dataset when available.
    # This avoids calling functions on optional NFL backends that may not
    # expose the same helper APIs (e.g., `nfl_data_py` vs `nflreadpy`).
    try:
        if "season" in working.columns:
            working["season_num"] = pd.to_numeric(working["season"], errors="coerce")
        elif "season_num" in working.columns:
            working["season_num"] = pd.to_numeric(
                working["season_num"], errors="coerce"
            )
        else:
            working["season_num"] = pd.NA

        if "week" in working.columns:
            working["week_num"] = pd.to_numeric(working["week"], errors="coerce")
        elif "week_num" in working.columns:
            working["week_num"] = pd.to_numeric(working["week_num"], errors="coerce")
        else:
            working["week_num"] = pd.NA
    except Exception:  # pragma: no cover - defensive
        working["season_num"] = pd.NA
        working["week_num"] = pd.NA

    # Try to find games that align with the approximated context.
    mask_context = (working["season_num"] == cur_season) & (
        working["week_num"] >= approx_week
    )
    context_games = working.loc[mask_context].copy()
    if not context_games.empty:
        week_val = int(context_games["week_num"].min())
        selection.update(
            {
                "strategy": "calendar",
                "context": context,
            }
        )
        return cur_season, week_val, selection

    # 3) Last resort: choose the latest week in the dataset for the latest season.
    valid_rows = working.dropna(subset=["season_num", "week_num"])
    if not valid_rows.empty:
        last_season = int(valid_rows["season_num"].max())
        last_week = int(
            valid_rows.loc[valid_rows["season_num"] == last_season, "week_num"].max()
        )
        selection.update(
            {
                "strategy": "latest",
            }
        )
        return last_season, last_week, selection

    # If all strategies fail, surface a clear error.
    raise ValueError("Unable to derive schedule scope from dataset; no valid rows found.")


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    """Container for app-wide state: config, data, models, and prediction history."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_manager = DataManager(config)
        self.model_manager = ModelManager(config)
        self.prediction_history: List[PredictionHistoryEntry] = []
        self.history_lock = Lock()

    @property
    def data(self) -> pd.DataFrame:
        df = self.data_manager.dataset
        if df is None:
            raise RuntimeError("Dataset not loaded")
        return df

    def initialize(self) -> Dict[str, Any]:
        """Load dataset and models at startup."""
        status: Dict[str, Any] = {"dataset": None, "models": None}

        try:
            self.data_manager.load_dataset()
            status["dataset"] = {"status": "ok"}
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("startup").exception("Dataset load failed")
            status["dataset"] = {"status": "error", "reason": str(exc)}

        try:
            # Load models eagerly during startup so /health and CI checks can
            # validate that the models were loaded successfully. This avoids
            # returning a terse False value in the startup status.
            model_result = self.model_manager.load_models()
            status["models"] = model_result
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("startup").exception("Model load failed")
            status["models"] = {"status": "error", "reason": str(exc)}

        return status

    # ------------------------------------------------------------------ #
    # Prediction history
    # ------------------------------------------------------------------ #

    def record_prediction(
        self,
        request: PredictionRequest,
        response: PredictionResponse,
        game_row: Optional[pd.Series] = None,
    ) -> None:
        """Append a prediction to in-memory history and persist a small CSV snapshot."""
        game_id = (
            f"{request.season}_{request.week}_{request.home_team}_{request.away_team}"
        )
        kickoff: Optional[str] = None
        if game_row is not None and "home_game_date" in game_row.index:
            kickoff = str(game_row["home_game_date"])

        predicted_winner = (
            request.home_team
            if response.home_win_probability >= response.away_win_probability
            else request.away_team
        )

        entry = PredictionHistoryEntry(
            timestamp=datetime.now(timezone.utc),
            game_id=game_id,
            season=request.season,
            week=request.week,
            home_team=request.home_team,
            away_team=request.away_team,
            home_score_pred=response.home_score,
            away_score_pred=response.away_score,
            home_win_probability=response.home_win_probability,
            away_win_probability=response.away_win_probability,
            point_diff=response.point_diff,
            mode=response.mode,
            prediction_source=response.prediction_source,
            win_threshold_used=response.win_threshold_used,
            predicted_winner=predicted_winner,
            kickoff=kickoff,
            actual_home_score=None,
            actual_away_score=None,
            actual_winner=None,
            confidence_score=response.confidence_score,
        )

        with self.history_lock:
            # Keep an in-memory record for quick access.
            self.prediction_history.append(entry)

            # Best-effort: persist the latest prediction to a CSV for external inspection.
            try:
                pd.DataFrame([entry.dict()]).to_csv(
                    self.config.data_dir / "latest_prediction.csv",
                    index=False,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logging.getLogger("history").warning(
                    "Failed to persist latest prediction: %s", exc
                )


# ---------------------------------------------------------------------------
# FastAPI application factory and lifespan management
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create FastAPI application with enhanced configuration."""

    # Initialize configuration
    config = Config()

    # Setup logging
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(config.log_dir / "api.log"),
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "encoding": "utf-8",
                },
            },
            "root": {"level": "DEBUG", "handlers": ["console", "file"]},
        }
    )

    log = logging.getLogger("api")

    # Create application state
    app_state = AppState(config)

    # ------------------------------------------------------------------ #
    # Lifespan manager
    # ------------------------------------------------------------------ #

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        """Application startup/shutdown context manager.

        Loads dataset and models up front so that the first prediction
        request does not incur heavy I/O.
        """
        log.info("Starting up NFL prediction API")
        status = app_state.initialize()
        log.info("Startup status: %s", status)
        try:
            yield
        finally:
            log.info("Shutting down NFL prediction API")

    app = FastAPI(lifespan=lifespan)

    # ------------------------------------------------------------------ #
    # CORS configuration
    # ------------------------------------------------------------------ #

    # Only restrict CORS if origins are explicitly configured; otherwise allow
    # all origins (suitable for local development). This preserves original
    # behaviour but makes the logic slightly easier to read.
    if config.cors_origins:
        allowed = config.cors_origins
    else:
        allowed = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Static files (if any compiled frontend assets exist)
    # ------------------------------------------------------------------ #

    static_dir = config.repo_root / "frontend" / "dist"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ------------------------------------------------------------------ #
    # Health and debug endpoints
    # ------------------------------------------------------------------ #

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Return health and configuration diagnostics."""
        dataset_status = (
            {"status": "unloaded"} if app_state.data_manager.dataset is None else {"status": "loaded"}
        )
        models_loaded = all(
            [
                app_state.model_manager.home_pipe is not None,
                app_state.model_manager.away_pipe is not None,
                app_state.model_manager.win_pipe is not None,
            ]
        )
        model_status = (
            {"status": "loaded", "metadata": app_state.model_manager.metadata}
            if models_loaded
            else {"status": "unloaded"}
        )

        try:
            repo_root_path = Path(app_state.data_manager.config.repo_root)
            repo_root_exists = repo_root_path.exists()
        except Exception:
            repo_root_exists = False

        context = get_current_nfl_context()

        return {
            "status": "ok",
            "mode": app_state.model_manager.metadata.get("mode", "unknown"),
            "dataset": dataset_status,
            "models": model_status,
            "repo_root_exists": repo_root_exists,
            "current_nfl_context": context,
        }

    # ------------------------------------------------------------------ #
    # Schedule endpoints
    # ------------------------------------------------------------------ #

    @app.get("/schedule/next-week")
    async def get_next_week_schedule() -> List[Dict[str, Any]]:
        """Return the upcoming week's schedule as a simple list of games.

        This primarily exists to support the React dashboard. Instead of
        blindly returning the smallest week in the dataset (which skewed toward
        archival 2018 rows), the handler relies on kickoff timestamps and a
        calendar-aware fallback to surface the true "next" slate.
        """
        # Resolve schedule path via Config so tests and deployments pick up
        # the correct file whether it's stored in backend/data or as an
        # application-specific artifact.
        schedule_path = str(config.resolve_schedule_path())
        log.debug("Using schedule path: %s", schedule_path)
        try:
            df = pd.read_csv(schedule_path, parse_dates=True)
        except FileNotFoundError as exc:
            log.error(
                "Schedule file not found at %s; expected file: backend/Nfl_schedule_2025.csv",
                schedule_path,
            )
            raise HTTPException(
                503,
                f"Schedule file not found: {schedule_path}. Expected: backend/Nfl_schedule_2025.csv",
            ) from exc

        try:
            target_season, target_week, selection = _select_schedule_scope(df)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise HTTPException(503, str(exc)) from exc

        numeric_season = pd.to_numeric(df.get("season"), errors="coerce")  # type: ignore
        numeric_week = pd.to_numeric(df.get("week"), errors="coerce")  # type: ignore
        week_rows = df.loc[
            (numeric_season == target_season) & (numeric_week == target_week)
        ].copy()
        if week_rows.empty:
            raise HTTPException(
                404,
                f"No games found for season {target_season} week {target_week}",
            )

        week_rows = week_rows.assign(
            season_num=pd.to_numeric(week_rows.get("season"), errors="coerce"),
            week_num=pd.to_numeric(week_rows.get("week"), errors="coerce"),
        )

        logging.getLogger("api").info(
            "Schedule scope -> season=%s week=%s via %s",
            target_season,
            target_week,
            selection.get("strategy"),
        )

        tm_logo = pd.read_csv("backend/team_logo.csv")
        # Build a lookup of team abbreviation -> display name and logo URL for fast mapping.
        logo_map = {
            str(row["abbr"]).upper(): {
                "team_name": row["team_name"],
                "logo_url": row["logo_url"],
            }
            for _, row in tm_logo.iterrows()
        }

        # Normalise into a compact schedule shape for the frontend.
        games: List[Dict[str, Any]] = []
        for _, row in week_rows.iterrows():
            kickoff_iso: Optional[str] = None
            if "gameday" in row.index:
                kickoff_dt = pd.to_datetime(
                    row.get("gameday"), errors="coerce", utc=True
                )
                kickoff_iso = (
                    kickoff_dt.isoformat() if pd.notna(kickoff_dt) else None
                )

            season_val = row.get("season_num")
            season_val = target_season if pd.isna(season_val) else int(season_val)

            week_val = row.get("week_num")
            week_val = target_week if pd.isna(week_val) else int(week_val)

            home_raw = str(row.get("home_team", "")).upper()
            away_raw = str(row.get("away_team", "")).upper()

            home_info = logo_map.get(home_raw)
            away_info = logo_map.get(away_raw)

            home_team = str(home_info["team_name"]) if home_info else home_raw
        
            home_logo = home_info["logo_url"] if home_info else None

            away_team = str(away_info["team_name"]) if away_info else away_raw
            away_logo = away_info["logo_url"] if away_info else None

            games.append(
                {
                    "season": season_val,
                    "week": week_val,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_logo": home_logo,
                    "away_logo": away_logo,
                    "game_id": str(
                        row.get(
                            "game_id",
                            f"{season_val}_{week_val}_{home_team}_{away_team}",
                        )
                    ),
                    "kickoff": kickoff_iso,
                    "venue": row.get("stadium_name") or row.get("stadium"),
                    "network": row.get("tv_network") or row.get("network"),
                }
            )

        return games

    # Optional debug endpoint to inspect CORS configuration (helpful for CI/deploys)
    @app.get("/debug")
    async def debug_config() -> Dict[str, Any]:
        """Return a small snapshot of current configuration."""
        return {
            "repo_root": str(config.repo_root),
            "data_dir": str(config.data_dir),
            "models_dir": str(config.models_dir),
            "schedule_file": str(config.schedule_file),
            "cors_origins": config.cors_origins,
        }

    # ------------------------------------------------------------------ #
    # Prediction history endpoint
    # ------------------------------------------------------------------ #

    @app.get("/history")
    async def history() -> List[PredictionHistoryEntry]:
        """Return the in-memory prediction history."""
        with app_state.history_lock:
            return list(app_state.prediction_history)

    # ------------------------------------------------------------------ #
    # Prediction endpoint
    # ------------------------------------------------------------------ #

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """Prediction endpoint with validation and simplified model/dataset checks."""
        # Validate application state: lazily ensure models and dataset are available.
        if not all(
            [
                app_state.model_manager.home_pipe,
                app_state.model_manager.away_pipe,
                app_state.model_manager.win_pipe,
            ]
        ):
            load_result = app_state.model_manager.load_models()
            if load_result.get("status") == "error" or not all(
                [
                    app_state.model_manager.home_pipe,
                    app_state.model_manager.away_pipe,
                    app_state.model_manager.win_pipe,
                ]
            ):
                raise HTTPException(503, "Models not loaded")

        if (
            app_state.data_manager.dataset is None
            and app_state.data_manager.load_dataset().get("status") != "ok"
        ):
            raise HTTPException(503, "Dataset not available")

        # Locate the corresponding game row in the engineered dataset.
        df = app_state.data
        df = app_state.data_manager._ensure_home_away_columns(df)

        try:
            mask = (
                (df["season"] == request.season)
                & (df["week"] == request.week)
                & (df["home_team"] == request.home_team.upper())
                & (df["away_team"] == request.away_team.upper())
            )
        except KeyError as exc:
            # Dataset missing required identifiers; treat as server misconfiguration.
            raise HTTPException(
                500, f"Dataset missing required column: {exc}"
            ) from exc

        game_rows = df.loc[mask]
        if game_rows.empty:
            raise HTTPException(
                404,
                (
                    f"Game not found in dataset for {request.season} week {request.week} "
                    f"({request.away_team} at {request.home_team})"
                ),
            )

        game_row = game_rows.iloc[0]

        # Build feature frame (single-row DataFrame for the model pipelines).
        feature_frame = game_row.to_frame().T

        try:
            home_scores, away_scores, win_proba = app_state.model_manager.predict_scores_and_win_prob(
                feature_frame
            )
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("predict").exception("Model prediction failed")
            raise HTTPException(500, "Internal prediction error") from exc

        home_score = float(home_scores[0])
        away_score = float(away_scores[0])
        home_win_probability = float(win_proba[0])
        away_win_probability = float(1.0 - home_win_probability)
        point_diff = home_score - away_score

        threshold = float(request.win_threshold)
        prediction_source = "calibrated_win_model"
        win_classifier_used = "calibrated"
        win_probability_source = "home_win_probability"

        confidence_score = (
            home_win_probability
            if home_win_probability >= threshold
            else away_win_probability
        )

        response = PredictionResponse(
            home_score=home_score,
            away_score=away_score,
            home_win_probability=home_win_probability,
            away_win_probability=away_win_probability,
            point_diff=point_diff,
            mode=app_state.model_manager.metadata.get("mode", "production"),
            prediction_source=prediction_source,
            win_classifier_used=win_classifier_used,
            win_probability_source=win_probability_source,
            win_threshold_used=threshold,
            confidence_score=confidence_score,
        )

        # Record prediction in in-memory history; failures here should not
        # affect the main prediction response.
        try:
            app_state.record_prediction(request, response, game_row)
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("history").warning(
                "Failed to record prediction history: %s", exc
            )

        return response

    return app


# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
