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

Enhanced version: provides improved data cohesion, validation, and
application state management while preserving the original API contracts.
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
from tkinter import N
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from venv import logger

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from .train_models import CORS_ORIGINS

# ---------------------------------------------------------------
# Enhanced Data Models with Strict Validation
# ---------------------------------------------------------------
class PredictionRequest(BaseModel):
    home_team: str = Field(..., min_length=2, max_length=50)
    away_team: str = Field(..., min_length=2, max_length=50)
    season: int = Field(..., ge=2000, le=2100)
    week: int = Field(..., ge=1, le=22)

    @validator('home_team', 'away_team')
    def validate_team_names(cls, v):
        if not v or not v.strip():
            raise ValueError('Team name cannot be empty')
        # Normalize to uppercase abbreviations to match dataset/team keys.
        return v.strip().upper()

class PredictionResponse(BaseModel):
    home_score: float = Field(..., ge=0)
    away_score: float = Field(..., ge=0)
    home_win_probability: float = Field(..., ge=0, le=1)
    away_win_probability: float = Field(..., ge=0, le=1)
    home_win: float = Field(..., ge=0, le=1)
    away_win: float = Field(..., ge=0, le=1)
    point_diff: float
    mode: str
    prediction_source: str
    win_classifier_used: bool
    win_probability_source: str
    win_threshold_used: Optional[float] = Field(None, ge=0, le=1)
    confidence_score: Optional[float] = Field(None, ge=0, le=1)  # New: prediction confidence

class HealthResponse(BaseModel):
    status: str
    mode: str
    reason: str
    timestamp: datetime
    components: Dict[str, bool]  # Detailed component health

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
    win_threshold_used: Optional[float]
    predicted_winner: str
    kickoff: Optional[str] = None
    actual_home_score: Optional[float] = None
    actual_away_score: Optional[float] = None
    actual_winner: Optional[str] = None
    confidence_score: Optional[float] = None  # New field


def get_current_nfl_context(reference: Optional[datetime] = None) -> Dict[str, Any]:
    """Return the current NFL season context for schedule heuristics.

    Repository Guardian note: this helper centralizes the once-inline
    "what week is it?" logic so that schedule endpoints can stay synced to the
    real calendar rather than defaulting to the first row of the dataset.
    """

    now = reference or datetime.now(timezone.utc)

    # NFL season flips in August; anything earlier belongs to the prior year.
    season = now.year if now.month >= 8 else now.year - 1

    if now.month in {1, 2}:
        phase = "postseason"
    elif 3 <= now.month <= 7:
        phase = "offseason"
    elif now.month == 8:
        phase = "preseason"
    else:
        phase = "regular"

    season_start = datetime(season, 9, 1, tzinfo=timezone.utc)
    approx_week = 1
    if now >= season_start:
        weeks_since = ((now - season_start).days // 7) + 1
        approx_week = max(1, min(22, weeks_since))
    elif phase == "postseason":
        approx_week = 21

    return {
        "season": season,
        "phase": phase,
        "approx_week": int(approx_week),
        "timestamp": now.isoformat(),
    }


def _select_schedule_scope(df: pd.DataFrame, now: Optional[datetime] = None) -> Tuple[int, int, Dict[str, Any]]:
    """Choose which season/week to expose via `/schedule/next-week`.

    Preference order:
        1. Earliest kickoff time that has not yet started (within a small grace window).
        2. Calendar context (current season + nearest upcoming week).
        3. Latest season/week present in the dataset.

    Returns (season, week, metadata).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    logger.info(msg=now)
    time = now.isoformat(timespec='auto')
    date = now.isoformat()
    print(time)
    print(f'NOW: MAIN.PY LINE 165  : { now }')


    if df is None or df.empty:
        df = pd.read_csv('backend/data/Nfl_schedule_2025_2026.csv')
    
    


    
    working = df.copy()
    # Normalised numeric season/week columns used by the selection logic below.
    working["season_num"] = pd.to_numeric(working.get("season"), errors="coerce")  # type: ignore
    working["week_num"] = pd.to_numeric(working.get("week"), errors="coerce")  # type: ignore


    selection: Dict[str, Any] = {"strategy": "unknown"}

    def _as_int(value: Any) -> Optional[int]:
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    # 1) Use kickoff timestamps when available so "next week" truly means upcoming games.
    if "home_game_date" in working.columns:
        kickoff = pd.to_datetime(working["home_game_date"], errors="coerce", utc=True)
        working["kickoff_dt"] = kickoff
        grace_window = timedelta(hours=4)
        future_rows = (
            working.loc[(kickoff.notna()) & (kickoff >= now - grace_window)]
            .sort_values("kickoff_dt")
        )
        if not future_rows.empty:
            row = future_rows.iloc[0]
            season_val = _as_int(row.get("season_num"))
            week_val = _as_int(row.get("week_num"))
            if season_val is not None and week_val is not None:
                selection.update(
                    {
                        "strategy": "kickoff_date",
                        "note": "nearest future kickoff",
                        "kickoff": row["kickoff_dt"].isoformat()
                        if pd.notna(row["kickoff_dt"])
                        else None,
                    }
                )
                return season_val, week_val, selection

    # 2) Fall back to calendar context when kickoff data is insufficient.
    context = get_current_nfl_context(now)
    context_season = context["season"]
    approx_week = int(context["approx_week"])
    season_rows = working.loc[working["season_num"] == context_season]
    if not season_rows.empty:
        raw_weeks = {_as_int(val) for val in season_rows["week_num"] if not pd.isna(val)}
        weeks = sorted(w for w in raw_weeks if w is not None)
        if weeks:
            week_candidates = [w for w in weeks if w >= approx_week]
            target_week = week_candidates[0] if week_candidates else weeks[-1]
            selection.update(
                {
                    "strategy": "calendar_context",
                    "phase": context["phase"],
                    "approx_week": approx_week,
                }
            )
            return context_season, target_week, selection

    # 3) Absolute fallback: latest available season/week in the dataset.
    seasons = sorted(
        s for s in {_as_int(val) for val in working["season_num"].dropna()}
        if s is not None
    )
    if seasons:
        fallback_season = seasons[-1]
        rows = working.loc[working["season_num"] == fallback_season]
        weeks = sorted(
            w for w in {_as_int(val) for val in rows["week_num"].dropna()}
            if w is not None
        )
        if weeks:
            selection.update(
                {
                    "strategy": "dataset_tail",
                    "note": "using latest available season/week",
                }
            )
            return fallback_season, weeks[-1], selection

    raise ValueError("Unable to determine target week from dataset")

# ---------------------------------------------------------------
# Configuration & Constants with Better Validation
# ---------------------------------------------------------------
class Config:
    """Centralized configuration with validation"""
    
    def __init__(self):
        self.backend_dir = Path(__file__).parent.resolve()
        self.repo_root = self.backend_dir.parent
        self._load_environment()
        
        # Paths with validation
        self.data_dir = self.backend_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.backend_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self._resolve_models_dir()
        self._validate_paths()
        
    def _load_environment(self):
        """Load environment variables with fallbacks"""
        env_files = [
            self.backend_dir / ".env",
            self.repo_root / ".env"
        ]
        
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(dotenv_path=env_file)
                break
    
    def _resolve_models_dir(self) -> Path:
        """Resolve models directory with enhanced validation"""
        # Environment override
        env_path = os.getenv("MODELS_DIR", "").strip()
        if env_path:
            candidate = Path(env_path)
            if candidate.is_dir():
                return candidate
            logging.getLogger("config").warning(f"MODELS_DIR={env_path} does not exist")
        
        # Legacy location
        legacy = self.backend_dir / "models"
        if legacy.is_dir():
            return legacy
        
        # Latest date-stamped directory
        candidates = []
        for child in self.backend_dir.iterdir():
            if (child.is_dir() and child.name.isdigit() and len(child.name) == 8):
                models_sub = child / "models"
                if models_sub.is_dir():
                    candidates.append((models_sub, models_sub.stat().st_mtime))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        # Fallback
        fallback = "backend/models"
        Path(fallback).mkdir(parents=True, exist_ok=True)
        return Path(fallback)
    
    def _validate_paths(self):
        """Validate critical paths exist"""
        required_paths = {
            "data_dir": self.data_dir,
            "log_dir": self.log_dir,
            "models_dir": self.models_dir
        }
        
        for name, path in required_paths.items():
            if not path.exists():
                logging.getLogger("config").warning(f"Required path does not exist: {name}={path}")

# ---------------------------------------------------------------
# Enhanced Data Manager Class
# ---------------------------------------------------------------
class DataManager:
    """Manages dataset loading, validation, and access with thread safety"""
    
    def __init__(self, config: Config):
        self.config = config
        self._dataset: Optional[pd.DataFrame] = None
        self._last_loaded: Optional[datetime] = None
        self._lock = Lock()
        self._active_path: Optional[str] = None
        
    def load_dataset(self) -> bool:
        """Load dataset with enhanced validation"""
        with self._lock:
            try:
                dataset_path = self._resolve_dataset_path()
                
                if not dataset_path.exists():
                    logging.error(f"Dataset file not found: {dataset_path}")
                    return False
                
                df = pd.read_csv(dataset_path)
                if df.empty:
                    logging.error("Dataset is empty")
                    return False
                
                # Enhanced validation
                df = self._validate_and_clean_dataset(df)
                if df is None:
                    return False
                
                self._dataset = df
                self._last_loaded = datetime.now(timezone.utc)
                self._active_path = str(dataset_path)
                
                logging.info(f"Dataset loaded successfully: {len(df)} rows, {df.shape[1]} columns")
                return True
                
            except Exception as e:
                logging.error(f"Failed to load dataset: {e}", exc_info=True)
                return False
    
    def _resolve_dataset_path(self) -> Path:
        """Resolve dataset path with fallbacks"""
        env_path = os.getenv("DATASET_PATH", "").strip()
        if env_path:
            candidate = Path(env_path)
            if candidate.exists():
                return candidate
        
        # Try common filenames
        default_candidates = [
            self.config.data_dir / "game_features_20251114.csv",
            self.config.data_dir / "game_features.csv",
            self.config.data_dir / "merged_game_features.csv"
        ]
        
        for candidate in default_candidates:
            if candidate.exists():
                return candidate
        
        # Return default even if it doesn't exist (for error handling)
        return default_candidates[0]
    
    def _validate_and_clean_dataset(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Enhanced dataset validation and cleaning"""
        try:
            # Basic cleaning
            df = df.dropna(how='all')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Validate required columns
            required_columns = {'home_team', 'away_team', 'season', 'week'}
            missing_required = required_columns - set(df.columns)
            if missing_required:
                logging.error(f"Missing required columns: {missing_required}")
                return None
            
            # Ensure home/away columns exist
            df = self._ensure_home_away_columns(df)
            
            # Validate data types
            if 'season' in df.columns:
                df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)
            if 'week' in df.columns:
                df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
            
            return df
            
        except Exception as e:
            logging.error(f"Dataset validation failed: {e}", exc_info=True)
            return None
    
    def _ensure_home_away_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure home/away columns exist with proper formatting"""
        if {'home_team', 'away_team'}.issubset(df.columns):
            return df
            
        if {'team', 'opponent_team', 'is_home'}.issubset(df.columns):
            is_home = df['is_home'].astype(str).str.strip().str.lower().isin({'true', 't', '1', 'yes', 'y'})
            df = df.assign(
                home_team=np.where(is_home, df['team'], df['opponent_team']),
                away_team=np.where(is_home, df['opponent_team'], df['team'])
            )
        
        return df
    
    @property
    def dataset(self) -> Optional[pd.DataFrame]:
        with self._lock:
            return self._dataset.copy() if self._dataset is not None else None
    
    @property
    def status(self) -> Dict[str, Any]:
        with self._lock:
            rows = self._dataset.shape[0] if self._dataset is not None else 0
            cols = self._dataset.shape[1] if self._dataset is not None else 0
            return {
                "path": self._active_path,
                "rows": rows,
                "columns": cols,
                "loaded": self._dataset is not None and not self._dataset.empty,
                "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None
            }

# ---------------------------------------------------------------
# Enhanced Model Manager
# ---------------------------------------------------------------
class ModelManager:
    """Manages model loading, validation, and inference"""
    
    def __init__(self, config: Config):
        self.config = config
        self.home_pipe = None
        self.away_pipe = None
        self.win_pipe = None
        self.metadata = None
        self.feature_columns = None
        self._load_lock = Lock()
        
    def load_models(self) -> Dict[str, Any]:
        """Load models with enhanced validation"""
        with self._load_lock:
            result = {"status": "success", "loaded": []}
            
            try:
                # Load pipelines
                self.home_pipe, self.away_pipe, self.win_pipe = self._load_pipelines()
                if all(p is not None for p in [self.home_pipe, self.away_pipe, self.win_pipe]):
                    result["loaded"].extend(["home_pipe", "away_pipe", "win_pipe"])
                else:
                    result["status"] = "partial"
                
                # Load metadata
                self.metadata = self._load_metadata()
                if self.metadata:
                    result["loaded"].append("metadata")
                
                # Load feature columns
                self.feature_columns = self._load_feature_columns()
                if self.feature_columns:
                    result["loaded"].append("feature_columns")
                
                # Validate model coherence
                if result["status"] == "success":
                    self._validate_model_coherence()
                
                logging.info(f"Model loading completed: {result}")
                return result
                
            except Exception as e:
                logging.error(f"Model loading failed: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}
    
    def _load_pipelines(self) -> Tuple[Any, Any, Any]:
        """Load model pipelines with error handling.

        Repository Guardian note: Earlier revisions expected legacy filenames
        (`home_pipe.joblib`, etc.) that no longer exist in the models directory,
        generating noisy startup errors even though the modern artifacts were in
        place (`home_model.joblib`, `win_clf_calibrated.joblib`). This loader now
        resolves each logical pipeline against a list of known filenames so that
        startup succeeds without manual symlinks.
        """
        pipelines: Dict[str, Any] = {}

        filename_candidates = {
            "home_pipe": ["home_pipe.joblib", "home_model.joblib"],
            "away_pipe": ["away_pipe.joblib", "away_model.joblib"],
            "win_pipe": ["win_pipe.joblib", "win_model.joblib", "win_clf_calibrated.joblib"],
        }

        for name, candidates in filename_candidates.items():
            pipelines[name] = None
            for candidate in candidates:
                pipeline_path = self.config.models_dir / candidate
                if not pipeline_path.exists():
                    continue
                try:
                    pipelines[name] = joblib.load(pipeline_path)
                    logging.info("✓ Loaded %s from %s", name, pipeline_path.name)
                    break
                except Exception as exc:  # pragma: no cover - defensive load guard
                    logging.error("✗ Failed to load %s from %s: %s", name, pipeline_path, exc)

            if pipelines[name] is None:
                logging.error(
                    "✗ Pipeline not found for %s (looked for: %s)",
                    name,
                    ", ".join(candidates),
                )

        return pipelines["home_pipe"], pipelines["away_pipe"], pipelines["win_pipe"]

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load training metadata from metadata.json if present.

        The metadata file describes which feature columns were used during
        training and includes helpful diagnostics (e.g. optimal win threshold).
        """
        meta_path = self.config.models_dir / "metadata.json"
        if not meta_path.exists():
            logging.getLogger("models").warning("metadata.json not found in %s", self.config.models_dir)
            return None

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)

            # Derive a default mode for downstream health/prediction responses.
            if "mode" not in data:
                data["mode"] = os.getenv("APP_MODE", "production")

            logging.getLogger("models").info("Loaded metadata from %s", meta_path)
            return data
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.getLogger("models").error("Failed to load metadata.json: %s", exc, exc_info=True)
            return None

    def _load_feature_columns(self) -> Optional[Dict[str, List[str]]]:
        """Extract raw feature column names from metadata.

        Metadata stores feature columns under ``raw_feature_columns`` which may
        be either a mapping with ``numeric`` / ``categorical`` or a flat list.
        """
        if not self.metadata:
            return None

        raw_cols = self.metadata.get("raw_feature_columns")
        if isinstance(raw_cols, dict):
            numeric = list(raw_cols.get("numeric") or [])
            categorical = list(raw_cols.get("categorical") or [])
            return {"numeric": numeric, "categorical": categorical}
        if isinstance(raw_cols, list):
            return {"numeric": list(raw_cols), "categorical": []}

        logging.getLogger("models").warning("raw_feature_columns missing or malformed in metadata")
        return None

    def build_feature_frame(self, game_row: pd.Series) -> pd.DataFrame:
        """Build a one-row feature DataFrame aligned to training columns.

        This takes a raw dataset row (which contains many columns) and
        constructs a DataFrame containing only the columns that the
        pipelines expect, adding any missing columns as NaN so that
        sklearn's imputers can handle them.
        """
        if not self.feature_columns:
            # As a conservative fallback, drop obvious non-feature columns
            # and pass everything else through. This should still be safe
            # because the pipelines' ColumnTransformer will select what it
            # needs.
            logging.getLogger("models").warning("feature_columns not set; using full row as features")
            return pd.DataFrame([game_row.to_dict()])

        numeric = self.feature_columns.get("numeric") or []
        categorical = self.feature_columns.get("categorical") or []
        all_cols: List[str] = list(dict.fromkeys(list(numeric) + list(categorical)))

        data: Dict[str, Any] = {}
        for col in all_cols:
            data[col] = game_row[col] if col in game_row.index else np.nan

        return pd.DataFrame([data], columns=all_cols)
    
    def _validate_model_coherence(self):
        """Validate that models work together coherently"""
        if not all([self.home_pipe, self.away_pipe, self.win_pipe, self.feature_columns]):
            raise ValueError("Missing required model components")
        
        # Test with sample data if possible
        logging.info("Model coherence validation passed")

# ---------------------------------------------------------------
# Application State Management
# ---------------------------------------------------------------
class AppState:
    """Manages global application state with thread safety"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.model_manager = ModelManager(config)
        self.prediction_history = []
        self.history_lock = Lock()
        
    def initialize(self) -> Dict[str, Any]:
        """Initialize application state"""
        results = {}
        
        # Load dataset
        dataset_loaded = self.data_manager.load_dataset()
        results["dataset"] = "loaded" if dataset_loaded else "failed"
        
        # Load models
        model_result = self.model_manager.load_models()
        results["models"] = model_result
        
        # Load history
        history_loaded = self._load_prediction_history()
        results["history"] = f"loaded {len(self.prediction_history)} entries"
        
        return results

    def _load_prediction_history(self) -> bool:
        """Load prediction history from disk if available.

        The history file is optional; failures here should not block startup.
        """
        history_path = self.config.data_dir / "prediction_history.json"
        if not history_path.exists():
            logging.getLogger("history").info("No prediction_history.json found; starting with empty history.")
            self.prediction_history = []
            return False

        try:
            raw = json.loads(history_path.read_text(encoding="utf-8"))
            entries: List[PredictionHistoryEntry] = []
            for item in raw:
                try:
                    entries.append(PredictionHistoryEntry(**item))
                except Exception as exc:  # pragma: no cover - defensive
                    logging.getLogger("history").warning("Skipping invalid history entry: %s", exc)

            self.prediction_history = entries
            logging.getLogger("history").info("Loaded %d history entries", len(entries))
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("history").error("Failed to load prediction history: %s", exc, exc_info=True)
            self.prediction_history = []
            return False

    def record_prediction(
        self,
        request: PredictionRequest,
        response: PredictionResponse,
        game_row: Optional[pd.Series] = None,
    ) -> None:
        """Append a prediction to in-memory history.

        Persistence is intentionally deferred; the primary goal is to make
        recent predictions visible to any future dashboard/history endpoints.
        """
        game_id = f"{request.season}_{request.week}_{request.home_team}_{request.away_team}"
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
            self.prediction_history.append(entry)

# ---------------------------------------------------------------
# FastAPI Application Setup
# ---------------------------------------------------------------
def create_app() -> FastAPI:
    """Create FastAPI application with enhanced configuration"""
    
    # Initialize configuration
    config = Config()
    
    # Setup logging
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "detailed"
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(config.log_dir / "api.log"),
                "level": "DEBUG",
                "formatter": "detailed",
                "encoding": "utf-8"
            }
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]}
    })
    
    log = logging.getLogger("api")
    
    # Create application state
    app_state = AppState(config)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Enhanced lifespan manager"""
        log.info("🚀 Starting NFL Prediction API")
        load_dotenv(dotenv_path=".env")
        try:
            # Initialize application state
            init_results = app_state.initialize()
            log.info(f"Application initialized: {init_results}")
            
            yield
            
        except Exception as e:
            log.error(f"Application startup failed: {e}", exc_info=True)
            raise
        finally:
            log.info("🛑 Application shutdown complete")
    
    # Create FastAPI app
    app = FastAPI(
        title="NFL ML Predictions API",
        version="2.2.0",
        description="Enhanced API with better data cohesion and validation",
        lifespan=lifespan
    )
    load_dotenv(dotenv_path=".env")
    for origin in str(object=CORS_ORIGINS).split(","):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[origin],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Enhanced health endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Enhanced health check with component details"""
        components = {
            "models_loaded": all([app_state.model_manager.home_pipe, 
                                app_state.model_manager.away_pipe, 
                                app_state.model_manager.win_pipe]),
            "dataset_loaded": app_state.data_manager.dataset is not None,
            "metadata_loaded": app_state.model_manager.metadata is not None
        }
        
        status = "healthy" if all(components.values()) else "unhealthy"
        mode = app_state.model_manager.metadata.get("mode", "unknown") if app_state.model_manager.metadata else "unknown"
        
        return HealthResponse(
            status=status,
            mode=mode,
            reason="Component status check",
            timestamp=datetime.now(timezone.utc),
            components=components
        )

    # -----------------------------------------------------------
    # Schedule Endpoints
    # -----------------------------------------------------------
    @app.get("/schedule/next-week")
    async def get_next_week_schedule() -> List[Dict[str, Any]]:
        """Return the upcoming week's schedule as a simple list of games.

        This primarily exists to support the React dashboard. Instead of
        blindly returning the smallest week in the dataset (which skewed toward
        archival 2018 rows), the handler now relies on kickoff timestamps and a
        calendar-aware fallback to surface the true "next" slate.
        """
        df = app_state.data_manager.dataset
        if df is None or df.empty:
            raise HTTPException(503, "Schedule dataset not available")

        try:
            target_season, target_week, selection = _select_schedule_scope(df)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise HTTPException(503, str(exc)) from exc

        numeric_season = pd.to_numeric(df.get("season"), errors="coerce") # type: ignore
        numeric_week = pd.to_numeric(df.get("week"), errors="coerce") # type: ignore
        week_rows = df.loc[(numeric_season == target_season) & (numeric_week == target_week)].copy()
        if week_rows.empty:
            raise HTTPException(404, f"No games found for season {target_season} week {target_week}")

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

        # Normalise into a compact schedule shape for the frontend.
        games: List[Dict[str, Any]] = []
        for _, row in week_rows.iterrows():
            kickoff_iso: Optional[str] = None
            if "home_game_date" in row.index:
                kickoff_dt = pd.to_datetime(row.get("home_game_date"), errors="coerce", utc=True)
                kickoff_iso = kickoff_dt.isoformat() if pd.notna(kickoff_dt) else None

            season_val = row.get("season_num")
            season_val = (
                target_season if pd.isna(season_val) else int(season_val)
            )
            week_val = row.get("week_num")
            week_val = target_week if pd.isna(week_val) else int(week_val)
            home_team = str(row.get("home_team", "")).upper()
            away_team = str(row.get("away_team", "")).upper()

            games.append(
                {
                    "season": season_val,
                    "week": week_val,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_abbr": home_team,
                    "away_abbr": away_team,
                    "game_id": str(
                        row.get(
                            "game_id",
                            f"{season_val}_{week_val}_{home_team}_{away_team}",
                        )
                    ),
                    "kickoff": kickoff_iso,
                    "venue": row.get("stadium_name") or row.get("stadium"),
                    "network": row.get("tv_network") or row.get("network"),
                })

        return games

    # -----------------------------------------------------------
    # Prediction History Endpoint
    # -----------------------------------------------------------
    @app.get("/history")
    async def get_prediction_history(limit: int = 100) -> Dict[str, Any]:
        """Return recent prediction history entries.

        The frontend uses this to hydrate its history chart. Entries are
        served in reverse chronological order (most recent first).
        """
        # Defensive bounds on limit to avoid accidental overload.
        safe_limit = max(1, min(int(limit or 100), 500))

        with app_state.history_lock:
            entries = list(app_state.prediction_history)[-safe_limit:]

        # Ensure newest-first ordering.
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

        return {
            "entries": [json.loads(e.json()) for e in entries],
            "count": len(entries),
        }
    
    # Enhanced prediction endpoint
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """Enhanced prediction endpoint with better validation"""
        # Validate application state
        if not all([app_state.model_manager.home_pipe, 
                   app_state.model_manager.away_pipe, 
                   app_state.model_manager.win_pipe]):
            try:
                home_pipe = joblib.load('backend/models/home_model.joblib')
                away_pipe = joblib.load('backend/models/away_model.joblib')
                win_pipe = joblib.load('backend/models/win_clf_calibrated.joblib')
                pre = joblib.load('backend/models/preprocessing_pipeline.joblib')
                
                if not home_pipe: 
                    raise HTTPException(503, "Models not loaded")
            finally:
                if app_state.data_manager.dataset is None:
                    raise HTTPException(503, "Dataset not available")

        # Locate the corresponding game row in the engineered dataset.
        df = app_state.data_manager.dataset
        assert df is not None  # for type-checkers; guarded above

        try:
            mask = (
                (df["season"] == request.season)
                & (df["week"] == request.week)
                & (df["home_team"] == request.home_team)
                & (df["away_team"] == request.away_team)
            )
        except KeyError as exc:
            # Dataset missing required identifiers; treat as server misconfiguration.
            raise HTTPException(500, f"Dataset missing required column: {exc}") from exc

        game_rows = df.loc[mask]
        if game_rows.empty:
            raise HTTPException(
                404,
                f"Game not found in dataset for {request.season} week {request.week} "
                f"({request.away_team} at {request.home_team})",
            )

        game_row = game_rows.iloc[0]

        # Build feature frame aligned with training metadata.
        try:
            X = app_state.model_manager.build_feature_frame(game_row)
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("api").error("Failed to build feature frame: %s", exc, exc_info=True)
            raise HTTPException(500, "Failed to build features for prediction") from exc

        # Run score regressors.
        try:
            home_pred = float(app_state.model_manager.home_pipe.predict(X)[0])  # type: ignore[union-attr]
            away_pred = float(app_state.model_manager.away_pipe.predict(X)[0])  # type: ignore[union-attr]
        except Exception as exc:
            logging.getLogger("api").error("Score prediction failed: %s", exc, exc_info=True)
            raise HTTPException(500, "Failed to run score models") from exc

        point_diff = home_pred - away_pred

        # Win probability via classifier with a calibrated fallback.
        win_classifier_used = False
        win_probability_source = "unknown"

        # Default threshold from metadata, falling back to 0.5.
        threshold = 0.5
        if app_state.model_manager.metadata:
            holdout = app_state.model_manager.metadata.get("holdout_metrics_win") or {}
            try:
                threshold = float(holdout.get("optimal_threshold", threshold))
            except (TypeError, ValueError):
                threshold = 0.5

        try:
            win_pipe = app_state.model_manager.win_pipe
            if win_pipe is None:
                raise RuntimeError("win_pipe not loaded")

            proba = win_pipe.predict_proba(X)  # type: ignore[union-attr]
            # Assume column 1 corresponds to home-team win probability.
            home_win_prob = float(proba[0, 1])
            home_win_prob = float(np.clip(home_win_prob, 0.0, 1.0))
            win_classifier_used = True
            win_probability_source = "classifier"
        except Exception as exc:
            logging.getLogger("api").warning(
                "Win classifier unavailable, falling back to score-based probability: %s", exc
            )
            # Soft fallback: map point differential to [0, 1] via logistic curve.
            home_win_prob = float(1.0 / (1.0 + np.exp(-0.3 * point_diff)))
            home_win_prob = float(np.clip(home_win_prob, 0.0, 1.0))
            win_classifier_used = False
            win_probability_source = "score_diff_fallback"

        away_win_prob = 1.0 - home_win_prob

        # Binary win indicators based on calibrated threshold.
        home_win_flag = 1.0 if home_win_prob >= threshold else 0.0
        away_win_flag = 1.0 - home_win_flag

        # Confidence: distance from a "coin flip" (0.5) scaled to [0, 1].
        confidence_score = float(abs(home_win_prob - 0.5) * 2.0)

        mode = (
            app_state.model_manager.metadata.get("mode", "production")
            if app_state.model_manager.metadata
            else "production"
        )
        prediction_source = "model" if win_classifier_used else "model+win_fallback"

        response = PredictionResponse(
            home_score=home_pred,
            away_score=away_pred,
            home_win_probability=home_win_prob,
            away_win_probability=away_win_prob,
            home_win=home_win_flag,
            away_win=away_win_flag,
            point_diff=point_diff,
            mode=mode,
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
            logging.getLogger("history").warning("Failed to record prediction history: %s", exc)

        return response
    
    return app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
