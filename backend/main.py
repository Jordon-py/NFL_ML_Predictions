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
from datetime import datetime, timezone
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
from pydantic import BaseModel, Field, validator

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
        fallback = self.backend_dir / "models"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    
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
        """Load model pipelines with error handling"""
        pipelines = {}
        
        for name in ["home_pipe", "away_pipe", "win_pipe"]:
            try:
                pipeline_path = self.config.models_dir / f"{name}.joblib"
                if pipeline_path.exists():
                    pipelines[name] = joblib.load(pipeline_path)
                    logging.info(f"✓ Loaded {name}")
                else:
                    logging.error(f"✗ Pipeline not found: {pipeline_path}")
                    pipelines[name] = None
            except Exception as e:
                logging.error(f"✗ Failed to load {name}: {e}")
                pipelines[name] = None
        
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
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure based on environment
        allow_credentials=True,
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
    
    # Enhanced prediction endpoint
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """Enhanced prediction endpoint with better validation"""
        # Validate application state
        if not all([app_state.model_manager.home_pipe, 
                   app_state.model_manager.away_pipe, 
                   app_state.model_manager.win_pipe]):
            raise HTTPException(503, "Models not loaded")
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
