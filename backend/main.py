#!/usr/bin/env python
"""
NFL Game Prediction API (FastAPI)
=================================
See docstring in original for purpose and endpoints. Route shapes unchanged.
Run: uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import logging.config
import os
import subprocess
import sys
import inspect
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pandas import Timestamp, NaT

# -----------------------------------------------------------------------------
# Paths (compute first so logging can use them)
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
BASE_DIR = BACKEND_DIR.parent
MODELS_DIR = BASE_DIR / "backend" / "models"
DATA_DIR = BASE_DIR / "backend" / "data"
LOG_DIR = BASE_DIR / "backend" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging (file exists by now)
# -----------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"},
        "simple": {"format": "%(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "simple", "stream": "ext://sys.stdout"},
        "file": {"class": "logging.FileHandler", "level": "DEBUG", "formatter": "detailed", "filename": str(LOG_DIR / "nfl_prediction.log"), "mode": "a"},
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    "loggers": {"nfl_prediction": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": False}},
})
logger = logging.getLogger("nfl_prediction")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_DATASET = DATA_DIR / "Nfl_data_sorted.csv"
DEFAULT_SCHEDULE = DATA_DIR / "Nfl_schedule_2025_2026.csv"

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
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
    status: str
    mode: Optional[str] = None
    reason: Optional[str] = None

class ScheduleGame(BaseModel):
    season: int
    week: int
    home_team: str
    home_abbr: str
    away_team: str
    away_abbr: str
    kickoff_iso: str
    game_id: str

# -----------------------------------------------------------------------------
# Team canon
# -----------------------------------------------------------------------------
TEAM_ABBREVIATIONS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
VALID_ABBRS = set(TEAM_ABBREVIATIONS.values())
VALID_TEAM_NAMES = set(TEAM_ABBREVIATIONS.keys())

def get_team_abbreviation(team_name: str) -> str:
    if team_name in VALID_ABBRS:
        return team_name
    if team_name in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[team_name]
    logger.error("Unknown team: %s", team_name)
    raise ValueError(f"Unknown team name: {team_name}")

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_objects() -> Dict[str, Any]:
    try:
        import joblib
    except ImportError as e:
        logger.error("joblib not available: %s", e)
        raise RuntimeError("Install requirements (joblib) before running the API.") from e

    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        logger.error("Model metadata missing: %s", meta_path)
        raise FileNotFoundError(f"Missing {meta_path}")
    
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        logger.info("Loaded model metadata from %s", meta_path)
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to read metadata: %s", e)
        raise RuntimeError(f"Invalid metadata file {meta_path}: {e}") from e

    preprocessor = joblib.load(MODELS_DIR / meta["preprocessor"])
    models_meta = meta.get("models", {})
    model_types = meta.get("model_types", {"home_model_type": "lgbm", "away_model_type": "lgbm"})

    # Check if TensorFlow is needed and available
    needs_tensorflow = (
        model_types.get("home_model_type") == "neural_network" or 
        model_types.get("away_model_type") == "neural_network"
    )
    
    tensorflow_available = False
    if needs_tensorflow:
        try:
            import tensorflow as tf
            tensorflow_available = True
            logger.info("TensorFlow loaded successfully, version: %s", tf.__version__)
        except ImportError as e:
            logger.error("TensorFlow required but not available: %s", e)
            raise RuntimeError(
                "TensorFlow is required for neural network models but is not installed. "
                "Install with: pip install tensorflow"
            ) from e

    # Home model
    home_model_path = MODELS_DIR / models_meta.get("home_model", "home_model.joblib")
    if not home_model_path.exists():
        logger.error("Home model file missing: %s", home_model_path)
        raise FileNotFoundError(f"Missing {home_model_path}")
    
    if model_types.get("home_model_type") == "neural_network":
        if not tensorflow_available:
            raise RuntimeError("TensorFlow required for home neural network model but not available")
        import tensorflow as tf
        try:
            home_model = tf.keras.models.load_model(home_model_path) # type: ignore
            logger.info("Loaded TensorFlow home model from %s", home_model_path)
        except Exception as e:
            logger.error("Failed to load TensorFlow home model: %s", e)
            raise RuntimeError(f"Failed to load neural network home model: {e}") from e
    else:
        try:
            home_model = joblib.load(home_model_path)
            logger.info("Loaded joblib home model from %s", home_model_path)
        except Exception as e:
            logger.error("Failed to load joblib home model: %s", e)
            raise RuntimeError(f"Failed to load home model: {e}") from e

    # Away model
    away_model_path = MODELS_DIR / models_meta.get("away_model", "away_model.joblib")
    if not away_model_path.exists():
        logger.error("Away model file missing: %s", away_model_path)
        raise FileNotFoundError(f"Missing {away_model_path}")
    
    if model_types.get("away_model_type") == "neural_network":
        if not tensorflow_available:
            raise RuntimeError("TensorFlow required for away neural network model but not available")
        import tensorflow as tf
        try:
            away_model = tf.keras.models.load_model(away_model_path) # type: ignore
            logger.info("Loaded TensorFlow away model from %s", away_model_path)
        except Exception as e:
            logger.error("Failed to load TensorFlow away model: %s", e)
            raise RuntimeError(f"Failed to load neural network away model: {e}") from e
    else:
        try:
            away_model = joblib.load(away_model_path)
            logger.info("Loaded joblib away model from %s", away_model_path)
        except Exception as e:
            logger.error("Failed to load joblib away model: %s", e)
            raise RuntimeError(f"Failed to load away model: {e}") from e

    logger.info("Successfully loaded all models - home: %s, away: %s", 
                model_types.get("home_model_type", "lgbm"), 
                model_types.get("away_model_type", "lgbm"))

    return {
        "mode": "models",
        "preprocessor": preprocessor,
        "home_model": home_model,
        "away_model": away_model,
        "model_types": model_types,
        "raw_feature_columns": meta.get("raw_feature_columns", {}),
    }

# -----------------------------------------------------------------------------
# App + lifespan
# -----------------------------------------------------------------------------
model_objects: Optional[Dict[str, Any]] = None
dataset_df: Optional[pd.DataFrame] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_objects, dataset_df
    logger.info("Startup: loading models and dataset")
    model_objects = load_objects()
    ds_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    if not ds_path.exists():
        logger.error("Dataset missing at %s", ds_path)
        raise RuntimeError(f"Dataset not found: {ds_path}")
    df = pd.read_csv(ds_path)
    if df.empty:
        raise RuntimeError("Dataset CSV is empty")
    df.columns = [c.strip() for c in df.columns]
    dataset_df = df
    logger.info("Loaded dataset rows=%d", len(df))
    yield
    logger.info("Shutdown complete")

app = FastAPI(title="NFL Game Prediction API", description="Predict home/away scores and win odds.", version="1.0.0", lifespan=lifespan)

# CORS configuration - read from environment for flexibility
allowed_origins = os.getenv("CORS_ORIGINS", "https://nfl-predict-ecf5a5bd34fe.herokuapp.com/").split(",")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_current_nfl_context() -> Dict[str, Any]:
    """Robust last-completed → next-week detection using season/week ordering."""
    now = datetime.utcnow()
    current_season = now.year if now.month >= 8 else now.year - 1

    global dataset_df
    if dataset_df is not None and not dataset_df.empty and {"season", "week", "home_points_for", "away_points_for"} <= set(dataset_df.columns):
        completed = dataset_df[dataset_df["home_points_for"].notna() & dataset_df["away_points_for"].notna()]
        if not completed.empty:
            ordered = completed.sort_values(["season", "week"], kind="mergesort")
            last = ordered.iloc[-1]
            last_completed_season = int(last["season"])
            last_completed_week = int(last["week"])
            next_season = last_completed_season
            next_week = last_completed_week + 1
            if next_week > 18:
                next_week = 1
                next_season += 1
            return {
                "current_season": current_season,
                "last_completed_season": last_completed_season,
                "last_completed_week": last_completed_week,
                "next_prediction_season": next_season,
                "next_prediction_week": next_week,
                "status": "nfl_season_active" if next_season == current_season else "offseason",
            }

    return {
        "current_season": current_season,
        "last_completed_season": current_season,
        "last_completed_week": 0,
        "next_prediction_season": current_season,
        "next_prediction_week": 1,
        "status": "preseason_or_early",
    }

def build_future_game_features(df: pd.DataFrame, home_team: str, away_team: str, season: int, week: int) -> pd.Series:
    """Use last known rolling stats strictly before target (season, week)."""
    def latest_team_rollups(team: str) -> Dict[str, float]:
        d = df.copy()
        d["time_key"] = d["season"].astype(int) * 100 + d["week"].astype(int)
        target_key = int(season) * 100 + int(week)
        team_rows = d[(d["home_team"] == team) | (d["away_team"] == team)]
        before = team_rows[team_rows["time_key"] < target_key]
        if before.empty:
            raise ValueError(f"No historical rows for team '{team}' before {season}W{week}")
        last = before.sort_values("time_key").iloc[-1]
        if last["home_team"] == team:
            return {
                "prior_pa_avg_3": float(last.get("home_prior_pa_avg_3", 0.0)),
                "prior_pa_avg_5": float(last.get("home_prior_pa_avg_5", 0.0)),
                "prior_pf_avg_3": float(last.get("home_prior_pf_avg_3", 0.0)),
                "prior_pf_avg_5": float(last.get("home_prior_pf_avg_5", 0.0)),
                "prior_win_pct_3": float(last.get("home_prior_win_pct_3", 0.5)),
                "prior_win_pct_5": float(last.get("home_prior_win_pct_5", 0.5)),
            }
        return {
            "prior_pa_avg_3": float(last.get("away_prior_pa_avg_3", 0.0)),
            "prior_pa_avg_5": float(last.get("away_prior_pa_avg_5", 0.0)),
            "prior_pf_avg_3": float(last.get("away_prior_pf_avg_3", 0.0)),
            "prior_pf_avg_5": float(last.get("away_prior_pf_avg_5", 0.0)),
            "prior_win_pct_3": float(last.get("away_prior_win_pct_3", 0.5)),
            "prior_win_pct_5": float(last.get("away_prior_win_pct_5", 0.5)),
        }

    home_roll = latest_team_rollups(home_team)
    away_roll = latest_team_rollups(away_team)
    return pd.Series({
        "home_prior_pa_avg_3": home_roll["prior_pa_avg_3"],
        "home_prior_pa_avg_5": home_roll["prior_pa_avg_5"],
        "home_prior_pf_avg_3": home_roll["prior_pf_avg_3"],
        "home_prior_pf_avg_5": home_roll["prior_pf_avg_5"],
        "home_prior_win_pct_3": home_roll["prior_win_pct_3"],
        "home_prior_win_pct_5": home_roll["prior_win_pct_5"],
        "away_prior_pa_avg_3": away_roll["prior_pa_avg_3"],
        "away_prior_pa_avg_5": away_roll["prior_pa_avg_5"],
        "away_prior_pf_avg_3": away_roll["prior_pf_avg_3"],
        "away_prior_pf_avg_5": away_roll["prior_pf_avg_5"],
        "away_prior_win_pct_3": away_roll["prior_win_pct_3"],
        "away_prior_win_pct_5": away_roll["prior_win_pct_5"],
    })

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    if model_objects is None:
        return HealthResponse(status="unhealthy", reason="models not loaded", mode=None)
    return HealthResponse(status="healthy", reason="all systems operational", mode=model_objects.get("mode"))

@app.get("/")
def root():
    context = get_current_nfl_context()
    return {
        "name": "NFL Game Prediction API",
        "version": "1.0.0",
        "nfl_context": context,
        "endpoints": {
            "/health": "Health check",
            "/debug": "System debug info",
            "/predict": "Predict one game",
            "/predict/next-week": "Predict next week's slate",
            "/schedule/next-week": "Next week's schedule",
            "/retrain": "Retrain models",
            "/update_data": "Rebuild datasets + retrain",
        },
    }

@app.get("/debug")
def debug_info():
    global model_objects, dataset_df
    debug_data: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models_loaded": model_objects is not None,
        "dataset_loaded": dataset_df is not None,
    }
    if model_objects:
        cols = model_objects.get("raw_feature_columns", {})
        debug_data.update({
            "model_mode": model_objects.get("mode"),
            "model_types": model_objects.get("model_types"),
            "feature_columns": list(cols.keys()),
            "numeric_features": len(cols.get("numeric", [])),
            "categorical_features": len(cols.get("categorical", [])),
        })
    if dataset_df is not None:
        debug_data.update({
            "dataset_rows": int(len(dataset_df)),
            "dataset_columns": int(len(dataset_df.columns)),
            "season_range": [int(dataset_df["season"].min()), int(dataset_df["season"].max())] if "season" in dataset_df else None,
            "week_range": [int(dataset_df["week"].min()), int(dataset_df["week"].max())] if "week" in dataset_df else None,
        })
    try:
        mpath = MODELS_DIR / "metadata.json"
        if mpath.exists():
            with open(mpath, "r") as f:
                meta = json.load(f)
            debug_data["training_metadata"] = {
                "training_timestamp": meta.get("training_timestamp"),
                "dataset_hash": meta.get("dataset_hash"),
                "training_samples": meta.get("training_samples"),
                "model_scores": meta.get("model_scores"),
            }
    except Exception as e:
        debug_data["training_metadata_error"] = str(e)
    return debug_data

@app.get("/schedule/next-week", response_model=List[ScheduleGame])
def get_next_week_schedule():
    try:
        schedule_path = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
        if not schedule_path.exists():
            raise HTTPException(status_code=404, detail=f"Schedule not found: {schedule_path}")
        df = pd.read_csv(schedule_path)
        if df.empty:
            raise HTTPException(status_code=404, detail="Schedule is empty")

        def to_kickoff_utc(row) -> pd.Timestamp:
            ts = f"{row.get('gameday','')} {row.get('gametime')}" if pd.notna(row.get("gametime")) else str(row.get("gameday",""))
            dt = pd.to_datetime(ts, errors="coerce", utc=True)
            return dt # type: ignore

        df["kickoff_ts_utc"] = df.apply(to_kickoff_utc, axis=1)
        now = pd.Timestamp.now(tz="UTC")
        future = df[df["kickoff_ts_utc"].notna() & (df["kickoff_ts_utc"] >= now)]
        current_week = int(future["week"].min()) if not future.empty else int(df["week"].max())

        week_games = df[df["week"] == current_week]
        games: List[ScheduleGame] = []
        for _, row in week_games.iterrows():
            games.append(ScheduleGame(
                season=int(row["season"]),
                week=int(row["week"]),
                home_team=str(row["home_team"]),
                home_abbr=get_team_abbreviation(str(row["home_team"])),
                away_team=str(row["away_team"]),
                away_abbr=get_team_abbreviation(str(row["away_team"])),
                kickoff_iso=(row["kickoff_ts_utc"].isoformat() if pd.notna(row["kickoff_ts_utc"]) else "TBD"),
                game_id=str(row.get("game_id", f"{row['season']}W{row['week']}-{row['away_team']}@{row['home_team']}")),
            ))
        logger.info("Schedule week %s games=%d", current_week, len(games))
        return games
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Schedule error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load schedule: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict_game(payload: PredictionRequest):
    global model_objects, dataset_df
    if model_objects is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
    if dataset_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    try:
        home_abbr = get_team_abbreviation(payload.home_team)
        away_abbr = get_team_abbreviation(payload.away_team)
        season = int(payload.season)
        week = int(payload.week)

        mask = (
            (dataset_df["season"] == season)
            & (dataset_df["week"] == week)
            & (dataset_df["home_team"] == home_abbr)
            & (dataset_df["away_team"] == away_abbr)
        )
        rows = dataset_df.loc[mask]

        if rows.empty:
            row = build_future_game_features(dataset_df, home_abbr, away_abbr, season, week)
        else:
            if len(rows) > 1:
                logger.warning("Multiple dataset matches; using first")
            row = rows.iloc[0]
            if pd.notna(row.get("home_points_for")) and pd.notna(row.get("away_points_for")):
                raise HTTPException(status_code=400, detail="Game already completed; no prediction produced.")

        feature_cols = [
            'home_prior_pf_avg_3','home_prior_pa_avg_3','home_prior_win_pct_3',
            'home_prior_pf_avg_5','home_prior_pa_avg_5','home_prior_win_pct_5',
            'away_prior_pf_avg_3','away_prior_pa_avg_3','away_prior_win_pct_3',
            'away_prior_pf_avg_5','away_prior_pa_avg_5','away_prior_win_pct_5','home_minus_away_pf_avg_3','home_minus_away_pa_avg_3','home_minus_away_win_pct_3','home_minus_away_pf_avg_5','home_minus_away_pa_avg_5','home_minus_away_win_pct_5',
            'home_minus_away_pf_avg_3','home_minus_away_pa_avg_3','home_minus_away_win_pct_3',
            'home_minus_away_pf_avg_5','home_minus_away_pa_avg_5','home_minus_away_win_pct_5',
        ]
        missing = [c for c in feature_cols if c not in row.index]
        if missing:
            raise HTTPException(status_code=500, detail=f"Missing feature columns: {missing}")
        input_df = pd.DataFrame({c: [row[c]] for c in feature_cols})

        X = model_objects["preprocessor"].transform(input_df)
        types = model_objects.get("model_types", {})

        if types.get("home_model_type") == "neural_network":
            home_pred = float(model_objects["home_model"].predict(X, verbose=0)[0][0])
        else:
            home_pred = float(model_objects["home_model"].predict(X)[0])

        if types.get("away_model_type") == "neural_network":
            away_pred = float(model_objects["away_model"].predict(X, verbose=0)[0][0])
        else:
            away_pred = float(model_objects["away_model"].predict(X)[0])

        home_score = round(max(0.0, min(70.0, home_pred)), 1)
        away_score = round(max(0.0, min(70.0, away_pred)), 1)
        point_diff = round(home_score - away_score, 1)

        k = 0.22
        home_win_prob = 1.0 / (1.0 + np.exp(-k * point_diff))
        away_win_prob = 1.0 / (1.0 + np.exp(k * point_diff))

        return PredictionResponse(
            home_score=home_score,
            away_score=away_score,
            home_win_probability=round(float(home_win_prob), 3),
            away_win_probability=round(float(1 - home_win_prob), 3),
            point_diff=point_diff,
            mode="models",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.get("/predict/next-week")
def predict_next_week():
    global model_objects
    if model_objects is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
    try:
        context = get_current_nfl_context()
        next_season = context["next_prediction_season"]
        next_week = context["next_prediction_week"]

        schedule_path = Path(os.getenv("SCHEDULE_PATH", str(DEFAULT_SCHEDULE)))
        if not schedule_path.exists():
            raise HTTPException(status_code=404, detail="Schedule data not found")
        schedule_df = pd.read_csv(schedule_path)
        games_df = schedule_df[(schedule_df["season"] == next_season) & (schedule_df["week"] == next_week)]

        preds: List[Dict[str, Any]] = []
        for _, g in games_df.iterrows():
            try:
                req = PredictionRequest(
                    home_team=str(g["home_team"]),
                    away_team=str(g["away_team"]),
                    season=int(g["season"]),
                    week=int(g["week"]),
                )
                pr = predict_game(req)
                preds.append({
                    "game_id": str(g.get("game_id", f"{g['season']}W{g['week']}-{g['away_team']}@{g['home_team']}")),
                    "season": int(g["season"]),
                    "week": int(g["week"]),
                    "home_team": str(g["home_team"]),
                    "away_team": str(g["away_team"]),
                    "kickoff": str(g.get("gameday", "TBD")),
                    "prediction": pr.model_dump_json(),
                })
            except Exception as e:
                preds.append({
                    "game_id": str(g.get("game_id", "unknown")),
                    "season": int(g["season"]),
                    "week": int(g["week"]),
                    "home_team": str(g["home_team"]),
                    "away_team": str(g["away_team"]),
                    "error": str(e),
                })
        return {"context": context, "games": preds, "total_games": len(preds), "successful_predictions": len([p for p in preds if "prediction" in p])}
    except Exception as e:
        logger.error("Next-week prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to predict next week: {e}")

@app.post("/retrain")
def retrain(new_data_path: Optional[str] = None):
    global model_objects
    try:
        subprocess.run([sys.executable, str(BACKEND_DIR / "train_models.py")], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e.stderr}")
    model_objects = load_objects()
    return {"detail": "Models retrained successfully."}

@app.post("/update_data")
def update_data():
    try:
        build = subprocess.run(
            [sys.executable, str(BASE_DIR / "scripts" / "build_csvs.py"), "--start", "2014", "--end", "2024", "--out-dir", str(DATA_DIR)],
            check=True, capture_output=True, text=True
        )
        logger.info("build_csvs stdout:\n%s", build.stdout)
        train = subprocess.run([sys.executable, str(BACKEND_DIR / "train_models.py")], check=True, capture_output=True, text=True)
        logger.info("train_models stdout:\n%s", train.stdout)
        return {"detail": "Data updated and models retrained."}
    except subprocess.CalledProcessError as e:
        logger.error("Update failed: %s", e.stderr)
        return {"detail": "Update failed", "stderr": e.stderr}

def your_function() -> Timestamp:
    """Process timestamp data with strict validation.
    
    Returns:
        Timestamp: Valid pandas timestamp
        
    Raises:
        ValueError: If timestamp is NaT or invalid
    """
    # Create current timestamp for NFL context processing
    dt = pd.Timestamp.now(tz='UTC')
    
    if pd.isna(dt) or dt is NaT:
        # Safe line number detection with fallback
        frame = inspect.currentframe()
        line_no = frame.f_lineno if frame is not None else "unknown"
        
        logger.error(
            "Timestamp validation failed: NaT detected in %s at line %s", 
            __name__, 
            line_no
        )
        raise ValueError(f"Invalid timestamp: Cannot process NaT value in {__name__}")
    
    # Type assertion: we've validated dt is not NaT, so it's definitely Timestamp
    logger.debug("Timestamp validation passed: %s", dt)
    return dt
    return validated_dt
