#!/usr/bin/env python
"""
NFL Game Prediction API (FastAPI)
=================================

Purpose
-------
Expose HTTP endpoints used by the frontend to check health, fetch the upcoming
schedule, trigger training, refresh data, and get game predictions.

Key Endpoints (kept EXACTLY as-is for frontend compatibility)
-------------------------------------------------------------
GET  /health               → service status (+ model loading mode)
GET  /                     → API metadata
GET  /schedule/next-week   → next week's scheduled games
POST /predict              → predicted scores, win probabilities, point diff
POST /retrain              → retrain models (synchronous)
POST /update_data          → rebuild CSVs then retrain (synchronous)

External Dependencies
---------------------
- fastapi, pydantic
- pandas, numpy
- joblib, lightgbm (loaded inside `load_objects`)
- pytz (schedule date handling)
- subprocess (shelling out to builder/trainer scripts)

Usage Notes
-----------
- Models are loaded once at startup via FastAPI lifespan.
- This module assumes artefacts exist in `backend/models/` with a `metadata.json`.
- Route *shapes and names* are intentionally unchanged to avoid breaking the UI.
- **IMPORTANT** TO RUN:
  uvicorn backend.main:app --reload --port 8000


Compatibility & Constraints
---------------------------
- No fallback prediction path is used at runtime; if models fail to load,
  the app startup fails fast. (The older comment about a "lightweight fallback"
  is marked for review below but not enabled to honor your “no fallback” policy.)

"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# App-level configuration
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths (kept consistent with current layout)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "backend" / "models"

# -----------------------------------------------------------------------------
# Data models (pydantic)
# -----------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Input payload for POST /predict."""
    home_team: str = Field(..., description="Home team (name or abbr)")
    away_team: str = Field(..., description="Away team (name or abbr)")
    season: int = Field(..., description="NFL season year")
    week: int = Field(..., description="Week number within season")


class PredictionResponse(BaseModel):
    """Output payload returned by POST /predict."""
    home_score: float = Field(..., description="Predicted home score")
    away_score: float = Field(..., description="Predicted away score")
    home_win_probability: float = Field(..., description="P(home wins)")
    away_win_probability: float = Field(..., description="P(away wins)")
    point_diff: float = Field(..., description="home_score - away_score")
    mode: str = Field(..., description="Prediction mode identifier")


class HealthResponse(BaseModel):
    """Returned by GET /health."""
    status: str = Field(..., description="healthy|unhealthy")
    mode: Optional[str] = Field(None, description="Model loading mode")
    reason: Optional[str] = Field(None, description="Diagnostic reason")


class ScheduleGame(BaseModel):
    """Schedule row returned by GET /schedule/next-week."""
    season: int
    week: int
    home_team: str
    home_abbr: str
    away_team: str
    away_abbr: str
    kickoff_iso: str
    game_id: str


# -----------------------------------------------------------------------------
# Team canonicalization helpers (kept as-is for UI expectations)
# -----------------------------------------------------------------------------

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
    "Los Angeles Rams": "LA",   # NOTE: consider standardizing to "LAR" (see enhancements)
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

def get_team_abbreviation(team_name: str) -> str:
    """Return an abbreviation for a given full team name; passthrough if unknown."""
    return TEAM_ABBREVIATIONS.get(team_name, team_name)


# -----------------------------------------------------------------------------
# Model artefact loading
# -----------------------------------------------------------------------------

def load_objects() -> Dict[str, Any]:
    """
    Load trained models and preprocessing artefacts from disk.

    Raises
    ------
    RuntimeError
        If heavy dependencies are missing.
    FileNotFoundError
        If required files are absent.

    Returns
    -------
    dict
        {'mode', 'preprocessor', 'nn_model', 'gbm_model', 'raw_feature_columns'}
    """
    # Lazy import: keeps API importable even if heavy libs are not installed.
    try:
        import joblib
        from lightgbm import Booster
    except ImportError as e:
        raise RuntimeError(
            "Heavy dependencies not installed. "
            "Please install requirements before running the API."
        ) from e

    metadata_path = MODELS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    preprocessor = joblib.load(MODELS_DIR / meta["preprocessor"])
    nn_model = joblib.load(MODELS_DIR / meta["models"]["nn_model"])
    gbm_model = Booster(model_file=str(MODELS_DIR / meta["models"]["gbm_model"]))
    return {
        "mode": "models",
        "preprocessor": preprocessor,
        "nn_model": nn_model,
        "gbm_model": gbm_model,
        "raw_feature_columns": meta.get("raw_feature_columns", {}),
    }


# -----------------------------------------------------------------------------
# FastAPI app factory (lifespan ensures models load at startup)
# -----------------------------------------------------------------------------

model_objects: Optional[Dict[str, Any]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models before serving any requests; fail fast if not available."""
    global model_objects
    model_objects = load_objects()
    logger.info("Models loaded successfully.")
    yield
    # No explicit teardown is required.

app = FastAPI(
    title="NFL Game Prediction API",
    description="Predict the probability of a home team winning an NFL game.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: wide-open for now (frontends can call from anywhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Routes (unchanged signatures)
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Return service health and model mode."""
    global model_objects
    if model_objects is None:
        return HealthResponse(status="unhealthy", reason="models not loaded")
    return HealthResponse(status="healthy", mode=model_objects.get("mode"))


@app.get("/")
def root():
    """API discovery endpoint."""
    return {
        "name": "NFL Game Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Predict game outcome with team names and return scores",
            "/predict_raw": "Predict with full feature set (reserved)",
            "/schedule/next-week": "Get next week's NFL game schedule",
            "/train": "Trigger model training process",
            "/retrain": "Retrain models (legacy)",
            "/update_data": "Rebuild datasets and retrain",
        },
        # NOTE: The comment about a "heuristic fallback" in older docs is outdated.
        # See "Suggested Enhancements" for remediation steps.
    }


@app.get("/schedule/next-week", response_model=List[ScheduleGame])
def get_next_week_schedule():
    """
    Return scheduled games for the *next* week relative to 'now' (UTC).

    Implementation notes
    --------------------
    - Reads a prebuilt schedule CSV.
    - Finds the first week with any future game (>= now).
    - If none exist, returns the latest available week.
    """
    try:
        schedule_path = BASE_DIR / "backend" / "data" / "Nfl_schedule_2025_2026.csv"
        if not schedule_path.exists():
            logger.error("Schedule data not found at %s", schedule_path)
            raise HTTPException(status_code=404, detail="Schedule data not found")

        df = pd.read_csv(schedule_path)
        now = datetime.now(pytz.UTC)

        # Determine the "current" (next) week by scanning in order
        current_week = None
        for _, row in df.iterrows():
            gd = row.get("gameday")
            if pd.isna(gd):
                continue
            try:
                game_dt = pd.to_datetime(gd).tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
            except Exception:
                # If tz_localize fails, fallback to naive then set UTC
                game_dt = pd.to_datetime(gd)
                game_dt = game_dt.tz_localize("UTC") if game_dt.tzinfo is None else game_dt
            if game_dt >= now:
                current_week = int(row["week"])
                break

        if current_week is None:
            current_week = int(df["week"].max())

        week_games = df[df["week"] == current_week]
        games: List[ScheduleGame] = []

        for _, row in week_games.iterrows():
            gd = pd.to_datetime(row["gameday"])
            time_str = row.get("gametime")
            if pd.isna(time_str):
                kickoff_iso = gd.isoformat()
            else:
                # best-effort parse "HH:MM" or ISO-like strings
                kickoff_iso = pd.to_datetime(f"{row['gameday']} {time_str}").isoformat()

            games.append(
                ScheduleGame(
                    season=int(row["season"]),
                    week=int(row["week"]),
                    home_team=str(row["home_team"]),
                    home_abbr=get_team_abbreviation(str(row["home_team"])),
                    away_team=str(row["away_team"]),
                    away_abbr=get_team_abbreviation(str(row["away_team"])),
                    kickoff_iso=kickoff_iso,
                    game_id=str(row["game_id"]),
                )
            )

        logger.info("Returning %d games for week %s", len(games), current_week)
        return games

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error loading schedule: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load schedule: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict_game(payload: PredictionRequest):
    """
    Predict outcome for (home_team vs away_team, season/week).

    Notes
    -----
    - Uses both NN and GBM probabilities; returns averaged ensemble.
    - Feature construction uses placeholder priors; replace with dataset
      lookups for production-grade accuracy.
    """
    global model_objects
    if model_objects is None:
        logger.error("Models not loaded - cannot make predictions")
        raise HTTPException(status_code=500, detail="Models not loaded.")

    # Validate model load state
    if model_objects.get("mode") != "models":
        msg = f"Models not properly loaded (mode={model_objects.get('mode')})"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)

    try:
        home_team = payload.home_team
        away_team = payload.away_team
        season = payload.season
        week = payload.week
        logger.info(
            "Predicting %s vs %s (season=%s, week=%s)",
            home_team, away_team, season, week
        )

        # Minimal feature frame — structure must match the preprocessor expectations.
        # NOTE: These priors are placeholders; connect to your built dataset to
        # fetch true rolling priors for the given teams/season/week.
        input_df = pd.DataFrame(
            {
                "season": [season],
                "week": [week],
                "home_team": [home_team],
                "away_team": [away_team],
                "home_prior_pa_avg_3": [22.5],
                "home_prior_pa_avg_5": [22.5],
                "home_prior_pf_avg_3": [23.0],
                "home_prior_pf_avg_5": [23.0],
                "home_prior_win_pct_3": [0.5],
                "home_prior_win_pct_5": [0.5],
                "away_prior_pa_avg_3": [22.5],
                "away_prior_pa_avg_5": [22.5],
                "away_prior_pf_avg_3": [23.0],
                "away_prior_pf_avg_5": [23.0],
                "away_prior_win_pct_3": [0.5],
                "away_prior_win_pct_5": [0.5],
            }
        )

        X = model_objects["preprocessor"].transform(input_df)
        nn_prob = float(model_objects["nn_model"].predict_proba(X)[:, 1][0])
        gbm_prob = float(model_objects["gbm_model"].predict_proba(X)[:, 1][0])

        # Ensemble average; simple score shaping around ~23-21 baseline
        home_win_prob = (nn_prob + gbm_prob) / 2.0
        adj = (home_win_prob - 0.5) * 10.0
        home_score = round(max(0.0, min(60.0, 23.0 + adj)), 1)
        away_score = round(max(0.0, min(60.0, 20.5 - adj)), 1)
        point_diff = round(home_score - away_score, 1)

        return PredictionResponse(
            home_score=home_score,
            away_score=away_score,
            home_win_probability=round(home_win_prob, 3),
            away_win_probability=round(1 - home_win_prob, 3),
            point_diff=point_diff,
            mode="models",
        )

    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.post("/retrain")
def retrain(new_data_path: Optional[str] = None):
    """
    Retrain models, then hot-reload artefacts in-process.

    Implementation detail
    ---------------------
    - Synchronous subprocess call; consider background tasks or a job queue
      if training time increases.
    """
    global model_objects
    train_script_path = Path(__file__).resolve().parent / "train_models.py"
    try:
        subprocess.run(["python", str(train_script_path)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e.stderr}")
    model_objects = load_objects()
    return {"detail": "Models retrained successfully."}


@app.post("/update_data")
def update_data():
    """
    Rebuild CSVs then retrain models (one-click refresh).
    """
    import sys
    try:
        # 1) Rebuild CSVs (paths kept as-is to avoid changing external scripts)
        build = subprocess.run(
            [sys.executable, "scripts/build_csvs.py", "--start", "2014", "--end", "2024", "--out-dir", "data"],
            check=True, capture_output=True, text=True,
        )
        logger.info("build_csvs stdout:\n%s", build.stdout)
        logger.info("build_csvs stderr:\n%s", build.stderr)

        # 2) Retrain
        train = subprocess.run([sys.executable, "train_models.py"], check=True, capture_output=True, text=True)
        logger.info("train_models stdout:\n%s", train.stdout)
        logger.info("train_models stderr:\n%s", train.stderr)

        return {"detail": "Data updated and models retrained."}
    except subprocess.CalledProcessError as e:
        logger.error("Update failed: %s", e.stderr)
        return {"detail": "Update failed", "stderr": e.stderr}


# -----------------------------------------------------------------------------
# Suggested Enhancements (for review)
# -----------------------------------------------------------------------------
# 1) Standardize team codes (e.g., use "LAR" vs "LA") to match dataset canon and
#    reduce mapping edge cases across builder → trainer → API.
# 2) Replace placeholder priors in /predict by joining the built game dataset
#    to fetch true rolling features for (season, week, home_team, away_team).
# 3) Convert /retrain and /update_data to background tasks or an external worker
#    to avoid blocking API threads; add auth on these endpoints for safety.

# 4) Implement logging to file with rotation instead of stdout; consider
#    structured logging (JSON) for easier parsing in log management systems.