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
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse


# -----------------------------------------------------------------------------
# App-level configuration
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths (kept consistent with current layout)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "backend" / "models"
DATASET_PATH = BASE_DIR / "Nfl_data_sorted.csv"

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
    "Los Angeles Rams": "LA",  # NOTE: consider standardizing to "LAR" (see enhancements)
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
    """
    Return an abbreviation for a team name. Handles both full names and abbreviations.
    Maintains fail-fast behavior for truly unknown teams.
    """
    # If input is already a valid abbreviation, return as-is
    valid_abbreviations = set(TEAM_ABBREVIATIONS.values())
    if team_name in valid_abbreviations:
        return team_name

    # If input is a full team name, convert to abbreviation
    if team_name in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[team_name]

    # If neither full name nor valid abbreviation, fail fast
    logger.error(
        "Unknown team name: %s. Available full names: %s, Valid abbreviations: %s",
        team_name,
        list(TEAM_ABBREVIATIONS.keys()),
        sorted(valid_abbreviations),
    )
    raise ValueError(f"Unknown team name: {team_name}")


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
        # Import tensorflow early to catch any issues
        import tensorflow.keras as keras
        logger.info("TensorFlow/Keras available for neural network models")
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
    models_meta = meta.get("models", {})
    model_types = meta.get("model_types", {"home_model_type": "lgbm", "away_model_type": "lgbm"})

    # Load home model based on its type
    home_model_file = models_meta.get("home_model", "home_model.joblib")
    home_model_path = MODELS_DIR / home_model_file
    if not home_model_path.exists():
        raise FileNotFoundError(f"Home model not found at {home_model_path}")

    if model_types.get("home_model_type") == "neural_network":
        try:
            home_model = keras.models.load_model(home_model_path)
            logger.info("Loaded home neural network model from %s", home_model_path)
        except Exception as e:
            logger.error("Failed to load home neural network model: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to load home neural network model: {e}") from e
    else:
        home_model = joblib.load(home_model_path)
        logger.info("Loaded home LightGBM model from %s", home_model_path)

    # Load away model based on its type
    away_model_file = models_meta.get("away_model", "away_model.joblib")
    away_model_path = MODELS_DIR / away_model_file
    if not away_model_path.exists():
        raise FileNotFoundError(f"Away model not found at {away_model_path}")

    if model_types.get("away_model_type") == "neural_network":
        try:
            away_model = keras.models.load_model(away_model_path)
            logger.info("Loaded away neural network model from %s", away_model_path)
        except Exception as e:
            logger.error("Failed to load away neural network model: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to load away neural network model: {e}") from e
    else:
        away_model = joblib.load(away_model_path)
        logger.info("Loaded away LightGBM model from %s", away_model_path)

    return {
        "mode": "models",
        "preprocessor": preprocessor,
        "home_model": home_model,
        "away_model": away_model,
        "model_types": model_types,
        "raw_feature_columns": meta.get("raw_feature_columns", {}),
    }


# -----------------------------------------------------------------------------
# FastAPI app factory (lifespan ensures models load at startup)
# -----------------------------------------------------------------------------

model_objects: Optional[Dict[str, Any]] = None
dataset_df: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models before serving any requests; fail fast if not available."""
    global model_objects, dataset_df
    model_objects = load_objects()
    logger.info("Models loaded successfully.")
    # Load rolling priors dataset used for feature construction
    if not DATASET_PATH.exists():
        logger.error("Dataset not found at %s", DATASET_PATH)
        raise RuntimeError(f"Dataset not found: {DATASET_PATH}")
    try:
        df = pd.read_csv(DATASET_PATH)
        df.columns = [c.strip() for c in df.columns]
        dataset_df = df
        logger.info(
            "Loaded dataset for inference: %s (rows=%d)", DATASET_PATH.name, len(dataset_df)
        )
    except Exception as e:
        logger.error("Failed to load dataset: %s", e, exc_info=True)
        raise
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
    allow_origins=["https://nfl-predict-ecf5a5bd34fe.herokuapp.com"],  # tighten in production
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
        return HealthResponse(status="unhealthy", mode="none", reason="models not loaded")
    return HealthResponse(status="healthy", mode=model_objects.get("mode"), reason="models loaded successfully")


def get_current_nfl_context() -> Dict[str, Any]:
    """
    Determine current NFL season state and next prediction target.
    Returns context about current week and what should be predicted next.
    """
    from datetime import datetime

    current_date = datetime.now()
    current_season = current_date.year

    # NFL season spans Sept-Feb, adjust if in early months
    if current_date.month <= 7:
        current_season -= 1

    # Try to determine completed week from existing data
    global dataset_df
    if dataset_df is not None and not dataset_df.empty:
        # Find the most recent completed game
        completed_games = dataset_df[
            dataset_df["home_points_for"].notna() & dataset_df["away_points_for"].notna()
        ]
        if not completed_games.empty:
            latest_game = completed_games.loc[completed_games.index[-1]]
            last_completed_season = int(latest_game["season"])
            last_completed_week = int(latest_game["week"])

            # Determine next prediction target
            next_week = last_completed_week + 1
            next_season = last_completed_season

            # Handle season rollover (Week 18 -> Week 1 of next season)
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

    # Default fallback - Week 1 of current season
    return {
        "current_season": current_season,
        "last_completed_season": current_season,
        "last_completed_week": 0,
        "next_prediction_season": current_season,
        "next_prediction_week": 1,
        "status": "preseason_or_early",
    }


@app.get("/")
def root():
    """API discovery endpoint with NFL season context."""
    context = get_current_nfl_context()

    return {
        "name": "NFL Game Prediction API",
        "version": "1.0.0",
        "nfl_context": context,
        "endpoints": {
            "/health": "Health check",
            "/predict": "Predict specific game outcome with team names and return scores",
            "/predict/next-week": "Predict all games for the next NFL week",
            "/predict_raw": "Predict with full feature set (reserved)",
            "/schedule/next-week": "Get next week's NFL game schedule",
            "/train": "Trigger model training process",
            "/retrain": "Retrain models (legacy)",
            "/update_data": "Rebuild datasets and retrain",
        },
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
            except Exception as e:
                logger.error("Failed to parse game date '%s': %s", gd, e)
                raise ValueError(f"Invalid game date format: {gd}") from e
            if game_dt >= now:
                current_week = int(row["week"])
                break

        if current_week is None:
            logger.error("No future games found in schedule data")
            raise HTTPException(status_code=404, detail="No future games found in schedule")

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

# serve the built Vite app (mounted last so API routes take precedence)
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="app")



# -----------------------------------------------------------------------------
# Prediction logic
# -----------------------------------------------------------------------------
def build_future_game_features(
    df: pd.DataFrame, home_team: str, away_team: str, season: int, week: int
) -> pd.Series:
    """
    Build rolling features for a future game by finding the most recent
    available data for each team.

    Args:
        df: The historical dataset
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        season: Game season
        week: Game week

    Returns:
        Series with rolling features for the matchup
    """

    def get_latest_team_features(
        team: str, target_season: int, target_week: int
    ) -> Dict[str, Any]:
        """Get the most recent rolling features for a team before the target game."""
        # Find all games for this team before the target date
        team_mask = (df["home_team"] == team) | (df["away_team"] == team)
        target_time_key = target_season * 100 + target_week

        # Get games before target week
        df["time_key"] = df["season"] * 100 + df["week"]
        before_target = df[team_mask & (df["time_key"] < target_time_key)]

        if before_target.empty:
            # No prior data - fail fast as per instructions
            logger.error(
                "No prior data available for team %s before season %d, week %d",
                team, target_season, target_week
            )
            raise ValueError(
                f"No prior data available for {team} in season {target_season}, week {target_week}. "
                f"Cannot build features for prediction."
            )

        # Get the most recent game features
        latest_idx = before_target["time_key"].idxmax()
        latest_game = before_target.loc[latest_idx]

        # Extract team's features based on home/away status
        if str(latest_game["home_team"]) == team:
            return {
                "home_prior_pa_avg_3": (latest_game.get("home_prior_pa_avg_3")),
                "home_prior_pa_avg_5": (latest_game.get("home_prior_pa_avg_5")),
                "home_prior_pf_avg_3": (latest_game.get("home_prior_pf_avg_3")),
                "home_prior_pf_avg_5": (latest_game.get("home_prior_pf_avg_5")),
                "home_prior_win_pct_3": (latest_game.get("home_prior_win_pct_3")),
                "home_prior_win_pct_5": (latest_game.get("home_prior_win_pct_5")),
            }
        else:
            return {
                "away_prior_pa_avg_3": (latest_game.get("away_prior_pa_avg_3")),
                "away_prior_pa_avg_5": (latest_game.get("away_prior_pa_avg_5")),
                "away_prior_pf_avg_3": (latest_game.get("away_prior_pf_avg_3")),
                "away_prior_pf_avg_5": (latest_game.get("away_prior_pf_avg_5")),
                "away_prior_win_pct_3": (latest_game.get("away_prior_win_pct_3")),
                "away_prior_win_pct_5": (latest_game.get("away_prior_win_pct_5")),
            }

    # Get latest features for both teams
    home_features = get_latest_team_features(home_team, season, week)
    away_features = get_latest_team_features(away_team, season, week)

    # Build the feature row in the same format as historical data
    feature_row = {}
    for stat in [
        "prior_pa_avg_3",
        "prior_pa_avg_5",
        "prior_pf_avg_3",
        "prior_pf_avg_5",
        "prior_win_pct_3",
        "prior_win_pct_5",
    ]:
        feature_row[f"home_{stat}"] = home_features[stat]
        feature_row[f"away_{stat}"] = away_features[stat]

    return pd.Series(feature_row)


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
        logger.info("Predicting %s vs %s (season=%s, week=%s)", home_team, away_team, season, week)

        # Canonicalize to abbreviations for dataset matching
        home_abbr = get_team_abbreviation(home_team)
        away_abbr = get_team_abbreviation(away_team)

        # Pull rolling priors from dataset - handle both historical and future games
        global dataset_df
        if dataset_df is None:
            logger.error("Dataset not loaded at startup; cannot build features")
            raise HTTPException(status_code=500, detail="Dataset not loaded")

        # First try exact match for historical games
        mask = (
            (dataset_df.get("season") == season)
            & (dataset_df.get("week") == week)
            & (dataset_df.get("home_team") == home_abbr)
            & (dataset_df.get("away_team") == away_abbr)
        )
        rows = dataset_df.loc[mask]

        if rows.empty:
            # Game not in dataset - likely a future game, build features from latest available data
            logger.info("Future game prediction - building features from latest team data")
            row = build_future_game_features(dataset_df, home_abbr, away_abbr, season, week)
        else:
            if len(rows) > 1:
                logger.warning("Multiple matches found for the same game; using the first row")
            row = rows.iloc[0]

            # CRITICAL: Check if game is already completed
            if pd.notna(row.get("home_points_for")) and pd.notna(row.get("away_points_for")):
                actual_home_score = int(row["home_points_for"])
                actual_away_score = int(row["away_points_for"])
                logger.warning(
                    "Attempted prediction on completed game: %s %d - %s %d",
                    away_abbr,
                    actual_away_score,
                    home_abbr,
                    actual_home_score,
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Game already completed: {away_abbr} {actual_away_score} - "
                        f"{home_abbr} {actual_home_score}. Cannot predict completed games."
                    ),
                )

        feature_cols = [
            "home_prior_pa_avg_3",
            "home_prior_pa_avg_5",
            "home_prior_pf_avg_3",
            "home_prior_pf_avg_5",
            "home_prior_win_pct_3",
            "home_prior_win_pct_5",
            "away_prior_pa_avg_3",
            "away_prior_pa_avg_5",
            "away_prior_pf_avg_3",
            "away_prior_pf_avg_5",
            "away_prior_win_pct_3",
            "away_prior_win_pct_5",
        ]
        missing = [c for c in feature_cols if c not in row.index]
        if missing:
            logger.error("Dataset missing required feature columns: %s", missing)
            raise HTTPException(status_code=500, detail=f"Missing feature columns: {missing}")
        input_df = pd.DataFrame({c: [row[c]] for c in feature_cols})

        X = model_objects["preprocessor"].transform(input_df)

        # Predict scores based on model types
        model_types = model_objects.get("model_types", {})

        # Home score prediction
        if model_types.get("home_model_type") == "neural_network":
            home_score = (model_objects["home_model"].predict(X, verbose=0)[0][0])
        else:
            home_score = (model_objects["home_model"].predict(X)[0])

        # Away score prediction
        if model_types.get("away_model_type") == "neural_network":
            away_score = (model_objects["away_model"].predict(X, verbose=0)[0][0])
        else:
            away_score = (model_objects["away_model"].predict(X)[0])
        # Clamp to a reasonable NFL score range
        home_score = round(max(0.0, min(70.0, home_score)), 1)
        away_score = round(max(0.0, min(70.0, away_score)), 1)
        point_diff = round(home_score - away_score, 1)

        # Derive win probabilities from point spread via logistic mapping
        # k tunes spread-to-probability steepness; can be calibrated later
        k = 0.22
        home_win_prob = 1.0 / (1.0 + np.exp(-k * point_diff))

        return PredictionResponse(
            home_score=home_score,
            away_score=away_score,
            home_win_probability=round((home_win_prob), 3),
            away_win_probability=round((1 - home_win_prob), 3),
            point_diff=point_diff,
            mode="models",
        )

    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.get("/predict/next-week")
def predict_next_week():
    """
    Predict outcomes for all games in the next NFL week.

    Automatically determines the current NFL state and predicts upcoming games.
    Returns predictions for the next week that should be played.
    """
    global model_objects, dataset_df

    if model_objects is None:
        logger.error("Models not loaded - cannot make predictions")
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        # Get current NFL context
        context = get_current_nfl_context()
        next_season = context["next_prediction_season"]
        next_week = context["next_prediction_week"]

        logger.info(
            "Predicting next week: %dW%d (last completed: %dW%d)",
            next_season,
            next_week,
            context["last_completed_season"],
            context["last_completed_week"],
        )

        # Load schedule for the next week
        schedule_path = BASE_DIR / "backend" / "data" / "Nfl_schedule_2025_2026.csv"
        if not schedule_path.exists():
            raise HTTPException(status_code=404, detail="Schedule data not found")

        schedule_df = pd.read_csv(schedule_path)

        # Filter to next week's games
        next_week_games = schedule_df[
            (schedule_df["season"] == next_season) & (schedule_df["week"] == next_week)
        ]

        if next_week_games.empty:
            return {
                "context": context,
                "games": [],
                "message": f"No games scheduled for {next_season}W{next_week}",
            }

        predictions = []
        for _, game in next_week_games.iterrows():
            try:
                # Create prediction request
                request = PredictionRequest(
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    season=int(game["season"]),
                    week=int(game["week"]),
                )

                # Get prediction (reuse existing logic)
                prediction = predict_game(request)

                predictions.append(
                    {
                        "game_id": str(
                            game.get(
                                "game_id",
                                f"{game['season']}W{game['week']}-"
                                f"{game['away_team']}@{game['home_team']}",
                            )
                        ),
                        "season": int(game["season"]),
                        "week": int(game["week"]),
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "kickoff": game.get("gameday", "TBD"),
                        "prediction": prediction.dict(),
                    }
                )

            except Exception as e:
                logger.warning(
                    "Failed to predict game %s @ %s: %s", game["away_team"], game["home_team"], e
                )
                predictions.append(
                    {
                        "game_id": str(game.get("game_id", "unknown")),
                        "season": int(game["season"]),
                        "week": int(game["week"]),
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "error": str(e),
                    }
                )

        return {
            "context": context,
            "games": predictions,
            "total_games": len(predictions),
            "successful_predictions": len([p for p in predictions if "prediction" in p]),
        }

    except Exception as e:
        logger.error("Next week prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to predict next week: {e}")


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
        subprocess.run(
            ["python", str(train_script_path)], check=True, capture_output=True, text=True
        )
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
            [
                sys.executable,
                "scripts/build_csvs.py",
                "--start",
                "2014",
                "--end",
                "2024",
                "--out-dir",
                "data",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("build_csvs stdout:\n%s", build.stdout)
        logger.info("build_csvs stderr:\n%s", build.stderr)

        # 2) Retrain
        train = subprocess.run(
            [sys.executable, "train_models.py"], check=True, capture_output=True, text=True
        )
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
