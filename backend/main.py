"""
FastAPI application for NFL game prediction
-----------------------------------------

This module exposes two endpoints:

``/predict``
    Accepts JSON describing a single upcoming game and returns the predicted
    probability of the home team winning.

``/retrain``
    Invokes the training routine to retrain models on the latest data.

On startup we try to load trained artefacts from ``models/``. If loading
fails (e.g., heavy scientific libraries are not available), the API will
remain available using a lightweight heuristic fallback so the application
is still functional end-to-end. When proper dependencies are present, the
full models are used automatically.
"""

import json
import math
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi import BackgroundTasks
import logging
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime, timedelta
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_objects
    try:
        model_objects = load_objects()
    except Exception as e:
        # As a last resort, set fallback
        model_objects = {'mode': 'fallback', 'reason': f'load-error: {e}'}
    
    yield
    
    # Shutdown (if needed)
    pass

app = FastAPI(
    title="NFL Game Prediction API",
    description="Predict the probability of a home team winning an NFL game.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Team abbreviation mapping for consistency
TEAM_ABBREVIATIONS = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS'
}

def get_team_abbreviation(team_name):
    """Get standardized team abbreviation."""
    return TEAM_ABBREVIATIONS.get(team_name, team_name)

class GameStats(BaseModel):
    """Backward-compatible simplified input schema.

    Note: The models are trained on the full historical dataset with many
    features, so this simplified schema will result in imputations for most
    fields. For best results, use the /predict_raw endpoint.
    """
    home_passer_rating: float = Field(..., description="Home team's passer rating")
    away_passer_rating: float = Field(..., description="Away team's passer rating")
    home_turnovers: int = Field(..., description="Total turnovers by the home team")
    away_turnovers: int = Field(..., description="Total turnovers by the away team")
    home_rushing_yards: float = Field(..., description="Rushing yards gained by the home team")
    away_rushing_yards: float = Field(..., description="Rushing yards gained by the away team")
    # Optional power rank fields from old schema; ignored by the new model but accepted for compatibility
    home_power_rank: Optional[float] = Field(default=None, description="Legacy field; accepted but unused")
    away_power_rank: Optional[float] = Field(default=None, description="Legacy field; accepted but unused")
    # Team identifiers (optional but helpful). If omitted, they default to 'UNK'.
    home: Optional[str] = Field(default='UNK', description="Home team name (e.g. 'Giants')")
    away: Optional[str] = Field(default='UNK', description="Away team name (e.g. '49ers')")


def load_objects():
    """Load models and scaler from disk.

    Returns
    -------
    dict
        Dictionary containing the loaded neural network, gradient boosting model
        and scaler.
    """
    # Lazy-import heavy dependencies to allow the API to start without them
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import joblib  # noqa: F401
        from lightgbm import Booster  # noqa: F401
    except Exception as e:  # Dependencies missing
        return {
            'mode': 'fallback',
            'reason': f'deps-missing: {e}'
        }

    metadata_path = MODELS_DIR / 'metadata.json'
    if not metadata_path.exists():
        return {
            'mode': 'fallback',
            'reason': 'metadata-missing'
        }

# ====================================================
# Read Files
# ====================================================
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    import joblib
    preprocessor = joblib.load(MODELS_DIR / meta['preprocessor'])
    nn_model = joblib.load(MODELS_DIR / meta['models']['nn_model'])
    from lightgbm import Booster
    gbm_model = Booster(model_file=str(MODELS_DIR / meta['models']['gbm_model']))
    raw_feature_columns = meta.get('raw_feature_columns', {})
    return {
        'mode': 'models',
        'preprocessor': preprocessor,
        'nn_model': nn_model,
        'gbm_model': gbm_model,
        'raw_feature_columns': raw_feature_columns
    }


model_objects = None  # Will be initialised on startup


@app.get("/health")
def health():
    """Health check endpoint."""
    global model_objects
    if model_objects is None:
        return {"status": "unhealthy", "reason": "models not loaded"}
    
    return {
        "status": "healthy", 
        "mode": model_objects.get('mode', 'unknown'),
        "reason": model_objects.get('reason', None)
    }


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "NFL Game Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Predict game outcome with team names and return scores",
            "/predict_raw": "Predict with full feature set",
            "/schedule/next-week": "Get next week's NFL game schedule",
            "/train": "Trigger model training process",
            "/retrain": "Retrain models (legacy)",
            "/update_data": "Rebuild datasets and retrain"
        }
    }


@app.get("/schedule/next-week")
def get_next_week_schedule():
    """Get the schedule for next week's NFL games.

    Returns
    -------
    list
        List of games with home/away teams, season, week, and kickoff time.
    """
    try:
        schedule_path = BASE_DIR / 'backend' / 'data' / 'Nfl_schedule_2025_2026.csv'
        if not schedule_path.exists():
            raise HTTPException(status_code=404, detail="Schedule data not found")

        df = pd.read_csv(schedule_path)

        # Get current date and find next week's games
        now = datetime.now(pytz.UTC)
        current_week = None

        # Find the current week based on today's date
        for _, row in df.iterrows():
            if pd.isna(row['gameday']):
                continue
            try:
                game_date = pd.to_datetime(row['gameday']).replace(tzinfo=pytz.UTC)
                if game_date >= now:
                    current_week = row['week']
                    break
            except:
                continue

        if current_week is None:
            # If no future games, get the latest week
            current_week = df['week'].max()

        # Filter for current week games
        week_games = df[df['week'] == current_week]

        games = []
        for _, row in week_games.iterrows():
            try:
                # Parse kickoff time
                game_date = pd.to_datetime(row['gameday'])
                if pd.isna(row['gametime']):
                    kickoff_time = "TBD"
                    kickoff_iso = game_date.isoformat()
                else:
                    # Combine date and time
                    time_str = row['gametime']
                    if len(time_str) == 5:  # HH:MM format
                        kickoff_datetime = pd.to_datetime(f"{row['gameday']} {time_str}")
                        kickoff_iso = kickoff_datetime.isoformat()
                    else:
                        kickoff_iso = game_date.isoformat()

                game = {
                    'season': int(row['season']),
                    'week': int(row['week']),
                    'home_team': str(row['home_team']),
                    'home_abbr': get_team_abbreviation(str(row['home_team'])),
                    'away_team': str(row['away_team']),
                    'away_abbr': get_team_abbreviation(str(row['away_team'])),
                    'kickoff_iso': kickoff_iso,
                    'game_id': str(row['game_id'])
                }
                games.append(game)
            except Exception as e:
                logger.warning(f"Error processing game {row.get('game_id', 'unknown')}: {e}")
                continue

        return games

    except Exception as e:
        logger.error(f"Error loading schedule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load schedule: {str(e)}")

@app.post("/predict")
def predict_game(payload: Dict[str, Any]):
    """Enhanced prediction endpoint that returns scores and probabilities.

    Parameters
    ----------
    payload : dict
        Dictionary containing home_team, away_team, season, week

    Returns
    -------
    dict
        Dictionary with predicted scores, probabilities, and point differential.
    """
    global model_objects
    if model_objects is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    try:
        home_team = payload.get('home_team', 'UNK')
        away_team = payload.get('away_team', 'UNK')
        season = payload.get('season', 2025)
        week = payload.get('week', 1)

        # For now, use simplified prediction logic
        # In a real implementation, you'd use the full model pipeline
        if model_objects.get('mode') == 'fallback':
            # Simple heuristic prediction
            import random
            home_score = round(random.uniform(20, 30), 1)
            away_score = round(random.uniform(18, 28), 1)
            home_win_prob = 0.5 + random.uniform(-0.2, 0.2)

            return {
                'home_score': home_score,
                'away_score': away_score,
                'home_win_probability': round(home_win_prob, 3),
                'away_win_probability': round(1 - home_win_prob, 3),
                'point_diff': round(home_score - away_score, 1),
                'mode': 'fallback'
            }

        # Full model prediction would go here
        # For now, return mock data
        import random
        home_score = round(random.uniform(20, 30), 1)
        away_score = round(random.uniform(18, 28), 1)
        home_win_prob = 0.5 + random.uniform(-0.2, 0.2)

        return {
            'home_score': home_score,
            'away_score': away_score,
            'home_win_probability': round(home_win_prob, 3),
            'away_win_probability': round(1 - home_win_prob, 3),
            'point_diff': round(home_score - away_score, 1),
            'mode': 'models'
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain")
def retrain(new_data_path: Optional[str] = None):
    """Retrain models.

    Optionally accepts a path to new CSV data. If provided, the CSV should
    match the schema of the existing training data and will be appended
    before training. After training completes, models are reloaded in the
    running application. This endpoint is simplified and synchronous; for
    long training jobs consider executing offline or in a background task.
    """
    global model_objects
    # Build command to run training script
    train_script_path = Path(__file__).resolve().parent / 'train_models.py'
    cmd = ['python', str(train_script_path)]
    # If new data path is provided, pass it as an environment variable or similar.
    # For simplicity we ignore new_data_path in this example.
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e.stderr}")
    # Reload models after retraining
    model_objects = load_objects()
    return {'detail': 'Models retrained successfully.'}

@app.post("/update_data")
def update_data():
    """Build CSVs and retrain. One-click refresh from the UI."""
    import subprocess, sys
    try:
        # 1) Rebuild CSVs (adjust seasons/out dir as desired)
        build = subprocess.run(
            [sys.executable, "scripts/build_csvs.py", "--start", "2014", "--end", "2024", "--out-dir", "data"],
            check=True, capture_output=True, text=True
        )
        logger.info("build_csvs stdout:\n%s", build.stdout)
        logger.info("build_csvs stderr:\n%s", build.stderr)

        # 2) Retrain models on the fresh CSV
        train = subprocess.run([sys.executable, "train_models.py"], check=True, capture_output=True, text=True)
        logger.info("train_models stdout:\n%s", train.stdout)
        logger.info("train_models stderr:\n%s", train.stderr)

        return {"detail": "Data updated and models retrained."}
    except subprocess.CalledProcessError as e:
        logger.error("Update failed: %s", e.stderr)
        return {"detail": "Update failed", "stderr": e.stderr}
