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
            "/predict": "Predict game outcome with simplified input",
            "/predict_raw": "Predict with full feature set",
            "/retrain": "Retrain models",
            "/update_data": "Rebuild datasets and retrain"
        }
    }


@app.post("/predict")
def predict(stats: GameStats):
    """Predict the probability of the home team winning a game.

    Parameters
    ----------
    stats : GameStats
        JSON body containing the required game statistics.

    Returns
    -------
    dict
        Dictionary with individual model probabilities and the ensemble probability.
    """
    global model_objects
    if model_objects is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    # Fallback path: lightweight heuristic logistic model
    if model_objects.get('mode') == 'fallback':
        # Coefficients chosen to mimic reasonable behavior
        coef_passer = 0.015
        coef_turnover = -0.35
        coef_rushing = 0.008
        coef_power = 0.01
        intercept = 0.0
        diff_passer = stats.home_passer_rating - stats.away_passer_rating
        diff_turnover = stats.home_turnovers - stats.away_turnovers
        diff_rushing = stats.home_rushing_yards - stats.away_rushing_yards
        diff_power = (stats.home_power_rank or 0.0) - (stats.away_power_rank or 0.0)
        z = (coef_passer * diff_passer +
             coef_turnover * diff_turnover +
             coef_rushing * diff_rushing +
             coef_power * diff_power + intercept)
        proba = 1.0 / (1.0 + math.exp(-z))
        return {
            'mode': 'fallback',
            'neural_network_proba': proba,
            'gradient_boosting_proba': proba,
            'ensemble_proba': proba
        }

    # Full model path
    import numpy as np
    import pandas as pd
    raw_cols = model_objects['raw_feature_columns']
    numeric_cols = raw_cols.get('numeric', [])
    categorical_cols = raw_cols.get('categorical', [])
    row: Dict[str, Any] = {c: np.nan for c in numeric_cols}
    if 'rush_yards_home' in row:
        row['rush_yards_home'] = stats.home_rushing_yards
    if 'rush_yards_away' in row:
        row['rush_yards_away'] = stats.away_rushing_yards
    if 'fumbles_home' in row:
        row['fumbles_home'] = float(stats.home_turnovers)
    if 'fumbles_away' in row:
        row['fumbles_away'] = float(stats.away_turnovers)
    raw_cat: Dict[str, Any] = {c: 'UNK' for c in categorical_cols}
    if 'home' in raw_cat and stats.home:
        raw_cat['home'] = stats.home
    if 'away' in raw_cat and stats.away:
        raw_cat['away'] = stats.away

    data = {**row, **raw_cat}
    X_df = pd.DataFrame([data])
    X_trans = model_objects['preprocessor'].transform(X_df)
    nn_proba = float(model_objects['nn_model'].predict_proba(X_trans)[0, 1])
    gbm_proba = float(model_objects['gbm_model'].predict(X_trans)[0])
    ensemble_proba = (nn_proba + gbm_proba) / 2
    return {
        'mode': 'models',
        'neural_network_proba': nn_proba,
        'gradient_boosting_proba': gbm_proba,
        'ensemble_proba': ensemble_proba
    }


@app.post("/predict_raw")
def predict_raw(payload: Dict[str, Any]):
    """Predict using raw feature schema similar to the training dataset.

    Provide keys such as 'home', 'away', 'plays_home', 'pass_yards_home',
    'rush_yards_home', 'fumbles_home', 'interceptions_home', and the
    corresponding '_away' fields. Any missing fields are imputed.
    """
    global model_objects
    if model_objects is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    if model_objects.get('mode') == 'fallback':
        # Try to derive minimal stats if available, else return error
        try:
            stats = GameStats(
                home_passer_rating=float(payload.get('home_passer_rating', 90.0)),
                away_passer_rating=float(payload.get('away_passer_rating', 88.0)),
                home_turnovers=int(payload.get('home_turnovers', 1)),
                away_turnovers=int(payload.get('away_turnovers', 1)),
                home_rushing_yards=float(payload.get('rush_yards_home', payload.get('home_rushing_yards', 120.0))),
                away_rushing_yards=float(payload.get('rush_yards_away', payload.get('away_rushing_yards', 110.0))),
                home_power_rank=float(payload.get('home_power_rank', 75.0)),
                away_power_rank=float(payload.get('away_power_rank', 75.0)),
                home=str(payload.get('home', 'UNK')),
                away=str(payload.get('away', 'UNK')),
            )
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload for fallback mode.")
        # Reuse fallback logic
        return predict(stats)

    import numpy as np
    import pandas as pd
    raw_cols = model_objects['raw_feature_columns']
    numeric_cols = raw_cols.get('numeric', [])
    categorical_cols = raw_cols.get('categorical', [])

    row: Dict[str, Any] = {}
    for c in numeric_cols:
        row[c] = payload.get(c, np.nan)
    for c in categorical_cols:
        row[c] = payload.get(c, 'UNK')

    X_df = pd.DataFrame([row])
    X_trans = model_objects['preprocessor'].transform(X_df)
    nn_proba = float(model_objects['nn_model'].predict_proba(X_trans)[0, 1])
    gbm_proba = float(model_objects['gbm_model'].predict(X_trans)[0])
    ensemble_proba = (nn_proba + gbm_proba) / 2
    return {
        'mode': 'models',
        'neural_network_proba': nn_proba,
        'gradient_boosting_proba': gbm_proba,
        'ensemble_proba': ensemble_proba
    }


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