#!/usr/bin/env python
"""
train_models.py
===============

Purpose
-------
Train two regressors (LightGBM) to directly predict home and away points.
The API will compute point_diff = home_points_for - away_points_for from these
predictions. This aligns the web app with score-based outputs.

Key Steps
---------
1) Load CSV: `<repo_root>/Nfl_data_sorted.csv`
2) Select engineered prior features (no outcome leakage)
3) ColumnTransformer: impute + scale numeric features
4) Train LGBMRegressor for home_points_for and away_points_for on ALL data
5) Persist artefacts to `<repo_root>/backend/models/` + write `metadata.json`

External Dependencies
---------------------
pandas, numpy, scikit-learn, lightgbm, joblib

Usage Notes
-----------
- Trains on all available rows to maximize data for production predictions.
- For offline evaluation (e.g., 2025 week 1), use `backend/test_train_models.py`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor

# -----------------------------------------------------------------------------
# Paths & logging
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
# Use the sorted dataset used elsewhere in the repo
DATA_PATH = BASE_DIR / "Nfl_data_sorted.csv"
MODELS_DIR = BASE_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def _get_feature_list(df: pd.DataFrame) -> List[str]:
    """Return the numeric prior features used for score prediction.

    This function validates presence and provides a single source of truth
    for features for both training and inference.
    """
    features = [
        'home_prior_pa_avg_3', 'home_prior_pa_avg_5', 'home_prior_pf_avg_3',
        'home_prior_pf_avg_5', 'home_prior_win_pct_3', 'home_prior_win_pct_5',
        'away_prior_pa_avg_3', 'away_prior_pa_avg_5', 'away_prior_pf_avg_3',
        'away_prior_pf_avg_5', 'away_prior_win_pct_3', 'away_prior_win_pct_5'
    ]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return features


# -----------------------------------------------------------------------------
# Modeling helpers
# -----------------------------------------------------------------------------

def _fit_lgbm_regressor(X: np.ndarray, y: np.ndarray) -> LGBMRegressor:
    """Train a LightGBM regressor with sensible defaults."""
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model


# -----------------------------------------------------------------------------
# Main training pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    # Validate dataset exists and load
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    
    # Production validation: ensure we have sufficient data for training
    if len(df) < 100:
        raise ValueError(f"Insufficient training data: only {len(df)} rows available")
    
    # Filter to completed games only (remove any future/scheduled games)
    valid_scores = df['home_points_for'].notna() & df['away_points_for'].notna()
    df = df[valid_scores].reset_index(drop=True)
    
    logger.info("Training on %d completed games from %s", len(df), DATA_PATH)
    logger.info("Season range: %d-%d, Week range: %d-%d", 
                df['season'].min(), df['season'].max(),
                df['week'].min(), df['week'].max())

    # Feature matrix with validation
    features = _get_feature_list(df)
    X = df[features]
    
    # Validate no missing features in production dataset
    missing_count = X.isnull().sum().sum()
    if missing_count > len(X) * 0.05:  # More than 5% missing
        raise ValueError(f"Too many missing feature values: {missing_count}/{len(X)}")

    # Targets with validation
    if not {'home_points_for', 'away_points_for'}.issubset(df.columns):
        raise ValueError("Dataset missing required target columns")
    
    y_home = df['home_points_for']
    y_away = df['away_points_for']
    
    # Validate target ranges (NFL scores should be 0-70)
    if y_home.min() < 0 or y_home.max() > 80 or y_away.min() < 0 or y_away.max() > 80:
        raise ValueError("Invalid score values detected in training data")

    # Production preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")), 
                ("scaler", StandardScaler())
            ]), features)
        ],
        remainder="drop",
    )

    X_proc = preprocessor.fit_transform(X)
    logger.info("Preprocessing pipeline fitted on %d samples with %d features", 
                X_proc.shape[0], X_proc.shape[1])

    # Train models with production validation
    home_model = _fit_lgbm_regressor(X_proc, y_home)
    away_model = _fit_lgbm_regressor(X_proc, y_away)
    
    # Validate model training success
    home_train_score = home_model.score(X_proc, y_home)
    away_train_score = away_model.score(X_proc, y_away)
    
    if home_train_score < 0.1 or away_train_score < 0.1:
        raise ValueError(f"Model training failed - poor R² scores: home={home_train_score:.3f}, away={away_train_score:.3f}")
    
    logger.info("Model training complete - R² scores: home=%.3f, away=%.3f", 
                home_train_score, away_train_score)

    # Production model persistence with validation
    try:
        joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
        joblib.dump(home_model, MODELS_DIR / "home_model.joblib") 
        joblib.dump(away_model, MODELS_DIR / "away_model.joblib")
        logger.info("Successfully saved models to %s", MODELS_DIR)
    except Exception as e:
        raise RuntimeError(f"Failed to save trained models: {e}") from e

    # Production metadata with comprehensive information
    feature_names: List[str] = []
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        logger.warning("Could not extract transformed feature names")
        feature_names = []

    # Calculate dataset hash for cache invalidation
    import hashlib
    dataset_hash = hashlib.md5(df.to_string().encode()).hexdigest()[:8]
    
    from datetime import datetime
    training_timestamp = datetime.now().isoformat()

    meta = {
        "training_timestamp": training_timestamp,
        "dataset_hash": dataset_hash,
        "dataset_path": str(DATA_PATH),
        "training_samples": len(df),
        "season_range": [int(df['season'].min()), int(df['season'].max())],
        "week_range": [int(df['week'].min()), int(df['week'].max())],
        "model_scores": {
            "home_r2": float(home_train_score), 
            "away_r2": float(away_train_score)
        },
        "raw_feature_columns": {"numeric": features, "categorical": []},
        "transformed_feature_names": feature_names,
        "models": {"home_model": "home_model.joblib", "away_model": "away_model.joblib"},
        "preprocessor": "preprocessor.joblib",
        "target_names": ["home_points_for", "away_points_for"],
    }
    
    try:
        with open(MODELS_DIR / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Model training complete - Ready for production use")
    except Exception as e:
        raise RuntimeError(f"Failed to save model metadata: {e}") from e


if __name__ == "__main__":
    main()

# -----------------------------
# Production Enhancement Notes  
# -----------------------------
# ✓ Enhanced validation of training data quality and completeness
# ✓ Added comprehensive metadata with training metrics and dataset versioning
# ✓ Implemented proper error handling and fail-fast validation
# ✓ Uses ALL available data including newly added weeks for optimal predictions
