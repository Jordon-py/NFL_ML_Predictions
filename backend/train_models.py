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
    # Load entire dataset
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)

    # Feature matrix
    features = _get_feature_list(df)
    X = df[features]

    # Targets
    if not {'home_points_for', 'away_points_for'}.issubset(df.columns):
        raise ValueError("Dataset is missing 'home_points_for' and/or 'away_points_for'.")
    y_home = df['home_points_for']
    y_away = df['away_points_for']

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)
        ],
        remainder="drop",
    )

    X_proc = preprocessor.fit_transform(X)
    logger.info("Preprocessing fitted on full dataset.")

    # Train regressors
    home_model = _fit_lgbm_regressor(X_proc, y_home)
    away_model = _fit_lgbm_regressor(X_proc, y_away)

    # Persist
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(home_model, MODELS_DIR / "home_model.joblib")
    joblib.dump(away_model, MODELS_DIR / "away_model.joblib")
    logger.info("Saved preprocessor and models to %s", MODELS_DIR)

    # Metadata for API loader
    feature_names: List[str] = []
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = []

    meta = {
        "raw_feature_columns": {"numeric": features, "categorical": []},
        "transformed_feature_names": feature_names,
        "models": {"home_model": "home_model.joblib", "away_model": "away_model.joblib"},
        "preprocessor": "preprocessor.joblib",
        "target_names": ["home_points_for", "away_points_for"],
        "dataset_path": str(DATA_PATH),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

# -----------------------------
# Suggested Enhancements
# -----------------------------
# 1) Add walk-forward evaluation to quantify MAE/RMSE per season/week.
# 2) Add post-processing to avoid impossible scores (clip 0–70).
# 3) Log model + data hashes (e.g., md5 of CSV, model params) into metadata.json
#    for reproducibility and cache invalidation in deployment.
