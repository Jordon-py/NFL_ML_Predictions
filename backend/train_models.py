#!/usr/bin/env python
"""
train_models.py
===============

Purpose
-------
Train a pair of classifiers (MLP "neural net" + LightGBM) to predict home win
probability using the game-level dataset built by `build_csv_datasets.py`.

Key Steps
---------
1) Load raw CSV: `<repo_root>/backend/data/Nfl_data.csv`
2) Prepare frames: derive target (`win = home_points_for > away_points_for`),
   drop outcome leakage columns, keep remaining features.
3) ColumnTransformer: scale numeric, one-hot encode categoricals.
4) Train MLPClassifier + LGBMClassifier; evaluate on 2024 validation/test splits.
5) Persist artefacts to `<repo_root>/backend/models/` + write `metadata.json`.

External Dependencies
---------------------
pandas, numpy, scikit-learn, lightgbm, joblib

Usage Notes
-----------
- Time-aware split: Train = seasons 2002–2023; Val/Test = 2024 (weeks ≤4 vs ≥5).
- Artefacts expected by API (`main.py`) via `metadata.json` keys:
- preprocessor.joblib, nn_model.joblib, gbm_model.txt

**IMPORTANT** TO RUN:
python backend/train_models.py  

"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pandas.api.types as pdt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

# -----------------------------------------------------------------------------
# Paths & logging
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "backend" / "data" / "Nfl_data.csv"
MODELS_DIR = BASE_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SimpleNN = MLPClassifier  # alias for clarity


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def build_prepared_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Produce (X, y) from raw game-level dataframe.

    Required columns
    ----------------
    - 'home_points_for', 'away_points_for', 'home_team', 'away_team'
    """
    df = df.copy()
    required = ["home_points_for", "away_points_for", "home_team", "away_team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Target: 1 if home wins, else 0
    y = (df["home_points_for"] > df["away_points_for"]).astype(int)

    # Drop target and direct outcome leakage columns
    drop_cols = ["home_points_for", "away_points_for", "point_diff", "winner"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return X, y


def load_raw_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CSV and create time-aware splits:
      - Train: seasons 2002–2023
      - Val:   2024 weeks ≤ 4
      - Test:  2024 weeks ≥ 5
    """
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")

    train_df = df[df["season"].between(2002, 2023)].reset_index(drop=True)
    season_2024 = df[df["season"] == 2024].reset_index(drop=True)

    if "week" in season_2024.columns and not season_2024.empty:
        val_df = season_2024[season_2024["week"] <= 4].reset_index(drop=True)
        test_df = season_2024[season_2024["week"] >= 5].reset_index(drop=True)
    else:
        # Deterministic fallback if 'week' missing; fail fast if empty.
        split_idx = max(1, int(0.25 * len(season_2024))) if len(season_2024) else 0
        val_df = season_2024.iloc[:split_idx].reset_index(drop=True)
        test_df = season_2024.iloc[split_idx:].reset_index(drop=True)

    if val_df.empty or test_df.empty:
        raise RuntimeError("Validation/test split is empty; check 2024 data and 'week' column.")
    return train_df, val_df, test_df


# -----------------------------------------------------------------------------
# Modeling helpers
# -----------------------------------------------------------------------------

def train_neural_network(
    X_train: np.ndarray, y_train: np.ndarray, num_epochs: int = 40, batch_size: int = 128, lr: float = 1e-3
) -> SimpleNN:
    """Fit a small MLP; returns trained estimator."""
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=num_epochs,
        batch_size=batch_size,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    return mlp


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Return accuracy, log loss, and Brier score for probability outputs."""
    y_hat = (y_proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_hat),
        "log_loss": log_loss(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
    }


def _dense2d(a) -> np.ndarray:
    """Ensure 2D dense numpy array from various estimator outputs."""
    toarray = getattr(a, "toarray", None)
    arr = np.asarray(toarray()) if callable(toarray) else np.asarray(a)
    return arr if arr.ndim == 2 else arr.reshape(-1, 2)


# -----------------------------------------------------------------------------
# Main training pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    # 1) Load time-aware splits
    train_raw, val_raw, test_raw = load_raw_splits()

    # 2) Prepare features/targets
    X_train_df, y_train = build_prepared_frames(train_raw)
    X_val_df, y_val = build_prepared_frames(val_raw)
    X_test_df, y_test = build_prepared_frames(test_raw)

    # 3) Identify column types
    all_cat = X_train_df.select_dtypes(include=["object"]).columns.tolist()
    exclude_cat = ["game_id", "team_name", "opponent_name"]  # highly specific keys; avoid overfit
    cat_cols = [c for c in all_cat if c not in exclude_cat]
    num_cols = [c for c in X_train_df.columns if c not in cat_cols and pdt.is_numeric_dtype(X_train_df[c])]

    logger.info("Categorical columns: %s", cat_cols)
    logger.info("Numeric columns: %d columns", len(num_cols))
    logger.info("Total features: %d", len(cat_cols) + len(num_cols))

    # 4) Preprocess: impute→scale numerics, impute→OHE categoricals
    transformers = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols)
        )
    if cat_cols:
        transformers.append(
            ("cat", Pipeline([("imputer", SimpleImputer(fill_value="UNK", strategy="constant")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Fit/transform (note: ensures dense arrays downstream)
    X_train = np.asarray(preprocessor.fit_transform(X_train_df))
    X_val = np.asarray(preprocessor.transform(X_val_df))
    X_test = np.asarray(preprocessor.transform(X_test_df))

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    # 5) Train models
    nn_model = train_neural_network(X_train, y_train)

    gbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ).fit(X_train, y_train)

    # 6) Evaluate
    nn_val = nn_model.predict_proba(X_val)[:, 1]
    nn_test = nn_model.predict_proba(X_test)[:, 1]

    gbm_val = _dense2d(gbm_model.predict_proba(X_val))[:, 1]
    gbm_test = _dense2d(gbm_model.predict_proba(X_test))[:, 1]

    ens_val = (nn_val + gbm_val) / 2.0
    ens_test = (nn_test + gbm_test) / 2.0

    logger.info("Validation metrics:")
    for name, proba in [("neural_network", nn_val), ("gradient_boosting", gbm_val), ("ensemble", ens_val)]:
        m = evaluate_predictions(y_val, proba)
        logger.info(" %s: accuracy=%.3f, log_loss=%.3f, brier_score=%.3f", name, m["accuracy"], m["log_loss"], m["brier_score"])

    logger.info("Test metrics:")
    for name, proba in [("neural_network", nn_test), ("gradient_boosting", gbm_test), ("ensemble", ens_test)]:
        m = evaluate_predictions(y_test, proba)
        logger.info(" %s: accuracy=%.3f, log_loss=%.3f, brier_score=%.3f", name, m["accuracy"], m["log_loss"], m["brier_score"])

    # 7) Persist artefacts
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(nn_model, MODELS_DIR / "nn_model.joblib")
    gbm_model.booster_.save_model(str(MODELS_DIR / "gbm_model.txt"))

    # 8) Metadata for API loader
    feature_names = []
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        pass

    meta = {
        "raw_feature_columns": {"numeric": num_cols, "categorical": cat_cols},
        "transformed_feature_names": feature_names,
        "models": {"nn_model": "nn_model.joblib", "gbm_model": "gbm_model.txt"},
        "preprocessor": "preprocessor.joblib",
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

# -----------------------------
# Suggested Enhancements
# -----------------------------
# 1) Replace fixed 2024-based split with season-aware walk-forward CV to better
#    estimate generalization for future weeks (saves fold metrics & plots).
# 2) Add calibrated probabilities (e.g., Platt/Isotonic on validation) to improve
#    probability quality used by the API for score shaping.
# 3) Log model + data hashes (e.g., md5 of CSV, model params) into metadata.json
#    for reproducibility and cache invalidation in deployment.
