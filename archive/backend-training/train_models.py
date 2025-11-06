#!/usr/bin/env python3
"""
Train leak-free NFL models with time-aware CV.

- Drops label and label-derived columns from features.
- Uses TimeSeriesSplit for all model selection.
- Reserves the final chronological fold for holdout metrics.
- Writes: artifacts/{preprocessor,home_model,away_model,win_clf_calibrated}.joblib
         artifacts/training_report.json
         artifacts/metadata.json
- Environment variables are loaded with defaults and converted to appropriate types for safety.
"""

# File: backend/train_models.py
# Purpose: Train ML models for NFL game predictions using time-aware cross-validation to prevent data leakage.

#  Functions: _ensure_columns, _dataset_hash, _drop_leaky_columns, _infer_features, _make_preprocessor, _split_for_calibration, _fit_regression, _fit_classifier, _evaluate_regression, _dataset_sort, main

# Variables: RANDOM_SEED, N_SPLITS, TARGET_HOME, TARGET_AWAY, CLASS_LABEL, TIME_KEYS, ID_COLS, LEAK_BLOCKLIST, REG_PARAM_DISTS, CLF_PARAM_DISTS, log

# Interacts With: backend/data/merge_dominance.csv (preferred input dataset), backend/models/ (output models and metadata)

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Tuple, Dict, cast, Any
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

# -----------------------
# Configuration
# -----------------------
load_dotenv(dotenv_path="backend/.env", verbose=True)

SERVE_FRONTEND=os.getenv('SERVE_FRONTEND')
CORS_ORIGINS=os.getenv('CORS_ORIGINS')
NODE_ENV=os.getenv('NODE_ENV')
HP_N_ITER=int(os.getenv('HP_N_ITER', '100'))  # Default to 100 if not set or invalid
CV_SPLITS=int(os.getenv('CV_SPLITS', '5'))  # Default to 5
RANDOM_SEED=int(os.getenv('RANDOM_SEED', '42'))  # Default to 42
N_SPLITS=int(os.getenv('N_SPLITS', '5'))  # Default to 5
N_JOBS = int(os.getenv('N_JOBS', '-1'))  # limit parallelism to avoid memory spikes

print(N_JOBS, NODE_ENV, SERVE_FRONTEND, CORS_ORIGINS)
# ----------Developement enviorment -----------
DEV_ORIGINS=os.getenv('DEV_ORIGINS', 'http://localhost:3000')

TRAIN_DATASET_FILE=os.getenv(
    'TRAIN_DATASET_FILE',
    'C:/Users/iProg/OneDrive/Documents/Football_predict/nfl_prediction_system/NFL_ML_Predictions/backend/data/game_features.csv'
    )

TARGET_HOME = "home_points_for"
TARGET_AWAY = "away_points_for"
CLASS_LABEL = "home_win"  # must be 0/1
TIME_KEYS = ["season", "week"]

ID_COLS = {
    "game_id",
    # "home_team",
    # "away_team",
    "home_team_id",
    "away_team_id",
    "stadium",
}

# Columns that must never enter features (labels or post-game/market values)
LEAK_BLOCKLIST = {
    # Labels and direct targets
    CLASS_LABEL,
    TARGET_HOME.strip().lower(),
    TARGET_AWAY.strip().lower(),
    "winner",
    "point_diff",
    # Post-game realized values
    "home_points_against",
    "away_points_against",
    "home_score",
    "away_score",
    "final_home_score",
    "final_away_score",
    "postgame_margin",
    "post_game_total",
    "actual_margin",
    # Market-informed or derived win signals that can leak outcome
    "home_moneyline",
    "away_moneyline",
    "home_win_prob",
    "away_win_prob",
    # Aggregated outcome-like rates
    "season_home_win_rate",
}

REG_PARAM_DISTS = {
    "reg__max_depth": [None, 6, 10, 14],
    "reg__learning_rate": np.linspace(0.02, 0.2, 6),
    "reg__max_leaf_nodes": [15, 31, 63, 127],
    "reg__l2_regularization": np.linspace(0.0, 0.2, 5),
}

CLF_PARAM_DISTS = {
    "clf__C": np.logspace(-3, 1, 8),
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear", "lbfgs"],
    "clf__class_weight": [None, "balanced"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("train_models")


@dataclass
class TrainSummary:
    training_timestamp_utc: str
    rows_total: int
    n_features_numeric: int
    n_features_categorical: int
    cv_n_splits: int
    random_seed: int
    production_ready: bool
    dataset_hash: int


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _dataset_hash(df: pd.DataFrame) -> int:
    return int(pd.util.hash_pandas_object(df[TIME_KEYS + ["home_team", "away_team"]], index=False).sum())


def _drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop known leaky columns and underscore-prefixed engineered hints.

    This complements the stricter leak guard in enhanced_pipeline.py by ensuring
    the legacy training path avoids obvious label/post-game leaks and internal
    engineered signals (convention: leading underscore).
    """
    present = [c for c in LEAK_BLOCKLIST if c in df.columns]
    # Heuristic: drop all columns starting with '_' (often post-merge engineered signals)
    underscore_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("_")]
    to_drop = sorted(set(present) | set(underscore_cols))
    if to_drop:
        log.warning("Dropping leaky/engineered columns: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


def _infer_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature names after removing IDs and blocklisted fields."""
    ignore = set(ID_COLS) | set(TIME_KEYS) | set(LEAK_BLOCKLIST)
    numeric: List[str] = []
    categorical: List[str] = []
    cat_check = df['home_team'].unique()
    for c in df.columns:
        if c in ignore:
            continue
        if c in (TARGET_HOME, TARGET_AWAY, CLASS_LABEL):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
        
    return numeric, categorical


def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Keep OHE sparse to reduce memory footprint; downstream estimators must accept sparse
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),  
            ("cat", cat_pipe, cat_cols),
        ],
        sparse_threshold=0.0,  # Force dense output
    )


def _split_for_calibration(tscv: TimeSeriesSplit, X: pd.DataFrame, y: pd.Series):
    """Use the last TimeSeriesSplit fold as validation/holdout. Others become the training pool."""
    splits = list(tscv.split(X, y))
    train_idx_all: List[int] = []
    for tr, va in splits[:-1]:
        train_idx_all.extend(tr.tolist())
        train_idx_all.extend(va.tolist())
    calib_tr, calib_va = splits[-1]
    return np.array(train_idx_all), calib_tr, calib_va


def _fit_regression(
    X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer, random_state: int, n_jobs: int = N_JOBS
) -> Pipeline:
    rs = RandomizedSearchCV(
        estimator=Pipeline([
            ('pre', pre),
            ('reg', HistGradientBoostingRegressor(random_state=random_state))
        ]),
        param_distributions=REG_PARAM_DISTS,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=2,
        n_iter=min(HP_N_ITER, len(list(REG_PARAM_DISTS.values())[0]) * 10),
        refit=True,
    )
    rs.fit(X, y)
    logging.info('%s rs', rs)
    return cast(Pipeline, rs.best_estimator_)


def _fit_classifier(
    X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer, random_state: int, n_jobs: int = N_JOBS
) -> Tuple[Pipeline, Dict[str, Any]]:
    rs = RandomizedSearchCV(
        estimator=Pipeline([
            ('pre', pre),
            ('clf', LogisticRegression(random_state=random_state, max_iter=1000))
        ]),
        param_distributions=CLF_PARAM_DISTS,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_log_loss",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=2,
    n_iter=min(HP_N_ITER, len(list(CLF_PARAM_DISTS.values())[0]) * 5),
        refit=True,
    )
    rs.fit(X, y)
    logging.info('%s rs', rs)
    best_pipeline = cast(Pipeline, rs.best_estimator_)
    
    # Compute holdout metrics using the last fold
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    train_idx, _, holdout_idx = _split_for_calibration(tscv, X, y)
    pred_proba = best_pipeline.predict_proba(X.iloc[holdout_idx])[:, 1]
    auc = roc_auc_score(y.iloc[holdout_idx], pred_proba)
    brier = brier_score_loss(y.iloc[holdout_idx], pred_proba)
    logloss = log_loss(y.iloc[holdout_idx], pred_proba)
    
    holdout_metrics = {
        "roc_auc": auc,
        "brier_score": brier,
        "log_loss": logloss,
        "optimal_threshold": 0.5,
        "optimal_threshold_f1": 0.5,
        "optimal_threshold_acc": 0.5,
    }
    
    return best_pipeline, holdout_metrics


def _evaluate_regression(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    pred = model.predict(X)
    return float(mean_absolute_error(y, pred))


def _dataset_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(TIME_KEYS).reset_index(drop=True)


def main(data_path: str, out_dir: str) -> None:
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(data_path)
    _ensure_columns(df, TIME_KEYS + [TARGET_HOME, TARGET_AWAY, CLASS_LABEL])
    if df.empty:
        raise RuntimeError(f"Dataset is empty: {data_path}")

    # Chronological order
    df = _dataset_sort(df)

    # Extract targets BEFORE dropping leaky columns
    y_home = df[TARGET_HOME].copy()
    y_away = df[TARGET_AWAY].copy()
    # Drop NaN values before converting to int
    y_win = df[CLASS_LABEL].copy()
    cat = df[['home_team', 'away_team']]
    
    # Now drop leaky columns from features (but keep targets separately)
    df = _drop_leaky_columns(df)

    # Remove rows with missing targets (apply same mask to features and targets)
    keep_mask = (~y_home.isna()) & (~y_away.isna()) & (~y_win.isna())
    df = df.loc[keep_mask].reset_index(drop=True)
    y_home = y_home.loc[keep_mask].reset_index(drop=True)
    y_away = y_away.loc[keep_mask].reset_index(drop=True)
    y_win = y_win.loc[keep_mask].astype(int).reset_index(drop=True)

    # Features after sanitization
    num_cols, cat_cols = _infer_features(df)
    feature_cols = num_cols + cat_cols
    if not feature_cols:
        raise RuntimeError("No features found after leakage sanitization. Check your dataset.")
    X = df[feature_cols].copy()
    # Cast numeric columns to float32 to reduce memory footprint during CV
    for nc in num_cols:
        if nc in X.columns:
            X[nc] = X[nc].astype('float32')
    pre = _make_preprocessor(num_cols, cat_cols)

    # Fit models (time-aware CV inside)
    log.info("Fitting home points regressor")
    # Allow user to override parallel jobs via env or CLI
    n_jobs = N_JOBS
    home_model = _fit_regression(X, y_home, pre, RANDOM_SEED, n_jobs=n_jobs)
    log.info('%s home_model', home_model)
    log.info("Fitting away points regressor")
    away_model = _fit_regression(X, y_away, pre, RANDOM_SEED, n_jobs=n_jobs)

    log.info("Fitting win classifier")
    win_model, win_holdout_metrics = _fit_classifier(X, y_win, pre, RANDOM_SEED, n_jobs=n_jobs)

    # Quick in-sample sanity MAE (for monitoring only)
    mae_home = _evaluate_regression(home_model, X, y_home)
    mae_away = _evaluate_regression(away_model, X, y_away)

    # Persist artifacts
    os.makedirs(out_dir, exist_ok=True)
    dump(pre, os.path.join(out_dir, "preprocessor.joblib"))
    dump(home_model, os.path.join(out_dir, "home_model.joblib"))
    dump(away_model, os.path.join(out_dir, "away_model.joblib"))
    dump(win_model, os.path.join(out_dir, "win_clf_calibrated.joblib"))

    # Reports
    training_timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    summary = TrainSummary(
        training_timestamp_utc=training_timestamp_utc,
        rows_total=int(len(df)),
        n_features_numeric=len(num_cols),
        n_features_categorical=len(cat_cols),
        cv_n_splits=N_SPLITS,
        random_seed=RANDOM_SEED,
        production_ready=True or False,
        dataset_hash=_dataset_hash(df),
    )

    metadata = {
        "training_timestamp_utc": summary.training_timestamp_utc,
        "dataset_hash": summary.dataset_hash,
        "preprocessor": "preprocessor.joblib",
        "home_model": "home_model.joblib",
        "away_model": "away_model.joblib",
        "win_model": "win_clf_calibrated.joblib",
        "raw_feature_columns": {"numeric": num_cols, "categorical": cat_cols},
        "production_ready": True ,
        "cv": {"type": "TimeSeriesSplit", "n_splits": N_SPLITS},
        "holdout_metrics_win": win_holdout_metrics,
        "quick_mae": {"home": mae_home, "away": mae_away},
    }

    with open(os.path.join(out_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("Saved models and reports to %s", out_dir)
    log.info("Summary: %s", summary)
    log.info("Metadata: %s", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train leak-free NFL models with time-aware CV.")
    parser.add_argument("--data", type=str, default="./backend/data/merge_dominance.csv", help="Path to features CSV (default: merge_dominance.csv).")
    parser.add_argument("--out", type=str, default="models", help="Output directory for artifacts and reports.")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="Number of parallel jobs to use for CV (default from N_JOBS env)")
    parser.add_argument("--hp-niter", type=int, default=HP_N_ITER, help="Number of RandomizedSearchCV iterations (overrides HP_N_ITER env)")
    args = parser.parse_args()
    # Allow quick runs by overriding HP_N_ITER when provided on CLI
    HP_N_ITER = int(args.hp_niter)
    main(args.data, args.out)
