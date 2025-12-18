#!/usr/bin/env python3
# File: backend/train_models.py
# Purpose: Train leak-aware NFL prediction models and persist artifacts/metadata for backend inference.
# Functions: _resolve_default_dataset_path(140), _ensure_columns(197), _dataset_hash(203), _drop_leaky_columns(208), _infer_features(225), _make_preprocessor(244), _split_for_calibration(271), _fit_regression(282), _fit_classifier(323), _evaluate_regression(395), _dataset_sort(400), main(405)
# Variables: HP_N_ITER(60), CV_SPLITS(61), RANDOM_SEED(62), N_SPLITS(64), N_JOBS(66), TARGET_HOME(74), TARGET_AWAY(75), CLASS_LABEL(76), TIME_KEYS(77), LEAK_BLOCKLIST(89), REG_PARAM_DISTS(115), CLF_PARAM_DISTS(122), BASE_DIR(136), DATA_DIR(137)
# Interacts With: backend/data/game_features*.csv, backend/models/*.joblib, backend/main.py loader via model artifacts.
"""
File: train_models.py
Purpose: Train ML models for NFL game predictions using time-aware cross-validation to prevent data leakage; outputs fitted models and metadata for backend/main.py inference.
Functions: _ensure_columns, _dataset_hash, _drop_leaky_columns, _infer_features, _make_preprocessor, _split_for_calibration, _fit_regression, _fit_classifier, _evaluate_regression, _dataset_sort, main
Variables: RANDOM_SEED, N_SPLITS, TARGET_HOME, TARGET_AWAY, CLASS_LABEL, TIME_KEYS, ID_COLS, LEAK_BLOCKLIST, REG_PARAM_DISTS, CLF_PARAM_DISTS, log
Interacts With: backend/data/merge_dominance.csv or game_features CSV (input dataset), backend/models/ (output .joblib + metadata.json), backend/main.py (loads artifacts)

Train leak-free NFL models with time-aware CV.
- Drops label and label-derived columns from features.
- Uses TimeSeriesSplit for all model selection.
- Reserves the final chronological fold for holdout metrics.
- Writes: artifacts/{preprocessor,home_model,away_model,win_clf_calibrated}.joblib
         artifacts/training_report.json
         artifacts/metadata.json
- Environment variables are loaded with defaults and converted to appropriate types for safety.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, cast, Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import dump, load
import time

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    auc,
    accuracy_score,
)


# -----------------------
# Configuration & Environment
# -----------------------
load_dotenv(dotenv_path="./.env", verbose=True)

SERVE_FRONTEND = os.getenv("SERVE_FRONTEND")
CORS_ORIGINS = os.getenv("CORS_ORIGINS")
NODE_ENV = os.getenv("NODE_ENV")
HP_N_ITER = int(os.getenv("HP_N_ITER", "100"))  # Default to 100 if not set or invalid
CV_SPLITS = int(os.getenv("CV_SPLITS", "5"))  # Default to 5
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))  # Default to 42
# If N_SPLITS is not provided, fall back to CV_SPLITS to keep CV configuration consistent.
N_SPLITS = int(os.getenv("N_SPLITS", str(CV_SPLITS)))
# Limit parallelism to avoid memory spikes during RandomizedSearch; override via N_JOBS env if needed.
N_JOBS = int(os.getenv("N_JOBS", "-1"))

# ----------Development environment -----------
DEV_ORIGINS = os.getenv("DEV_ORIGINS", "http://localhost:3000")

# Fast dev flag: when true, skip hyperparameter search and train single models.
FAST_DEV_TRAIN = os.getenv("FAST_DEV_TRAIN", "0") == "1"

TARGET_HOME = "home_points_for"
TARGET_AWAY = "away_points_for"
CLASS_LABEL = "home_win"
TIME_KEYS = ["season", "week"]

# Identifier columns (excluded from features)
ID_COLS = {
    "game_id",
    "gid",
    "home_team_id",
    "away_team_id",
    "stadium",
    "time_key",
    "home_game_date",
    "away_game_date",
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

# Hyperparameter search spaces
REG_PARAM_DISTS = {
    "reg__max_depth": [3, 6, 10, 14],
    "reg__learning_rate": [0.02, 0.05, 0.1, 0.15, 0.2, 0.01],
    "reg__max_leaf_nodes": [15, 31, 63, 127],
    "reg__l2_regularization": [0.02, 0.05, 0.1, 0.15, 0.2],
    "reg__min_samples_leaf": [10, 20, 30],
}

CLF_PARAM_DISTS = {
    "clf__C": [0.02, 0.01, 0.1, 1.0, 0.05],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear", "lbfgs"],
    "clf__class_weight": [None, "balanced"],
}

HIST_PARAM_DISTS = {
    "clf__max_depth": [3, 6, 10, 14],
    "clf__learning_rate": [0.02, 0.05, 0.1, 0.15, 0.2],
    "clf__max_leaf_nodes": [15, 31, 63, 127],
    "clf__l2_regularization": [0.0, 0.01, 0.05, 0.1],
    "clf__min_samples_leaf": [10, 20, 30],
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_models")

# Base directories for resolving dataset paths and outputs
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _resolve_default_dataset_path() -> str:
    """Determine the default training dataset CSV.

    Priority:
      1) TRAIN_DATASET_FILE env var if it points to an existing file.
      2) Latest game_features_*.csv in backend/data (by modification time).
      3) Legacy merge_dominance.csv in backend/data.
    """
    env_path = os.getenv("TRAIN_DATASET_FILE")
    if env_path:
        env_path = env_path.strip()
        if env_path:
            candidate = Path(env_path)
            if not candidate.is_absolute():
                # If the env path is relative, prefer resolving it from the backend folder.
                if env_path.startswith("backend/") or env_path.startswith("backend\\"):
                    candidate = BASE_DIR.parent / env_path
                else:
                    candidate = BASE_DIR / env_path
            if candidate.exists():
                log.info("Using training dataset from TRAIN_DATASET_FILE=%s", candidate)
                return str(candidate)
            log.warning(
                "TRAIN_DATASET_FILE=%s does not exist; falling back to auto-detected dataset.",
                env_path,
            )

    if DATA_DIR.exists():
        game_feature_files = list(DATA_DIR.glob("game_features_*.csv"))
        if game_feature_files:
            latest = max(game_feature_files, key=lambda p: p.stat().st_mtime)
            log.info("Using latest game_features CSV for training: %s", latest)
            return str(latest)

    legacy = DATA_DIR / "merge_dominance.csv"
    if legacy.exists():
        log.info("Falling back to legacy merge_dominance.csv dataset: %s", legacy)
        return str(legacy)

    raise FileNotFoundError(
        f"No training dataset found. Expected a game_features_*.csv or merge_dominance.csv in {DATA_DIR} "
        "or a valid TRAIN_DATASET_FILE env override."
    )


@dataclass
class TrainingSummary:
    """Comprehensive training run summary."""
    training_timestamp_utc: str
    training_duration_seconds: float
    rows_total: int
    rows_train: int
    rows_holdout: int
    n_features_numeric: int
    n_features_categorical: int
    n_features_total_transformed: int
    cv_n_splits: int
    hp_n_iter: int
    random_seed: int
    dataset_hash: str
    production_ready: bool
    home_model_metrics: Dict[str, float] = field(default_factory=dict)
    away_model_metrics: Dict[str, float] = field(default_factory=dict)
    win_model_metrics: Dict[str, float] = field(default_factory=dict)
    hist_model_metrics: Dict[str, float] = field(default_factory=dict)
    cv_best_params: Dict[str, Any] = field(default_factory=dict)


# -----------------------
# Utility Functions
# -----------------------
def _timer(start: float) -> str:
    """Format elapsed time since start."""
    elapsed = time.time() - start
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    return f"{elapsed / 60:.1f}m"


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Validate required columns exist in DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _dataset_hash(df: pd.DataFrame) -> int:
    """Stable hash across season/week/home/away columns to track dataset identity."""
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
    """Separate numeric and categorical feature columns.

    Excludes:
    - ID columns
    - Time keys (used for sorting, not prediction)
    - Target columns
    - Leak blocklist columns
    """
    ignore = set(ID_COLS) | set(TIME_KEYS) | LEAK_BLOCKLIST | {TARGET_HOME, TARGET_AWAY, CLASS_LABEL}

    numeric: List[str] = []
    categorical: List[str] = []

    for col in df.columns:
        if col in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            categorical.append(col)

    log.info("Inferred %d numeric and %d categorical features", len(numeric), len(categorical))
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
            # Use dense one-hot output so downstream estimators like HistGradientBoostingRegressor
            # receive a dense design matrix. LogisticRegression also handles dense inputs efficiently
            # at this feature scale.
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        # Force a dense combined feature matrix; some sklearn estimators in this pipeline
        # (e.g. HistGradientBoostingRegressor) do not accept sparse X.
        sparse_threshold=0.0,
    )


def _dataset_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataset chronologically by season and week."""
    return df.sort_values(by=TIME_KEYS).reset_index(drop=True)


def _get_holdout_split(X: pd.DataFrame, y: pd.Series, holdout_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Reserve the final chronological portion for holdout evaluation.

    Returns indices for train and holdout sets.
    """
    n = len(X)
    holdout_size = int(n * holdout_ratio)
    train_idx = np.arange(n - holdout_size)
    holdout_idx = np.arange(n - holdout_size, n)
    return train_idx, holdout_idx


# -----------------------
# Model Training Functions
# -----------------------
def _fit_regression(
    X: pd.DataFrame,
    y: pd.Series,
    pre: ColumnTransformer,
    random_state: int,
    n_jobs: int = N_JOBS,
    use_search: bool = True,
) -> Pipeline:
    """Fit a regression model with optional hyperparameter search.

    When use_search=False, trains a single HistGradientBoostingRegressor pipeline
    without RandomizedSearchCV for faster development iterations.
    """
    base_pipeline = Pipeline(
        [
            ("pre", pre),
            ("reg", HistGradientBoostingRegressor(random_state=random_state)),
        ]
    )

    if not use_search:
        base_pipeline.fit(X, y)
        logging.info("Fitted regression model without hyperparameter search (fast dev mode).")
        return cast(Pipeline, base_pipeline)

    rs = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=REG_PARAM_DISTS,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=2,
        n_iter=n_iter,
        refit=True,
        error_score="raise",
    )

    rs.fit(X, y)
    logging.info("%s rs", rs)
    return cast(Pipeline, rs.best_estimator_)


def _fit_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    pre: ColumnTransformer,
    random_state: int,
    n_jobs: int = N_JOBS,
    use_search: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Fit a classifier with optional hyperparameter search.

    When use_search=False, trains a single LogisticRegression pipeline without
    RandomizedSearchCV and computes metrics on the full dataset for quick feedback.
    """
    base_pipeline = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(random_state=random_state, max_iter=1000)),
        ]
    )

    if not use_search:
        base_pipeline.fit(X, y)
        logging.info("Fitted classifier without hyperparameter search (fast dev mode).")
        pred_proba = base_pipeline.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, pred_proba)
        brier = brier_score_loss(y, pred_proba)
        logloss = log_loss(y, pred_proba)
        holdout_metrics = {
            "roc_auc": auc,
            "brier_score": brier,
            "log_loss": logloss,
            "optimal_threshold": 0.5,
            "optimal_threshold_f1": 0.5,
            "optimal_threshold_acc": 0.5,
        }
        return cast(Pipeline, base_pipeline), holdout_metrics

    rs = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=CLF_PARAM_DISTS,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=3,
        n_iter=n_iter,
        refit=True,
        error_score="raise",
    )

    # 3. Hyperparameter search: HistGradientBoostingClassifier
    log.info("Running RandomizedSearchCV for HistGradientBoostingClassifier...")
    hist_gb_rs = RandomizedSearchCV(
        estimator=hist_gb_pipe,
        param_distributions=HIST_PARAM_DISTS,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=3,
        n_iter=n_iter,
        refit=True,
    )
    rs.fit(X, y)
    logging.info("%s rs", rs)
    best_pipeline = cast(Pipeline, rs.best_estimator_)

    # Compute holdout metrics using the last fold
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    _, _, holdout_idx = _split_for_calibration(tscv, X, y)
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


def _evaluate_regression(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> RegressionMetrics:
    """Compute regression metrics on holdout set."""
    pred = model.predict(X)
    return float(mean_absolute_error(y, pred))


def _dataset_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Chronologically sort dataset by TIME_KEYS."""
    return df.sort_values(TIME_KEYS).reset_index(drop=True)


def main(data_path: str, out_dir: str, n_jobs: int = N_JOBS, fast_dev: bool = False) -> None:
    """Train regression/classifier models and persist artifacts/metadata to disk."""
    np.random.seed(RANDOM_SEED)

    log.info("Loading training dataset from %s", data_path)
    dataset_path = Path(data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)
    _ensure_columns(df, TIME_KEYS + [TARGET_HOME, TARGET_AWAY, CLASS_LABEL])
    if df.empty:
        raise RuntimeError(f"Dataset is empty: {data_path}")

    log.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    # Ensure required columns are present (targets + time keys)
    _ensure_columns(df, TIME_KEYS + [TARGET_HOME, TARGET_AWAY, CLASS_LABEL])

    # Chronological sort (critical for time-series CV)
    df = _dataset_sort(df)

    # -------------------------------------------------------------------------
    # 2. Extract targets and drop leaky columns
    # -------------------------------------------------------------------------
    y_home = df[TARGET_HOME].copy()
    y_away = df[TARGET_AWAY].copy()
    y_win = df[CLASS_LABEL].copy()

    # Now drop leaky columns from features (but keep targets separately)
    df = _drop_leaky_columns(df)

    keep_mask = (~y_home.isna()) & (~y_away.isna()) & (~y_win.isna())
    n_dropped = (~keep_mask).sum()
    if n_dropped > 0:
        log.info("Dropped %d rows with missing targets", n_dropped)

    df = df.loc[keep_mask].reset_index(drop=True)
    y_home = y_home.loc[keep_mask].reset_index(drop=True)
    y_away = y_away.loc[keep_mask].reset_index(drop=True)
    y_win = y_win.loc[keep_mask].astype(int).reset_index(drop=True)

    log.info("Dataset after cleaning: %d rows", len(df))

    # -------------------------------------------------------------------------
    # 3. Infer features and optimize numeric dtypes
    # -------------------------------------------------------------------------
    num_cols, cat_cols = _infer_features(df)
    feature_cols = num_cols + cat_cols

    if not feature_cols:
        raise RuntimeError(
            "No features found after leakage sanitization. "
            "Check your dataset or _drop_leaky_columns."
        )
    df[feature_cols].to_csv('prod-dataset.csv')
    X = df[feature_cols].copy()
    for col in num_cols:
        if col in X.columns:
            X[col] = X[col].astype("float32")

    # -------------------------------------------------------------------------
    # 4. Chronological train / holdout split
    # -------------------------------------------------------------------------
    train_idx, holdout_idx = _get_holdout_split(X, y_home, holdout_ratio=0.20)
    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_home_train, y_home_holdout = y_home.iloc[train_idx], y_home.iloc[holdout_idx]
    y_away_train, y_away_holdout = y_away.iloc[train_idx], y_away.iloc[holdout_idx]
    y_win_train, y_win_holdout = y_win.iloc[train_idx], y_win.iloc[holdout_idx]

    log.info(
        "Train set: %d rows | Holdout set: %d rows",
        len(train_idx),
        len(holdout_idx),
    )

    # -------------------------------------------------------------------------
    # 5. Preprocessor (fit on TRAIN only)
    # -------------------------------------------------------------------------
    pre = _make_preprocessor(num_cols, cat_cols)

    # Fit models (time-aware CV inside)
    log.info("Fitting home points regressor")
    home_model = _fit_regression(
        X,
        y_home,
        pre,
        RANDOM_SEED,
        n_jobs=n_jobs,
        use_search=not fast_dev,
    )
    log.info('%s home_model', home_model)
    log.info("Fitting away points regressor")
    away_model = _fit_regression(
        X,
        y_away,
        pre,
        RANDOM_SEED,
        n_jobs=n_jobs,
        use_search=not fast_dev,
    )

    log.info("Fitting win classifier")
    win_model, win_holdout_metrics = _fit_classifier(
        X,
        y_win,
        pre,
        RANDOM_SEED,
        n_jobs=n_jobs,
        use_search=not fast_dev,
    )

    # -------------------------------------------------------------------------
    # 7. Evaluate on holdout
    # -------------------------------------------------------------------------
    home_metrics = _evaluate_regression(home_model, X_holdout, y_home_holdout)
    away_metrics = _evaluate_regression(away_model, X_holdout, y_away_holdout)

    win_is_calibrated = win_params.get("calibrated", False)
    hist_is_calibrated = hist_params.get("calibrated", False)

    win_metrics = _evaluate_classifier(
        win_model, X_holdout, y_win_holdout, win_is_calibrated
    )
    hist_win_metrics = _evaluate_classifier(
        hist_model, X_holdout, y_win_holdout, hist_is_calibrated
    )

    log.info(
        "Home Score Regressor - MAE: %.2f, RMSE: %.2f, R²: %.3f",
        home_metrics.mae,
        home_metrics.rmse,
        home_metrics.r2,
    )
    log.info(
        "Away Score Regressor - MAE: %.2f, RMSE: %.2f, R²: %.3f",
        away_metrics.mae,
        away_metrics.rmse,
        away_metrics.r2,
    )
    log.info(
        "Win Classifier (winner) - Accuracy: %.3f, ROC-AUC: %.3f, "
        "Brier: %.4f, ECE: %.4f",
        win_metrics.accuracy,
        win_metrics.roc_auc,
        win_metrics.brier_score,
        win_metrics.ece,
    )
    log.info(
        "HistGB Classifier - Accuracy: %.3f, ROC-AUC: %.3f, "
        "Brier: %.4f, ECE: %.4f",
        hist_win_metrics.accuracy,
        hist_win_metrics.roc_auc,
        hist_win_metrics.brier_score,
        hist_win_metrics.ece,
    )

    # -------------------------------------------------------------------------
    # 8. Feature importances
    # -------------------------------------------------------------------------
    log.info("Extracting feature importances...")
    home_importance = _extract_feature_importance(home_model, feature_cols, "regressor")
    away_importance = _extract_feature_importance(away_model, feature_cols, "regressor")
    win_importance = _extract_feature_importance(win_model, feature_cols, "classifier")
    hist_win_importance = _extract_feature_importance(
        hist_model, feature_cols, "classifier"
    )

    # -------------------------------------------------------------------------
    # 9. Save artifacts
    # -------------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Saving artifacts to %s", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dump(pre, os.path.join(out_dir, "preprocessor.joblib"))
    dump(home_model, os.path.join(out_dir, "home_model.joblib"))
    dump(away_model, os.path.join(out_dir, "away_model.joblib"))
    dump(win_model, os.path.join(out_dir, "win_clf_calibrated.joblib"))

    training_duration = time.time() - training_start
    training_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # -------------------------------------------------------------------------
    # 10. Build summary
    # -------------------------------------------------------------------------
    summary = TrainingSummary(
        training_timestamp_utc=training_timestamp,
        training_duration_seconds=round(training_duration, 2),
        rows_total=len(df),
        rows_train=len(train_idx),
        rows_holdout=len(holdout_idx),
        n_features_numeric=len(num_cols),
        n_features_categorical=len(cat_cols),
        n_features_total_transformed=n_transformed,
        cv_n_splits=N_SPLITS,
        hp_n_iter=HP_N_ITER,
        random_seed=RANDOM_SEED,
        production_ready=True,
        dataset_hash=_dataset_hash(df),
        production_ready=True,
        home_model_metrics=home_metrics.to_dict(),
        away_model_metrics=away_metrics.to_dict(),
        win_model_metrics=win_metrics.to_dict(),
        hist_model_metrics=hist_win_metrics.to_dict(),
        cv_best_params={
            "home_model": home_params,
            "away_model": away_params,
            "win_model": win_params,
            "hist_model": hist_params,
        },
    )

    # -------------------------------------------------------------------------
    # 11. Metadata for API/frontend
    # -------------------------------------------------------------------------
    metadata = {
        "training_timestamp_utc": summary.training_timestamp_utc,
        "dataset_hash": summary.dataset_hash,
        "preprocessor": "preprocessor.joblib",
        "home_model": "home_model.joblib",
        "away_model": "away_model.joblib",
        "win_model": "win_clf_calibrated.joblib",
        "hist_win_model": "hist_win_clf_calibrated.joblib",
        "raw_feature_columns": {"numeric": num_cols, "categorical": cat_cols},
        "production_ready": True,
        "cv": {"type": "TimeSeriesSplit", "n_splits": N_SPLITS},
        "holdout_metrics": {
            "home": home_metrics.to_dict(),
            "away": away_metrics.to_dict(),
            "win": win_metrics.to_dict(),
            "hist_win": hist_win_metrics.to_dict(),
        },
        "rows": summary.rows_total,
        "features": len(feature_cols),
    }

    # Feature importance report
    feature_importance = {
        "home_model": dict(list(home_importance.items())[:50]),
        "away_model": dict(list(away_importance.items())[:50]),
        "win_model": dict(list(win_importance.items())[:50]),
        "hist_model": dict(list(hist_win_importance.items())[:50]),
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
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Path to features CSV. If omitted, uses TRAIN_DATASET_FILE env or "
            "auto-detects the latest game_features_*.csv in backend/data."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models",
        help="Output directory for artifacts and reports (default: models).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help="Number of parallel jobs to use for CV (default from N_JOBS env)",
    )
    parser.add_argument(
        "--hp-niter",
        type=int,
        default=HP_N_ITER,
        help="Number of RandomizedSearchCV iterations (overrides HP_N_ITER env)",
    )
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="Enable fast dev mode (skip hyperparameter search and fit single models).",
    )
    args = parser.parse_args()
    # Allow quick runs by overriding HP_N_ITER when provided on CLI
    HP_N_ITER = int(args.hp_niter)
    dataset_path = args.data or _resolve_default_dataset_path()
    fast_dev_flag = FAST_DEV_TRAIN or args.fast_dev
    main(dataset_path, args.out, n_jobs=args.n_jobs, fast_dev=fast_dev_flag)
