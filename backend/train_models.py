#!/usr/bin/env python3
"""
Train leak-free NFL models with time-aware CV.

- Drops label and label-derived columns from features.
- Uses TimeSeriesSplit for all model selection.
- Reserves the final chronological fold for holdout metrics.
- Writes: artifacts/{preprocessor,home_model,away_model,win_clf_calibrated}.joblib
         artifacts/training_report.json
         artifacts/metadata.json
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Tuple, Dict, cast

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
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

# -----------------------
# Configuration
# -----------------------

RANDOM_SEED = 42
N_SPLITS = 5

TARGET_HOME = "home_points_for"
TARGET_AWAY = "away_points_for"
CLASS_LABEL = "home_win"  # must be 0/1
TIME_KEYS = ["season", "week"]

ID_COLS = {
    "game_id",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "stadium",
}

# Columns that must never enter features (labels or post-game values)
LEAK_BLOCKLIST = {
    CLASS_LABEL,
    "point_diff",
    "winner",
    TARGET_HOME.strip().lower(),
    TARGET_AWAY.strip().lower(),
    "home_points_against",
    "away_points_against",
    "home_score",
    "away_score",
    "final_home_score",
    "final_away_score",
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
    present = [c for c in LEAK_BLOCKLIST if c in df.columns]
    if present:
        log.warning("Dropping leaky columns: %s", present)
        df = df.drop(columns=present)
    return df


def _infer_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature names after removing IDs and blocklisted fields."""
    ignore = set(ID_COLS) | set(TIME_KEYS) | set(LEAK_BLOCKLIST)
    numeric: List[str] = []
    categorical: List[str] = []
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
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # Dense output for HistGradientBoosting
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


def _fit_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    pre: ColumnTransformer,
    random_state: int,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Fit a calibrated classifier pipeline using RandomizedSearchCV, with holdout evaluation.
    
    Purpose: Trains a LogisticRegression classifier with hyperparameter tuning,
    applies isotonic calibration, and evaluates on a chronological holdout fold.
    
    Key Logic Flow:
    1. Split data: Use training folds for hyperparameter search, last fold as holdout.
    2. Perform hyperparameter search on training folds.
    3. Calibrate the best model on training folds.
    4. Evaluate calibrated model on holdout fold (ROC AUC, Brier score, log loss).
    5. Return the calibrated pipeline and holdout metrics.
    
    Dependencies: Requires scikit-learn (RandomizedSearchCV, CalibratedClassifierCV, metrics).
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    train_idx, calib_tr, calib_va = _split_for_calibration(tscv, X, y)
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_holdout = X.iloc[calib_va]
    y_holdout = y.iloc[calib_va]
    
    base = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            random_state=random_state,
            max_iter=1000,  # Increased for convergence on larger datasets
        )),
    ])
    
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=CLF_PARAM_DISTS,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=N_SPLITS - 1),  # Use remaining folds for CV
        scoring="neg_brier_score",
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
        refit=True,
    )
    rs.fit(X_train, y_train)
    
    best_pipeline = cast(Pipeline, rs.best_estimator_)
    
    # Calibrate on training folds
    cal = CalibratedClassifierCV(best_pipeline, method="isotonic", cv="prefit")
    cal.fit(X_train, y_train)
    
    # Evaluate on holdout
    y_prob = cal.predict_proba(X_holdout)[:, 1]
    y_pred = cal.predict(X_holdout)
    holdout_metrics = {
        "roc_auc": float(roc_auc_score(y_holdout, y_prob)),
        "brier_score": float(brier_score_loss(y_holdout, y_prob)),
        "log_loss": float(log_loss(y_holdout, y_prob)),
    }
    
    calibrated_pipeline = Pipeline([("pre", pre), ("cal", cal)])
    return calibrated_pipeline, holdout_metrics


def _fit_regression(
    X: pd.DataFrame,
    y: pd.Series,
    pre: ColumnTransformer,
    random_state: int,
) -> Pipeline:
    """
    Fit a regression pipeline using RandomizedSearchCV.
    
    Purpose: Trains a HistGradientBoostingRegressor with hyperparameter tuning
    for point prediction tasks.
    
    Key Logic Flow:
    1. Construct base pipeline (preprocessor + regressor).
    2. Perform hyperparameter search with time-series cross-validation.
    3. Return the best pipeline.
    
    Dependencies: Requires scikit-learn (RandomizedSearchCV).
    """
    base = Pipeline([
        ("pre", pre),
        ("reg", HistGradientBoostingRegressor(
            random_state=random_state,
            max_iter=200,
        )),
    ])
    
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=REG_PARAM_DISTS,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
        refit=True,
    )
    rs.fit(X, y)
    
    # Fixed: Cast rs.best_estimator_ to Pipeline to match return type hint.
    # This resolves the type checker error while preserving runtime behavior.
    return cast(Pipeline, rs.best_estimator_)


def _evaluate_regression(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    pred = model.predict(X)
    return float(mean_absolute_error(y, pred))


def _save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


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
    pre = _make_preprocessor(num_cols, cat_cols)

    # Fit models (time-aware CV inside)
    log.info("Fitting home points regressor")
    home_model = _fit_regression(X, y_home, pre, RANDOM_SEED)

    log.info("Fitting away points regressor")
    away_model = _fit_regression(X, y_away, pre, RANDOM_SEED)

    log.info("Fitting win classifier")
    win_model, win_holdout_metrics = _fit_classifier(X, y_win, pre, RANDOM_SEED)

    # Quick in-sample sanity MAE (for monitoring only)
    mae_home = _evaluate_regression(home_model, X, y_home)
    mae_away = _evaluate_regression(away_model, X, y_away)

    # Persist artifacts
    os.makedirs(out_dir, exist_ok=True)
    _save(pre, os.path.join(out_dir, "preprocessor.joblib"))
    _save(home_model, os.path.join(out_dir, "home_model.joblib"))
    _save(away_model, os.path.join(out_dir, "away_model.joblib"))
    _save(win_model, os.path.join(out_dir, "win_clf_calibrated.joblib"))

    # Reports
    training_timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    summary = TrainSummary(
        training_timestamp_utc=training_timestamp_utc,
        rows_total=int(len(df)),
        n_features_numeric=len(num_cols),
        n_features_categorical=len(cat_cols),
        cv_n_splits=N_SPLITS,
        random_seed=RANDOM_SEED,
        production_ready=False,
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
        "production_ready": False,
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
    parser.add_argument("--data", type=str, default="data/game_features.csv", help="Path to features CSV.")
    parser.add_argument("--out", type=str, default="models", help="Output directory for artifacts and reports.")
    args = parser.parse_args()
    main(args.data, args.out)
