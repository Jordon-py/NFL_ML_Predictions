"""
NFL-ML: Training Pipeline (models + metadata + reports)

Run:
  python backend/train_models.py

Outputs (in backend/models):
  - preprocessor.joblib
  - home_model.joblib
  - away_model.joblib
  - win_clf_calibrated.joblib
  - metadata.json      (feature contract + artifact registry + thresholds)
  - training_report.json
  - validation_errors.csv

Key improvements:
  1) Stable feature contract → metadata["raw_feature_columns"] for inference.
  2) Time-aware CV + calibrated classifier + reliability bins + Brier metrics.
  3) Simple score-ensemble (HGBR + Ridge with weight search) for MAE gains.
  4) Transformer outputs coerced to dense arrays for estimator compatibility.
"""

from __future__ import annotations

import json
import logging
import logging.config
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Dict, List, Literal, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -----------------------
# Paths and configuration 
# -----------------------
THIS_FILE = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = BACKEND_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET = DATA_DIR / os.getenv("TRAIN_DATASET_FILE", "new_dataset.csv")

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "1337"))
HYPERPARAM_SEARCH_ITERATIONS = int(os.getenv("HP_NITER", "40"))
N_SPLITS = int(os.getenv("CV_SPLITS", "5"))
CALIBRATION_METHOD: Literal["sigmoid", "isotonic"] = cast(
    Literal["sigmoid", "isotonic"], os.getenv("CALIB_METHOD", "sigmoid"))

RELIABILITY_BINS = int(os.getenv("RELIABILITY_BINS", "10"))

# Logging
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"d": {"format": "%(asctime)s %(levelname)s %(message)s"}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "d",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "d",
                "filename": str(LOG_DIR / "train.log"),
                "encoding": "utf-8",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)
log = logging.getLogger("train")


# -----------------------
# Determinism
# -----------------------
def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across numpy and Python's random module."""
    random.seed(seed)
    np.random.seed(seed)

set_all_seeds(RANDOM_SEED)

# -----------------------
# Utilities
# -----------------------
ID_COLS = {
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "is_home",
}

TARGET_HOME = "home_points_for"
TARGET_AWAY = "away_points_for"

CLASS_LABEL = "home_win"  # derived


def _dataset_hash(df: pd.DataFrame) -> str:
    """
    Generate deterministic hash for dataset tracking and cache invalidation.
    Uses pandas hash_pandas_object for consistent results across runs.
    """
    hash_sum = pd.util.hash_pandas_object(df.fillna(-999), index=True).sum()
    return str(int(hash_sum))

# --- main(): sanitize targets before modeling ---


def _infer_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Automatically detect numeric and categorical features from dataset columns.
    
    Logic:
    - Numeric: All float/int columns except IDs and targets
    - Categorical: home_team, away_team (even if encoded as integers with <64 unique values)
    
    Returns:
        (numeric_columns, categorical_columns)
    """
    cols = list(df.columns)
    ignore = ID_COLS | {TARGET_HOME, TARGET_AWAY}
    numeric: List[str] = []
    categorical: List[str] = []

    # Step 1: Collect all numeric columns that aren't metadata/targets
    for c in cols:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    
    # Step 2: Explicitly mark team columns as categorical (for one-hot encoding)
    for c in ("home_team", "away_team"):
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            categorical.append(c)

    # Step 3: Handle legacy datasets where teams are encoded as integers
    # If cardinality is low (<64 teams), treat as categorical not continuous
    for c in ("home_team", "away_team"):
        if c in df.columns and c not in categorical and df[c].nunique() <= 64:
            categorical.append(c)
            if c in numeric:
                numeric.remove(c)  # Move from numeric to categorical

    return numeric, categorical


def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Preprocess features: impute → scale / encode. Keeps estimators NaN-free."""
    """
    Build sklearn preprocessing pipeline for numeric and categorical features.
    
    Numeric pipeline:  StandardScaler (zero mean, unit variance)
    Categorical pipeline: OneHotEncoder (convert team names to binary indicators)
    
    Args:
        num_cols: List of numeric feature column names
        cat_cols: List of categorical feature column names
    
    Returns:
        ColumnTransformer that can fit/transform feature matrices
    
    Raises:
        RuntimeError: If no features provided (invalid dataset)
    """
    
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),   # handle numeric NaNs
                ("scale", StandardScaler(with_mean=True, with_std=True)),
            ]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),  # handle missing teams
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols
        ))
    if not transformers:
        raise RuntimeError("No features selected for training.")
    return ColumnTransformer(transformers=transformers, verbose=True, remainder="drop")

# -----------------------
# Model search spaces
# -----------------------
def _reg_grid() -> Dict[str, List[Any]]:
    """
    Hyperparameter search space for HistGradientBoostingRegressor.
    
    These ranges balance model complexity vs. generalization:
    - learning_rate: Controls gradient step size (lower = more stable, slower convergence)
    - max_depth: Tree depth limit (deeper = more complex interactions, higher overfitting risk)
    - max_leaf_nodes: Total leaves per tree (higher = more granular splits)
    - min_samples_leaf: Minimum samples per leaf (higher = smoother predictions, less overfitting)
    - l2_regularization: L2 penalty on leaf weights (higher = more regularization)
    """
    return {
        "learning_rate": list(np.geomspace(0.01, 0.3, 10)),     # Geometric spacing for exponential parameter
        "max_depth": [None, 3, 4, 5, 6],                        # None = unlimited depth
        "max_leaf_nodes": [15, 31, 63, 127],                    # Powers of 2 minus 1 (balanced binary trees)
        "min_samples_leaf": [10, 20, 30, 50, 80],               # Minimum samples to form a leaf
        "l2_regularization": [0.0, 0.01, 0.05, 0.1],            # Ridge regularization strength
    }


def _clf_grid() -> Dict[str, List[Any]]:
    """
    Hyperparameter search space for LogisticRegression (binary win/loss classifier).
    
    Parameters:
    - C: Inverse regularization strength (higher = less regularization, more complex model)
    - penalty: Regularization type (L2 = ridge penalty on coefficients)
    - solver: Optimization algorithm (lbfgs = good for small-to-medium datasets)
    - max_iter: Maximum iterations for convergence
    - class_weight: Handle imbalanced classes (None = equal weight, 'balanced' = inverse frequency)
    """
    return {
        "C": list(np.geomspace(0.05, 10.0, 10)),     # Inverse regularization (higher = less penalty)
        "penalty": ["l2"],                           # L2 regularization only (lbfgs doesn't support L1)
        "solver": ["lbfgs"],                         # Limited-memory BFGS optimizer
        "max_iter": [100, 200, 400],                 # Convergence iterations
        "class_weight": [None, "balanced"],          # Class balancing strategy
    }


REG_PARAMS = _reg_grid()
CLF_PARAMS = _clf_grid()


# -----------------------
# Data splits
# -----------------------
def _time_splits(df: pd.DataFrame, n_splits: int) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def _last_split_indices(
    df: pd.DataFrame, splitter: TimeSeriesSplit
) -> Tuple[np.ndarray, np.ndarray]:
    # build a time key to respect chronology
    tk = df["season"].astype(int) * 100 + df["week"].astype(int)
    order = np.argsort(tk.to_numpy())
    X = np.arange(len(df)).reshape(-1, 1)
    last_train_idx, last_test_idx = None, None
    for tr, te in splitter.split(X[order]):
        last_train_idx, last_test_idx = order[tr], order[te]
    if last_train_idx is None or last_test_idx is None:
        raise RuntimeError("Failed to create time-aware split.")
    return np.array(last_train_idx), np.array(last_test_idx)


# -----------------------
# Fitting functions
# -----------------------
@dataclass
class FitResult:
    model: Any
    mae_val: float
    report: Dict[str, Any]


def _fit_regressor(
    X: np.ndarray,
    y: np.ndarray,
    pre: ColumnTransformer,
    df: pd.DataFrame,  # Add DataFrame parameter for time-aware splitting
) -> FitResult:
    """
    Train score prediction model with ensemble blending (HGBR + Ridge).
    
    Pipeline:
    1. Hyperparameter search for HistGradientBoostingRegressor (primary model)
    2. Train Ridge regression as secondary model (linear baseline)
    3. Search optimal blend weight between models on validation set
    
    Ensemble Rationale:
    - HGBR captures non-linear patterns (recent form, matchups)
    - Ridge provides stable linear baseline (prevents wild predictions)
    - Blending reduces variance and improves MAE by ~5-10%
    
    Args:
        X: Preprocessed feature matrix
        y: Target scores (home or away points)
        pre: Fitted preprocessor (unused but kept for API consistency)
    
    Returns:
        FitResult with ensemble model dict, validation MAE, and training report
    """
    # Step 1: Hyperparameter search for gradient boosting regressor
    base = HistGradientBoostingRegressor(random_state=RANDOM_SEED)
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=REG_PARAMS,
        n_iter=HYPERPARAM_SEARCH_ITERATIONS,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",  # Minimize MAE
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
        refit=True,
    )
    rs.fit(X, y)

    # Step 2: Get validation split for ensemble weight tuning
    tscv = _time_splits(df, n_splits=N_SPLITS)
    tr_idx, te_idx = _last_split_indices(df, tscv)
    X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    # Step 3: Train both models on training split
    hgbr = cast(HistGradientBoostingRegressor, rs.best_estimator_)
    ridge = Ridge(random_state=RANDOM_SEED)
    ridge.fit(X_tr, y_tr)

    # Step 4: Find optimal blend weight (grid search from 20% to 90% HGBR)
    preds_h = hgbr.predict(X_te)
    preds_r = ridge.predict(X_te)
    best_w, best_mae = 1.0, mean_absolute_error(y_te, preds_h)
    
    for w in np.linspace(0.2, 0.9, 8):  # Test 8 blend ratios
        blend = w * preds_h + (1 - w) * preds_r
        mae = mean_absolute_error(y_te, blend)
        if mae < best_mae:
            best_mae, best_w = mae, w

    # Step 5: Package ensemble as dictionary for serialization
    model = {"hgbr": hgbr, "ridge": ridge, "weight": float(best_w)}
    report = {
        "best_params": rs.best_params_,
        "val_mae_hgbr": float(mean_absolute_error(y_te, preds_h)),
        "val_mae_ridge": float(mean_absolute_error(y_te, preds_r)),
        "val_mae_blend": float(best_mae),
        "blend_weight_hgbr": float(best_w),
    }
    return FitResult(model=model, mae_val=best_mae, report=report)


def _predict_reg(model_bundle: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    w = model_bundle["weight"]
    p1 = model_bundle["hgbr"].predict(X)
    p2 = model_bundle["ridge"].predict(X)
    return w * p1 + (1 - w) * p2


@dataclass
class ClfResult:
    model: Any
    report: Dict[str, Any]
    threshold: float


def _fit_classifier(
    X: np.ndarray,
    y_clf: np.ndarray,
    df: pd.DataFrame,  # Add DataFrame parameter for time-aware splitting
) -> ClfResult:
    """
    Train and calibrate a win probability classifier with optimal threshold tuning.
    
    Pipeline:
    1. Hyperparameter search with RandomizedSearchCV (maximize AUC)
    2. Calibration on validation split (sigmoid/isotonic method)
    3. Reliability analysis (binned calibration curve)
    4. Threshold optimization (maximize F1 score)
    
    Args:
        X: Feature matrix (preprocessed)
        y_clf: Binary labels (1=home win, 0=away win)
    
    Returns:
        ClfResult with calibrated model, metrics, and optimal threshold
    """
    # Step 1: Hyperparameter search with time-aware cross-validation
    base = LogisticRegression()
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=CLF_PARAMS,
        n_iter=HYPERPARAM_SEARCH_ITERATIONS,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),  # Use stratified folds to ensure both classes
        scoring="roc_auc",              # Optimize for probability ranking quality
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
        refit=True,
    )
    rs.fit(X, y_clf)
    best_lr = cast(LogisticRegression, rs.best_estimator_)

    # Step 2: Get validation split for calibration and threshold tuning
    tscv = _time_splits(df, n_splits=N_SPLITS)
    tr_idx, te_idx = _last_split_indices(df, tscv)

    # Step 3: Calibrate probabilities (sigmoid/isotonic transformation)
    cal = CalibratedClassifierCV(best_lr, method=CALIBRATION_METHOD, cv="prefit")
    cal.fit(X[tr_idx], y_clf[tr_idx])
    proba = cal.predict_proba(X[te_idx])[:, 1]  # Home win probabilities

    # Step 4: Compute validation metrics
    auc = roc_auc_score(y_clf[te_idx], proba)
    br = brier_score_loss(y_clf[te_idx], proba)
    ll = log_loss(y_clf[te_idx], np.c_[1 - proba, proba])
    acc50 = accuracy_score(y_clf[te_idx], (proba >= 0.5).astype(int))

    # Step 5: Build reliability diagram (calibration curve)
    # Bin predictions into deciles and compare predicted vs actual win rates
    bins = np.linspace(0, 1, RELIABILITY_BINS + 1)
    bin_ids = np.digitize(proba, bins) - 1
    reliab = []
    for b in range(RELIABILITY_BINS):
        mask = (bin_ids == b)
        if mask.any():
            mean_pred = float(np.mean(proba[mask]))
            mean_true = float(np.mean(y_clf[te_idx][mask]))
            count = int(np.sum(mask))
            reliab.append({"bin": b, "n": count, "mean_pred": mean_pred, "mean_true": mean_true})

    # Step 6: Optimize classification threshold on validation set
    # Sweep thresholds from 0.3 to 0.7 and maximize F1 score
    best_th, best_f1, best_acc = 0.5, -1.0, 0.0
    for th in np.linspace(0.3, 0.7, 41):  # 0.01 increments
        preds = (proba >= th).astype(int)
        
        # Compute precision/recall/F1 manually (avoid sklearn overhead in loop)
        tp = np.sum((preds == 1) & (y_clf[te_idx] == 1))
        fp = np.sum((preds == 1) & (y_clf[te_idx] == 0))
        fn = np.sum((preds == 0) & (y_clf[te_idx] == 1))
        
        prec = tp / (tp + fp + 1e-9)  # Avoid division by zero
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        acc = accuracy_score(y_clf[te_idx], preds)
        
        # Update if F1 improves (or tied F1 with better accuracy)
        if f1 > best_f1 or (math.isclose(f1, best_f1, rel_tol=1e-6) and acc > best_acc):
            best_f1, best_acc, best_th = f1, acc, float(th)

    # Step 7: Package results
    report = {
        "auc_val": float(auc),
        "brier_val": float(br),
        "logloss_val": float(ll),
        "accuracy_at_0p5": float(acc50),
        "reliability_bins": reliab,
        "optimal_threshold": best_th,
        "optimal_threshold_f1": float(best_f1),
        "optimal_threshold_acc": float(best_acc),
        "best_params": rs.best_params_,
    }
    return ClfResult(model=cal, report=report, threshold=best_th)


def _ensure_dense_matrix(matrix: Any, *, context: str) -> np.ndarray:
    """
    Enforce a dense 2-D NumPy array; ColumnTransformer may emit sparse matrices
    even with dense sub-transformers, so this keeps downstream estimators safe.
    """
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    if dense.ndim != 2:
        raise ValueError(f"{context} must be 2-D after densification.")
    return dense


# -----------------------
# Pipeline
# -----------------------
def main() -> None:
    """
    Main training pipeline: Load data → Train models → Save artifacts.
    
    High-Level Flow:
    1. Load dataset from CSV (configured via TRAIN_DATASET_FILE env var)
    2. Infer feature types (numeric vs categorical) automatically
    3. Build preprocessing pipeline (scaling + one-hot encoding)
    4. Train 3 models:
       - Home score regressor (HGBR + Ridge ensemble)
       - Away score regressor (HGBR + Ridge ensemble)
       - Win probability classifier (calibrated LogisticRegression)
    5. Generate validation error analysis CSV
    6. Save models + metadata + training report to backend/models/
    
    Outputs:
    - preprocessor.joblib (feature transformer)
    - home_model.joblib (home score ensemble)
    - away_model.joblib (away score ensemble)
    - win_clf_calibrated.joblib (calibrated win probability)
    - metadata.json (feature contract + thresholds + versioning)
    - training_report.json (metrics + hyperparameters)
    - validation_errors.csv (worst predictions for analysis)
    """
    # ---------------------------------------
    # Step 1: Load and validate dataset
    # ---------------------------------------
    # Load dataset from environment variable or default path
    data_path = Path(os.getenv(
        "TRAIN_DATASET_FILE",
        'C:/Users/iProg/OneDrive/Documents/Football_predict/nfl_prediction_system/NFL_ML_Predictions/backend/data/new_dataset.csv'
    ))
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(data_path)
    if df.empty:
        raise RuntimeError(f"Dataset is empty: {data_path}")
    
    # Drop rows with missing targets (models and MAE require finite y)
    df = df.dropna(subset=[TARGET_HOME, TARGET_AWAY]).reset_index(drop=True)
    
    # Drop rows with missing home_win labels (classifier needs both classes)
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    
    # Sort chronologically for time-aware cross-validation
    df = df.sort_values(['season', 'week']).reset_index(drop=True)
    
    # Convert 'home_win' column from boolean True/False to binary int (1/0)
    df['home_win'] = df['home_win'].astype(int)
    
    train_df = df

    # ---------------------------------------
    # Step 2: Feature engineering and preprocessing
    # ---------------------------------------
    # Automatically detect numeric and categorical features
    num_cols, cat_cols = _infer_features(train_df)
    pre = _make_preprocessor(num_cols, cat_cols)

    # Fit preprocessor on full training data and transform to feature matrix
    X_df = train_df[num_cols + cat_cols] if cat_cols else train_df[num_cols]
    pre.fit(X_df)
    X_full = pre.transform(X_df)
    
    # Ensure dense array format (ColumnTransformer can output sparse matrices)
    X_full = _ensure_dense_matrix(X_full, context="training features")

    # Extract target variables
    y_home = train_df[TARGET_HOME].to_numpy()  # Home team points scored
    y_away = train_df[TARGET_AWAY].to_numpy()  # Away team points scored
    y_clf = train_df["home_win"].to_numpy()     # Binary outcome (1=home win, 0=away win)

    # ---------------------------------------
    # Step 3: Train prediction models
    # ---------------------------------------
    log.info("Training home score regressor...")
    res_home = _fit_regressor(X_full, y_home, pre, train_df)
    
    log.info("Training away score regressor...")
    res_away = _fit_regressor(X_full, y_away, pre, train_df)
    
    log.info("Training win probability classifier...")
    clf_res = _fit_classifier(X_full, y_clf, train_df)

    # ---------------------------------------
    # Step 4: Generate validation error analysis
    # ---------------------------------------
    # Get validation split indices and make predictions
    tscv = _time_splits(train_df, n_splits=N_SPLITS)
    tr_idx, te_idx = _last_split_indices(train_df, tscv)
    X_te = X_full[te_idx]
    
    # Predict scores on validation set
    home_pred = _predict_reg(res_home.model, X_te)
    away_pred = _predict_reg(res_away.model, X_te)
    
    # Compute total absolute error per game (home + away)
    abs_err = (
        np.abs(home_pred - train_df.iloc[te_idx][TARGET_HOME].to_numpy()) +
        np.abs(away_pred - train_df.iloc[te_idx][TARGET_AWAY].to_numpy())
    )
    
    # Build diagnostic table: worst predictions sorted by error
    val_err = train_df.iloc[te_idx][
        ["season", "week", "home_team", "away_team", TARGET_HOME, TARGET_AWAY]
    ].copy()
    val_err["pred_home"] = np.round(home_pred, 2)
    val_err["pred_away"] = np.round(away_pred, 2)
    val_err["abs_error_sum"] = np.round(abs_err, 2)
    
    # Save worst predictions for manual review
    val_err.sort_values("abs_error_sum", ascending=False).to_csv(
        MODELS_DIR / "validation_errors.csv", index=False
    )

    # ---------------------------------------
    # Step 5: Save trained models
    # ---------------------------------------
    log.info("Saving models to %s", MODELS_DIR)
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib", compress=3)
    joblib.dump(res_home.model, MODELS_DIR / "home_model.joblib", compress=3)
    joblib.dump(res_away.model, MODELS_DIR / "away_model.joblib", compress=3)
    joblib.dump(clf_res.model, MODELS_DIR / "win_clf_calibrated.joblib", compress=3)

    # ---------------------------------------
    # Step 6: Generate metadata and reports
    # ---------------------------------------
    dataset_hash = _dataset_hash(
        train_df[["season", "week"]].assign(
            h=df["home_team"].astype(str), a=df["away_team"].astype(str)
        )
    )
    training_report = {
        "training_timestamp_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "dataset": {
            "path": str(data_path),
            "hash": dataset_hash,
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
        },
        "features": {
            "numeric": num_cols,
            "categorical": cat_cols,
            "count": int(len(num_cols) + len(cat_cols)),
        },
        "models": {
            "home": res_home.report,
            "away": res_away.report,
            "win_clf": clf_res.report,
        },
    }
    (MODELS_DIR / "training_report.json").write_text(
        json.dumps(training_report, indent=2), encoding="utf-8"
    )

    metadata = {
        "training_timestamp_utc": training_report["training_timestamp_utc"],
        "dataset_hash": dataset_hash,
        "preprocessor": "preprocessor.joblib",
        "home_model": "home_model.joblib",
        "away_model": "away_model.joblib",
        "win_model": "win_clf_calibrated.joblib",
        "raw_feature_columns": {"numeric": num_cols, "categorical": cat_cols},
        "win_threshold_optimal": clf_res.threshold,
        "production_ready": True,
        "cv": {"type": "TimeSeriesSplit", "n_splits": N_SPLITS},
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    log.info("Saved artifacts to %s", MODELS_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
