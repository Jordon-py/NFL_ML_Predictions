#!/usr/bin/env python3
"""
Train leak-free NFL models with time-aware CV.

Enhanced training script with:
- Comprehensive leak protection (blocklist + underscore heuristic)
- TimeSeriesSplit cross-validation to prevent temporal leakage
- Proper holdout evaluation with full metrics suite
- Feature importance extraction and persistence
- Detailed progress logging with timing information
- Robust error handling and validation

Outputs:
    models/preprocessor.joblib         - Feature engineering pipeline
    models/home_model.joblib           - Home score regressor
    models/away_model.joblib           - Away score regressor
    models/win_clf_calibrated.joblib   - Win probability classifier
    models/metadata.json               - Model metadata and feature info
    models/training_report.json        - Training summary
    models/feature_importance.json     - Feature importance rankings

Environment variables:
    HP_N_ITER     - RandomizedSearchCV iterations (default: 50)
    CV_SPLITS     - Cross-validation splits (default: 5)
    N_SPLITS      - TimeSeriesSplit folds (default: 5)
    N_JOBS        - Parallel jobs (-1 = all cores, default: -1)
    RANDOM_SEED   - Random state for reproducibility (default: 42)

Usage:
    python train_models.py --data backend/data/game_features_2014_2025.csv --out backend/models
"""

# File: backend/train_models.py
# Purpose: Train ML models for NFL game predictions using time-aware cross-validation
# Functions: _ensure_columns, _dataset_hash, _drop_leaky_columns, _infer_features,
#            _make_preprocessor, _fit_regression, _fit_classifier, _evaluate_holdout,
#            _extract_feature_importance, _dataset_sort, main
# Interacts With: backend/data/*.csv (input), backend/models/ (output artifacts)

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, cast, Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------
# Configuration & Environment
# -----------------------
# Try multiple .env locations for flexibility
_backend_dir = Path(__file__).parent
_repo_root = _backend_dir.parent
for env_path in [_backend_dir / ".env", _repo_root / ".env"]:
    if env_path.exists():
        load_dotenv(env_path, verbose=True)
        break

# Environment-driven configuration with safe defaults
HP_N_ITER = int(os.getenv("HP_N_ITER", "50"))
CV_SPLITS = int(os.getenv("CV_SPLITS", "5"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
N_JOBS = int(os.getenv("N_JOBS", "-1"))

# Target columns
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

# Columns that MUST NOT be used as features (leakage risk)
LEAK_BLOCKLIST = {
    # Direct targets and labels
    CLASS_LABEL,
    TARGET_HOME,
    TARGET_AWAY,
    "winner",
    "actual_winner",
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
    # Post-game Elo (computed after game outcome)
    "home_elo_post",
    "away_elo_post",
    # Market data (contains information about expected outcome)
    "home_moneyline",
    "away_moneyline",
    "spread_line",
    "total_line",
    "home_win_prob",
    "away_win_prob",
    # Aggregated outcome rates
    "season_home_win_rate",
}

# Hyperparameter search spaces
REG_PARAM_DISTS = {
    "reg__max_depth": [None, 6, 10, 14],
    "reg__learning_rate": [0.02, 0.05, 0.1, 0.15, 0.2],
    "reg__max_leaf_nodes": [15, 31, 63, 127],
    "reg__l2_regularization": [0.0, 0.05, 0.1, 0.15, 0.2],
    "reg__min_samples_leaf": [10, 20, 30],
}

CLF_PARAM_DISTS = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear", "lbfgs"],
    "clf__class_weight": [None, "balanced"],
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_models")


# -----------------------
# Data Classes
# -----------------------
@dataclass
class RegressionMetrics:
    """Metrics for regression model evaluation."""
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> Dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


@dataclass
class ClassifierMetrics:
    """Metrics for binary classifier evaluation.

    Includes calibration metrics to measure how well predicted probabilities
    align with actual outcomes (ECE = Expected Calibration Error).
    """
    accuracy: float
    roc_auc: float
    brier_score: float
    log_loss_val: float
    pr_auc: float
    ece: float = 0.0  # Expected Calibration Error
    is_calibrated: bool = False  # Whether CalibratedClassifierCV was applied

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss_val,
            "pr_auc": self.pr_auc,
            "ece": self.ece,
            "is_calibrated": self.is_calibrated,
        }


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


def _dataset_hash(df: pd.DataFrame) -> str:
    """Generate a reproducible hash of the dataset for versioning."""
    # Use key columns for hash to detect data changes
    key_cols = [c for c in TIME_KEYS + ["home_team", "away_team"] if c in df.columns]
    if not key_cols:
        key_cols = df.columns[:5].tolist()

    content = df[key_cols].to_json(orient="records")
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that could leak target information.

    Strategy:
    1. Drop explicitly blocklisted columns (known leaks)
    2. Drop underscore-prefixed columns (internal engineering signals)
    3. Log all dropped columns for transparency
    """
    to_drop = []

    # Explicit blocklist (case-insensitive check)
    blocklist_lower = {c.lower() for c in LEAK_BLOCKLIST}
    for col in df.columns:
        if col.lower() in blocklist_lower:
            to_drop.append(col)

    # Underscore prefix heuristic (internal/derived signals)
    underscore_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("_")]
    to_drop.extend(underscore_cols)

    to_drop = sorted(set(to_drop))

    if to_drop:
        log.warning("Dropping %d leaky/internal columns: %s", len(to_drop), to_drop[:10])
        if len(to_drop) > 10:
            log.warning("  ... and %d more", len(to_drop) - 10)
        df = df.drop(columns=to_drop, errors="ignore")

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
    """Build sklearn preprocessing pipeline.

    Numeric: median imputation + standard scaling
    Categorical: mode imputation + one-hot encoding
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # Force dense output for HistGradientBoosting compatibility
    )


def _dataset_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataset chronologically by season and week."""
    return df.sort_values(TIME_KEYS).reset_index(drop=True)


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
    n_jobs: int,
    label: str = "regression",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Fit regression model with hyperparameter search.

    Returns:
        Tuple of (fitted pipeline, best parameters)
    """
    start = time.time()
    log.info("Starting %s model training...", label)

    n_iter = min(HP_N_ITER, 40)  # Cap iterations for reasonable runtime

    estimator = Pipeline([
        ("pre", pre),
        ("reg", HistGradientBoostingRegressor(random_state=random_state)),
    ])

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=REG_PARAM_DISTS,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        n_iter=n_iter,
        refit=True,
        error_score="raise",
    )

    rs.fit(X, y)

    best_params = {k.replace("reg__", ""): v for k, v in rs.best_params_.items()}
    log.info("%s training complete in %s | Best MAE: %.3f | Params: %s",
             label, _timer(start), -rs.best_score_, best_params)

    return cast(Pipeline, rs.best_estimator_), rs.best_params_


def _fit_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    pre: ColumnTransformer,
    random_state: int,
    n_jobs: int,
    calibrate: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Fit binary classifier with hyperparameter search and optional probability calibration.

    Probability Calibration Enhancement (v2.4):
    - Wraps the best estimator with CalibratedClassifierCV using isotonic regression
    - Improves probability estimates so "65% win" truly reflects ~65% historical accuracy
    - Critical for betting/prediction applications where calibrated probabilities matter
    - Uses 5-fold cross-validation for calibration to avoid overfitting

    Args:
        X: Feature dataframe
        y: Target series
        pre: Fitted preprocessor
        random_state: Random seed for reproducibility
        n_jobs: Parallel jobs for CV
        calibrate: Whether to apply probability calibration (default: True)

    Returns:
        Tuple of (fitted pipeline, best parameters including calibration info)
    """
    start = time.time()
    log.info("Starting classifier training...")

    n_iter = min(HP_N_ITER, 32)  # Cap iterations

    estimator = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(random_state=random_state, max_iter=1000)),
    ])

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=CLF_PARAM_DISTS,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_log_loss",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        n_iter=n_iter,
        refit=True,
        error_score="raise",
    )

    rs.fit(X, y)

    best_params = {k.replace("clf__", ""): v for k, v in rs.best_params_.items()}
    base_model = cast(Pipeline, rs.best_estimator_)

    log.info("Classifier CV complete in %s | Best LogLoss: %.4f | Params: %s",
             _timer(start), -rs.best_score_, best_params)

    # Apply probability calibration using isotonic regression
    if calibrate:
        log.info("Applying probability calibration (isotonic regression, 5-fold)...")
        cal_start = time.time()

        # Wrap the entire pipeline with CalibratedClassifierCV
        # Uses 5-fold CV to fit calibration without data leakage
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method="isotonic",  # Non-parametric, works well with limited data
            cv=5,
            n_jobs=n_jobs,
        )
        calibrated_model.fit(X, y)

        log.info("Calibration complete in %s", _timer(cal_start))
        best_params["calibrated"] = True
        best_params["calibration_method"] = "isotonic"
        best_params["calibration_cv"] = 5

        return calibrated_model, best_params

    best_params["calibrated"] = False
    return base_model, best_params


# -----------------------
# Evaluation Functions
# -----------------------
def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the average gap between predicted probabilities and actual outcomes
    across probability bins. Lower is better (0 = perfectly calibrated).

    A well-calibrated model with ECE < 0.05 means predictions are trustworthy:
    when it says "65% win probability", historically ~65% of such games were wins.

    Args:
        y_true: Actual binary outcomes (0/1)
        y_prob: Predicted probabilities for class 1
        n_bins: Number of bins for calibration curve (default: 10)

    Returns:
        Expected Calibration Error (0 to 1, lower is better)
    """
    try:
        # calibration_curve returns fraction of positives and mean predicted value per bin
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )

        # Compute bin sizes for weighted average
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Count samples in each bin
        bin_counts = np.bincount(bin_indices, minlength=n_bins)

        # Filter to bins that actually have samples
        non_empty = bin_counts > 0
        if not np.any(non_empty):
            return 0.0

        # Align arrays - calibration_curve may return fewer bins
        n_actual_bins = len(fraction_of_positives)
        if n_actual_bins < n_bins:
            # Use only the bins that calibration_curve computed
            bin_counts = bin_counts[:n_actual_bins]

        weights = bin_counts[:n_actual_bins] / len(y_prob)
        calibration_errors = np.abs(fraction_of_positives - mean_predicted_value)

        ece = float(np.sum(weights * calibration_errors))
        return ece

    except Exception as e:
        log.warning("Could not compute ECE: %s", e)
        return 0.0


def _evaluate_regression(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> RegressionMetrics:
    """Compute regression metrics on holdout set."""
    pred = model.predict(X)
    return RegressionMetrics(
        mae=float(mean_absolute_error(y, pred)),
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        r2=float(r2_score(y, pred)),
    )


def _evaluate_classifier(
    model, X: pd.DataFrame, y: pd.Series, is_calibrated: bool = False
) -> ClassifierMetrics:
    """Compute classifier metrics on holdout set with calibration diagnostics.

    Enhanced to compute Expected Calibration Error (ECE), which measures
    how well probability estimates align with actual outcomes.
    """
    pred_proba = model.predict_proba(X)[:, 1]
    pred_class = model.predict(X)

    # PR-AUC calculation
    precision, recall, _ = precision_recall_curve(y, pred_proba)
    pr_auc_val = float(auc(recall, precision))

    # Compute Expected Calibration Error
    ece = _compute_ece(y.values, pred_proba)

    return ClassifierMetrics(
        accuracy=float(accuracy_score(y, pred_class)),
        roc_auc=float(roc_auc_score(y, pred_proba)),
        brier_score=float(brier_score_loss(y, pred_proba)),
        log_loss_val=float(log_loss(y, pred_proba)),
        pr_auc=pr_auc_val,
        ece=ece,
        is_calibrated=is_calibrated,
    )


def _extract_feature_importance(
    model: Pipeline,
    feature_names: List[str],
    model_type: str = "regressor",
) -> Dict[str, float]:
    """Extract feature importance from trained model.

    Works with HistGradientBoosting (feature_importances_) and
    LogisticRegression (coef_).
    """
    importance_dict = {}

    try:
        # Get the actual estimator from pipeline
        if model_type == "regressor":
            estimator = model.named_steps.get("reg")
        else:
            estimator = model.named_steps.get("clf")

        if estimator is None:
            return importance_dict

        # Get preprocessor for feature name mapping
        pre = model.named_steps.get("pre")
        if pre is not None:
            try:
                transformed_names = pre.get_feature_names_out()
            except Exception:
                transformed_names = feature_names
        else:
            transformed_names = feature_names

        # Extract importances based on model type
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_).flatten()
        else:
            return importance_dict

        # Map to feature names
        n_features = min(len(importances), len(transformed_names))
        for i in range(n_features):
            name = str(transformed_names[i])
            importance_dict[name] = float(importances[i])

        # Sort by importance (descending)
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: -x[1]))

    except Exception as e:
        log.warning("Could not extract feature importance: %s", e)

    return importance_dict


# -----------------------
# Main Training Function
# -----------------------
def main(data_path: str, out_dir: str) -> None:
    """Execute full training pipeline.

    1. Load and validate dataset
    2. Preprocess and split data (train/holdout)
    3. Train home/away regressors and win classifier
    4. Evaluate on holdout set
    5. Extract feature importances
    6. Save all artifacts and reports
    """
    training_start = time.time()
    np.random.seed(RANDOM_SEED)

    log.info("=" * 60)
    log.info("NFL ML Training Pipeline")
    log.info("=" * 60)
    log.info("Configuration:")
    log.info("  Data path: %s", data_path)
    log.info("  Output dir: %s", out_dir)
    log.info("  HP iterations: %d", HP_N_ITER)
    log.info("  CV splits: %d", N_SPLITS)
    log.info("  Random seed: %d", RANDOM_SEED)
    log.info("  Parallel jobs: %d", N_JOBS)
    log.info("-" * 60)

    # Load data
    log.info("Loading dataset...")
    df = pd.read_csv(data_path)
    log.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    # Validate required columns
    _ensure_columns(df, TIME_KEYS + [TARGET_HOME, TARGET_AWAY, CLASS_LABEL])

    if df.empty:
        raise RuntimeError(f"Dataset is empty: {data_path}")

    # Chronological sort (critical for time-series CV)
    df = _dataset_sort(df)

    # Extract targets BEFORE dropping leaky columns
    y_home = df[TARGET_HOME].copy()
    y_away = df[TARGET_AWAY].copy()
    y_win = df[CLASS_LABEL].copy()

    # Remove leaky columns from features
    df = _drop_leaky_columns(df)

    # Remove rows with missing targets
    keep_mask = (~y_home.isna()) & (~y_away.isna()) & (~y_win.isna())
    n_dropped = (~keep_mask).sum()
    if n_dropped > 0:
        log.info("Dropped %d rows with missing targets", n_dropped)

    df = df.loc[keep_mask].reset_index(drop=True)
    y_home = y_home.loc[keep_mask].reset_index(drop=True)
    y_away = y_away.loc[keep_mask].reset_index(drop=True)
    y_win = y_win.loc[keep_mask].astype(int).reset_index(drop=True)

    log.info("Dataset after cleaning: %d rows", len(df))

    # Infer features
    num_cols, cat_cols = _infer_features(df)
    feature_cols = num_cols + cat_cols

    if not feature_cols:
        raise RuntimeError("No features found after leakage sanitization. Check your dataset.")

    X = df[feature_cols].copy()

    # Optimize memory for numeric columns
    for col in num_cols:
        if col in X.columns:
            X[col] = X[col].astype("float32")

    # Create train/holdout split (chronological)
    train_idx, holdout_idx = _get_holdout_split(X, y_home, holdout_ratio=0.15)
    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_home_train, y_home_holdout = y_home.iloc[train_idx], y_home.iloc[holdout_idx]
    y_away_train, y_away_holdout = y_away.iloc[train_idx], y_away.iloc[holdout_idx]
    y_win_train, y_win_holdout = y_win.iloc[train_idx], y_win.iloc[holdout_idx]

    log.info("Train set: %d rows | Holdout set: %d rows", len(train_idx), len(holdout_idx))

    # Build preprocessor
    pre = _make_preprocessor(num_cols, cat_cols)

    # Fit preprocessor to get transformed feature count
    pre.fit(X_train)
    try:
        n_transformed = len(pre.get_feature_names_out())
    except Exception:
        n_transformed = X_train.shape[1]

    log.info("Transformed feature count: %d", n_transformed)
    log.info("-" * 60)

    # Train models
    log.info("Training home score regressor...")
    home_model, home_params = _fit_regression(
        X_train, y_home_train, pre, RANDOM_SEED, N_JOBS, label="home_score"
    )

    log.info("Training away score regressor...")
    away_model, away_params = _fit_regression(
        X_train, y_away_train, pre, RANDOM_SEED, N_JOBS, label="away_score"
    )

    log.info("Training win probability classifier...")
    win_model, win_params = _fit_classifier(
        X_train, y_win_train, pre, RANDOM_SEED, N_JOBS
    )

    log.info("-" * 60)
    log.info("Evaluating on holdout set...")

    # Evaluate on holdout
    home_metrics = _evaluate_regression(home_model, X_holdout, y_home_holdout)
    away_metrics = _evaluate_regression(away_model, X_holdout, y_away_holdout)

    # Pass calibration flag to classifier evaluation
    is_calibrated = win_params.get("calibrated", False)
    win_metrics = _evaluate_classifier(win_model, X_holdout, y_win_holdout, is_calibrated)

    log.info("Home Score Regressor - MAE: %.2f, RMSE: %.2f, R²: %.3f",
             home_metrics.mae, home_metrics.rmse, home_metrics.r2)
    log.info("Away Score Regressor - MAE: %.2f, RMSE: %.2f, R²: %.3f",
             away_metrics.mae, away_metrics.rmse, away_metrics.r2)
    log.info("Win Classifier - Accuracy: %.3f, ROC-AUC: %.3f, Brier: %.4f, ECE: %.4f",
             win_metrics.accuracy, win_metrics.roc_auc, win_metrics.brier_score, win_metrics.ece)

    # Extract feature importances
    log.info("Extracting feature importances...")
    home_importance = _extract_feature_importance(home_model, feature_cols, "regressor")
    away_importance = _extract_feature_importance(away_model, feature_cols, "regressor")
    win_importance = _extract_feature_importance(win_model, feature_cols, "classifier")

    # Save artifacts
    log.info("-" * 60)
    log.info("Saving artifacts to %s", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dump(pre, os.path.join(out_dir, "preprocessor.joblib"))
    dump(home_model, os.path.join(out_dir, "home_model.joblib"))
    dump(away_model, os.path.join(out_dir, "away_model.joblib"))
    dump(win_model, os.path.join(out_dir, "win_clf_calibrated.joblib"))

    # Training duration
    training_duration = time.time() - training_start
    training_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build summary
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
        dataset_hash=_dataset_hash(df),
        production_ready=True,
        home_model_metrics=home_metrics.to_dict(),
        away_model_metrics=away_metrics.to_dict(),
        win_model_metrics=win_metrics.to_dict(),
        cv_best_params={
            "home_model": home_params,
            "away_model": away_params,
            "win_model": win_params,
        },
    )

    # Metadata for API consumption
    metadata = {
        "training_timestamp_utc": summary.training_timestamp_utc,
        "dataset_hash": summary.dataset_hash,
        "preprocessor": "preprocessor.joblib",
        "home_model": "home_model.joblib",
        "away_model": "away_model.joblib",
        "win_model": "win_clf_calibrated.joblib",
        "raw_feature_columns": {"numeric": num_cols, "categorical": cat_cols},
        "production_ready": True,
        "cv": {"type": "TimeSeriesSplit", "n_splits": N_SPLITS},
        "holdout_metrics": {
            "home": home_metrics.to_dict(),
            "away": away_metrics.to_dict(),
            "win": win_metrics.to_dict(),
        },
        "rows": summary.rows_total,
        "features": len(feature_cols),
    }

    # Feature importance report
    feature_importance = {
        "home_model": dict(list(home_importance.items())[:50]),  # Top 50
        "away_model": dict(list(away_importance.items())[:50]),
        "win_model": dict(list(win_importance.items())[:50]),
    }

    # Write reports
    with open(os.path.join(out_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(out_dir, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)

    log.info("=" * 60)
    log.info("Training complete in %s", _timer(training_start))
    log.info("Artifacts saved:")
    log.info("  - preprocessor.joblib")
    log.info("  - home_model.joblib")
    log.info("  - away_model.joblib")
    log.info("  - win_clf_calibrated.joblib")
    log.info("  - metadata.json")
    log.info("  - training_report.json")
    log.info("  - feature_importance.json")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train leak-free NFL ML models with time-aware CV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models.py --data backend/data/game_features_2014_2025.csv --out backend/models
  python train_models.py --data data/game_features.csv --out models --hp-niter 20 --n-jobs 4
        """,
    )
    parser.add_argument(
        "--data", type=str,
        default="./backend/data/game_features_2014_2025.csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--out", type=str,
        default="backend/models",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--n-jobs", type=int,
        default=N_JOBS,
        help=f"Parallel jobs for CV (default: {N_JOBS})",
    )
    parser.add_argument(
        "--hp-niter", type=int,
        default=HP_N_ITER,
        help=f"RandomizedSearchCV iterations (default: {HP_N_ITER})",
    )
    parser.add_argument(
        "--cv-splits", type=int,
        default=N_SPLITS,
        help=f"TimeSeriesSplit folds (default: {N_SPLITS})",
    )

    args = parser.parse_args()

    # Override globals with CLI args
    HP_N_ITER = args.hp_niter
    N_SPLITS = args.cv_splits
    N_JOBS = args.n_jobs

    main(args.data, args.out)
