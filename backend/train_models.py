#!/usr/bin/env python
"""
train_models.py — Production-Ready Training Pipeline for NFL Predictions

This script orchestrates the end-to-end training, evaluation, and serialization of
all models required for the NFL prediction API. It is designed for automated,
production environments.

Key Responsibilities:
- Load and validate the master dataset (`merged_nfl_data.csv`).
- Split data into training and testing sets using a time-series approach based on the latest season's progress.
- Train and tune regressors (LightGBM) to predict home and away scores.
- Train and tune a calibrated classifier (LightGBM) to predict home win probability.
- Apply recency weighting to prioritize modern game data.
- Generate and save comprehensive artifacts:
  - Preprocessing pipeline (`preprocessor.joblib`).
  - Trained models (`home_model.joblib`, `away_model.joblib`, `win_clf_calibrated.joblib`).
  - Detailed performance metrics (`training_report.json`).
  - Model metadata for the API (`metadata.json`).
  - Test set predictions and errors for analysis (`test_predictions.csv`).

Execution:
  Run from the repository root or ensure the backend directory is in the Python path.
  `python backend/train_models.py`
"""

from __future__ import annotations

import json
import logging
import logging.config
import time
import warnings
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# --- Configuration ---

# Paths
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
force_col_wise = True
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
LOG_DIR = BACKEND_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "default",
            "filename": LOG_DIR / "training.log",
            "mode": "w",  # Overwrite log file for each training run
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
}
logging.config.dictConfig(LOGGING_CONFIG)
log = logging.getLogger(__name__)

# Training & Evaluation Parameters
# Defines the time-series split for the most recent season.
# The model trains on all data *before* the latest season, plus these initial weeks.
# It then tests on the subsequent weeks of that same season.
CURRENT_SEASON_TRAIN_WEEKS = (
    3  # Number of weeks from the latest season to include in training
)
CURRENT_SEASON_TEST_WEEKS = 2  # Number of weeks to use for hold-out testing
RANDOM_SEED = 42  # Ensures reproducibility for stochastic processes
HYPERPARAM_SEARCH_ITERATIONS = 25  # Number of iterations for RandomizedSearchCV

# --- End Configuration ---


def _resolve_dataset_path() -> Path:
    """
    Locates and validates the primary dataset file.

    Returns:
        Path: The validated path to the merged dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    dataset_path = DATA_DIR / "merged_nfl_data.csv"
    if not dataset_path.exists():
        log.error("Dataset file not found at '%s'.", dataset_path)
        raise FileNotFoundError(
            f"Missing dataset file: {dataset_path}. "
            "Ensure historical data is merged and available before training."
        )
    log.info("Dataset found at '%s'.", dataset_path)
    return dataset_path


def _load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads the merged dataset and splits it into training and testing sets.

    The split is time-series aware:
    - Training set: All seasons before the latest one, plus the first `CURRENT_SEASON_TRAIN_WEEKS` of the latest season.
    - Testing set: The `CURRENT_SEASON_TEST_WEEKS` immediately following the training weeks in the latest season.

    Returns:
        A tuple containing:
        - pd.DataFrame: The training data.
        - pd.DataFrame: The testing data.
        - Dict[str, Any]: Metadata about the split (e.g., season, weeks).
    """
    dataset_path = _resolve_dataset_path()
    try:
        full_df = pd.read_csv(dataset_path)
        full_df.columns = [c.strip() for c in full_df.columns]
        log.info("Successfully loaded dataset with %d rows.", len(full_df))
    except Exception as e:
        log.exception("Failed to load or parse dataset from '%s'.", dataset_path)
        raise IOError(f"Could not read dataset file: {e}") from e

    required_cols = set(BASE_FEATURES) | {
        "season",
        "week",
        "home_points_for",
        "away_points_for",
    }
    missing_cols = required_cols - set(full_df.columns)
    if missing_cols:
        log.error("Dataset is missing required columns: %s", sorted(list(missing_cols)))
        raise ValueError(
            f"Missing required columns in merged dataset: {sorted(list(missing_cols))}"
        )

    completed_mask = (
        full_df["home_points_for"].notna() & full_df["away_points_for"].notna()
    )
    dropped_rows = int((~completed_mask).sum())
    if dropped_rows:
        log.warning(
            "Dropped %d rows from the dataset that were missing final scores.",
            dropped_rows,
        )
    full_df = full_df[completed_mask].copy()
    if full_df.empty:
        log.error(
            "Dataset at '%s' contains no completed games after filtering.", dataset_path
        )
        raise ValueError(f"Merged dataset at {dataset_path} has no completed games.")

    full_df = full_df.sort_values(["season", "week"]).reset_index(drop=True)

    latest_season = int(full_df["season"].max())
    latest_weeks = sorted(
        full_df.loc[full_df["season"] == latest_season, "week"]
        .dropna()
        .unique()
        .astype(int)
    )

    required_weeks = CURRENT_SEASON_TRAIN_WEEKS + CURRENT_SEASON_TEST_WEEKS
    if len(latest_weeks) < required_weeks:
        log.error(
            "Season %d has only %d completed weeks, but %d are required for the train/test split.",
            latest_season,
            len(latest_weeks),
            required_weeks,
        )
        raise ValueError(
            f"Season {latest_season} only has {len(latest_weeks)} completed weeks. "
            f"At least {required_weeks} are required for the train/test split."
        )

    train_weeks_latest = latest_weeks[:CURRENT_SEASON_TRAIN_WEEKS]
    test_weeks_latest = latest_weeks[
        CURRENT_SEASON_TRAIN_WEEKS : CURRENT_SEASON_TRAIN_WEEKS
        + CURRENT_SEASON_TEST_WEEKS
    ]

    train_mask = (full_df["season"] < latest_season) | (
        (full_df["season"] == latest_season)
        & (full_df["week"].isin(train_weeks_latest))
    )
    test_mask = (full_df["season"] == latest_season) & (
        full_df["week"].isin(test_weeks_latest)
    )

    train_df = full_df[train_mask].copy()
    test_df = full_df[test_mask].copy()

    if train_df.empty or test_df.empty:
        log.error(
            "Train/test split resulted in an empty dataframe. Check dataset coverage and split logic."
        )
        raise ValueError(
            "Train/test split resulted in an empty dataframe. Verify merged dataset coverage."
        )

    # Add target variable 'home_win' to both dataframes
    for frame in (train_df, test_df):
        frame["home_win"] = (
            frame["home_points_for"] > frame["away_points_for"]
        ).astype(int)

    log.info(
        "Data split complete: %d training games, %d testing games.",
        len(train_df),
        len(test_df),
    )
    log.info(
        "Training on all data before season %d, plus weeks %s. Testing on weeks %s.",
        latest_season,
        train_weeks_latest,
        test_weeks_latest,
    )

    split_details = {
        "latest_season": latest_season,
        "train_weeks": train_weeks_latest,
        "test_weeks": test_weeks_latest,
        "dataset_path": str(dataset_path),
    }
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        split_details,
    )


# --- Feature Engineering & Preprocessing ---

# Feature specification: Defines the set of features used for modeling.
WINDOWS = (3, 5)  # Rolling window sizes for prior performance metrics
PRIOR_METRICS = [
    "pf_avg",
    "pa_avg",
    "win_pct",
    "off_epa_per_play",
    "off_success_rate",
    "off_explosive_rate",
    "off_third_down_pct",
    "off_pass_over_expected",
    "off_turnover_rate",
    "def_epa_per_play",
    "def_success_rate_allowed",
    "def_explosive_rate_allowed",
    "def_takeaway_rate",
]


def _side_prior_features(side: str) -> List[str]:
    """Generate feature names for one side (home/away)."""
    return [f"{side}_prior_{metric}_{w}" for metric in PRIOR_METRICS for w in WINDOWS]


# Combine all feature groups into the final list
SIDE_FEATURES = _side_prior_features("home") + _side_prior_features("away")
DIFF_FEATURES = [
    f"home_minus_away_{metric}_{w}" for metric in PRIOR_METRICS for w in WINDOWS
]
BETTING_CONTEXT_FEATURES = [
    "home_moneyline_prob",
    "away_moneyline_prob",
    "moneyline_prob_diff",
    "spread_line",
    "total_line",
    "home_rest",
    "away_rest",
    "rest_diff",
]
BASE_FEATURES = sorted(
    list(dict.fromkeys(SIDE_FEATURES + DIFF_FEATURES + BETTING_CONTEXT_FEATURES))
)


def _compute_recency_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Generates sample weights that prioritize more recent games.
    This helps the model adapt to league-wide trends and changes in play style.

    Args:
        df (pd.DataFrame): The training dataframe, must contain 'season' and 'week'.

    Returns:
        np.ndarray: An array of weights, one for each row in the dataframe.
    """
    if "season" not in df.columns or "week" not in df.columns:
        log.error("Recency weighting requires 'season' and 'week' columns.")
        raise ValueError("Recency weighting requires 'season' and 'week' columns.")

    seasons = df["season"].to_numpy(dtype=float)
    weeks = df["week"].to_numpy(dtype=float)

    season_span = max(seasons.max() - seasons.min(), 1.0)
    season_norm = (seasons - seasons.min()) / season_span
    week_norm = weeks / max(weeks.max(), 1.0)

    # Formula: 40% base weight, 40% from season progress, 20% from week progress
    weights = 0.4 + 0.4 * season_norm + 0.2 * week_norm
    normalized_weights = weights / weights.mean()
    log.info(
        "Applied recency weighting. Mean: %.3f, Max: %.3f, Min: %.3f",
        normalized_weights.mean(),
        normalized_weights.max(),
        normalized_weights.min(),
    )
    return normalized_weights


def _create_preprocessor(features: List[str]) -> ColumnTransformer:
    """
    Creates the preprocessing pipeline for numerical features.
    - Imputes missing values using the median.
    - Scales features to have zero mean and unit variance.

    Args:
        features (List[str]): The list of numerical feature names to process.

    Returns:
        ColumnTransformer: The scikit-learn preprocessor.
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )


# --- Model Training & Tuning ---


def _get_lgbm_reg_grid() -> Dict[str, List[Any]]:
    """Hyperparameter search space for the score regressors."""
    return {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [15, 25, 31],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.1, 0.2, 0.5],
        "reg_lambda": [0.1, 0.2, 0.5],
        "min_child_samples": [20, 30],
    }


def _get_lgbm_clf_grid() -> Dict[str, List[Any]]:
    """Hyperparameter search space for the win probability classifier."""
    return {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [15, 25, 31],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.1, 0.2, 0.5],
        "reg_lambda": [0.1, 0.2, 0.5],
        "min_child_samples": [20, 30],
        "class_weight": [None, "balanced"],
    }


def _fit_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[LGBMRegressor, Dict[str, Any]]:
    """
    Fits and tunes a LightGBM regressor.

    Args:
        X_train, y_train: Training data and targets.
        X_test, y_test: Testing data and targets for evaluation.
        name (str): Name of the model for logging (e.g., "home_score").
        sample_weight (Optional[np.ndarray]): Weights for training samples.

    Returns:
        A tuple containing:
        - LGBMRegressor: The best trained regressor.
        - Dict[str, Any]: A dictionary of performance metrics.
    """
    log.info("Starting training for '%s' regressor...", name)
    lgbm = LGBMRegressor(
        objective="regression", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    )

    cv = TimeSeriesSplit(n_splits=3)
    rs = RandomizedSearchCV(
        estimator=cast(BaseEstimator, lgbm),
        param_distributions=_get_lgbm_reg_grid(),
        n_iter=HYPERPARAM_SEARCH_ITERATIONS,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_SEED,
    )

    t0 = time.time()
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    rs.fit(X_train, y_train, **fit_kwargs)
    best = cast(LGBMRegressor, rs.best_estimator_)
    search_time = time.time() - t0

    yhat_train = best.predict(X_train)
    yhat_test = best.predict(X_test)

    metrics = {
        "best_params": rs.best_params_,
        "cv_rmse": -rs.best_score_,
        "train_r2": r2_score(y_train, yhat_train),
        "train_mae": mean_absolute_error(y_train, yhat_train),
        "test_r2": r2_score(y_test, yhat_test),
        "test_mae": mean_absolute_error(y_test, yhat_test),
        "search_time_s": search_time,
    }
    log.info(
        "'%s' regressor training complete. Test R²: %.3f, Test MAE: %.3f",
        name,
        metrics["test_r2"],
        metrics["test_mae"],
    )

    if metrics["train_r2"] < 0.1:
        log.warning(
            "'%s' regressor may have underfit with Train R² of %.3f.",
            name,
            metrics["train_r2"],
        )

    return best, metrics


def _fit_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame]:
    """
    Fits, tunes, and calibrates a LightGBM classifier.

    Args:
        X_train, y_train: Training data and targets.
        X_test, y_test: Testing data and targets for evaluation.
        sample_weight (Optional[np.ndarray]): Weights for training samples.

    Returns:
        A tuple containing:
        - BaseEstimator: The best trained and calibrated classifier.
        - Dict[str, Any]: A dictionary of performance metrics.
        - pd.DataFrame: Predictions and probabilities on the test set.
    """
    log.info("Starting training for win probability classifier...")
    base = LGBMClassifier(
        objective="binary", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    )

    cv_splitter = TimeSeriesSplit(n_splits=4)
    rs = RandomizedSearchCV(
        estimator=cast(BaseEstimator, base),
        param_distributions=_get_lgbm_clf_grid(),
        n_iter=HYPERPARAM_SEARCH_ITERATIONS,
        cv=cv_splitter,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_SEED,
    )

    t0 = time.time()
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    rs.fit(X_train, y_train, **fit_kwargs)
    best_uncalibrated = rs.best_estimator_

    # Calibrate the best model to produce reliable probabilities
    log.info("Calibrating classifier using isotonic regression...")
    calib = CalibratedClassifierCV(best_uncalibrated, cv=cv_splitter, method="isotonic")
    calib.fit(X_train, y_train, sample_weight=sample_weight)
    search_time = time.time() - t0

    # Evaluate on holdout test set
    prob_home_win = calib.predict_proba(X_test)[:, 1]
    pred_test = (prob_home_win >= 0.5).astype(int)

    preds_df = pd.DataFrame(
        {
            "idx": np.arange(len(X_test)),
            "prob_home_win": prob_home_win,
            "predicted_outcome": pred_test,
        }
    )

    # Evaluate on training set for comparison
    prob_train = calib.predict_proba(X_train)[:, 1]
    pred_train = (prob_train >= 0.5).astype(int)

    metrics = {
        "best_params": rs.best_params_,
        "cv_auc": rs.best_score_,
        "train_auc": roc_auc_score(y_train, prob_train),
        "train_accuracy": accuracy_score(y_train, pred_train),
        "train_precision": precision_score(y_train, pred_train),
        "train_recall": recall_score(y_train, pred_train),
        "train_f1": f1_score(y_train, pred_train),
        "train_brier": brier_score_loss(y_train, prob_train),
        "test_auc": roc_auc_score(y_test, prob_home_win),
        "test_accuracy": accuracy_score(y_test, pred_test),
        "test_precision": precision_score(y_test, pred_test),
        "test_recall": recall_score(y_test, pred_test),
        "test_f1": f1_score(y_test, pred_test),
        "test_brier": brier_score_loss(y_test, prob_home_win),
        "search_time_s": search_time,
    }
    log.info(
        "Classifier training complete. Test AUC: %.3f, Test Brier Score: %.3f",
        metrics["test_auc"],
        metrics["test_brier"],
    )

    return calib, metrics, preds_df


def _save_artifacts(
    pre: ColumnTransformer,
    home_reg: LGBMRegressor,
    away_reg: LGBMRegressor,
    win_clf: BaseEstimator,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_preds: pd.DataFrame,
    split_info: Dict[str, Any],
    train_weights: np.ndarray,
    home_res: Dict[str, Any],
    away_res: Dict[str, Any],
    win_res: Dict[str, Any],
) -> None:
    """
    Saves all training artifacts to the `models` directory.

    This includes:
    - The preprocessor.
    - All trained models.
    - Test predictions for analysis.
    - A detailed training report.
    - API metadata.
    """
    log.info("Saving all training artifacts to '%s'...", MODELS_DIR)

    # Save models
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(home_reg, MODELS_DIR / "home_model.joblib")
    joblib.dump(away_reg, MODELS_DIR / "away_model.joblib")
    joblib.dump(win_clf, MODELS_DIR / "win_clf_calibrated.joblib")

    # Save test set predictions with metadata for analysis
    test_preds_full = test_preds.merge(
        test_df.reset_index().rename(columns={"index": "idx"})[
            ["idx", "season", "week", "home_team", "away_team", "home_win"]
        ],
        on="idx",
        how="left",
    )
    test_preds_full["abs_error"] = (
        test_preds_full["prob_home_win"] - test_preds_full["home_win"]
    ).abs()
    test_preds_full.to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    log.info("Saved test set predictions to 'test_predictions.csv'.")

    # Generate a hash of the dataset for lineage tracking
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    hash_bytes = pd.util.hash_pandas_object(combined_df, index=False).values.tobytes()
    dataset_hash = hashlib.md5(hash_bytes).hexdigest()

    # Create detailed training report
    training_report = {
        "training_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset": {
            "source_path": split_info["dataset_path"],
            "hash": dataset_hash,
            "total_rows": len(combined_df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "split_details": {
                "latest_season": split_info["latest_season"],
                "train_weeks_in_latest_season": split_info["train_weeks"],
                "test_weeks_in_latest_season": split_info["test_weeks"],
            },
        },
        "features": {"count": len(BASE_FEATURES), "names": BASE_FEATURES},
        "sample_weighting": {
            "strategy": "recency_linear",
            "mean": train_weights.mean(),
            "max": train_weights.max(),
            "min": train_weights.min(),
        },
        "models": {
            "home_score_regressor": home_res,
            "away_score_regressor": away_res,
            "win_probability_classifier": win_res,
        },
        "production_readiness": {
            "win_model_auc_threshold": 0.60,
            "is_ready": win_res["test_auc"] >= 0.60,
        },
    }
    (MODELS_DIR / "training_report.json").write_text(
        json.dumps(training_report, indent=2, default=str)
    )
    log.info("Saved detailed training report to 'training_report.json'.")

    # Create metadata file for the API
    metadata = {
        "training_timestamp_utc": training_report["training_timestamp_utc"],
        "dataset_hash": dataset_hash,
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "raw_feature_columns": {"numeric": BASE_FEATURES, "categorical": []},
        "models": {
            "home_model": "home_model.joblib",
            "away_model": "away_model.joblib",
            "win_model": "win_clf_calibrated.joblib",
        },
        "preprocessor": "preprocessor.joblib",
        "model_scores": {
            "home_test_r2": home_res["test_r2"],
            "away_test_r2": away_res["test_r2"],
            "win_test_auc": win_res["test_auc"],
            "win_test_brier_score": win_res["test_brier"],
        },
        "production_ready": training_report["production_readiness"]["is_ready"],
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    log.info("Saved API metadata to 'metadata.json'.")

    if not metadata["production_ready"]:
        log.warning(
            "Win model did not meet the production readiness threshold (Test AUC %.3f < %.3f).",
            win_res["test_auc"],
            training_report["production_readiness"]["win_model_auc_threshold"],
        )
    log.info("All artifacts saved successfully.")


def main() -> None:
    """Main function to orchestrate the model training pipeline."""
    log.info("--- Starting NFL Model Training Pipeline ---")
    start_time = time.time()

    try:
        # 1. Load and split data
        train_df, test_df, split_info = _load_dataset()

        # 2. Prepare data for modeling
        X_train_raw = train_df[BASE_FEATURES]
        y_train_home = train_df["home_points_for"].astype(float).values
        y_train_away = train_df["away_points_for"].astype(float).values
        y_train_win = train_df["home_win"].astype(int).values
        train_weights = _compute_recency_weights(train_df)

        X_test_raw = test_df[BASE_FEATURES]
        y_test_home = test_df["home_points_for"].astype(float).values
        y_test_away = test_df["away_points_for"].astype(float).values
        y_test_win = test_df["home_win"].astype(int).values

        # 3. Fit preprocessor and transform data
        preprocessor = _create_preprocessor(BASE_FEATURES)
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)
        log.info(
            "Preprocessor fitted. Transformed data shape: Train=%s, Test=%s",
            X_train_proc.shape,
            X_test_proc.shape,
        )

        # 4. Train score regressors
        home_reg, home_res = _fit_regressor(
            X_train_proc,
            y_train_home,
            X_test_proc,
            y_test_home,
            "home_score",
            train_weights,
        )
        away_reg, away_res = _fit_regressor(
            X_train_proc,
            y_train_away,
            X_test_proc,
            y_test_away,
            "away_score",
            train_weights,
        )

        # 5. Train win probability classifier
        win_clf, win_res, test_preds = _fit_classifier(
            X_train_proc, y_train_win, X_test_proc, y_test_win, train_weights
        )

        # 6. Save all artifacts
        _save_artifacts(
            pre=preprocessor,
            home_reg=home_reg,
            away_reg=away_reg,
            win_clf=win_clf,
            train_df=train_df,
            test_df=test_df,
            test_preds=test_preds,
            split_info=split_info,
            train_weights=train_weights,
            home_res=home_res,
            away_res=away_res,
            win_res=win_res,
        )

    except (FileNotFoundError, ValueError, IOError) as e:
        log.exception("A critical error occurred during the training pipeline: %s", e)
        # In a real production system, you might exit with a non-zero status code
        # sys.exit(1)
    except Exception as e:
        log.exception("An unexpected error occurred: %s", e)
        # sys.exit(1)

    finally:
        total_time = time.time() - start_time
        log.info(
            "--- NFL Model Training Pipeline Finished in %.2f seconds ---", total_time
        )


if __name__ == "__main__":
    main()
