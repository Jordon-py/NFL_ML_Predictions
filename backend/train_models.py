#!/usr/bin/env python
"""
train_models.py — Production training for NFL predictions
- Regressors: predict home/away scores (LightGBM + GridSearchCV)
- Classifier: predict home win probability (LightGBMClassifier + calibration)
- Saves: preprocessor.joblib, home_model.joblib, away_model.joblib, win_clf_calibrated.joblib
- Reports: models/metadata.json, models/training_report.json, models/validation_errors.csv
"""
from __future__ import annotations

import json, logging, time, warnings, hashlib, os
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss, mean_absolute_error, r2_score,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    


warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
force_col_wise = True
# Paths
BACKEND_DIR = Path(__file__).resolve().parent


def _resolve_train_test_paths() -> Tuple[Path, Path]:
    env_train = os.getenv("NFL_TRAIN_PATH")
    env_test = os.getenv("NFL_TEST_PATH")
    default_train = BACKEND_DIR / "data" / "train.csv"
    default_test = BACKEND_DIR / "data" / "test.csv"
    train_path = Path(env_train) if env_train else default_train
    test_path = Path(env_test) if env_test else default_test
    missing = [str(p) for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files: " + ", ".join(missing) + ". "
            "Expected train/test CSVs in backend/data or via NFL_TRAIN_PATH/NFL_TEST_PATH."
        )
    return train_path, test_path

MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("train")

# Feature spec
WINDOWS = (3, 5)
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
    return [f"{side}_prior_{metric}_{w}" for metric in PRIOR_METRICS for w in WINDOWS]

SIDE_FEATURES = _side_prior_features("home") + _side_prior_features("away")
DIFF_FEATURES = [
    f"home_minus_away_{metric}_{w}"
    for metric in PRIOR_METRICS
    for w in WINDOWS
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

BASE_FEATURES = list(dict.fromkeys(SIDE_FEATURES + DIFF_FEATURES + BETTING_CONTEXT_FEATURES))

def _load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split train/test datasets from CSV files."""
    train_path, test_path = _resolve_train_test_paths()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_raw_len = len(train_df)
    test_raw_len = len(test_df)
    train_df = train_df[train_df["home_points_for"].notna() & train_df["away_points_for"].notna()].copy()
    test_df = test_df[test_df["home_points_for"].notna() & test_df["away_points_for"].notna()].copy()
    if len(train_df) < train_raw_len:
        log.warning(
            "Dropped %d train rows without final scores", train_raw_len - len(train_df)
        )
    if len(test_df) < test_raw_len:
        log.warning(
            "Dropped %d test rows without final scores", test_raw_len - len(test_df)
        )

    for split_name, split_df, split_path in (("train", train_df, train_path), ("test", test_df, test_path)):
        split_df.columns = [c.strip() for c in split_df.columns]
        if split_df.empty:
            raise ValueError(f"{split_name.capitalize()} dataset is empty at {split_path}")

    for frame in (train_df, test_df):
        frame["home_win"] = (frame["home_points_for"] > frame["away_points_for"]).astype(int)

    log.info("Loaded %d train games from %s", len(train_df), train_path)
    log.info("Loaded %d test games from %s", len(test_df), test_path)

    missing = [c for c in BASE_FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing required features in train dataset: {missing}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _compute_recency_weights(df: pd.DataFrame) -> np.ndarray:
    """Boost recent games so the model tracks league drift."""
    if "season" not in df.columns or "week" not in df.columns:
        raise ValueError("Recency weighting requires 'season' and 'week' columns.")
    seasons = df["season"].to_numpy(dtype=float)
    weeks = df["week"].to_numpy(dtype=float)
    season_span = max(seasons.max() - seasons.min(), 1.0)
    season_norm = (seasons - seasons.min()) / season_span
    week_norm = weeks / max(weeks.max(), 1.0)
    weights = 0.4 + 0.4 * season_norm + 0.2 * week_norm
    return weights / weights.mean()

def _preprocessor(features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                       ("scaler", StandardScaler())]), features)],
        remainder="drop",
    )

def _grid_lgbm_reg() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [75, 150, 175],
        "learning_rate": [0.05, 0.1, .02],
        "max_depth": [4, 10],
        "num_leaves": [15, 25],
        "subsample": [0.4, 0.9],
        "colsample_bytree": [0.4, 0.9],
        "reg_alpha": [0.2, 0.1],
        "reg_lambda": [0.2, 0.1],
        "min_child_samples": [20, 35],
    }

def _grid_lgbm_clf() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [75, 150, 175],
        "learning_rate": [0.05, 0.1, .02],
        "max_depth": [4, 10],
        "num_leaves": [15, 25],
        "subsample": [0.4, 0.9],
        "colsample_bytree": [0.4, 0.9],
        "reg_alpha": [0.2, 0.1],
        "reg_lambda": [0.2, 0.1],
        "min_child_samples": [20, 35],
        "class_weight": [None, "balanced"],
    }

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

def _fit_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    name: str,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[LGBMRegressor, Dict[str, Any]]:
    """Fit regressor on train set, validate on test set."""
    lgbm = LGBMRegressor(objective="regression", random_state=4, n_jobs=-1, verbose=1)
    
    # Use 3-fold TimeSeriesSplit on training data for hyperparameter tuning
    cv = TimeSeriesSplit(n_splits=3)

    # Cast the estimator to satisfy type checker
    rs = RandomizedSearchCV(
        estimator=cast(BaseEstimator, lgbm),
        param_distributions=_grid_lgbm_reg(),
        n_iter=15,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
        refit=True,
        random_state=4
    )
    t0 = time.time()
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    rs.fit(X_train, y_train, **fit_kwargs)
    best = cast(LGBMRegressor, rs.best_estimator_)
    
    # Evaluate on training set
    yhat_train = np.asarray(best.predict(X_train)).ravel()
    
    # Evaluate on holdout test set
    yhat_test = np.asarray(best.predict(X_test)).ravel()
    
    res = {
        "best_params": rs.best_params_,
        "cv_rmse": float(rs.best_score_),
        "train_r2": float(r2_score(y_train, yhat_train)),
        "train_mae": float(mean_absolute_error(y_train, yhat_train)),
        "test_r2": float(r2_score(y_test, yhat_test)),
        "test_mae": float(mean_absolute_error(y_test, yhat_test)),
        "search_time_s": time.time() - t0,
        "n_candidates": len(rs.cv_results_["params"]),
    }
    if res["train_r2"] < -0.2:
        raise ValueError(f"{name} regressor underfit: R²={res['train_r2']:.3f}")
    return best, res

def _fit_classifier_optimized(
    X_train,
    y_train,
    X_test,
    y_test,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[BaseEstimator, Dict[str, Any], pd.DataFrame]:
    """
    Fit classifier on train set, validate on holdout test set.
    Returns calibrated classifier, metrics, and test set predictions.
    """

    # Use RandomizedSearchCV for faster hyperparameter tuning
    base = LGBMClassifier(objective="binary",
                          random_state=4,
                          n_jobs=-1,
                          verbose=0,
                          learning_rate=0.1
                          )

    # Use 4-fold TimeSeriesSplit on training data
    cv_splitter = TimeSeriesSplit(n_splits=4)
    
    # Use RandomizedSearchCV with a limited number of iterations
    rs = RandomizedSearchCV(
        estimator=cast(BaseEstimator, base),
        param_distributions=_grid_lgbm_clf(), 
        n_iter=30, 
        cv=cv_splitter, 
        scoring="roc_auc",
        n_jobs=-1, 
        verbose=0, 
        return_train_score=True,
        refit=True,
        random_state=4
    )
    
    t0 = time.time()
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    rs.fit(X_train, y_train, **fit_kwargs)
    best_uncalibrated = rs.best_estimator_
    
    # Calibrate the best model on training data
    calib = CalibratedClassifierCV(best_uncalibrated, cv=cv_splitter, method="isotonic")
    calib.fit(X_train, y_train, sample_weight=sample_weight)
    
    # Generate predictions on holdout test set
    prob_confidence = calib.predict_proba(X_test)[:, 1]
    prob_home_win_pct = np.round(prob_confidence * 100, 1)
    
    # Determine high confidence predictions: check if >=75% of games are outside 40-65% range
    is_high_confidence_batch = np.mean((prob_confidence >= 0.65) | (prob_confidence <= 0.40)) >= 0.75
    is_high_confidence_per_game = (prob_confidence >= 0.65) | (prob_confidence <= 0.40)
    
    pred_test = (prob_confidence >= 0.50).astype(int)

    # Create test predictions DataFrame
    preds_df = pd.DataFrame({
        "idx": np.arange(len(X_test)),
        "fold": -1,  # -1 indicates holdout test set
        "prob_home_win": prob_confidence,
        "prob_home_win_pct": prob_home_win_pct,
        "is_high_confidence": is_high_confidence_per_game
    })
    
    # Evaluate on training set for comparison
    prob_train = calib.predict_proba(X_train)[:, 1]
    pred_train = (prob_train >= 0.5).astype(int)
    
    metrics = {
        "best_params": rs.best_params_,
        "cv_auc": float(rs.best_score_),
        "train_auc": float(roc_auc_score(y_train, prob_train)),
        "train_accuracy": float(accuracy_score(y_train, pred_train)),
        "train_precision": float(precision_score(y_train, pred_train)),
        "train_recall": float(recall_score(y_train, pred_train)),
        "train_f1": float(f1_score(y_train, pred_train)),
        "train_brier": float(brier_score_loss(y_train, prob_train)),
        "test_auc": float(roc_auc_score(y_test, prob_confidence)),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_precision": float(precision_score(y_test, pred_test)),
        "test_recall": float(recall_score(y_test, pred_test)),
        "test_f1": float(f1_score(y_test, pred_test)),
        "test_brier": float(brier_score_loss(y_test, prob_confidence)),
        "search_time_s": time.time() - t0,
        "n_candidates": len(rs.cv_results_["params"]),
    }
    
    return calib, metrics, preds_df


def main() -> None:
    # Load data with train/test split
    train_df, test_df = _load_dataset()
    
    # Prepare training data
    X_train_raw = train_df[BASE_FEATURES]
    y_train_home = train_df["home_points_for"].astype(float).values
    y_train_away = train_df["away_points_for"].astype(float).values
    y_train_win = train_df["home_win"].astype(int).values
    train_weights = _compute_recency_weights(train_df)
    
    # Prepare test data
    X_test_raw = test_df[BASE_FEATURES]
    y_test_home = test_df["home_points_for"].astype(float).values
    y_test_away = test_df["away_points_for"].astype(float).values
    y_test_win = test_df["home_win"].astype(int).values

    # Fit preprocessor on training data only
    pre = _preprocessor(BASE_FEATURES)
    X_train_proc = pre.fit_transform(X_train_raw)
    X_test_proc = pre.transform(X_test_raw)

    log.info("Fitted preprocessor on %d train samples, %d test samples", 
             X_train_proc.shape[0], X_test_proc.shape[0])
    log.info("Features: %d", X_train_proc.shape[1])
    log.info(
        "Recency weighting applied: mean=%.3f, max=%.3f, min=%.3f",
        float(train_weights.mean()),
        float(train_weights.max()),
        float(train_weights.min()),
    )

    # Regressors
    log.info("Training LightGBM regressors (home, away)...")
    home_reg, home_res = _fit_regressor(
        X_train_proc, y_train_home, X_test_proc, y_test_home, "home", sample_weight=train_weights
    )
    away_reg, away_res = _fit_regressor(
        X_train_proc, y_train_away, X_test_proc, y_test_away, "away", sample_weight=train_weights
    )
    log.info("Home regressor - Train R²: %.3f, Test R²: %.3f", home_res["train_r2"], home_res["test_r2"])
    log.info("Away regressor - Train R²: %.3f, Test R²: %.3f", away_res["train_r2"], away_res["test_r2"])

    # Classifier
    log.info("Training LightGBM classifier for win probability...")
    win_clf, win_res, test_preds = _fit_classifier_optimized(
        X_train_proc, y_train_win, X_test_proc, y_test_win, sample_weight=train_weights
    )
    log.info("Win classifier - Train AUC: %.3f, Test AUC: %.3f", win_res["train_auc"], win_res["test_auc"])

    # Persist models (trained on full training set)
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(home_reg, MODELS_DIR / "home_model.joblib")
    joblib.dump(away_reg, MODELS_DIR / "away_model.joblib")
    joblib.dump(win_clf, MODELS_DIR / "win_clf_calibrated.joblib")

    # Save test set predictions with metadata
    test_preds = test_preds.merge(test_df.reset_index().rename(columns={"index":"idx"})[
        ["idx","season","week","home_team","away_team","home_win"]
    ], on="idx", how="left")
    test_preds["abs_error"] = (test_preds["prob_home_win"] - test_preds["home_win"]).abs()
    test_preds.to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    log.info("Saved test set predictions to test_predictions.csv")

    # Reports
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    hash_values = pd.util.hash_pandas_object(combined_df, index=False).values
    hash_bytes = np.asarray(hash_values).tobytes()
    dataset_hash = hashlib.md5(hash_bytes).hexdigest()[:10]
    
    training_report = {
        "dataset": {
            "total_rows": int(len(combined_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "hash": dataset_hash
        },
        "features_used": BASE_FEATURES,
        "sample_weighting": {
            "strategy": "recency_linear",
            "formula": "0.4 base + 0.4*season_norm + 0.2*week_norm",
            "mean": float(train_weights.mean()),
            "max": float(train_weights.max()),
            "min": float(train_weights.min()),
        },
        "regression": {"home": home_res, "away": away_res},
        "classification": win_res,
        "thresholds": {"win_auc_min": 0.65},
        "production_ready_win_model": bool(win_res["test_auc"] >= 0.60),
    }
    (MODELS_DIR / "training_report.json").write_text(json.dumps(training_report, indent=2))

    metadata = {
        "training_timestamp": pd.Timestamp.now().isoformat(),
        "dataset_hash": dataset_hash,
        "training_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "train_cutoff": "2025 Week 3",
        "test_period": "2025 Week 4+",
        "raw_feature_columns": {"numeric": BASE_FEATURES, "categorical": []},
        "models": {
            "home_model": "home_model.joblib",
            "away_model": "away_model.joblib",
            "win_model": "win_clf_calibrated.joblib",
        },
        "preprocessor": "preprocessor.joblib",
        "targets": ["home_points_for","away_points_for","home_win"],
        "sample_weighting": {
            "strategy": "recency_linear",
            "formula": "0.4 base + 0.4*season_norm + 0.2*week_norm",
            "mean": float(train_weights.mean()),
            "max": float(train_weights.max()),
            "min": float(train_weights.min()),
        },
        "model_scores": {
            "home_train_r2": float(home_res["train_r2"]),
            "home_test_r2": float(home_res["test_r2"]),
            "away_train_r2": float(away_res["train_r2"]),
            "away_test_r2": float(away_res["test_r2"]),
            "win_cv_auc": float(win_res["cv_auc"]),
            "win_train_auc": float(win_res["train_auc"]),
            "win_test_auc": float(win_res["test_auc"]),
        },
        "production_ready_win_model": bool(win_res["test_auc"] >= 0.65),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    log.info("Saved models and reports to %s", MODELS_DIR)
    if not metadata["production_ready_win_model"]:
        log.warning("Win model below AUC threshold (%.3f < 0.65). Not production-ready.", win_res["test_auc"])

if __name__ == "__main__":
    main()
