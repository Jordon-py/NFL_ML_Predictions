#!/usr/bin/env python3
# ==========================================
# File: backend/train_models.py
# Role: Production-grade trainer for NFL score + win-probability models.
# Input Data: Clean feature datasets (CSV).
# Output Data: Trained model pipelines (.joblib), Performance reports.
# Dependencies: sklearn, pandas, numpy, joblib
# Notes: Implements staging/promotion gating to prevent regression in production.
# ==========================================

# File: backend/train_models.py
# Purpose: Production-grade trainer for NFL score + win-probability models with staging/promotion gating.

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import dump
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
try:
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - sklearn < 1.6
    FrozenEstimator = None
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_MODELS_DIR = BASE_DIR / "models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.utils.ops_reporting import file_sha256, resolve_latest_dataset  # noqa: E402

load_dotenv(BASE_DIR / ".env")


TARGET_HOME = "home_points_for"
TARGET_AWAY = "away_points_for"
TARGET_WIN = "home_win"
WIN_PROBA_FEATURE = "nn_home_win_proba"
TIME_KEYS: Tuple[str, str] = ("season", "week")
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "season",
    "week",
    "home_team",
    "away_team",
    TARGET_HOME,
    TARGET_AWAY,
    TARGET_WIN,
)
ID_COLUMNS: Tuple[str, ...] = (
    "game_id",
    "gid",
    "home_team_id",
    "away_team_id",
    "stadium",
    "home_game_date",
    "away_game_date",
)
LEAK_BLOCKLIST: Tuple[str, ...] = (
    TARGET_HOME,
    TARGET_AWAY,
    TARGET_WIN,
    "winner",
    "point_diff",
    "home_points_against",
    "away_points_against",
    "home_score",
    "away_score",
    "final_home_score",
    "final_away_score",
    "postgame_margin",
    "post_game_total",
    "actual_margin",
)

REG_PARAM_DISTS: Dict[str, Sequence[Any]] = {
    "max_depth": [3, 6, 10, 14],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63, 127],
    "l2_regularization": [0.0, 0.01, 0.05, 0.1],
    "min_samples_leaf": [10, 20, 30],
}
CLF_PARAM_DISTS: Dict[str, Sequence[Any]] = {
    "hidden_layer_sizes": [(64,), (96,), (128,), (64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "learning_rate_init": [3e-4, 1e-3, 3e-3],
    "batch_size": [32, 64, 128],
}


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: Optional[float]


@dataclass
class ClassificationMetrics:
    accuracy: float
    brier: float
    roc_auc: Optional[float]
    log_loss: Optional[float]


def _log_feature_importance(model, feature_names, target_name, log: logging.Logger):
    """Extract and log feature importance for estimators that expose it."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            top_n = 15
            log.info(f"Top {top_n} features for {target_name}:")
            for i in range(min(top_n, len(feature_names))):
                idx = sorted_idx[i]
                log.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    except Exception as e:
        log.warning(f"Could not extract feature importance for {target_name}: {e}")


def _plot_training_metrics(report: Dict[str, Any], out_dir: Path):
    """Generate a compact training metrics chart when matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning(f"Skipping training metrics plot because matplotlib is unavailable: {e}")
        return

    try:
        metrics = report.get("metrics", {})
        reg = metrics.get("regression", {})
        cls = metrics.get("classification", {})

        plt.figure(figsize=(10, 6))
        labels = ["Combined MAE", "Brier Score"]
        values = [reg.get("combined_mae"), cls.get("brier")]
        valid = [(l, v) for l, v in zip(labels, values) if v is not None]
        if not valid:
            plt.close()
            return
        l_final, v_final = zip(*valid)
        plt.bar(l_final, v_final, color=["skyblue", "salmon"])
        generated_at = str(report.get("generated_at", ""))[:10]
        plt.title(f"Training Performance Summary - {generated_at}")
        plt.ylabel("Error Metric (Lower is Better)")
        plot_path = out_dir / "training_metrics_plot.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Training metrics plot saved to {plot_path}")
    except Exception as e:
        logging.warning(f"Failed to generate training plot: {e}")

def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("train_models")


def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_dataset_path(explicit: Optional[str], log: logging.Logger) -> Path:
    env_override = os.getenv("TRAIN_DATASET_FILE")
    candidate = explicit or env_override
    try:
        path = resolve_latest_dataset(DATA_DIR, explicit_path=candidate)
    except FileNotFoundError:
        if candidate:
            log.warning(
                "Requested dataset path not found (%s). Falling back to latest game_features*.csv.",
                candidate,
            )
        path = resolve_latest_dataset(DATA_DIR, explicit_path=None)
    return path.resolve()


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {missing}")


def _safe_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2 compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", _safe_onehot()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        sparse_threshold=0.0,
    )


def _drop_leaky_columns(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    to_drop: List[str] = []
    blocked = set(LEAK_BLOCKLIST) | set(ID_COLUMNS)
    for col in df.columns:
        if col in blocked:
            to_drop.append(col)
            continue
        if isinstance(col, str) and col.startswith("_"):
            to_drop.append(col)
    if to_drop:
        log.info("Dropping non-feature columns: %s", sorted(set(to_drop)))
        return df.drop(columns=sorted(set(to_drop)), errors="ignore")
    return df


def _coerce_binary_label(y: pd.Series) -> pd.Series:
    if y.dtype == bool:
        return y.astype(int)
    if pd.api.types.is_numeric_dtype(y):
        coerced = pd.to_numeric(y, errors="coerce")
        return coerced.round().clip(0, 1).astype("Int64")
    normalized = y.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y", "home", "win"}
    mapped = normalized.map(lambda x: 1 if x in truthy else 0)
    return mapped.astype("Int64")


def _infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    ignore = set(LEAK_BLOCKLIST) | {TARGET_HOME, TARGET_AWAY, TARGET_WIN}
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in df.columns:
        if col in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _make_group_labels(df: pd.DataFrame) -> np.ndarray:
    seasons = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    weeks = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    return np.asarray(
        [
            f"{int(season)}-{int(week)}" if pd.notna(season) and pd.notna(week) else f"row-{idx}"
            for idx, (season, week) in enumerate(zip(seasons, weeks))
        ],
        dtype=object,
    )


def _ordered_unique_groups(group_labels: Sequence[Any]) -> List[Any]:
    ordered: List[Any] = []
    seen: set[Any] = set()
    for label in group_labels:
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def _group_time_series_splits(
    group_labels: Sequence[Any],
    requested_splits: int,
    *,
    embargo_groups: int = 1,
    min_train_groups: int = 2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(list(group_labels), dtype=object)
    ordered_groups = _ordered_unique_groups(labels)
    if len(ordered_groups) < (min_train_groups + max(0, embargo_groups) + 1):
        return []

    max_splits = min(int(requested_splits), len(ordered_groups) - 1)
    if max_splits < 2:
        return []

    group_index = np.arange(len(ordered_groups), dtype=int)
    splitter = TimeSeriesSplit(n_splits=max_splits)
    ordered_arr = np.asarray(ordered_groups, dtype=object)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_group_idx, val_group_idx in splitter.split(group_index):
        if embargo_groups > 0:
            if len(train_group_idx) <= embargo_groups:
                continue
            train_group_idx = train_group_idx[:-embargo_groups]
        if len(train_group_idx) < min_train_groups or len(val_group_idx) == 0:
            continue

        train_groups = ordered_arr[train_group_idx]
        val_groups = ordered_arr[val_group_idx]
        train_idx = np.flatnonzero(np.isin(labels, train_groups))
        val_idx = np.flatnonzero(np.isin(labels, val_groups))
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def _split_train_holdout_indices(
    group_labels: Sequence[Any],
    *,
    holdout_ratio: float,
    embargo_groups: int = 1,
    min_train_groups: int = 4,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    labels = np.asarray(list(group_labels), dtype=object)
    ordered_groups = _ordered_unique_groups(labels)
    required_groups = min_train_groups + max(0, embargo_groups) + 1
    if len(ordered_groups) < required_groups:
        raise RuntimeError(
            f"Not enough (season, week) groups for grouped holdout splitting: {len(ordered_groups)} < {required_groups}."
        )

    target_holdout_rows = max(20, int(len(labels) * float(holdout_ratio)))
    group_counts = {group: int(np.sum(labels == group)) for group in ordered_groups}

    holdout_groups: List[Any] = []
    holdout_rows = 0
    for idx in range(len(ordered_groups) - 1, -1, -1):
        if idx < (min_train_groups + max(0, embargo_groups)):
            break
        group = ordered_groups[idx]
        holdout_groups.append(group)
        holdout_rows += group_counts[group]
        if holdout_rows >= target_holdout_rows:
            break

    holdout_groups = list(reversed(holdout_groups))
    if not holdout_groups:
        holdout_groups = [ordered_groups[-1]]

    holdout_start = ordered_groups.index(holdout_groups[0])
    train_group_end = holdout_start - max(0, embargo_groups)
    if train_group_end < min_train_groups:
        raise RuntimeError("Grouped holdout split would leave too few training groups after embargo.")

    train_groups = ordered_groups[:train_group_end]
    holdout_groups = ordered_groups[holdout_start:]
    embargo_slice = ordered_groups[train_group_end:holdout_start]

    train_idx = np.flatnonzero(np.isin(labels, np.asarray(train_groups, dtype=object)))
    holdout_idx = np.flatnonzero(np.isin(labels, np.asarray(holdout_groups, dtype=object)))
    if train_idx.size == 0 or holdout_idx.size == 0:
        raise RuntimeError("Grouped holdout split produced an empty train or holdout partition.")

    info = {
        "group_key_columns": list(TIME_KEYS),
        "train_groups": [str(group) for group in train_groups],
        "holdout_groups": [str(group) for group in holdout_groups],
        "embargo_groups": [str(group) for group in embargo_slice],
        "requested_holdout_ratio": float(holdout_ratio),
    }
    return train_idx, holdout_idx, info


def _fit_regressor(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    numeric_cols: List[str],
    categorical_cols: List[str],
    group_labels: Sequence[Any],
    random_seed: int,
    hp_n_iter: int,
    cv_splits: int,
    embargo_groups: int,
    n_jobs: int,
    fast_dev: bool,
) -> Tuple[HistGradientBoostingRegressor, Dict[str, Any]]:
    base = HistGradientBoostingRegressor(random_state=random_seed)
    grouped_splits = _group_time_series_splits(
        group_labels,
        cv_splits,
        embargo_groups=embargo_groups,
    )
    if fast_dev or len(grouped_splits) < 2:
        return clone(base), {
            "mode": "single_fit",
            "random_state": random_seed,
            "cv_strategy": "group_time_series",
            "cv_folds": int(len(grouped_splits)),
            "embargo_groups": int(embargo_groups),
        }

    search_pipeline = Pipeline(
        steps=[
            ("pre", _make_preprocessor(numeric_cols, categorical_cols)),
            ("reg", base),
        ]
    )

    rs = RandomizedSearchCV(
        estimator=search_pipeline,
        param_distributions={f"reg__{key}": values for key, values in REG_PARAM_DISTS.items()},
        n_iter=max(1, int(hp_n_iter)),
        cv=grouped_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
        random_state=random_seed,
        refit=True,
        verbose=1,
        error_score="raise",
    )
    rs.fit(X, y)
    best_regressor = clone(rs.best_estimator_.named_steps["reg"])
    best_params = {
        key.replace("reg__", "", 1): value for key, value in dict(rs.best_params_).items()
    }
    return best_regressor, {
        "mode": "search",
        "best_params": best_params,
        "cv_strategy": "group_time_series",
        "cv_folds": int(len(grouped_splits)),
        "embargo_groups": int(embargo_groups),
    }


def _fit_classifier_base(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    numeric_cols: List[str],
    categorical_cols: List[str],
    group_labels: Sequence[Any],
    random_seed: int,
    hp_n_iter: int,
    cv_splits: int,
    embargo_groups: int,
    n_jobs: int,
    fast_dev: bool,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    base = MLPClassifier(
        random_state=random_seed,
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        batch_size=64,
        solver="adam",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        max_iter=500,
    )
    grouped_splits = _group_time_series_splits(
        group_labels,
        cv_splits,
        embargo_groups=embargo_groups,
    )
    if fast_dev or len(grouped_splits) < 2:
        return clone(base), {
            "mode": "single_fit",
            "random_state": random_seed,
            "algorithm": "mlp",
            "cv_strategy": "group_time_series",
            "cv_folds": int(len(grouped_splits)),
            "embargo_groups": int(embargo_groups),
        }
    search_pipeline = Pipeline(
        steps=[
            ("pre", _make_preprocessor(numeric_cols, categorical_cols)),
            ("clf", base),
        ]
    )

    rs = RandomizedSearchCV(
        estimator=search_pipeline,
        param_distributions={f"clf__{key}": values for key, values in CLF_PARAM_DISTS.items()},
        n_iter=max(1, int(hp_n_iter)),
        cv=grouped_splits,
        scoring="neg_log_loss",
        n_jobs=n_jobs,
        random_state=random_seed,
        refit=True,
        verbose=1,
        error_score="raise",
    )
    rs.fit(X, y)
    best_classifier = clone(rs.best_estimator_.named_steps["clf"])
    return best_classifier, {
        "mode": "search",
        "best_params": {key.replace("clf__", "", 1): value for key, value in dict(rs.best_params_).items()},
        "algorithm": "mlp",
        "cv_strategy": "group_time_series",
        "cv_folds": int(len(grouped_splits)),
        "embargo_groups": int(embargo_groups),
    }


def _clone_classifier_for_fit(
    base_clf: BaseEstimator,
    y_fit: np.ndarray,
    *,
    disable_early_stopping: bool = False,
) -> BaseEstimator:
    """Disable MLP early stopping when a small fold cannot support stratified splitting."""
    fitted = clone(base_clf)
    if isinstance(fitted, MLPClassifier):
        class_counts = np.bincount(np.asarray(y_fit, dtype=int), minlength=2)
        if disable_early_stopping or len(y_fit) < 25 or int(class_counts.min()) < 2:
            fitted.set_params(early_stopping=False)
    return fitted


def _make_calibrator(
    estimator: BaseEstimator,
    *,
    cv: Any,
) -> CalibratedClassifierCV:
    if cv == "prefit":
        if FrozenEstimator is not None:
            return CalibratedClassifierCV(
                estimator=FrozenEstimator(estimator),
                method="sigmoid",
            )
        try:
            return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
        except TypeError:
            return CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=cv)
    try:
        return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=cv)


def _calibrate_classifier(
    base_clf: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    min_class = int(class_counts.min())

    # Chronology-safe prefit calibration on the tail of train data, when possible.
    calib_size = max(120, int(len(y_train) * 0.2))
    if (
        len(y_train) > (calib_size + 150)
        and min_class > 1
        and len(np.unique(y_train[-calib_size:])) == 2
    ):
        fit_end = len(y_train) - calib_size
        prefitted = _clone_classifier_for_fit(
            base_clf,
            y_train[:fit_end],
            disable_early_stopping=True,
        ).fit(
            X_train[:fit_end],
            y_train[:fit_end],
        )
        calibrated = _make_calibrator(prefitted, cv="prefit")
        calibrated.fit(X_train[fit_end:], y_train[fit_end:])
        return calibrated, {
            "mode": "prefit_tail",
            "fit_rows": int(fit_end),
            "calibration_rows": int(calib_size),
            "algorithm": type(base_clf).__name__,
        }

    # Fallback CV calibration when class counts support it.
    if min_class >= 3:
        cv = 3
    elif min_class >= 2:
        cv = 2
    else:
        cv = None

    if cv is None:
        fitted = _clone_classifier_for_fit(
            base_clf,
            y_train,
            disable_early_stopping=True,
        ).fit(X_train, y_train)
        return fitted, {
            "mode": "uncalibrated",
            "reason": "insufficient_minority_class_examples",
            "class_counts": class_counts.tolist(),
            "algorithm": type(base_clf).__name__,
        }

    calibrated = _make_calibrator(
        _clone_classifier_for_fit(
            base_clf,
            y_train,
            disable_early_stopping=True,
        ),
        cv=cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated, {
        "mode": "cv",
        "cv_folds": int(cv),
        "class_counts": class_counts.tolist(),
        "algorithm": type(base_clf).__name__,
    }


def _predict_positive_class_proba(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    if raw.ndim != 2:
        raise ValueError("predict_proba must return a 2D matrix.")
    if raw.shape[1] == 1:
        return np.clip(raw[:, 0].astype(float), 1e-6, 1 - 1e-6)

    classes = getattr(model, "classes_", None)
    if classes is not None:
        class_arr = np.asarray(classes)
        positive_matches = np.where(class_arr == 1)[0]
        positive_idx = int(positive_matches[0]) if len(positive_matches) else int(len(class_arr) - 1)
    else:
        positive_idx = 1

    return np.clip(raw[:, positive_idx].astype(float), 1e-6, 1 - 1e-6)


def _prior_home_win_probabilities(y_win: np.ndarray, neutral_prob: float = 0.5) -> np.ndarray:
    """Baseline chronology-safe probabilities using only prior outcomes."""
    y = np.asarray(y_win, dtype=float)
    if y.size == 0:
        return np.asarray([], dtype=float)

    probs = np.full(y.shape[0], float(neutral_prob), dtype=float)
    if y.shape[0] == 1:
        return probs

    cumulative_wins = np.cumsum(y[:-1])
    counts = np.arange(1, y.shape[0], dtype=float)
    probs[1:] = cumulative_wins / counts
    return np.clip(probs, 1e-6, 1 - 1e-6)


def _fallback_home_win_probabilities(
    X: pd.DataFrame,
    *,
    neutral_prob: float = 0.5,
) -> np.ndarray:
    """Fallback score-stack probabilities from market priors, else a neutral prior."""
    probs = np.full(len(X), float(neutral_prob), dtype=float)
    if "home_moneyline_prob" not in X.columns or len(X) == 0:
        return np.clip(probs, 1e-6, 1 - 1e-6)

    moneyline = pd.to_numeric(X["home_moneyline_prob"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(moneyline)
    if np.any(valid):
        probs[valid] = moneyline[valid]
    return np.clip(probs, 1e-6, 1 - 1e-6)


def _generate_stacked_train_probabilities(
    X: pd.DataFrame,
    y_win: np.ndarray,
    *,
    tuned_estimator: BaseEstimator,
    numeric_cols: List[str],
    categorical_cols: List[str],
    group_labels: Sequence[Any],
    cv_splits: int,
    embargo_groups: int,
    fallback_probabilities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate chronology-safe win probabilities for score-model training."""
    if fallback_probabilities is None:
        probs = np.full(len(y_win), 0.5, dtype=float)
        fallback_mode = "neutral"
    else:
        probs = np.clip(np.asarray(fallback_probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if len(probs) != len(y_win):
            raise ValueError("Fallback probability length must match y_win.")
        fallback_mode = "home_moneyline_prob_or_neutral"
    grouped_splits = _group_time_series_splits(
        group_labels,
        cv_splits,
        embargo_groups=embargo_groups,
    )
    if not grouped_splits:
        return probs, {
            "mode": "fallback_only",
            "coverage": 0.0,
            "predicted_rows": 0,
            "total_rows": int(len(y_win)),
            "fallback_mode": fallback_mode,
            "cv_strategy": "group_time_series",
            "cv_folds": 0,
            "embargo_groups": int(embargo_groups),
        }

    predicted_rows = 0
    fold_summaries: List[Dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(grouped_splits, start=1):
        y_fold = y_win[train_idx]
        if len(np.unique(y_fold)) < 2:
            fold_summaries.append(
                {
                    "fold": fold_idx,
                    "train_rows": int(len(train_idx)),
                    "val_rows": int(len(val_idx)),
                    "mode": "prior_fallback",
                }
            )
            continue

        fold_preprocessor = _make_preprocessor(numeric_cols, categorical_cols)
        X_fold_train = X.iloc[train_idx].reset_index(drop=True)
        X_fold_val = X.iloc[val_idx].reset_index(drop=True)
        X_fold_train_proc = np.asarray(fold_preprocessor.fit_transform(X_fold_train))
        X_fold_val_proc = np.asarray(fold_preprocessor.transform(X_fold_val))
        fold_clf, fold_calibration = _calibrate_classifier(
            clone(tuned_estimator),
            X_fold_train_proc,
            y_fold,
        )
        fold_probs = _predict_positive_class_proba(fold_clf, X_fold_val_proc)
        probs[val_idx] = fold_probs
        predicted_rows += len(val_idx)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "mode": str(fold_calibration.get("mode", "unknown")),
            }
        )

    return probs, {
        "mode": "time_series_oof",
        "coverage": float(predicted_rows / max(1, len(y_win))),
        "predicted_rows": int(predicted_rows),
        "total_rows": int(len(y_win)),
        "fallback_mode": fallback_mode,
        "cv_strategy": "group_time_series",
        "cv_folds": int(len(grouped_splits)),
        "embargo_groups": int(embargo_groups),
        "folds": fold_summaries,
    }


def _augment_score_features(X: pd.DataFrame, win_prob: np.ndarray) -> pd.DataFrame:
    """Append the raw home-win probability feature used by score regressors."""
    X_aug = X.copy()
    clipped = np.clip(np.asarray(win_prob, dtype=float), 1e-6, 1 - 1e-6)
    if len(X_aug) != len(clipped):
        raise ValueError("Score feature augmentation length mismatch.")
    X_aug[WIN_PROBA_FEATURE] = clipped
    return X_aug


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2: Optional[float]
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = None
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2)


def _compute_classifier_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ClassificationMetrics:
    probs = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    pred = (probs >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_true, pred))
    brier = float(brier_score_loss(y_true, probs))

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, probs))
        ll = float(log_loss(y_true, probs, labels=[0, 1]))
    else:
        roc_auc = None
        ll = None

    return ClassificationMetrics(
        accuracy=accuracy,
        brier=brier,
        roc_auc=roc_auc,
        log_loss=ll,
    )


def _load_previous_report(out_dir: Path) -> Optional[Dict[str, Any]]:
    path = out_dir / "training_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_gate_metrics(report: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    cls = report.get("metrics", {}).get("classification", {})
    reg = report.get("metrics", {}).get("regression", {})
    brier = cls.get("brier")
    combined_mae = reg.get("combined_mae")
    try:
        brier_f = None if brier is None else float(brier)
    except Exception:
        brier_f = None
    try:
        mae_f = None if combined_mae is None else float(combined_mae)
    except Exception:
        mae_f = None
    return brier_f, mae_f


def _gate_result(
    *,
    current_report: Dict[str, Any],
    previous_report: Optional[Dict[str, Any]],
    max_brier_delta: float,
    max_mae_delta: float,
    disable_gate: bool,
) -> Dict[str, Any]:
    if disable_gate:
        return {"enabled": False, "passed": True, "reason": "gate disabled"}

    curr_brier, curr_mae = _extract_gate_metrics(current_report)
    prev_brier: Optional[float] = None
    prev_mae: Optional[float] = None

    if previous_report is not None:
        prev_brier, prev_mae = _extract_gate_metrics(previous_report)

    failures: List[str] = []
    if curr_brier is None or curr_mae is None:
        failures.append("missing current metrics")

    if prev_brier is not None and curr_brier is not None:
        if curr_brier > (prev_brier + max_brier_delta):
            failures.append(
                f"brier regression: current={curr_brier:.4f} previous={prev_brier:.4f} "
                f"max_delta={max_brier_delta:.4f}"
            )

    if prev_mae is not None and curr_mae is not None:
        if curr_mae > (prev_mae + max_mae_delta):
            failures.append(
                f"mae regression: current={curr_mae:.4f} previous={prev_mae:.4f} "
                f"max_delta={max_mae_delta:.4f}"
            )

    return {
        "enabled": True,
        "passed": len(failures) == 0,
        "failures": failures,
        "current": {"brier": curr_brier, "combined_mae": curr_mae},
        "previous": {"brier": prev_brier, "combined_mae": prev_mae},
        "thresholds": {
            "max_brier_delta": float(max_brier_delta),
            "max_mae_delta": float(max_mae_delta),
        },
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stage_bundle(
    *,
    stage_dir: Path,
    score_preprocessor: ColumnTransformer,
    win_preprocessor: ColumnTransformer,
    home_reg: HistGradientBoostingRegressor,
    away_reg: HistGradientBoostingRegressor,
    win_clf: BaseEstimator,
    metadata: Dict[str, Any],
    report: Dict[str, Any],
) -> List[str]:
    stage_dir.mkdir(parents=True, exist_ok=True)

    home_pipe = Pipeline([("pre", score_preprocessor), ("reg", home_reg)])
    away_pipe = Pipeline([("pre", score_preprocessor), ("reg", away_reg)])
    win_pipe = Pipeline([("pre", win_preprocessor), ("clf", win_clf)])

    dump(score_preprocessor, stage_dir / "preprocessor.joblib")
    dump(score_preprocessor, stage_dir / "score_preprocessor.joblib")
    dump(win_preprocessor, stage_dir / "win_preprocessor.joblib")
    dump(home_reg, stage_dir / "home_model.joblib")
    dump(away_reg, stage_dir / "away_model.joblib")
    dump(win_clf, stage_dir / "win_clf_calibrated.joblib")
    dump(home_pipe, stage_dir / "home_pipe.joblib")
    dump(away_pipe, stage_dir / "away_pipe.joblib")
    dump(win_pipe, stage_dir / "win_pipe.joblib")

    _write_json(stage_dir / "metadata.json", metadata)
    _write_json(stage_dir / "training_report.json", report)

    return [
        "preprocessor.joblib",
        "score_preprocessor.joblib",
        "win_preprocessor.joblib",
        "home_model.joblib",
        "away_model.joblib",
        "win_clf_calibrated.joblib",
        "home_pipe.joblib",
        "away_pipe.joblib",
        "win_pipe.joblib",
        "metadata.json",
        "training_report.json",
    ]


def _promote_stage(stage_dir: Path, out_dir: Path, files: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = stage_dir / name
        if not src.exists():
            continue
        tmp = out_dir / f".{name}.tmp"
        shutil.copy2(src, tmp)
        tmp.replace(out_dir / name)
    # Ensure runtime model discovery that sorts by directory mtime selects this bundle.
    os.utime(out_dir, None)


def _mirror_to_dated_bundle(out_dir: Path, files: List[str], run_date_tag: str) -> Optional[Path]:
    """
    Mirror promoted artifacts into backend/YYYYMMDD/models to match runtime discovery logic.
    """
    if out_dir.resolve() != DEFAULT_MODELS_DIR.resolve():
        return None
    dated_dir = BASE_DIR / run_date_tag / "models"
    dated_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = out_dir / name
        if not src.exists():
            continue
        shutil.copy2(src, dated_dir / name)
    os.utime(dated_dir, None)
    return dated_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NFL score/win models with leak-safe preprocessing and promotion gating."
    )
    parser.add_argument("--data", type=str, default=None, help="Path to training CSV.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_MODELS_DIR.resolve()),
        help="Output models directory.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_safe_int_env("N_JOBS", -1),
        help="Parallel jobs used by randomized search.",
    )
    parser.add_argument(
        "--hp-niter",
        type=int,
        default=_safe_int_env("HP_N_ITER", 30),
        help="Number of randomized search iterations per model.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=_safe_int_env("CV_SPLITS", 5),
        help="Expanding grouped (season, week) folds for randomized search.",
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=_safe_int_env("EMBARGO_GROUPS", 1),
        help="Number of trailing (season, week) groups excluded between train and validation windows.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=_safe_int_env("RANDOM_SEED", 42),
        help="Random seed for model training.",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Tail holdout ratio used for evaluation/gating.",
    )
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="Skip hyperparameter search for quick local iterations.",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="After evaluation, refit chosen models on the full labeled dataset.",
    )
    parser.add_argument(
        "--bundle-version",
        type=str,
        default=os.getenv("BUNDLE_VERSION", "v1"),
        help="Bundle version tag written to metadata.",
    )
    parser.add_argument(
        "--max-brier-delta",
        type=float,
        default=0.01,
        help="Maximum allowed Brier regression vs previous training report.",
    )
    parser.add_argument(
        "--max-mae-delta",
        type=float,
        default=0.5,
        help="Maximum allowed combined MAE regression vs previous report.",
    )
    parser.add_argument(
        "--disable-gate",
        action="store_true",
        help="Disable quality gate checks against previous report.",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Promote staged artifacts even if gate fails.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Write staged artifacts but do not promote to current.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = _setup_logging()
    start_ts = time.time()

    dataset_path = _resolve_dataset_path(args.data, log)
    out_dir = Path(args.out).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_dir = out_dir / "staging" / run_id

    log.info("Using dataset: %s", dataset_path)
    df = pd.read_csv(dataset_path)
    _ensure_required_columns(df)
    if df.empty:
        raise RuntimeError(f"Training dataset is empty: {dataset_path}")

    for col in TIME_KEYS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(TIME_KEYS)).copy()
    df[list(TIME_KEYS)] = df[list(TIME_KEYS)].astype(int)
    df = df.sort_values(list(TIME_KEYS)).reset_index(drop=True)

    y_home = pd.to_numeric(df[TARGET_HOME], errors="coerce")
    y_away = pd.to_numeric(df[TARGET_AWAY], errors="coerce")
    y_win = _coerce_binary_label(df[TARGET_WIN])

    keep_mask = y_home.notna() & y_away.notna() & y_win.notna()
    dropped = int((~keep_mask).sum())
    if dropped:
        log.info("Dropped %d rows with invalid target labels.", dropped)

    df = df.loc[keep_mask].reset_index(drop=True)
    y_home = y_home.loc[keep_mask].astype(float).reset_index(drop=True)
    y_away = y_away.loc[keep_mask].astype(float).reset_index(drop=True)
    y_win = y_win.loc[keep_mask].astype(int).reset_index(drop=True)

    if len(df) < 80:
        raise RuntimeError(
            f"Dataset too small for reliable production training (rows={len(df)}). Need at least 80 rows."
        )

    feature_df = _drop_leaky_columns(df.copy(), log)
    numeric_cols, categorical_cols = _infer_feature_columns(feature_df)
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        raise RuntimeError("No usable feature columns were inferred after leak filtering.")

    X = feature_df[feature_cols].copy()
    group_labels = _make_group_labels(df.loc[:, list(TIME_KEYS)])
    holdout_ratio = float(np.clip(args.holdout_ratio, 0.05, 0.4))
    train_idx, holdout_idx, holdout_split_info = _split_train_holdout_indices(
        group_labels,
        holdout_ratio=holdout_ratio,
        embargo_groups=int(args.embargo),
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_holdout = X.iloc[holdout_idx].reset_index(drop=True)
    y_home_train = y_home.iloc[train_idx].to_numpy()
    y_home_hold = y_home.iloc[holdout_idx].to_numpy()
    y_away_train = y_away.iloc[train_idx].to_numpy()
    y_away_hold = y_away.iloc[holdout_idx].to_numpy()
    y_win_train = y_win.iloc[train_idx].to_numpy()
    y_win_hold = y_win.iloc[holdout_idx].to_numpy()
    train_group_labels = group_labels[train_idx]
    holdout_group_labels = group_labels[holdout_idx]

    log.info(
        "Rows total=%d train=%d holdout=%d features=%d train_groups=%d holdout_groups=%d embargo_groups=%d",
        len(X),
        len(X_train),
        len(X_holdout),
        len(feature_cols),
        len(_ordered_unique_groups(train_group_labels)),
        len(_ordered_unique_groups(holdout_group_labels)),
        int(args.embargo),
    )

    win_base, win_train_info = _fit_classifier_base(
        X_train,
        y_win_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        group_labels=train_group_labels,
        random_seed=args.random_seed,
        hp_n_iter=args.hp_niter,
        cv_splits=max(2, int(args.splits)),
        embargo_groups=int(args.embargo),
        n_jobs=int(args.n_jobs),
        fast_dev=bool(args.fast_dev),
    )
    win_preprocessor = _make_preprocessor(numeric_cols, categorical_cols)
    win_preprocessor.fit(X_train)
    X_train_proc = np.asarray(win_preprocessor.transform(X_train))
    X_hold_proc = np.asarray(win_preprocessor.transform(X_holdout))
    win_clf, win_calibration_info = _calibrate_classifier(
        win_base,
        X_train_proc,
        y_win_train,
    )

    fallback_train_win_prob = _fallback_home_win_probabilities(X_train)
    train_win_prob, train_stack_info = _generate_stacked_train_probabilities(
        X_train,
        y_win_train,
        tuned_estimator=win_base,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        group_labels=train_group_labels,
        cv_splits=max(2, int(args.splits)),
        embargo_groups=int(args.embargo),
        fallback_probabilities=fallback_train_win_prob,
    )
    holdout_win_prob = _predict_positive_class_proba(win_clf, X_hold_proc)

    X_train_score = _augment_score_features(X_train, train_win_prob)
    X_holdout_score = _augment_score_features(X_holdout, holdout_win_prob)
    score_numeric_cols = numeric_cols + [WIN_PROBA_FEATURE]
    score_feature_cols = score_numeric_cols + categorical_cols

    home_reg, home_train_info = _fit_regressor(
        X_train_score,
        y_home_train,
        numeric_cols=score_numeric_cols,
        categorical_cols=categorical_cols,
        group_labels=train_group_labels,
        random_seed=args.random_seed,
        hp_n_iter=args.hp_niter,
        cv_splits=max(2, int(args.splits)),
        embargo_groups=int(args.embargo),
        n_jobs=int(args.n_jobs),
        fast_dev=bool(args.fast_dev),
    )
    away_reg, away_train_info = _fit_regressor(
        X_train_score,
        y_away_train,
        numeric_cols=score_numeric_cols,
        categorical_cols=categorical_cols,
        group_labels=train_group_labels,
        random_seed=args.random_seed,
        hp_n_iter=args.hp_niter,
        cv_splits=max(2, int(args.splits)),
        embargo_groups=int(args.embargo),
        n_jobs=int(args.n_jobs),
        fast_dev=bool(args.fast_dev),
    )
    score_preprocessor = _make_preprocessor(score_numeric_cols, categorical_cols)
    score_preprocessor.fit(X_train_score)
    X_train_score_proc = np.asarray(score_preprocessor.transform(X_train_score))
    X_hold_score_proc = np.asarray(score_preprocessor.transform(X_holdout_score))

    home_reg = clone(home_reg).fit(X_train_score_proc, y_home_train)
    away_reg = clone(away_reg).fit(X_train_score_proc, y_away_train)
    try:
        score_model_feature_names = list(score_preprocessor.get_feature_names_out())
    except Exception:
        score_model_feature_names = [f"feature_{idx}" for idx in range(X_train_score_proc.shape[1])]
    _log_feature_importance(home_reg, score_model_feature_names, "home_score", log)
    _log_feature_importance(away_reg, score_model_feature_names, "away_score", log)

    home_pred = home_reg.predict(X_hold_score_proc)
    away_pred = away_reg.predict(X_hold_score_proc)
    win_prob = holdout_win_prob

    home_metrics = _compute_regression_metrics(y_home_hold, home_pred)
    away_metrics = _compute_regression_metrics(y_away_hold, away_pred)
    cls_metrics = _compute_classifier_metrics(y_win_hold, win_prob)
    combined_mae = float((home_metrics.mae + away_metrics.mae) / 2.0)

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_hash": file_sha256(dataset_path),
        "rows": {
            "total": int(len(X)),
            "train": int(len(X_train)),
            "holdout": int(len(X_holdout)),
            "embargo_excluded": int(len(X) - len(X_train) - len(X_holdout)),
        },
        "features": {
            "win": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "count": int(len(feature_cols)),
            },
            "score": {
                "numeric": score_numeric_cols,
                "categorical": categorical_cols,
                "count": int(len(score_feature_cols)),
            },
            "generated": [WIN_PROBA_FEATURE],
        },
        "metrics": {
            "regression": {
                "home": asdict(home_metrics),
                "away": asdict(away_metrics),
                "combined_mae": combined_mae,
            },
            "classification": asdict(cls_metrics),
        },
        "train_info": {
            "home": home_train_info,
            "away": away_train_info,
            "win_base": win_train_info,
            "win_calibration": win_calibration_info,
            "score_stack": train_stack_info,
            "holdout_split": holdout_split_info,
            "cv_splits": int(args.splits),
            "embargo_groups": int(args.embargo),
            "hp_niter": int(args.hp_niter),
            "fast_dev": bool(args.fast_dev),
        },
    }

    # Optional production refit on full labeled data after quality evaluation.
    if args.production:
        log.info("Production mode enabled: refitting selected models on full labeled dataset.")
        win_preprocessor = _make_preprocessor(numeric_cols, categorical_cols)
        win_preprocessor.fit(X)
        X_full_proc = np.asarray(win_preprocessor.transform(X))
        win_clf, win_calibration_info = _calibrate_classifier(
            clone(win_base),
            X_full_proc,
            y_win.to_numpy(),
        )
        fallback_full_win_prob = _fallback_home_win_probabilities(X)
        full_train_win_prob, full_stack_info = _generate_stacked_train_probabilities(
            X,
            y_win.to_numpy(),
            tuned_estimator=win_base,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            group_labels=group_labels,
            cv_splits=max(2, int(args.splits)),
            embargo_groups=int(args.embargo),
            fallback_probabilities=fallback_full_win_prob,
        )
        X_full_score = _augment_score_features(X, full_train_win_prob)
        score_preprocessor = _make_preprocessor(score_numeric_cols, categorical_cols)
        score_preprocessor.fit(X_full_score)
        X_full_score_proc = np.asarray(score_preprocessor.transform(X_full_score))
        home_reg = clone(home_reg).fit(X_full_score_proc, y_home.to_numpy())
        away_reg = clone(away_reg).fit(X_full_score_proc, y_away.to_numpy())
        report["train_info"]["win_calibration_full"] = win_calibration_info
        report["train_info"]["score_stack_full"] = full_stack_info

    previous_report = _load_previous_report(out_dir)
    gate = _gate_result(
        current_report=report,
        previous_report=previous_report,
        max_brier_delta=float(args.max_brier_delta),
        max_mae_delta=float(args.max_mae_delta),
        disable_gate=bool(args.disable_gate),
    )

    bundle_timestamp = datetime.now(timezone.utc).isoformat()
    metadata: Dict[str, Any] = {
        "timestamp": bundle_timestamp,
        "training_timestamp_utc": bundle_timestamp,
        "bundle_timestamp_utc": bundle_timestamp,
        "bundle_version": str(args.bundle_version),
        "bundle_contract_version": 2,
        "sklearn_version": sklearn.__version__,
        "training_script": "backend/train_models.py",
        "dataset_path": str(dataset_path),
        "dataset_hash": report["dataset_hash"],
        "rows_total": int(len(X)),
        "rows_train": int(len(X_train)),
        "rows_holdout": int(len(X_holdout)),
        "raw_feature_columns": {
            "win": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
            },
            "score": {
                "numeric": score_numeric_cols,
                "categorical": categorical_cols,
            },
        },
        "feature_names": score_feature_cols,
        "feature_names_win": feature_cols,
        "feature_manifests": {
            "win": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
            },
            "score": {
                "numeric": score_numeric_cols,
                "categorical": categorical_cols,
            },
            "scores": {
                "numeric": score_numeric_cols,
                "categorical": categorical_cols,
            },
        },
        "generated_features": {
            WIN_PROBA_FEATURE: {
                "source": "winner_model_predict_proba",
                "fallback_column": "home_moneyline_prob",
                "default": 0.5,
                "used_by": ["home", "away"],
            }
        },
        "serving_mode": "pipeline_primary",
        "targets": {
            "home": TARGET_HOME,
            "away": TARGET_AWAY,
            "win": TARGET_WIN,
        },
        "stacking": {
            "win_probability_feature": WIN_PROBA_FEATURE,
            "score_models_use_win_probability": True,
            "winner_model_algorithm": "calibrated_mlp",
        },
        "metrics": report["metrics"],
        "gate": gate,
        "artifacts": {
            "preprocessor": "score_preprocessor.joblib",
            "score_preprocessor": "score_preprocessor.joblib",
            "win_preprocessor": "win_preprocessor.joblib",
            "home_model": "home_pipe.joblib",
            "away_model": "away_pipe.joblib",
            "win_model": "win_pipe.joblib",
            "home_estimator": "home_model.joblib",
            "away_estimator": "away_model.joblib",
            "win_estimator": "win_clf_calibrated.joblib",
        },
        "bundle_contract": {
            "serving_mode": "pipeline_primary",
            "bundle_timestamp_utc": "bundle_timestamp_utc",
            "dataset_hash": "dataset_hash",
            "sklearn_version": "sklearn_version",
            "feature_manifests": "feature_manifests",
            "generated_features": "generated_features",
            "preprocessor": "score_preprocessor.joblib",
            "score_preprocessor": "score_preprocessor.joblib",
            "win_preprocessor": "win_preprocessor.joblib",
            "reg_home": "home_pipe.joblib",
            "reg_away": "away_pipe.joblib",
            "clf_home_win": "win_pipe.joblib",
            "legacy_reg_home": "home_model.joblib",
            "legacy_reg_away": "away_model.joblib",
            "legacy_clf_home_win": "win_clf_calibrated.joblib",
        },
    }

    promoted_files = _stage_bundle(
        stage_dir=stage_dir,
        score_preprocessor=score_preprocessor,
        win_preprocessor=win_preprocessor,
        home_reg=home_reg,
        away_reg=away_reg,
        win_clf=win_clf,
        metadata=metadata,
        report=report,
    )

    allow_promote = gate["passed"] or bool(args.force_promote)
    if args.no_promote:
        allow_promote = False

    dated_bundle_dir: Optional[Path] = None
    if allow_promote:
        _promote_stage(stage_dir, out_dir, promoted_files)
        dated_bundle_dir = _mirror_to_dated_bundle(
            out_dir=out_dir,
            files=promoted_files,
            run_date_tag=run_id[:8],
        )
        status = "PROMOTED"
    else:
        status = "STAGED_ONLY"

    summary_payload = {
        "status": status,
        "staging_dir": str(stage_dir),
        "models_dir": str(out_dir),
        "dated_bundle_dir": str(dated_bundle_dir) if dated_bundle_dir else None,
        "gate": gate,
        "duration_seconds": round(time.time() - start_ts, 2),
    }
    _write_json(stage_dir / "run_summary.json", summary_payload)
    if status == "PROMOTED":
        _write_json(out_dir / "run_summary.json", summary_payload)

    # Generate training metrics plot
    _plot_training_metrics(report, out_dir)

    log.info(
        "Training finished. status=%s gate_passed=%s stage=%s out=%s",
        status,
        gate.get("passed"),
        stage_dir,
        out_dir,
    )
    print(json.dumps(summary_payload, indent=2))

    if gate.get("passed") or args.disable_gate or args.force_promote:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
