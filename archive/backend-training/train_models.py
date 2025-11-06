#!/usr/bin/env python3
"""Bias-aware trainer for NFL score regressors and calibrated win classifier.

This module orchestrates model retraining for the FastAPI backend. It consumes
``backend/data/game_features.csv`` (emitted by :mod:`backend.build_dataset`) and
produces refreshed artifacts under ``backend/models``. The implementation
prioritises reproducibility, leakage control, and fairness-aware diagnostics:

- Deterministic random state across preprocessing, model initialisation, and CV.
- Time-aware cross-validation (``TimeSeriesSplit``) with a chronologically
  isolated validation fold provided by the dataset's ``split`` column.
- Unified preprocessing pipeline shared by both score regressors and the win
  classifier to keep feature handling consistent.
- Built-in bias probes (confusion matrix, classification report, calibration
  metrics) highlighting home/away prediction symmetry.
- Automatic metadata + report emission to track deltas versus prior training
  runs for easier operational auditing.

Run from repository root::

    python backend/train_models.py --data backend/data/game_features.csv         --models backend/models --random-state 42
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
N_SPLITS = 5
TARGET_HOME = "home_points_for"
TARGET_AWAY = "away_points_for"
CLASS_LABEL = "home_win"
TIME_KEYS = ["season", "week"]
ID_COLS = ["game_id", "home_team", "away_team"]
LEAK_BLOCKLIST = {
    TARGET_HOME,
    TARGET_AWAY,
    CLASS_LABEL,
    "point_diff",
    "home_score",
    "away_score",
    "home_win_prob",
    "away_win_prob",
    "winner",
}
REG_PARAM_DISTS = {
    "model__learning_rate": np.logspace(-2.3, -0.5, num=10),
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__max_leaf_nodes": [15, 31, 63, 127, 255],
    "model__min_samples_leaf": [3, 5, 8, 12, 20, 32],
    "model__l2_regularization": np.logspace(-2, 1, num=8),
}
CLF_PARAM_DISTS = {
    "model__C": np.logspace(-3, 2, num=12),
    "model__solver": ["lbfgs", "liblinear", "saga"],
    "model__penalty": ["l2"],
    "model__class_weight": ["balanced", None],
}


@dataclass
class TrainerConfig:
    """Runtime configuration for training."""

    data_path: Path
    models_dir: Path
    metrics_dir: Path
    random_state: int = RANDOM_SEED
    n_jobs: int = -1
    max_search_iter: int = 25

    @property
    def report_path(self) -> Path:
        return self.models_dir / "training_report.json"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "metadata.json"

    @property
    def preprocessor_path(self) -> Path:
        return self.models_dir / "preprocessor.joblib"

    @property
    def home_model_path(self) -> Path:
        return self.models_dir / "home_model.joblib"

    @property
    def away_model_path(self) -> Path:
        return self.models_dir / "away_model.joblib"

    @property
    def win_clf_path(self) -> Path:
        return self.models_dir / "win_clf_calibrated.joblib"


def load_dataset(cfg: TrainerConfig) -> pd.DataFrame:
    """Read dataset and validate mandatory columns."""

    df = pd.read_csv(cfg.data_path)
    required = set(TIME_KEYS + ID_COLS + [TARGET_HOME, TARGET_AWAY, CLASS_LABEL, "split"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty; run backend/build_dataset.py first.")

    df = df.sort_values(TIME_KEYS + ["game_id"]).reset_index(drop=True)
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that would leak targets or post-game outcomes."""

    cols = [c for c in df.columns if c not in LEAK_BLOCKLIST and not c.startswith("_")]
    return df[cols]


def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature names for preprocessing."""

    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in TIME_KEYS
        and not c.startswith("split")
        and c not in LEAK_BLOCKLIST
    ]
    categorical_cols = [c for c in ("home_team", "away_team", "game_type") if c in df.columns]
    return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: Iterable[str], categorical_cols: Iterable[str]) -> ColumnTransformer:
    """Build a reusable preprocessing pipeline for all estimators."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_cols)),
            ("cat", categorical_pipeline, list(categorical_cols)),
        ],
        remainder="drop",
        n_jobs=None,
    )


def _time_series_split(X: pd.DataFrame, n_splits: int) -> TimeSeriesSplit:
    """Utility wrapper to keep split logic centralised."""

    min_train_size = max(1, len(X) // (n_splits + 1))
    return TimeSeriesSplit(n_splits=n_splits, test_size=min_train_size)


def fit_regressor(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cfg: TrainerConfig,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Fit a HistGradientBoostingRegressor with randomized hyper-parameter search."""

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=cfg.random_state,
                    max_iter=500,
                    l2_regularization=1.0,
                    early_stopping=True,
                ),
            ),
        ]
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=REG_PARAM_DISTS,
        n_iter=cfg.max_search_iter,
        cv=_time_series_split(X, N_SPLITS),
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        scoring="neg_mean_absolute_error",
        verbose=0,
    )
    search.fit(X, y)

    best_model: Pipeline = search.best_estimator_
    preds = best_model.predict(X)
    metrics = {
        "search_score": float(search.best_score_),
        "mae_train": float(mean_absolute_error(y, preds)),
    }
    logging.info("%s regressor best params: %s", name, search.best_params_)
    return best_model, metrics


def fit_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cfg: TrainerConfig,
) -> Tuple[CalibratedClassifierCV, Dict[str, Any]]:
    """Train a class-weighted logistic regression with calibration."""

    base_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    random_state=cfg.random_state,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=CLF_PARAM_DISTS,
        n_iter=min(cfg.max_search_iter, 40),
        cv=_time_series_split(X, N_SPLITS),
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        scoring="roc_auc",
        verbose=0,
    )
    search.fit(X, y)

    calibrated = CalibratedClassifierCV(
        estimator=search.best_estimator_,
        method="isotonic",
        cv=3,
        n_jobs=cfg.n_jobs,
    )
    calibrated.fit(X, y)

    preds = calibrated.predict(X)
    proba = calibrated.predict_proba(X)[:, 1]
    proba_safe = np.clip(proba, 1e-6, 1 - 1e-6)  # Guards log_loss against 0/1 probabilities on newer sklearn builds.

    metrics = {
        "search_score": float(search.best_score_),
        "accuracy_train": float(accuracy_score(y, preds)),
        "balanced_accuracy_train": float(balanced_accuracy_score(y, preds)),
        "f1_macro_train": float(f1_score(y, preds, average="macro")),
        "brier_train": float(brier_score_loss(y, proba)),
        "logloss_train": float(log_loss(y, proba_safe)),
    }
    logging.info("Classifier best params: %s", search.best_params_)
    return calibrated, metrics


def evaluate_regressors(
    home_model: Pipeline,
    away_model: Pipeline,
    X_holdout: pd.DataFrame,
    y_home: pd.Series,
    y_away: pd.Series,
) -> Dict[str, Any]:
    """Compute MAE on holdout split for both score regressors."""

    home_preds = home_model.predict(X_holdout)
    away_preds = away_model.predict(X_holdout)
    return {
        "home_mae": float(mean_absolute_error(y_home, home_preds)),
        "away_mae": float(mean_absolute_error(y_away, away_preds)),
    }


def find_optimal_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Return the threshold that maximises macro-F1, keeping class recall parity in view."""

    best_threshold, best_score = 0.5, -np.inf
    for threshold in np.arange(step, 1.0, step):
        preds = (proba >= threshold).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score:
            best_threshold, best_score = threshold, score

    return best_threshold, best_score


def evaluate_classifier(
    clf: CalibratedClassifierCV,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> Dict[str, Any]:
    """Return fairness-oriented diagnostics for the win classifier."""

    proba = clf.predict_proba(X_holdout)[:, 1]
    threshold, threshold_metric = find_optimal_threshold(y_holdout, proba)
    proba_safe = np.clip(proba, 1e-6, 1 - 1e-6)  # Keep evaluation numerically stable for log-loss comparisons.
    preds = (proba >= threshold).astype(int)

    report = classification_report(y_holdout, preds, output_dict=True)
    cm = ConfusionMatrixDisplay.from_predictions(
        y_holdout,
        preds,
        display_labels=["Away win", "Home win"],
    )
    confusion = cm.confusion_matrix.tolist()
    tn, fp, fn, tp = cm.confusion_matrix.ravel()

    return {
        "accuracy": float(accuracy_score(y_holdout, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_holdout, preds)),
        "f1_macro": float(f1_score(y_holdout, preds, average="macro")),
        "roc_auc": float(roc_auc_score(y_holdout, proba)),
        "brier": float(brier_score_loss(y_holdout, proba)),
        "logloss": float(log_loss(y_holdout, proba_safe)),
        "confusion_matrix": confusion,
        "classification_report": report,
        "home_prediction_rate": float(preds.mean()),
        "threshold": float(threshold),
        "threshold_metric": float(threshold_metric),
        "youden_j": float((tp / (tp + fn + 1e-12)) - (fp / (fp + tn + 1e-12))),
    }


def score_based_classifier(
    home_model: Pipeline,
    away_model: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate probabilities from score differential using logistic mapping."""

    home_scores = home_model.predict(X)
    away_scores = away_model.predict(X)
    diff = home_scores - away_scores
    return 1 / (1 + np.exp(-0.35 * diff))


def load_previous_report(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_artifacts(
    cfg: TrainerConfig,
    preprocessor: ColumnTransformer,
    home_model: Pipeline,
    away_model: Pipeline,
    classifier: CalibratedClassifierCV,
) -> None:
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, cfg.preprocessor_path)
    joblib.dump(home_model, cfg.home_model_path)
    joblib.dump(away_model, cfg.away_model_path)
    joblib.dump(classifier, cfg.win_clf_path)
    logging.info("Artifacts persisted to %s", cfg.models_dir)


def write_report(cfg: TrainerConfig, report: Dict[str, Any]) -> None:
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    with cfg.report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logging.info("Training report written to %s", cfg.report_path)


def write_metadata(
    cfg: TrainerConfig,
    numeric_cols: List[str],
    categorical_cols: List[str],
    train_rows: int,
    holdout_rows: int,
    win_probability_threshold: float,
) -> None:
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "dataset": str(cfg.data_path),
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "win_probability_threshold": win_probability_threshold,
        "raw_feature_columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
        },
    }
    with cfg.metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logging.info("Metadata written to %s", cfg.metadata_path)


def train(cfg: TrainerConfig) -> Dict[str, Any]:
    """Execute the training workflow end-to-end and return aggregated metrics."""

    np.random.seed(cfg.random_state)

    raw = load_dataset(cfg)
    modeling_df = raw[raw["split"].isin(["train", "validation"])].copy()
    inference_rows = raw[raw["split"] == "inference"].shape[0]

    if modeling_df[CLASS_LABEL].isna().any():
        modeling_df = modeling_df.dropna(subset=[CLASS_LABEL, TARGET_HOME, TARGET_AWAY])

    logging.info("Loaded %d rows (%d inference rows held out).", len(raw), inference_rows)

    features_df = drop_leakage_columns(modeling_df)
    numeric_cols, categorical_cols = infer_feature_columns(features_df)
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    train_mask = modeling_df["split"] == "train"
    holdout_mask = modeling_df["split"] == "validation"

    X_train = features_df.loc[train_mask, :]
    X_holdout = features_df.loc[holdout_mask, :]
    y_home_train = modeling_df.loc[train_mask, TARGET_HOME]
    y_away_train = modeling_df.loc[train_mask, TARGET_AWAY]
    y_class_train = modeling_df.loc[train_mask, CLASS_LABEL].astype(int)

    home_model, home_metrics = fit_regressor("home", X_train, y_home_train, preprocessor, cfg)
    away_model, away_metrics = fit_regressor("away", X_train, y_away_train, preprocessor, cfg)
    clf, clf_metrics = fit_classifier(X_train, y_class_train, preprocessor, cfg)

    evaluation: Dict[str, Any] = {}
    decision_threshold = 0.5
    if not X_holdout.empty:
        evaluation["regression_holdout"] = evaluate_regressors(
            home_model,
            away_model,
            X_holdout,
            modeling_df.loc[holdout_mask, TARGET_HOME],
            modeling_df.loc[holdout_mask, TARGET_AWAY],
        )

        clf_holdout_metrics = evaluate_classifier(
            clf,
            X_holdout,
            modeling_df.loc[holdout_mask, CLASS_LABEL].astype(int),
        )
        evaluation["classifier_holdout"] = clf_holdout_metrics
        decision_threshold = clf_holdout_metrics.get("threshold", decision_threshold)

        score_proba = score_based_classifier(home_model, away_model, X_holdout)
        evaluation["score_model_bias"] = {
            "home_prediction_rate": float((score_proba >= 0.5).mean()),
            "mean_probability": float(score_proba.mean()),
        }

    previous_report = load_previous_report(cfg.report_path)
    report = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "data_path": str(cfg.data_path),
        "models_dir": str(cfg.models_dir),
        "random_state": cfg.random_state,
        "train_rows": int(train_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "home_regressor": home_metrics,
        "away_regressor": away_metrics,
        "classifier": clf_metrics,
        "evaluation": evaluation,
        "previous_report": previous_report,
        "decision_threshold": decision_threshold,
    }

    save_artifacts(cfg, preprocessor, home_model, away_model, clf)
    write_report(cfg, report)
    write_metadata(
        cfg,
        numeric_cols,
        categorical_cols,
        int(train_mask.sum()),
        int(holdout_mask.sum()),
        decision_threshold,
    )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NFL prediction models with fairness diagnostics")
    parser.add_argument("--data", type=Path, default=Path("backend/data/game_features.csv"), help="Dataset path emitted by backend/build_dataset.py")
    parser.add_argument("--models", type=Path, default=Path("backend/models"), help="Directory to persist trained artifacts")
    parser.add_argument("--metrics", type=Path, default=Path("metrics/training"), help="Directory to store supplemental metrics (reserved for future use)")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED, help="Random seed for deterministic behaviour")
    parser.add_argument("--max-search-iter", type=int, default=25, help="RandomizedSearch iterations per estimator")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def main() -> None:
    load_dotenv("backend/.env", override=False)

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = TrainerConfig(
        data_path=args.data,
        models_dir=args.models,
        metrics_dir=args.metrics,
        random_state=args.random_state,
        max_search_iter=args.max_search_iter,
    )

    logging.info("Starting training with dataset %s", cfg.data_path)
    report = train(cfg)
    logging.info("Training complete: %s", json.dumps(report["evaluation"], indent=2)[:800])


if __name__ == "__main__":
    main()
