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
<<<<<<< HEAD
from sklearn.linear_model import LogisticRegression
=======
from sklearn.linear_model import LogisticRegression, Ridge
>>>>>>> 572910b09d95a69b5bb241dda05ce6698620f8e9
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
<<<<<<< HEAD
    """Return numeric and categorical feature names after removing IDs and blocklisted fields."""
    ignore = set(ID_COLS) | set(TIME_KEYS) | set(LEAK_BLOCKLIST)
=======
    """
    Numeric features default:
      - any float/int columns that are not identifiers or targets
    Categorical:
      - home_team, away_team if present
    """
    cols = list(df.columns)
    ignore = ID_COLS | {TARGET_HOME, TARGET_AWAY, CLASS_LABEL}
>>>>>>> 572910b09d95a69b5bb241dda05ce6698620f8e9
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
<<<<<<< HEAD
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
=======
    transformers = []
    if num_cols:
        # Add imputer to handle NaN values in numeric columns
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True))
        ])
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )
    if not transformers:
        raise RuntimeError("No features selected. Check dataset and feature inference.")
    return ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=None)
>>>>>>> 572910b09d95a69b5bb241dda05ce6698620f8e9


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
<<<<<<< HEAD
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
    
=======
    df: pd.DataFrame = None,
) -> FitResult:
    base = HistGradientBoostingRegressor(random_state=RANDOM_SEED)
>>>>>>> 572910b09d95a69b5bb241dda05ce6698620f8e9
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
<<<<<<< HEAD
    
    # Fixed: Cast rs.best_estimator_ to Pipeline to match return type hint.
    # This resolves the type checker error while preserving runtime behavior.
    return cast(Pipeline, rs.best_estimator_)
=======

    # Simple 2-model blend: HGBR + Ridge; search blend weight on validation slice
    # Prepare validation slice
    if df is not None:
        tscv = _time_splits(df, n_splits=N_SPLITS)
        tr_idx, te_idx = _last_split_indices(df, tscv)
    else:
        tscv = _time_splits(pd.DataFrame(index=np.arange(len(y))), n_splits=N_SPLITS)
        tr_idx, te_idx = _last_split_indices(pd.DataFrame(index=np.arange(len(y))), tscv)
    X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    hgbr = cast(HistGradientBoostingRegressor, rs.best_estimator_)
    ridge = Ridge(random_state=RANDOM_SEED)
    ridge.fit(X_tr, y_tr)

    preds_h = hgbr.predict(X_te)
    preds_r = ridge.predict(X_te)
    best_w, best_mae = 1.0, mean_absolute_error(y_te, preds_h)
    for w in np.linspace(0.2, 0.9, 8):
        blend = w * preds_h + (1 - w) * preds_r
        mae = mean_absolute_error(y_te, blend)
        if mae < best_mae:
            best_mae, best_w = mae, w

    # Wrap ensemble
    model = {"hgbr": hgbr, "ridge": ridge, "weight": float(best_w)}
    report = {
        "best_params": rs.best_params_,
        "val_mae_hgbr": float(mean_absolute_error(y_te, preds_h)),
        "val_mae_ridge": float(mean_absolute_error(y_te, preds_r)),
        "val_mae_blend": float(best_mae),
        "blend_weight_hgbr": float(best_w),
    }
    return FitResult(model=model, mae_val=best_mae, report=report)
>>>>>>> 572910b09d95a69b5bb241dda05ce6698620f8e9


def _evaluate_regression(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    pred = model.predict(X)
    return float(mean_absolute_error(y, pred))


def _save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


<<<<<<< HEAD
def _dataset_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(TIME_KEYS).reset_index(drop=True)
=======
def _fit_classifier(
    X: np.ndarray,
    y_clf: np.ndarray,
    df: pd.DataFrame = None,
) -> ClfResult:
    base = LogisticRegression()
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=CLF_PARAMS,
        n_iter=HYPERPARAM_SEARCH_ITERATIONS,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
        refit=True,
    )
    rs.fit(X, y_clf)
    best_lr = cast(LogisticRegression, rs.best_estimator_)

    # Final calibration on last split
    # Build a synthetic df to reuse the same splitter
    if df is not None:
        tscv = _time_splits(df, n_splits=N_SPLITS)
        tr_idx, te_idx = _last_split_indices(df, tscv)
    else:
        df_idx = pd.DataFrame(index=np.arange(len(y_clf)))
        tscv = _time_splits(df_idx, n_splits=N_SPLITS)
        tr_idx, te_idx = _last_split_indices(df_idx, tscv)

    cal = CalibratedClassifierCV(best_lr, method=CALIBRATION_METHOD, cv="prefit")
    cal.fit(X[tr_idx], y_clf[tr_idx])
    proba = cal.predict_proba(X[te_idx])[:, 1]

    # Metrics
    auc = roc_auc_score(y_clf[te_idx], proba)
    br = brier_score_loss(y_clf[te_idx], proba)
    ll = log_loss(y_clf[te_idx], np.c_[1 - proba, proba])
    acc50 = accuracy_score(y_clf[te_idx], (proba >= 0.5).astype(int))

    # Reliability bins
    bins = np.linspace(0, 1, RELIABILITY_BINS + 1)
    bin_ids = np.digitize(proba, bins) - 1
    reliab = []
    for b in range(RELIABILITY_BINS):
        m = bin_ids == b
        if m.any():
            mean_p = float(np.mean(proba[m]))
            mean_y = float(np.mean(y_clf[te_idx][m]))
            n = int(np.sum(m))
            reliab.append({"bin": b, "n": n, "mean_pred": mean_p, "mean_true": mean_y})

    # Threshold sweep on validation to maximize F1, tie-break to accuracy
    best_th, best_f1, best_acc = 0.5, -1.0, 0.0
    for th in np.linspace(0.3, 0.7, 41):
        preds = (proba >= th).astype(int)
        tp = np.sum((preds == 1) & (y_clf[te_idx] == 1))
        fp = np.sum((preds == 1) & (y_clf[te_idx] == 0))
        fn = np.sum((preds == 0) & (y_clf[te_idx] == 1))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        acc = accuracy_score(y_clf[te_idx], preds)
        if f1 > best_f1 or (math.isclose(f1, best_f1, rel_tol=1e-6) and acc > best_acc):
            best_f1, best_acc, best_th = f1, acc, float(th)

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
>>>>>>> 572910b09d95a69b5bb241dda05ce6698620f8e9


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

<<<<<<< HEAD
    log.info("Fitting win classifier")
    win_model, win_holdout_metrics = _fit_classifier(X, y_win, pre, RANDOM_SEED)

    # Quick in-sample sanity MAE (for monitoring only)
    mae_home = _evaluate_regression(home_model, X, y_home)
    mae_away = _evaluate_regression(away_model, X, y_away)
=======
    # Train regressors with small ensemble
    res_home = _fit_regressor(X_full, y_home, pre, train_df)
    res_away = _fit_regressor(X_full, y_away, pre, train_df)

    # Train classifier with calibration and threshold sweep
    clf_res = _fit_classifier(X_full, y_clf, train_df)
>>>>>>> 572910b09d95a69b5bb241dda05ce6698620f8e9

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
    parser.add_argument("--data", type=str, default="backend/data/game_features.csv", help="Path to features CSV.")
    parser.add_argument("--out", type=str, default="backend/models", help="Output directory for artifacts and reports.")
    args = parser.parse_args()
    main(args.data, args.out)
