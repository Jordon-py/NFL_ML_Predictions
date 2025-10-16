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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import sparse

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

DEFAULT_DATASET = DATA_DIR / os.getenv("TRAIN_DATASET_FILE", "merged_game_features.csv")

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "1337"))
HYPERPARAM_SEARCH_ITERATIONS = int(os.getenv("HP_NITER", "40"))
N_SPLITS = int(os.getenv("CV_SPLITS", "5"))
CALIBRATION_METHOD: Literal["sigmoid", "isotonic"] = cast(
    Literal["sigmoid", "isotonic"], os.getenv("CALIB_METHOD", "sigmoid")
)  # 'sigmoid' or 'isotonic'
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
    return (
        pd.util.hash_pandas_object(df.fillna(-999), index=True)
        .sum()
        .__int__()
        .__str__()
    )


def _infer_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Numeric features default:
      - any float/int columns that are not identifiers or targets
    Categorical:
      - home_team, away_team if present
    """
    cols = list(df.columns)
    ignore = ID_COLS | {TARGET_HOME, TARGET_AWAY, CLASS_LABEL}
    numeric: List[str] = []
    categorical: List[str] = []

    for c in cols:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    for c in ("home_team", "away_team"):
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            categorical.append(c)

    # Allow legacy numeric team codes to be treated as categorical if low-cardinality
    for c in ("home_team", "away_team"):
        if c in df.columns and c not in categorical and df[c].nunique() <= 64:
            categorical.append(c)
            if c in numeric:
                numeric.remove(c)

    return numeric, categorical


def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
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


# -----------------------
# Model search spaces
# -----------------------
def _reg_grid() -> Dict[str, List[Any]]:
    return {
        "learning_rate": list(np.geomspace(0.01, 0.3, 10)),
        "max_depth": [None, 3, 4, 5, 6],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [10, 20, 30, 50, 80],
        "l2_regularization": [0.0, 0.01, 0.05, 0.1],
    }


def _clf_grid() -> Dict[str, List[Any]]:
    return {
        "C": list(np.geomspace(0.05, 10.0, 10)),
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [100, 200, 400],
        "class_weight": [None, "balanced"],
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
    df: pd.DataFrame = None,
) -> FitResult:
    base = HistGradientBoostingRegressor(random_state=RANDOM_SEED)
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=REG_PARAMS,
        n_iter=HYPERPARAM_SEARCH_ITERATIONS,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
        refit=True,
    )
    rs.fit(X, y)

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


def _compute_recency_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Computes sample weights that give more importance to more recent games.
    This helps the model prioritize learning from the latest team dynamics.
    """
    # A unique, sortable key for each game
    tk = df["season"].astype(str) + df["week"].astype(str).str.zfill(2)

    # Use .to_numpy() to get a reliable numpy array. .values can return a
    # pandas ExtensionArray, which is incompatible with np.argsort.
    sorted_indices = np.argsort(tk.to_numpy())

    # Create a ranking where the most recent game has the highest rank
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(tk))

    # Scale ranks to a [0.1, 1.0] range and return as weights
    scaled_weights = (ranks / len(tk)) * 0.9 + 0.1
    return scaled_weights.astype(float)


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
    data_path = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET)))
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    if df.empty:
        raise RuntimeError("Dataset is empty")

    # Filter rows with outcomes for supervised learning
    have_scores = df[TARGET_HOME].notna() & df[TARGET_AWAY].notna()
    train_df = df.loc[have_scores].copy()
    train_df["home_win"] = (train_df[TARGET_HOME] > train_df[TARGET_AWAY]).astype(int)

    # Infer features
    num_cols, cat_cols = _infer_features(train_df)
    pre = _make_preprocessor(num_cols, cat_cols)

    # Fit preprocessor and transform full training matrix
    X_df = train_df[num_cols + cat_cols] if cat_cols else train_df[num_cols]
    pre.fit(X_df)
    X_full = pre.transform(X_df)
    # ChangeLog 2024-10-07: Coerce features to dense arrays to eliminate sparse typing errors and keep training stable.
    X_full = _ensure_dense_matrix(X_full, context="training features")

    # Targets
    y_home = train_df[TARGET_HOME].to_numpy()
    y_away = train_df[TARGET_AWAY].to_numpy()
    y_clf = train_df["home_win"].to_numpy()

    # Train regressors with small ensemble
    res_home = _fit_regressor(X_full, y_home, pre, train_df)
    res_away = _fit_regressor(X_full, y_away, pre, train_df)

    # Train classifier with calibration and threshold sweep
    clf_res = _fit_classifier(X_full, y_clf, train_df)

    # Build a validation error table on last split for diagnostics
    tscv = _time_splits(train_df, n_splits=N_SPLITS)
    tr_idx, te_idx = _last_split_indices(train_df, tscv)
    X_te = X_full[te_idx]
    home_pred = _predict_reg(res_home.model, X_te)
    away_pred = _predict_reg(res_away.model, X_te)
    abs_err = np.abs(
        home_pred - train_df.iloc[te_idx][TARGET_HOME].to_numpy()
    ) + np.abs(away_pred - train_df.iloc[te_idx][TARGET_AWAY].to_numpy())
    val_err = train_df.iloc[te_idx][
        ["season", "week", "home_team", "away_team", TARGET_HOME, TARGET_AWAY]
    ].copy()
    val_err["pred_home"] = np.round(home_pred, 2)
    val_err["pred_away"] = np.round(away_pred, 2)
    val_err["abs_error_sum"] = np.round(abs_err, 2)
    val_err.sort_values("abs_error_sum", ascending=False).to_csv(
        MODELS_DIR / "validation_errors.csv", index=False
    )

    # Save artifacts
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib", compress=3)
    joblib.dump(res_home.model, MODELS_DIR / "home_model.joblib", compress=3)
    joblib.dump(res_away.model, MODELS_DIR / "away_model.joblib", compress=3)
    joblib.dump(clf_res.model, MODELS_DIR / "win_clf_calibrated.joblib", compress=3)

    # Reports
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
