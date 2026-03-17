#!/usr/bin/env python3
"""
backend/train_models.py

Merged + enhanced training script implementing:

Evolution path 1:
  Rolling walk-forward evaluation for home_win
    train <= Y-1, calibrate Y, test Y+1
  Aggregated OOS metrics with bootstrap confidence intervals

Evolution path 2:
  Export predictions_test_2025.csv with per-game probs + decile bins
  Export predictions_future.csv for rows where home_win is null

Other improvements:
  - Drop near-empty columns (default >=95% missing)
  - Drop suspicious post-game columns via missingness mismatch vs future rows
  - Exclude home_team/away_team from training features (kept for reporting)
  - Time-aware calibration using CalibratedClassifierCV(cv="prefit") on calibration season
  - Fold-safe categorical imputation using constant fill (prevents "all-missing" fold failures)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.calibration import FrozenEstimator
except ImportError:
    FrozenEstimator = None  # Fallback for older sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------
# Environment + knobs
# -----------------------
_backend_dir = Path(__file__).parent
_repo_root = _backend_dir.parent
for env_path in [_backend_dir / ".env", _repo_root / ".env"]:
    if env_path.exists():
        load_dotenv(env_path, verbose=True)
        break

HP_N_ITER = int(os.getenv("HP_N_ITER", "50"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
N_JOBS = int(os.getenv("N_JOBS", "-1"))

TARGET_HOME = "home_points_for"
TARGET_AWAY = "away_points_for"
CLASS_LABEL = "home_win"
TIME_KEYS = ["season", "week"]

# Reporting-only columns (kept in outputs, excluded from training features)
REPORT_COLS = ["season", "week", "game_id", "home_team", "away_team"]
REPORT_ONLY = {"home_team", "away_team"}

# ID columns excluded from training features
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

# Hard leakage blocklist (case-insensitive)
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
    # Market (policy choice: treated as do-not-train here)
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
    "reg__max_depth": [3, 6],
    "reg__learning_rate": [0.1],
}

CLF_PARAM_DISTS = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__class_weight": [None, "balanced"],
}

HIST_PARAM_DISTS = {
    "clf__max_depth": [3, 6, 10],
    "clf__learning_rate": [0.05, 0.1],
    "clf__l2_regularization": [0.0, 0.1, 1.0],
}


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_models")


# -----------------------
# Data classes
# -----------------------
@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> Dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


@dataclass
class WalkForwardSummary:
    start_calib_season: int
    end_calib_season: int
    n_folds_used: int
    oos_rows: int
    metrics_overall: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingSummary:
    training_timestamp_utc: str
    training_duration_seconds: float
    rows_total: int
    rows_completed: int
    rows_future: int
    rows_train_reg: int
    rows_holdout_reg: int
    dataset_hash: str
    production_ready: bool
    home_model_metrics: Dict[str, Any] = field(default_factory=dict)
    away_model_metrics: Dict[str, Any] = field(default_factory=dict)
    deploy_win_metrics: Dict[str, Any] = field(default_factory=dict)
    walkforward_win_summary: Dict[str, Any] = field(default_factory=dict)
    cv_best_params: Dict[str, Any] = field(default_factory=dict)


# -----------------------
# Utility
# -----------------------
def _timer(start: float) -> str:
    elapsed = time.time() - start
    return f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _dataset_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=TIME_KEYS).reset_index(drop=True)


def _dataset_hash(df: pd.DataFrame) -> str:
    key_cols = [c for c in TIME_KEYS + ["home_team", "away_team"] if c in df.columns]
    if not key_cols:
        key_cols = df.columns[:5].tolist()
    content = df[key_cols].to_json(orient="records")
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        # Weight by bin occupancy inferred from y_prob distribution
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.digitize(y_prob, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        counts = np.bincount(bin_idx, minlength=n_bins)
        if counts.sum() == 0:
            return 0.0
        k = len(frac_pos)
        weights = counts[:k] / counts.sum()
        return float(np.sum(weights * np.abs(frac_pos - mean_pred)))
    except Exception as e:
        log.warning("ECE failed: %s", e)
        return 0.0


def _calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.astype(int), "p": y_prob.astype(float)})
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["p"], bins=edges, include_lowest=True)
    return (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), avg_pred=("p", "mean"), win_rate=("y", "mean"))
        .reset_index()
    )


def _bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point_val = metric_fn(y_true, y_prob)
    point = None if point_val is None else float(point_val)

    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        try:
            v = metric_fn(yt, yp)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            vals.append(float(v))
        except Exception:
            continue

    if len(vals) < 50:
        return {"point": point, "ci_low": None, "ci_high": None, "n_boot_used": len(vals)}

    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot_used": len(vals)}


# -----------------------
# Feature hygiene
# -----------------------
def _drop_leaky_and_internal(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    to_drop: List[str] = []
    block_lower = {c.lower() for c in LEAK_BLOCKLIST}
    for c in df.columns:
        if c.lower() in block_lower:
            to_drop.append(c)
        if isinstance(c, str) and c.startswith("_"):
            to_drop.append(c)

    # Never drop reporting columns here
    to_drop = [c for c in sorted(set(to_drop)) if c not in set(REPORT_COLS)]

    if to_drop:
        log.warning("Dropping %d leaky/internal cols (sample): %s", len(to_drop), to_drop[:12])
        df = df.drop(columns=to_drop, errors="ignore")

    return df, to_drop


def _find_near_empty_cols(df: pd.DataFrame, threshold: float) -> List[str]:
    miss = df.isna().mean()
    return miss[miss >= threshold].index.tolist()


def _find_suspicious_postgame_cols(
    df_complete: pd.DataFrame,
    df_future: pd.DataFrame,
    candidate_cols: List[str],
    complete_missing_max: float,
    future_missing_min: float,
) -> List[str]:
    if len(df_future) == 0:
        return []
    miss_complete = df_complete[candidate_cols].isna().mean()
    miss_future = df_future[candidate_cols].isna().mean()
    mask = (miss_complete <= complete_missing_max) & (miss_future >= future_missing_min)
    return miss_complete[mask].index.tolist()


def _drop_constant_in_train(X_train: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if X_train[c].nunique(dropna=True) <= 1]


def _coerce_numeric_object_cols(
    X_train: pd.DataFrame,
    others: List[pd.DataFrame],
    min_parse_rate: float,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[str]]:
    """
    If an object column is *mostly* parseable as numeric in training, coerce it to numeric
    consistently across calibration/test/future matrices.
    """
    X_train = X_train.copy()
    others = [o.copy() for o in others]
    converted: List[str] = []

    obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        s = X_train[c]
        non_null = int(s.notna().sum())
        if non_null == 0:
            continue
        s_num = pd.to_numeric(s, errors="coerce")
        parse_rate = float(s_num.notna().sum() / non_null)
        if parse_rate >= min_parse_rate:
            X_train[c] = s_num
            for i in range(len(others)):
                others[i][c] = pd.to_numeric(others[i][c], errors="coerce")
            converted.append(c)

    return X_train, others, converted


def _infer_feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    ignore = set(ID_COLS) | set(TIME_KEYS) | set(REPORT_ONLY) | {TARGET_HOME, TARGET_AWAY, CLASS_LABEL} | set(LEAK_BLOCKLIST)
    num_cols: List[str] = []
    cat_cols: List[str] = []

    for c in df.columns:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        elif pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            cat_cols.append(c)

    return num_cols, cat_cols


def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Constant categorical imputer prevents failures when a fold has all-missing categories.
    """
    # Explicitly use simple imputer to ensure it gets the correct versioned attributes
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="constant", fill_value="__MISSING__")

    num_pipe = Pipeline(
        [("imputer", num_imputer), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", cat_imputer),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )


# -----------------------
# Model training + evaluation
# -----------------------
def _get_holdout_split_indices(n: int, holdout_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    h = int(n * holdout_ratio)
    train_idx = np.arange(n - h)
    hold_idx = np.arange(n - h, n)
    return train_idx, hold_idx


def _evaluate_regression(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> RegressionMetrics:
    pred = model.predict(X)
    return RegressionMetrics(
        mae=float(mean_absolute_error(y, pred)),
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        r2=float(r2_score(y, pred)),
    )


def _tune_regressor(X_train: pd.DataFrame, y_train: pd.Series, pre: ColumnTransformer, random_state: int, n_jobs: int, label: str) -> Tuple[Pipeline, Dict[str, Any]]:
    log.info("Tuning %s regressor...", label)
    n_iter = min(HP_N_ITER, 40)
    estimator = Pipeline([("pre", pre), ("reg", HistGradientBoostingRegressor(random_state=random_state))])
    rs = RandomizedSearchCV(
        estimator=estimator,
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
    try:
        rs.fit(X_train, y_train)
    except Exception as e:
        log.error("%s regressor tuning failed: %s. Falling back to default estimator.", label, str(e))
        # Ensure we fit the estimator before returning
        estimator.fit(X_train, y_train)
        return estimator, {}
    log.info("%s regressor best CV MAE: %.4f", label, -float(rs.best_score_))
    return cast(Pipeline, rs.best_estimator_), dict(rs.best_params_)


def _tune_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pre: ColumnTransformer,
    random_state: int,
    n_jobs: int,
    n_iter: int,
) -> Tuple[Pipeline, Dict[str, Any], float, Pipeline, Dict[str, Any], float]:
    """
    Tune LogisticRegression + HistGradientBoostingClassifier using time-series CV.
    Returns best estimators (already fit on full X_train via refit=True) + their CV LogLoss.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    lr_pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(random_state=random_state, max_iter=2000, solver='lbfgs', penalty='l2'))])
    hgb_pipe = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(random_state=random_state, max_iter=350))])

    lr_rs = RandomizedSearchCV(
        estimator=lr_pipe,
        param_distributions=CLF_PARAM_DISTS,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
        n_iter=n_iter,
        refit=True,
        error_score="raise",
    )
    hgb_rs = RandomizedSearchCV(
        estimator=hgb_pipe,
        param_distributions=HIST_PARAM_DISTS,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
        n_iter=n_iter,
        refit=True,
        error_score="raise",
    )

    try:
        lr_rs.fit(X_train, y_train)
        lr_best = cast(Pipeline, lr_rs.best_estimator_)
        lr_params = dict(lr_rs.best_params_)
        lr_ll = float(-lr_rs.best_score_)
    except Exception as e:
        log.error("LogisticRegression tuning failed: %s. Using default fit.", str(e))
        lr_best = lr_pipe
        lr_best.fit(X_train, y_train)
        lr_params = {}
        lr_ll = 0.693

    try:
        hgb_rs.fit(X_train, y_train)
        hgb_best = cast(Pipeline, hgb_rs.best_estimator_)
        hgb_params = dict(hgb_rs.best_params_)
        hgb_ll = float(-hgb_rs.best_score_)
    except Exception as e:
        log.error("HistGradientBoosting tuning failed: %s. Using default fit.", str(e))
        hgb_best = hgb_pipe
        hgb_best.fit(X_train, y_train)
        hgb_params = {}
        hgb_ll = 0.693

    return lr_best, lr_params, lr_ll, hgb_best, hgb_params, hgb_ll


def _calibrate_prefit(
    fitted_model: BaseEstimator,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    method: str,
    n_jobs: int,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    if len(X_calib) == 0 or len(np.unique(y_calib)) < 2:
        return fitted_model, {"calibrated": False, "method": None, "cv": "prefit_skipped"}

    # sklearn 1.6+ deprecated cv='prefit' in favor of FrozenEstimator
    if FrozenEstimator is not None:
        cal = CalibratedClassifierCV(estimator=FrozenEstimator(fitted_model), method=method, cv="prefit", n_jobs=n_jobs)
    else:
        cal = CalibratedClassifierCV(estimator=fitted_model, method=method, cv="prefit", n_jobs=n_jobs)
    cal.fit(X_calib, y_calib)
    return cal, {"calibrated": True, "method": method, "cv": "prefit"}


def _pick_winner(lr_ll: float, hgb_ll: float) -> str:
    return "hist_gb" if hgb_ll < lr_ll else "log_reg"


def _add_probability_bins(df_pred: pd.DataFrame, prob_col: str, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df_pred = df_pred.copy()
    df_pred["proba_bin"] = pd.cut(df_pred[prob_col], bins=edges, include_lowest=True)
    df_pred["proba_bin_id"] = pd.cut(df_pred[prob_col], bins=edges, include_lowest=True, labels=False)
    return df_pred


def _walk_forward_eval(
    df_complete: pd.DataFrame,
    y_win: pd.Series,
    base_feature_cols: List[str],
    walk_start_calib: int,
    walk_end_calib: int,
    n_boot: int,
    seed: int,
    min_parse_rate: float,
    near_empty_threshold: float,
    complete_missing_max: float,
    future_missing_min: float,
) -> Tuple[WalkForwardSummary, pd.DataFrame]:
    """
    For each calibration season Y:
      train seasons <= Y-1
      calibrate season == Y
      test season == Y+1
    """
    oos_rows: List[pd.DataFrame] = []
    folds_used = 0

    for calib_season in range(walk_start_calib, walk_end_calib + 1):
        test_season = calib_season + 1

        train_mask = df_complete["season"] <= (calib_season - 1)
        calib_mask = df_complete["season"] == calib_season
        test_mask = df_complete["season"] == test_season

        if train_mask.sum() == 0 or calib_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = df_complete.loc[train_mask, base_feature_cols].copy()
        X_calib = df_complete.loc[calib_mask, base_feature_cols].copy()
        X_test = df_complete.loc[test_mask, base_feature_cols].copy()

        y_train = y_win.loc[train_mask].astype(int)
        y_calib = y_win.loc[calib_mask].astype(int)
        y_test = y_win.loc[test_mask].astype(int)

        # Drop constants for this fold (keeps imputer safe)
        const_cols = _drop_constant_in_train(X_train, base_feature_cols)
        feat_cols = [c for c in base_feature_cols if c not in set(const_cols)]
        X_train = X_train[feat_cols]
        X_calib = X_calib[feat_cols]
        X_test = X_test[feat_cols]

        # Coerce numeric-like object cols consistently
        X_train, [X_calib, X_test], coerced_cols = _coerce_numeric_object_cols(
            X_train, [X_calib, X_test], min_parse_rate=min_parse_rate
        )

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        pre = _make_preprocessor(num_cols, cat_cols)

        # Keep walk-forward tuning modest (many folds)
        tune_iter = min(HP_N_ITER, 12)
        lr_best, lr_params, lr_ll, hgb_best, hgb_params, hgb_ll = _tune_classifiers(
            X_train, y_train, pre, RANDOM_SEED, N_JOBS, n_iter=tune_iter
        )

        winner = _pick_winner(lr_ll, hgb_ll)
        if winner == "hist_gb":
            base = hgb_best
            base_ll = hgb_ll
        else:
            base = lr_best
            base_ll = lr_ll

        cal_model, cal_info = _calibrate_prefit(base, X_calib, y_calib, method="sigmoid", n_jobs=N_JOBS)

        p = cal_model.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)

        out = df_complete.loc[test_mask, REPORT_COLS].copy()
        if "home_game_date" in df_complete.columns:
            out["home_game_date"] = df_complete.loc[test_mask, "home_game_date"]

        out["y_true"] = y_test.to_numpy(dtype=int)
        out["home_win_proba"] = p
        out["predicted_home_win"] = pred
        out["fold_train_max_season"] = calib_season - 1
        out["fold_calib_season"] = calib_season
        out["fold_test_season"] = test_season
        out["winner_algorithm"] = winner
        out["winner_cv_logloss"] = float(base_ll)
        out["calibrated"] = bool(cal_info.get("calibrated", False))
        out["calibration_method"] = cal_info.get("method")

        oos_rows.append(out)
        folds_used += 1

        log.info(
            "WF fold: train<=%d calib=%d test=%d | rows=%d | winner=%s | CV LogLoss=%.4f | calibrated=%s",
            calib_season - 1,
            calib_season,
            test_season,
            len(out),
            winner,
            base_ll,
            out["calibrated"].iloc[0],
        )

    if folds_used == 0:
        return WalkForwardSummary(walk_start_calib, walk_end_calib, 0, 0, {}), pd.DataFrame()

    oos = pd.concat(oos_rows, axis=0).reset_index(drop=True)
    y_true_all = oos["y_true"].to_numpy(dtype=int)
    y_prob_all = oos["home_win_proba"].to_numpy(dtype=float)

    metrics_ci = {
        "roc_auc": _bootstrap_ci(y_true_all, y_prob_all, _safe_roc_auc, n_boot=n_boot, seed=seed),
        "log_loss": _bootstrap_ci(y_true_all, y_prob_all, lambda yt, yp: float(log_loss(yt, yp, normalize=True)), n_boot=n_boot, seed=seed + 1),
        "brier": _bootstrap_ci(y_true_all, y_prob_all, lambda yt, yp: float(brier_score_loss(yt, yp)), n_boot=n_boot, seed=seed + 2),
        "ece_10": _bootstrap_ci(y_true_all, y_prob_all, lambda yt, yp: float(_compute_ece(yt, yp, n_bins=10)), n_boot=n_boot, seed=seed + 3),
        "accuracy@0.5": {
            "point": float(((y_prob_all >= 0.5).astype(int) == y_true_all).mean()),
            "ci_low": None,
            "ci_high": None,
            "n_boot_used": None,
        },
    }

    summary = WalkForwardSummary(
        start_calib_season=walk_start_calib,
        end_calib_season=walk_end_calib,
        n_folds_used=folds_used,
        oos_rows=int(len(oos)),
        metrics_overall=metrics_ci,
    )
    return summary, oos


# -----------------------
# Main
# -----------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFL ML models (merged + walk-forward)")
    parser.add_argument("--data", type=str, default="data/game_features_20250109.csv")
    parser.add_argument("--out", type=str, default="prod-models/models")

    # Hygiene knobs
    parser.add_argument("--near_empty_threshold", type=float, default=0.95)
    parser.add_argument("--complete_missing_max", type=float, default=0.20)
    parser.add_argument("--future_missing_min", type=float, default=0.95)
    parser.add_argument("--numeric_object_parse_rate", type=float, default=0.98)

    # Walk-forward knobs
    parser.add_argument("--walk_start_calib", type=int, default=2019)
    parser.add_argument("--walk_end_calib", type=int, default=2024)
    parser.add_argument("--bootstrap_samples", type=int, default=1500)

    # Prediction knobs
    parser.add_argument("--threshold", type=float, default=0.54)

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    np.random.seed(RANDOM_SEED)

    log.info("=" * 72)
    log.info("TRAIN MODELS (MERGED) | data=%s | out=%s", args.data, str(out_dir))
    log.info("=" * 72)

    df_raw = pd.read_csv(args.data)
    if df_raw.empty:
        raise RuntimeError(f"Dataset is empty: {args.data}")
    df_raw = _normalize_columns(df_raw)
    _ensure_columns(df_raw, TIME_KEYS + [TARGET_HOME, TARGET_AWAY, CLASS_LABEL] + REPORT_COLS)
    df_raw = _dataset_sort(df_raw)
    ds_hash = _dataset_hash(df_raw)

    # Extract labels BEFORE dropping
    y_home_all = df_raw[TARGET_HOME].copy()
    y_away_all = df_raw[TARGET_AWAY].copy()
    y_win_all = df_raw[CLASS_LABEL].copy()

    complete_mask = y_win_all.notna()
    future_mask = y_win_all.isna()

    df_future_raw = df_raw.loc[future_mask].copy()

    # 1) Global leakage sanitization
    df_feat, dropped_leaky_internal = _drop_leaky_and_internal(df_raw.copy())

    # Near-empty drop
    near_empty = [c for c in _find_near_empty_cols(df_feat, args.near_empty_threshold) if c not in set(REPORT_COLS)]
    if near_empty:
        log.info("Dropping %d near-empty cols (>=%.2f missing): %s", len(near_empty), args.near_empty_threshold, near_empty[:12])
        df_feat = df_feat.drop(columns=near_empty, errors="ignore")

    # Suspicious post-game drop (using future rows signal)
    df_complete_feat = df_feat.loc[complete_mask].copy()
    df_future_feat = df_feat.loc[future_mask].copy()
    candidate = [c for c in df_feat.columns if c not in set(REPORT_COLS) and c not in set(ID_COLS) and c not in set(TIME_KEYS)]
    suspicious = _find_suspicious_postgame_cols(
        df_complete=df_complete_feat,
        df_future=df_future_feat,
        candidate_cols=candidate,
        complete_missing_max=args.complete_missing_max,
        future_missing_min=args.future_missing_min,
    )
    suspicious = [c for c in suspicious if c not in set(REPORT_COLS)]
    if suspicious:
        log.warning("Dropping %d suspicious post-game cols (sample): %s", len(suspicious), suspicious[:12])
        df_feat = df_feat.drop(columns=suspicious, errors="ignore")

    # 2) Infer base feature columns (excluding team names)
    num_cols, cat_cols = _infer_feature_cols(df_feat)
    base_feature_cols = num_cols + cat_cols
    if not base_feature_cols:
        raise RuntimeError("No training features remain after sanitization. Check drop rules.")

    # Align complete/future frames
    df_complete = df_feat.loc[complete_mask].copy().reset_index(drop=True)
    df_future = df_feat.loc[future_mask].copy().reset_index(drop=True)

    y_home = y_home_all.loc[complete_mask].reset_index(drop=True)
    y_away = y_away_all.loc[complete_mask].reset_index(drop=True)
    y_win = y_win_all.loc[complete_mask].astype(int).reset_index(drop=True)

    # -------------------------
    # Regression (home/away points)
    # -------------------------
    reg_mask = (~y_home.isna()) & (~y_away.isna())
    df_reg = df_complete.loc[reg_mask].copy().reset_index(drop=True)
    y_home_reg = y_home.loc[reg_mask].reset_index(drop=True)
    y_away_reg = y_away.loc[reg_mask].reset_index(drop=True)

    X_reg = df_reg[base_feature_cols].copy()

    const_reg = _drop_constant_in_train(X_reg, base_feature_cols)
    feature_cols_reg = [c for c in base_feature_cols if c not in set(const_reg)]
    if const_reg:
        log.info("Dropping %d constant cols for regression (sample): %s", len(const_reg), const_reg[:12])
        X_reg = X_reg[feature_cols_reg]

    X_reg, _, coerced_reg = _coerce_numeric_object_cols(X_reg, [], min_parse_rate=args.numeric_object_parse_rate)
    cat_reg = X_reg.select_dtypes(include=["object", "category"]).columns.tolist()
    num_reg = [c for c in X_reg.columns if c not in cat_reg]
    pre_reg = _make_preprocessor(num_reg, cat_reg)

    train_idx, hold_idx = _get_holdout_split_indices(len(X_reg), holdout_ratio=0.20)
    X_train_reg, X_hold_reg = X_reg.iloc[train_idx], X_reg.iloc[hold_idx]
    y_home_train, y_home_hold = y_home_reg.iloc[train_idx], y_home_reg.iloc[hold_idx]
    y_away_train, y_away_hold = y_away_reg.iloc[train_idx], y_away_reg.iloc[hold_idx]

    log.info("-" * 72)
    log.info("Training regressors (chronological holdout 20%%)...")
    home_model, home_params = _tune_regressor(X_train_reg, y_home_train, pre_reg, RANDOM_SEED, N_JOBS, label="home_points")
    away_model, away_params = _tune_regressor(X_train_reg, y_away_train, pre_reg, RANDOM_SEED, N_JOBS, label="away_points")

    home_metrics = _evaluate_regression(home_model, X_hold_reg, y_home_hold)
    away_metrics = _evaluate_regression(away_model, X_hold_reg, y_away_hold)
    log.info("Holdout regression metrics | home=%s | away=%s", home_metrics.to_dict(), away_metrics.to_dict())

    # Save fitted preprocessor from a fitted pipeline (important!)
    fitted_pre = home_model.named_steps["pre"]

    # -------------------------
    # Walk-forward evaluation for win classifier (Evolution Path 1)
    # -------------------------
    log.info("-" * 72)
    log.info("Walk-forward evaluation (train<=Y-1, calib=Y, test=Y+1)...")

    walk_summary, oos = _walk_forward_eval(
        df_complete=df_complete,
        y_win=y_win,
        base_feature_cols=base_feature_cols,
        walk_start_calib=args.walk_start_calib,
        walk_end_calib=args.walk_end_calib,
        n_boot=args.bootstrap_samples,
        seed=RANDOM_SEED,
        min_parse_rate=args.numeric_object_parse_rate,
        near_empty_threshold=args.near_empty_threshold,
        complete_missing_max=args.complete_missing_max,
        future_missing_min=args.future_missing_min,
    )

    oos_path = out_dir / "walkforward_oos_predictions.csv"
    if len(oos) > 0:
        oos.to_csv(oos_path, index=False)
        log.info("Saved OOS walk-forward predictions: %s", str(oos_path))

    # -------------------------
    # Export predictions_test_2025.csv (Evolution Path 2)
    # -------------------------
    test_2025_path = out_dir / "predictions_test_2025.csv"
    if len(oos) > 0 and (oos["season"] == 2025).any():
        df_test_2025 = oos.loc[oos["season"] == 2025].copy()
        df_test_2025 = _add_probability_bins(df_test_2025, prob_col="home_win_proba", n_bins=10)
        df_test_2025 = df_test_2025.sort_values(["season", "week", "game_id"], kind="stable")
        df_test_2025.to_csv(test_2025_path, index=False)
        log.info("Saved test predictions with bins: %s", str(test_2025_path))
        tab = _calibration_table(df_test_2025["y_true"].to_numpy(dtype=int), df_test_2025["home_win_proba"].to_numpy(dtype=float), n_bins=10)
        log.info("Calibration table (2025 test):\n%s", tab.to_string(index=False))
    else:
        log.warning("No 2025 test predictions produced (check walkforward range / dataset coverage).")

    # -------------------------
    # Train deployment win model (train<=2024, calib=2025) + predict future
    # -------------------------
    log.info("-" * 72)
    log.info("Deployment win model: train<=2024, calib=2025, predict future rows...")

    train_deploy = df_complete["season"] <= 2024
    calib_deploy = df_complete["season"] == 2025

    # Fallback if 2025 isn't present in completed data
    if calib_deploy.sum() == 0:
        log.warning("No completed 2025 rows found; falling back to last 20%% of completed games as calibration set.")
        n = len(df_complete)
        idx_train, idx_cal = _get_holdout_split_indices(n, holdout_ratio=0.20)
        train_deploy = pd.Series(False, index=df_complete.index)
        calib_deploy = pd.Series(False, index=df_complete.index)
        train_deploy.iloc[idx_train] = True
        calib_deploy.iloc[idx_cal] = True

    X_train_w = df_complete.loc[train_deploy, base_feature_cols].copy()
    y_train_w = y_win.loc[train_deploy].astype(int)
    X_calib_w = df_complete.loc[calib_deploy, base_feature_cols].copy()
    y_calib_w = y_win.loc[calib_deploy].astype(int)

    const_w = _drop_constant_in_train(X_train_w, base_feature_cols)
    feat_w = [c for c in base_feature_cols if c not in set(const_w)]
    X_train_w = X_train_w[feat_w]
    X_calib_w = X_calib_w[feat_w]
    X_future_w = df_future[feat_w].copy() if len(df_future) > 0 else pd.DataFrame()

    X_train_w, [X_calib_w, X_future_w], coerced_w = _coerce_numeric_object_cols(
        X_train_w, [X_calib_w, X_future_w], min_parse_rate=args.numeric_object_parse_rate
    )

    cat_w = X_train_w.select_dtypes(include=["object", "category"]).columns.tolist()
    num_w = [c for c in X_train_w.columns if c not in cat_w]
    pre_w = _make_preprocessor(num_w, cat_w)

    tune_iter_deploy = min(HP_N_ITER, 24)
    try:
        lr_best, lr_params, lr_ll, hgb_best, hgb_params, hgb_ll = _tune_classifiers(
            X_train_w, y_train_w, pre_w, RANDOM_SEED, N_JOBS, n_iter=tune_iter_deploy
        )
    except Exception as e:
        log.error("Deployment win model tuning failed: %s. Falling back to default fit.", str(e))
        # Fallback: Just fit default models if tuning crashes
        lr_best = Pipeline([("pre", pre_w), ("clf", LogisticRegression(random_state=RANDOM_SEED, max_iter=2000, solver='lbfgs', penalty='l2'))])
        hgb_best = Pipeline([("pre", pre_w), ("clf", HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_iter=350))])
        lr_best.fit(X_train_w, y_train_w)
        hgb_best.fit(X_train_w, y_train_w)
        lr_params, lr_ll = {}, 0.693  # Placeholder
        hgb_params, hgb_ll = {}, 0.693 # Placeholder

    winner = _pick_winner(lr_ll, hgb_ll)
    lr_cal, lr_cal_info = _calibrate_prefit(lr_best, X_calib_w, y_calib_w, method="sigmoid", n_jobs=N_JOBS)
    hgb_cal, hgb_cal_info = _calibrate_prefit(hgb_best, X_calib_w, y_calib_w, method="sigmoid", n_jobs=N_JOBS)

    if winner == "hist_gb":
        win_model = hgb_cal
        win_cal_info = hgb_cal_info
        win_cv_ll = hgb_ll
    else:
        win_model = lr_cal
        win_cal_info = lr_cal_info
        win_cv_ll = lr_ll

    # Predict future
    future_path = out_dir / "predictions_future.csv"
    if len(df_future) > 0:
        p_future = win_model.predict_proba(X_future_w)[:, 1]
        pred_future = (p_future >= args.threshold).astype(int)

        out_future = df_future_raw[REPORT_COLS].copy()
        out_future["home_win_proba"] = p_future
        out_future["predicted_home_win"] = pred_future
        out_future = out_future.sort_values(["season", "week", "game_id"], kind="stable")
        out_future.to_csv(future_path, index=False)
        log.info("Saved future predictions: %s", str(future_path))
    else:
        log.info("No future rows detected; skipping predictions_future.csv")

    # Quick calibration metrics on calibration slice (if valid)
    deploy_metrics: Dict[str, Any] = {}
    if len(X_calib_w) > 0 and len(np.unique(y_calib_w)) >= 2:
        p_cal = win_model.predict_proba(X_calib_w)[:, 1]
        pred_cal = (p_cal >= 0.5).astype(int)
        precision, recall, _ = precision_recall_curve(y_calib_w.to_numpy(dtype=int), p_cal)
        pr_auc_val = float(auc(recall, precision))
        deploy_metrics = {
            "accuracy": float(accuracy_score(y_calib_w, pred_cal)),
            "roc_auc": _safe_roc_auc(y_calib_w.to_numpy(dtype=int), p_cal),
            "brier": float(brier_score_loss(y_calib_w, p_cal)),
            "log_loss": float(log_loss(y_calib_w, p_cal, eps=1e-15)),
            "pr_auc": pr_auc_val,
            "ece_10": float(_compute_ece(y_calib_w.to_numpy(dtype=int), p_cal, n_bins=10)),
            "calibrated": bool(win_cal_info.get("calibrated", False)),
        }

    # -------------------------
    # Save artifacts
    # -------------------------
    log.info("-" * 72)
    log.info("Saving model artifacts + reports...")

    dump(fitted_pre, out_dir / "preprocessor.joblib")
    dump(home_model, out_dir / "home_model.joblib")
    dump(away_model, out_dir / "away_model.joblib")

    dump(win_model, out_dir / "win_clf_calibrated.joblib")
    dump(hgb_cal, out_dir / "hist_win_clf_calibrated.joblib")
    dump(lr_cal, out_dir / "log_win_clf_calibrated.joblib")

    duration = time.time() - t0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    summary = TrainingSummary(
        training_timestamp_utc=ts,
        training_duration_seconds=float(round(duration, 2)),
        rows_total=int(len(df_raw)),
        rows_completed=int(complete_mask.sum()),
        rows_future=int(future_mask.sum()),
        rows_train_reg=int(len(train_idx)),
        rows_holdout_reg=int(len(hold_idx)),
        dataset_hash=ds_hash,
        production_ready=True,
        home_model_metrics=home_metrics.to_dict(),
        away_model_metrics=away_metrics.to_dict(),
        deploy_win_metrics=deploy_metrics,
        walkforward_win_summary=asdict(walk_summary),
        cv_best_params={
            "home_regressor": home_params,
            "away_regressor": away_params,
            "deploy_win": {
                "winner": winner,
                "winner_cv_logloss": float(win_cv_ll),
                "logreg": {"cv_logloss": float(lr_ll), "best_params": lr_params, "calibration": lr_cal_info},
                "histgb": {"cv_logloss": float(hgb_ll), "best_params": hgb_params, "calibration": hgb_cal_info},
            },
        },
    )

    metadata = {
        "trained_at_utc": ts,
        "dataset_hash": ds_hash,
        "production_ready": True,
        "artifacts": {
            "preprocessor": "preprocessor.joblib",
            "reg_home": "home_model.joblib",
            "reg_away": "away_model.joblib",
            "clf_home_win": "win_clf_calibrated.joblib",
        },
        "raw_feature_columns": {
            "numeric": num_cols,
            "categorical": cat_cols,
        },
        "dropped_columns": {
            "leaky_internal": dropped_leaky_internal,
            "near_empty": near_empty,
            "suspicious_postgame": suspicious,
        },
        "notes": {
            "home_team_away_team_excluded_from_training_features": True,
            "calibration": "CalibratedClassifierCV(cv='prefit') on a dedicated calibration slice",
            "exports": {
                "walkforward_oos_predictions": str(oos_path.name),
                "predictions_test_2025": str(test_2025_path.name),
                "predictions_future": str(future_path.name),
            },
        },
    }

    try:
        (out_dir / "training_report.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        log.info("=" * 72)
        log.info("DONE in %s | dataset_hash=%s", _timer(t0), ds_hash)
        log.info("Report: %s", str(out_dir / "training_report.json"))
        log.info("Meta  : %s", str(out_dir / "metadata.json"))
        log.info("=" * 72)
    except Exception as e:
        log.error("Final reporting failed: %s. Models should be saved in %s", str(e), str(out_dir))


if __name__ == "__main__":
    main()
