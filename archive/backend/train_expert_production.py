#!/usr/bin/env python3
"""
train_expert_production.py

Enhanced leak-free NFL model trainer combining:
  - Strict leakage prevention (from train_home_win_expert.py)
  - Walk-forward validation with bootstrap CIs (from train_models.py)
  - Multi-output training: HOME/AWAY regressors + calibrated WIN classifier

Key Features:
  - Automatic post-game column detection via missingness mismatch
  - Walk-forward evaluation (train ≤ Y-1, calibrate Y, test Y+1)
  - Bootstrap confidence intervals for robust metrics
  - Platt scaling calibration on dedicated calibration season
  - Comprehensive metadata export with feature manifest

Usage:
    python train_expert_production.py \
      --data backend/data/datasets/game_features_20260109_clean.csv \
      --out prod-models/models \
      --walk_start_calib 2020 \
      --walk_end_calib 2024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Dataset contract
TARGET_HOME = "home_points_for"
TARGET_AWAY = "away_points_for"
CLASS_LABEL = "home_win"
TIME_KEYS = ["season", "week"]
REPORT_COLS = ["season", "week", "game_id", "home_team", "away_team"]

# Feature exclusions
ID_COLS = {
    "game_id", "gid", "home_team_id", "away_team_id",
    "stadium", "time_key", "home_game_date", "away_game_date",
    "home_team", "away_team",  # Reporting only
}

# Hard leakage blocklist
LEAK_BLOCKLIST = {
    # Direct targets
    CLASS_LABEL, TARGET_HOME, TARGET_AWAY, "winner", "actual_winner", "point_diff",
    # Post-game realized values
    "home_points_against", "away_points_against",
    "home_score", "away_score", "final_home_score", "final_away_score",
    "postgame_margin", "post_game_total", "actual_margin",
    # Post-game Elo
    "home_elo_post", "away_elo_post",
    # Market lines (policy: exclude from training)
    "home_moneyline", "away_moneyline", "spread_line", "total_line",
    "home_win_prob", "away_win_prob", "season_home_win_rate",
}

# Hyperparameter search spaces
REG_PARAM_DISTS = {
    "reg__max_depth": [3, 6, 10, 14],
    "reg__learning_rate": [0.02, 0.05, 0.1, 0.15],
    "reg__max_leaf_nodes": [15, 31, 63],
    "reg__l2_regularization": [0.02, 0.05, 0.1],
    "reg__min_samples_leaf": [10, 20, 30],
}

CLF_PARAMS = {
    "learning_rate": 0.06,
    "max_iter": 250,
    "max_leaf_nodes": 31,
    "max_depth": 6,
    "min_samples_leaf": 30,
    "l2_regularization": 0.05,
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_expert")


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------
@dataclass
class Config:
    """Training configuration."""
    data_path: Path
    out_dir: Path
    near_empty_threshold: float = 0.95
    complete_missing_max: float = 0.20
    future_missing_min: float = 0.95
    numeric_object_parse_rate: float = 0.98
    walk_start_calib: int = 2020
    walk_end_calib: int = 2024
    bootstrap_samples: int = 1500
    hp_n_iter: int = 30
    n_jobs: int = -1
    reg_holdout_ratio: float = 0.20


@dataclass
class FeatureManifest:
    """Audit trail for feature engineering."""
    n_features_final: int
    features_final: List[str]
    dropped: Dict[str, List[str]] = field(default_factory=dict)
    coerced_numeric: List[str] = field(default_factory=list)


@dataclass
class RegressionMetrics:
    """Regression evaluation metrics."""
    mae: float
    rmse: float
    r2: float


@dataclass
class WalkForwardMetrics:
    """Walk-forward validation summary."""
    n_folds: int
    oos_rows: int
    roc_auc: Dict[str, Any]
    log_loss: Dict[str, Any]
    brier: Dict[str, Any]
    ece: Dict[str, Any]
    accuracy: float


@dataclass
class ProductionMetadata:
    """Complete training metadata for production deployment."""
    training_timestamp_utc: str
    training_duration_seconds: float
    dataset_path: str
    dataset_hash: str
    random_seed: int
    
    # Row counts
    rows_total: int
    rows_completed: int
    rows_future: int
    rows_train_reg: int
    rows_holdout_reg: int
    
    # Feature engineering
    feature_manifest: Dict[str, Any]
    
    # Model training
    home_model_best_params: Dict[str, Any]
    away_model_best_params: Dict[str, Any]
    win_clf_params: Dict[str, Any]
    
    # Validation
    home_metrics_holdout: Dict[str, float]
    away_metrics_holdout: Dict[str, float]
    walkforward_metrics: Dict[str, Any]
    
    # Deployment readiness
    production_ready: bool
    calibration_method: str


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def timer(start: float) -> str:
    """Format elapsed time."""
    elapsed = time.time() - start
    return f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove BOM and whitespace from column names."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def dataset_hash(df: pd.DataFrame) -> str:
    """Compute hash of dataset for provenance tracking."""
    key_cols = [c for c in TIME_KEYS + ["home_team", "away_team"] if c in df.columns]
    if not key_cols:
        key_cols = df.columns[:5].tolist()
    content = df[key_cols].to_json(orient="records")
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Compute ROC AUC, return None if single-class."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    try:
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        
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
        log.warning(f"ECE computation failed: {e}")
        return 0.0


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int
) -> Dict[str, Any]:
    """Bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    point_val = metric_fn(y_true, y_prob)
    point = None if point_val is None else float(point_val)
    
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_prob[idx])
            if v is not None and not np.isnan(v):
                vals.append(float(v))
        except Exception:
            continue
    
    if len(vals) < 50:
        return {"point": point, "ci_low": None, "ci_high": None}
    
    return {
        "point": point,
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


# -----------------------------------------------------------------------------
# Feature Engineering (Leak-Safe)
# -----------------------------------------------------------------------------
def find_near_empty_cols(df: pd.DataFrame, threshold: float) -> List[str]:
    """Find columns with ≥threshold fraction missing."""
    miss = df.isna().mean()
    return miss[miss >= threshold].index.tolist()


def find_constant_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Find columns that are constant or all-missing."""
    return [c for c in cols if df[c].nunique(dropna=True) <= 1]


def find_suspicious_postgame_cols(
    df_complete: pd.DataFrame,
    df_future: pd.DataFrame,
    candidate_cols: List[str],
    complete_missing_max: float,
    future_missing_min: float,
) -> List[str]:
    """
    Detect post-game columns via missingness mismatch.
    
    If a column is mostly present in completed games but almost always
    missing in future games, it's likely not available pre-game.
    """
    if len(df_future) == 0:
        return []
    
    miss_complete = df_complete[candidate_cols].isna().mean()
    miss_future = df_future[candidate_cols].isna().mean()
    
    suspicious_mask = (miss_complete <= complete_missing_max) & (miss_future >= future_missing_min)
    return miss_complete[suspicious_mask].index.tolist()


def coerce_numeric_object_cols(
    X_train: pd.DataFrame,
    others: List[pd.DataFrame],
    min_parse_rate: float = 0.98,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[str]]:
    """
    Convert object columns that are mostly numeric to numeric dtype.
    
    Prevents treating accidental string columns as high-cardinality categoricals.
    """
    X_train = X_train.copy()
    others = [x.copy() for x in others]
    converted = []
    
    obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        s = X_train[c]
        non_null = s.notna().sum()
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


def build_feature_manifest(
    df: pd.DataFrame,
    df_future: pd.DataFrame,
    config: Config,
) -> Tuple[List[str], FeatureManifest]:
    """
    Build leak-safe feature list with audit trail.
    
    Drops:
        1. Explicit targets and leakage blocklist
        2. ID/reporting columns  
        3. Near-empty columns
        4. Suspicious post-game columns
        5. Constant-in-train columns
    """
    df_complete = df[df[CLASS_LABEL].notna()].copy()
    
    # 1. Explicit drops
    explicit_drop = set(LEAK_BLOCKLIST) | set(ID_COLS)
    
    # 2. Near-empty
    near_empty = find_near_empty_cols(df, threshold=config.near_empty_threshold)
    
    # 3. Suspicious post-game
    candidate = [c for c in df.columns if c not in explicit_drop and c not in near_empty]
    suspicious = find_suspicious_postgame_cols(
        df_complete=df_complete,
        df_future=df_future,
        candidate_cols=candidate,
        complete_missing_max=config.complete_missing_max,
        future_missing_min=config.future_missing_min,
    )
    
    # 4. Constant-in-train (use training slice only)
    df_train = df_complete[df_complete["season"] <= (config.walk_start_calib - 1)].copy()
    candidate_after_suspicious = [c for c in candidate if c not in suspicious]
    constant = find_constant_cols(df_train, candidate_after_suspicious)
    
    # Final feature list
    features = [c for c in candidate_after_suspicious if c not in constant]
    
    # Manifest
    manifest = FeatureManifest(
        n_features_final=len(features),
        features_final=features,
        dropped={
            "explicit_leakage": sorted([c for c in df.columns if c in explicit_drop]),
            "near_empty": sorted(near_empty),
            "suspicious_postgame": sorted(suspicious),
            "constant_in_train": sorted(constant),
        },
    )
    
    log.info(f"Feature engineering: {len(features)} features retained")
    log.info(f"  Dropped: {len(near_empty)} near-empty, {len(suspicious)} suspicious, {len(constant)} constant")
    
    return features, manifest


# -----------------------------------------------------------------------------
# Model Building
# -----------------------------------------------------------------------------
def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Create preprocessing pipeline.
    
    Numeric: median impute + standard scaling
    Categorical: most frequent impute + ordinal encoding
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def tune_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    label: str,
    config: Config,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tune regressor with TimeSeriesSplit CV."""
    log.info(f"Tuning {label} regressor (n_iter={config.hp_n_iter})...")
    
    pipe = Pipeline([
        ("pre", preprocessor),
        ("reg", HistGradientBoostingRegressor(random_state=RANDOM_SEED)),
    ])
    
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=REG_PARAM_DISTS,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
        n_jobs=config.n_jobs,
        random_state=RANDOM_SEED,
        n_iter=config.hp_n_iter,
        verbose=0,
        refit=True,
    )
    
    rs.fit(X_train, y_train)
    log.info(f"{label} best CV MAE: {-rs.best_score_:.3f}")
    
    return rs.best_estimator_, dict(rs.best_params_)


def evaluate_regression(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> RegressionMetrics:
    """Evaluate regressor on holdout."""
    pred = model.predict(X)
    return RegressionMetrics(
        mae=float(mean_absolute_error(y, pred)),
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        r2=float(r2_score(y, pred)),
    )




# -----------------------------------------------------------------------------
# Walk-Forward Validation
# -----------------------------------------------------------------------------
def walk_forward_validate(
    df_complete: pd.DataFrame,
    features: List[str],
    y_win: pd.Series,
    config: Config,
) -> Tuple[WalkForwardMetrics, pd.DataFrame]:
    """
    Walk-forward validation for win classifier.
    
    For each calib year Y in [2020..2024]:
        train ≤ Y-1, calibrate Y, test Y+1
    
    Returns aggregated OOS metrics with bootstrap CIs.
    """
    oos_predictions = []
    folds_used = 0
    
    for calib_year in range(config.walk_start_calib, config.walk_end_calib + 1):
        test_year = calib_year + 1
        
        train_mask = df_complete["season"] <= (calib_year - 1)
        calib_mask = df_complete["season"] == calib_year
        test_mask = df_complete["season"] == test_year
        
        if train_mask.sum() == 0 or calib_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        
        X_train = df_complete.loc[train_mask, features].copy()
        X_calib = df_complete.loc[calib_mask, features].copy()
        X_test = df_complete.loc[test_mask, features].copy()
        
        y_train = y_win[train_mask].astype(int)
        y_calib = y_win[calib_mask].astype(int)
        y_test = y_win[test_mask].astype(int)
        
        # Drop constants for this fold
        fold_const = find_constant_cols(X_train, features)
        fold_features = [c for c in features if c not in fold_const]
        
        X_train = X_train[fold_features]
        X_calib = X_calib[fold_features]
        X_test = X_test[fold_features]
        
        # Coerce numeric
        X_train, [X_calib, X_test], _ = coerce_numeric_object_cols(
            X_train, [X_calib, X_test], min_parse_rate=config.numeric_object_parse_rate
        )
        
        # Build preprocessor
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        pre = make_preprocessor(num_cols, cat_cols)
        
        # Train classifier
        clf = Pipeline([
            ("pre", pre),
            ("clf", HistGradientBoostingClassifier(
                random_state=RANDOM_SEED,
                **CLF_PARAMS
            )),
        ])
        clf.fit(X_train, y_train)
        
        # Calibrate
        if len(np.unique(y_calib)) >= 2:
            cal = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit", n_jobs=config.n_jobs)
            cal.fit(X_calib, y_calib)
        else:
            cal = clf  # Skip calibration if single-class
        
        # Test
        p_test = cal.predict_proba(X_test)[:, 1]
        
        # Record OOS predictions
        test_df = df_complete.loc[test_mask, REPORT_COLS].copy()
        test_df["y_true"] = y_test.to_numpy()
        test_df["home_win_proba"] = p_test
        test_df["fold_calib_year"] = calib_year
        test_df["fold_test_year"] = test_year
        
        oos_predictions.append(test_df)
        folds_used += 1
        
        log.info(
            f"WF fold: train≤{calib_year-1} calib={calib_year} test={test_year} | "
            f"rows={len(test_df)} | const_dropped={len(fold_const)}"
        )
    
    if folds_used == 0:
        raise RuntimeError("No valid walk-forward folds found. Check season range.")
    
    # Aggregate OOS results
    oos_df = pd.concat(oos_predictions, axis=0).reset_index(drop=True)
    y_true_all = oos_df["y_true"].to_numpy()
    y_prob_all = oos_df["home_win_proba"].to_numpy()
    
    # Bootstrap CIs
    metrics = WalkForwardMetrics(
        n_folds=folds_used,
        oos_rows=len(oos_df),
        roc_auc=bootstrap_ci(y_true_all, y_prob_all, safe_roc_auc, config.bootstrap_samples, RANDOM_SEED),
        log_loss=bootstrap_ci(y_true_all, y_prob_all, lambda yt, yp: float(log_loss(yt, yp)), config.bootstrap_samples, RANDOM_SEED+1),
        brier=bootstrap_ci(y_true_all, y_prob_all, lambda yt, yp: float(brier_score_loss(yt, yp)), config.bootstrap_samples, RANDOM_SEED+2),
        ece=bootstrap_ci(y_true_all, y_prob_all, lambda yt, yp: compute_ece(yt, yp), config.bootstrap_samples, RANDOM_SEED+3),
        accuracy=float(((y_prob_all >= 0.5).astype(int) == y_true_all).mean()),
    )
    
    log.info(f"Walk-forward validation complete: {folds_used} folds, {len(oos_df)} OOS predictions")
    log.info(f"  ROC AUC: {metrics.roc_auc['point']:.4f} [{metrics.roc_auc['ci_low']:.4f}, {metrics.roc_auc['ci_high']:.4f}]")
    log.info(f"  LogLoss: {metrics.log_loss['point']:.4f}")
    log.info(f"  ECE:     {metrics.ece['point']:.4f}")
    
    return metrics, oos_df


# -----------------------------------------------------------------------------
# Main Training Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    """Main training orchestration."""
    parser = argparse.ArgumentParser(description="Enhanced Leak-Free NFL Model Trainer")
    parser.add_argument("--data", type=str, required=True, help="Path to game features CSV")
    parser.add_argument("--out", type=str, default="prod-models/models", help="Output directory for models")
    parser.add_argument("--near_empty_threshold", type=float, default=0.95)
    parser.add_argument("--complete_missing_max", type=float, default=0.20)
    parser.add_argument("--future_missing_min", type=float, default=0.95)
    parser.add_argument("--numeric_object_parse_rate", type=float, default=0.98)
    parser.add_argument("--walk_start_calib", type=int, default=2020)
    parser.add_argument("--walk_end_calib", type=int, default=2024)
    parser.add_argument("--bootstrap_samples", type=int, default=1500)
    parser.add_argument("--hp_n_iter", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=-1)
    
    args = parser.parse_args()
    
    config = Config(
        data_path=Path(args.data),
        out_dir=Path(args.out),
        near_empty_threshold=args.near_empty_threshold,
        complete_missing_max=args.complete_missing_max,
        future_missing_min=args.future_missing_min,
        numeric_object_parse_rate=args.numeric_object_parse_rate,
        walk_start_calib=args.walk_start_calib,
        walk_end_calib=args.walk_end_calib,
        bootstrap_samples=args.bootstrap_samples,
        hp_n_iter=args.hp_n_iter,
        n_jobs=args.n_jobs,
    )
    
    config.out_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    log.info("=" * 80)
    log.info(f"ENHANCED LEAK-FREE TRAINER | data={config.data_path} | out={config.out_dir}")
    log.info("=" * 80)
    
    # 1. Load data
    if not config.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.data_path}")
    
    df_raw = normalize_columns(pd.read_csv(config.data_path))
    if df_raw.empty:
        raise RuntimeError(f"Dataset is empty: {config.data_path}")
    
    # Validate required columns
    required = set(TIME_KEYS + [TARGET_HOME, TARGET_AWAY, CLASS_LABEL] + REPORT_COLS)
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df_raw = df_raw.sort_values(by=TIME_KEYS).reset_index(drop=True)
    ds_hash = dataset_hash(df_raw)
    
    # Split completed vs future
    complete_mask = df_raw[CLASS_LABEL].notna()
    df_complete = df_raw[complete_mask].copy().reset_index(drop=True)
    df_future = df_raw[~complete_mask].copy().reset_index(drop=True)
    
    y_home_all = df_complete[TARGET_HOME].copy()
    y_away_all = df_complete[TARGET_AWAY].copy()
    y_win_all = df_complete[CLASS_LABEL].map({True: 1, False: 0}).astype(int)
    
    log.info(f"Dataset: {len(df_raw)} total, {len(df_complete)} completed, {len(df_future)} future")
    
    # 2. Feature engineering
    log.info("-" * 80)
    log.info("Building leak-safe feature manifest...")
    features, manifest = build_feature_manifest(df_raw, df_future, config)
    manifest.coerced_numeric = []  # Will be populated during preprocessing
    
    # 3. Train HOME/AWAY regressors
    log.info("-" * 80)
    log.info("Training HOME/AWAY score regressors...")
    
    reg_mask = y_home_all.notna() & y_away_all.notna()
    X_reg = df_complete.loc[reg_mask, features].copy()
    y_home_reg = y_home_all[reg_mask].reset_index(drop=True)
    y_away_reg = y_away_all[reg_mask].reset_index(drop=True)
    
    # Drop constants for regression
    const_reg = find_constant_cols(X_reg, features)
    features_reg = [c for c in features if c not in const_reg]
    X_reg = X_reg[features_reg]
    
    # Numeric coercion
    X_reg, _, coerced = coerce_numeric_object_cols(X_reg, [], min_parse_rate=config.numeric_object_parse_rate)
    manifest.coerced_numeric = coerced
    
    # Build preprocessor
    cat_reg = X_reg.select_dtypes(include=["object", "category"]).columns.tolist()
    num_reg = [c for c in X_reg.columns if c not in cat_reg]
    pre_reg = make_preprocessor(num_reg, cat_reg)
    
    # Chronological split for regression
    holdout_idx = int(len(X_reg) * (1 - config.reg_holdout_ratio))
    X_train_reg, X_hold_reg = X_reg.iloc[:holdout_idx], X_reg.iloc[holdout_idx:]
    y_home_train, y_home_hold = y_home_reg.iloc[:holdout_idx], y_home_reg.iloc[holdout_idx:]
    y_away_train, y_away_hold = y_away_reg.iloc[:holdout_idx], y_away_reg.iloc[holdout_idx:]
    
    # Tune regressors
    home_model, home_params = tune_regressor(X_train_reg, y_home_train, pre_reg, "HOME", config)
    away_model, away_params = tune_regressor(X_train_reg, y_away_train, pre_reg, "AWAY", config)
    
    # Evaluate on holdout
    home_metrics = evaluate_regression(home_model, X_hold_reg, y_home_hold)
    away_metrics = evaluate_regression(away_model, X_hold_reg, y_away_hold)
    
    log.info(f"HOME holdout: MAE={home_metrics.mae:.2f}, RMSE={home_metrics.rmse:.2f}, R²={home_metrics.r2:.3f}")
    log.info(f"AWAY holdout: MAE={away_metrics.mae:.2f}, RMSE={away_metrics.rmse:.2f}, R²={away_metrics.r2:.3f}")
    
    # Save final regressors (trained on all regression data)
    home_model.fit(X_reg, y_home_reg)
    away_model.fit(X_reg, y_away_reg)
    
    dump(home_model, config.out_dir / "home_model.joblib")
    dump(away_model, config.out_dir / "away_model.joblib")
    log.info("Saved: home_model.joblib, away_model.joblib")
    
    # Extract fitted preprocessor
    fitted_pre = home_model.named_steps["pre"]
    dump(fitted_pre, config.out_dir / "preprocessor.joblib")
    log.info("Saved: preprocessor.joblib")
    
    # 4. Walk-forward validation for WIN classifier
    log.info("-" * 80)
    log.info(f"Walk-forward validation ({config.walk_start_calib}-{config.walk_end_calib})...")
    
    wf_metrics, oos_df = walk_forward_validate(df_complete, features, y_win_all, config)
    
    # Save OOS predictions
    oos_df.to_csv(config.out_dir / "walkforward_oos_predictions.csv", index=False)
    log.info("Saved: walkforward_oos_predictions.csv")
    
    # 5. Train final WIN classifier for production
    log.info("-" * 80)
    log.info("Training final production WIN classifier...")
    
    X_win = df_complete[features].copy()
    
    # Drop constants
    const_win = find_constant_cols(X_win, features)
    features_win = [c for c in features if c not in const_win]
    X_win = X_win[features_win]
    
    # Numeric coercion
    X_win, _, _ = coerce_numeric_object_cols(X_win, [], min_parse_rate=config.numeric_object_parse_rate)
    
    # Split for calibration (last 20% chronologically)
    calib_idx = int(len(X_win) * 0.80)
    X_train_win, X_calib_win = X_win.iloc[:calib_idx], X_win.iloc[calib_idx:]
    y_train_win, y_calib_win = y_win_all.iloc[:calib_idx], y_win_all.iloc[calib_idx:]
    
    # Build preprocessor
    cat_win = X_train_win.select_dtypes(include=["object", "category"]).columns.tolist()
    num_win = [c for c in X_train_win.columns if c not in cat_win]
    pre_win = make_preprocessor(num_win, cat_win)
    
    # Train base classifier
    win_clf = Pipeline([
        ("pre", pre_win),
        ("clf", HistGradientBoostingClassifier(
            random_state=RANDOM_SEED,
            **CLF_PARAMS
        )),
    ])
    win_clf.fit(X_train_win, y_train_win)
    
    # Calibrate
    if len(np.unique(y_calib_win)) >= 2:
        win_clf_cal = CalibratedClassifierCV(
            estimator=win_clf,
            method="sigmoid",
            cv="prefit",
            n_jobs=config.n_jobs
        )
        win_clf_cal.fit(X_calib_win, y_calib_win)
        calibration_method = "sigmoid_prefit"
    else:
        win_clf_cal = win_clf
        calibration_method = "none_single_class"
    
    # Save calibrated classifier
    dump(win_clf_cal, config.out_dir / "win_clf_calibrated.joblib")
    log.info("Saved: win_clf_calibrated.joblib")
    
    # 6. Generate metadata
    log.info("-" * 80)
    log.info("Generating metadata...")
    
    metadata = ProductionMetadata(
        training_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        training_duration_seconds=time.time() - t0,
        dataset_path=str(config.data_path),
        dataset_hash=ds_hash,
        random_seed=RANDOM_SEED,
        rows_total=len(df_raw),
        rows_completed=len(df_complete),
        rows_future=len(df_future),
        rows_train_reg=holdout_idx,
        rows_holdout_reg=len(X_hold_reg),
        feature_manifest=asdict(manifest),
        home_model_best_params=home_params,
        away_model_best_params=away_params,
        win_clf_params=CLF_PARAMS,
        home_metrics_holdout=asdict(home_metrics),
        away_metrics_holdout=asdict(away_metrics),
        walkforward_metrics=asdict(wf_metrics),
        production_ready=True,
        calibration_method=calibration_method,
    )
    
    metadata_path = config.out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    log.info(f"Saved: metadata.json")
    
    # 7. Summary
    log.info("=" * 80)
    log.info(f"TRAINING COMPLETE | Duration: {timer(t0)}")
    log.info(f"Models saved to: {config.out_dir.absolute()}")
    log.info(f"  - home_model.joblib, away_model.joblib, preprocessor.joblib")
    log.info(f"  - win_clf_calibrated.joblib, metadata.json")
    log.info(f"  - walkforward_oos_predictions.csv")
    log.info("=" * 80)
    
    # Final validation summary
    print("\n=== FINAL VALIDATION METRICS ===")
    print(f"OOS ROC AUC:  {wf_metrics.roc_auc['point']:.4f} [{wf_metrics.roc_auc['ci_low']:.4f}, {wf_metrics.roc_auc['ci_high']:.4f}]")
    print(f"OOS LogLoss:  {wf_metrics.log_loss['point']:.4f}")
    print(f"OOS Brier:    {wf_metrics.brier['point']:.4f}")
    print(f"OOS ECE:      {wf_metrics.ece['point']:.4f}")
    print(f"OOS Accuracy: {wf_metrics.accuracy:.4f}")
    print(f"\nProduction Ready: {metadata.production_ready}")


if __name__ == "__main__":
    main()
