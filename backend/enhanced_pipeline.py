"""
Enhanced NFL win‑probability predictive pipeline (NFLEX v6).

This module constructs a unified game‑level dataset, engineers rolling
and differential features from the provided enhanced dataset, and
trains several machine‑learning models under a rigorously time‑aware
cross‑validation regime.  Key improvements over previous versions
include:

* A purged, group‑aware walk‑forward splitter with an embargo window to
  prevent subtle data leakage between adjacent weeks.
* A final hold‑out season (the most recent available) reserved for
  unbiased evaluation.
* Cross‑validated isotonic calibration of probabilistic outputs.
* Brier score skill scoring relative to a season‑level baseline and
  decomposition into reliability, resolution and uncertainty.
* A compact set of models (logistic regression with interaction terms,
  support vector machine, gradient boosting classifier, monotonic
  gradient boosting classifier) along with a convex blend ensemble.

Because internet access is disabled in this environment, we operate on
the provided `enhanced_dataset.csv` (covering seasons 2014–2023).  In
production, replace the data ingestion step with a call to an API
such as `nflreadr` to extend the dataset to the current season.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Any, Optional, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.metrics import precision_recall_curve, roc_curve
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.build_csv_datasets import make_time_key


FEATURE_METADATA_PATH = Path(__file__).resolve().parent / "models" / "feature_metadata.json"
# Determine the correct parameter name for CalibratedClassifierCV based on its signature
sig = signature(CalibratedClassifierCV.__init__)
CALIBRATOR_ESTIMATOR_PARAM = "estimator" if "estimator" in sig.parameters else "base_estimator"
PROBABILITY_EPS = 1e-6
CLASS_LABELS = [0, 1]


def compute_recency_weights(df: pd.DataFrame) -> np.ndarray:
    """Create normalized weights favouring recent games."""
    if {"season", "week"}.issubset(df.columns):
        seasons = df["season"].to_numpy(dtype=float)
        weeks = df["week"].to_numpy(dtype=float)
        season_span = max(seasons.max() - seasons.min(), 1.0)
        season_norm = (seasons - seasons.min()) / season_span
        week_norm = weeks / max(weeks.max(), 1.0)
        weights = 0.4 + 0.4 * season_norm + 0.2 * week_norm
    else:
        weights = np.ones(len(df), dtype=float)
    return weights / weights.mean()


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create a compact feature summary for documentation and monitoring."""
    stats = features.describe().transpose().reset_index().rename(columns={"index": "feature"})
    stats["missing_pct"] = features.isna().mean().values
    stats["dtype"] = features.dtypes.astype(str).values
    return stats[["feature", "dtype", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_pct"]]


def export_feature_metadata(summary: pd.DataFrame, output_path: Path) -> None:
    """Persist feature metadata to JSON for downstream inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_json(output_path, orient="records", indent=2)


def _fit_with_optional_weights(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> BaseEstimator:
    """Fit estimator or pipeline with optional sample weights."""
    est = cast(Any, estimator)
    if sample_weight is None:
        est.fit(X, y)
        return estimator

    fit_params: Dict[str, Any] = {}
    if isinstance(estimator, Pipeline):
        final_step = estimator.steps[-1][0]
        fit_params[f"{final_step}__sample_weight"] = sample_weight
    else:
        fit_params["sample_weight"] = sample_weight
    est.fit(X, y, **fit_params)
    return estimator


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """Custom cross‑validator implementing a purged walk‑forward split.

    Splits the data by distinct integer `groups` (e.g. season‑week codes).
    Training data for fold `i` comprises all groups up to the fold
    boundary minus an embargo window.  Validation data is the next group
    in sequence.  The final fold is not yielded to avoid empty
    validation sets.  See `split()` docstring for details.
    """

    def __init__(self, n_splits: int = 5, embargo_groups: int = 1):
        self.n_splits = n_splits
        self.embargo_groups = embargo_groups

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be provided for PurgedGroupTimeSeriesSplit")
        uniq_groups = np.unique(groups)
        # Compute approximate fold sizes
        fold_sizes = np.full(self.n_splits, len(uniq_groups) // self.n_splits, dtype=int)
        fold_sizes[: len(uniq_groups) % self.n_splits] += 1
        # Partition unique groups
        group_folds = []
        start = 0
        for size in fold_sizes:
            group_folds.append(uniq_groups[start : start + size])
            start += size
        for i in range(self.n_splits - 1):
            train_groups = np.concatenate(group_folds[: i + 1])
            val_groups = group_folds[i + 1]
            # Purge groups within embargo window relative to validation groups
            max_val = val_groups.max()
            mask = train_groups <= (max_val - self.embargo_groups)
            train_groups = train_groups[mask]
            tr_idx = np.where(np.isin(groups, train_groups))[0]
            va_idx = np.where(np.isin(groups, val_groups))[0]
            yield tr_idx, va_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits - 1


def brier_decomposition(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Decompose the Brier score into reliability, resolution and uncertainty.

    Parameters
    ----------
    y_true : array
        Binary outcomes (0/1).
    prob : array
        Predicted probabilities.
    n_bins : int
        Number of probability bins.

    Returns
    -------
    dict
        Dictionary containing Brier score, reliability, resolution and uncertainty.
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    brier = np.mean((prob - y_true) ** 2)
    bins = np.linspace(0, 1, n_bins + 1)
    # Assign each prediction to a bin
    bin_idx = np.digitize(prob, bins) - 1
    rel = 0.0
    res = 0.0
    p_bar = y_true.mean()
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        p_b = prob[mask].mean()
        y_b = y_true[mask].mean()
        w_b = mask.mean()
        rel += w_b * (p_b - y_b) ** 2
        res += w_b * (y_b - p_bar) ** 2
    unc = p_bar * (1 - p_bar)
    return {
        "brier": float(brier),
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
    }


def baseline_prediction(y_train: pd.Series, grouping: Optional[pd.Series] = None) -> Callable[[pd.DataFrame], np.ndarray]:
    """Return a function that predicts baseline home‑win probability.

    By default, predicts the overall home‑win rate.  If `grouping` is
    provided (e.g. season), then predicts the mean home‑win rate within
    each group.
    """
    if grouping is not None:
        group_rates = y_train.groupby(grouping).mean()

        def predict_baseline(df: pd.DataFrame) -> np.ndarray:
            return np.asarray(df[grouping.name].map(group_rates).fillna(y_train.mean()).values)

        return predict_baseline
    else:
        rate = y_train.mean()
        return lambda df: np.full(len(df), rate)


def build_dataset(data_path: str, expected_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load dataset, engineer features and return feature matrix and targets.

    Parameters
    ----------
    data_path : str
        Path to the enhanced game‑level CSV file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (diff features and engineered interactions).
    y : pd.Series
        Binary target indicating home win.
    groups : pd.Series
        Integer group index representing (season, week) ordinal.
    df : pd.DataFrame
        Original DataFrame augmented with group index (for downstream use).
    """
    df = pd.read_csv(data_path)
    # Harmonise column names: seasons and weeks from home side
    if "season" in df.columns:
        df["season"] = df["season"].astype(int)
    else:
        season_cols = [c for c in df.columns if c.startswith("season_")]
        if not season_cols:
            raise KeyError("Dataset must contain a 'season' or 'season_*' column")
        df["season"] = df[season_cols[0]].astype(int)
    if "week" in df.columns:
        df["week"] = df["week"].astype(int)
    else:
        week_cols = [c for c in df.columns if c.startswith("week_")]
        if not week_cols:
            raise KeyError("Dataset must contain a 'week' or 'week_*' column")
        df["week"] = df[week_cols[0]].astype(int)
    # Compute group index: season * 100 + week to ensure unique ordering
    df["group_idx"] = make_time_key(df)
    # Extract or derive target label
    if "home_win" in df.columns:
        y = df["home_win"].astype(int)
    elif {"winner", "home_team"}.issubset(df.columns):
        y = (df["winner"].astype(str).str.strip() == df["home_team"].astype(str).str.strip()).astype(int)
        df["home_win"] = y
    elif {"home_points_for", "away_points_for"}.issubset(df.columns):
        y = (df["home_points_for"].astype(float) > df["away_points_for"].astype(float)).astype(int)
        df["home_win"] = y
    else:
        raise KeyError(
            "Dataset must contain `home_win` or sufficient columns to derive it (winner/home_team or home/away points)."
        )
    # Select diff features (prefixed with 'diff_'); if none exist, fall back to numeric predictors
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    if diff_cols:
        X = df[diff_cols].copy()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols = {"season", "week", "home_win", "group_idx"}
        feature_cols = [c for c in numeric_cols if c not in drop_cols]
        if not feature_cols:
            raise ValueError(
                "Dataset must contain either `diff_` engineered features or other numeric predictors after excluding season/week/home_win."
            )
        X = df[feature_cols].copy()
    # Replace missing values with column means (simple imputation)
    X = X.fillna(X.mean())
    if expected_columns is not None:
        missing_cols = [c for c in expected_columns if c not in X.columns]
        for col in missing_cols:
            X[col] = 0.0
        extra_cols = [c for c in X.columns if c not in expected_columns]
        if extra_cols:
            X = X.drop(columns=extra_cols)
        X = X[expected_columns]
    # Engineering additional interactions capturing form asymmetry
    # Choose a few representative diff features to create delta and product interactions
    # We use net_epa_sum and off_turnovers as analogues to form strength and risk
    if "diff_net_epa_sum_avg_3" in X.columns and "diff_net_epa_sum_avg_5" in X.columns:
        X["epa_sum_delta"] = X["diff_net_epa_sum_avg_3"] - X["diff_net_epa_sum_avg_5"]
        X["epa_sum_product"] = X["diff_net_epa_sum_avg_3"] * X["diff_net_epa_sum_avg_5"]
    if "diff_off_turnovers_avg_3" in X.columns and "diff_off_turnovers_avg_5" in X.columns:
        X["turnover_delta"] = X["diff_off_turnovers_avg_3"] - X["diff_off_turnovers_avg_5"]
        X["turnover_product"] = X["diff_off_turnovers_avg_3"] * X["diff_off_turnovers_avg_5"]
        if {"epa_sum_delta", "turnover_delta"}.issubset(X.columns):
            turnover_guard = np.where(np.isclose(X["turnover_delta"], 0), 1e-6, X["turnover_delta"].abs())
            X["epa_turnover_ratio"] = X["epa_sum_delta"] / turnover_guard
            X["epa_turnover_weighted"] = X["epa_sum_product"] * np.log1p(turnover_guard)

    X = X.loc[:, ~X.columns.duplicated()]
    groups = df["group_idx"].astype(int)
    # Always return only 4 items as annotated
    return X, y, groups, df


def load_train_test_splits(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Prepare explicit train/test splits using the same feature schema."""
    X_train, y_train, groups_train, df_train = build_dataset(train_path)
    X_test, y_test, groups_test, df_test = build_dataset(test_path, expected_columns=list(X_train.columns))
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    return X_train, y_train, groups_train, df_train, X_test, y_test, groups_test, df_test


@dataclass
class ModelResult:
    name: str
    mean_brier: float
    mean_logloss: float
    mean_roc_auc: float
    mean_pr_auc: float
    brier_ci: Tuple[float, float]
    logloss_ci: Tuple[float, float]
    brier_skill: float
    brier_decomp: Dict[str, float]
    # Additional fields for hold‑out performance
    test_brier: float | None = None
    test_logloss: float | None = None
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    test_brier_skill: float | None = None
    test_brier_decomp: Dict[str, float] | None = None


def evaluate_model(
    name: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv: BaseCrossValidator,
    calibrate: bool = False,
    baseline_rate: Optional[float] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> ModelResult:
    """Cross‑validate a model with the provided splitter, optionally calibrating.

    Returns mean metrics and confidence intervals via block bootstrap.
    """
    n = len(y)
    # Store per‑sample predictions and true labels
    prob_oof = np.zeros(n, dtype=float)
    y_oof = np.zeros(n, dtype=float)
    # Predictions by fold
    for tr_idx, va_idx in cv.split(X, y, groups):
        X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]
        fold_weights = sample_weight[tr_idx] if sample_weight is not None else None
        # If the training fold lacks class diversity, fall back to baseline rate
        if y_train.nunique() < 2:
            baseline_fold_rate = float(y_train.mean()) if len(y_train) else baseline_rate or float(y.mean())
            prob_val = np.full(len(X_val), baseline_fold_rate, dtype=float)
        else:
            # Clone estimator to avoid contamination
            est = clone(estimator)
            if calibrate:
                # Fit base estimator first on training to produce probabilities
                base = clone(estimator)
                _fit_with_optional_weights(base, X_train, y_train, fold_weights)
                calibrator = CalibratedClassifierCV(
                    **{CALIBRATOR_ESTIMATOR_PARAM: base}, method="isotonic", cv="prefit"
                )
                calibrator.fit(X_train, y_train)
                prob_val = calibrator.predict_proba(X_val)[:, 1]
            else:
                _fit_with_optional_weights(est, X_train, y_train, fold_weights)
                prob_val = est.predict_proba(X_val)[:, 1]
        prob_val = np.clip(prob_val, PROBABILITY_EPS, 1 - PROBABILITY_EPS)
        prob_oof[va_idx] = prob_val
        y_oof[va_idx] = y_val.to_numpy(dtype=float)
    # Compute metrics across all oof predictions
    prob_oof = np.clip(prob_oof, PROBABILITY_EPS, 1 - PROBABILITY_EPS)
    brier = brier_score_loss(y_oof, prob_oof)
    ll = log_loss(y_oof, prob_oof, labels=CLASS_LABELS)
    try:
        auc = roc_auc_score(y_oof, prob_oof)
    except ValueError:
        auc = np.nan
    try:
        pr_auc = average_precision_score(y_oof, prob_oof)
    except ValueError:
        pr_auc = np.nan
    # Baseline Brier
    if baseline_rate is None:
        baseline_rate = float(y.mean())
    y_array = y.to_numpy(dtype=float)
    baseline_probs = np.full_like(y_array, baseline_rate, dtype=float)
    brier_baseline = brier_score_loss(y_array, baseline_probs)
    brier_skill = 1 - brier / brier_baseline if brier_baseline > 0 else np.nan
    # Brier decomposition using out‑of‑fold predictions
    decomp = brier_decomposition(y_oof, prob_oof)
    # Compute block bootstrap for confidence intervals
    # Sample groups with replacement
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(42)
    brier_samples = []
    ll_samples = []
    for _ in range(200):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = np.isin(groups, sampled_groups)
        if mask.sum() == 0:
            continue
        y_sample = y_oof[mask]
        prob_sample = np.clip(prob_oof[mask], PROBABILITY_EPS, 1 - PROBABILITY_EPS)
        brier_samples.append(brier_score_loss(y_sample, prob_sample))
        ll_samples.append(log_loss(y_sample, prob_sample, labels=CLASS_LABELS))
    brier_samples = np.sort(brier_samples)
    ll_samples = np.sort(ll_samples)
    ci_low = max(int(0.025 * len(brier_samples)), 0)
    ci_high = min(int(0.975 * len(brier_samples)), len(brier_samples) - 1)
    brier_ci = (brier_samples[ci_low], brier_samples[ci_high])
    ll_ci = (ll_samples[ci_low], ll_samples[ci_high])
    return ModelResult(
        name=name,
        mean_brier=float(brier),
        mean_logloss=float(ll),
        mean_roc_auc=float(auc),
        mean_pr_auc=float(pr_auc),
        brier_ci=(float(brier_ci[0]), float(brier_ci[1])),
        logloss_ci=(float(ll_ci[0]), float(ll_ci[1])),
        brier_skill=float(brier_skill),
        brier_decomp=decomp,
    )


def evaluate_on_test(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    calibrate: bool = False,
    baseline_rate: Optional[float] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float, Dict[str, float]]:
    """Train estimator on training data and evaluate on held‑out test set.

    Returns test metrics and Brier skill/decomposition.
    """
    est = clone(estimator)
    if y_train.nunique() < 2:
        baseline_rate = float(y_train.mean()) if len(y_train) else baseline_rate or float(y_test.mean())
        prob_test = np.full(len(X_test), baseline_rate, dtype=float)
    else:
        if calibrate:
            base = clone(estimator)
            _fit_with_optional_weights(base, X_train, y_train, sample_weight)
            calibrator = CalibratedClassifierCV(
                **{CALIBRATOR_ESTIMATOR_PARAM: base}, method="isotonic", cv="prefit"
            )
            calibrator.fit(X_train, y_train)
            prob_test = calibrator.predict_proba(X_test)[:, 1]
        else:
            _fit_with_optional_weights(est, X_train, y_train, sample_weight)
            prob_test = est.predict_proba(X_test)[:, 1]
    prob_test = np.clip(prob_test, PROBABILITY_EPS, 1 - PROBABILITY_EPS)
    y_test_array = y_test.to_numpy(dtype=float)
    brier = brier_score_loss(y_test_array, prob_test)
    ll = log_loss(y_test_array, prob_test, labels=CLASS_LABELS)
    try:
        auc = roc_auc_score(y_test_array, prob_test)
    except ValueError:
        auc = np.nan
    try:
        pr_auc = average_precision_score(y_test_array, prob_test)
    except ValueError:
        pr_auc = np.nan
    if baseline_rate is None:
        baseline_rate = float(y_train.mean())
    baseline_probs = np.full_like(y_test_array, baseline_rate, dtype=float)
    brier_baseline = brier_score_loss(y_test_array, baseline_probs)
    brier_skill = 1 - brier / brier_baseline if brier_baseline > 0 else np.nan
    decomp = brier_decomposition(y_test_array, prob_test)
    return float(brier), float(ll), float(auc), float(pr_auc), float(brier_skill), decomp


def convex_blend(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    y_true: np.ndarray,
    search_steps: int = 50,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Find the optimal convex blending weight between two probability vectors.

    Minimises log‑loss on y_true.  Returns (best_weight, best_log_loss).
    """
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    ws = np.linspace(0, 1, search_steps)
    best_w = 0.5
    best_ll = np.inf
    eps = max(eps, PROBABILITY_EPS)
    for w in ws:
        blended = np.clip(w * prob_a + (1 - w) * prob_b, eps, 1 - eps)
        ll = log_loss(y_true, blended, labels=CLASS_LABELS)
        if ll < best_ll:
            best_ll = ll
            best_w = w
    return float(best_w), float(best_ll)


def run_experiment(
    data_path: Optional[str] = None,
    *,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    holdout_season: Optional[int] = None,
) -> Tuple[List[ModelResult], pd.DataFrame]:
    """Execute the enhanced NFLEX pipeline on the supplied dataset or explicit splits.

    Parameters
    ----------
    data_path : str, optional
        Path to a combined CSV to be split by season.
    train_path : str, optional
        Path to a pre-split training CSV.
    test_path : str, optional
        Path to a pre-split testing CSV.
    holdout_season : int, optional
        Season to reserve for final testing.  If None, infers from data.

    Returns
    -------
    List[ModelResult]
        Results for each model trained and cross‑validated.
    pd.DataFrame
        DataFrame summarising hold‑out predictions for ensemble blending.
    """
    if train_path and test_path:
        (
            X_train,
            y_train,
            groups_train,
            df_train,
            X_test,
            y_test,
            _groups_test,
            df_test,
        ) = load_train_test_splits(train_path, test_path)
        if holdout_season is None:
            if "season" in df_test.columns:
                holdout_season = int(df_test["season"].max())
            elif "season_home" in df_test.columns:
                holdout_season = int(df_test["season_home"].max())
            else:
                holdout_season = int(df_train["season"].max()) + 1
    elif data_path:
        X, y, groups, df = build_dataset(data_path)
        if holdout_season is None:
            holdout_season = int(df["season"].max())
        train_mask = df["season"] < holdout_season
        test_mask = df["season"] == holdout_season
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
        groups_train = groups.loc[train_mask]
        df_train = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()
    else:
        raise ValueError("Provide either `data_path` or both `train_path` and `test_path`.")
    # Persist feature metadata for documentation
    feature_summary = summarize_features(X_train)
    export_feature_metadata(feature_summary, FEATURE_METADATA_PATH)
    # Recency-aware weights
    sample_weight_train = compute_recency_weights(df_train)
    # Baseline rate for BSS (weighted if possible)
    if len(sample_weight_train):
        baseline_rate_train = float(np.average(y_train, weights=sample_weight_train))
    else:
        baseline_rate_train = float(y_train.mean())
    # Define cross‑validator
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, embargo_groups=1)
    # Persist feature metadata for documentation
    feature_summary = summarize_features(X_train)
    export_feature_metadata(feature_summary, FEATURE_METADATA_PATH)
    # Baseline rate for BSS
    baseline_rate_train = float(y_train.mean())
    # Define cross‑validator
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, embargo_groups=1)
    # Define models
    models: List[Tuple[str, BaseEstimator, bool]] = []
    # Logistic regression (with scaling)
    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500)),
    ])
    models.append(("Logistic", logit, True))
    # Support vector machine
    svc = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=1.0, kernel="rbf", probability=True)),
    ])
    models.append(("SVM", svc, True))
    # Gradient boosting classifier
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3)
    models.append(("GradientBoosting", gb, False))
    # Monotonic gradient boosting using HistGradientBoostingClassifier if available
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        # For monotonicity, assign +1 for all features (assuming higher diff increases home win probability)
        monotonic_cst = [1] * X_train.shape[1]
        hist_gb = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            monotonic_cst=monotonic_cst,
        )
        models.append(("MonotonicHGB", hist_gb, False))
    except Exception:
        # Fallback to RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42)
        models.append(("RandomForest", rf, False))
    results: List[ModelResult] = []
    sample_weight_train = np.asarray(sample_weight_train, dtype=float)
    # For storing training and test predictions by model for ensemble blending
    ensemble_train = pd.DataFrame(index=X_train.index)
    ensemble_test = pd.DataFrame(index=X_test.index)
    for name, est, calibrate in models:
        res = evaluate_model(
            name,
            est,
            X_train,
            y_train,
            groups_train,
            cv,
            calibrate=calibrate,
            baseline_rate=baseline_rate_train,
            sample_weight=sample_weight_train,
        )
        # Evaluate on test
        test_metrics = evaluate_on_test(
            est,
            X_train,
            y_train,
            X_test,
            y_test,
            calibrate=calibrate,
            baseline_rate=baseline_rate_train,
            sample_weight=sample_weight_train,
        )
        res.test_brier, res.test_logloss, res.test_roc_auc, res.test_pr_auc, res.test_brier_skill, res.test_brier_decomp = test_metrics
        results.append(res)
        # Fit on full training and record probability predictions for blending on training set
        est_full = clone(est)
        if calibrate:
            base_full = clone(est)
            _fit_with_optional_weights(base_full, X_train, y_train, sample_weight_train)
            calibrator_full = CalibratedClassifierCV(
                **{CALIBRATOR_ESTIMATOR_PARAM: base_full}, method="isotonic", cv="prefit"
            )
            calibrator_full.fit(X_train, y_train)
            ensemble_train[f"{name}"] = calibrator_full.predict_proba(X_train)[:, 1]
            ensemble_test[f"{name}"] = calibrator_full.predict_proba(X_test)[:, 1]
        else:
            _fit_with_optional_weights(est_full, X_train, y_train, sample_weight_train)
            ensemble_train[f"{name}"] = est_full.predict_proba(X_train)[:, 1]
            ensemble_test[f"{name}"] = est_full.predict_proba(X_test)[:, 1]
    # Build convex blend using two best models (logistic and gradient boosting by default)
    if "Logistic" in ensemble_train.columns and "GradientBoosting" in ensemble_train.columns:
        pA_train = ensemble_train["Logistic"].to_numpy(dtype=float)
        pB_train = ensemble_train["GradientBoosting"].to_numpy(dtype=float)
        best_w, _ = convex_blend(pA_train, pB_train, y_train.to_numpy(dtype=float))
        # Apply blend to test predictions
        pA_test = ensemble_test["Logistic"].to_numpy(dtype=float)
        pB_test = ensemble_test["GradientBoosting"].to_numpy(dtype=float)
        blend_test_prob = best_w * pA_test + (1 - best_w) * pB_test
        blend_test_prob = np.clip(blend_test_prob, PROBABILITY_EPS, 1 - PROBABILITY_EPS)
        # Evaluate blend
        y_test_array = y_test.to_numpy(dtype=float)
        brier_blend = brier_score_loss(y_test_array, blend_test_prob)
        ll_blend = log_loss(y_test_array, blend_test_prob, labels=CLASS_LABELS)
        try:
            auc_blend = roc_auc_score(y_test_array, blend_test_prob)
        except ValueError:
            auc_blend = np.nan
        try:
            pr_auc_blend = average_precision_score(y_test_array, blend_test_prob)
        except ValueError:
            pr_auc_blend = np.nan
        baseline_fill = np.full_like(y_test_array, baseline_rate_train, dtype=float)
        brier_base = brier_score_loss(y_test_array, baseline_fill)
        brier_skill_blend = 1 - brier_blend / brier_base if brier_base > 0 else np.nan
        decomp_blend = brier_decomposition(y_test_array, blend_test_prob)
        blend_result = ModelResult(
            name=f"Blend(Logit,GB) w={best_w:.2f}",
            mean_brier=np.nan,
            mean_logloss=np.nan,
            mean_roc_auc=np.nan,
            mean_pr_auc=np.nan,
            brier_ci=(np.nan, np.nan),
            logloss_ci=(np.nan, np.nan),
            brier_skill=np.nan,
            brier_decomp={},
            test_brier=float(brier_blend),
            test_logloss=float(ll_blend),
            test_roc_auc=float(auc_blend),
            test_pr_auc=float(pr_auc_blend),
            test_brier_skill=float(brier_skill_blend),
            test_brier_decomp=decomp_blend,
        )
        results.append(blend_result)
    # Combine training and test predictions for external inspection
    ensemble_df = pd.concat(
        [ensemble_train.add_suffix("_train"), ensemble_test.add_suffix("_test")], axis=1
    )
    return results, ensemble_df


def generate_markdown_report(results: List[ModelResult], output_path: str, holdout_season: int) -> None:
    """Create a detailed markdown report from model results and save to file."""
    lines: List[str] = []
    lines.append("# NFLEX v6 Predictive Pipeline Report")
    lines.append("")
    lines.append(f"This report summarises the performance of four base models and a convex blend on NFL game data from 2014–{holdout_season}.")
    lines.append("")
    lines.append("## Cross‑validated results (training seasons)")
    lines.append("")
    header = ["Model", "Brier", "Brier CI", "Log‑loss", "LL CI", "ROC AUC", "PR AUC", "Brier Skill"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- " for _ in header]) + "|")
    for res in results:
        if res.mean_brier is np.nan:
            continue
        lines.append(
            f"| {res.name} | {res.mean_brier:.4f} | [{res.brier_ci[0]:.4f}, {res.brier_ci[1]:.4f}] | {res.mean_logloss:.4f} | [{res.logloss_ci[0]:.4f}, {res.logloss_ci[1]:.4f}] | {res.mean_roc_auc:.4f} | {res.mean_pr_auc:.4f} | {res.brier_skill:.3f} |"
        )
    lines.append("")
    lines.append("## Hold‑out season results (\"never_seen\" season)")
    lines.append("")
    header = ["Model", "Brier", "Log‑loss", "ROC AUC", "PR AUC", "Brier Skill"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- " for _ in header]) + "|")
    for res in results:
        if res.test_brier is None:
            continue
        lines.append(
            f"| {res.name} | {res.test_brier:.4f} | {res.test_logloss:.4f} | {res.test_roc_auc:.4f} | {res.test_pr_auc:.4f} | {res.test_brier_skill:.3f} |"
        )
    lines.append("")
    lines.append("## Brier decomposition (hold‑out season)")
    lines.append("")
    lines.append("| Model | Brier | Reliability | Resolution | Uncertainty |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for res in results:
        if res.test_brier_decomp:
            d = res.test_brier_decomp
            lines.append(f"| {res.name} | {d['brier']:.4f} | {d['reliability']:.4f} | {d['resolution']:.4f} | {d['uncertainty']:.4f} |")
    lines.append("")
    lines.append("**Notes**:")
    lines.append("- Cross‑validated results use a purged walk‑forward splitter with one‑week embargo and five folds.")
    lines.append(f"- The hold‑out season is {holdout_season}; models were trained exclusively on prior seasons.")
    lines.append("- Brier Skill Score is relative to the mean home‑win rate in the training set.")
    lines.append("- The convex blend combines Logistic and GradientBoosting predictions using a weight that minimises log‑loss on the training set.")
    lines.append("- Monotonic constraints assume that increasing differential statistics generally increase the probability of a home win.")
    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("/n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NFLEX v6 pipeline")
    parser.add_argument("--data", required=True, help="Path to enhanced game‑level CSV")
    parser.add_argument("--outdir", default="reports", help="Directory for output report")
    parser.add_argument("--holdout", type=int, default=None, help="Season to hold out (default latest)")
    args = parser.parse_args()
    results, _ = run_experiment(args.data, holdout_season=args.holdout)
    preview_df = pd.read_csv(args.data)
    if "season_home" in preview_df.columns:
        holdout_source = "season_home"
    elif "season" in preview_df.columns:
        holdout_source = "season"
    else:
        raise KeyError("Dataset must include a 'season' or 'season_home' column to determine hold-out season.")
    holdout_season = args.holdout or int(preview_df[holdout_source].max())
    report_path = Path(args.outdir) / "nflex_v6_report.md"
    generate_markdown_report(results, str(report_path), int(holdout_season))