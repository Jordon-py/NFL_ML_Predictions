"""
Enhanced NFL win-probability predictive pipeline (NFLEX v6).

This module builds a unified game-level dataset, engineers rolling/differential
features (assumed precomputed in the enhanced CSV), and trains several ML models
under a purged, group-aware walk-forward CV with an embargo to reduce leakage.

Key properties
--------------
• Purged walk-forward CV over (season, week) groups with an embargo window.
• Optional final hold-out season (never-seen) OR production mode using all data.
• Optional isotonic calibration applied per fold (train-only) to avoid leakage.
• Brier Skill Score (relative to baseline) and Brier decomposition (reliability,
  resolution, uncertainty) provided for deeper diagnostic insight.
• Compact model zoo: Logistic (with scaling), SVM (with scaling), GB, and a
  monotonic HGB fallbacking to RandomForest when HGB is unavailable.
• Simple convex blend of top two models (Logit, GB) on the hold-out set.

Notes on calibration
--------------------
We use CalibratedClassifierCV with cv='prefit' inside each fold:
  1) fit base estimator on the *training* split
  2) pass that prefit estimator to the calibrator
  3) calibrator.fit(X_train, y_train) learns calibration mapping on train only
  4) predict_proba on validation split → calibrated, leakage-minimized OOF preds
A fully nested CV would be even stricter but heavier; this is a pragmatic middle.

CLI
---
python enhanced_pipeline.py --data path/to/enhanced_dataset.csv [--holdout 2024]
                            [--outdir reports] [--production]

• --production trains on all rows (no hold-out block; report omitted accordingly).

"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
import json
from inspect import signature
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Any, Optional, cast, Iterable

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
import joblib

# (Optional) Unused imports removed for cleanliness:
# from sklearn.metrics import precision_recall_curve, roc_curve
# from scipy.special import expit

# -----------------------------
# Project path (import helpers)
# -----------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except IndexError:
    PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Optional import; safe to remove if not needed.
try:
    from backend.build_csv_datasets import make_time_key  # noqa: F401
except Exception:
    pass

FEATURE_METADATA_PATH = Path(__file__).resolve().parent / "models" / "feature_metadata.json"
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Calibrator signature compat (scikit-learn changed param name over time)
sig = signature(CalibratedClassifierCV.__init__)
CALIBRATOR_ESTIMATOR_PARAM = "estimator" if "estimator" in sig.parameters else "base_estimator"

# Numerics / labels
PROBABILITY_EPS = 1e-6
CLASS_LABELS = [0, 1]
RNG_SEED = 42


# ==============================
# Utilities & Feature Summaries
# ==============================
def compute_recency_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Create normalized weights favoring recent games.
    If 'season'/'week' are missing (should not be after harmonization),
    returns uniform weights (mean = 1).
    """
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
    """
    Compact feature summary for documentation/monitoring.
    Assumes features are numeric by construction (see build_dataset).
    """
    stats = features.describe().transpose().reset_index().rename(columns={"index": "feature"})
    stats["missing_pct"] = features.isna().mean().values
    stats["dtype"] = features.dtypes.astype(str).values
    return stats[
        ["feature", "dtype", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_pct"]
    ]


def export_feature_metadata(summary: pd.DataFrame, output_path: Path) -> None:
    """Persist feature metadata to JSON for downstream inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # pandas >= 1.1 supports indent; if older, remove indent param.
    summary.to_json(output_path, orient="records", indent=2)


# ==========================
# Leakage guard definitions
# ==========================
def is_leak_feature(name: str, *, allow_prefixes: Iterable[str] = ("prior_", "home_prior_", "away_prior_", "diff_", "home_minus_away_", "trend_")) -> bool:
    """
    Return True if a column name is considered leakage-prone for training.

    Rules (conservative):
    - Any column starting with '_' is reserved/diagnostic (often target-derived) → drop.
    - Explicit forbidden names known to encode outcomes or empirical win rates derived from labels.
    - Bare points-for/against without explicit prior/diff/trend context are dropped in build_dataset, not here.

    allow_prefixes exists to avoid flagging legitimate engineered pregame features that may contain
    strings like 'win_pct' but are computed from prior windows.
    """
    if not isinstance(name, str):
        return False
    n = name.strip()
    if not n:
        return False

    # Allow-list prefixes take precedence
    if any(n.startswith(p) for p in allow_prefixes):
        return False

    # 1) Underscore-prefixed diagnostics/targets
    if n.startswith("_"):
        return True

    # 2) Explicit known leakage terms
    forbidden_exact = {
        "home_win",  # target itself
        "winner", "winner_team",
        "home_win_prob", "away_win_prob",
        "season_home_win_rate",  # risky unless time-sliced; conservatively drop
        "_home_win_derived", "_dom_delta_emp_home_win", "_dom_delta",
    }
    if n in forbidden_exact:
        return True

    # 3) Patterns indicating empirical outcome mapping
    if "emp_home_win" in n or "derived_win" in n:
        return True

    return False


def _fit_with_optional_weights(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
) -> BaseEstimator:
    """
    Fit estimator or pipeline with optional sample weights. For Pipeline, forwards
    weights to the final step as '<finalstep>__sample_weight'.
    """
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


# ===========================================
# Purged, Group-aware Time Series Split (CV)
# ===========================================
class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Purged walk-forward split over integer group IDs (e.g., season*100+week).

    For fold i:
      • Train on groups up to the boundary of fold i (excluding an embargo window)
      • Validate on the next (i+1) group block
    The last block is not yielded as a validation fold.
    """

    def __init__(self, n_splits: int = 5, embargo_groups: int = 1):
        self.n_splits = n_splits
        self.embargo_groups = embargo_groups

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be provided for PurgedGroupTimeSeriesSplit")
        uniq_groups = np.unique(groups)  # sorted ascending
        # Slice unique groups into n_splits contiguous blocks
        fold_sizes = np.full(self.n_splits, len(uniq_groups) // self.n_splits, dtype=int)
        fold_sizes[: len(uniq_groups) % self.n_splits] += 1

        group_folds = []
        start = 0
        for size in fold_sizes:
            group_folds.append(uniq_groups[start : start + size])
            start += size

        for i in range(self.n_splits - 1):
            train_groups = np.concatenate(group_folds[: i + 1])
            val_groups = group_folds[i + 1]
            # Embargo: remove groups too close to the validation max group
            max_val = val_groups.max()
            mask = train_groups <= (max_val - self.embargo_groups)
            train_groups = train_groups[mask]

            tr_idx = np.where(np.isin(groups, train_groups))[0]
            va_idx = np.where(np.isin(groups, val_groups))[0]
            yield tr_idx, va_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits - 1


# ===========================
# Brier utilities & baselines
# ===========================
def brier_decomposition(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Decompose Brier into reliability, resolution, uncertainty."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    brier = np.mean((prob - y_true) ** 2)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(prob, bins) - 1  # last bin inclusive of prob==1

    rel, res = 0.0, 0.0
    p_bar = y_true.mean()
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        p_b = prob[mask].mean()
        y_b = y_true[mask].mean()
        w_b = mask.mean()  # fraction of samples in this bin
        rel += w_b * (p_b - y_b) ** 2
        res += w_b * (y_b - p_bar) ** 2
    unc = p_bar * (1 - p_bar)
    return {"brier": float(brier), "reliability": float(rel), "resolution": float(res), "uncertainty": float(unc)}


def baseline_prediction(y_train: pd.Series, grouping: Optional[pd.Series] = None) -> Callable[[pd.DataFrame], np.ndarray]:
    """
    Baseline home-win rate function:
      • global rate if grouping is None
      • per-group (e.g., per season) rate otherwise
    """
    if grouping is not None:
        group_rates = y_train.groupby(grouping).mean()

        def predict_baseline(df: pd.DataFrame) -> np.ndarray:
            return np.asarray(df[grouping.name].map(group_rates).fillna(y_train.mean()).values)

        return predict_baseline
    else:
        rate = y_train.mean()
        return lambda df: np.full(len(df), rate)


# ==========================
# Data loading & processing
# ==========================
def _extract_season_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'season' and 'week' exist as integer-typed columns.
    Falls back to first 'season_*' / 'week_*' if canonical names are missing.
    """
    # season
    if "season" in df.columns:
        df["season"] = df["season"].astype(int)
    else:
        season_cols = [c for c in df.columns if c.startswith("season_")]
        if not season_cols:
            raise KeyError("Dataset must contain a 'season' or 'season_*' column")
        df["season"] = df[season_cols[0]].astype(int)

    # week
    if "week" in df.columns:
        df["week"] = df["week"].astype(int)
    else:
        week_cols = [c for c in df.columns if c.startswith("week_")]
        if not week_cols:
            raise KeyError("Dataset must contain a 'week' or 'week_*' column")
        df["week"] = df[week_cols[0]].astype(int)

    return df


def build_dataset(
    data_path: str,
    expected_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Load and harmonize the dataset, return (X, y, groups, df_raw).

    Target derivation priority:
      1) 'home_win' exists → use it
      2) ('winner','home_team') → home_win = (winner == home_team)
      3) ('home_points_for','away_points_for') → home_win = (home_points_for > away_points_for)

    Feature selection:
      • Prefer engineered 'diff_*' columns.
      • Otherwise all numeric predictors excluding {season, week, home_win, group_idx}.
      • Mean imputation for simplicity; pipelines downstream can be extended for MICE/median.
    """
    df = pd.read_csv(data_path)
    df = _extract_season_week(df)

    # Group index (season*100 + week) enforces chronological order across seasons.
    df["group_idx"] = df["season"].astype(int) * 100 + df["week"].astype(int)

    # Target (defer casting to int until after NaN filtering)
    if "home_win" in df.columns:
        y_raw = df["home_win"]
    elif {"winner", "home_team"}.issubset(df.columns):
        y_raw = (df["winner"].astype(str).str.strip() == df["home_team"].astype(str).str.strip()).astype(float)
        df["home_win"] = y_raw
    elif {"home_points_for", "away_points_for"}.issubset(df.columns):
        y_raw = (df["home_points_for"].astype(float) > df["away_points_for"].astype(float)).astype(float)
        df["home_win"] = y_raw
    else:
        raise KeyError(
            "Dataset must contain `home_win` or sufficient columns to derive it "
            "(winner/home_team or home_points_for/away_points_for)."
        )

    # Filter out rows with missing targets (future/incomplete games), then cast to int
    valid_mask = pd.Series(True, index=df.index)
    if y_raw is not None:
        valid_mask = pd.Series(y_raw).notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    y = pd.Series(y_raw).loc[valid_mask].reset_index(drop=True).astype(int)

    # Feature frame on filtered df
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    if diff_cols:
        X = df[diff_cols].copy()
    else:
        # Broad numeric selection, then explicitly drop ANY post-game/outcome columns to prevent leakage.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols = {
            "season", "week", "home_win", "group_idx",
            # Outcome columns (post-game): must not leak into training features
            "home_points_for", "away_points_for", "point_diff",
        }
        # Drop any numeric columns that strongly imply outcome if present
        leak_like_prefixes = (
            # generalized points columns without prior_ guard
            "points_for", "points_against",
        )
        leak_like_exact = {"winner", "winner_team", "home_win_prob", "away_win_prob"}
        safe_cols: List[str] = []
        for c in numeric_cols:
            if c in drop_cols:
                continue
            if c in leak_like_exact:
                continue
            # Centralized leakage guard (conservative)
            if is_leak_feature(c):
                continue
            # Exclude bare points_* columns unless they are properly prefixed as priors/diffs/trends
            if any(c.startswith(pref) for pref in ("home_points_for", "away_points_for")) and not (
                c.startswith("home_prior_") or c.startswith("away_prior_") or c.startswith("diff_") or c.startswith("home_minus_away_")
            ):
                continue
            if any(c.endswith(suf) for suf in ("_points_for", "_points_against")) and not c.startswith("prior_"):
                continue
            # keep
            safe_cols.append(c)
        if not safe_cols:
            raise ValueError(
                "Dataset must contain either `diff_` engineered features or other numeric predictors "
                "that are pre-game (no outcome leakage)."
            )
        X = df[safe_cols].copy()

    # Imputation: column means where available, then 0.0 for any remaining NaNs
    X = X.fillna(X.mean(numeric_only=True))
    if X.isna().any().any():
        X = X.fillna(0.0)

    # Optional schema enforcement (test must match train)
    if expected_columns is not None:
        missing_cols = [c for c in expected_columns if c not in X.columns]
        for col in missing_cols:
            X[col] = 0.0
        extra_cols = [c for c in X.columns if c not in expected_columns]
        if extra_cols:
            X = X.drop(columns=extra_cols)
        X = X[expected_columns]

    # Small, interpretable interactions (if available)
    if "diff_net_epa_sum_avg_3" in X.columns and "diff_net_epa_sum_avg_5" in X.columns:
        X["epa_sum_delta"] = X["diff_net_epa_sum_avg_3"] - X["diff_net_epa_sum_avg_5"]
        X["epa_sum_product"] = X["diff_net_epa_sum_avg_3"] * X["diff_net_epa_sum_avg_5"]
    if "diff_off_turnovers_avg_3" in X.columns and "diff_off_turnovers_avg_5" in X.columns:
        X["turnover_delta"] = X["diff_off_turnovers_avg_3"] - X["diff_off_turnovers_avg_5"]
        X["turnover_product"] = X["diff_off_turnovers_avg_3"] * X["diff_off_turnovers_avg_5"]
        if {"epa_sum_delta", "turnover_delta"}.issubset(X.columns):
            turnover_guard = np.where(np.isclose(X["turnover_delta"], 0), 1e-6, X["turnover_delta"].abs())
            X["epa_turnover_ratio"] = X["epa_sum_delta"] / turnover_guard
            if "epa_sum_product" in X.columns:
                X["epa_turnover_weighted"] = X["epa_sum_product"] * np.log1p(turnover_guard)

    # Remove accidental duplicates
    X = X.loc[:, ~X.columns.duplicated()]

    groups = df["group_idx"].astype(int)
    return X, y, groups, df


def load_train_test_splits(
    train_path: str,
    test_path: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Prepare explicit train/test splits using the same feature schema."""
    X_train, y_train, groups_train, df_train = build_dataset(train_path)
    X_test, y_test, groups_test, df_test = build_dataset(test_path, expected_columns=list(X_train.columns))
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    return X_train, y_train, groups_train, df_train, X_test, y_test, groups_test, df_test


# =======================
# Results data structure
# =======================
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
    # Hold-out performance
    test_brier: float | None = None
    test_logloss: float | None = None
    test_roc_auc: float | None = None
    test_pr_auc: float | None = None
    test_brier_skill: float | None = None
    test_brier_decomp: Dict[str, float] | None = None


# ======================
# Evaluation primitives
# ======================
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
    """
    Cross-validate a model with the provided splitter. If `calibrate=True`,
    fit a prefit calibrator on the fold's *training* data to avoid leakage.
    """
    n = len(y)
    prob_oof = np.zeros(n, dtype=float)
    y_oof = np.zeros(n, dtype=float)

    for tr_idx, va_idx in cv.split(X, y, groups):
        X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]
        fold_weights = sample_weight[tr_idx] if sample_weight is not None else None

        if y_train.nunique() < 2:
            baseline_fold_rate = float(y_train.mean()) if len(y_train) else baseline_rate or float(y.mean())
            prob_val = np.full(len(X_val), baseline_fold_rate, dtype=float)
        else:
            if calibrate:
                base = clone(estimator)
                _fit_with_optional_weights(base, X_train, y_train, fold_weights)
                calibrator = CalibratedClassifierCV(
                    **{CALIBRATOR_ESTIMATOR_PARAM: base}, method="isotonic", cv="prefit"
                )
                # Calibrate on TRAIN ONLY to avoid leakage
                calibrator.fit(X_train, y_train)
                prob_val = calibrator.predict_proba(X_val)[:, 1]
            else:
                est = clone(estimator)
                _fit_with_optional_weights(est, X_train, y_train, fold_weights)
                prob_val = est.predict_proba(X_val)[:, 1]

        prob_val = np.clip(prob_val, PROBABILITY_EPS, 1 - PROBABILITY_EPS)
        prob_oof[va_idx] = prob_val
        y_oof[va_idx] = y_val.to_numpy(dtype=float)

    # Aggregate OOF metrics
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

    # Baseline Brier → Skill
    if baseline_rate is None:
        baseline_rate = float(y.mean())
    y_array = y.to_numpy(dtype=float)
    baseline_probs = np.full_like(y_array, baseline_rate, dtype=float)
    brier_baseline = brier_score_loss(y_array, baseline_probs)
    brier_skill = 1 - brier / brier_baseline if brier_baseline > 0 else np.nan

    # Decomposition
    decomp = brier_decomposition(y_oof, prob_oof)

    # Block bootstrap CI over groups
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(RNG_SEED)
    brier_samples, ll_samples = [], []
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
    """
    Fit on all training rows and evaluate on the hold-out test set.
    Mirrors the calibrator behavior used during CV.
    """
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
            est = clone(estimator)
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
    """
    Simple convex weight search minimizing log-loss.
    Returns (best_weight, best_log_loss).
    """
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    ws = np.linspace(0, 1, search_steps)
    best_w, best_ll = 0.5, np.inf
    eps = max(eps, PROBABILITY_EPS)
    for w in ws:
        blended = np.clip(w * prob_a + (1 - w) * prob_b, eps, 1 - eps)
        ll = log_loss(y_true, blended, labels=CLASS_LABELS)
        if ll < best_ll:
            best_ll, best_w = ll, w
    return float(best_w), float(best_ll)


# ======================
# Experiment orchestration
# ======================
def run_experiment(
    data_path: Optional[str] = None,
    *,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    holdout_season: Optional[int] = None,
    holdout_week: Optional[int] = None,
    holdout_week_end: Optional[int] = None,
) -> Tuple[List[ModelResult], pd.DataFrame]:
    """
    Execute the NFLEX pipeline.

    Inputs:
      • Either a single combined CSV via `data_path` (and inferred/explicit hold-out),
        or explicit train/test CSVs via `train_path`/`test_path`.
      • Set `holdout_season=None` explicitly to run in production mode.
      • If week-aware holdout is used, provide holdout_week and optional holdout_week_end.

    Returns:
      • List[ModelResult] for each model (and optional blend)
      • DataFrame with per-model train/test probabilities (for analysis/blends)
    """
    # ----- Load data -----
    if data_path:
        X, y, groups, df = build_dataset(data_path)
        if holdout_season is None:
            # Production mode: use all rows for training, no test set
            X_train, y_train, groups_train, df_train = X, y, groups, df.copy()
            X_test, y_test, df_test = X.iloc[0:0], y.iloc[0:0], df.iloc[0:0].copy()
        else:
            # Week-aware or full-season holdout within holdout_season
            if holdout_week is None:
                test_mask = (df["season"] == int(holdout_season))
            else:
                if holdout_week_end is None:
                    test_mask = (df["season"] == int(holdout_season)) & (df["week"] >= int(holdout_week))
                else:
                    test_mask = (df["season"] == int(holdout_season)) & (
                        df["week"].between(int(holdout_week), int(holdout_week_end))
                    )
            train_mask = ~test_mask
            X_train, X_test = X.loc[train_mask], X.loc[test_mask]
            y_train, y_test = y.loc[train_mask], y.loc[test_mask]
            groups_train = groups.loc[train_mask]
            df_train = df.loc[train_mask].copy()
            df_test = df.loc[test_mask].copy()

    elif train_path and test_path:
        X_train, y_train, groups_train, df_train, X_test, y_test, _groups_test, df_test = load_train_test_splits(
            train_path, test_path
        )
        # In explicit split mode, ignore holdout_* controls (splits already provided).
    else:
        raise ValueError("Provide either `data_path` or both `train_path` and `test_path`.")

    # ----- Feature documentation -----
    feature_summary = summarize_features(X_train)
    export_feature_metadata(feature_summary, FEATURE_METADATA_PATH)

    # ----- Weights & baseline -----
    sample_weight_train = compute_recency_weights(df_train)
    baseline_rate_train = (
        float(np.average(y_train, weights=sample_weight_train)) if len(sample_weight_train) else float(y_train.mean())
    )

    # ----- Cross-validator -----
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, embargo_groups=1)

    # ----- Models -----
    models: List[Tuple[str, BaseEstimator, bool]] = []

    logit = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500, random_state=RNG_SEED)),
        ]
    )
    models.append(("Logistic", logit, True))  # calibrate logistic

    svc = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=RNG_SEED)),
        ]
    )
    models.append(("SVM", svc, True))  # calibrate SVM

    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, random_state=RNG_SEED)
    models.append(("GradientBoosting", gb, False))  # GB often well-calibrated already

    # Monotonic HGB if available; otherwise RF fallback
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
        monotonic_cst = [1] * X_train.shape[1]  # assume higher diffs ↑ P(home win)
        hist_gb = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.03,
            max_iter=300,
            monotonic_cst=monotonic_cst,
            random_state=RNG_SEED,
        )
        models.append(("MonotonicHGB", hist_gb, False))
    except Exception:
        rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=RNG_SEED)
        models.append(("RandomForest", rf, False))

    # ----- Train / CV / (optional) hold-out -----
    results: List[ModelResult] = []
    sample_weight_train = np.asarray(sample_weight_train, dtype=float)

    ensemble_train = pd.DataFrame(index=X_train.index)
    ensemble_test = pd.DataFrame(index=X_test.index)

    for name, est, calibrate in models:
        res = evaluate_model(
            name=name,
            estimator=est,
            X=X_train,
            y=y_train,
            groups=groups_train,
            cv=cv,
            calibrate=calibrate,
            baseline_rate=baseline_rate_train,
            sample_weight=sample_weight_train,
        )

        # Evaluate on hold-out if present
        if len(X_test) > 0:
            test_metrics = evaluate_on_test(
                estimator=est,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                calibrate=calibrate,
                baseline_rate=baseline_rate_train,
                sample_weight=sample_weight_train,
            )
        else:
            test_metrics = (None, None, None, None, None, None)

        (
            res.test_brier,
            res.test_logloss,
            res.test_roc_auc,
            res.test_pr_auc,
            res.test_brier_skill,
            res.test_brier_decomp,
        ) = test_metrics

        results.append(res)

        # Fit on full train for blend inputs
        if calibrate:
            base_full = clone(est)
            _fit_with_optional_weights(base_full, X_train, y_train, sample_weight_train)
            calibrator_full = CalibratedClassifierCV(
                **{CALIBRATOR_ESTIMATOR_PARAM: base_full}, method="isotonic", cv="prefit"
            )
            calibrator_full.fit(X_train, y_train)
            ensemble_train[name] = calibrator_full.predict_proba(X_train)[:, 1]
            ensemble_test[name] = (
                calibrator_full.predict_proba(X_test)[:, 1] if len(X_test) > 0 else pd.Series(dtype=float, index=X_test.index)
            )
        else:
            est_full = clone(est)
            _fit_with_optional_weights(est_full, X_train, y_train, sample_weight_train)
            ensemble_train[name] = est_full.predict_proba(X_train)[:, 1]
            ensemble_test[name] = (
                est_full.predict_proba(X_test)[:, 1] if len(X_test) > 0 else pd.Series(dtype=float, index=X_test.index)
            )

    # ----- Simple convex blend on hold-out (Logit + GB) -----
    if {"Logistic", "GradientBoosting"}.issubset(set(ensemble_train.columns)) and len(X_test) > 0:
        pA_train = ensemble_train["Logistic"].to_numpy(dtype=float)
        pB_train = ensemble_train["GradientBoosting"].to_numpy(dtype=float)
        best_w, _ = convex_blend(pA_train, pB_train, y_train.to_numpy(dtype=float))

        pA_test = ensemble_test["Logistic"].to_numpy(dtype=float)
        pB_test = ensemble_test["GradientBoosting"].to_numpy(dtype=float)
        blend_test_prob = np.clip(best_w * pA_test + (1 - best_w) * pB_test, PROBABILITY_EPS, 1 - PROBABILITY_EPS)

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

        brier_base = brier_score_loss(y_test_array, np.full_like(y_test_array, baseline_rate_train, dtype=float))
        brier_skill_blend = 1 - brier_blend / brier_base if brier_base > 0 else np.nan
        decomp_blend = brier_decomposition(y_test_array, blend_test_prob)

        results.append(
            ModelResult(
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
        )

    # Combined train/test probabilities (external analysis)
    ensemble_df = pd.concat([ensemble_train.add_suffix("_train"), ensemble_test.add_suffix("_test")], axis=1)

    # ===== Persist best model for production use =====
    # Map model names to (estimator, calibrate) used above
    name_to_cfg: Dict[str, Tuple[BaseEstimator, bool]] = {m[0]: (m[1], m[2]) for m in models}
    chosen_name = _select_best_model(results)
    est_base, calibrate_flag = name_to_cfg[chosen_name]

    # Fit on full training set and (optionally) calibrate
    if calibrate_flag:
        base_full = clone(est_base)
        _fit_with_optional_weights(base_full, X_train, y_train, sample_weight_train)
        calibrator_full = CalibratedClassifierCV(**{CALIBRATOR_ESTIMATOR_PARAM: base_full}, method="isotonic", cv="prefit")
        calibrator_full.fit(X_train, y_train)
        final_model = calibrator_full
    else:
        est_full = clone(est_base)
        _fit_with_optional_weights(est_full, X_train, y_train, sample_weight_train)
        final_model = est_full

    # Save artifacts (model, report, metadata)
    save_artifacts(
        model_name=chosen_name,
        model_obj=final_model,
        X_train=X_train,
        results=results,
        holdout_season=holdout_season,
    )

    return results, ensemble_df


# =======================
# Artifact saving helpers
# =======================
def _select_best_model(results: List[ModelResult]) -> str:
    """Pick best model name using mean_logloss, then mean_brier as tiebreaker."""
    df = pd.DataFrame([
        {
            "name": r.name,
            "mean_logloss": r.mean_logloss,
            "mean_brier": r.mean_brier,
        }
        for r in results
        if not np.isnan(r.mean_logloss)
    ])
    if df.empty:
        # Fallback: pick first
        return results[0].name
    df = df.sort_values(["mean_logloss", "mean_brier"], ascending=[True, True])
    return str(df.iloc[0]["name"])


def _build_training_report(results: List[ModelResult], chosen: str, holdout_season: Optional[int]) -> Dict[str, Any]:
    """Construct a JSON-serializable training report structure."""
    def _to_float(x):
        try:
            return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
        except Exception:
            return None

    rep = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "holdout_season": holdout_season,
        "chosen_model": chosen,
        "models": {}
    }
    for r in results:
        rep["models"][r.name] = {
            "cv": {
                "brier": _to_float(r.mean_brier),
                "logloss": _to_float(r.mean_logloss),
                "roc_auc": _to_float(r.mean_roc_auc),
                "pr_auc": _to_float(r.mean_pr_auc),
                "brier_ci": [ _to_float(r.brier_ci[0]), _to_float(r.brier_ci[1]) ],
                "brier_skill": _to_float(r.brier_skill),
            },
            "holdout": {
                "brier": _to_float(r.test_brier),
                "logloss": _to_float(r.test_logloss),
                "roc_auc": _to_float(r.test_roc_auc),
                "pr_auc": _to_float(r.test_pr_auc),
                "brier_skill": _to_float(r.test_brier_skill),
            }
        }
    return rep


def _update_models_metadata(feature_cols: List[str], win_model_relpath: str) -> None:
    """Update models/metadata.json with raw_feature_columns and win_model path.

    Preserves existing keys when possible.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MODELS_DIR / "metadata.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["raw_feature_columns"] = {"numeric": feature_cols, "categorical": []}
    meta["win_model"] = win_model_relpath
    # Keep existing preprocessor/home_model/away_model if present.
    meta.setdefault("preprocessor", meta.get("preprocessor"))
    meta.setdefault("home_model", meta.get("home_model"))
    meta.setdefault("away_model", meta.get("away_model"))
    meta.setdefault("mode", meta.get("mode", "production"))
    meta.setdefault("win_threshold_optimal", meta.get("win_threshold_optimal", 0.5))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_artifacts(
    model_name: str,
    model_obj: BaseEstimator,
    X_train: pd.DataFrame,
    results: List[ModelResult],
    holdout_season: Optional[int],
) -> None:
    """Persist trained classifier, feature metadata, training report, and metadata.json.

    - Saves win model to models/win_clf_calibrated.joblib
    - Saves feature metadata (already generated) to models/feature_metadata.json
    - Writes training report models/training_reportYYYYMMDD_HHMMSS.json
    - Updates models/metadata.json with raw_feature_columns and win_model path
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Save classifier
    win_path = MODELS_DIR / "win_clf_calibrated.joblib"
    joblib.dump(model_obj, win_path)

    # 2) Save feature metadata already written by export_feature_metadata
    # (No-op here; FEATURE_METADATA_PATH is written earlier.)

    # 3) Training report
    rep = _build_training_report(results, model_name, holdout_season)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = MODELS_DIR / f"training_report_{ts}.json"
    report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # 4) Update models/metadata.json
    _update_models_metadata(list(X_train.columns), win_model_relpath=win_path.name)


# =======================
# Markdown report writer
# =======================
def generate_markdown_report(results: List[ModelResult], output_path: str, holdout_season: int) -> None:
    """Write a human-readable report for cross-val and hold-out metrics."""
    lines: List[str] = []
    lines.append("# NFLEX v6 Predictive Pipeline Report\n")
    lines.append(
        f"This report summarises the performance of base models and a convex blend on NFL game data up to {holdout_season}.\n"
    )

    lines.append("## Cross-validated results (training seasons)\n")
    header = ["Model", "Brier", "Brier CI", "Log-loss", "LL CI", "ROC AUC", "PR AUC", "Brier Skill"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- " for _ in header]) + "|")
    for res in results:
        if np.isnan(res.mean_brier):
            continue
        lines.append(
            f"| {res.name} | {res.mean_brier:.4f} | [{res.brier_ci[0]:.4f}, {res.brier_ci[1]:.4f}] | "
            f"{res.mean_logloss:.4f} | [{res.logloss_ci[0]:.4f}, {res.logloss_ci[1]:.4f}] | "
            f"{res.mean_roc_auc:.4f} | {res.mean_pr_auc:.4f} | {res.brier_skill:.3f} |"
        )

    lines.append("\n## Hold-out season results (\"never_seen\" season)\n")
    header = ["Model", "Brier", "Log-loss", "ROC AUC", "PR AUC", "Brier Skill"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- " for _ in header]) + "|")
    for res in results:
        if res.test_brier is None:
            continue
        lines.append(
            f"| {res.name} | {res.test_brier:.4f} | {res.test_logloss:.4f} | {res.test_roc_auc:.4f} | "
            f"{res.test_pr_auc:.4f} | {res.test_brier_skill:.3f} |"
        )

    lines.append("\n## Brier decomposition (hold-out season)\n")
    lines.append("| Model | Brier | Reliability | Resolution | Uncertainty |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for res in results:
        if res.test_brier_decomp:
            d = res.test_brier_decomp
            lines.append(
                f"| {res.name} | {d['brier']:.4f} | {d['reliability']:.4f} | {d['resolution']:.4f} | {d['uncertainty']:.4f} |"
            )

    lines.append("\n**Notes**:")
    lines.append("- Purged walk-forward CV uses one-group embargo and five folds.")
    lines.append("- Hold-out season models are trained strictly on prior seasons.")
    lines.append("- Brier Skill Score baseline = weighted mean home-win rate on train.")
    lines.append("- Blend = convex log-loss-minimizing weight over Logistic and GB.")
    lines.append("- Monotonic constraints assume increasing diffs → higher home-win probability.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========
# CLI entry
# =========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NFLEX v6 pipeline")
    parser.add_argument("--data", required=True, help="Path to enhanced game-level CSV")
    parser.add_argument("--outdir", default="reports", help="Directory for output report")

    # Backward-compat (season-only)
    parser.add_argument("--holdout", type=int, default=None, help="Season to hold out (season-level)")

    # New: week-aware holdout
    parser.add_argument("--holdout-season", type=int, default=None, help="Season for week-level hold-out")
    parser.add_argument("--holdout-week", type=int, default=None, help="Start week (inclusive) within hold-out season")
    parser.add_argument("--holdout-week-end", type=int, default=None, help="End week (inclusive); if omitted, holds from start week to season end")

    parser.add_argument(
        "--production",
        action="store_true",
        help="Train on the entire dataset (no hold-out).",
    )
    args = parser.parse_args()

    if args.production:
        holdout_season = None
        holdout_week = None
        holdout_week_end = None
    else:
        # Precedence: explicit week-aware > season-only > auto-latest
        if args.holdout_season is not None:
            holdout_season = int(args.holdout_season)
            holdout_week = args.holdout_week
            holdout_week_end = args.holdout_week_end
        elif args.holdout is not None:
            holdout_season = int(args.holdout)
            holdout_week = None
            holdout_week_end = None
        else:
            preview_df = pd.read_csv(args.data)
            src = "season" if "season" in preview_df.columns else (
                "season_home" if "season_home" in preview_df.columns else None
            )
            if src is None:
                raise KeyError("Dataset must include a 'season' or 'season_home' column.")
            holdout_season = int(preview_df[src].max())
            holdout_week = None
            holdout_week_end = None

    results, _ = run_experiment(
        data_path=args.data,
        holdout_season=holdout_season,
        holdout_week=holdout_week,
        holdout_week_end=holdout_week_end,
    )

    report_path = Path(args.outdir) / "nflex_v6_report.md"
    if holdout_season is not None:
        generate_markdown_report(results, str(report_path), int(holdout_season))
    else:
        print("Training completed in PRODUCTION mode (no hold-out season). Report not generated.")
