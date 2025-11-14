#!/usr/bin/env python3
"""
Enhanced NFL Score + Win-Probability Training Pipeline
======================================================

Purpose
-------
A single, reliable training entrypoint that:
  • Trains HOME and AWAY score regressors, plus a calibrated WIN classifier.
  • Uses a purged, group-aware time-series CV (season*100+week) for evaluation.
  • Produces verbose logs, plots, and human-readable reports.
  • Saves/overwrites **all artifacts** into ./data/ on every run:
      - data/home_model.joblib
      - data/away_model.joblib
      - data/preprocessor.joblib
      - data/win_CLF_calibrated.joblib
      - data/metadata.json
      - data/feature_metadata.json
      - data/training_report.png
      - data/training_report.txt

CLI
---
python pipeline_enhanced.py --data path/to/enhanced_dataset.csv \
    [--production] \
    [--holdout-season 2025 --holdout-week 6 --holdout-week-end 9] \
    [--splits 5 --embargo 1]

Notes
-----
• In --production mode, trains on the full dataset (no hold-out).
• Otherwise, a week-aware holdout is created within the specified season. If only
  a season is provided, the entire season is used as hold-out.
• Random states are fixed for reproducibility.
"""

# ============================================================
# — Imports & Global Settings
# ============================================================
from __future__ import annotations

import argparse
import logging
import math
from ssl import HAS_NPN
import warnings
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, 
    log_loss, accuracy_score, brier_score_loss
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import matplotlib as plt

from typing import Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

# ============================================================
# — Utility: Time Block Context Manager
# ============================================================
class Block:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logging.info(f"\n=== {self.name} ===")
        self.start = time.time()

    def __exit__(self, exc_type, exc, tb):
        took = time.time() - self.start
        logging.info(f"{self.name} done in {took:.2f}s")


# ----------------------
# Globals / Paths
# ----------------------
RANDOM_STATE = 4211
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "models"
DATASET_PATH = HERE / "data" / "game_features_20251108.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Artifact paths (required)
P_HOME = DATA_DIR / "home_model.joblib"
P_AWAY = DATA_DIR / "away_model.joblib"
P_PREP = DATA_DIR / "preprocessor.joblib"
P_WIN  = DATA_DIR / "win_clf_calibrated.joblib"
P_META = DATA_DIR / "metadata.json"
P_FEAT = DATA_DIR / "feature_metadata.json"
P_PNG  = DATA_DIR / "training_report.png"
P_TXT  = DATA_DIR / "training_report.txt"
# New structured outputs
P_FOLDS_CSV = DATA_DIR / "cv_fold_metrics.csv"
P_SUMMARY_JSON = DATA_DIR / "training_summary.json"

# ----------------------
# Logging helpers
# ----------------------
class Block:
    def __init__(self, title: str):
        self.title = title
        self.t0 = 0.0
    def __enter__(self):
        logging.info("\n=== %s ===", self.title)
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        logging.info("%s done in %.2fs", self.title, dt)


def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ----------------------
# CV splitter (purged, group-aware)
# ----------------------
class PurgedGroupTimeSeriesSplit:
    """Walk-forward over integer group IDs (e.g., season*100+week) with embargo."""
    def __init__(self, n_splits: int = 5, embargo_groups: int = 1):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_groups = embargo_groups

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be provided")
        uniq = np.unique(groups)
        fold_sizes = np.full(self.n_splits, len(uniq) // self.n_splits, dtype=int)
        fold_sizes[: len(uniq) % self.n_splits] += 1
        blocks = []
        start = 0
        for sz in fold_sizes:
            blocks.append(uniq[start : start + sz])
            start += sz
        for i in range(self.n_splits - 1):
            train_groups = np.concatenate(blocks[: i + 1])
            val_groups = blocks[i + 1]
            max_val = val_groups.max()
            train_groups = train_groups[train_groups <= (max_val - self.embargo_groups)]
            tr_idx = np.where(np.isin(groups, train_groups))[0]
            va_idx = np.where(np.isin(groups, val_groups))[0]
            yield tr_idx, va_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits - 1

# ----------------------
# Data loading / features
# ----------------------
@dataclass
class DataBundle:
    X: pd.DataFrame
    y_home: pd.Series
    y_away: pd.Series
    y_win: pd.Series
    groups: pd.Series
    df_raw: pd.DataFrame


def ensure_season_week(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1 — Remove footer rows where no scoring is present
    score_cols = ["home_points_for", "away_points_for"]
    if all(c in df.columns for c in score_cols):
        bad_footer = df[score_cols].isna().all(axis=1)
        if bad_footer.any():
            logging.warning(f"Dropping {bad_footer.sum()} rows with NaN scores.")
            df = df.loc[~bad_footer].copy()

    # 2 — Season/Week creation
    if "season" not in df.columns:
        df["season"] = df["home_season"]

    if "week" not in df.columns:
        df["week"] = df["home_week"]

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")

    df = df.dropna(subset=["season", "week"])
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    return df

def leak_harden_features(df: pd.DataFrame, feature_cols: list, y_win: pd.Series):
    forbidden_exact = {
        "_home_win_derived",
        "_dom_delta_emp_home_win",
        "season_home_win_rate",
        "tl_home_home_win_rate_when_home",
        "tl_away_home_win_rate_when_home",
        "tl_away_away_win_rate_when_away",
        "tl_home_away_win_rate_when_away",
        "home_elo_post",
        "away_elo_post",
    }

    forbidden_tokens = [
        "point_diff", "winner", "outcome", "final", "score"
    ]

    forbidden_suffixes = [
        "_post"
    ]

    safe, dropped = [], []

    for c in feature_cols:
        lc = c.lower()

        if c in forbidden_exact:
            dropped.append(c)
            continue

        if any(tok in lc for tok in forbidden_tokens):
            dropped.append(c)
            continue

        if any(lc.endswith(sfx) for sfx in forbidden_suffixes):
            dropped.append(c)
            continue

        safe.append(c)

    return safe, dropped



def load_dataset(path: str):
    df = pd.read_csv(filepath_or_buffer=path)
    df = ensure_season_week(df=df)

    # 1 — Create group index for time awareness
    df["group_idx"] = df["season"] * 100 + df["week"]

    # 2 — Targets
    y_home = df["home_points_for"]
    y_away = df["away_points_for"]

    if "home_win" in df.columns:
        y_win = df["home_win"].astype(int)
    elif "winner" in df.columns:
        y_win = (df["winner"] == df["home_team"]).astype(int)
    else:
        y_win = (df["home_points_for"] > df["away_points_for"]).astype(int)

    # 3 — Feature selection
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    remove_cols = ["season", "week", "group_idx", 
                   "home_points_for", "away_points_for", "home_win"]

    feature_cols = [c for c in numeric_cols if c not in remove_cols]

    # 4 — Leak-harden
    safe_cols, dropped = leak_harden_features(df=df, feature_cols=feature_cols, y_win=y_win)

    if dropped:
        logging.warning(f"Dropping leaky columns: {dropped}")

    X = df[safe_cols].copy()

    # 5 — Missing-value hygiene (no double-imputation)
    X = X.fillna(X.mean(numeric_only=True))
    return X, y_away, y_home, df


def build_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

X, y_away, y_home, df = load_dataset(path=DATASET_PATH)





# ----------------------
# Models / preprocessing
# ----------------------
numeric_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transform, X)], remainder="drop")


def make_models():
    params = dict(
        random_state=4211,
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=1.0,
    )

    return {
        "home": GradientBoostingRegressor(**params),
        "away": GradientBoostingRegressor(**params),
    }

        # NOTE: For CV we will train LogisticRegression on preprocessed arrays directly (no extra scaler).

# ----------------------
# CV / evaluation
# ----------------------
@dataclass
class FoldScores:
    train: List[float]
    valid: List[float]

@dataclass
class EvalBundle:
    home_mae: FoldScores
    home_rmse: FoldScores 
    away_mae: FoldScores
    away_rmse: FoldScores
    win_brier: FoldScores
    win_logloss: FoldScores
    win_accuracy: FoldScores


def _rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def cross_validate_models(
    X: pd.DataFrame,
    y_home: pd.Series,
    y_away: pd.Series,
    y_win: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    embargo: int = 1,
) -> Tuple[EvalBundle, Dict[str, Any]]:
    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, embargo_groups=embargo)

    # Prepare collectors
    home_mae_tr, home_mae_va, home_rmse_tr, home_rmse_va = [], [], [], []
    away_mae_tr, away_mae_va, away_rmse_tr, away_rmse_va = [], [], [], []
    win_brier_tr, win_brier_va, win_ll_tr, win_ll_va, win_acc_tr, win_acc_va = [], [], [], [], [], []

    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y_win, groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        yh_tr, yj_tr = y_home.iloc[tr_idx], y_away.iloc[tr_idx]
        yw_tr, yw_va = y_win.iloc[tr_idx], y_win.iloc[va_idx]

        # Fit preprocessor on TRAIN only
        prep = build_preprocessor(list(X.columns))
        Xtr_tf = prep.fit_transform(X_tr)
        Xva_tf = prep.transform(X_va)

        models = make_models()

        # --- HOME regressor ---
        home = clone(models["home_reg"]).fit(Xtr_tf, yh_tr)
        yh_tr_pred = home.predict(Xtr_tf)
        yh_va_pred = home.predict(Xva_tf)
        home_mae_tr.append(mean_absolute_error(yh_tr, yh_tr_pred))
        home_mae_va.append(mean_absolute_error(y_home.iloc[va_idx], yh_va_pred))
        home_rmse_tr.append(_rmse(yh_tr, yh_tr_pred))
        home_rmse_va.append(_rmse(y_home.iloc[va_idx], yh_va_pred))

        # --- AWAY regressor ---
        away = clone(models["away_reg"]).fit(Xtr_tf, yj_tr)
        yj_tr_pred = away.predict(Xtr_tf)
        yj_va_pred = away.predict(Xva_tf)
        away_mae_tr.append(mean_absolute_error(yj_tr, yj_tr_pred))
        away_mae_va.append(mean_absolute_error(y_away.iloc[va_idx], yj_va_pred))
        away_rmse_tr.append(_rmse(yj_tr, yj_tr_pred))
        away_rmse_va.append(_rmse(y_away.iloc[va_idx], yj_va_pred))

        # --- WIN classifier (consistent with final training: logistic on preprocessed X) ---
        clf = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=600,
            class_weight="balanced", random_state=RANDOM_STATE
        ).fit(Xtr_tf, yw_tr)

        p_tr = np.clip(clf.predict_proba(Xtr_tf)[:, 1], 1e-6, 1 - 1e-6)
        p_va = np.clip(clf.predict_proba(Xva_tf)[:, 1], 1e-6, 1 - 1e-6)

        brier_tr = brier_score_loss(yw_tr, p_tr)
        brier_va = brier_score_loss(yw_va, p_va)
        ll_tr = log_loss(yw_tr, p_tr, labels=[0, 1])
        ll_va = log_loss(yw_va, p_va, labels=[0, 1])
        acc_tr = accuracy_score(yw_tr, (p_tr >= 0.5).astype(int))
        acc_va = accuracy_score(yw_va, (p_va >= 0.5).astype(int))

        win_brier_tr.append(brier_tr)
        win_brier_va.append(brier_va)
        win_ll_tr.append(ll_tr)
        win_ll_va.append(ll_va)
        win_acc_tr.append(acc_tr)
        win_acc_va.append(acc_va)

        # Persist per-fold row (for CSV)
        fold_rows.append({
            "fold": fold,
            "home_mae_train": float(home_mae_tr[-1]),
            "home_mae_val":   float(home_mae_va[-1]),
            "home_rmse_train": float(home_rmse_tr[-1]),
            "home_rmse_val":   float(home_rmse_va[-1]),
            "away_mae_train": float(away_mae_tr[-1]),
            "away_mae_val":   float(away_mae_va[-1]),
            "away_rmse_train": float(away_rmse_tr[-1]),
            "away_rmse_val":   float(away_rmse_va[-1]),
            "win_brier_train": float(brier_tr),
            "win_brier_val":   float(brier_va),
            "win_logloss_train": float(ll_tr),
            "win_logloss_val":   float(ll_va),
            "win_acc_train":   float(acc_tr),
            "win_acc_val":     float(acc_va),
            "class_balance_train": float(yw_tr.mean()),
            "class_balance_val":   float(yw_va.mean()),
        })

        # Class balance + leak sentinel
        logging.info(
            "Fold %d class balance — train p(home=1)=%.3f | val=%.3f",
            fold, float(yw_tr.mean()), float(yw_va.mean())
        )
        if acc_va > 0.995 or brier_va < 1e-4:
            logging.warning(
                "Leak sentinel tripped on fold %d: Acc=%.3f Brier=%.6f — check features!",
                fold, acc_va, brier_va
            )

        logging.info(
            "Fold %d | HOME MAE %.3f/%.3f | AWAY MAE %.3f/%.3f | WIN Brier %.3f/%.3f | Acc %.3f/%.3f",
            fold,
            home_mae_tr[-1], home_mae_va[-1],
            away_mae_tr[-1], away_mae_va[-1],
            brier_tr, brier_va,
            acc_tr, acc_va,
        )

    eval_bundle = EvalBundle(
        home_mae=FoldScores(train=home_mae_tr, valid=home_mae_va),
        home_rmse=FoldScores(train=home_rmse_tr, valid=home_rmse_va),
        away_mae=FoldScores(train=away_mae_tr, valid=away_mae_va),
        away_rmse=FoldScores(train=away_rmse_tr, valid=away_rmse_va),
        win_brier=FoldScores(train=win_brier_tr, valid=win_brier_va),
        win_logloss=FoldScores(train=win_ll_tr, valid=win_ll_va),
        win_accuracy=FoldScores(train=win_acc_tr, valid=win_acc_va),
    )

    # Summary metrics for metadata/report (FIX: home RMSE now correct)
    summary = {
        "home": {
            "MAE_mean_val": float(np.mean(home_mae_va)),
            "RMSE_mean_val": float(np.mean(eval_bundle.home_rmse.valid)),
        },
        "away": {
            "MAE_mean_val": float(np.mean(away_mae_va)),
            "RMSE_mean_val": float(np.mean(eval_bundle.away_rmse.valid)),
        },
        "win": {
            "Brier_mean_val": float(np.mean(win_brier_va)),
            "LogLoss_mean_val": float(np.mean(win_ll_va)),
            "Acc_mean_val": float(np.mean(win_acc_va)),
        },
    }
    return eval_bundle, summary, fold_rows

# ----------------------
# Train / final fit
# ----------------------
@dataclass
class TrainedModels:
    preprocessor: ColumnTransformer
    home_model: BaseEstimator
    away_model: BaseEstimator
    win_model: CalibratedClassifierCV



def train_models(X_train: pd.DataFrame,
        y_home: pd.Series, y_away: pd.Series, y_win: pd.Series) -> TrainedModels:
    
    # Shared preprocessor fit on all training rows
    preprocessor = build_preprocessor(feature_names=list(X_train.columns))
    Xtf = preprocessor.fit_transform(X=X_train)

    models = make_models()

    # Regressors (fit on transformed)
    home = clone(estimator=models["home_reg"]).fit(Xtf, y_home)
    away = clone(estimator=models["away_reg"]).fit(Xtf, y_away)

    # Classifier: logistic on preprocessed X, then calibrate
    # Classifier: logistic on preprocessed X, then attempt isotonic calibration.
    # Use class_weight='balanced' to mitigate any imbalance during final training.
    win_base = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=600,
        class_weight="balanced", random_state=RANDOM_STATE  
    )
    win_base.fit(Xtf, y_win)

    # Calibrate probabilities: prefer isotonic, fall back to sigmoid if isotonic
    # fails (numerical / monotonicity issues). If both fail, return an
    # uncalibrated wrapper around the fitted logistic (last resort).
    try:
        win_cal = CalibratedClassifierCV(win_base, method="isotonic", cv=3)
        win_cal.fit(Xtf, y_win)
    except Exception as exc_isotonic:  # pragma: no cover - runtime fallback
        logging.warning("Isotonic calibration failed: %s; falling back to sigmoid", exc_isotonic)
        try:
            win_cal = CalibratedClassifierCV(win_base, method="sigmoid", cv=3)
            win_cal.fit(Xtf, y_win)
        except Exception as exc_sigmoid:  # pragma: no cover - runtime fallback
            logging.error(
                "Sigmoid calibration also failed: %s; using uncalibrated logistic as fallback",
                exc_sigmoid,
            )
            # Last-resort: use the uncalibrated logistic model (still scikit-learn compatible)
            win_cal = win_base

    return TrainedModels(preprocessor=preprocessor, home_model=home, away_model=away, win_model=win_cal)

# ----------------------
# Plotting
# ----------------------
def plot_training_curves(evalb: EvalBundle, save_path: Path) -> None:
    if not HAS_NPN:
        logging.warning("matplotlib not available; skipping plot generation")
        return
    folds = range(1, len(evalb.home_mae.valid) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Panel 1: HOME/AWAY MAE (validation)
    ax = axes[0]
    ax.plot(folds, evalb.home_mae.valid, marker="o", label="HOME MAE (val)")
    ax.plot(folds, evalb.away_mae.valid, marker="o", label="AWAY MAE (val)")
    ax.set_title("Validation Errors per Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: HOME/AWAY RMSE (validation)
    ax = axes[1]
    ax.plot(folds, evalb.home_rmse.valid, marker="o", label="HOME RMSE (val)")
    ax.plot(folds, evalb.away_rmse.valid, marker="o", label="AWAY RMSE (val)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: WIN metrics (validation)
    ax = axes[2]
    ax.plot(folds, evalb.win_brier.valid, marker="o", label="Brier (val)")
    ax.plot(folds, evalb.win_logloss.valid, marker="o", label="LogLoss (val)")
    ax.plot(folds, evalb.win_accuracy.valid, marker="o", label="Accuracy (val)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training/Validation Performance (CV)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ----------------------
# Persistence / Reports
# ----------------------
def _save_joblib(obj: Any, path: Path) -> None:
    joblib.dump(obj,path)
    if path.exists() and path.stat().st_size > 0:
        logging.info("✅ Saved: %s", path)
    else:
        logging.warning("⚠️  Save verification failed for: %s", path)


def _save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    logging.info("✅ Saved: %s", path)


def generate_feature_metadata(X: pd.DataFrame) -> List[Dict[str, Any]]:
    meta = []
    for col in X.columns:
        ser = X[col]
        is_num = ser.dtype.kind in "if"
        meta.append({
            "feature": col,
            "dtype": str(ser.dtype),
            "missing_pct": float(pd.isna(ser).mean()),
            "mean": float(np.nanmean(ser)) if is_num else None,
            "std": float(np.nanstd(ser)) if is_num else None,
        })
    return meta


def write_training_report_txt(
    save_path: Path,
    *,
    dataset_rows: int,
    dataset_cols: int,
    features: List[str],
    metrics_summary: Dict[str, Any],
    artifacts: Dict[str, str]
) -> None:
    lines = []
    lines.append("# Training Report\n")
    lines.append(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")
    lines.append(f"Dataset: {dataset_rows} rows, {dataset_cols} cols\n")
    lines.append(f"Features used ({len(features)}): {', '.join(features)}\n\n")

    lines.append("## Metrics (validation means)\n")
    lines.append(json.dumps(metrics_summary, indent=2))
    lines.append("\n\n## Artifacts\n")
    for k, v in artifacts.items():
        lines.append(f"- {k}: {v}")

    save_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("✅ Saved: %s", save_path)


def save_artifacts(models: TrainedModels, X_train: pd.DataFrame, eval_summary: Dict[str, Any]) -> None:
    # Save models & preprocessor
    _save_joblib(models.preprocessor, P_PREP)
    _save_joblib(models.home_model, P_HOME)
    _save_joblib(models.away_model, P_AWAY)
    _save_joblib(models.win_model,  P_WIN)

    # Feature metadata
    feat_meta = generate_feature_metadata(X_train)
    _save_json(feat_meta, P_FEAT)

    # Metadata.json (training params + scores)
    meta = {
        "timestamp": datetime.now().isoformat() + "Z",
        "random_state": RANDOM_STATE,
        "models": {
            "home_reg": {
                "type": type(models.home_model).__name__,
                "params": getattr(models.home_model, "get_params", lambda: {})(),
            },
            "away_reg": {
                "type": type(models.away_model).__name__,
                "params": getattr(models.away_model, "get_params", lambda: {})(),
            },
            "win_clf_calibrated": {
                "type": type(models.win_model).__name__,
                "base_estimator": type(models.win_model.calibrated_classifiers_).__name__ if hasattr(models.win_model, "calibrated_classifiers_") else None,
                "method": getattr(models.win_model, "method", "isotonic"),
            },
        },
        "validation_summary": eval_summary,
        "artifacts": {
            "home_model": str(P_HOME),
            "away_model": str(P_AWAY),
            "preprocessor": str(P_PREP),
            "win_clf_calibrated": str(P_WIN),
            "cv_fold_metrics_csv": str(P_FOLDS_CSV),
            "training_summary_json": str(P_SUMMARY_JSON),
            "feature_metadata": str(P_FEAT),
            "training_report_png": str(P_PNG),
            "training_report_txt": str(P_TXT),
        },
    }
    _save_json(meta, P_META)

# ----------------------
# Orchestration
# ----------------------
def evaluate_models(
    X: pd.DataFrame,
    y_home: pd.Series,
    y_away: pd.Series,
    y_win: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    embargo: int = 1
) -> Tuple[EvalBundle, Dict[str, Any]]:
    with Block("Evaluating Models"):
        eval_bundle, summary, fold_rows = cross_validate_models(
            X, y_home, y_away, y_win, groups, n_splits=n_splits, embargo=embargo
        )

    # Persist per-fold metrics and a summary JSON for downstream inspection
    try:
        if fold_rows:
            df_folds = pd.DataFrame(fold_rows)
            df_folds.to_csv(P_FOLDS_CSV, index=False)
            logging.info("Saved CV fold metrics to %s", P_FOLDS_CSV)
    except Exception as ex:  # pragma: no cover - I/O safety
        logging.exception("Failed to write CV fold metrics CSV: %s", ex)

    try:
        _save_json(summary, P_SUMMARY_JSON)
        logging.info("Saved training summary JSON to %s", P_SUMMARY_JSON)
    except Exception as ex:
        logging.exception("Failed to write training summary JSON: %s", ex)

    return eval_bundle, summary


def main():
    parser = argparse.ArgumentParser(description="Train NFL score + win models and save artifacts to ./data/")
    parser.add_argument("--data", required=True, help="Path to enhanced game-level CSV")
    parser.add_argument("--production", action="store_true", help="Train on all rows (no hold-out)")
    parser.add_argument("--holdout-season", type=int, default=None, help="Hold-out season (for week-aware holdout)")
    parser.add_argument("--holdout-week", type=int, default=None, help="Start week (inclusive) within the hold-out season")
    parser.add_argument("--holdout-week-end", type=int, default=None, help="End week (inclusive); default = season end")
    parser.add_argument("--splits", type=int, default=5, help="CV splits (default 5)")
    parser.add_argument("--embargo", type=int, default=1, help="Embargo groups between train/val (default 1)")
    args = parser.parse_args()

    setup_logging(verbose=True)

    # --------------------
    # 1) Load & preprocess
    # --------------------
    with Block("Loading and Preprocessing Data"):
        bundle = load_dataset(path='game_features_20251110.csv')
        df = bundle
        if args.production:
            train_mask = np.ones(len(df), dtype=bool)
            test_mask = np.zeros(len(df), dtype=bool)
        else:
            if args.holdout_season is None:
                # default: last season entirely as hold-out
                hold_season = int(df["season"].max())
                hold_start, hold_end = None, None
            else:
                hold_season = int(args.holdout_season)
                hold_start = args.holdout_week
                hold_end = args.holdout_week_end

            if hold_start is None:
                test_mask = (df["season"] == hold_season)
            else:
                if hold_end is None:
                    test_mask = (df["season"] == hold_season) & (df["week"] >= int(hold_start))
                else:
                    test_mask = (df["season"] == hold_season) & (df["week"].between(int(hold_start), int(hold_end)))
            train_mask = ~test_mask

        X_train = bundle.X.loc[train_mask]
        yh_tr = bundle.y_home.loc[train_mask]
        yj_tr = bundle.y_away.loc[train_mask]
        yw_tr = bundle.y_win.loc[train_mask]
        groups_tr = bundle.groups.loc[train_mask]

        # Filter unlabeled rows (NaNs in any target) to avoid NaN->int cast errors later
        labeled_mask = (~yh_tr.isna()) & (~yj_tr.isna()) & (~yw_tr.isna())
        if not bool(labeled_mask.all()):
            X_train = X_train.loc[labeled_mask]
            yh_tr = yh_tr.loc[labeled_mask]
            yj_tr = yj_tr.loc[labeled_mask]
            yw_tr = yw_tr.loc[labeled_mask]
            groups_tr = groups_tr.loc[labeled_mask]
        # Ensure classifier labels are integers (0/1)
        yw_tr = yw_tr.astype(int)

        X_test = bundle.X.loc[test_mask]
        yw_te = bundle.y_win.loc[test_mask]  # currently unused; preserved for future eval

        logging.info("Data shape (train): %s | (test): %s", X_train.shape, X_test.shape)
        logging.info("Features used: %d", X_train.shape[1])

    # --------------------
    # 2) Cross-validated eval
    # --------------------
    eval_bundle, eval_summary = evaluate_models(
        X_train, yh_tr, yj_tr, yw_tr, groups_tr, n_splits=args.splits, embargo=args.embargo
    )

    # --------------------
    # 3) Final training
    # --------------------
    with Block("Training Models"):
        models = train_models(X_train, yh_tr, yj_tr, yw_tr)
        logging.info("Model params | HOME: %s", getattr(models.home_model, "get_params", lambda: {})())
        logging.info("Model params | AWAY: %s", getattr(models.away_model, "get_params", lambda: {})())
        logging.info("Calibrated WIN model: %s", type(models.win_model).__name__)

    # --------------------
    # 4) Plot curves
    # --------------------
    with Block("Plotting Training Curves"):
        plot_training_curves(eval_bundle, P_PNG)
        logging.info("Figure saved: %s", P_PNG)

    # --------------------
    # 5) Save artifacts
    # --------------------
    with Block("Saving Artifacts"):
        save_artifacts(models, X_train, eval_summary)

    # --------------------
    # 6) Write text report
    # --------------------
    with Block("Generating Reports"):
        artifacts = {
            "home_model": str(P_HOME),
            "away_model": str(P_AWAY),
            "preprocessor": str(P_PREP),
            "win_CLF_calibrated": str(P_WIN),
            "feature_metadata": str(P_FEAT),
            "training_report_png": str(P_PNG),
            "training_report_txt": str(P_TXT),
        }
        write_training_report_txt(
            P_TXT,
            dataset_rows=int(X_train.shape[0]),
            dataset_cols=int(X_train.shape[1]),
            features=list(X_train.columns),
            metrics_summary=eval_summary,
            artifacts=artifacts,
        )

    print(f"\n\n🏁 Training complete — all artifacts saved to {DATA_DIR.resolve()}/\n")


if __name__ == "__main__":
    main()