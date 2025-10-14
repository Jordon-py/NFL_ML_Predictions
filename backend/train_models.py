#!/usr/bin/env python
"""
train_models_converted.py — Flexible Training Pipeline for the user's merged_nfl_data.csv

Purpose
-------
Adapt the original team-vs-team training pipeline to work with a generic team‑game dataset like
`/mnt/data/merged_nfl_data.csv` that may not include home/away score columns.

Supported label configurations (auto-detected in this order):
1) Game-level home/away labels:
   - Required columns: ['season','week','home_team','away_team','home_points_for','away_points_for']
   - Trains two regressors for home/away scores and a classifier for home win.

2) Team-level points labels:
   - Required columns: ['season','week','team','opponent_team','points_for','points_against']  (names are flexible; see ALIASES below)
   - Trains a regressor for team points_for and a classifier for team win = points_for > points_against.

3) Precomputed team-level win label:
   - Required columns: ['season','week','team','opponent_team','win'] where win ∈ {0,1}
   - Trains only the classifier for win.

If none are present, the script stops early with a precise error message explaining how to add labels.

Feature handling
----------------
- Numeric features: all numeric columns except ID/time columns and label columns.
- Categorical features: ['team','opponent_team','season_type'] if present.
- Boolean/binary: 'is_home' (treated numeric if present).

Outputs
-------
- models/preprocessor.joblib
- models/home_model.joblib, models/away_model.joblib (when available)
- models/team_points_model.joblib (team-level regression when available)
- models/win_clf_calibrated.joblib
- models/test_predictions.csv
- models/training_report.json
- models/metadata.json

Run
---
python train_models_converted.py --data /mnt/data/merged_nfl_data.csv

Notes
-----
- This script is intentionally conservative. It refuses to invent labels.
- If you need labels, merge an official schedule/scores file to produce either:
  (home_points_for, away_points_for) or (points_for, points_against) or a boolean 'win'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------- Config -----------------

RANDOM_SEED = 42
HYPERPARAM_SEARCH_ITERATIONS = 25

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOGGING_CONFIG)
log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Aliases for flexible column names
ALIASES = {
    "team": ["team", "home_team", "club"],
    "opponent_team": ["opponent_team", "away_team", "opponent", "opp_team"],
    "points_for": ["points_for", "pf", "team_points", "score", "pts_for"],
    "points_against": [
        "points_against",
        "pa",
        "opp_points",
        "opp_score",
        "pts_against",
    ],
    "win": ["win", "is_win", "team_win", "home_win"],
    "season_type": ["season_type", "type"],
}

# ----------------- Utilities -----------------


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_paths(user_path: Optional[str]) -> Path:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Dataset not found at --data {user_path}")
    # Try common defaults
    for p in [
        Path("/mnt/data/merged_nfl_data.csv"),
        BACKEND_DIR / "data" / "merged_nfl_data.csv",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate merged_nfl_data.csv. Pass --data <path>.")


def _split_latest_season(
    df: pd.DataFrame, train_weeks: int = 3, test_weeks: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if "season" not in df.columns or "week" not in df.columns:
        raise ValueError("Dataset must include 'season' and 'week'.")
    df = df.dropna(subset=["season", "week"]).copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    latest = int(df["season"].max())
    weeks = sorted(df.loc[df["season"] == latest, "week"].unique().astype(int))
    if len(weeks) < train_weeks + test_weeks:
        raise ValueError(
            f"Season {latest} has only {len(weeks)} weeks. Need >= {train_weeks + test_weeks}."
        )

    train_w = weeks[:train_weeks]
    test_w = weeks[train_weeks : train_weeks + test_weeks]

    train_df = df[
        (df["season"] < latest)
        | ((df["season"] == latest) & (df["week"].isin(train_w)))
    ].copy()
    test_df = df[(df["season"] == latest) & (df["week"].isin(test_w))].copy()

    split_info = {"latest_season": latest, "train_weeks": train_w, "test_weeks": test_w}
    return train_df, test_df, split_info


def _recency_weights(df: pd.DataFrame) -> np.ndarray:
    seasons = df["season"].to_numpy(dtype=float)
    weeks = df["week"].to_numpy(dtype=float)
    sspan = max(seasons.max() - seasons.min(), 1.0)
    season_norm = (seasons - seasons.min()) / sspan
    week_norm = weeks / max(weeks.max(), 1.0)
    w = 0.4 + 0.4 * season_norm + 0.2 * week_norm
    return w / w.mean()


def _build_feature_lists(
    df: pd.DataFrame, label_cols: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    non_features = set(["season", "week", "game_id", "idx"] + label_cols)
    num_cols = [
        c for c in df.select_dtypes(include=["number"]).columns if c not in non_features
    ]
    cat_cols = [c for c in ["team", "opponent_team", "season_type"] if c in df.columns]
    bin_cols = [c for c in ["is_home"] if c in df.columns]
    # keep order stable
    return num_cols + bin_cols, cat_cols, list(non_features)


def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _reg_grid() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [15, 25, 31],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.1, 0.2, 0.5],
        "reg_lambda": [0.1, 0.2, 0.5],
        "min_child_samples": [20, 30],
    }


def _clf_grid() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [15, 25, 31],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.1, 0.2, 0.5],
        "reg_lambda": [0.1, 0.2, 0.5],
        "min_child_samples": [20, 30],
        "class_weight": [None, "balanced"],
    }


def _fit_regressor(
    X_train: Union[np.ndarray, spmatrix],
    y_train: np.ndarray,
    X_test: Union[np.ndarray, spmatrix],
    y_test: np.ndarray,
    name: str,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[LGBMRegressor, Dict[str, Any]]:
    """Fits and evaluates a LightGBM regressor using RandomizedSearchCV."""
    log.info("--- Fitting Regressor: %s ---", name)
    estimator = LGBMRegressor(random_state=42)
    # RandomizedSearchCV is used to find the best hyperparameters in a timely manner
    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=REG_PARAMS,
        n_iter=50,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    rs.fit(
        X_train,
        y_train,
        **({"sample_weight": sample_weight} if sample_weight is not None else {}),
    )
    best_estimator = rs.best_estimator_
    if best_estimator is None:
        raise RuntimeError("Hyperparameter search returned no regressor.")
    best = cast(LGBMRegressor, best_estimator)
    # Change log 2025-02-14: Cast aligns LightGBM regressor with static typing and preserves predict().
    # Change log 2025-02-14: Removed unused training predictions to reduce clutter and keep evaluation focused on test metrics.
    yhat_te = np.asarray(best.predict(X_test), dtype=float).ravel()

    # Evaluate on the test set
    r2 = r2_score(y_test, yhat_te)
    mae = mean_absolute_error(y_test, yhat_te)

    log.info(f"  {name} Test R2: {r2:.4f}, MAE: {mae:.4f}")

    metrics = {
        f"{name}_test_r2": r2,
        f"{name}_test_mae": mae,
        f"{name}_best_params": rs.best_params_,
    }
    return best_estimator, metrics


def _fit_classifier(
    X_train: Union[np.ndarray, spmatrix],
    y_train: np.ndarray,
    X_test: Union[np.ndarray, spmatrix],
    y_test: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[CalibratedClassifierCV, Dict[str, Any], pd.DataFrame]:
    """Fits and evaluates a LightGBM classifier with calibration."""
    log.info("--- Fitting Classifier: home_win ---")
    estimator = LGBMClassifier(random_state=42)
    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=CLF_PARAMS,
        n_iter=50,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    rs.fit(
        X_train,
        y_train,
        **({"sample_weight": sample_weight} if sample_weight is not None else {}),
    )
    best_uncal_est = rs.best_estimator_
    if best_uncal_est is None:
        raise RuntimeError("Hyperparameter search returned no classifier.")
    best_uncal = cast(LGBMClassifier, best_uncal_est)
    calib = CalibratedClassifierCV(
        estimator=cast(BaseEstimator, best_uncal),
        cv=TimeSeriesSplit(n_splits=4),
        method="isotonic",
    )
    # Change log 2025-02-14: Cast clarifies estimator type for calibration and appeases type checkers.
    calib.fit(X_train, y_train, sample_weight=sample_weight)

    # Evaluate on the test set
    y_prob_test = calib.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test > 0.5).astype(int)

    metrics = {
        "test_auc": roc_auc_score(y_test, y_prob_test),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test),
        "test_brier": brier_score_loss(y_test, y_prob_test),
    }
    preds = pd.DataFrame({"prob_win": y_prob_test, "pred_win": y_pred_test})
    log.info(
        "win classifier → test AUC=%.3f, Brier=%.3f",
        metrics["test_auc"],
        metrics["test_brier"],
    )
    return calib, metrics, preds


def _hash_df(df: pd.DataFrame) -> str:
    hb = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.md5(hb).hexdigest()


# ----------------- Main flow -----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Path to merged_nfl_data.csv")
    ap.add_argument("--train_weeks", type=int, default=3)
    ap.add_argument("--test_weeks", type=int, default=2)
    args = ap.parse_args()

    data_path = _detect_paths(args.data)
    log.info("Loading dataset: %s", data_path)
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]

    # Detect label configuration
    has_home = all(
        c in df.columns
        for c in ["home_team", "away_team", "home_points_for", "away_points_for"]
    )
    team_col = _first_present(df, ALIASES["team"])
    opp_col = _first_present(df, ALIASES["opponent_team"])
    pf_col = _first_present(df, ALIASES["points_for"])
    pa_col = _first_present(df, ALIASES["points_against"])
    win_col = _first_present(df, ALIASES["win"])

    if has_home:
        mode = "game_home_away"
        label_cols = ["home_points_for", "away_points_for"]
    elif team_col and opp_col and pf_col and pa_col:
        mode = "team_points"
        label_cols = [pf_col, pa_col]
    elif team_col and opp_col and win_col:
        mode = "team_win_only"
        label_cols = [win_col]
    else:
        need = (
            "Either: ['home_team','away_team','home_points_for','away_points_for'] "
            "or: ['team','opponent_team','points_for','points_against'] "
            "or: ['team','opponent_team','win']"
        )
        raise SystemExit(
            f"Missing labels. Provide {need}. Present columns: {list(df.columns)[:20]} ..."
        )

    # Split
    train_df, test_df, split_info = _split_latest_season(
        df, args.train_weeks, args.test_weeks
    )

    # Build features
    num_cols, cat_cols, _ = _build_feature_lists(df, label_cols)
    pre = _make_preprocessor(num_cols, cat_cols)

    # Fit preprocessor
    Xtr = pre.fit_transform(
        train_df[num_cols + cat_cols] if cat_cols else train_df[num_cols]
    )
    Xte = pre.transform(test_df[num_cols + cat_cols] if cat_cols else test_df[num_cols])
    w = _recency_weights(train_df)

    artifacts = {}
    metrics_report = {}
    preds_export = None

    if mode == "game_home_away":
        # Targets
        ytr_home = train_df["home_points_for"].astype(float).to_numpy()
        ytr_away = train_df["away_points_for"].astype(float).to_numpy()
        yte_home = test_df["home_points_for"].astype(float).to_numpy()
        yte_away = test_df["away_points_for"].astype(float).to_numpy()
        train_df["home_win"] = (
            train_df["home_points_for"] > train_df["away_points_for"]
        ).astype(int)
        test_df["home_win"] = (
            test_df["home_points_for"] > test_df["away_points_for"]
        ).astype(int)
        ytr_win = train_df["home_win"].to_numpy()
        yte_win = test_df["home_win"].to_numpy()

        home_reg, home_res = _fit_regressor(
            Xtr, ytr_home, Xte, yte_home, "home_score", w
        )
        away_reg, away_res = _fit_regressor(
            Xtr, ytr_away, Xte, yte_away, "away_score", w
        )
        win_clf, win_res, preds = _fit_classifier(Xtr, ytr_win, Xte, yte_win, w)

        joblib.dump(home_reg, MODELS_DIR / "home_model.joblib")
        joblib.dump(away_reg, MODELS_DIR / "away_model.joblib")
        joblib.dump(win_clf, MODELS_DIR / "win_clf_calibrated.joblib")
        artifacts.update(
            {
                "home_model": "home_model.joblib",
                "away_model": "away_model.joblib",
                "win_model": "win_clf_calibrated.joblib",
            }
        )
        metrics_report.update(
            {
                "home_score_regressor": home_res,
                "away_score_regressor": away_res,
                "win_classifier": win_res,
            }
        )
        preds_export = preds.assign(
            season=test_df["season"].to_numpy(),
            week=test_df["week"].to_numpy(),
            home_team=test_df.get(
                "home_team", pd.Series(index=test_df.index, dtype="object")
            ),
            away_team=test_df.get(
                "away_team", pd.Series(index=test_df.index, dtype="object")
            ),
            true_home_win=(test_df["home_points_for"] > test_df["away_points_for"])
            .astype(int)
            .to_numpy(),
        )
    elif mode == "team_points":
        # Team-level regression + win classifier
        ytr_pts = train_df[pf_col].astype(float).to_numpy()
        yte_pts = test_df[pf_col].astype(float).to_numpy()
        ytr_win = (train_df[pf_col] > train_df[pa_col]).astype(int).to_numpy()
        yte_win = (test_df[pf_col] > test_df[pa_col]).astype(int).to_numpy()

        team_reg, team_res = _fit_regressor(
            Xtr, ytr_pts, Xte, yte_pts, "team_points_for", w
        )
        win_clf, win_res, preds = _fit_classifier(Xtr, ytr_win, Xte, yte_win, w)

        joblib.dump(team_reg, MODELS_DIR / "team_points_model.joblib")
        joblib.dump(win_clf, MODELS_DIR / "win_clf_calibrated.joblib")
        artifacts.update(
            {
                "team_points_model": "team_points_model.joblib",
                "win_model": "win_clf_calibrated.joblib",
            }
        )
        metrics_report.update(
            {"team_points_regressor": team_res, "win_classifier": win_res}
        )
        preds_export = preds.assign(
            season=test_df["season"].to_numpy(),
            week=test_df["week"].to_numpy(),
            team=test_df[team_col].to_numpy(),
            opponent=test_df[opp_col].to_numpy(),
            true_win=(test_df[pf_col] > test_df[pa_col]).astype(int).to_numpy(),
        )
    else:  # team_win_only
        ytr_win = train_df[win_col].astype(int).to_numpy()
        yte_win = test_df[win_col].astype(int).to_numpy()

        win_clf, win_res, preds = _fit_classifier(Xtr, ytr_win, Xte, yte_win, w)
        joblib.dump(win_clf, MODELS_DIR / "win_clf_calibrated.joblib")
        artifacts.update({"win_model": "win_clf_calibrated.joblib"})
        metrics_report.update({"win_classifier": win_res})
        preds_export = preds.assign(
            season=test_df["season"].to_numpy(),
            week=test_df["week"].to_numpy(),
            team=test_df[team_col].to_numpy(),
            opponent=test_df[opp_col].to_numpy(),
            true_win=test_df[win_col].astype(int).to_numpy(),
        )

    # Save preprocessor
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")

    # Save predictions
    if preds_export is not None:
        preds_export.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    # Training report and metadata
    combined = pd.concat([train_df, test_df], ignore_index=True)
    dataset_hash = _hash_df(combined)
    training_report = {
        "training_timestamp_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "dataset": {
            "path": str(data_path),
            "hash": dataset_hash,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "split": split_info,
        },
        "features": {
            "numeric": num_cols,
            "categorical": cat_cols,
            "count": len(num_cols) + len(cat_cols),
        },
        "models": metrics_report,
    }
    (MODELS_DIR / "training_report.json").write_text(
        json.dumps(training_report, indent=2)
    )

    metadata = {
        "training_timestamp_utc": training_report["training_timestamp_utc"],
        "dataset_hash": dataset_hash,
        "models": artifacts,
        "preprocessor": "preprocessor.joblib",
        "production_ready": True,  # caller should apply thresholds downstream
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    log.info("Saved artifacts under %s", MODELS_DIR)


if __name__ == "__main__":
    main()
