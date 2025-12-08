"""
evaluation_seasonal.py
======================

Time-aware evaluation for NFL models using the game_features dataset.

Features:
  - Chronological train/holdout split by (season, week)
  - Score regression evaluation (home_points_for, away_points_for)
  - Win classification evaluation (home_win vs model prob)
  - Baseline comparison:
      * always-home classifier
      * moneyline-implied probability (home_moneyline_prob), if available
  - Per-season metric tables for the holdout period

Usage:
  python eval_models.py --csv-path 'game_features_20251208.csv' --train-end-season 2023 --train-end-week 18
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Season-based, time-aware evaluation for NFL models."
    )
    p.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to game_features CSV (e.g. backend/data/game_features_YYYYMMDD.csv).",
    )
    p.add_argument(
        "--train-end-season",
        type=int,
        default=2023,
        help="Last season to include in the training set (inclusive).",
    )
    p.add_argument(
        "--train-end-week",
        type=int,
        default=18,
        help="Last week in the final train season to include (inclusive).",
    )
    return p.parse_args()


def make_time_split_mask(
    df: pd.DataFrame,
    train_end_season: int,
    train_end_week: int,
) -> Tuple[pd.Series, pd.Series]:
    if not {"season", "week"}.issubset(df.columns):
        raise ValueError("Dataset must have 'season' and 'week' columns.")

    season = df["season"].astype(int)
    week = df["week"].astype(int)

    is_train = (season < train_end_season) | (
        (season == train_end_season) & (week <= train_end_week)
    )
    is_holdout = ~is_train

    return is_train, is_holdout


def build_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Decide which columns are numeric vs categorical features.

    We explicitly drop:
      - IDs and time columns
      - Target-like columns: home_points_for, away_points_for, point_diff, home_win
      - Known leak-prone columns: away_win, *_elo_post
    """
    id_cols = [
        "season",
        "week",
        "game_id",
        "game_date",
        "home_game_date",
    ]
    target_like = [
        "home_points_for",
        "away_points_for",
        "point_diff",
        "home_win",
    ]
    leak_prone = [c for c in df.columns if c.endswith("_elo_post")]
    if "away_win" in df.columns:
        leak_prone.append("away_win")

    drop_from_features = set(id_cols + target_like + leak_prone)

    numeric_features = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in drop_from_features
    ]

    categorical_candidates = []
    for col in ["home_team", "away_team", "game_type"]:
        if col in df.columns:
            categorical_candidates.append(col)

    return numeric_features, categorical_candidates


def make_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_depth=3, max_iter=300)),
        ]
    )


def make_classifier_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------


def evaluate_regressor(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasons: np.ndarray,
) -> pd.DataFrame:
    """
    Return a per-season metrics table for a regressor.
    """
    records = []
    for season in sorted(np.unique(seasons)):
        mask = seasons == season
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        records.append(
            {
                "model": name,
                "season": int(season),
                "n_games": int(mask.sum()),
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
            }
        )
    return pd.DataFrame(records)


def evaluate_classifier_with_baselines(
    name: str,
    y_true: np.ndarray,
    prob_model: np.ndarray,
    seasons: np.ndarray,
    prob_moneyline: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate classifier vs baselines, overall and per season.

    y_true: boolean or {0,1} for home_win.
    prob_model: model predicted probabilities for home team winning.
    prob_moneyline: optional baseline from home_moneyline_prob.
    """
    y_true = y_true.astype(int)

    # Always-home baseline
    prob_home_always = np.full_like(prob_model, 0.5, dtype=float)
    pred_home_always = np.ones_like(y_true, dtype=int)

    records_overall = []

    def safe_auc(truth, probs) -> float | None:
        try:
            if np.unique(truth).size < 2:
                return None
            return float(roc_auc_score(truth, probs))
        except Exception:
            return None

    def safe_logloss(truth, probs) -> float | None:
        try:
            eps = 1e-15
            probs_clipped = np.clip(probs, eps, 1 - eps)
            return float(log_loss(truth, probs_clipped))
        except Exception:
            return None

    # Overall model metrics
    overall_model = {
        "model": name,
        "scope": "overall",
        "Brier": float(brier_score_loss(y_true, prob_model)),
        "LogLoss": safe_logloss(y_true, prob_model),
        "AUC": safe_auc(y_true, prob_model),
        "Accuracy": float(
            accuracy_score(y_true, (prob_model >= 0.5).astype(int))
        ),
    }
    records_overall.append(overall_model)

    # Overall always-home baseline
    overall_home = {
        "model": "baseline_always_home",
        "scope": "overall",
        "Brier": float(brier_score_loss(y_true, prob_home_always)),
        "LogLoss": safe_logloss(y_true, prob_home_always),
        "AUC": None,
        "Accuracy": float(accuracy_score(y_true, pred_home_always)),
    }
    records_overall.append(overall_home)

    # Overall moneyline baseline, if available
    if prob_moneyline is not None:
        overall_ml = {
            "model": "baseline_moneyline",
            "scope": "overall",
            "Brier": float(brier_score_loss(y_true, prob_moneyline)),
            "LogLoss": safe_logloss(y_true, prob_moneyline),
            "AUC": safe_auc(y_true, prob_moneyline),
            "Accuracy": float(
                accuracy_score(y_true, (prob_moneyline >= 0.5).astype(int))
            ),
        }
        records_overall.append(overall_ml)

    df_overall = pd.DataFrame(records_overall)

    # Per-season
    season_records = []
    for season in sorted(np.unique(seasons)):
        mask = seasons == season
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        pm = prob_model[mask]
        ph = prob_home_always[mask]
        rec_model = {
            "season": int(season),
            "model": name,
            "n_games": int(mask.sum()),
            "Brier": float(brier_score_loss(yt, pm)),
            "LogLoss": safe_logloss(yt, pm),
            "AUC": safe_auc(yt, pm),
            "Accuracy": float(
                accuracy_score(yt, (pm >= 0.5).astype(int))
            ),
        }
        season_records.append(rec_model)

        rec_home = {
            "season": int(season),
            "model": "baseline_always_home",
            "n_games": int(mask.sum()),
            "Brier": float(brier_score_loss(yt, ph)),
            "LogLoss": safe_logloss(yt, ph),
            "AUC": None,
            "Accuracy": float(
                accuracy_score(yt, np.ones_like(yt, dtype=int))
            ),
        }
        season_records.append(rec_home)

        if prob_moneyline is not None:
            pml = prob_moneyline[mask]
            rec_ml = {
                "season": int(season),
                "model": "baseline_moneyline",
                "n_games": int(mask.sum()),
                "Brier": float(brier_score_loss(yt, pml)),
                "LogLoss": safe_logloss(yt, pml),
                "AUC": safe_auc(yt, pml),
                "Accuracy": float(
                    accuracy_score(yt, (pml >= 0.5).astype(int))
                ),
            }
            season_records.append(rec_ml)

    df_season = pd.DataFrame(season_records)
    return df_overall, df_season


# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")

    # Drop rows without targets for training/eval
    mask_has_scores = df["home_points_for"].notna() & df["away_points_for"].notna()
    mask_has_win = df["home_win"].notna()
    df = df.loc[mask_has_scores & mask_has_win].copy()

    is_train, is_holdout = make_time_split_mask(
        df, args.train_end_season, args.train_end_week
    )

    print(
        f"Train rows: {is_train.sum()}, "
        f"Holdout rows: {is_holdout.sum()} "
        f"(train_end={args.train_end_season} wk {args.train_end_week})"
    )

    numeric_features, categorical_features = build_feature_sets(df)
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {categorical_features}")

    # For this evaluation harness we only use numeric features.
    X = df[numeric_features].copy()
    y_home_score = df["home_points_for"].to_numpy()
    y_away_score = df["away_points_for"].to_numpy()
    y_home_win = df["home_win"].astype(int).to_numpy()
    seasons = df["season"].astype(int).to_numpy()

    X_train = X.loc[is_train].to_numpy()
    X_holdout = X.loc[is_holdout].to_numpy()
    y_home_train = y_home_score[is_train]
    y_home_holdout = y_home_score[is_holdout]
    y_away_train = y_away_score[is_train]
    y_away_holdout = y_away_score[is_holdout]
    y_win_train = y_home_win[is_train]
    y_win_holdout = y_home_win[is_holdout]
    seasons_holdout = seasons[is_holdout]

    # ----------------------------
    # Train simple score regressors
    # ----------------------------
    reg_home = make_regression_pipeline()
    reg_away = make_regression_pipeline()

    reg_home.fit(X_train, y_home_train)
    reg_away.fit(X_train, y_away_train)

    pred_home = reg_home.predict(X_holdout)
    pred_away = reg_away.predict(X_holdout)

    df_reg_home = evaluate_regressor(
        "home_score_reg", y_home_holdout, pred_home, seasons_holdout
    )
    df_reg_away = evaluate_regressor(
        "away_score_reg", y_away_holdout, pred_away, seasons_holdout
    )

    print("\n=== Home score regressor (holdout, per season) ===")
    print(df_reg_home.to_string(index=False))

    print("\n=== Away score regressor (holdout, per season) ===")
    print(df_reg_away.to_string(index=False))

    # ----------------------------
    # Train win classifier
    # ----------------------------
    clf = make_classifier_pipeline()
    clf.fit(X_train, y_win_train)
    prob_win_holdout = clf.predict_proba(X_holdout)[:, 1]

    prob_moneyline = None
    if "home_moneyline_prob" in df.columns:
        prob_moneyline = df.loc[is_holdout, "home_moneyline_prob"].to_numpy()

    df_clf_overall, df_clf_season = evaluate_classifier_with_baselines(
        "win_classifier", y_win_holdout, prob_win_holdout, seasons_holdout, prob_moneyline
    )

    print("\n=== Win classifier vs baselines (overall) ===")
    print(df_clf_overall.to_string(index=False))

    print("\n=== Win classifier vs baselines (holdout, per season) ===")
    print(df_clf_season.sort_values(["season", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
