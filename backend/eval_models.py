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
    confusion_matrix,      # NEW
    precision_score,       # NEW
    recall_score,          # NEW
    f1_score,              # NEW
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
    Return a per-season metrics table for a score regressor.

    Metrics
    -------
    - MAE  (Mean Absolute Error):
        Average absolute difference |y_true - y_pred|.
        Intuition: "On average, how many points off are we per game?"

    - RMSE (Root Mean Squared Error):
        sqrt(mean((y_true - y_pred)^2)).
        Intuition: Heavily penalizes large misses; sensitive to blow-out errors.

    - R² (coefficient of determination):
        1 - SS_res / SS_tot.
        Intuition: Fraction of variance in scores explained by the model.
        R² ~ 1   → very strong explanatory power.
        R² ~ 0   → no better than predicting the mean.
        R² < 0   → worse than just predicting the average every time.
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

    Parameters
    ----------
    y_true : array-like of {0,1} or bool
        Ground truth labels for home_win (1 = home team wins, 0 = loses).

    prob_model : array-like of float
        Model-predicted probabilities P(home team wins).

    prob_moneyline : array-like of float, optional
        Implied win probabilities from sportsbook moneyline odds, if present.

    Metrics (probability-based)
    ---------------------------
    - Brier score (mean squared probability error):
        mean((prob_pred - y_true)^2).
        Intuition: measures both calibration and sharpness; lower is better.

    - LogLoss (cross-entropy loss):
        -mean(y*log(p) + (1-y)*log(1-p)).
        Intuition: punishes overconfident wrong predictions very harshly.
        Lower is better; extremely wrong 0.99 vs 0 labels hurt a lot.

    - ROC AUC:
        Probability that a randomly chosen positive (home win) has a higher
        predicted probability than a randomly chosen negative (home loss).
        Intuition: ranking quality independent of a fixed threshold.
        0.5 ~ random; 0.7–0.8 decent; 0.9+ strong.

    - Accuracy (at 0.48 threshold):
        Fraction of games where (prob >= 0.48) matches the true outcome.
        Intuition: easy to read, but can be misleading on imbalanced data or
        if we care about probability quality rather than just final decisions.
    """
    y_true = y_true.astype(int)

    prob_model = prob_model.astype(float)


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
# Threshold-based confusion matrix + precision/recall/F1 snapshot
# ---------------------------------------------------------------------


def build_threshold_diagnostics(
    y_true: np.ndarray,
    prob_model: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a confusion-matrix-style table and threshold metrics for the classifier.

    Parameters
    ----------
    y_true : array-like of {0,1}
        True labels for home_win (1 = home team wins).

    prob_model : array-like of float
        Predicted probabilities P(home team wins).

    threshold : float, default=0.5
        Decision threshold: prob >= threshold => predict "home win".

    Returns
    -------
    cm_df : pd.DataFrame
        Confusion matrix table with counts:
            rows    = Actual class (0 = home_loss, 1 = home_win)
            columns = Predicted class at given threshold.

    metrics : pd.Series
        Single-row series containing:
            - threshold
            - accuracy
            - precision
            - recall
            - f1

    Intuition (translation layer)
    -----------------------------
    - Confusion matrix:
        Breaks performance into:
          * True Positives  (TP): correctly predicted home wins.
          * True Negatives  (TN): correctly predicted home losses.
          * False Positives (FP): predicted win but actually lost.
          * False Negatives (FN): predicted loss but actually won.

    - Precision:
        Of all games we *called* home wins, how many were actually wins?
        (Trustworthiness of positive predictions.)

    - Recall (sensitivity):
        Of all true home wins, how many did we correctly predict as wins?
        (How many wins we "caught".)

    - F1 score:
        Harmonic mean of precision and recall; high only if both are high.
        Useful when we care about a balance between catching wins and avoiding
        false alarms.
    """
    y_true = y_true.astype(int)
    y_pred = (prob_model >= threshold).astype(int)

    # Confusion matrix counts: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    cm_df = pd.DataFrame(
        {
            "Pred_0_home_loss": [tn, fn],
            "Pred_1_home_win": [fp, tp],
        },
        index=["Actual_0_home_loss", "Actual_1_home_win"],
    )

    metrics = pd.Series(
        {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }
    )

    return cm_df, metrics

# ---------------------------------------------------------------------
# Textual report synthesis
# ---------------------------------------------------------------------


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Compute a simple weighted mean with protection against division by zero."""
    w = weights.astype(float)
    s = series.astype(float)
    total_weight = w.sum()
    if total_weight <= 0:
        return float("nan")
    return float((s * w).sum() / total_weight)


def _describe_r2(r2: float) -> str:
    """Short, human-readable interpretation bucket for R²."""
    if pd.isna(r2):
        return "R² is undefined (no variance or missing data)."
    if r2 >= 0.75:
        return "R² is high; the model explains most of the score variability."
    if r2 >= 0.5:
        return "R² is moderate; the model captures a meaningful share of score variability."
    if r2 >= 0.0:
        return "R² is low; the model barely improves over predicting the average score."
    return "R² is negative; the model performs worse than a constant average-score baseline."


def _format_improvement(model_val: float, baseline_val: float, higher_is_better: bool) -> str:
    """
    Turn a pair of metric values into a short comparative sentence fragment.

    Example outputs:
      - "better by ~12% (0.18 vs 0.21)"
      - "roughly on par (0.19 vs 0.20)"
      - "worse by ~8% (0.23 vs 0.21)"
    """
    if pd.isna(model_val) or pd.isna(baseline_val):
        return "comparison not available (missing values)."

    if baseline_val == 0:
        diff = model_val - baseline_val
        if (higher_is_better and diff > 0) or (not higher_is_better and diff < 0):
            direction = "better"
        elif diff == 0:
            direction = "roughly on par"
        else:
            direction = "worse"
        return f"{direction} ({model_val:.3f} vs {baseline_val:.3f})"

    if higher_is_better:
        improvement = model_val - baseline_val
        frac = improvement / abs(baseline_val)
    else:
        improvement = baseline_val - model_val
        frac = improvement / abs(baseline_val)

    pct = frac * 100.0
    if abs(pct) < 3:
        direction = "roughly on par"
    elif pct > 0:
        direction = f"better by ~{pct:.1f}%"
    else:
        direction = f"worse by ~{abs(pct):.1f}%"

    return f"{direction} ({model_val:.3f} vs {baseline_val:.3f})"


def build_textual_report(
    dataset_name: str,
    train_end_season: int,
    train_end_week: int,
    df_reg_home: pd.DataFrame,
    df_reg_away: pd.DataFrame,
    df_clf_overall: pd.DataFrame,
    df_clf_season: pd.DataFrame,
    cm_df: pd.DataFrame,
    thresh_metrics: pd.Series,
) -> str:
    """
    Construct a structured, human-readable evaluation report.

    The report has three layers:
      1) Factual: echo key metric values and tables.
      2) Analytical: highlight whether the model beats baselines.
      3) Intuitive: explain what this means in plain language.

    Output
    ------
    Multi-line Markdown-formatted string suitable for saving to a file.
    """
    lines: list[str] = []

    # Header
    lines.append("# NFL Model Evaluation Report")
    lines.append("")
    lines.append(f"- **Dataset**: `{dataset_name}`")
    lines.append(f"- **Train end boundary**: season `{train_end_season}`, week `{train_end_week}`")
    lines.append("")

    # -----------------------
    # Regression performance
    # -----------------------
    for label, df_reg in [("Home score", df_reg_home), ("Away score", df_reg_away)]:
        lines.append(f"## {label} regression performance")
        if df_reg.empty:
            lines.append("No holdout games found for this regressor.")
            lines.append("")
            continue

        w = df_reg["n_games"]
        mae = _weighted_mean(df_reg["MAE"], w)
        rmse = _weighted_mean(df_reg["RMSE"], w)
        r2 = _weighted_mean(df_reg["R2"], w)

        lines.append(
            f"- **Average MAE**: {mae:.2f} points. "
            "On average, the model's score prediction is this many points off."
        )
        lines.append(
            f"- **Average RMSE**: {rmse:.2f} points. "
            "Large mistakes are amplified here, so a big gap vs MAE indicates occasional blowouts."
        )
        lines.append(f"- **Average R²**: {r2:.3f}. {_describe_r2(r2)}")
        lines.append("")
        lines.append("Per-season breakdown:")
        lines.append("")
        lines.append(df_reg.sort_values("season").to_string(index=False))
        lines.append("")

    # -----------------------
    # Classification summary
    # -----------------------
    lines.append("## Win classifier vs baselines")
    if df_clf_overall.empty:
        lines.append("No overall classifier metrics available.")
    else:
        clf_row = df_clf_overall[
            (df_clf_overall["model"] == "win_classifier")
            & (df_clf_overall["scope"] == "overall")
        ].iloc[0]

        base_home = df_clf_overall[
            (df_clf_overall["model"] == "baseline_always_home")
            & (df_clf_overall["scope"] == "overall")
        ].iloc[0]

        base_ml = None
        ml_mask = (df_clf_overall["model"] == "baseline_moneyline") & (
            df_clf_overall["scope"] == "overall"
        )
        if ml_mask.any():
            base_ml = df_clf_overall[ml_mask].iloc[0]

        lines.append(
            "- **Brier score vs always-home**: "
            + _format_improvement(clf_row["Brier"], base_home["Brier"], higher_is_better=False)
        )
        lines.append(
            "- **LogLoss vs always-home**: "
            + _format_improvement(clf_row["LogLoss"], base_home["LogLoss"], higher_is_better=False)
        )
        lines.append(
            "- **Accuracy vs always-home**: "
            + _format_improvement(clf_row["Accuracy"], base_home["Accuracy"], higher_is_better=True)
        )

        if base_ml is not None:
            lines.append(
                "- **Brier score vs moneyline**: "
                + _format_improvement(clf_row["Brier"], base_ml["Brier"], higher_is_better=False)
            )
            lines.append(
                "- **LogLoss vs moneyline**: "
                + _format_improvement(
                    clf_row["LogLoss"], base_ml["LogLoss"], higher_is_better=False
                )
            )
            lines.append(
                "- **AUC vs moneyline**: "
                + _format_improvement(clf_row["AUC"], base_ml["AUC"], higher_is_better=True)
            )

        lines.append("")
        lines.append("Overall classifier metrics table:")
        lines.append("")
        lines.append(df_clf_overall.to_string(index=False))

    lines.append("")
    lines.append("Per-season classifier metrics:")
    lines.append("")
    if df_clf_season.empty:
        lines.append("No per-season classifier metrics available.")
    else:
        lines.append(df_clf_season.sort_values(["season", "model"]).to_string(index=False))

    # -----------------------
    # Threshold diagnostics
    # -----------------------
    lines.append("")
    lines.append("## Threshold diagnostics at 0.5")
    if cm_df is None or thresh_metrics is None:
        lines.append("Threshold diagnostics were not computed.")
    else:
        acc = thresh_metrics["accuracy"]
        prec = thresh_metrics["precision"]
        rec = thresh_metrics["recall"]
        f1_val = thresh_metrics["f1"]

        lines.append(
            f"- **Accuracy**: {acc:.3f} → overall fraction of correctly classified games."
        )
        lines.append(
            f"- **Precision (home win)**: {prec:.3f} → among predicted home wins, this fraction were actually wins."
        )
        lines.append(
            f"- **Recall (home win)**: {rec:.3f} → among all true home wins, this fraction were correctly predicted."
        )
        lines.append(
            f"- **F1 score (home win)**: {f1_val:.3f} → balance between precision and recall for predicting home wins."
        )

        lines.append("")
        lines.append("Confusion matrix (rows = actual, columns = predicted):")
        lines.append("")
        lines.append(cm_df.to_string())

        tn = cm_df.loc["Actual_0_home_loss", "Pred_0_home_loss"]
        fp = cm_df.loc["Actual_0_home_loss", "Pred_1_home_win"]
        fn = cm_df.loc["Actual_1_home_win", "Pred_0_home_loss"]
        tp = cm_df.loc["Actual_1_home_win", "Pred_1_home_win"]

        total = tn + fp + fn + tp
        if total > 0:
            fp_rate = fp / total
            fn_rate = fn / total
            if fp_rate > fn_rate * 1.2:
                lines.append(
                    f"- The model commits more **false positives** (predicted home win, actually loss): about {fp_rate:.1%} of all games."
                )
            elif fn_rate > fp_rate * 1.2:
                lines.append(
                    f"- The model commits more **false negatives** (predicted home loss, actually win): about {fn_rate:.1%} of all games."
                )
            else:
                lines.append(
                    "- The balance between false positives and false negatives is relatively even."
                )

    lines.append("")
    lines.append(
        "_This report was generated automatically by eval_models.py to provide both raw metrics "
        "and an interpretation layer that is readable by humans working with the model._"
    )
    lines.append("")

    return "\n".join(lines)

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

    # -----------------------------------------------------------------
    # Threshold-based confusion matrix & precision/recall/F1 snapshot
    # -----------------------------------------------------------------
    cm_df, thresh_metrics = build_threshold_diagnostics(
        y_true=y_win_holdout,
        prob_model=prob_win_holdout,
        threshold=0.5,
    )

    print("\n=== Win classifier confusion matrix (holdout, threshold=0.5) ===")
    print(cm_df.to_string())

    print("\n=== Win classifier threshold metrics (holdout, threshold=0.5) ===")
    # Convert Series to single-row DataFrame for nicer alignment.
    print(thresh_metrics.to_frame().T.to_string(index=False))

    # -----------------------------------------------------------------
    # Structured textual report saved to disk
    # -----------------------------------------------------------------
    report_text = build_textual_report(
        dataset_name=csv_path.name,
        train_end_season=args.train_end_season,
        train_end_week=args.train_end_week,
        df_reg_home=df_reg_home,
        df_reg_away=df_reg_away,
        df_clf_overall=df_clf_overall,
        df_clf_season=df_clf_season,
        cm_df=cm_df,
        thresh_metrics=thresh_metrics,
    )

    report_path = csv_path.with_name(csv_path.stem + "_eval_report.md")
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n=== Saved textual evaluation report to {report_path} ===")



if __name__ == "__main__":
    main()
