"""
train_home_win_expert.py

Leak-safe, time-split training + calibrated inference for NFL home_win.

Key features:
- Time split: train <= 2023, calibrate on 2024, test on 2025
- Strict leakage prevention:
    * drops explicit targets and ID/reporting columns
    * drops near-empty columns (>=95% missing)
    * auto-drops suspicious "post-game" columns via missingness mismatch
    * drops constant-in-train columns
- Robust preprocessing:
    * median impute numeric NaNs
    * most-frequent + ordinal-encode categoricals
    * coerces numeric-looking object columns to numeric
- Calibration:
    * Platt scaling fitted on 2024 (fallback if single-class)
- Metrics:
    * ROC AUC (if valid), LogLoss, Brier, ECE + a calibration table
- Outputs:
    * predictions_future.csv (for rows where home_win is null)
    * feature_manifest.json (audit trail: features used and why columns were dropped)

Run:
    python train_home_win_expert.py --data backend/data/datasets/game_features_20260109_clean.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


# -----------------------------
# Reproducibility
# -----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# -----------------------------
# Dataset contract
# -----------------------------
TARGET_COLS = [
    "home_points_for",
    "away_points_for",
    "home_win",
    "winner",
    "point_diff",
]

# Keep only for output/reporting, never for training features
REPORT_COLS = ["season", "week", "game_id", "home_team", "away_team"]

# ID-like columns to exclude from model features
ID_COLS = [
    "season",
    "week",
    "game_id",
    "home_game_date",
    "home_team",
    "away_team",
]


@dataclass(frozen=True)
class TimeSplit:
    """Season-based forward split (deployment-realistic)."""
    train_max_season: int = 2023
    calib_season: int = 2024
    test_season: int = 2025


# -----------------------------
# Utilities
# -----------------------------
def stable_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stable logit transform for probabilities in (0,1)."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin predicted probabilities and compare avg prediction vs empirical win rate."""
    df = pd.DataFrame({"y": y_true.astype(int), "p": y_prob.astype(float)})
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["p"], bins=bins, include_lowest=True)
    out = (
        df.groupby("bin", observed=True)
          .agg(n=("y", "size"), avg_pred=("p", "mean"), win_rate=("y", "mean"))
          .reset_index()
    )
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum over bins of |avg_pred - win_rate| weighted by bin frequency."""
    tab = calibration_table(y_true, y_prob, n_bins=n_bins)
    total = tab["n"].sum()
    if total == 0:
        return float("nan")
    return float((tab["n"] / total * (tab["avg_pred"] - tab["win_rate"]).abs()).sum())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Defend against BOM/whitespace in column names."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def find_near_empty_cols(df: pd.DataFrame, threshold: float) -> List[str]:
    """Columns with missing fraction >= threshold."""
    miss = df.isna().mean()
    return miss[miss >= threshold].index.tolist()


def find_constant_cols(df_train: pd.DataFrame, cols: List[str]) -> List[str]:
    """Columns constant or all-missing in the training subset."""
    constant = []
    for c in cols:
        if df_train[c].nunique(dropna=True) <= 1:
            constant.append(c)
    return constant


def find_suspicious_postgame_cols(
    df_complete: pd.DataFrame,
    df_future: pd.DataFrame,
    candidate_cols: List[str],
    complete_missing_max: float,
    future_missing_min: float,
) -> List[str]:
    """
    Auto-detect likely post-game columns via missingness mismatch:

    If column is mostly present in completed games (missing <= complete_missing_max)
    but almost always missing in future games (missing >= future_missing_min),
    it's probably not available pregame (boxscore/outcome dependent) => drop.

    This is a practical leakage shield for mixed "historical+future" datasets.
    """
    if len(df_future) == 0:
        return []

    miss_complete = df_complete[candidate_cols].isna().mean()
    miss_future = df_future[candidate_cols].isna().mean()

    suspicious_mask = (miss_complete <= complete_missing_max) & (miss_future >= future_missing_min)
    return miss_complete[suspicious_mask].index.tolist()


def coerce_numeric_object_cols(
    X_train: pd.DataFrame,
    X_other: List[pd.DataFrame],
    min_parse_rate: float = 0.98,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[str]]:
    """
    Convert object columns that are "really numeric" into numeric dtype.

    Why:
    - A single bad string can force pandas to store a numeric column as object.
    - Treating that as categorical is usually worse than numeric impute + tree split.

    Rule:
    - For an object column, try to_numeric(errors="coerce")
    - If >= min_parse_rate of non-null values are parseable, we convert it everywhere.
    """
    converted = []
    X_train = X_train.copy()
    others = [x.copy() for x in X_other]

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
    split: TimeSplit,
    near_empty_threshold: float,
    complete_missing_max: float,
    future_missing_min: float,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Build leak-safe feature list + dropped-by-reason manifest.
    """
    df_complete = df[df["home_win"].notna()].copy()
    df_future = df[df["home_win"].isna()].copy()

    near_empty = find_near_empty_cols(df, threshold=near_empty_threshold)

    base_drop = set(TARGET_COLS) | set(ID_COLS) | set(near_empty)
    candidate = [c for c in df.columns if c not in base_drop]

    # Constant-in-train uses only the training slice
    df_train = df_complete[df_complete["season"] <= split.train_max_season].copy()

    suspicious_postgame = find_suspicious_postgame_cols(
        df_complete=df_complete,
        df_future=df_future,
        candidate_cols=candidate,
        complete_missing_max=complete_missing_max,
        future_missing_min=future_missing_min,
    )

    candidate_after_postgame = [c for c in candidate if c not in set(suspicious_postgame)]
    constant_in_train = find_constant_cols(df_train, candidate_after_postgame)

    features = [c for c in candidate_after_postgame if c not in set(constant_in_train)]

    dropped = {
        "targets": sorted([c for c in df.columns if c in set(TARGET_COLS)]),
        "ids_reporting": sorted([c for c in df.columns if c in set(ID_COLS)]),
        "near_empty": sorted(near_empty),
        "suspicious_postgame": sorted(suspicious_postgame),
        "constant_in_train": sorted(constant_in_train),
    }
    return features, dropped


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Minimal, strong baseline:
    - Numeric: median impute
    - Categorical: most frequent impute + ordinal encoding
    - Model: HistGradientBoostingClassifier (strong in pure sklearn)
    """
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            ), cat_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_SEED,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def try_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """ROC AUC requires both classes; return None if not computable."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="game_features_20260109.csv")
    ap.add_argument("--out", type=str, default="predictions_future.csv")
    ap.add_argument("--manifest", type=str, default="feature_manifest.json")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for predicted_home_win")

    # Leakage guard knobs
    ap.add_argument("--near_empty_threshold", type=float, default=0.95)
    ap.add_argument("--complete_missing_max", type=float, default=0.20)
    ap.add_argument("--future_missing_min", type=float, default=0.95)

    # Numeric coercion knob
    ap.add_argument("--numeric_object_parse_rate", type=float, default=0.98)

    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path.resolve()}")

    df = normalize_columns(pd.read_csv(data_path))

    # Contract checks
    required = set(TARGET_COLS) | set(ID_COLS)
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df_complete = df[df["home_win"].notna()].copy()
    df_future = df[df["home_win"].isna()].copy()

    # Target mapping
    y_series = df_complete["home_win"].map({True: 1, False: 0})
    if y_series.isna().any():
        bad = df_complete.loc[y_series.isna(), "home_win"].unique().tolist()
        raise ValueError(f"home_win has unexpected non-bool values: {bad}")
    y = y_series.astype(int).to_numpy()

    split = TimeSplit()

    # Features + manifest
    features, dropped = build_feature_manifest(
        df=df,
        split=split,
        near_empty_threshold=args.near_empty_threshold,
        complete_missing_max=args.complete_missing_max,
        future_missing_min=args.future_missing_min,
    )

    # Time-based split masks
    train_mask = df_complete["season"] <= split.train_max_season
    calib_mask = df_complete["season"] == split.calib_season
    test_mask = df_complete["season"] == split.test_season

    if train_mask.sum() == 0 or calib_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("One of the time splits is empty; check seasons and TimeSplit settings.")

    X_all = df_complete[features].copy()
    X_train = X_all.loc[train_mask]
    X_calib = X_all.loc[calib_mask]
    X_test = X_all.loc[test_mask]

    y_train = y[train_mask.to_numpy()]
    y_calib = y[calib_mask.to_numpy()]
    y_test = y[test_mask.to_numpy()]

    # Coerce numeric-like object cols (train + all other splits, plus future)
    other_frames = [X_calib, X_test, df_future[features].copy()] if len(df_future) > 0 else [X_calib, X_test]
    X_train, coerced_others, coerced_cols = coerce_numeric_object_cols(
        X_train,
        other_frames,
        min_parse_rate=args.numeric_object_parse_rate,
    )
    X_calib = coerced_others[0]
    X_test = coerced_others[1]
    X_future = coerced_others[2] if len(df_future) > 0 else None

    # Build + fit base pipeline
    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    # Calibration (Platt) with safe fallback
    calibrator: Optional[LogisticRegression] = None
    use_calibration = len(np.unique(y_calib)) >= 2

    p_calib_base = pipe.predict_proba(X_calib)[:, 1]
    if use_calibration:
        z = stable_logit(p_calib_base).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", random_state=RANDOM_SEED)
        lr.fit(z, y_calib)
        calibrator = lr

    def predict_proba(X: pd.DataFrame) -> np.ndarray:
        base_p = pipe.predict_proba(X)[:, 1]
        if calibrator is None:
            return base_p
        z_local = stable_logit(base_p).reshape(-1, 1)
        return calibrator.predict_proba(z_local)[:, 1]

    # Metrics on held-out test season
    p_test = predict_proba(X_test)

    auc = try_roc_auc(y_test, p_test)
    ll = float(log_loss(y_test, p_test, normalize=True))
    brier = float(brier_score_loss(y_test, p_test))
    ece = expected_calibration_error(y_test, p_test, n_bins=10)

    # Manifest (audit trail)
    manifest = {
        "dataset_path": str(data_path),
        "row_counts": {
            "total": int(len(df)),
            "completed": int(len(df_complete)),
            "future": int(len(df_future)),
        },
        "time_split": {
            "train_max_season": split.train_max_season,
            "calib_season": split.calib_season,
            "test_season": split.test_season,
        },
        "leakage_guard": {
            "near_empty_threshold": args.near_empty_threshold,
            "complete_missing_max": args.complete_missing_max,
            "future_missing_min": args.future_missing_min,
        },
        "numeric_object_coercion": {
            "min_parse_rate": args.numeric_object_parse_rate,
            "coerced_columns": coerced_cols,
        },
        "calibration": {
            "method": "platt_logit" if calibrator is not None else "none_fallback",
            "calib_season": split.calib_season,
            "single_class_in_calib": bool(len(np.unique(y_calib)) < 2),
        },
        "n_features_final": int(len(features)),
        "dropped": dropped,
        "features_final": features,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Console report
    print("\n=== Data split ===")
    print(f"Train (<= {split.train_max_season}): {int(train_mask.sum())} rows")
    print(f"Calib ({split.calib_season}):        {int(calib_mask.sum())} rows")
    print(f"Test  ({split.test_season}):         {int(test_mask.sum())} rows")
    print(f"Future (home_win null):             {len(df_future)} rows")

    print("\n=== Leakage / feature hygiene ===")
    print(f"Near-empty dropped (>= {args.near_empty_threshold:.2f} missing): {len(dropped['near_empty'])}")
    print(f"Suspicious post-game dropped:                              {len(dropped['suspicious_postgame'])}")
    print(f"Constant-in-train dropped:                                 {len(dropped['constant_in_train'])}")
    print(f"Final feature count:                                       {len(features)}")
    print(f"Manifest saved: {Path(args.manifest).resolve()}")

    print("\n=== Test metrics (season 2025) ===")
    if auc is None:
        print("ROC AUC : n/a (single-class y_test)")
    else:
        print(f"ROC AUC : {auc:.4f}")
    print(f"LogLoss : {ll:.4f}")
    print(f"Brier   : {brier:.4f}")
    print(f"ECE(10) : {ece:.4f}")

    print("\n=== Calibration table (season 2025) ===")
    tab = calibration_table(y_test, p_test, n_bins=10)
    print(tab.to_string(index=False))

    # Future inference + export
    out_path = Path(args.out)
    if len(df_future) > 0:
        assert X_future is not None
        p_future = predict_proba(X_future)
        pred_future = (p_future >= args.threshold).astype(int)

        out = df_future[REPORT_COLS].copy()
        out["home_win_proba"] = p_future
        out["predicted_home_win"] = pred_future

        # Sorting improves human scanning
        out = out.sort_values(["season", "week", "game_id"], kind="stable")

        out.to_csv(out_path, index=False)
        print(f"\nWrote future predictions: {out_path.resolve()}")
    else:
        print("\nNo future rows found. Nothing to export.")


if __name__ == "__main__":
    main()
