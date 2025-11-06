#!/usr/bin/env python3
"""NFL dataset builder with leakage guards, stratified splits, and diagnostics.

This module replaces ad-hoc CSV assembly by ingesting an engineered base table
(`merge_dominance.csv`), stripping targets and stale columns, and emitting an
analysis-ready dataset plus diagnostic artifacts. Key improvements:

* Enforces deterministic random state handling for repeatable experiments.
* Detects stale or duplicate games before writing a new dataset version.
* Adds a time-aware validation split (latest completed season) and inference flag
  for unlabeled future games so training code can isolate hold-out rows safely.
* Computes label balance, correlation matrices, and a quick logistic baseline
  (confusion matrix + classification report) to highlight home/away bias before
  model training even begins.
* Persists diagnostics under ``metrics/dataset/`` for fast iteration.

Run from repo root::

    python backend/build_dataset.py --input backend/data/merge_dominance.csv \
        --output backend/data/game_features.csv --random-state 42

"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration dataclasses and helpers
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Runtime configuration for dataset assembly."""

    input_path: Path
    output_path: Path
    metrics_dir: Path
    random_state: int = 42
    validation_strategy: str = "last-season"
    correlation_top_k: int = 15

    @property
    def diagnostics_path(self) -> Path:
        return self.metrics_dir / "dataset_diagnostics.json"


# ---------------------------------------------------------------------------
# Core processing steps
# ---------------------------------------------------------------------------


LEAKAGE_COLUMNS = {
    "home_points_for",
    "away_points_for",
    "point_diff",
    "winner",
    "home_score",
    "away_score",
    "home_win",
    "home_win_prob",
    "away_win_prob",
    "season_home_win_rate",
}


def load_raw_table(cfg: DatasetConfig) -> pd.DataFrame:
    """Load the engineered dominance merge table and ensure expected columns."""

    df = pd.read_csv(cfg.input_path)
    if df.empty:
        raise ValueError(f"Input dataset {cfg.input_path} is empty.")

    required = {"season", "week", "game_id", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df.drop_duplicates(subset=["season", "week", "game_id"])
    df = df.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return df


def detect_stale_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Return high-level quality checks to surface stale or corrupt data."""

    duplicate_rows = int(df.duplicated(subset=["game_id"]).sum())
    seasons = sorted(df["season"].unique())
    weeks_per_season = df.groupby("season")["week"].nunique().to_dict()
    newest_completed_season = (
        df.loc[df["home_win"].notna(), "season"].max()
        if df["home_win"].notna().any()
        else None
    )

    return {
        "duplicate_games": duplicate_rows,
        "season_range": [int(seasons[0]), int(seasons[-1])] if seasons else [],
        "weeks_per_season": {int(k): int(v) for k, v in weeks_per_season.items()},
        "latest_completed_season": int(newest_completed_season)
        if newest_completed_season
        else None,
    }


def strip_leakage_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop high-risk leakage columns while keeping labels for evaluation."""

    present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    features = df.drop(columns=present, errors="ignore")
    return features, present


def add_split_column(df: pd.DataFrame) -> pd.Series:
    """Create a split label: train, validation, or inference for future games."""

    completed_mask = df["home_win"].notna()
    if completed_mask.any():
        latest_season = int(df.loc[completed_mask, "season"].max())
    else:
        latest_season = int(df["season"].max())

    split = pd.Series("train", index=df.index, dtype="string")
    split.loc[df["season"] == latest_season] = "validation"
    split.loc[~completed_mask] = "inference"
    return split


def select_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature column names used for modeling."""

    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("_")
    ]
    categorical_cols = [
        c
        for c in ("home_team", "away_team", "game_type")
        if c in df.columns
    ]
    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------


def compute_label_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize label balance overall and per-season."""

    completed = df[df["home_win"].notna()].copy()
    label_counts = (
        completed["home_win"].value_counts(normalize=True, dropna=False).to_dict()
    )
    season_means = completed.groupby("season")["home_win"].mean().to_dict()

    return {
        "total_games": int(len(completed)),
        "home_win_rate": {str(k): float(v) for k, v in label_counts.items()},
        "season_home_win_rate": {str(int(k)): float(v) for k, v in season_means.items()},
    }


def render_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: List[str],
    cfg: DatasetConfig,
) -> Path:
    """Render a correlation heatmap for the top-K features vs. home_win."""

    if "home_win" not in df.columns:
        raise ValueError("Column 'home_win' is required for correlation diagnostics.")

    completed = df[df["home_win"].notna()].copy()
    if completed.empty:
        raise ValueError("Cannot compute correlation heatmap with no labeled games.")

    # Re-derive numeric candidates from the raw table to avoid surprises from casting
    candidate_numeric = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("_")
    ]
    # keep backward compatibility with caller-provided numeric_cols by intersecting
    filtered_cols = [
        c for c in candidate_numeric if c in numeric_cols and c not in {"season", "week"}
    ]
    if not filtered_cols:
        filtered_cols = [c for c in candidate_numeric if c not in {"season", "week"}]

    corr = (
        completed[filtered_cols + ["home_win"]]
        .apply(pd.to_numeric, errors="coerce")
        .corr(numeric_only=True)["home_win"]
        .abs()
        .sort_values(ascending=False)
    )
    top_cols = [c for c in corr.index if c != "home_win"][: cfg.correlation_top_k]

    heatmap_data = (
        completed[top_cols + ["home_win"]]
        .apply(pd.to_numeric, errors="coerce")
        .corr(numeric_only=True)
    )

    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_path = cfg.metrics_dir / "correlation_heatmap.png"
    plt.figure(figsize=(0.6 * len(top_cols), 8))
    plt.imshow(heatmap_data, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45, ha="right")
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    plt.title("Feature correlation heatmap (top |corr with home_win|")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def logistic_bias_diagnostics(
    df: pd.DataFrame, numeric_cols: List[str], cfg: DatasetConfig
) -> Dict[str, Any]:
    """Train a quick logistic baseline to expose home/away bias patterns."""

    completed = df[df["home_win"].notna()].copy()
    if completed.empty:
        return {
            "note": "No completed games available for bias diagnostics.",
            "metrics": {},
        }

    baseline_features = [
        c
        for c in (
            "moneyline_prob_diff",
            "spread_line",
            "rest_diff",
            "home_prior_pf_avg_3",
            "away_prior_pf_avg_3",
            "home_prior_win_pct_3",
            "away_prior_win_pct_3",
        )
        if c in numeric_cols
    ]
    if len(baseline_features) < 3:
        return {
            "note": "Insufficient shared numeric features for bias diagnostics.",
            "metrics": {},
        }

    X = completed[baseline_features].copy().astype(float)
    X = X.fillna(X.mean(numeric_only=True))
    y = completed["home_win"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=cfg.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=200, random_state=cfg.random_state)
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)

    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()
    home_pred_rate = float(preds.mean())

    return {
        "features_used": baseline_features,
        "home_prediction_rate": home_pred_rate,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def build_dataset(cfg: DatasetConfig) -> Dict[str, Any]:
    """Execute the dataset pipeline and persist diagnostics."""

    np.random.seed(cfg.random_state)

    raw = load_raw_table(cfg)
    stale_info = detect_stale_dataset(raw)
    features, dropped = strip_leakage_columns(raw)

    features["split"] = add_split_column(raw)
    numeric_cols, categorical_cols = select_feature_columns(features)

    diagnostics: Dict[str, Any] = {
        "input_path": str(cfg.input_path),
        "output_path": str(cfg.output_path),
        "random_state": cfg.random_state,
        "total_rows": int(len(raw)),
        "dropped_leakage_columns": dropped,
    }
    diagnostics.update(stale_info)
    diagnostics["label_distribution"] = compute_label_distribution(raw)

    heatmap_path = render_correlation_heatmap(raw, numeric_cols, cfg)
    diagnostics["correlation_heatmap"] = str(heatmap_path)

    bias = logistic_bias_diagnostics(raw, numeric_cols, cfg)
    diagnostics["bias_diagnostics"] = bias

    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    with cfg.diagnostics_path.open("w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)

    # Persist dataset (features + targets for downstream training)
    output_df = raw.copy()
    output_df["split"] = features["split"]
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(cfg.output_path, index=False)

    logging.info("Dataset written to %s (%d rows, %d columns)", cfg.output_path, len(output_df), len(output_df.columns))
    logging.info("Diagnostics saved to %s", cfg.diagnostics_path)
    return diagnostics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned NFL dataset with diagnostics")
    parser.add_argument("--input", type=Path, default=Path("backend/data/merge_dominance.csv"), help="Input engineered dataset path")
    parser.add_argument("--output", type=Path, default=Path("backend/data/game_features.csv"), help="Output CSV path")
    parser.add_argument("--metrics-dir", type=Path, default=Path("metrics/dataset"), help="Directory for diagnostic artifacts")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for deterministic operations")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    cfg = DatasetConfig(
        input_path=args.input,
        output_path=args.output,
        metrics_dir=args.metrics_dir,
        random_state=args.random_state,
    )
    diagnostics = build_dataset(cfg)
    logging.info("Diagnostics summary: %s", json.dumps(diagnostics, indent=2)[:800])


if __name__ == "__main__":
    main()
