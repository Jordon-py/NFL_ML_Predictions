"""
backend/enhanced_pipeline.py
----------------------------
Training-time helpers for building a supervised dataset safely.

This module exists primarily to:
  - centralize "feature leakage" rules in one place
  - provide a small, testable `build_dataset()` API used by unit tests

It does NOT implement the full historical feature-engineering pipeline (that lives
in dataset build scripts). Instead, it focuses on the final training-frame hygiene:
removing post-game outcome columns and conservative leakage candidates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


# Columns that directly encode outcomes or are derived from outcomes.
OUTCOME_COLS = {
    "home_points_for",
    "away_points_for",
    "point_diff",
    "winner",
    "home_win",
    "actual_winner",
}


def is_leak_feature(name: str) -> bool:
    """Return True if a column name is considered leakage for training.

    Rule of thumb:
      - columns starting with "_" are internal/derived artifacts -> leakage
      - season-level aggregates often leak future weeks unless time-sliced
      - explicit outcome columns are leakage by definition
    """
    if not name:
        return False
    n = str(name).strip()
    if not n:
        return False

    if n in OUTCOME_COLS:
        return True

    # Internal/derived columns (typically computed from targets).
    if n.startswith("_"):
        return True

    # Conservative: season-level win rates are often computed using future games.
    if n in {"season_home_win_rate", "season_away_win_rate"}:
        return True

    return False


def _build_target_home_win(df: pd.DataFrame) -> pd.Series:
    """Create the binary target y = 1 if home wins else 0."""
    if "home_win" in df.columns:
        # Normalize to {0, 1} (allow bool, int, string)
        raw = df["home_win"]
        if raw.dtype == "bool":
            return raw.astype(int)
        lowered = raw.astype(str).str.strip().str.lower()
        return lowered.isin({"1", "true", "t", "yes", "y"}).astype(int)

    if {"home_points_for", "away_points_for"}.issubset(df.columns):
        home_pf = pd.to_numeric(df["home_points_for"], errors="coerce")
        away_pf = pd.to_numeric(df["away_points_for"], errors="coerce")
        return (home_pf > away_pf).astype(int)

    raise ValueError("Cannot derive target: need home_win or home_points_for/away_points_for")


def build_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load a CSV and return (X, y, groups, df_raw).

    - X: feature matrix with leakage/outcome columns removed
    - y: binary target (home_win)
    - groups: grouping key for time-aware CV (defaults to season)
    - df_raw: original dataframe (useful for debugging)
    """
    path = Path(csv_path)
    df_raw = pd.read_csv(path)

    y = _build_target_home_win(df_raw)
    groups = (
        pd.to_numeric(df_raw.get("season"), errors="coerce").fillna(0).astype(int)
        if "season" in df_raw.columns
        else pd.Series([0] * len(df_raw))
    )

    # Build feature frame by dropping explicit outcomes + conservative leakage.
    drop_cols = set(OUTCOME_COLS)
    drop_cols.update([c for c in df_raw.columns if is_leak_feature(c)])
    X = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns], errors="ignore")

    return X, y, groups, df_raw

