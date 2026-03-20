#!/usr/bin/env python
# ==========================================
# File: backend/utils/feature_helpers.py
# Role: Backend helper module.
# Input Data: Function inputs.
# Output Data: Module outputs.
# Dependencies: __future__, logging, os, pathlib
# Notes: Shared utilities.
# ==========================================

# -*- coding: utf-8 -*-
"""
feature_helpers.py (Enhanced v1.1)
===================================

Safe, minimal helpers shared with backend/main.py and dataset builders.

This module intentionally stays lightweight:
  - Provides deterministic team/season/week normalization
  - Supplies leak-safe prior feature utilities used during dataset prep
  - Exposes resolve_model_path so env overrides work consistently

All functions are defensive and no-op when inputs are missing.

Version History:
  v1.1 - Enhanced: Fixed syntax, improved docs, added type safety
  v1.0 - Initial implementation

Example Usage:
    >>> from feature_helpers import to_team_abbr, coerce_season_week
    >>> to_team_abbr("wsh")  # Returns "WAS"
    >>> df = coerce_season_week(raw_dataframe)
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from backend.config import MODELS_DIR

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def resolve_model_path(key: str, filename: str) -> Path:
    """
    Resolve a model artifact path with optional env override MODEL_PATH_<KEY>.

    Args:
        key: Model identifier (e.g., 'home_model', 'preprocessor')
        filename: Default filename if env var not set

    Returns:
        Path: Resolved absolute path to model artifact

    Example:
        >>> resolve_model_path('home_model', 'home_model.joblib')
        Path('/path/to/home_model.joblib')
    """
    env_val = os.getenv(f"MODEL_PATH_{key.upper()}")
    if env_val and str(env_val).strip():
        return Path(env_val).expanduser().resolve()

    p = Path(filename)
    if not p.is_absolute():
        p = MODELS_DIR / filename
    return p.resolve()


def to_pandas_safe(obj) -> pd.DataFrame:
    """Accept pandas or Polars DataFrame/LazyFrame; return pandas.DataFrame."""
    if obj.__class__.__module__.startswith("pandas"):
        return obj
    if hasattr(obj, "collect"):
        obj = obj.collect()
    if hasattr(obj, "to_pandas"):
        try:
            return obj.to_pandas(use_pyarrow_extension_array=False)
        except TypeError:
            return obj.to_pandas()
    raise TypeError(f"Unsupported table type for to_pandas_safe: {type(obj)}")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def to_team_abbr(t: str) -> str:
    """
    Normalize team abbreviations to canonical form.

    Args:
        t: Team abbreviation or name

    Returns:
        str: Canonical team abbreviation (uppercase)

    Example:
        >>> to_team_abbr("WSH")  # Returns "WAS"
        >>> to_team_abbr("HOU")  # Returns "HOU" (unchanged)
    """
    fix_map = {
        "WSH": "WAS", "HST": "HOU", "CLV": "CLE",
        "BLT": "BAL", "ARZ": "ARI", "LA": "LAR",
        "STL": "LAR", "SD": "LAC", "OAK": "LV"
    }
    return fix_map.get(str(t).upper(), str(t).upper())


def coerce_season_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely convert season/week columns to nullable integer types.

    Args:
        df: DataFrame with potential season/week columns

    Returns:
        pd.DataFrame: Copy with coerced season/week columns

    Example:
        >>> df = pd.DataFrame({'season': ['2023', '2024'], 'week': ['1', '2']})
        >>> result = coerce_season_week(df)
        >>> result.dtypes['season']  # Int64
    """
    out = df.copy()
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    return out


def make_time_key(df: pd.DataFrame) -> pd.Series:
    """
    Build a monotonic key for chronological sorting (season*100 + week).

    Args:
        df: DataFrame containing season and week columns

    Returns:
        pd.Series: Numeric time key for sorting

    Example:
        >>> df = pd.DataFrame({'season': [2023, 2023], 'week': [1, 2]})
        >>> make_time_key(df)  # Returns [202301, 202302]
    """
    season_num = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
    week_num = pd.to_numeric(df.get("week", 0), errors="coerce").fillna(0).astype(int)
    return season_num * 100 + week_num


def _normalize_feature_cols(raw: Any) -> List[str]:
    """
    Flatten raw_feature_columns metadata into a simple list of column names.

    Args:
        raw: Feature columns input (dict, list, or single value)

    Returns:
        List[str]: Flat list of feature column names

    Example:
        >>> _normalize_feature_cols({'numeric': ['a', 'b'], 'categorical': ['c']})
        ['a', 'b', 'c']
    """
    if raw is None:
        return []

    if isinstance(raw, dict):
        cols: List[str] = []
        for key in ("numeric", "categorical"):
            vals = raw.get(key)
            if isinstance(vals, (list, tuple, set, np.ndarray, pd.Index)):
                cols.extend([str(c) for c in vals])
        return cols

    if isinstance(raw, (list, tuple, set, np.ndarray, pd.Index)):
        return [str(c) for c in raw]

    return [str(raw)]


# ---------------------------------------------------------------------------
# Leak-safe priors (used by dataset prep and defensive runtime filling)
# ---------------------------------------------------------------------------
def _rolling_prior_stats(
    team_game_stats: pd.DataFrame,
    window: int,
    advanced_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute rolling *prior* stats per team with shift(1) to avoid leakage.

    Args:
        team_game_stats: DataFrame of team-game level stats
        window: Rolling window size
        advanced_cols: Additional columns to compute rolling stats for

    Returns:
        pd.DataFrame: Enhanced with prior_* columns

    Note:
        Uses shift(1) to ensure no future information leakage
    """
    df = team_game_stats.copy()
    mask_completed = df[["points_for", "points_against"]].notna().all(axis=1)

    def safe_rolling_mean(series: pd.Series) -> pd.Series:
        s = series.where(mask_completed)
        return (
            s.groupby(df["team"], observed=True)
            .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    df[f"prior_pf_avg_{window}"] = safe_rolling_mean(df["points_for"])
    df[f"prior_pa_avg_{window}"] = safe_rolling_mean(df["points_against"])

    win_flag = (df["points_for"] > df["points_against"]).where(mask_completed)
    df[f"prior_win_pct_{window}"] = (
        win_flag.groupby(df["team"], observed=True)
        .apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    if advanced_cols:
        for col in advanced_cols:
            if col in df.columns:
                df[f"prior_{col}_{window}"] = safe_rolling_mean(df[col])

    return df


def _ffill_prior_features(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill prior_* columns per team to keep future weeks NaN-safe.

    Args:
        wide: Wide-format DataFrame with prior_* columns

    Returns:
        pd.DataFrame: With forward-filled prior features
    """
    out = wide.copy()
    if "time_key" not in out.columns:
        out["time_key"] = make_time_key(out)

    home_prior_cols = [c for c in out.columns if c.startswith("home_prior_")]
    away_prior_cols = [c for c in out.columns if c.startswith("away_prior_")]

    if home_prior_cols and "home_team" in out.columns:
        out = out.sort_values(["home_team", "time_key", "game_id"]).copy()
        out[home_prior_cols] = out.groupby("home_team", group_keys=False)[home_prior_cols].ffill()

    if away_prior_cols and "away_team" in out.columns:
        out = out.sort_values(["away_team", "time_key", "game_id"]).copy()
        out[away_prior_cols] = out.groupby("away_team", group_keys=False)[away_prior_cols].ffill()

    return out.sort_values(["time_key", "game_id"]).reset_index(drop=True)


def _impute_remaining_prior_nans(
    wide: pd.DataFrame,
    baseline_medians: Optional[Union[pd.Series, dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Fill remaining prior_* NaNs with neutral medians derived from completed rows.

    Args:
        wide: DataFrame with prior_* columns

    Returns:
        pd.DataFrame: With imputed missing values
    """
    out = wide.copy()
    prior_cols = [c for c in out.columns if c.startswith(("home_prior_", "away_prior_"))]
    if not prior_cols:
        return out

    numeric_priors = out[prior_cols].apply(pd.to_numeric, errors="coerce")
    out[prior_cols] = numeric_priors

    if baseline_medians is None:
        if {"home_points_for", "away_points_for"}.issubset(out.columns):
            completed_mask = (
                pd.to_numeric(out["home_points_for"], errors="coerce").notna()
                & pd.to_numeric(out["away_points_for"], errors="coerce").notna()
            )
        else:
            completed_mask = pd.Series(True, index=out.index)

        baseline = numeric_priors.loc[completed_mask].median(numeric_only=True)
        if baseline.dropna().empty:
            baseline = numeric_priors.median(numeric_only=True)
    elif isinstance(baseline_medians, pd.Series):
        baseline = pd.to_numeric(baseline_medians, errors="coerce")
    else:
        baseline = pd.to_numeric(pd.Series(baseline_medians), errors="coerce")

    baseline = baseline.reindex(prior_cols)
    if baseline.dropna().empty:
        baseline = pd.Series(
            {
                col: (0.5 if ("win_pct" in col or "completion_pct" in col) else 0.0)
                for col in prior_cols
            }
        )

    out[prior_cols] = out[prior_cols].fillna(baseline)

    for col in prior_cols:
        if out[col].isna().any():
            fallback = baseline.get(col)
            if pd.isna(fallback):
                fallback = 0.5 if ("win_pct" in col or "completion_pct" in col) else 0.0
            out[col] = out[col].fillna(float(fallback))

    return out


def ensure_actual_winner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure `home_win` boolean and `actual_winner` string columns exist.

    Args:
        df: DataFrame with game outcome information

    Returns:
        pd.DataFrame: With home_win and actual_winner columns

    Raises:
        ValueError: If unable to infer game outcome from available columns
    """
    out = df.copy()


    if "home_win" in out.columns:
        win_series = pd.Series(out["home_win"], index=out.index, dtype="boolean")
    elif {"home_points_for", "away_points_for"}.issubset(out.columns):
        # Handle NA scores (future games) gracefully
        # If scores are NA, comparison is NA (using nullable boolean)
        pf = pd.to_numeric(out["home_points_for"], errors="coerce")
        pa = pd.to_numeric(out["away_points_for"], errors="coerce")

        # Where both valid, calculate win
        mask_valid = pf.notna() & pa.notna()
        win_series = pd.Series(pd.NA, index=out.index, dtype="boolean")
        win_series.loc[mask_valid] = (pf[mask_valid] > pa[mask_valid])

    elif "winner" in out.columns:
        winner_col = out["winner"]
        if pd.api.types.is_bool_dtype(winner_col.dtype):
            win_series = pd.Series(winner_col, index=out.index, dtype="boolean")
        else:
            win_series = pd.Series(pd.NA, index=out.index, dtype="boolean")
            win_series.loc[winner_col == out["home_team"]] = True
            win_series.loc[winner_col == out["away_team"]] = False
    else:
        # If still failing, check if we can just default to all NA (e.g. inference only)
        # But for training this is bad. For live prediction, it's expected for limited rows.
        # Let's inspect columns to help debugging
        raise ValueError(f"Need either 'home_win', scores, or 'winner' to infer outcome. Found: {list(out.columns)}")

    out["home_win"] = win_series
    actual = pd.Series(pd.NA, index=out.index, dtype="string")
    actual.loc[out["home_win"] == True] = out.loc[out["home_win"] == True, "home_team"].astype("string")
    actual.loc[out["home_win"] == False] = out.loc[out["home_win"] == False, "away_team"].astype("string")
    out["actual_winner"] = actual
    return out


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe preprocessing hook: coerce season/week and return copy.

    Args:
        df: Raw dataset DataFrame

    Returns:
        pd.DataFrame: Preprocessed copy

    Note:
        Gracefully handles errors by returning original DataFrame
    """
    try:
        return coerce_season_week(df)
    except Exception as e:
        log.warning("process_dataset failed; returning input. Error: %s", e)
        return df


# Export public API
__all__ = [
    'resolve_model_path',
    'to_team_abbr',
    'coerce_season_week',
    'make_time_key',
    'ensure_actual_winner',
    'process_dataset',
    '_rolling_prior_stats'
]
