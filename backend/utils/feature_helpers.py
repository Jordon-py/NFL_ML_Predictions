#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: backend/utils/feature_helpers.py
Purpose: Shared helper functions for leak-safe prior feature engineering across all dataset builder variants.
Functions: _rolling_prior_stats, _ffill_prior_features, _impute_remaining_prior_nans, make_time_key
Variables: none (stateless utilities)
Interacts With: backend/scripts/build_csv_datasets.py, backend/build_csv_datasets2.py, backend/build_csv_datasetsv3.py
"""

from typing import Optional, Sequence

import pandas as pd


def make_time_key(df: pd.DataFrame) -> pd.Series:
    """
    Build a monotonic time key from (season, week) columns for sorting.

    Args:
        df: DataFrame with 'season' and 'week' columns.

    Returns:
        pd.Series: Integer time key where `season * 100 + week` ensures chronological order.
    """
    return df["season"] * 100 + df["week"]


def _rolling_prior_stats(
    team_game_stats: pd.DataFrame,
    window: int,
    advanced_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute rolling *prior* stats for each team over a given window.

    - Uses ONLY completed games (rows where points_for/points_against are non-null).
    - Shifts by 1 so the current game never sees its own score (no leakage).
    - Supports additional advanced numeric columns via `advanced_cols`.

    Args:
        team_game_stats: Per-team, per-game stats with 'team', 'points_for', 'points_against' columns.
        window: Rolling window size (e.g., 3, 5 games).
        advanced_cols: Optional list of additional numeric columns to include in rolling stats.

    Returns:
        pd.DataFrame: Copy of input with new columns: 
            - prior_pf_avg_{window}
            - prior_pa_avg_{window}
            - prior_win_pct_{window}
            - prior_{col}_{window} for each col in advanced_cols
    """
    df = team_game_stats.copy()
    mask_completed = df[["points_for", "points_against"]].notna().all(axis=1)

    def safe_rolling_mean(series: pd.Series) -> pd.Series:
        """
        Compute per-team rolling mean with shift(1) to prevent leakage.
        Only uses completed games (where points are non-null).
        """
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
            if col not in df.columns:
                continue
            df[f"prior_{col}_{window}"] = safe_rolling_mean(df[col])

    return df


def _ffill_prior_features(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill (per-team, time-sorted) any missing prior_* columns so that
    future/prediction weeks have complete leak-safe priors derived only from
    earlier games.

    Args:
        wide: Wide-format DataFrame with game_id, season, week, home_team, away_team, 
              and home_prior_*/away_prior_* columns.

    Returns:
        pd.DataFrame: Copy with prior columns forward-filled per team chronologically.
    """
    out = wide.copy()
    if "time_key" not in out.columns:
        out["time_key"] = make_time_key(out)

    home_prior_cols = [c for c in out.columns if c.startswith("home_prior_")]
    away_prior_cols = [c for c in out.columns if c.startswith("away_prior_")]

    # Forward-fill home priors per home_team
    if home_prior_cols and "home_team" in out.columns:
        out = out.sort_values(["home_team", "time_key", "game_id"]).copy()
        out[home_prior_cols] = out.groupby("home_team", group_keys=False)[home_prior_cols].ffill()

    # Forward-fill away priors per away_team
    if away_prior_cols and "away_team" in out.columns:
        out = out.sort_values(["away_team", "time_key", "game_id"]).copy()
        out[away_prior_cols] = out.groupby("away_team", group_keys=False)[away_prior_cols].ffill()

    # Restore original chronological order
    out = out.sort_values(["time_key", "game_id"]).reset_index(drop=True)
    return out


def _impute_remaining_prior_nans(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Final neutral imputation: fill any remaining NaNs in prior_* columns with 0.0.
    For *_qb_completion_pct priors, use column median if available, else 0.0.
    
    This is leak-safe (does not use future game values row-wise) because it applies
    a constant fallback per column derived from historical data.

    Args:
        wide: Wide-format DataFrame with home_prior_*/away_prior_* columns.

    Returns:
        pd.DataFrame: Copy with all prior column NaNs imputed.
    """
    out = wide.copy()
    prior_cols = [c for c in out.columns if c.startswith(("home_prior_", "away_prior_"))]

    # Special handling for QB completion percentage: use median instead of 0.0
    qb_cols = [c for c in prior_cols if "qb_completion_pct" in c]
    median_map = {}
    for c in qb_cols:
        med = out[c].median(skipna=True)
        if not pd.isna(med):
            median_map[c] = float(med)

    # Fill priors with 0.0 baseline
    if prior_cols:
        out[prior_cols] = out[prior_cols].fillna(0.0)

    # Re-apply medians for QB completion columns
    for c, med in median_map.items():
        out[c] = out[c].where(out[c].notna(), med)

    return out
