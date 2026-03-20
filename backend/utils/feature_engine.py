# ==========================================
# File: backend/utils/feature_engine.py
# Role: Backend helper module.
# Input Data: Function inputs.
# Output Data: Module outputs.
# Dependencies: typing, pandas, numpy, logging
# Notes: Shared utilities.
# ==========================================

"""
feature_engine.py
=================

Centralized feature engineering logic for the NFL ML Prediction system.
Shared by:
  1. build_csv_datasetsv3.py (Training Data)
  2. services/live_predictor.py (Live Inference)

Ensures that "training" and "inference" use identical math for features.
"""

from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import logging

from utils.feature_helpers import (
    make_time_key,
    _rolling_prior_stats,
)

log = logging.getLogger(__name__)


def calculate_team_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw play-by-play data into per-team per-game advanced metrics.

    Returns:
        DataFrame with columns: [season, week, game_id, team, off_epa_per_play, ...]
    """
    if pbp.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "team"])

    # Ensure critical cols exist
    required = ["season", "week", "game_id", "posteam", "epa", "success", "pass", "yards_gained", "rush_attempt"]
    for c in required:
        if c not in pbp.columns:
            # If missing, try to be safe but warn
            log.warning(f"Missing PBP column {c}, metrics may be zeroed.")
            pbp[c] = 0

    mask_valid = pbp["posteam"].notna()
    df = pbp.loc[mask_valid].copy()

    # Numeric safety
    cols_to_fill = ["epa", "success", "pass", "yards_gained", "rush_attempt", "interception", "fumble_lost"]
    for c in cols_to_fill:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    df["turnover"] = df["interception"] + df["fumble_lost"]
    df["explosive_play"] = (
        ((df["pass"] == 1.0) & (df["yards_gained"] >= 20)) |
        ((df["rush_attempt"] == 1.0) & (df["yards_gained"] >= 15))
    )

    # Aggregation
    grp = ["season", "week", "game_id", "posteam"]
    agg = df.groupby(grp).agg(
        off_epa_per_play=("epa", "mean"),
        off_success_rate=("success", "mean"),
        off_turnovers=("turnover", "sum"),
        off_explosive_rate=("explosive_play", "mean"),
        off_pass_attempts=("pass", "sum"), # approximation
        off_rush_attempts=("rush_attempt", "sum")
    ).reset_index()

    agg["off_total_plays"] = agg["off_pass_attempts"] + agg["off_rush_attempts"]
    agg["off_turnover_rate"] = np.where(
        agg["off_total_plays"] > 0,
        agg["off_turnovers"] / agg["off_total_plays"],
        0.0
    )

    # We also need Defensive EPA (opponent's offensive EPA)
    # This requires Self-Join logic or processing defteam.
    # For simplicity/speed in live inference, we'll process 'defteam'

    def_grp = ["season", "week", "game_id", "defteam"]
    def_agg = df.groupby(def_grp).agg(
        def_epa_allowed=("epa", "mean"),
        def_success_rate_allowed=("success", "mean"),
        def_explosive_rate_allowed=("explosive_play", "mean"),
        def_takeaways=("turnover", "sum"),
        def_plays_faced=("epa", "count") # rough count
    ).reset_index()

    def_agg["def_epa_per_play"] = -1 * def_agg["def_epa_allowed"]
    def_agg["def_takeaway_rate"] = np.where(
        def_agg["def_plays_faced"] > 0,
        def_agg["def_takeaways"] / def_agg["def_plays_faced"],
        0.0
    )

    # Merge Off + Def
    # Rename 'posteam' -> 'team' and 'defteam' -> 'team'
    off_stats = agg.rename(columns={"posteam": "team"})
    def_stats = def_agg.rename(columns={"defteam": "team"})

    # Outer merge to handle teams that might have only played one side (shouldn't happen in standard games)
    merged = off_stats.merge(
        def_stats,
        on=["season", "week", "game_id", "team"],
        how="outer"
    ).fillna(0.0)

    # Drop temp cols
    merged = merged.drop(columns=[
        "off_pass_attempts", "off_rush_attempts", "off_turnovers",
        "off_total_plays", "def_epa_allowed", "def_takeaways", "def_plays_faced"
    ], errors="ignore")

    return merged


def calculate_rolling_features(
    team_games: pd.DataFrame,
    windows: Tuple[int, ...] = (3, 5)
) -> pd.DataFrame:
    """
    Computes rolling averages for a long-format DataFrame (one row per team-game).

    Args:
        team_games: DataFrame sorted by team, season, week. Must allow multiple seasons.
                    Columns: [team, time_key, off_epa_per_play, ...]
    """
    df = team_games.copy()

    if "time_key" not in df.columns:
        df["time_key"] = make_time_key(df)

    # Columns to roll - basically any numeric stat that isn't ID or Result
    exclude = {"season", "week", "game_id", "team", "opponent", "time_key", "is_home", "win", "points_for", "points_against", "game_date"}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # Use existing helper which handles the shift(1) logic securely
    # _rolling_prior_stats expects specific column naming or we can loop manually
    # The existing helper in build_csv_dataset is robust, let's reuse/wrap it logic here
    # or rely on the imported `_rolling_prior_stats`.

    for w in windows:
        # Note: _rolling_prior_stats modifies in place or returns new?
        # It returns new DF with `prior_X_N` columns
        df = _rolling_prior_stats(df, window=w, advanced_cols=numeric_cols)

    return df
