#!/usr/bin/env python
"""
build_csv_datasets.py
=====================

Purpose
-------
Build a *single*, prediction-ready NFL game-level dataset (one row per game)
with leak-free rolling features, advanced EPA-derived priors, betting context,
and normalized team codes.

Key Functions
-------------
- setup_logger(out_dir): configure file + console logging
- load_schedules(seasons): fetch schedule/scores per game
- add_features(sch, windows): add leak-free rolling priors (home_/away_ prefixed)
- build_dataset(start, end, out_dir): orchestrates load → features → write

External Dependencies
---------------------
pandas, numpy, nfl_data_py

Usage Notes
-----------
- Output: single chronologically sorted CSV ``merged_game_features.csv`` written to
    the specified ``out_dir`` (no duplicate root-level copy).
- Rolling stats use ``groupby().rolling(...)`` to prevent future leakage.
- Team codes are minimally normalized to limit join mismatches (LA→LAR, STL→LAR, ...).

**IMPORTANT** TO RUN:
python backend/build_csv_datasets.py --start 2016 --end 2026 --out-dir backend/data

"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl


# -----------------------------
# Configuration and constants
# -----------------------------

ABBR_FIX: Dict[str, str] = {
    "LA": "LAR",  # Rams short
    "STL": "LAR",  # Rams legacy
    "SD": "LAC",  # Chargers legacy
    "OAK": "LV",  # Raiders legacy
    "WSH": "WAS",  # Commanders legacy
}
OUTPUT_DATASET_NAME = "merged_game_features.csv"
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

def make_time_key(df: pd.DataFrame) -> pd.Series:
    """Return sortable integer key YYYYWW from 'season' and 'week' (assumes ints)."""
    return (df["season"].astype(int) * 100) + df["week"].astype(int)


# -----------------------------
# Logging
# -----------------------------


def setup_logger(out_dir: Path) -> None:
    """
    Initialize both file and console logging so CLI users get progress feedback.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "build_csv_datasets.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info("Logger initialized. Writing to %s", log_file)


# -----------------------------
# Data loading and normalization
# -----------------------------


def _normalize_codes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Normalize legacy team abbreviations to modern codes in the specified columns."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace(ABBR_FIX)
    return out


def _moneyline_to_prob(ml: pd.Series) -> pd.Series:
    """Convert American moneyline odds to implied win probability."""
    ml_numeric = pd.to_numeric(ml, errors="coerce")
    probs = pd.Series(np.nan, index=ml.index, dtype=float)
    if probs.empty:
        return probs
    negative = ml_numeric < 0
    positive = ml_numeric >= 0
<<<<<<< HEAD
<<<<<<< HEAD
    probs.loc[negative] = (-ml_numeric.loc[negative]) / (
        (-ml_numeric.loc[negative]) + 100
    )
    probs.loc[positive & ml_numeric.notna()] = 100 / (
        ml_numeric.loc[positive & ml_numeric.notna()] + 100
    )
=======
    probs.loc[negative] = (-ml_numeric.loc[negative]) / ((-ml_numeric.loc[negative]) + 100)
    probs.loc[positive & ml_numeric.notna()] = 100 / (ml_numeric.loc[positive & ml_numeric.notna()] + 100)
>>>>>>> main
=======
    probs.loc[negative] = (-ml_numeric.loc[negative]) / ((-ml_numeric.loc[negative]) + 100)
    probs.loc[positive & ml_numeric.notna()] = 100 / (ml_numeric.loc[positive & ml_numeric.notna()] + 100)
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    return probs


def load_team_game_metrics(pbp_path: Path) -> pd.DataFrame:
    """Aggregate play-by-play data to per-team, per-game advanced metrics using nflreadpy."""

    # If nflreadpy available, load directly from nflverse; otherwise use cached CSV
    try:
        # Extract season range from pbp_path naming or use default
        # For now, load recent seasons (2016+) where advanced metrics available
        seasons_to_load = list(range(2016, 2026))
<<<<<<< HEAD
<<<<<<< HEAD
        logging.info(
            "Loading play-by-play via nflreadpy for seasons %s", seasons_to_load
        )
=======
        logging.info("Loading play-by-play via nflreadpy for seasons %s", seasons_to_load)
>>>>>>> main
=======
        logging.info("Loading play-by-play via nflreadpy for seasons %s", seasons_to_load)
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

        pbp = nfl.load_pbp(seasons=seasons_to_load).to_pandas()
        logging.info("Loaded %d play-by-play rows from nflreadpy", len(pbp))
    except Exception as exc:
<<<<<<< HEAD
<<<<<<< HEAD
        logging.warning(
            "nflreadpy PBP load failed (%s); falling back to cached CSV", exc
        )
=======
        logging.warning("nflreadpy PBP load failed (%s); falling back to cached CSV", exc)
>>>>>>> main
=======
        logging.warning("nflreadpy PBP load failed (%s); falling back to cached CSV", exc)
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
        if not pbp_path.exists():
            logging.warning("Cached PBP missing; advanced features disabled")
            return pd.DataFrame(columns=["season", "week", "game_id", "team"])
        pbp = pd.read_csv(pbp_path, low_memory=False)
<<<<<<< HEAD
<<<<<<< HEAD

=======
    
>>>>>>> main
=======
    
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    # Ensure required columns exist
    required_cols = ["season", "week", "game_id", "posteam"]
    missing = [c for c in required_cols if c not in pbp.columns]
    if missing:
        logging.warning("PBP missing columns %s; advanced features disabled", missing)
        return pd.DataFrame(columns=["season", "week", "game_id", "team"])

    mask_valid_team = pbp["posteam"].notna()
    pbp = pbp.loc[mask_valid_team].copy()
    if pbp.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "team"])

    pbp["season"] = pbp["season"].astype(int)
    pbp["week"] = pbp["week"].astype(int)

    pbp["epa"] = pbp["epa"].fillna(0.0)
    pbp["success"] = pbp["success"].fillna(0.0)
    pbp["pass"] = pbp["pass"].fillna(0.0)
    pbp["xpass"] = pbp["xpass"].fillna(0.0)
    pbp["pass_attempt"] = pbp["pass_attempt"].fillna(0.0)
    pbp["rush_attempt"] = pbp["rush_attempt"].fillna(0.0)
    pbp["third_down_converted"] = pbp["third_down_converted"].fillna(0.0)
    pbp["third_down_failed"] = pbp["third_down_failed"].fillna(0.0)
    pbp["interception"] = pbp["interception"].fillna(0.0)
    pbp["fumble_lost"] = pbp["fumble_lost"].fillna(0.0)
    pbp["yards_gained"] = pbp["yards_gained"].fillna(0.0)

    pbp["turnover"] = pbp["interception"] + pbp["fumble_lost"]
    pbp["explosive_play"] = ((pbp["pass"] == 1.0) & (pbp["yards_gained"] >= 20)) | (
        (pbp["rush_attempt"] == 1.0) & (pbp["yards_gained"] >= 15)
    )

    off_group = ["season", "week", "game_id", "posteam"]
<<<<<<< HEAD
<<<<<<< HEAD
    off_agg = (
        pbp.groupby(off_group)
        .agg(
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
            off_pass_rate=("pass", "mean"),
            off_expected_pass_rate=("xpass", "mean"),
            off_pass_attempts=("pass_attempt", "sum"),
            off_rush_attempts=("rush_attempt", "sum"),
            off_turnovers=("turnover", "sum"),
            off_explosive_rate=("explosive_play", "mean"),
            off_third_down_conv=("third_down_converted", "sum"),
            off_third_down_fail=("third_down_failed", "sum"),
        )
        .reset_index()
    )

    off_agg["off_third_down_total"] = (
        off_agg["off_third_down_conv"] + off_agg["off_third_down_fail"]
    )
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    off_agg = pbp.groupby(off_group).agg(
        off_epa_per_play=("epa", "mean"),
        off_success_rate=("success", "mean"),
        off_pass_rate=("pass", "mean"),
        off_expected_pass_rate=("xpass", "mean"),
        off_pass_attempts=("pass_attempt", "sum"),
        off_rush_attempts=("rush_attempt", "sum"),
        off_turnovers=("turnover", "sum"),
        off_explosive_rate=("explosive_play", "mean"),
        off_third_down_conv=("third_down_converted", "sum"),
        off_third_down_fail=("third_down_failed", "sum"),
    ).reset_index()

    off_agg["off_third_down_total"] = off_agg["off_third_down_conv"] + off_agg["off_third_down_fail"]
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    off_agg["off_third_down_pct"] = np.where(
        off_agg["off_third_down_total"] > 0,
        off_agg["off_third_down_conv"] / off_agg["off_third_down_total"],
        np.nan,
    )
<<<<<<< HEAD
<<<<<<< HEAD
    off_agg["off_pass_over_expected"] = (
        off_agg["off_pass_rate"] - off_agg["off_expected_pass_rate"]
    )

    def_group = ["season", "week", "game_id", "defteam"]
    def_agg = (
        pbp.groupby(def_group)
        .agg(
            def_epa_allowed=("epa", "mean"),
            def_success_rate_allowed=("success", "mean"),
            def_explosive_rate_allowed=("explosive_play", "mean"),
            def_takeaways=("turnover", "sum"),
            def_pass_attempts_faced=("pass_attempt", "sum"),
            def_rush_attempts_faced=("rush_attempt", "sum"),
        )
        .reset_index()
    )
    def_agg["def_epa_per_play"] = -def_agg["def_epa_allowed"]
    def_agg["def_takeaway_rate"] = np.where(
        (def_agg["def_pass_attempts_faced"] + def_agg["def_rush_attempts_faced"]) > 0,
        def_agg["def_takeaways"]
        / (def_agg["def_pass_attempts_faced"] + def_agg["def_rush_attempts_faced"]),
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    off_agg["off_pass_over_expected"] = off_agg["off_pass_rate"] - off_agg["off_expected_pass_rate"]

    def_group = ["season", "week", "game_id", "defteam"]
    def_agg = pbp.groupby(def_group).agg(
        def_epa_allowed=("epa", "mean"),
        def_success_rate_allowed=("success", "mean"),
        def_explosive_rate_allowed=("explosive_play", "mean"),
        def_takeaways=("turnover", "sum"),
        def_pass_attempts_faced=("pass_attempt", "sum"),
        def_rush_attempts_faced=("rush_attempt", "sum"),
    ).reset_index()
    def_agg["def_epa_per_play"] = -def_agg["def_epa_allowed"]
    def_agg["def_takeaway_rate"] = np.where(
        (def_agg["def_pass_attempts_faced"] + def_agg["def_rush_attempts_faced"]) > 0,
        def_agg["def_takeaways"] / (def_agg["def_pass_attempts_faced"] + def_agg["def_rush_attempts_faced"]),
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
        np.nan,
    )

    metrics = off_agg.rename(columns={"posteam": "team"}).merge(
        def_agg.rename(columns={"defteam": "team"}),
        on=["season", "week", "game_id", "team"],
        how="outer",
    )

<<<<<<< HEAD
<<<<<<< HEAD
    metrics["off_total_plays"] = metrics["off_pass_attempts"].fillna(0) + metrics[
        "off_rush_attempts"
    ].fillna(0)
=======
    metrics["off_total_plays"] = metrics["off_pass_attempts"].fillna(0) + metrics["off_rush_attempts"].fillna(0)
>>>>>>> main
=======
    metrics["off_total_plays"] = metrics["off_pass_attempts"].fillna(0) + metrics["off_rush_attempts"].fillna(0)
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    metrics["off_turnover_rate"] = np.where(
        metrics["off_total_plays"] > 0,
        metrics["off_turnovers"].fillna(0) / metrics["off_total_plays"],
        np.nan,
    )

<<<<<<< HEAD
<<<<<<< HEAD
    metrics = metrics.drop(
        columns=[
            "off_third_down_total",
            "def_epa_allowed",
            "def_pass_attempts_faced",
            "def_rush_attempts_faced",
            "off_pass_attempts",
            "off_rush_attempts",
            "off_turnovers",
            "off_third_down_conv",
            "off_third_down_fail",
            "def_takeaways",
            "off_total_plays",
            "off_pass_rate",
            "off_expected_pass_rate",
        ],
        errors="ignore",
    )
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    metrics = metrics.drop(columns=[
        "off_third_down_total",
        "def_epa_allowed",
        "def_pass_attempts_faced",
        "def_rush_attempts_faced",
        "off_pass_attempts",
        "off_rush_attempts",
        "off_turnovers",
        "off_third_down_conv",
        "off_third_down_fail",
        "def_takeaways",
        "off_total_plays",
        "off_pass_rate",
        "off_expected_pass_rate",
    ], errors="ignore")
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

    return metrics.fillna(np.nan)


def load_player_game_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Load weekly player-level stats from nflreadpy and aggregate to team-game level.
    Provides QB efficiency, RB production, WR targets, etc.
    """
<<<<<<< HEAD
<<<<<<< HEAD

    try:
        logging.info("Loading player stats via nflreadpy for seasons %s", seasons)
        player_stats = nfl.load_player_stats(
            seasons=seasons, summary_level="week"
        ).to_pandas()
        logging.info("Loaded %d player-week records", len(player_stats))

        # Aggregate key metrics by team-game
        if "recent_team" not in player_stats.columns:
            logging.warning(
                "player_stats missing 'recent_team' column; using 'team' if available"
            )
            team_col = "team" if "team" in player_stats.columns else None
        else:
            team_col = "recent_team"

        if not team_col or team_col not in player_stats.columns:
            logging.warning("Cannot determine team column in player_stats; skipping")
            return pd.DataFrame(columns=["season", "week", "team"])

        # QB stats aggregation (top QB per team-week)
        qb_stats = player_stats[player_stats["position"] == "QB"].copy()
        qb_agg = (
            qb_stats.groupby(["season", "week", team_col])
            .agg(
                team_qb_pass_yards=("passing_yards", "sum"),
                team_qb_pass_tds=("passing_tds", "sum"),
                team_qb_interceptions=("interceptions", "sum"),
                team_qb_sacks=("sacks", "sum"),
                team_qb_completions=("completions", "sum"),
                team_qb_attempts=("attempts", "sum"),
            )
            .reset_index()
        )
        qb_agg["team_qb_completion_pct"] = np.where(
            qb_agg["team_qb_attempts"] > 0,
            qb_agg["team_qb_completions"] / qb_agg["team_qb_attempts"],
            np.nan,
        )

        # RB stats aggregation
        rb_stats = player_stats[player_stats["position"] == "RB"].copy()
        rb_agg = (
            rb_stats.groupby(["season", "week", team_col])
            .agg(
                team_rb_rush_yards=("rushing_yards", "sum"),
                team_rb_rush_tds=("rushing_tds", "sum"),
                team_rb_receptions=("receptions", "sum"),
                team_rb_receiving_yards=("receiving_yards", "sum"),
            )
            .reset_index()
        )

        # WR+TE stats aggregation
        pass_catchers = player_stats[player_stats["position"].isin(["WR", "TE"])].copy()
        wr_agg = (
            pass_catchers.groupby(["season", "week", team_col])
            .agg(
                team_wr_targets=("targets", "sum"),
                team_wr_receptions=("receptions", "sum"),
                team_wr_receiving_yards=("receiving_yards", "sum"),
                team_wr_receiving_tds=("receiving_tds", "sum"),
            )
            .reset_index()
        )

        # Merge all player aggregations
        player_team_stats = qb_agg.merge(
            rb_agg, on=["season", "week", team_col], how="outer"
        )
        player_team_stats = player_team_stats.merge(
            wr_agg, on=["season", "week", team_col], how="outer"
        )
        player_team_stats = player_team_stats.rename(columns={team_col: "team"})

        return player_team_stats.fillna(0)

    except Exception as exc:
        logging.warning(
            "Failed to load player stats (%s); player features disabled", exc
        )
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    
    try:
        logging.info("Loading player stats via nflreadpy for seasons %s", seasons)
        player_stats = nfl.load_player_stats(seasons=seasons, summary_level="week").to_pandas()
        logging.info("Loaded %d player-week records", len(player_stats))
        
        # Aggregate key metrics by team-game
        if "recent_team" not in player_stats.columns:
            logging.warning("player_stats missing 'recent_team' column; using 'team' if available")
            team_col = "team" if "team" in player_stats.columns else None
        else:
            team_col = "recent_team"
        
        if not team_col or team_col not in player_stats.columns:
            logging.warning("Cannot determine team column in player_stats; skipping")
            return pd.DataFrame(columns=["season", "week", "team"])
        
        # QB stats aggregation (top QB per team-week)
        qb_stats = player_stats[player_stats["position"] == "QB"].copy()
        qb_agg = qb_stats.groupby(["season", "week", team_col]).agg(
            team_qb_pass_yards=("passing_yards", "sum"),
            team_qb_pass_tds=("passing_tds", "sum"),
            team_qb_interceptions=("interceptions", "sum"),
            team_qb_sacks=("sacks", "sum"),
            team_qb_completions=("completions", "sum"),
            team_qb_attempts=("attempts", "sum"),
        ).reset_index()
        qb_agg["team_qb_completion_pct"] = np.where(
            qb_agg["team_qb_attempts"] > 0,
            qb_agg["team_qb_completions"] / qb_agg["team_qb_attempts"],
            np.nan
        )
        
        # RB stats aggregation
        rb_stats = player_stats[player_stats["position"] == "RB"].copy()
        rb_agg = rb_stats.groupby(["season", "week", team_col]).agg(
            team_rb_rush_yards=("rushing_yards", "sum"),
            team_rb_rush_tds=("rushing_tds", "sum"),
            team_rb_receptions=("receptions", "sum"),
            team_rb_receiving_yards=("receiving_yards", "sum"),
        ).reset_index()
        
        # WR+TE stats aggregation
        pass_catchers = player_stats[player_stats["position"].isin(["WR", "TE"])].copy()
        wr_agg = pass_catchers.groupby(["season", "week", team_col]).agg(
            team_wr_targets=("targets", "sum"),
            team_wr_receptions=("receptions", "sum"),
            team_wr_receiving_yards=("receiving_yards", "sum"),
            team_wr_receiving_tds=("receiving_tds", "sum"),
        ).reset_index()
        
        # Merge all player aggregations
        player_team_stats = qb_agg.merge(rb_agg, on=["season", "week", team_col], how="outer")
        player_team_stats = player_team_stats.merge(wr_agg, on=["season", "week", team_col], how="outer")
        player_team_stats = player_team_stats.rename(columns={team_col: "team"})
        
        return player_team_stats.fillna(0)
        
    except Exception as exc:
        logging.warning("Failed to load player stats (%s); player features disabled", exc)
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
        return pd.DataFrame(columns=["season", "week", "team"])


def load_team_weekly_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Load official team-level stats from nflreadpy at weekly granularity.
    Provides points scored/allowed, yards, turnovers, etc.
    """
<<<<<<< HEAD
<<<<<<< HEAD

    try:
        logging.info("Loading team stats via nflreadpy for seasons %s", seasons)
        team_stats = nfl.load_team_stats(
            seasons=seasons, summary_level="week"
        ).to_pandas()
        logging.info("Loaded %d team-week records", len(team_stats))

        # Select relevant columns for features
        feature_cols = [
            "season",
            "week",
            "team",
            "points_scored",
            "points_allowed",
            "total_yards",
            "total_yards_allowed",
            "turnovers",
            "turnovers_forced",
            "third_down_conversions",
            "third_down_attempts",
            "fourth_down_conversions",
            "fourth_down_attempts",
            "time_of_possession",
        ]
        available_cols = [c for c in feature_cols if c in team_stats.columns]

        return team_stats[available_cols].fillna(0)

=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    
    try:
        logging.info("Loading team stats via nflreadpy for seasons %s", seasons)
        team_stats = nfl.load_team_stats(seasons=seasons, summary_level="week").to_pandas()
        logging.info("Loaded %d team-week records", len(team_stats))
        
        # Select relevant columns for features
        feature_cols = [
            "season", "week", "team",
            "points_scored", "points_allowed",
            "total_yards", "total_yards_allowed",
            "turnovers", "turnovers_forced",
            "third_down_conversions", "third_down_attempts",
            "fourth_down_conversions", "fourth_down_attempts",
            "time_of_possession",
        ]
        available_cols = [c for c in feature_cols if c in team_stats.columns]
        
        return team_stats[available_cols].fillna(0)
        
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    except Exception as exc:
        logging.warning("Failed to load team stats (%s); team features disabled", exc)
        return pd.DataFrame(columns=["season", "week", "team"])

<<<<<<< HEAD
<<<<<<< HEAD

def load_schedules(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
    """
    Load schedules + final scores for given seasons using nflreadpy.

=======
def load_schedules(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
    """
    Load schedules + final scores for given seasons using nflreadpy.
=======
def load_schedules(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
    """
    Load schedules + final scores for given seasons using nflreadpy.
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    
>>>>>>> main
    Args:
        seasons: List of seasons to load
        include_future: If True, includes scheduled games without scores for prediction

    Returns
    -------
    DataFrame with:
      ['season','week','game_id','game_date','home_team','away_team',
       'home_score','away_score', 'spread_line', 'total_line', 'away_rest', 'home_rest']
    """
<<<<<<< HEAD
<<<<<<< HEAD

    # nflreadpy returns Polars DataFrame—convert to pandas
=======
 
    
        # nflreadpy returns Polars DataFrame—convert to pandas
>>>>>>> main
=======
 
    
        # nflreadpy returns Polars DataFrame—convert to pandas
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    sch = nfl.load_schedules(seasons=seasons).to_pandas()

    logging.info("Raw schedules loaded: %d games", len(sch))

    needed = [
<<<<<<< HEAD
        "season",
        "week",
        "game_id",
        "gameday",  # nflverse uses 'gameday'
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "game_type",
        "away_moneyline",
        "home_moneyline",
        "spread_line",
        "total_line",
        "away_rest",
        "home_rest",
=======
        "season", "week", "game_id", "gameday",  # nflverse uses 'gameday'
        "home_team", "away_team", "home_score", "away_score",
        "game_type", "away_moneyline", "home_moneyline",
        "spread_line", "total_line", "away_rest", "home_rest"
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    ]
    missing = [c for c in needed if c not in sch.columns]
    if missing:
        raise RuntimeError(f"Missing schedule columns: {missing}")

    sch = _normalize_codes(sch, ["home_team", "away_team"])
    sch["week"] = sch["week"].astype(int)  # enforce int for monotonic keys
    sch = sch.rename(columns={"gameday": "game_date"})
<<<<<<< HEAD
    sch = sch[
        [
            "season",
            "week",
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "game_type",
            "away_moneyline",
            "home_moneyline",
            "spread_line",
            "total_line",
            "away_rest",
            "home_rest",
        ]
    ].copy()
=======
    sch = sch[[
        "season", "week", "game_id", "game_date",
        "home_team", "away_team", "home_score", "away_score",
        "game_type", "away_moneyline", "home_moneyline", "spread_line", "total_line",
        "away_rest", "home_rest"
    ]].copy()
>>>>>>> main

    if include_future:
        # Keep both completed and scheduled games
        completed = sch.dropna(subset=["home_score", "away_score"]).reset_index(
            drop=True
        )

        # For future games, keep the schedule but mark scores as None
        future = sch[sch["home_score"].isna() | sch["away_score"].isna()].copy()
        future["home_score"] = None
        future["away_score"] = None

        # Only include regular season games for future predictions
        future = future[future["game_type"] == "REG"].reset_index(drop=True)

        logging.info(
            "Loaded %d completed games + %d future games", len(completed), len(future)
        )
        return pd.concat([completed, future], ignore_index=True)
    else:
        # Keep only completed games (original behavior)
        sch = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        logging.info("Schedules loaded: %d completed games", len(sch))
        return sch


# -----------------------------
# Feature engineering (leak-free)
# -----------------------------


def _team_game_long(sch: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-game schedule to *per-team per-game* long format to compute priors.
    Handles both completed games (with scores) and future games (without scores).
    """
    # Home perspective
    home = sch.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "points_for",
            "away_score": "points_against",
        }
    ).copy()
    home["is_home"] = 1

    # Away perspective
    away = sch.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "points_for",
            "home_score": "points_against",
        }
    ).copy()
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)

    # Only compute win for completed games
    completed_mask = long["points_for"].notna() & long["points_against"].notna()
    long["win"] = np.where(
        completed_mask,
        (long["points_for"] > long["points_against"]).astype(float),
        np.nan,
    )

    long["time_key"] = make_time_key(long)

    # Sorted so that groupby() yields strictly prior games
    return long.sort_values(["team", "time_key", "game_id"]).reset_index(drop=True)


<<<<<<< HEAD
<<<<<<< HEAD
def _rolling_prior_stats(
    long: pd.DataFrame, window: int = 3, advanced_cols: Optional[List[str]] = None
) -> pd.DataFrame:
=======
def _rolling_prior_stats(long: pd.DataFrame, window: int = 3, advanced_cols: Optional[List[str]] = None) -> pd.DataFrame:
>>>>>>> main
=======
def _rolling_prior_stats(long: pd.DataFrame, window: int = 3, advanced_cols: Optional[List[str]] = None) -> pd.DataFrame:
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    """
    Compute prior rolling means and win% per team with strict leakage protection.
    Only uses completed games to build priors for future game prediction.
    """
    grp = long.groupby("team", group_keys=False)

    def safe_rolling_mean(s):
        """Rolling mean that ignores NaN values (future games)"""
        # Shift by 1 to exclude current game, then apply rolling mean only to non-NaN values
        shifted = s.shift(1)  # Prior games only
        return shifted.rolling(window=window, min_periods=1).mean()

    # Use safe rolling that only considers completed games for priors
    long[f"prior_pf_avg_{window}"] = grp["points_for"].apply(safe_rolling_mean)
    long[f"prior_pa_avg_{window}"] = grp["points_against"].apply(safe_rolling_mean)
    long[f"prior_win_pct_{window}"] = grp["win"].apply(safe_rolling_mean)

    if advanced_cols:
        for col in advanced_cols:
            if col not in long.columns:
                continue
            long[f"prior_{col}_{window}"] = grp[col].apply(safe_rolling_mean)
<<<<<<< HEAD
<<<<<<< HEAD

=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    
>>>>>>> main
    return long


def add_features(
    sch: pd.DataFrame,
    windows: Tuple[int, ...] = (3, 5),
    advanced_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build one-row-per-game features by:
      1) creating per-team rows,
      2) computing prior rolling stats,
      3) re-pivoting to wide with home_/away_ prefixes.
    """
    long = _team_game_long(sch)

    advanced_cols: List[str] = []
    if advanced_metrics is not None and not advanced_metrics.empty:
        advanced_cols = [
<<<<<<< HEAD
<<<<<<< HEAD
            c
            for c in advanced_metrics.columns
            if c not in {"season", "week", "game_id", "team"}
        ]
        long = long.merge(
            advanced_metrics, on=["season", "week", "game_id", "team"], how="left"
        )
=======
            c for c in advanced_metrics.columns if c not in {"season", "week", "game_id", "team"}
        ]
        long = long.merge(advanced_metrics, on=["season", "week", "game_id", "team"], how="left")
>>>>>>> main
=======
            c for c in advanced_metrics.columns if c not in {"season", "week", "game_id", "team"}
        ]
        long = long.merge(advanced_metrics, on=["season", "week", "game_id", "team"], how="left")
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

    for w in windows:
        long = _rolling_prior_stats(long, window=w, advanced_cols=advanced_cols)

    if advanced_cols:
        long = long.drop(columns=advanced_cols, errors="ignore")

    base_cols = [
        "season",
        "week",
        "game_id",
        "game_date",
        "team",
        "opponent",
        "points_for",
        "points_against",
        "win",
    ]
    prior_cols = [c for c in long.columns if c.startswith("prior_")]
    carry = base_cols + prior_cols

    home_side = long[long["is_home"] == 1][carry].add_prefix("home_")
    away_side = long[long["is_home"] == 0][carry].add_prefix("away_")

    # Merge back to one row per game (home + away)
    wide = home_side.merge(
        away_side, left_on="home_game_id", right_on="away_game_id", how="inner"
    )
    wide = wide.rename(columns={"home_game_id": "game_id"}).drop(
        columns=["away_game_id"]
    )

    # Convenience fields at game-level (handle NaN scores for future games)
    wide["point_diff"] = np.where(
        wide["home_points_for"].notna() & wide["away_points_for"].notna(),
        wide["home_points_for"] - wide["away_points_for"],
        np.nan,
    )
    wide["winner"] = np.where(
        wide["point_diff"].notna() & (wide["point_diff"] > 0),
        wide["home_team"],
        np.where(
            wide["point_diff"].notna() & (wide["point_diff"] < 0),
            wide["away_team"],
            np.where(
                wide["point_diff"].notna() & (wide["point_diff"] == 0), "TIE", "TBD"
            ),
        ),
    )

    # Chronological sort across seasons/weeks
    wide["season"] = wide["home_season"].astype(int)
    wide["week"] = wide["home_week"].astype(int)
    wide["time_key"] = make_time_key(wide)
    wide = wide.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    # Add opponent-relative (differential) features: home_minus_away_*
    prior_pairs = [c for c in wide.columns if c.startswith("home_prior_")]
    for home_col in prior_pairs:
        suffix = home_col[len("home_prior_") :]
        away_col = f"away_prior_{suffix}"
        if away_col in wide.columns:
            wide[f"home_minus_away_{suffix}"] = wide[home_col] - wide[away_col]

    # Column ordering: identifiers + outcomes, then priors, then differentials
    ordered_cols = [
        "season",
        "week",
        "game_id",
        "home_game_date",
        "home_team",
        "away_team",
        "home_points_for",
        "away_points_for",
        "point_diff",
        "winner",
    ]

    # Add all prior and differential columns to the ordered list
    prior_feature_cols = [
        c for c in wide.columns if c.startswith(("home_prior_", "away_prior_"))
    ]
    diff_feature_cols = [c for c in wide.columns if c.startswith("home_minus_away_")]
    final_cols = ordered_cols + prior_feature_cols + diff_feature_cols

<<<<<<< HEAD
<<<<<<< HEAD
    schedule_extras = sch.drop_duplicates("game_id")[
        [
            "game_id",
            "home_moneyline",
            "away_moneyline",
            "spread_line",
            "total_line",
            "home_rest",
            "away_rest",
        ]
    ]
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    schedule_extras = sch.drop_duplicates("game_id")[[
        "game_id",
        "home_moneyline",
        "away_moneyline",
        "spread_line",
        "total_line",
        "home_rest",
        "away_rest",
    ]]
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    wide = wide.merge(schedule_extras, on="game_id", how="left")

    wide["home_moneyline"] = pd.to_numeric(wide["home_moneyline"], errors="coerce")
    wide["away_moneyline"] = pd.to_numeric(wide["away_moneyline"], errors="coerce")
    wide["home_moneyline_prob"] = _moneyline_to_prob(wide["home_moneyline"])
    wide["away_moneyline_prob"] = _moneyline_to_prob(wide["away_moneyline"])
<<<<<<< HEAD
<<<<<<< HEAD
    wide["moneyline_prob_diff"] = (
        wide["home_moneyline_prob"] - wide["away_moneyline_prob"]
    )
=======
    wide["moneyline_prob_diff"] = wide["home_moneyline_prob"] - wide["away_moneyline_prob"]
>>>>>>> main
=======
    wide["moneyline_prob_diff"] = wide["home_moneyline_prob"] - wide["away_moneyline_prob"]
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

    wide["spread_line"] = pd.to_numeric(wide["spread_line"], errors="coerce")
    wide["total_line"] = pd.to_numeric(wide["total_line"], errors="coerce")
    wide["home_rest"] = pd.to_numeric(wide["home_rest"], errors="coerce")
    wide["away_rest"] = pd.to_numeric(wide["away_rest"], errors="coerce")
    wide["rest_diff"] = wide["home_rest"] - wide["away_rest"]

<<<<<<< HEAD
<<<<<<< HEAD
    final_cols.extend(
        [
            "home_moneyline_prob",
            "away_moneyline_prob",
            "moneyline_prob_diff",
            "spread_line",
            "total_line",
            "home_rest",
            "away_rest",
            "rest_diff",
        ]
    )

    # Return the properly ordered DataFrame
    return wide[final_cols]
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    final_cols.extend([
        "home_moneyline_prob",
        "away_moneyline_prob",
        "moneyline_prob_diff",
        "spread_line",
        "total_line",
        "home_rest",
        "away_rest",
        "rest_diff",
    ])
    
    # Return the properly ordered DataFrame
    return wide[final_cols]


def _merge_team_week_stats(
    game_df: pd.DataFrame,
    team_week_stats: pd.DataFrame,
    prefix: str
) -> pd.DataFrame:
    """
    Merge team-week level stats into game-level dataframe for both home and away teams.
<<<<<<< HEAD
=======
    
    Args:
        game_df: Game-level dataframe with season, week, home_team, away_team
        team_week_stats: Team-week stats with season, week, team, [stat_columns]
        prefix: Prefix to add to merged columns (e.g. 'player' or 'team')
    
    Returns:
        game_df with additional home_{prefix}_* and away_{prefix}_* columns
    """
    stat_cols = [c for c in team_week_stats.columns if c not in {"season", "week", "team"}]
    
    # Merge home team stats
    home_stats = team_week_stats.copy()
    home_stats.columns = ["season", "week", "home_team"] + [f"home_{prefix}_{c}" for c in stat_cols]
    game_df = game_df.merge(home_stats, on=["season", "week", "home_team"], how="left")
    
    # Merge away team stats
    away_stats = team_week_stats.copy()
    away_stats.columns = ["season", "week", "away_team"] + [f"away_{prefix}_{c}" for c in stat_cols]
    game_df = game_df.merge(away_stats, on=["season", "week", "away_team"], how="left")
    
    return game_df
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    
    Args:
        game_df: Game-level dataframe with season, week, home_team, away_team
        team_week_stats: Team-week stats with season, week, team, [stat_columns]
        prefix: Prefix to add to merged columns (e.g. 'player' or 'team')
    
    Returns:
        game_df with additional home_{prefix}_* and away_{prefix}_* columns
    """
    stat_cols = [c for c in team_week_stats.columns if c not in {"season", "week", "team"}]
    
    # Merge home team stats
    home_stats = team_week_stats.copy()
    home_stats.columns = ["season", "week", "home_team"] + [f"home_{prefix}_{c}" for c in stat_cols]
    game_df = game_df.merge(home_stats, on=["season", "week", "home_team"], how="left")
    
    # Merge away team stats
    away_stats = team_week_stats.copy()
    away_stats.columns = ["season", "week", "away_team"] + [f"away_{prefix}_{c}" for c in stat_cols]
    game_df = game_df.merge(away_stats, on=["season", "week", "away_team"], how="left")
    
    return game_df
    
>>>>>>> main


def _merge_team_week_stats(
    game_df: pd.DataFrame, team_week_stats: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    """
    Merge team-week level stats into game-level dataframe for both home and away teams.

    Args:
        game_df: Game-level dataframe with season, week, home_team, away_team
        team_week_stats: Team-week stats with season, week, team, [stat_columns]
        prefix: Prefix to add to merged columns (e.g. 'player' or 'team')

    Returns:
        game_df with additional home_{prefix}_* and away_{prefix}_* columns
    """
    stat_cols = [
        c for c in team_week_stats.columns if c not in {"season", "week", "team"}
    ]

    # Merge home team stats
    home_stats = team_week_stats.copy()
    home_stats.columns = ["season", "week", "home_team"] + [
        f"home_{prefix}_{c}" for c in stat_cols
    ]
    game_df = game_df.merge(home_stats, on=["season", "week", "home_team"], how="left")

    # Merge away team stats
    away_stats = team_week_stats.copy()
    away_stats.columns = ["season", "week", "away_team"] + [
        f"away_{prefix}_{c}" for c in stat_cols
    ]
    game_df = game_df.merge(away_stats, on=["season", "week", "away_team"], how="left")

    return game_df


def build_regression_pipeline(
    numeric_features: List[str], categorical_features: List[str], alpha: float = 1.0
) -> Pipeline:
    """
    Returns a fit-ready sklearn Pipeline:
      - Numeric: median impute -> scale
      - Categorical: most-frequent impute -> one-hot
      - Estimator: Ridge (L2-regularized linear regression)

    Why StandardScaler(with_mean=False)?
      OneHotEncoder produces a sparse matrix; centering would densify it.
    """

    numeric_steps = Pipeline(
        [
            ("num_impute", SimpleImputer(strategy="median")),
            ("num_scale", StandardScaler(with_mean=False)),
        ]
    )

    categorical_steps = Pipeline(
        [
            ("cat_impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("num", numeric_steps, numeric_features),
            ("cat", categorical_steps, categorical_features),
        ]
    )

    model = Ridge(alpha=alpha)  # Deterministic given inputs; no random_state

    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    return pipeline


def ts_split_by_season_week(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    train_end: Tuple[int, int],  # (season, week), inclusive
):
    """
    Chronological split that prevents leakage.
    Train ≤ train_end, Val (train_end, val_end], Test > val_end.
    """
    data = df.copy()

    # Validate required columns early to fail fast
    required_cols = {"season", "week", *features, target}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Always sort time so cuts behave as expected
    data["time_key"] = make_time_key(data)
    data = data.sort_values(["time_key"]).reset_index(drop=True)

    # Boolean masks read like English - create proper train mask
    train_end_season, train_end_week = train_end
    is_train = (data["season"] < train_end_season) | (
        (data["season"] == train_end_season) & (data["week"] <= train_end_week)
    )

    # Split sets
    train_df = data.loc[is_train]

    # Final matrices
    X_train, y_train = train_df[features], train_df[target]

    # Clean up helper column before returning
    return (X_train, y_train), df, data


# -----------------------------
# Orchestration (CLI)
# -----------------------------


def get_current_nfl_week() -> tuple[int, int]:
    """
    Determine current NFL season and week based on current date and available data.

    Returns
    -------
    Tuple[int, int]
        Most recently completed (season, week) if historical data is available;
        otherwise defaults to Week 1 of the inferred season.
    """
    from datetime import datetime

    current_date = datetime.now()
    current_season = current_date.year

    # NFL season spans Sept-Feb, adjust if in early months
    if current_date.month <= 7:
        current_season -= 1

    try:
        import pandas as pd  # deferred import to keep CLI snappy
    except ImportError:
        pd = None

    if pd is not None:
        candidates = [
            Path("backend/data") / OUTPUT_DATASET_NAME,
            (Path("../backend/data").resolve() / OUTPUT_DATASET_NAME),
            Path(OUTPUT_DATASET_NAME),
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    df = pd.read_csv(candidate)
                except Exception:
                    continue
                if not df.empty:
                    latest_row = df.iloc[-1]
                    return int(latest_row["season"]), int(latest_row["week"])
                break

    # Default to Week 1 if no data available
    return current_season, 1

<<<<<<< HEAD
<<<<<<< HEAD

# Removed duplicate import pandas as pd here (was at line 738)

# ...existing code...

# ========= CONFIG (edit this if your column names differ) =========
HAS_winner_BOOL = True  # set to False if you only have scores
TIME_COLS_IN_ORDER = (
    ['season', 'week']  # auto-detect; or set to e.g. ["season","week"] or ["game_date"]
)

=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
import pandas as pd
import numpy as np

# ========= CONFIG (edit this if your column names differ) =========
HAS_winner_BOOL = True   # set to False if you only have scores
TIME_COLS_IN_ORDER = None  # auto-detect; or set to e.g. ["season","week"] or ["game_date"]
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

# ========= Helpers =========
def ensure_actual_winner(df):
    df = df.copy()

    if "home_win" in df.columns:
        win_series = pd.Series(df["home_win"], index=df.index, dtype="boolean")
    else:
        if not HAS_winner_BOOL:
            if {"home_points", "away_points"}.issubset(df.columns):
                win_series = pd.Series(
<<<<<<< HEAD
<<<<<<< HEAD
                    df["home_points"] > df["away_points"],
                    index=df.index,
                    dtype="boolean",
                )
            else:
                raise ValueError(
                    "Need either 'winner' bool or score columns 'home_points'/'away_points'."
                )
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
                    df["home_points"] > df["away_points"], index=df.index, dtype="boolean"
                )
            else:
                raise ValueError("Need either 'winner' bool or score columns 'home_points'/'away_points'.")
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
        else:
            winner_col = df["winner"]
            if pd.api.types.is_bool_dtype(winner_col.dtype):
                win_series = pd.Series(winner_col, index=df.index, dtype="boolean")
            else:
                win_series = pd.Series(pd.NA, index=df.index, dtype="boolean")
                win_series.loc[winner_col == df["home_team"]] = True
                win_series.loc[winner_col == df["away_team"]] = False

    df["home_win"] = win_series

    actual = np.where(win_series.fillna(False), df["home_team"], df["away_team"])
    df["actual_winner"] = pd.Series(actual, index=df.index)
    df.loc[win_series.isna(), "actual_winner"] = pd.NA
    return df

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
def detect_time_cols(df):
    if TIME_COLS_IN_ORDER is not None:
        return TIME_COLS_IN_ORDER
    if "game_date" in df.columns:
        return ["game_date"]
<<<<<<< HEAD
<<<<<<< HEAD
    if {"season", "week"}.issubset(df.columns):
        return ["season", "week"]
    # fallback: use current order; not ideal but keeps going
    return []


=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    if {"season","week"}.issubset(df.columns):
        return ["season","week"]
    # fallback: use current order; not ideal but keeps going
    return []

<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
def make_long_edges(df):
    """
    Produce a 'long' dataframe with two rows per game:
      - (team = home, opponent = away, team_won = winner)
      - (team = away, opponent = home, team_won = not winner)
    Includes 'gid' (row index of original df) to merge features back later.
    """
<<<<<<< HEAD
<<<<<<< HEAD
    base = df.reset_index(drop=False).rename(columns={"index": "gid"}).copy()
=======
    base = df.reset_index(drop=False).rename(columns={"index":"gid"}).copy()
>>>>>>> main
=======
    base = df.reset_index(drop=False).rename(columns={"index":"gid"}).copy()
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

    if "home_win" in base.columns:
        win_series = pd.Series(base["home_win"], index=base.index, dtype="boolean")
    else:
        winner_col = base["winner"]
        if pd.api.types.is_bool_dtype(winner_col.dtype):
            win_series = pd.Series(winner_col, index=base.index, dtype="boolean")
        else:
            win_series = pd.Series(pd.NA, index=base.index, dtype="boolean")
            win_series.loc[winner_col == base["home_team"]] = True
            win_series.loc[winner_col == base["away_team"]] = False

<<<<<<< HEAD
<<<<<<< HEAD
    home_rows = base[["gid", "home_team", "away_team"]].copy()
    home_rows = home_rows.rename(columns={"home_team": "team", "away_team": "opponent"})
    home_rows["team_won"] = win_series.reindex(home_rows.index)
    home_rows = home_rows[["gid", "team", "opponent", "team_won"]]

    away_rows = base[["gid", "home_team", "away_team"]].copy()
    away_rows = away_rows.rename(columns={"away_team": "team", "home_team": "opponent"})
    away_rows["team_won"] = (~win_series).reindex(away_rows.index)
    away_rows = away_rows[["gid", "team", "opponent", "team_won"]]
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    home_rows = base[["gid","home_team","away_team"]].copy()
    home_rows = home_rows.rename(columns={"home_team":"team","away_team":"opponent"})
    home_rows["team_won"] = win_series.reindex(home_rows.index)
    home_rows = home_rows[["gid","team","opponent","team_won"]]

    away_rows = base[["gid","home_team","away_team"]].copy()
    away_rows = away_rows.rename(columns={"away_team":"team","home_team":"opponent"})
    away_rows["team_won"] = (~win_series).reindex(away_rows.index)
    away_rows = away_rows[["gid","team","opponent","team_won"]]
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

    long = pd.concat([home_rows, away_rows], ignore_index=True)

    # Bring time columns to long for leak-free sorting
    time_cols = detect_time_cols(df)
    if time_cols:
        long = long.merge(base[["gid"] + time_cols], on="gid", how="left")
    return long

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
def pairwise_table(long):
    """
    Static, all-time pairwise records (order matters: (team, opponent)).
    dominance = wins - losses = 2*wins - games
    """
<<<<<<< HEAD
<<<<<<< HEAD
    g = long.groupby(["team", "opponent"], as_index=False, observed=True)
    agg = g.agg(games=("team", "size"), wins=("team_won", "sum"))
    agg["losses"] = agg["games"] - agg["wins"]
    agg["win_pct"] = np.where(agg["games"] > 0, agg["wins"] / agg["games"], np.nan)
    agg["dominance"] = agg["wins"] - agg["losses"]  # e.g., 8 vs 3 -> dominance = +5
    # If you want your “+8 / -3” flavor, also expose raw wins per side:
    agg["signed_wins_style"] = agg[
        "wins"
    ]  # for team’s +8 (and opponent’s +3 on their row)
    return agg


=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    g = long.groupby(["team","opponent"], as_index=False, observed=True)
    agg = g.agg(games=("team","size"),
                wins=("team_won","sum"))
    agg["losses"] = agg["games"] - agg["wins"]
    agg["win_pct"] = np.where(agg["games"] > 0, agg["wins"]/agg["games"], np.nan)
    agg["dominance"] = agg["wins"] - agg["losses"]        # e.g., 8 vs 3 -> dominance = +5
    # If you want your “+8 / -3” flavor, also expose raw wins per side:
    agg["signed_wins_style"] = agg["wins"]                # for team’s +8 (and opponent’s +3 on their row)
    return agg

<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
def leak_free_pregame_features(long):
    """
    For each (team, opponent), compute *prior-to-game* counts:
      prior_games, prior_wins, prior_losses, prior_dominance
    Returns a dataframe with one row per (gid, team, opponent).
    """
<<<<<<< HEAD
<<<<<<< HEAD
    time_cols = [c for c in long.columns if c in ("game_date", "season", "week")]
    sort_cols = time_cols + ["gid"] if time_cols else ["gid"]
    long = long.sort_values(sort_cols).copy()

    grp = long.groupby(["team", "opponent"], group_keys=False, observed=True)
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    time_cols = [c for c in long.columns if c in ("game_date","season","week")]
    sort_cols = time_cols + ["gid"] if time_cols else ["gid"]
    long = long.sort_values(sort_cols).copy()

    grp = long.groupby(["team","opponent"], group_keys=False, observed=True)
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

    # Prior games is simply the index within the group before the current row:
    long["prior_games"] = grp.cumcount()

    # Prior wins: cumulative sum of team_won excluding current row
    long["prior_wins"] = grp["team_won"].cumsum() - long["team_won"]

    long["prior_losses"] = long["prior_games"] - long["prior_wins"]
<<<<<<< HEAD
<<<<<<< HEAD
    long["prior_dom"] = (
        long["prior_wins"] - long["prior_losses"]
    )  # wins - losses so far
    long["prior_win_pct"] = np.where(
        long["prior_games"] > 0, long["prior_wins"] / long["prior_games"], np.nan
    )

    keep_cols = [
        "gid",
        "team",
        "opponent",
        "prior_games",
        "prior_wins",
        "prior_losses",
        "prior_dom",
        "prior_win_pct",
    ]
    return long[keep_cols]


=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    long["prior_dom"] = long["prior_wins"] - long["prior_losses"]          # wins - losses so far
    long["prior_win_pct"] = np.where(long["prior_games"]>0,
                                     long["prior_wins"]/long["prior_games"], np.nan)

    keep_cols = ["gid","team","opponent","prior_games","prior_wins","prior_losses","prior_dom","prior_win_pct"]
    return long[keep_cols]

<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
def attach_pregame_to_wide(df, pre):
    """
    Attach pre-game features back to the original wide df:
      home_vs_away_* and away_vs_home_*.
    """
<<<<<<< HEAD
<<<<<<< HEAD
    df = df.reset_index(drop=False).rename(columns={"index": "gid"}).copy()

    # Home perspective: (team=home_team, opponent=away_team)
    pre_home = pre.rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "prior_games": "home_vs_away_prior_games",
            "prior_wins": "home_vs_away_prior_wins",
            "prior_losses": "home_vs_away_prior_losses",
            "prior_dom": "home_vs_away_prior_dom",
            "prior_win_pct": "home_vs_away_prior_win_pct",
        }
    )

    # Away perspective: (team=away_team, opponent=home_team)
    pre_away = pre.rename(
        columns={
            "team": "away_team",
            "opponent": "home_team",
            "prior_games": "away_vs_home_prior_games",
            "prior_wins": "away_vs_home_prior_wins",
            "prior_losses": "away_vs_home_prior_losses",
            "prior_dom": "away_vs_home_prior_dom",
            "prior_win_pct": "away_vs_home_prior_win_pct",
        }
    )

    out = df.merge(
        pre_home[
            [
                "gid",
                "home_team",
                "away_team",
                "home_vs_away_prior_games",
                "home_vs_away_prior_wins",
                "home_vs_away_prior_losses",
                "home_vs_away_prior_dom",
                "home_vs_away_prior_win_pct",
            ]
        ],
        on=["gid", "home_team", "away_team"],
        how="left",
    )

    out = out.merge(
        pre_away[
            [
                "gid",
                "home_team",
                "away_team",
                "away_vs_home_prior_games",
                "away_vs_home_prior_wins",
                "away_vs_home_prior_losses",
                "away_vs_home_prior_dom",
                "away_vs_home_prior_win_pct",
            ]
        ],
        on=["gid", "home_team", "away_team"],
        how="left",
    )

    return out


=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    df = df.reset_index(drop=False).rename(columns={"index":"gid"}).copy()

    # Home perspective: (team=home_team, opponent=away_team)
    pre_home = pre.rename(columns={
        "team":"home_team", "opponent":"away_team",
        "prior_games":"home_vs_away_prior_games",
        "prior_wins":"home_vs_away_prior_wins",
        "prior_losses":"home_vs_away_prior_losses",
        "prior_dom":"home_vs_away_prior_dom",
        "prior_win_pct":"home_vs_away_prior_win_pct"
    })

    # Away perspective: (team=away_team, opponent=home_team)
    pre_away = pre.rename(columns={
        "team":"away_team", "opponent":"home_team",
        "prior_games":"away_vs_home_prior_games",
        "prior_wins":"away_vs_home_prior_wins",
        "prior_losses":"away_vs_home_prior_losses",
        "prior_dom":"away_vs_home_prior_dom",
        "prior_win_pct":"away_vs_home_prior_win_pct"
    })

    out = df.merge(pre_home[["gid","home_team","away_team",
                             "home_vs_away_prior_games","home_vs_away_prior_wins",
                             "home_vs_away_prior_losses","home_vs_away_prior_dom",
                             "home_vs_away_prior_win_pct"]],
                   on=["gid","home_team","away_team"], how="left")

    out = out.merge(pre_away[["gid","home_team","away_team",
                               "away_vs_home_prior_games","away_vs_home_prior_wins",
                               "away_vs_home_prior_losses","away_vs_home_prior_dom",
                               "away_vs_home_prior_win_pct"]],
                    on=["gid","home_team","away_team"], how="left")

    return out

<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
def team_level_prediction_accuracy(df):
    """
    Accuracy of your classifier *when a given team plays*.
    Expects df['prob_winner'] in [0,1] and df['winner'] bool.
    """
    if "prob_winner" not in df.columns:
        raise ValueError("Need a 'prob_winner' column for prediction accuracy.")

    df = df.copy()
<<<<<<< HEAD
<<<<<<< HEAD
    df["pred_winner"] = np.where(
        df["prob_winner"] >= 0.5, df["home_team"], df["away_team"]
    )
=======
    df["pred_winner"] = np.where(df["prob_winner"] >= 0.5, df["home_team"], df["away_team"])
>>>>>>> main
=======
    df["pred_winner"] = np.where(df["prob_winner"] >= 0.5, df["home_team"], df["away_team"])
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    df["actual_winner"] = np.where(df["winner"], df["home_team"], df["away_team"])
    df["pred_correct"] = (df["pred_winner"] == df["actual_winner"]).astype(int)

    # Give the same correctness stamp to both participants in that game:
<<<<<<< HEAD
<<<<<<< HEAD
    home_part = df[["home_team", "pred_correct"]].rename(columns={"home_team": "team"})
    away_part = df[["away_team", "pred_correct"]].rename(columns={"away_team": "team"})
=======
    home_part = df[["home_team","pred_correct"]].rename(columns={"home_team":"team"})
    away_part = df[["away_team","pred_correct"]].rename(columns={"away_team":"team"})
>>>>>>> main
=======
    home_part = df[["home_team","pred_correct"]].rename(columns={"home_team":"team"})
    away_part = df[["away_team","pred_correct"]].rename(columns={"away_team":"team"})
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    team_games = pd.concat([home_part, away_part], ignore_index=True)

    acc = team_games.groupby("team")["pred_correct"].mean().sort_values(ascending=False)
    return acc

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
# ========= Main flow =========
def build_dominance_features(dff):
    df = ensure_actual_winner(dff)

    # Long form (two rows per game)
    long = make_long_edges(df)

    # Static all-time pairwise table (order matters: team vs opponent)
    pair_table = pairwise_table(long)

    # Leak-free pregame features
    pre = leak_free_pregame_features(long)
    df_with_pregame = attach_pregame_to_wide(df, pre)

    # Optional: team-level prediction accuracy
<<<<<<< HEAD
<<<<<<< HEAD
    # team_acc = team_level_prediction_accuracy(df_with_pregame) if "prob_winner" in df_with_pregame.columns else None

    # Optional: dominance matrix (index=team, columns=opponent) for quick lookup
    dom_matrix = (
        pair_table.pivot(index="team", columns="opponent", values="dominance")
        .fillna(0)
        .astype(int)
    )

    return {
        "pair_table": pair_table.sort_values(["team", "opponent"]).reset_index(
            drop=True
        ),
        "dominance_matrix": dom_matrix,
        "df_with_pregame": df_with_pregame,
        # "team_prediction_accuracy": team_acc
    }


=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
   # team_acc = team_level_prediction_accuracy(df_with_pregame) if "prob_winner" in df_with_pregame.columns else None

    # Optional: dominance matrix (index=team, columns=opponent) for quick lookup
    dom_matrix = pair_table.pivot(index="team", columns="opponent", values="dominance").fillna(0).astype(int)

    return {
        "pair_table": pair_table.sort_values(["team","opponent"]).reset_index(drop=True),
        "dominance_matrix": dom_matrix,
        "df_with_pregame": df_with_pregame,
       # "team_prediction_accuracy": team_acc
    }

<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
# ========= Example usage =========


# # Pre-game feature columns available per row:
# # home_vs_away_prior_dom, away_vs_home_prior_dom, etc.


def build_dataset(
    start: int,
    end: int,
    out_dir: Path,
    production_mode: bool = True,
    include_future: bool = True,
    legacy_root_copy: bool = True,
):
    """
    Build production-ready NFL dataset with completed games + future scheduled games for prediction.

    Args:
        start: Starting season year
        end: Ending season year
        out_dir: Output directory path
        production_mode: If True, outputs only essential files and uses current NFL timing
        include_future: If True, includes future scheduled games for prediction
        legacy_root_copy: If True, also writes ``merged_game_features.csv`` to the
            repository root for legacy workflows (defaults to False).
    """
    seasons = list(range(int(start), int(end) + 1))

    if production_mode:
        current_season, current_week = get_current_nfl_week()
<<<<<<< HEAD
        logging.info(
            "Production dataset build - Current NFL state: %dW%d",
            current_season,
            current_week,
        )

    logging.info(
        "Building dataset for seasons=%s (include_future=%s)", seasons, include_future
    )

    # Stage 1: Load base schedules with betting lines
    schedules = load_schedules(seasons, include_future=include_future)

=======
        logging.info("Production dataset build - Current NFL state: %dW%d", current_season, current_week)
    
    logging.info("Building dataset for seasons=%s (include_future=%s)", seasons, include_future)
    
    # Stage 1: Load base schedules with betting lines
    schedules = load_schedules(seasons, include_future=include_future)
    
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    # Stage 2: Load advanced play-by-play metrics
    data_dir = Path(__file__).resolve().parent / "data"
    pbp_metrics = load_team_game_metrics(data_dir / "pbp_clean.csv")
    if not pbp_metrics.empty:
        pbp_metrics = pbp_metrics[pbp_metrics["season"].isin(seasons)]
<<<<<<< HEAD
<<<<<<< HEAD

    # Stage 3: Load player-level stats (QB, RB, WR aggregations)
    player_stats = load_player_game_stats(seasons)

    # Stage 4: Load team-level stats (official stats)
    team_stats = load_team_weekly_stats(seasons)

    # Stage 5: Engineer rolling features with PBP advanced metrics
    final_df = add_features(schedules, windows=(3, 5), advanced_metrics=pbp_metrics)

=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    
    # Stage 3: Load player-level stats (QB, RB, WR aggregations)
    player_stats = load_player_game_stats(seasons)
    
    # Stage 4: Load team-level stats (official stats)
    team_stats = load_team_weekly_stats(seasons)
    
    # Stage 5: Engineer rolling features with PBP advanced metrics
    final_df = add_features(schedules, windows=(3, 5), advanced_metrics=pbp_metrics)
    
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    # Stage 6: Merge player and team stats
    if not player_stats.empty:
        final_df = _merge_team_week_stats(final_df, player_stats, prefix="player")
        logging.info("Merged player stats: now %d columns", len(final_df.columns))
<<<<<<< HEAD
<<<<<<< HEAD

=======
    
>>>>>>> main
=======
    
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    if not team_stats.empty:
        final_df = _merge_team_week_stats(final_df, team_stats, prefix="teamstat")
        logging.info("Merged team stats: now %d columns", len(final_df.columns))

    # Robust data preparation for production
    prior_mask = final_df.filter(regex=r"^(home|away)_prior_").columns
    diff_mask = final_df.filter(regex=r"^home_minus_away_").columns

    # Use median imputation (more robust than mean)
    final_df[prior_mask] = final_df[prior_mask].fillna(final_df[prior_mask].median())
    final_df[diff_mask] = final_df[diff_mask].fillna(final_df[diff_mask].median())

    # Ensure data integrity - only remove rows with null values in critical feature columns
    # Keep future games (which have null scores but valid feature columns)
    critical_feature_cols = [
        c
        for c in final_df.columns
        if c.startswith(("home_prior_", "away_prior_", "home_minus_away_"))
    ]

    if include_future:
        # For production with future games, only drop rows missing critical features
        final_df = final_df.dropna(subset=critical_feature_cols).reset_index(drop=True)
        logging.info("Kept future games - dropped only rows missing feature data")
    else:
        # For training data, drop any rows with null values
        final_df = final_df.dropna().reset_index(drop=True)
        logging.info("Dropped all rows with any null values (training mode)")

    final_df = final_df.sort_values(by="home_game_date").reset_index(drop=True)
<<<<<<< HEAD

=======
    
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    # Create boolean home_win column before dominance analysis (nullable for ties/future games)
    home_win = pd.Series(pd.NA, index=final_df.index, dtype="boolean")
    home_win.loc[final_df["winner"] == final_df["home_team"]] = True
    home_win.loc[final_df["winner"] == final_df["away_team"]] = False
    final_df["home_win"] = home_win
<<<<<<< HEAD
<<<<<<< HEAD

=======
    
>>>>>>> main
=======
    
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    # run df through pipeline to ensure no errors
    for col in final_df.select_dtypes(include=["object"]).columns:
        final_df[col] = final_df[col].astype(dtype="category")

    numeric_features = final_df.select_dtypes(include=[np.number]).columns.tolist()

    categorical_features = ["home_team", "away_team"]

    dff = final_df.copy()
    # Include winner and home_win for dominance analysis
    required_cols = categorical_features + numeric_features + ["winner", "home_win"]
    # Filter to only include columns that exist
    required_cols = [c for c in required_cols if c in dff.columns]
    dff = dff[required_cols]
    result = build_dominance_features(dff)
    logging.info("Pairwise dominance table:\n%s", result["pair_table"].head(10))
    logging.info("Dominance matrix:\n%s", result["dominance_matrix"].head(10))
<<<<<<< HEAD
<<<<<<< HEAD
    logging.info(
        "DataFrame with pregame features:\n%s", result["df_with_pregame"].head(10)
    )
=======
    logging.info("DataFrame with pregame features:\n%s", result["df_with_pregame"].head(10))
>>>>>>> main
=======
    logging.info("DataFrame with pregame features:\n%s", result["df_with_pregame"].head(10))
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    team_acc = result.get("team_prediction_accuracy")
    if team_acc is not None:
        logging.info("Team prediction accuracy:\n%s", team_acc.head(10))
    result["df_with_pregame"].to_csv(out_dir / "df_with_pregame.csv", index=False)

    # Production output
    out_dir.mkdir(parents=True, exist_ok=True)
    main_output = out_dir / OUTPUT_DATASET_NAME
    final_df.to_csv(main_output, index=False)

    if legacy_root_copy:
        final_df.to_csv(OUTPUT_DATASET_NAME, index=False)
        logging.info("Legacy root-level copy created for compatibility across scripts.")

    logging.info("Production dataset ready: %s (%d games)", main_output, len(final_df))

    # Export team mapping for API consistency
    abbr_json_path = out_dir / "team_abbr_map.json"
    with open(abbr_json_path, "w") as f:
        json.dump(ABBR_FIX, f, indent=2)

    return main_output, final_df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for season range and output directory."""
    p = argparse.ArgumentParser(
        description="Build NFL game-level dataset (one row per game)."
    )
    p.add_argument("--start", type=int, default=2014, help="Start season (inclusive).")
    p.add_argument("--end", type=int, default=2025, help="End season (inclusive).")
<<<<<<< HEAD
    p.add_argument(
        "--out-dir", type=str, default="backend/data", help="Output directory."
    )
=======
    p.add_argument("--out-dir", type=str, default="backend/data", help="Output directory.")
<<<<<<< HEAD
>>>>>>> main
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    p.add_argument(
        "--legacy-root-copy",
        action="store_true",
        help=f"Also write {OUTPUT_DATASET_NAME} to the repository root for backwards compatibility.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for CLI usage with logging setup."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    setup_logger(out_dir)
    build_dataset(
        args.start,
        args.end,
        out_dir,
        legacy_root_copy=args.legacy_root_copy,
        production_mode=True,
        include_future=True,
    )
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5
    
>>>>>>> main


if __name__ == "__main__":
    main()
    
<<<<<<< HEAD

=======
>>>>>>> 9dcc198ee49ca6bd9f3bdbc57b1660740b2c15b5

# -----------------------------
# Suggested Enhancements
# -----------------------------
# 1) Persist & reuse a canonical TEAM_MAP shared with the API to avoid LA/LAR
#    drift; consider exporting it into metadata alongside the dataset build.
# 2) Add opponent-relative features (home_minus_away of priors) to reduce
#    collinearity and match many sports modeling baselines.
# 3) Provide a "strict" mode that drops rows with insufficient history instead
#    of imputing means, to allow unbiased validation when desired.
