#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_csv_datasets.py
=====================

Production-ready builder for NFL game-level datasets (one row per game).

**PURPOSE: DATASET BUILDING ONLY** - Model training is handled in enhanced_pipeline.py

Features:
  • Leak-free rolling features (strict shift(1) before rolling)
  • Team-game advanced metrics (EPA, success, explosive, turnover rates)
  • Pairwise dominance features (static table, leak-free pre-game priors, matrix values)
  • Market context (moneyline → implied probability, spread/total, rest diffs)
  • Optional team encodings (one-hot for home/away)
  • Optional calibration rows for downstream harnesses

Quick start
-----------
python build_csv_datasets.py --start 2016 --end 2025 --out-dir ./metrics/data --save-dominance-matrix --no-calibration-rows --dominance-log

Key CLI flags
-------------
--dominance-log               Optional text file with "Pairwise dominance table" section
--encode {onehot,none}        Team encoding mode (default: onehot)
--save-dominance-matrix       Also export dominance_matrix.csv
--no-calibration-rows         Do NOT append the 2 blank calibration rows

Outputs
-------
• CSV: {out_dir}/game_features_new.csv
• Log: {out_dir}/build_csv_datasets.log
• (Optional) dominance_matrix.csv
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
from typing import List, Dict, Tuple, Optional
import re
import numbers

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration and constants
# -----------------------------------------------------------------------------

ABBR_FIX: Dict[str, str] = {
    "LA": "LAR", "STL": "LAR",
    "SD": "LAC", "OAK": "LV",
    "WSH": "WAS",
}
OUTPUT_DATASET_NAME = "game_features.csv"

def make_time_key(df: pd.DataFrame) -> pd.Series:
    """Return sortable integer key YYYYWW from 'season' and 'week' (assumes ints)."""
    return (df["season"].astype(int) * 100) + df["week"].astype(int)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


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
    # Use ASCII-safe arrow to avoid Windows console encoding issues
    logging.info("Logger initialized -> %s", log_file)


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


# -----------------------------------------------------------------------------
# Backend selection (nflreadpy preferred, fallback to nfl_data_py)
# -----------------------------------------------------------------------------

NFL_BACKEND = "nfl_data_py"
nfl = None
_fallback_reason = None

def _note_backend(msg: str, level: int = logging.INFO) -> None:
    logging.log(level, msg)

try:
    import nflreadpy as _nfl
    try:
        _probe = _nfl.load_schedules(seasons=[2024])
        if hasattr(_probe, "to_pandas"):
            _ = _probe.head(1).to_pandas()
        NFL_BACKEND = "nflreadpy"
        nfl = _nfl
        _note_backend("Using backend 'nflreadpy'")
    except Exception as e:
        _fallback_reason = f"nflreadpy probe failed: {e}"
        import nfl_data_py as _nfl
        nfl = _nfl
        NFL_BACKEND = "nfl_data_py"
        _note_backend(f"Using fallback backend '{NFL_BACKEND}' — {_fallback_reason}", logging.WARNING)
except Exception as e:
    _fallback_reason = f"nflreadpy import failed: {e}"
    try:
        import nfl_data_py as _nfl
        nfl = _nfl
        NFL_BACKEND = "nfl_data_py"
        _note_backend(f"Using fallback backend '{NFL_BACKEND}' — {_fallback_reason}", logging.WARNING)
    except Exception as e2:
        nfl = None
        _note_backend(f"No NFL backend available: {e2}", logging.ERROR)



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
    positive = ml_numeric > 0
    probs.loc[negative] = (-ml_numeric.loc[negative]) / ((-ml_numeric.loc[negative]) + 100.0)
    probs.loc[positive] = 100.0 / (ml_numeric.loc[positive] + 100.0)
    return probs


# -----------------------------------------------------------------------------
# Loading sources
# -----------------------------------------------------------------------------

def load_team_game_metrics(pbp_path: Path) -> pd.DataFrame:
    """Compute team-level offensive and defensive metrics from play-by-play data."""
    if not pbp_path.exists():
        logging.warning("PBP cache not found at %s", pbp_path)
        return pd.DataFrame()

    try:
        logging.info("Loading PBP data from %s", pbp_path)
        
        # Load in chunks to avoid memory issues with dtype specification
        chunks = []
        chunk_size = 100000
        
        for chunk in pd.read_csv(pbp_path, chunksize=chunk_size, low_memory=False):
            # Filter immediately within each chunk
            chunk = chunk[chunk["posteam"].notna()].copy()
            
            # Only keep necessary columns for metrics (check which exist)
            required_cols = [
                "season", "week", "game_id", "posteam", "defteam",
                "epa", "success", "pass", "xpass", "pass_attempt", 
                "rush_attempt", "turnover", "explosive_play"
            ]
            # Keep only columns that exist
            existing_cols = [col for col in required_cols if col in chunk.columns]
            chunk = chunk[existing_cols]
            
            chunks.append(chunk)
        
        if not chunks:
            logging.warning("No valid PBP data found")
            return pd.DataFrame()
            
        pbp = pd.concat(chunks, ignore_index=True)
        logging.info("Loaded %d plays with columns: %s", len(pbp), list(pbp.columns))
        
    except Exception as exc:
        logging.error("Failed to load PBP from %s: %s", pbp_path, exc)
        return pd.DataFrame()

    # Offensive aggregation - build agg dict only for available columns
    off_group = ["season", "week", "game_id", "posteam"]
    off_agg_dict = {}
    
    if "epa" in pbp.columns:
        off_agg_dict["off_epa_per_play"] = ("epa", "mean")
    if "success" in pbp.columns:
        off_agg_dict["off_success_rate"] = ("success", "mean")
    if "pass" in pbp.columns:
        off_agg_dict["off_pass_rate"] = ("pass", "mean")
    if "xpass" in pbp.columns:
        off_agg_dict["off_expected_pass_rate"] = ("xpass", "mean")
    if "pass_attempt" in pbp.columns:
        off_agg_dict["off_pass_attempts"] = ("pass_attempt", "sum")
    if "rush_attempt" in pbp.columns:
        off_agg_dict["off_rush_attempts"] = ("rush_attempt", "sum")
    if "turnover" in pbp.columns:
        off_agg_dict["off_turnovers"] = ("turnover", "sum")
    if "explosive_play" in pbp.columns:
        off_agg_dict["off_explosive_rate"] = ("explosive_play", "mean")
    
    if not off_agg_dict:
        logging.warning("No valid offensive metrics columns found in PBP data")
        return pd.DataFrame()
    
    off_agg = pbp.groupby(off_group, as_index=False).agg(**off_agg_dict)

    # Defensive aggregation - build agg dict only for available columns
    def_group = ["season", "week", "game_id", "defteam"]
    def_agg_dict = {}
    
    if "epa" in pbp.columns:
        def_agg_dict["def_epa_allowed"] = ("epa", "mean")
    if "success" in pbp.columns:
        def_agg_dict["def_success_rate_allowed"] = ("success", "mean")
    if "explosive_play" in pbp.columns:
        def_agg_dict["def_explosive_rate_allowed"] = ("explosive_play", "mean")
    if "turnover" in pbp.columns:
        def_agg_dict["def_takeaways"] = ("turnover", "sum")
    if "pass_attempt" in pbp.columns:
        def_agg_dict["def_pass_attempts_faced"] = ("pass_attempt", "sum")
    if "rush_attempt" in pbp.columns:
        def_agg_dict["def_rush_attempts_faced"] = ("rush_attempt", "sum")
    
    if not def_agg_dict:
        logging.warning("No valid defensive metrics columns found in PBP data")
        return pd.DataFrame()
    
    def_agg = pbp.groupby(def_group, as_index=False).agg(**def_agg_dict)
    
    if "def_epa_allowed" in def_agg.columns:
        def_agg["def_epa_per_play"] = -def_agg["def_epa_allowed"]

    metrics = off_agg.rename(columns={"posteam": "team"}).merge(
        def_agg.rename(columns={"defteam": "team"}),
        on=["season", "week", "game_id", "team"], how="outer",
    )

    # Derived rates - only compute if we have the necessary columns
    if "off_pass_attempts" in metrics.columns and "off_rush_attempts" in metrics.columns:
        metrics["off_total_plays"] = metrics["off_pass_attempts"].fillna(0) + metrics["off_rush_attempts"].fillna(0)
        
        if "off_turnovers" in metrics.columns:
            metrics["off_turnover_rate"] = np.where(
                metrics["off_total_plays"] > 0,
                metrics["off_turnovers"].fillna(0) / metrics["off_total_plays"],
                np.nan,
            )
    
    if "def_takeaways" in metrics.columns and "def_pass_attempts_faced" in metrics.columns and "def_rush_attempts_faced" in metrics.columns:
        metrics["def_takeaway_rate"] = np.where(
            (metrics["def_pass_attempts_faced"].fillna(0) + metrics["def_rush_attempts_faced"].fillna(0)) > 0,
            metrics["def_takeaways"].fillna(0) / (metrics["def_pass_attempts_faced"].fillna(0) + metrics["def_rush_attempts_faced"].fillna(0)),
            np.nan,
        )

    # Drop intermediate columns that exist
    drop_cols = [
        "off_pass_attempts","off_rush_attempts","off_turnovers",
        "def_pass_attempts_faced","def_rush_attempts_faced",
        "def_takeaways","def_epa_allowed",
    ]
    metrics.drop(columns=[c for c in drop_cols if c in metrics.columns], inplace=True, errors="ignore")
    return metrics.fillna(np.nan)

    off_agg["off_third_down_total"] = (
        off_agg["off_third_down_conv"] + off_agg["off_third_down_fail"]
    )
    off_agg["off_third_down_pct"] = np.where(
        off_agg["off_third_down_total"] > 0,
        off_agg["off_third_down_conv"] / off_agg["off_third_down_total"],
        np.nan,
    )

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
        np.nan,
    )

    metrics = off_agg.rename(columns={"posteam": "team"}).merge(
        def_agg.rename(columns={"defteam": "team"}),
        on=["season", "week", "game_id", "team"],
        how="outer",
    )


    metrics["off_total_plays"] = metrics["off_pass_attempts"].fillna(0) + metrics[
        "off_rush_attempts"
    ].fillna(0)
    metrics["off_turnover_rate"] = np.where(
        metrics["off_total_plays"] > 0,
        metrics["off_turnovers"].fillna(0) / metrics["off_total_plays"],
        np.nan,
    )


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

    return metrics.fillna(np.nan)


def load_player_game_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Load weekly player-level stats from nflreadpy and aggregate to team-game level.
    Provides QB efficiency, RB production, WR targets, etc.
    """


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
        return pd.DataFrame(columns=["season", "week", "team"])


def load_team_weekly_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Load official team-level stats from nflreadpy at weekly granularity.
    Provides points scored/allowed, yards, turnovers, etc.
    """


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

    except Exception as exc:
        logging.warning("Failed to load team stats (%s); team features disabled", exc)
        return pd.DataFrame(columns=["season", "week", "team"])



def load_schedules(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
    """
    Load schedules + final scores for given seasons using nflreadpy.

    Args:
        seasons: List of seasons to load
        include_future: If True, includes scheduled games without scores for prediction

    Returns
    -------
    DataFrame with:
      ['season','week','game_id','game_date','home_team','away_team',
       'home_score','away_score', 'spread_line', 'total_line', 'away_rest', 'home_rest']
    """
    # Duplicate/broken implementation removed; see later load_schedules() definition.
    pass


def load_player_game_stats(seasons: List[int]) -> pd.DataFrame:
    """Load weekly player stats and aggregate to team-week level."""
    if nfl is None:
        logging.warning("No NFL backend available; player features disabled")
        return pd.DataFrame(columns=["season", "week", "team"])
    try:
        logging.info("Loading player stats via %s for seasons %s", NFL_BACKEND, seasons)
        player_stats = None
        if NFL_BACKEND == "nflreadpy":
            loader = getattr(nfl, "load_player_stats", None)
            if loader:
                player_stats = to_pandas_safe(loader(seasons=seasons, summary_level="week"))
        else:
            loader = getattr(nfl, "import_weekly_data", None)
            if loader:
                player_stats = loader(seasons)
        if player_stats is None or player_stats.empty:
            logging.warning("Player stats unavailable; skipping.")
            return pd.DataFrame(columns=["season", "week", "team"])

        team_col = "recent_team" if "recent_team" in player_stats.columns else ("team" if "team" in player_stats.columns else None)
        if not team_col:
            logging.warning("Cannot determine team column in player_stats; skipping")
            return pd.DataFrame(columns=["season", "week", "team"])

        # QB aggregation - check for required columns first
        qb = player_stats[player_stats.get("position") == "QB"].copy() if "position" in player_stats.columns else pd.DataFrame()
        
        qb_cols = {
            "passing_yards": "team_qb_pass_yards",
            "passing_tds": "team_qb_pass_tds",
            "interceptions": "team_qb_interceptions",
            "sacks": "team_qb_sacks",
            "completions": "team_qb_completions",
            "attempts": "team_qb_attempts",
        }
        
        # Build aggregation dict only for available columns
        qb_agg_dict = {}
        for src_col, tgt_col in qb_cols.items():
            if src_col in qb.columns:
                qb_agg_dict[tgt_col] = (src_col, "sum")
        
        if qb_agg_dict and not qb.empty:
            qb_agg = qb.groupby(["season", "week", team_col], as_index=False).agg(**qb_agg_dict)
            
            # Calculate completion percentage if both columns exist
            if "team_qb_completions" in qb_agg.columns and "team_qb_attempts" in qb_agg.columns:
                qb_agg["team_qb_completion_pct"] = np.where(
                    qb_agg["team_qb_attempts"] > 0,
                    qb_agg["team_qb_completions"] / qb_agg["team_qb_attempts"],
                    np.nan,
                )
        else:
            qb_agg = pd.DataFrame(columns=["season", "week", team_col])

        # RB aggregation
        rb = player_stats[player_stats["position"].isin(["RB"])].copy() if "position" in player_stats.columns else pd.DataFrame()
        
        rb_cols = {
            "rushing_yards": "team_rb_rush_yards",
            "rushing_tds": "team_rb_rush_tds",
            "receptions": "team_rb_receptions",
            "receiving_yards": "team_rb_receiving_yards",
        }
        
        rb_agg_dict = {tgt: (src, "sum") for src, tgt in rb_cols.items() if src in rb.columns}
        
        if rb_agg_dict and not rb.empty:
            rb_agg = rb.groupby(["season", "week", team_col], as_index=False).agg(**rb_agg_dict)
        else:
            rb_agg = pd.DataFrame(columns=["season", "week", team_col])

        # WR+TE aggregation
        pass_catchers = player_stats[player_stats["position"].isin(["WR", "TE"])].copy() if "position" in player_stats.columns else pd.DataFrame()
        
        wr_cols = {
            "targets": "team_wr_targets",
            "receptions": "team_wr_receptions",
            "receiving_yards": "team_wr_receiving_yards",
            "receiving_tds": "team_wr_receiving_tds",
        }
        
        wr_agg_dict = {tgt: (src, "sum") for src, tgt in wr_cols.items() if src in pass_catchers.columns}
        
        if wr_agg_dict and not pass_catchers.empty:
            wr_agg = pass_catchers.groupby(["season", "week", team_col], as_index=False).agg(**wr_agg_dict)
        else:
            wr_agg = pd.DataFrame(columns=["season", "week", team_col])

        # Merge to one table
        out = qb_agg
        if not rb_agg.empty:
            out = out.merge(rb_agg, on=["season", "week", team_col], how="outer")
        if not wr_agg.empty:
            out = out.merge(wr_agg, on=["season", "week", team_col], how="outer")
        
        if not out.empty:
            out = out.rename(columns={team_col: "team"}).fillna(0)
        else:
            out = pd.DataFrame(columns=["season", "week", "team"])
        
        return out

    except Exception as exc:
        logging.warning("Failed to load player stats (%s); player features disabled", exc)
        return pd.DataFrame(columns=["season", "week", "team"])


def load_team_weekly_stats(seasons: List[int]) -> pd.DataFrame:
    """Load team-level weekly stats when available (nflreadpy path)."""
    if nfl is not None and NFL_BACKEND == "nflreadpy":
        try:
            logging.info("Loading team weekly stats via nflreadpy for seasons %s", seasons)
            loader = getattr(nfl, "load_team_stats", None)
            if loader:
                team_stats = to_pandas_safe(loader(seasons=seasons, summary_level="week"))
                feature_cols = [
                    "season", "week", "team",
                    "points_scored", "points_allowed",
                    "total_yards", "total_yards_allowed",
                    "turnovers", "turnovers_forced",
                ]
                available = [c for c in feature_cols if c in team_stats.columns]
                return team_stats[available].fillna(0)
        except Exception as exc:
            logging.warning("Team stats load failed (%s); team features disabled", exc)
    return pd.DataFrame(columns=["season", "week", "team"])


def load_schedules(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
    """Load schedules + scores."""
    sch = None
    if nfl is not None:
        if NFL_BACKEND == "nflreadpy":
            loader = getattr(nfl, "load_schedules", None)
            if loader:
                sch = to_pandas_safe(loader(seasons=seasons))
        else:
            loader = getattr(nfl, "import_schedules", None)
            if loader:
                sch = loader(seasons)
    if sch is None or sch.empty:
        raise RuntimeError("Could not load schedules from any backend.")

    need = [
        "season","week","game_id","gameday","home_team","away_team","home_score","away_score","game_type",
        "away_moneyline","home_moneyline","spread_line","total_line","away_rest","home_rest",
    ]
    missing = [c for c in need if c not in sch.columns]
    if missing:
        raise RuntimeError(f"Missing schedule columns: {missing}")

    sch = _normalize_codes(sch, ["home_team","away_team"])
    sch["week"] = sch["week"].astype(int)
    sch = sch.rename(columns={"gameday": "game_date"})
    sch = sch[
        ["season","week","game_id","game_date","home_team","away_team","home_score","away_score",
         "game_type","away_moneyline","home_moneyline","spread_line","total_line","away_rest","home_rest"]
    ].copy()

    if include_future:
        completed = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        future = sch[sch["home_score"].isna() | sch["away_score"].isna()].copy()
        future = future[future["game_type"] == "REG"].reset_index(drop=True)
        if not future.empty:
            future[["home_score", "away_score"]] = np.nan
            shared_cols = [c for c in future.columns if c in completed.columns]
            future = future.reindex(columns=completed.columns, fill_value=pd.NA)
            for col in shared_cols:
                try:
                    future[col] = future[col].astype(completed[col].dtype)
                except (TypeError, ValueError):
                    continue
        logging.info("Loaded %d completed + %d future games", len(completed), len(future))
        # Only concatenate if we have future games to avoid FutureWarning
        if not future.empty:
            return pd.concat([completed, future], ignore_index=True)
        return completed

    return sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Core feature building
# -----------------------------------------------------------------------------


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



def _rolling_prior_stats(
    long: pd.DataFrame, window: int = 3, advanced_cols: Optional[List[str]] = None
) -> pd.DataFrame:
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

    # Remove duplicate block referencing undefined safe_roll
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
            c
            for c in advanced_metrics.columns
            if c not in {"season", "week", "game_id", "team"}
        ]
        long = long.merge(
            advanced_metrics, on=["season", "week", "game_id", "team"], how="left"
        )

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

    # Opponent-relative diffs
    for home_col in [c for c in wide.columns if c.startswith("home_prior_")]:
        suffix = home_col[len("home_prior_") :]
        away_col = f"away_prior_{suffix}"
        if away_col in wide.columns:
            wide[f"home_minus_away_{suffix}"] = wide[home_col] - wide[away_col]

    ordered_cols = [
        "season","week","game_id","home_game_date","home_team","away_team",
        "home_points_for","away_points_for","point_diff","winner",
    ]
    prior_feature_cols = [c for c in wide.columns if c.startswith(("home_prior_", "away_prior_"))]
    diff_feature_cols = [c for c in wide.columns if c.startswith("home_minus_away_")]

    # Market/rest context
    schedule_extras = sch.drop_duplicates("game_id")[
        ["game_id","home_moneyline","away_moneyline","spread_line","total_line","home_rest","away_rest"]
    ]
    wide = wide.merge(schedule_extras, on="game_id", how="left")
    for c in ["home_moneyline","away_moneyline","spread_line","total_line","home_rest","away_rest"]:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")

    wide["home_moneyline_prob"] = _moneyline_to_prob(wide["home_moneyline"])
    wide["away_moneyline_prob"] = _moneyline_to_prob(wide["away_moneyline"])
    wide["moneyline_prob_diff"] = wide["home_moneyline_prob"] - wide["away_moneyline_prob"]
    wide["rest_diff"] = wide["home_rest"] - wide["away_rest"]

    final_cols = ordered_cols + prior_feature_cols + diff_feature_cols + [
        "home_moneyline_prob","away_moneyline_prob","moneyline_prob_diff",
        "spread_line","total_line","home_rest","away_rest","rest_diff"
    ]
    return wide[final_cols]


def _merge_team_week_stats(game_df: pd.DataFrame, team_week_stats: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Merge team-week aggregates to both home_ and away_ namespaces."""
    if team_week_stats.empty:
        return game_df
    stat_cols = [c for c in team_week_stats.columns if c not in {"season", "week", "team"}]

    home_stats = team_week_stats.copy()
    home_stats.columns = ["season", "week", "home_team"] + [f"home_{prefix}_{c}" for c in stat_cols]
    out = game_df.merge(home_stats, on=["season", "week", "home_team"], how="left")

    away_stats = team_week_stats.copy()
    away_stats.columns = ["season", "week", "away_team"] + [f"away_{prefix}_{c}" for c in stat_cols]
    out = out.merge(away_stats, on=["season", "week", "away_team"], how="left")
    return out


# -----------------------------------------------------------------------------
# Outcomes + Pairwise dominance
# -----------------------------------------------------------------------------

def ensure_actual_winner(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to home_win (boolean) and winner_team (string)."""
    out = df.copy()
    if "home_win" in out.columns and pd.api.types.is_bool_dtype(out["home_win"]):
        hw = out["home_win"].astype("boolean")
        out["home_win"] = hw
        out["winner_team"] = pd.Series(pd.NA, index=out.index, dtype="string")
        out.loc[hw == True, "winner_team"] = out.loc[hw == True, "home_team"].astype("string")
        out.loc[hw == False, "winner_team"] = out.loc[hw == False, "away_team"].astype("string")
        return out

    if "winner" in out.columns and out["winner"].dtype == object:
        winner_str = out["winner"].astype("string")
        hw = pd.Series(pd.NA, index=out.index, dtype="boolean")
        hw.loc[winner_str == out["home_team"]] = True
        hw.loc[winner_str == out["away_team"]] = False
        out["home_win"] = hw
        out["winner_team"] = winner_str
        return out

    if {"home_points_for", "away_points_for"}.issubset(out.columns):
        pf = pd.to_numeric(out["home_points_for"], errors="coerce")
        pa = pd.to_numeric(out["away_points_for"], errors="coerce")
        hw = pd.Series(pd.NA, index=out.index, dtype="boolean")
        mask = pf.notna() & pa.notna()
        hw.loc[mask & (pf > pa)] = True
        hw.loc[mask & (pf < pa)] = False
        out["home_win"] = hw
        out["winner_team"] = pd.Series(pd.NA, index=out.index, dtype="string")
        out.loc[hw == True, "winner_team"] = out.loc[hw == True, "home_team"].astype("string")
        out.loc[hw == False, "winner_team"] = out.loc[hw == False, "away_team"].astype("string")
        out.loc[mask & (pf == pa), "winner_team"] = "TIE"
        return out

    out["home_win"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["winner_team"] = pd.Series(pd.NA, index=out.index, dtype="string")
    return out


def _detect_time_cols(df: pd.DataFrame) -> List[str]:
    """Pick chronological sort columns."""
    # Default to auto-detection; constant may be unset in some builds
    TIME_COLS_IN_ORDER: Optional[List[str]] = None
    if TIME_COLS_IN_ORDER:
        return TIME_COLS_IN_ORDER
    if "game_date" in df.columns:
        return ["game_date"]
    if {"season", "week"}.issubset(df.columns):
        return ["season", "week"]
    return []


def _to_pairwise_long_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (home + away) with team_won flag."""
    base = df.reset_index(drop=False).rename(columns={"index": "gid"}).copy()
    base = ensure_actual_winner(base)
    win_series = pd.Series(base["home_win"], index=base.index, dtype="boolean")

    home_rows = base[["gid", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    ).copy()
    home_rows["team_won"] = win_series.reindex(home_rows.index)
    home_rows = home_rows[["gid", "team", "opponent", "team_won"]]

    away_rows = base[["gid", "home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    ).copy()
    away_rows["team_won"] = (~win_series).reindex(away_rows.index)
    away_rows = away_rows[["gid", "team", "opponent", "team_won"]]

    long = pd.concat([home_rows, away_rows], ignore_index=True)
    time_cols = _detect_time_cols(df)
    if time_cols:
        long = long.merge(base[["gid"] + time_cols], on="gid", how="left")
    return long


def build_pairwise_record_table(long: pd.DataFrame) -> pd.DataFrame:
    """Static, all-time pairwise records (team, opponent)."""
    g = long.groupby(["team", "opponent"], as_index=False)
    agg = g.agg(games=("team", "size"), wins=("team_won", "sum"))
    agg["losses"] = agg["games"] - agg["wins"]
    agg["win_pct"] = np.where(agg["games"] > 0, agg["wins"] / agg["games"], np.nan)
    agg["dominance"] = agg["wins"] - agg["losses"]
    return agg


def leak_free_pregame_features(long: pd.DataFrame) -> pd.DataFrame:
    """Prior-to-game counts within (team, opponent)."""
    time_cols = [c for c in long.columns if c in ("game_date", "season", "week")]
    sort_cols = time_cols + ["gid"] if time_cols else ["gid"]
    long = long.sort_values(sort_cols).copy()

    grp = long.groupby(["team", "opponent"], group_keys=False)
    long["prior_games"] = grp.cumcount()
    long["prior_wins"] = grp["team_won"].cumsum() - long["team_won"].fillna(False)
    long["prior_losses"] = long["prior_games"] - long["prior_wins"]
    long["prior_dom"] = long["prior_wins"] - long["prior_losses"]
    long["prior_win_pct"] = np.where(long["prior_games"] > 0, long["prior_wins"] / long["prior_games"], np.nan)

    keep = ["gid","team","opponent","prior_games","prior_wins","prior_losses","prior_dom","prior_win_pct"]
    return long[keep]


def attach_pairwise_priors_to_games(df: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-game features to wide df: home_vs_away_* and away_vs_home_*."""
    df = df.reset_index(drop=False).rename(columns={"index": "gid"}).copy()

    pre_home = pre.rename(
        columns={
            "team": "home_team", "opponent": "away_team",
            "prior_games": "home_vs_away_prior_games",
            "prior_wins": "home_vs_away_prior_wins",
            "prior_losses": "home_vs_away_prior_losses",
            "prior_dom": "home_vs_away_prior_dom",
            "prior_win_pct": "home_vs_away_prior_win_pct",
        }
    )
    pre_away = pre.rename(
        columns={
            "team": "away_team", "opponent": "home_team",
            "prior_games": "away_vs_home_prior_games",
            "prior_wins": "away_vs_home_prior_wins",
            "prior_losses": "away_vs_home_prior_losses",
            "prior_dom": "away_vs_home_prior_dom",
            "prior_win_pct": "away_vs_home_prior_win_pct",
        }
    )

    out = df.merge(
        pre_home[
            ["gid","home_team","away_team",
             "home_vs_away_prior_games","home_vs_away_prior_wins","home_vs_away_prior_losses",
             "home_vs_away_prior_dom","home_vs_away_prior_win_pct"]
        ],
        on=["gid", "home_team", "away_team"], how="left",
    )
    out = out.merge(
        pre_away[
            ["gid","home_team","away_team",
             "away_vs_home_prior_games","away_vs_home_prior_wins","away_vs_home_prior_losses",
             "away_vs_home_prior_dom","away_vs_home_prior_win_pct"]
        ],
        on=["gid", "home_team", "away_team"], how="left",
    )
    return out


def build_dominance_features(df_games_wide: pd.DataFrame) -> dict:
    """Build static table, leak-free pregame columns, and dominance matrix."""
    df_games_wide = ensure_actual_winner(df_games_wide)
    long = _to_pairwise_long_edges(df_games_wide)
    pair_table = build_pairwise_record_table(long)
    pre = leak_free_pregame_features(long)
    df_with_pregame = attach_pairwise_priors_to_games(df_games_wide, pre)
    dom_matrix = pair_table.pivot(index="team", columns="opponent", values="dominance").fillna(0).astype(int)
    return {
        "pair_table": pair_table.sort_values(["team","opponent"]).reset_index(drop=True),
        "dominance_matrix": dom_matrix,
        "df_with_pregame": df_with_pregame,
    }


def _parse_pairwise_table_from_log_txt(txt_path: Path) -> Optional[pd.DataFrame]:
    """Parse 'Pairwise dominance table' from text file (best-effort)."""
    if not txt_path.exists():
        logging.warning("Dominance log not found at %s; skipping.", txt_path)
        return None
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Pairwise dominance table:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            logging.warning("Pairwise dominance table section not found in %s", txt_path)
            return None
        section = re.split(r"\n\s*\n", m.group(1), maxsplit=1)[0]
        lines = section.strip().splitlines()

        header_i = None
        for i, ln in enumerate(lines):
            if re.search(r"^\s*team\s+opponent\s+games\s+wins\s+losses\s+win_pct\s+dominance", ln, re.I):
                header_i = i
                break
        if header_i is None:
            logging.warning("Dominance table header not found")
            return None

        rows = []
        for ln in lines[header_i+1:]:
            parts = ln.split()
            if len(parts) < 7:
                break
            try:
                team, opponent = parts[0], parts[1]
                games, wins, losses = int(parts[2]), int(parts[3]), int(parts[4])
                win_pct = float(parts[5])
                dominance = int(parts[6])
                rows.append({"team": team.upper(), "opponent": opponent.upper(),
                             "games": games, "wins": wins, "losses": losses,
                             "win_pct": win_pct, "dominance": dominance})
            except Exception:
                continue
        if not rows:
            logging.warning("No rows parsed from dominance table")
            return None
        return pd.DataFrame(rows)
    except Exception as exc:
        logging.warning("Failed to parse dominance table from %s: %s", txt_path, exc)
        return None


def merge_dominance_table(game_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    """Join external pairwise dominance table twice (home_vs_away_* / away_vs_home_*)."""
    if pair_df is None or pair_df.empty:
        return game_df
    out = game_df.copy()
    keep = [c for c in ["games","wins","losses","win_pct","dominance"] if c in pair_df.columns]

    home_join = (
        out[["home_team","away_team"]]
        .rename(columns={"home_team":"team","away_team":"opponent"})
        .merge(pair_df, on=["team","opponent"], how="left")
        .rename(columns={c: f"home_vs_away_{c}" for c in keep})
    )
    away_join = (
        out[["home_team","away_team"]]
        .rename(columns={"away_team":"team","home_team":"opponent"})
        .merge(pair_df, on=["team","opponent"], how="left")
        .rename(columns={c: f"away_vs_home_{c}" for c in keep})
    )
    out = pd.concat([out, home_join[[c for c in home_join.columns if c.startswith("home_vs_away_")]]], axis=1)
    out = pd.concat([out, away_join[[c for c in away_join.columns if c.startswith("away_vs_home_")]]], axis=1)
    return out


def merge_dominance_matrix_values(game_df: pd.DataFrame, dom_matrix: pd.DataFrame) -> pd.DataFrame:
    """Merge dominance matrix cell values per game (home_vs_away_dom_matrix_val, away_vs_home_dom_matrix_val)."""
    if dom_matrix is None or dom_matrix.empty:
        return game_df
    out = game_df.copy()

    dom = dom_matrix.copy()
    dom.index = dom.index.astype(str).str.upper()
    dom.columns = dom.columns.astype(str).str.upper()

    def lookup(row, is_home=True):
        team = str(row["home_team" if is_home else "away_team"]).upper()
        opp = str(row["away_team" if is_home else "home_team"]).upper()
        try:
            raw_val = dom.at[team, opp]
        except (KeyError, ValueError):
            return np.nan
        if pd.isna(raw_val):
            return np.nan
        if isinstance(raw_val, numbers.Integral):
            return int(raw_val)
        if isinstance(raw_val, numbers.Real):
            return int(round(raw_val))
        return np.nan

    out["home_vs_away_dom_matrix_val"] = out.apply(lambda r: lookup(r, True), axis=1)
    out["away_vs_home_dom_matrix_val"] = out.apply(lambda r: lookup(r, False), axis=1)
    return out


# -----------------------------------------------------------------------------
# Optional encodings, calibration rows, save helpers
# -----------------------------------------------------------------------------

def encode_team_features(df: pd.DataFrame, method: str = "onehot") -> pd.DataFrame:
    """Default: one-hot encode 'home_team' and 'away_team'."""
    if method != "onehot":
        return df
    out = df.copy()
    if "home_team" in out.columns:
        out = pd.concat([out, pd.get_dummies(out["home_team"], prefix="team_home", dtype=np.uint8)], axis=1)
    if "away_team" in out.columns:
        out = pd.concat([out, pd.get_dummies(out["away_team"], prefix="team_away", dtype=np.uint8)], axis=1)
    return out


def add_calibration_rows(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Append n blank rows for calibration; flagged by 'is_calibration_row'."""
    if n <= 0:
        return df
    out = df.copy()
    if "is_calibration_row" not in out.columns:
        out["is_calibration_row"] = False
    blanks = pd.DataFrame({c: [np.nan]*n for c in out.columns})
    blanks["is_calibration_row"] = True
    return pd.concat([out, blanks], ignore_index=True)


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Chronologically sort and normalize dtypes."""
    out = df.copy()
    if {"season","week"}.issubset(out.columns):
        out["time_key"] = make_time_key(out)
        out = out.sort_values(["time_key", "game_id"]).drop(columns=["time_key"], errors="ignore")
    return out


def save_dataset(df: pd.DataFrame, out_dir: Path, legacy_root_copy: bool = False) -> None:
    """Save the final dataset to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_DATASET_NAME
    df.to_csv(out_path, index=False)
    logging.info("Wrote dataset → %s (%d rows, %d cols)", out_path, len(df), len(df.columns))
    if legacy_root_copy:
        root_copy = Path(".") / OUTPUT_DATASET_NAME
        df.to_csv(root_copy, index=False)
        logging.info("Legacy copy → %s", root_copy)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Build NFL game-level dataset (one row per game).")
    p.add_argument("--start", type=int, default=2014, help="Start season (inclusive).")
    p.add_argument("--end", type=int, default=2025, help="End season (inclusive).")
    p.add_argument("--out-dir", type=str, default="metrics/data", help="Output directory.")
    p.add_argument("--legacy-root-copy", action="store_true", help=f"Copy {OUTPUT_DATASET_NAME} to repo root.")
    return p.parse_args()


def create_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ELO rating features for teams with proper initialization and updates.
    
    ELO ratings track team strength over time:
    - Start each team at 1500
    - Update after each game based on actual vs expected result
    - K-factor controls how much ratings change per game
    """
    out = df.copy()
    
    # Check if we have the necessary columns
    if not {"season", "week", "home_team", "away_team"}.issubset(out.columns):
        logging.warning("Missing required columns for ELO features; skipping")
        return out
    
    # Initialize ELO ratings dict
    elo_ratings = {}
    K_FACTOR = 32  # Standard chess ELO k-factor
    
    def get_elo(team: str, default: float = 1500.0) -> float:
        """Get current ELO rating for a team."""
        return elo_ratings.get(team, default)
    
    def expected_score(elo_a: float, elo_b: float) -> float:
        """Calculate expected score for team A vs team B."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    
    def update_elo(team: str, actual: float, expected: float) -> float:
        """Update ELO rating based on game result."""
        current = get_elo(team)
        new_elo = current + K_FACTOR * (actual - expected)
        elo_ratings[team] = new_elo
        return new_elo
    
    # Add ELO columns
    out["home_elo_pre"] = np.nan
    out["away_elo_pre"] = np.nan
    out["elo_diff_pre"] = np.nan
    out["home_elo_post"] = np.nan
    out["away_elo_post"] = np.nan
    
    # Sort by time to process games chronologically
    if "time_key" not in out.columns:
        out["time_key"] = make_time_key(out)
    out = out.sort_values(["time_key", "game_id"]).reset_index(drop=True)
    
    # Process each game
    for idx, row in out.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        
        # Get pre-game ELO ratings
        home_elo_pre = get_elo(home_team)
        away_elo_pre = get_elo(away_team)
        
        out.at[idx, "home_elo_pre"] = home_elo_pre
        out.at[idx, "away_elo_pre"] = away_elo_pre
        out.at[idx, "elo_diff_pre"] = home_elo_pre - away_elo_pre
        
        # Update ELO if game is completed
        if pd.notna(row.get("home_points_for")) and pd.notna(row.get("away_points_for")):
            home_score = row["home_points_for"]
            away_score = row["away_points_for"]
            
            # Determine actual result (1 = home win, 0.5 = tie, 0 = away win)
            if home_score > away_score:
                home_actual, away_actual = 1.0, 0.0
            elif home_score < away_score:
                home_actual, away_actual = 0.0, 1.0
            else:
                home_actual, away_actual = 0.5, 0.5
            
            # Calculate expected scores
            home_expected = expected_score(home_elo_pre, away_elo_pre)
            away_expected = 1.0 - home_expected
            
            # Update ratings
            home_elo_post = update_elo(home_team, home_actual, home_expected)
            away_elo_post = update_elo(away_team, away_actual, away_expected)
            
            out.at[idx, "home_elo_post"] = home_elo_post
            out.at[idx, "away_elo_post"] = away_elo_post
        else:
            # Future game - use current ratings as post-game ratings
            out.at[idx, "home_elo_post"] = home_elo_pre
            out.at[idx, "away_elo_post"] = away_elo_pre
    
    logging.info("Created ELO features for %d teams", len(elo_ratings))
    return out


def create_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic game-level features from schedule data.
    
    Features include:
    - Game date parsing and day of week
    - Home field advantage indicators
    - Rest days differential
    - Division/conference matchup flags (if data available)
    """
    out = df.copy()
    
    # Parse game date if available
    if "game_date" in out.columns:
        out["game_date_parsed"] = pd.to_datetime(out["game_date"], errors="coerce")
        out["game_day_of_week"] = out["game_date_parsed"].dt.dayofweek  # 0=Monday, 6=Sunday
        out["is_weekend"] = out["game_day_of_week"].isin([5, 6])  # Saturday or Sunday
    
    # Home field advantage indicator (already present via is_home in other functions)
    if "home_team" in out.columns and "away_team" in out.columns:
        out["has_home_team"] = out["home_team"].notna() & out["away_team"].notna()
    
    # Rest differential if not already present
    if "home_rest" in out.columns and "away_rest" in out.columns and "rest_diff" not in out.columns:
        out["rest_diff"] = pd.to_numeric(out["home_rest"], errors="coerce") - pd.to_numeric(out["away_rest"], errors="coerce")
    
    # Game type indicators (regular season vs playoffs)
    if "game_type" in out.columns:
        out["is_regular_season"] = out["game_type"] == "REG"
        out["is_playoff"] = out["game_type"].isin(["WC", "DIV", "CON", "SB"])
    
    logging.info("Created basic game features")
    return out


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling window statistics for teams.
    
    Computes rolling averages over last N games for:
    - Points scored/allowed
    - Win percentage
    - Advanced metrics (if available)
    
    Uses strict shift(1) to avoid data leakage.
    """
    out = df.copy()
    
    # Check required columns
    if not {"season", "week", "home_team", "away_team"}.issubset(out.columns):
        logging.warning("Missing required columns for rolling features; skipping")
        return out
    
    # Sort chronologically
    if "time_key" not in out.columns:
        out["time_key"] = make_time_key(out)
    out = out.sort_values(["time_key", "game_id"]).reset_index(drop=True)
    
    # Convert to team-game long format for rolling calculation
    home_games = out[["season", "week", "game_id", "time_key", "home_team", "away_team", 
                      "home_points_for", "away_points_for"]].copy()
    home_games = home_games.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_points_for": "points_for",
        "away_points_for": "points_against"
    })
    home_games["is_home"] = 1
    
    away_games = out[["season", "week", "game_id", "time_key", "home_team", "away_team",
                      "home_points_for", "away_points_for"]].copy()
    away_games = away_games.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_points_for": "points_for",
        "home_points_for": "points_against"
    })
    away_games["is_home"] = 0
    
    # Combine
    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(["team", "time_key", "game_id"]).reset_index(drop=True)
    
    # Calculate win indicator
    completed = team_games["points_for"].notna() & team_games["points_against"].notna()
    team_games["win"] = np.where(
        completed,
        (team_games["points_for"] > team_games["points_against"]).astype(float),
        np.nan
    )
    
    # Compute rolling stats per team (with shift to avoid leakage)
    def safe_rolling(series, window):
        """Apply shift(1) then rolling to avoid current-game leakage."""
        return series.shift(1).rolling(window=window, min_periods=1).mean()
    
    for window in [3, 5, 10]:
        grouped = team_games.groupby("team", group_keys=False)
        team_games[f"rolling_pf_{window}"] = grouped["points_for"].apply(lambda x: safe_rolling(x, window))
        team_games[f"rolling_pa_{window}"] = grouped["points_against"].apply(lambda x: safe_rolling(x, window))
        team_games[f"rolling_win_pct_{window}"] = grouped["win"].apply(lambda x: safe_rolling(x, window))
    
    # Merge back to game-level (home and away)
    roll_cols = [c for c in team_games.columns if c.startswith("rolling_")]
    
    home_rolling = team_games[team_games["is_home"] == 1][["game_id", "team"] + roll_cols].copy()
    home_rolling.columns = ["game_id", "home_team"] + [f"home_{c}" for c in roll_cols]
    
    away_rolling = team_games[team_games["is_home"] == 0][["game_id", "team"] + roll_cols].copy()
    away_rolling.columns = ["game_id", "away_team"] + [f"away_{c}" for c in roll_cols]
    
    out = out.merge(home_rolling, on=["game_id", "home_team"], how="left")
    out = out.merge(away_rolling, on=["game_id", "away_team"], how="left")
    
    logging.info("Created rolling window features (windows: 3, 5, 10)")
    return out


def create_qb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create quarterback-specific features if QB data is available.
    
    Features include:
    - QB rating/performance metrics
    - QB experience (games started)
    - Backup QB indicators
    
    Note: Requires player-level data; gracefully skips if unavailable.
    """
    out = df.copy()
    
    # QB features require player stats which may not be available
    # Check if we have QB-related columns from player stats merge
    qb_cols = [c for c in out.columns if "qb" in c.lower()]
    
    if not qb_cols:
        logging.info("No QB data available; skipping QB-specific features")
        return out
    
    # If we have team_qb_* columns from player stats, create derived features
    if "team_qb_completions" in out.columns and "team_qb_attempts" in out.columns:
        # Completion percentage (if not already computed)
        if "team_qb_completion_pct" not in out.columns:
            out["home_qb_completion_pct"] = np.where(
                out.get("home_team_qb_attempts", 0) > 0,
                out.get("home_team_qb_completions", 0) / out.get("home_team_qb_attempts", 1),
                np.nan
            )
            out["away_qb_completion_pct"] = np.where(
                out.get("away_team_qb_attempts", 0) > 0,
                out.get("away_team_qb_completions", 0) / out.get("away_team_qb_attempts", 1),
                np.nan
            )
    
    # Yards per attempt
    if "team_qb_pass_yards" in out.columns and "team_qb_attempts" in out.columns:
        out["home_qb_ypa"] = np.where(
            out.get("home_team_qb_attempts", 0) > 0,
            out.get("home_team_qb_pass_yards", 0) / out.get("home_team_qb_attempts", 1),
            np.nan
        )
        out["away_qb_ypa"] = np.where(
            out.get("away_team_qb_attempts", 0) > 0,
            out.get("away_team_qb_pass_yards", 0) / out.get("away_team_qb_attempts", 1),
            np.nan
        )
    
    # TD to INT ratio
    if "team_qb_pass_tds" in out.columns and "team_qb_interceptions" in out.columns:
        out["home_qb_td_int_ratio"] = np.where(
            out.get("home_team_qb_interceptions", 0) > 0,
            out.get("home_team_qb_pass_tds", 0) / out.get("home_team_qb_interceptions", 1),
            out.get("home_team_qb_pass_tds", 0)  # If no INTs, just use TDs
        )
        out["away_qb_td_int_ratio"] = np.where(
            out.get("away_team_qb_interceptions", 0) > 0,
            out.get("away_team_qb_pass_tds", 0) / out.get("away_team_qb_interceptions", 1),
            out.get("away_team_qb_pass_tds", 0)
        )
    
    logging.info("Created QB-derived features from available stats")
    return out


def create_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive point differential, win flags, and winner label."""
    out = df.copy()
    if {"home_points_for", "away_points_for"}.issubset(out.columns):
        pf = pd.to_numeric(out["home_points_for"], errors="coerce")
        pa = pd.to_numeric(out["away_points_for"], errors="coerce")
        out["point_diff"] = pf - pa
        win_mask = pf.notna() & pa.notna()
        home_win = pd.Series(pd.NA, index=out.index, dtype="boolean")
        home_win.loc[win_mask] = pf.loc[win_mask] > pa.loc[win_mask]
        out["home_win"] = home_win
        winner = pd.Series(pd.NA, index=out.index, dtype="string")
        winner.loc[home_win == True] = out.loc[home_win == True, "home_team"].astype("string")
        winner.loc[home_win == False] = out.loc[home_win == False, "away_team"].astype("string")
        tie_mask = win_mask & (pf == pa)
        winner.loc[tie_mask] = "TIE"
        out["winner"] = winner
    else:
        out.setdefault("point_diff", np.nan)
        out.setdefault("home_win", pd.Series(pd.NA, index=out.index, dtype="boolean"))
        out.setdefault("winner", pd.Series(pd.NA, index=out.index, dtype="string"))
    return out


def build_dataset(
    start_season: int,
    end_season: int,
    out_dir: Path,
    legacy_root_copy: bool = False,
    production_mode: bool = True,
    include_future: bool = True,
) -> pd.DataFrame:
    """Build the full modeling dataset and save to CSV.
    
    Args:
        start_season: First season to include (inclusive)
        end_season: Last season to include (inclusive)
        out_dir: Directory to save output CSV
        legacy_root_copy: If True, also copy to repo root
        production_mode: If True, include production-ready features
        include_future: If True, include scheduled future games with null scores
        
    Returns:
        Final dataset DataFrame
    """
    seasons = list(range(start_season, end_season + 1))
    logging.info("Building dataset for seasons %d-%d", start_season, end_season)
    
    # Load all data sources
    logging.info("Loading data sources...")
    # Try multiple locations for PBP cache
    pbp_candidates = [
        Path("pbp_cache.csv"),
        Path("backend/pbp_cache.csv"),
        Path("backend/data/pbp_cache.csv"),
        Path("data/pbp_cache.csv"),
    ]
    pbp_path = next((p for p in pbp_candidates if p.exists()), Path("pbp_cache.csv"))
    team_game_metrics = load_team_game_metrics(pbp_path)
    team_stats = load_team_weekly_stats(seasons)
    player_stats = load_player_game_stats(seasons)
    schedule = load_schedules(seasons, include_future=include_future)

    # Start with schedule as base
    if schedule is None or schedule.empty:
        logging.error("No schedule data available. Cannot build dataset.")
        return pd.DataFrame()
    
    logging.info("Loaded %d games from schedule", len(schedule))
    
    # Build features using the existing add_features pipeline
    df = add_features(
        sch=schedule,
        windows=(3, 5),
        advanced_metrics=team_game_metrics if not team_game_metrics.empty else None
    )
    
    logging.info("Built base features, dataset has %d rows", len(df))
    
    # Merge additional team stats if available
    if team_stats is not None and not team_stats.empty:
        df = _merge_team_week_stats(df, team_stats, "team_stats")
        logging.info("Merged team weekly stats")
    
    # Merge player stats if available
    if player_stats is not None and not player_stats.empty:
        df = _merge_team_week_stats(df, player_stats, "player_stats")
        logging.info("Merged player stats")
    
    # Apply feature engineering functions
    logging.info("Applying feature engineering...")
    df = create_game_features(df)
    df = create_elo_features(df)
    df = create_rolling_features(df)
    df = create_qb_features(df)
    df = create_target_features(df)
    
    # Finalize and save
    df = finalize_dataset(df)
    
    if df is not None and not df.empty:
        save_dataset(df, out_dir, legacy_root_copy=legacy_root_copy)
        logging.info("Dataset build complete: %d rows, %d columns", len(df), len(df.columns))
    else:
        logging.warning("Final dataset is empty, nothing to save")
    
    return df


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------

def main() -> None:
    """Main entry point for CLI execution."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    setup_logger(out_dir)
    
    logging.info("=" * 60)
    logging.info("NFL Dataset Builder - PRODUCTION MODE")
    logging.info("Focus: Dataset building only (no model training)")
    logging.info("=" * 60)
    
    # Build dataset with correct parameters
    df = build_dataset(
        start_season=args.start,
        end_season=args.end,
        out_dir=out_dir,
        legacy_root_copy=args.legacy_root_copy,
        production_mode=True,
        include_future=True,
    )
    
    if df is not None and not df.empty:
        logging.info("=" * 60)
        logging.info("SUCCESS: Dataset ready for model training")
        logging.info("Output: %s/%s", out_dir, OUTPUT_DATASET_NAME)
        logging.info("Rows: %d, Columns: %d", len(df), len(df.columns))
        logging.info("Next step: Use enhanced_pipeline.py for model training")
        logging.info("=" * 60)
    else:
        logging.error("=" * 60)
        logging.error("FAILED: Dataset build produced no data")
        logging.error("=" * 60)


if __name__ == "__main__":
    main()

# -----------------------------
# Suggested Enhancements
# -----------------------------
# 1) Persist & reuse a canonical TEAM_MAP shared with the API to avoid LA/LAR
#    drift; consider exporting it into metadata alongside the dataset build.
# 2) Add opponent-relative features (home_minus_away of priors) to reduce
#    collinearity and match many sports modeling baselines.
# 3) Provide a "strict" mode that drops rows with insufficient history instead
#    of imputing means, to allow unbiased validation when desired.
