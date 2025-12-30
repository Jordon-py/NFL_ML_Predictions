<<<<<<< HEAD
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: build_csv_datasets.py
Purpose: CLI-driven dataset builder for NFL game-level features; produces one row per game with leak-safe rolling priors, market context, dominance features, and optional encodings.
Functions: parse_args, build_dataset, add_features, load_schedules, load_team_game_metrics, load_player_game_stats, _rolling_prior_stats, _ffill_prior_features, _team_game_long, create_elo_features, create_rolling_features, build_dominance_features, setup_logger, main
Variables: ABBR_FIX (team code map), OUTPUT_DATASET_NAME, NFL_BACKEND (nflreadpy or nfl_data_py), nfl (data loader), HAS_winner_BOOL, TIME_COLS_IN_ORDER
Interacts With: nflreadpy/nfl_data_py (schedules/PBP), backend/data/ (optional pbp_cache.csv), backend/train_models.py (consumes output CSV), backend/data/ (output directory)
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Sequence, Any
from datetime import datetime, timezone

import argparse
import json
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------

ABBR_FIX: Dict[str, str] = {
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
    "WSH": "WAS",
}

# Name of the output CSV file for the generated dataset.
# Includes current date in YYYYMMDD format for traceability/versioning.
OUTPUT_DATASET_NAME = f"game_features_{datetime.now().strftime('%Y%m%d')}.csv"

# Pairwise dominance helpers
HAS_winner_BOOL = True  # if you only have scores, set False
TIME_COLS_IN_ORDER: Optional[Sequence[str]] = None  # auto-detect if None


def make_time_key(df: pd.DataFrame) -> pd.Series:
    """Return sortable integer key YYYYWW from 'season' and 'week' (assumes ints)."""
    season = df["season"].fillna(0).astype(int)
    week = df["week"].fillna(0).astype(int)
    return (season * 100) + week



# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logger(out_dir: Path) -> None:
    """Initialize both file and console logging so CLI users get progress feedback."""
    
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "build_csv_datasets.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info("Logger initialized → %s", log_file)



# ---------------------------------------------------------------------
# Backend selection (nflreadpy preferred, fallback to nfl_data_py)
# ---------------------------------------------------------------------
NFL_BACKEND: Optional[str] = None
nfl: Optional[str] = None
_fallback_reason = None

def _note_backend(msg: str, level: int = logging.INFO) -> None:
    logging.log(level, msg)

try:
    import nflreadpy as nfl
    
    try:
        _probe = nfl.load_schedules(seasons=[2025])
        print("Probing nflreadpy backend...", _probe, flush=True)
        if hasattr(_probe, "to_pandas"):
            _data = _probe.head(1).to_pandas()
            nfl = "nflreadpy"
            NFL_BACKEND = nfl
        _note_backend(msg="Using backend 'nflreadpy'", level=4) 
    

    except Exception as e:
        _fallback_reason = f"nflreadpy probe failed: {e}"
        import nfl_data_py as n
        if n is not None and nfl is None:
            nfl = n
            NFL_BACKEND = "nfl_data_py"
        _note_backend(
            f"Using fallback backend '{NFL_BACKEND}' — {_fallback_reason}", logging.WARNING,
        )
    finally:
        del _probe
        del _data
       
except Exception as e:
    print(f"Failed to import nflreadpy: {e}", flush=True)



# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
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

    probs.loc[negative] = (-ml_numeric.loc[negative]) / (
        (-ml_numeric.loc[negative]) + 100
    )
    probs.loc[positive & ml_numeric.notna()] = 100 / (
        ml_numeric.loc[positive & ml_numeric.notna()] + 100
    )
    return probs



# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
pbp_cache_path = Path("../pbp_cache.csv")

def load_team_game_metrics(pbp_path: Path) -> pd.DataFrame:
    """Aggregate play-by-play data to per-team, per-game advanced metrics."""
    # If nflreadpy available, load directly; otherwise use cached CSV
    if nfl is not None and NFL_BACKEND == "nflreadpy":
        try:
            seasons_to_load: list[int] =(
                    [int(s) for s in input('Enter seasons to load (comma-separated, e.g., start=2010, end=2022):2010,2022\n').strip().split(',')]
            )

            logging.info(
                "Loading play-by-play via nflreadpy for seasons %s", seasons_to_load
            )
            pbp_raw = nfl.load_pbp(seasons=seasons_to_load)
            pbp = to_pandas_safe(pbp_raw)
            logging.info("Loaded %d play-by-play rows from nflreadpy", len(pbp))
            
        except Exception as exc:
                logging.warning(
                    "nflreadpy PBP load failed (%s); falling back to cached CSV", exc
                )
                if len(pbp.columns) == 0:
                    raise ValueError("No columns loaded from nflreadpy PBP data")
    else:
        if pbp is None:
            if not pbp_path.exists():
                logging.warning(
                    "Cached PBP missing at %s; advanced PBP features disabled", pbp_path
                )
                return pd.DataFrame(columns=["season", "week", "game_id", "team"])
            pbp = pd.read_csv(pbp_path, low_memory=False)
            logging.info("Loaded %d play-by-play rows from CSV", len(pbp))

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

    for col in [
        "epa",
        "success",
        "pass",
        "xpass",
        "pass_attempt",
        "rush_attempt",
        "third_down_converted",
        "third_down_failed",
        "interception",
        "fumble_lost",
        "yards_gained",
    ]:
        if col in pbp.columns:
            pbp[col] = pbp[col].fillna(0.0)
        else:
            pbp[col] = 0.0

    pbp["turnover"] = pbp["interception"] + pbp["fumble_lost"]
    pbp["explosive_play"] = ((pbp["pass"] == 1.0) & (pbp["yards_gained"] >= 20)) | (
        (pbp["rush_attempt"] == 1.0) & (pbp["yards_gained"] >= 15)
    )

    off_group = ["season", "week", "game_id", "posteam"]
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
    total_def_plays = (
        def_agg["def_pass_attempts_faced"] + def_agg["def_rush_attempts_faced"]
    )
    def_agg["def_takeaway_rate"] = np.where(
        total_def_plays > 0,
        def_agg["def_takeaways"] / total_def_plays,
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

    return metrics.reset_index(drop=True)


def load_player_game_stats(seasons: List[int]) -> pd.DataFrame:
    """Load weekly player stats and aggregate to team-week level."""
    if nfl is None:
        logging.warning("No NFL backend available; player features disabled")
        return pd.DataFrame(columns=["season", "week", "team"])

    try:
        logging.info("Loading player stats via %s for seasons %s", NFL_BACKEND, seasons)
        player_stats: Optional[pd.DataFrame] = None

        if NFL_BACKEND == "nflreadpy":
            loader = getattr(nfl, "load_player_stats", None)
            if loader:
                player_stats = to_pandas_safe(
                    loader(seasons=seasons, summary_level="week")
                )
        else:
            loader = getattr(nfl, "import_weekly_data", None)
            if loader:
                player_stats = loader(seasons)

        if player_stats is None or player_stats.empty:
            logging.warning("Player stats unavailable; skipping.")
            return pd.DataFrame(columns=["season", "week", "team"])

        team_col = "recent_team" if "recent_team" in player_stats.columns else (
            "team" if "team" in player_stats.columns else None
        )
        if not team_col:
            logging.warning("Cannot determine team column in player_stats; skipping")
            return pd.DataFrame(columns=["season", "week", "team"])

        # QB aggregation
        if "position" in player_stats.columns:
            qb = player_stats[player_stats["position"] == "QB"].copy()
        else:
            qb = pd.DataFrame()

        qb_cols = {
            "passing_yards": "team_qb_pass_yards",
            "passing_tds": "team_qb_pass_tds",
            "interceptions": "team_qb_interceptions",
            "sacks": "team_qb_sacks",
            "completions": "team_qb_completions",
            "attempts": "team_qb_attempts",
        }

        qb_agg_dict = {
            tgt: (src, "sum") for src, tgt in qb_cols.items() if src in qb.columns
        }
        if qb_agg_dict and not qb.empty:
            qb_agg = qb.groupby(["season", "week", team_col], as_index=False).agg(
                **qb_agg_dict
            )
            if {
                "team_qb_completions",
                "team_qb_attempts",
            }.issubset(qb_agg.columns):
                qb_agg["team_qb_completion_pct"] = np.where(
                    qb_agg["team_qb_attempts"] > 0,
                    qb_agg["team_qb_completions"] / qb_agg["team_qb_attempts"],
                    np.nan,
                )
        else:
            qb_agg = pd.DataFrame(columns=["season", "week", team_col])

        # RB aggregation
        if "position" in player_stats.columns:
            rb = player_stats[player_stats["position"].isin(["RB"])].copy()
        else:
            rb = pd.DataFrame()

        rb_cols = {
            "rushing_yards": "team_rb_rush_yards",
            "rushing_tds": "team_rb_rush_tds",
            "receptions": "team_rb_receptions",
            "receiving_yards": "team_rb_receiving_yards",
        }
        rb_agg_dict = {
            tgt: (src, "sum") for src, tgt in rb_cols.items() if src in rb.columns
        }
        if rb_agg_dict and not rb.empty:
            rb_agg = rb.groupby(["season", "week", team_col], as_index=False).agg(
                **rb_agg_dict
            )
        else:
            rb_agg = pd.DataFrame(columns=["season", "week", team_col])

        # WR+TE aggregation
        if "position" in player_stats.columns:
            pass_catchers = player_stats[
                player_stats["position"].isin(["WR", "TE"])
            ].copy()
        else:
            pass_catchers = pd.DataFrame()

        wr_cols = {
            "targets": "team_wr_targets",
            "receptions": "team_wr_receptions",
            "receiving_yards": "team_wr_receiving_yards",
            "receiving_tds": "team_wr_receiving_tds",
        }
        wr_agg_dict = {
            tgt: (src, "sum")
            for src, tgt in wr_cols.items()
            if src in pass_catchers.columns
        }
        if wr_agg_dict and not pass_catchers.empty:
            wr_agg = pass_catchers.groupby(
                ["season", "week", team_col], as_index=False
            ).agg(**wr_agg_dict)
        else:
            wr_agg = pd.DataFrame(columns=["season", "week", team_col])

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
    """Load team-level weekly stats when available."""
    if nfl is None:
        logging.warning("No NFL backend; team-week stats disabled")
        return pd.DataFrame(columns=["season", "week", "team"])

    try:
        logging.info(
            "Loading team weekly stats via %s for seasons %s", NFL_BACKEND, seasons
        )
        if NFL_BACKEND == "nflreadpy":
            loader = getattr(nfl, "load_team_stats", None)
            if loader:
                team_stats = to_pandas_safe(
                    loader(seasons=seasons, summary_level="week")
                )
            else:
                team_stats = None
        else:
            loader = getattr(nfl, "import_team_desc", None)
            if loader:
                # nfl_data_py doesn't have a direct weekly team stats equivalent;
                # if unavailable, just skip.
                team_stats = None
            else:
                team_stats = None

        if team_stats is None or team_stats.empty:
            logging.warning("Team weekly stats unavailable; skipping")
            return pd.DataFrame(columns=["season", "week", "team"])

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
        ]
        available = [c for c in feature_cols if c in team_stats.columns]
        return team_stats[available].fillna(0).reset_index(drop=True)

    except Exception as exc:
        logging.warning("Team stats load failed (%s); team features disabled", exc)
        return pd.DataFrame(columns=["season", "week", "team"])


def load_schedules(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
    """Load schedules + final scores for given seasons."""
    sch: Optional[pd.DataFrame] = None

    if nfl is not None:
        try:
            if NFL_BACKEND == "nflreadpy":
                loader = getattr(nfl, "load_schedules", None)
                if loader:
                    sch = to_pandas_safe(loader(seasons=seasons))
            else:
                loader = getattr(nfl, "import_schedules", None)
                if loader:
                    sch = loader(seasons)
        except Exception as exc:
            logging.warning("Schedule load failed via backend (%s)", exc)
            sch = None

    if sch is None or sch.empty:
        raise RuntimeError("Could not load schedules from any backend.")

    need = [
        "season",
        "week",
        "game_id",
        "gameday",
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
    missing = [c for c in need if c not in sch.columns]
    if missing:
        raise RuntimeError(f"Missing schedule columns: {missing}")

    sch = _normalize_codes(sch, ["home_team", "away_team"])
    sch = sch.rename(columns={"gameday": "game_date"})
    sch["week"] = sch["week"].astype(int)

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

    if include_future:
        completed = sch.dropna(subset=["home_score", "away_score"]).reset_index(
            drop=True
        )
        future = sch[
            sch["home_score"].isna() | sch["away_score"].isna()
        ].copy()
        future["home_score"] = None
        future["away_score"] = None
        future = future[future["game_type"] == "REG"].reset_index(drop=True)

        logging.info(
            "Loaded %d completed games + %d future games", len(completed), len(future)
        )
        return pd.concat([completed, future], ignore_index=True)

    sch = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    logging.info("Schedules loaded: %d completed games", len(sch))
    return sch


# ---------------------------------------------------------------------
# Core feature building
# ---------------------------------------------------------------------


def _team_game_long(sch: pd.DataFrame) -> pd.DataFrame:
    """Convert per-game schedule to per-team per-game long format."""
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

    completed_mask = long["points_for"].notna() & long["points_against"].notna()
    long["win"] = np.where(
        completed_mask, (long["points_for"] > long["points_against"]).astype(float), np.nan
    )

    long["time_key"] = make_time_key(long)
    return long.sort_values(["team", "time_key", "game_id"]).reset_index(drop=True)


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
            if col not in df.columns:
                continue
            df[f"prior_{col}_{window}"] = safe_rolling_mean(df[col])

    return df


def add_features(
    sch: pd.DataFrame,
    windows: Tuple[int, ...] = (3, 5),
    advanced_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build one-row-per-game features by:
      1) creating per-team rows,
      2) computing prior rolling stats,
      3) re-pivoting to wide with home_/away_ prefixes,
      4) adding market & rest features.
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
            right=advanced_metrics, on=["season", "week", "game_id", "team"], how="left"
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

    wide = home_side.merge(
        away_side,
        left_on="home_game_id",
        right_on="away_game_id",
        how="inner",
    )
    wide = wide.rename(columns={"home_game_id": "game_id"}).drop(
        columns=["away_game_id"]
    )

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

    wide["season"] = wide["home_season"].astype(int)
    wide["week"] = wide["home_week"].astype(int)
    wide["time_key"] = make_time_key(wide)
    wide = wide.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    prior_pairs = [c for c in wide.columns if c.startswith("home_prior_")]
    for home_col in prior_pairs:
        suffix = home_col[len("home_prior_") :]
        away_col = f"away_prior_{suffix}"
        if away_col in wide.columns:
            wide[f"home_minus_away_{suffix}"] = wide[home_col] - wide[away_col]

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
    prior_feature_cols = [
        c for c in wide.columns if c.startswith(("home_prior_", "away_prior_"))
    ]
    diff_feature_cols = [c for c in wide.columns if c.startswith("home_minus_away_")]
    final_cols = ordered_cols + prior_feature_cols + diff_feature_cols

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
    wide = wide.merge(schedule_extras, on="game_id", how="left")

    wide["home_moneyline"] = pd.to_numeric(wide["home_moneyline"], errors="coerce")
    wide["away_moneyline"] = pd.to_numeric(wide["away_moneyline"], errors="coerce")
    wide["home_moneyline_prob"] = _moneyline_to_prob(wide["home_moneyline"])
    wide["away_moneyline_prob"] = _moneyline_to_prob(wide["away_moneyline"])
    wide["moneyline_prob_diff"] = (
        wide["home_moneyline_prob"] - wide["away_moneyline_prob"]
    )

    wide["spread_line"] = pd.to_numeric(wide["spread_line"], errors="coerce")
    wide["total_line"] = pd.to_numeric(wide["total_line"], errors="coerce")
    wide["home_rest"] = pd.to_numeric(wide["home_rest"], errors="coerce")
    wide["away_rest"] = pd.to_numeric(wide["away_rest"], errors="coerce")
    wide["rest_diff"] = wide["home_rest"] - wide["away_rest"]

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

    # Ensure we only select existing columns
    final_cols = [c for c in final_cols if c in wide.columns]

    return wide[final_cols].reset_index(drop=True)


def _merge_team_week_stats(
    game_df: pd.DataFrame, team_week_stats: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    """
    Merge team-week level stats into game-level dataframe for both home and away teams.
    """
    if team_week_stats is None or team_week_stats.empty:
        return game_df

    stat_cols = [c for c in team_week_stats.columns if c not in {"season", "week", "team"}]

    home_stats = team_week_stats.copy()
    home_stats.columns = ["season", "week", "home_team"] + [
        f"home_{prefix}_{c}" for c in stat_cols
    ]
    game_df = game_df.merge(home_stats, on=["season", "week", "home_team"], how="left")

    away_stats = team_week_stats.copy()
    away_stats.columns = ["season", "week", "away_team"] + [
        f"away_{prefix}_{c}" for c in stat_cols
    ]
    game_df = game_df.merge(away_stats, on=["season", "week", "away_team"], how="left")

    return game_df


# ---------------------------------------------------------------------
# Simple regression pipeline + time split (kept for reuse)
# ---------------------------------------------------------------------


def build_regression_pipeline(
    numeric_features: List[str], categorical_features: List[str], alpha: float = 1.0
) -> Pipeline:
    """Simple numeric+categorical -> Ridge regression pipeline."""
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

    model = Ridge(alpha=alpha)
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    return pipeline


def ts_split_by_season_week(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    train_end: Tuple[int, int],
):
    """Chronological split that prevents leakage; returns (X_train, y_train), df, sorted_df."""
    data = df.copy()
    required_cols = {"season", "week", *features, target}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["time_key"] = make_time_key(data)
    data = data.sort_values(["time_key"]).reset_index(drop=True)

    train_end_season, train_end_week = train_end
    is_train = (data["season"] < train_end_season) | (
        (data["season"] == train_end_season) & (data["week"] <= train_end_week)
    )

    train_df = data.loc[is_train]
    X_train, y_train = train_df[features], train_df[target]
    return (X_train, y_train), df, data


# ---------------------------------------------------------------------
# Dominance / pairwise priors
# ---------------------------------------------------------------------


def ensure_actual_winner(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'home_win' (bool) and 'actual_winner' (team name) columns exist."""
    df = df.copy()

    if "home_win" in df.columns:
        win_series = pd.Series(df["home_win"], index=df.index, dtype="boolean")
    else:
        if not HAS_winner_BOOL:
            if {"home_points_for", "away_points_for"}.issubset(df.columns):
                win_series = pd.Series(
                    df["home_points_for"] > df["away_points_for"],
                    index=df.index,
                    dtype="boolean",
                )
            else:
                raise ValueError(
                    "Need either 'home_win' bool or score columns "
                    "'home_points_for'/'away_points_for'."
                )
        else:
            if "winner" not in df.columns:
                raise ValueError("Need 'winner' column when HAS_winner_BOOL=True.")
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


def detect_time_cols(df: pd.DataFrame) -> List[str]:
    if TIME_COLS_IN_ORDER is not None:
        return list(TIME_COLS_IN_ORDER)
    if "game_date" in df.columns:
        return ["game_date"]
    if {"season", "week"}.issubset(df.columns):
        return ["season", "week"]
    return []


def make_long_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a 'long' dataframe with two rows per game:
      - (team = home, opponent = away, team_won = winner)
      - (team = away, opponent = home, team_won = not winner)
    Includes 'gid' (row index of original df) to merge features back later.
    """
    base = df.reset_index(drop=False).rename(columns={"index": "gid"}).copy()

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

    home_rows = base[["gid", "home_team", "away_team"]].copy()
    home_rows = home_rows.rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    home_rows["team_won"] = win_series.reindex(home_rows.index)
    home_rows = home_rows[["gid", "team", "opponent", "team_won"]]

    away_rows = base[["gid", "home_team", "away_team"]].copy()
    away_rows = away_rows.rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    away_rows["team_won"] = (~win_series).reindex(away_rows.index)
    away_rows = away_rows[["gid", "team", "opponent", "team_won"]]

    long = pd.concat([home_rows, away_rows], ignore_index=True)

    time_cols = detect_time_cols(df)
    if time_cols:
        long = long.merge(base[["gid"] + time_cols], on="gid", how="left")
    return long


def pairwise_table(long: pd.DataFrame) -> pd.DataFrame:
    """Static, all-time pairwise records (order matters: (team, opponent))."""
    g = long.groupby(["team", "opponent"], as_index=False, observed=True)
    agg = g.agg(games=("team", "size"), wins=("team_won", "sum"))
    agg["losses"] = agg["games"] - agg["wins"]
    agg["win_pct"] = np.where(agg["games"] > 0, agg["wins"] / agg["games"], np.nan)
    agg["dominance"] = agg["wins"] - agg["losses"]
    agg["signed_wins_style"] = agg["wins"]
    return agg


def leak_free_pregame_features(long: pd.DataFrame) -> pd.DataFrame:
    """
    For each (team, opponent), compute *prior-to-game* counts:
      prior_games, prior_wins, prior_losses, prior_dominance, prior_win_pct
    """
    time_cols = [c for c in long.columns if c in ("game_date", "season", "week")]
    sort_cols = time_cols + ["gid"] if time_cols else ["gid"]
    long = long.sort_values(sort_cols).copy()

    grp = long.groupby(["team", "opponent"], group_keys=False, observed=True)

    long["prior_games"] = grp.cumcount()
    long["prior_wins"] = grp["team_won"].cumsum() - long["team_won"]
    long["prior_losses"] = long["prior_games"] - long["prior_wins"]
    long["prior_dom"] = long["prior_wins"] - long["prior_losses"]
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


def attach_pregame_to_wide(df: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-game features back to the original wide df."""
    df = df.reset_index(drop=False).rename(columns={"index": "gid"}).copy()

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


def team_level_prediction_accuracy(df: pd.DataFrame) -> pd.Series:
    """
    Accuracy of your classifier *when a given team plays*.

    Expects:
      - df['prob_winner'] in [0,1] (probability home team wins)
      - df['home_win'] boolean
      - df['home_team'], df['away_team']
    """
    required = {"prob_winner", "home_win", "home_team", "away_team"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns for team accuracy: {required - set(df.columns)}")

    df = df.copy()
    df["pred_winner"] = np.where(
        df["prob_winner"] >= 0.5, df["home_team"], df["away_team"]
    )

    actual = pd.Series(pd.NA, index=df.index, dtype="string")
    actual.loc[df["home_win"] == True] = df.loc[df["home_win"] == True, "home_team"].astype("string")
    actual.loc[df["home_win"] == False] = df.loc[df["home_win"] == False, "away_team"].astype("string")
    df["actual_winner"] = actual

    df["pred_correct"] = (df["pred_winner"] == df["actual_winner"]).astype(int)

    home_part = df[["home_team", "pred_correct"]].rename(columns={"home_team": "team"})
    away_part = df[["away_team", "pred_correct"]].rename(columns={"away_team": "team"})
    team_games = pd.concat([home_part, away_part], ignore_index=True)

    acc = team_games.groupby("team")["pred_correct"].mean().sort_values(ascending=False)
    return acc


def build_dominance_features(dff: pd.DataFrame) -> Dict[str, Any]:
    """Compute pairwise dominance tables and leak-free pregame dominance features."""
    df = ensure_actual_winner(dff)
    long = make_long_edges(df)
    pair_table = pairwise_table(long)
    pre = leak_free_pregame_features(long)
    df_with_pregame = attach_pregame_to_wide(df, pre)

    dom_matrix = pair_table.pivot(
        index="team", columns="opponent", values="dominance"
    ).fillna(0).astype(int)

    team_acc: Optional[pd.Series] = None
    if "prob_winner" in df_with_pregame.columns:
        try:
            team_acc = team_level_prediction_accuracy(df_with_pregame)
        except Exception as exc:
            logging.warning("Failed to compute team prediction accuracy: %s", exc)

    return {
        "pair_table": pair_table.sort_values(["team", "opponent"]).reset_index(
            drop=True
        ),
        "dominance_matrix": dom_matrix,
        "df_with_pregame": df_with_pregame,
        "team_prediction_accuracy": team_acc,
    }


# ---------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------


def build_game_feature_metadata(
    df: pd.DataFrame,
    *,
    include_future: bool,
) -> Dict[str, Any]:
    """Inspect the final dataframe and build a structured metadata dict."""
    meta: Dict[str, Any] = {}

    meta["build_timestamp_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    meta["rows_total"] = int(df.shape[0])
    meta["n_features_total"] = int(df.shape[1])
    meta["include_future_games"] = bool(include_future)

    if "season" in df.columns:
        meta["season_min"] = int(df["season"].min())
        meta["season_max"] = int(df["season"].max())

    id_cols = [
        c
        for c in (
            "season",
            "week",
            "game_id",
            "game_date",
            "home_game_date",
            "home_team",
            "away_team",
        )
        if c in df.columns
    ]
    target_cols = [
        c
        for c in (
            "home_points_for",
            "away_points_for",
            "home_win",
            "winner",
            "point_diff",
        )
        if c in df.columns
    ]
    prior_cols = sorted(
        c for c in df.columns if c.startswith(("home_prior_", "away_prior_"))
    )
    prior_diff_cols = sorted(
        c for c in df.columns if c.startswith("home_minus_away_")
    )
    market_cols = sorted(
        c
        for c in df.columns
        if c
        in {
            "home_moneyline_prob",
            "away_moneyline_prob",
            "moneyline_prob_diff",
            "spread_line",
            "total_line",
        }
    )
    rest_cols = sorted(c for c in df.columns if "rest" in c)
    dominance_cols = sorted(
        c for c in df.columns if c.startswith(("home_vs_away_", "away_vs_home_"))
    )

    meta["columns"] = {
        "id": id_cols,
        "target": target_cols,
        "priors": prior_cols,
        "priors_diff": prior_diff_cols,
        "market": market_cols,
        "rest": rest_cols,
        "dominance": dominance_cols,
    }

    windows = set()
    window_pattern = re.compile(r"_(\d+)$")
    for col in prior_cols:
        m = window_pattern.search(col)
        if m:
            try:
                windows.add(int(m.group(1)))
            except ValueError:
                pass

    meta["priors_config"] = {
        "windows": sorted(windows),
    }

    has_scores = {"home_points_for", "away_points_for"}.issubset(df.columns)
    if has_scores:
        train_mask = df["home_points_for"].notna() & df["away_points_for"].notna()
    else:
        train_mask = pd.Series(True, index=df.index)

    if prior_cols:
        train_slice = df.loc[train_mask, prior_cols]
        priors_median = train_slice.median(numeric_only=True)
        priors_median_json = {
            k: (None if pd.isna(v) else float(v)) for k, v in priors_median.items()
        }
    else:
        priors_median_json = {}

    meta["priors_imputation"] = {
        "strategy": "median_over_training_rows_with_history",
        "training_rows": int(train_mask.sum()),
        "baseline_medians": priors_median_json,
    }

    return meta


# ---------------------------------------------------------------------
# Additional feature engineering helpers
# ---------------------------------------------------------------------


def create_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ELO rating features for teams with proper initialization and updates."""
    out = df.copy()

    required = {"season", "week", "home_team", "away_team", "game_id"}
    if not required.issubset(out.columns):
        logging.warning("Missing required columns for ELO features; skipping")
        return out

    elo_ratings: Dict[str, float] = {}
    K_FACTOR = 32.0

    def get_elo(team: str, default: float = 1500.0) -> float:
        return float(elo_ratings.get(team, default))

    def expected_score(elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def update_elo(team: str, actual: float, expected: float) -> float:
        current = get_elo(team)
        new_elo = current + K_FACTOR * (actual - expected)
        elo_ratings[team] = new_elo
        return new_elo

    for col in [
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff_pre",
        "home_elo_post",
        "away_elo_post",
    ]:
        out[col] = np.nan

    if "time_key" not in out.columns:
        out["time_key"] = make_time_key(out)

    out = out.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    for idx, row in out.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_elo_pre = get_elo(home_team)
        away_elo_pre = get_elo(away_team)

        out.at[idx, "home_elo_pre"] = home_elo_pre
        out.at[idx, "away_elo_pre"] = away_elo_pre
        out.at[idx, "elo_diff_pre"] = home_elo_pre - away_elo_pre

        home_score = row.get("home_points_for")
        away_score = row.get("away_points_for")

        if pd.notna(home_score) and pd.notna(away_score):
            if home_score > away_score:
                home_actual, away_actual = 1.0, 0.0
            elif home_score < away_score:
                home_actual, away_actual = 0.0, 1.0
            else:
                home_actual, away_actual = 0.5, 0.5

            home_expected = expected_score(home_elo_pre, away_elo_pre)
            away_expected = 1.0 - home_expected

            home_elo_post = update_elo(home_team, home_actual, home_expected)
            away_elo_post = update_elo(away_team, away_actual, away_expected)

            out.at[idx, "home_elo_post"] = home_elo_post
            out.at[idx, "away_elo_post"] = away_elo_post
        else:
            out.at[idx, "home_elo_post"] = home_elo_pre
            out.at[idx, "away_elo_post"] = away_elo_pre

    logging.info("Created ELO features for %d teams", len(elo_ratings))
    return out


def create_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic game-level features from schedule data."""
    out = df.copy()

    if "game_date" in out.columns:
        out["game_date_parsed"] = pd.to_datetime(out["game_date"], errors="coerce")
        out["game_day_of_week"] = out["game_date_parsed"].dt.dayofweek
        out["is_weekend"] = out["game_day_of_week"].isin([5, 6])

    if {"home_team", "away_team"}.issubset(out.columns):
        out["has_home_team"] = out["home_team"].notna() & out["away_team"].notna()

    if (
        "home_rest" in out.columns
        and "away_rest" in out.columns
        and "rest_diff" not in out.columns
    ):
        out["rest_diff"] = (
            pd.to_numeric(out["home_rest"], errors="coerce")
            - pd.to_numeric(out["away_rest"], errors="coerce")
        )

    if "game_type" in out.columns:
        out["is_regular_season"] = out["game_type"] == "REG"
        out["is_playoff"] = out["game_type"].isin(["WC", "DIV", "CON", "SB"])

    logging.info("Created basic game features")
    return out


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling window statistics for teams (points + win%)."""
    out = df.copy()

    if not {"season", "week", "home_team", "away_team", "game_id"}.issubset(out.columns):
        logging.warning("Missing required columns for rolling features; skipping")
        return out

    if "time_key" not in out.columns:
        out["time_key"] = make_time_key(out)
    out = out.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    home_games = out[
        [
            "season",
            "week",
            "game_id",
            "time_key",
            "home_team",
            "away_team",
            "home_points_for",
            "away_points_for",
        ]
    ].copy()
    home_games = home_games.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_points_for": "points_for",
            "away_points_for": "points_against",
        }
    )
    home_games["is_home"] = 1

    away_games = out[
        [
            "season",
            "week",
            "game_id",
            "time_key",
            "home_team",
            "away_team",
            "home_points_for",
            "away_points_for",
        ]
    ].copy()
    away_games = away_games.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_points_for": "points_for",
            "home_points_for": "points_against",
        }
    )
    away_games["is_home"] = 0

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(
        ["team", "time_key", "game_id"]
    ).reset_index(drop=True)

    completed = team_games["points_for"].notna() & team_games["points_against"].notna()
    team_games["win"] = np.where(
        completed, (team_games["points_for"] > team_games["points_against"]).astype(float), np.nan
    )

    def safe_rolling(series: pd.Series, window: int) -> pd.Series:
        return series.shift(1).rolling(window=window, min_periods=1).mean()

    for window in [3, 5, 10]:
        grouped = team_games.groupby("team", group_keys=False)
        team_games[f"rolling_pf_{window}"] = grouped["points_for"].apply(
            lambda x: safe_rolling(x, window)
        )
        team_games[f"rolling_pa_{window}"] = grouped["points_against"].apply(
            lambda x: safe_rolling(x, window)
        )
        team_games[f"rolling_win_pct_{window}"] = grouped["win"].apply(
            lambda x: safe_rolling(x, window)
        )

    roll_cols = [c for c in team_games.columns if c.startswith("rolling_")]

    home_rolling = team_games[team_games["is_home"] == 1][
        ["game_id", "team"] + roll_cols
    ].copy()
    home_rolling.columns = ["game_id", "home_team"] + [f"home_{c}" for c in roll_cols]

    away_rolling = team_games[team_games["is_home"] == 0][
        ["game_id", "team"] + roll_cols
    ].copy()
    away_rolling.columns = ["game_id", "away_team"] + [f"away_{c}" for c in roll_cols]

    out = out.merge(home_rolling, on=["game_id", "home_team"], how="left")
    out = out.merge(away_rolling, on=["game_id", "away_team"], how="left")

    logging.info("Created rolling window features (windows: 3, 5, 10)")
    return out


def create_qb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create quarterback-specific features if QB data is available."""
    out = df.copy()
    qb_cols = [c for c in out.columns if "qb" in c.lower()]
    if not qb_cols:
        logging.info("No QB data available; skipping QB-specific features")
        return out

    # These names only exist if you merged player stats in a compatible way.
    if {
        "home_player_team_qb_attempts",
        "home_player_team_qb_completions",
    }.issubset(out.columns):
        out["home_qb_completion_pct"] = np.where(
            out["home_player_team_qb_attempts"] > 0,
            out["home_player_team_qb_completions"]
            / out["home_player_team_qb_attempts"],
            np.nan,
        )
    if {
        "away_player_team_qb_attempts",
        "away_player_team_qb_completions",
    }.issubset(out.columns):
        out["away_qb_completion_pct"] = np.where(
            out["away_player_team_qb_attempts"] > 0,
            out["away_player_team_qb_completions"]
            / out["away_player_team_qb_attempts"],
            np.nan,
        )

    logging.info("Created QB-derived features from available stats (if any)")
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
        winner.loc[home_win == True] = out.loc[
            home_win == True, "home_team"
        ].astype("string")
        winner.loc[home_win == False] = out.loc[
            home_win == False, "away_team"
        ].astype("string")
        tie_mask = win_mask & (pf == pa)
        winner.loc[tie_mask] = "TIE"
        out["winner"] = winner
    else:
        if "point_diff" not in out.columns:
            out["point_diff"] = np.nan
        if "home_win" not in out.columns:
            out["home_win"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        if "winner" not in out.columns:
            out["winner"] = pd.Series(pd.NA, index=out.index, dtype="string")
    return out


# ---------------------------------------------------------------------
# Dataset builder (canonical)
# ---------------------------------------------------------------------


def build_dataset(
    start_season: int,
    end_season: int,
    out_dir: Path,
    legacy_root_copy: bool = False,
    production_mode: bool = True,
    include_future: bool = True,
    *,
    encode: str = "onehot",
    save_dominance_matrix: bool = False,
    no_calibration_rows: bool = True,
    dominance_log: Optional[str] = None,
) -> pd.DataFrame:
    """Build the full modeling dataset and save to CSV."""
    seasons = list(range(start_season, end_season + 1))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Building dataset for seasons %d-%d", start_season, end_season)
    logging.info("include_future=%s, encode=%s", include_future, encode)

    # PBP cache lookup
    pbp_candidates = [
        Path("pbp_cache.csv"),
        Path("backend/pbp_cache.csv"),
        Path("backend/data/pbp_cache.csv"),
        Path("data/pbp_cache.csv"),
    ]
    pbp_path = next((p for p in pbp_candidates if p.exists()), pbp_candidates[0])
    logging.info("Using PBP cache at %s (exists=%s)", pbp_path, pbp_path.exists())

    team_game_metrics = load_team_game_metrics(pbp_path)
    team_stats = load_team_weekly_stats(seasons)
    player_stats = load_player_game_stats(seasons)
    schedule = load_schedules(seasons, include_future=include_future)

    if schedule is None or schedule.empty:
        logging.error("No schedule data available. Cannot build dataset.")
        return pd.DataFrame()

    logging.info("Loaded %d games from schedule", len(schedule))

    df = add_features(
        sch=schedule,
        windows=(3, 5),
        advanced_metrics=team_game_metrics if not team_game_metrics.empty else None,
    )
    logging.info("Built base features; dataset has %d rows and %d columns", len(df), len(df.columns))

    if not team_stats.empty:
        df = _merge_team_week_stats(df, team_stats, "teamstat")
        logging.info("Merged team weekly stats; now %d columns", len(df.columns))

    if not player_stats.empty:
        df = _merge_team_week_stats(df, player_stats, "player")
        logging.info("Merged player stats; now %d columns", len(df.columns))

    df = create_game_features(df)
    df = create_elo_features(df)
    df = create_rolling_features(df)
    df = create_qb_features(df)
    df = create_target_features(df)

    # Optional encoding for teams
    try:
        if encode == "onehot" and {"home_team", "away_team"}.issubset(df.columns):
            home_ohe = pd.get_dummies(df["home_team"], prefix="home_team")
            away_ohe = pd.get_dummies(df["away_team"], prefix="away_team")
            df = pd.concat([df, home_ohe, away_ohe], axis=1)
            logging.info(
                "Applied one-hot encoding for home_team and away_team (encode=onehot)"
            )
    except Exception:
        logging.exception("Failed to apply one-hot encoding; continuing without encoding")

    # Optional calibration rows
    if not no_calibration_rows:
        try:
            calib_rows = pd.DataFrame([{c: pd.NA for c in df.columns} for _ in range(2)])
            df = pd.concat([df, calib_rows], ignore_index=True)
            logging.info(
                "Appended 2 blank calibration rows to dataset (no_calibration_rows=False)"
            )
        except Exception:
            logging.exception("Failed to append calibration rows; continuing")

    # Optional dominance features (pair table + dominance matrix)
    if save_dominance_matrix or dominance_log:
        try:
            categorical_features = ["home_team", "away_team"]
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            dff = df.copy()
            required_cols = [
                c
                for c in (categorical_features + numeric_features + ["winner", "home_win"])
                if c in dff.columns
            ]
            dff = dff[required_cols]
            result = build_dominance_features(dff)
            pair_table = result.get("pair_table")
            dom_matrix = result.get("dominance_matrix")

            if save_dominance_matrix and dom_matrix is not None:
                dom_path = out_dir / "dominance_matrix.csv"
                dom_matrix.to_csv(dom_path, index=True)
                logging.info("Wrote dominance matrix: %s", dom_path)

            if dominance_log and pair_table is not None:
                try:
                    dlog_path = Path(dominance_log)
                    dlog_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dlog_path, "w", encoding="utf-8") as fh:
                        fh.write("Pairwise dominance table\n")
                        fh.write(pair_table.to_string(index=False))
                    logging.info("Wrote dominance pairwise log: %s", dlog_path)
                except Exception:
                    logging.exception("Failed to write dominance log %s", dominance_log)
        except Exception:
            logging.exception("Failed to compute/export dominance features")

    # Write metadata
    try:
        meta = build_game_feature_metadata(df, include_future=include_future)
        meta_path = out_dir / "game_features_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logging.info("Wrote game_features_metadata.json to %s", meta_path)
    except Exception:
        logging.exception("Failed to write game_features_metadata.json")

    # Save main dataset
    dataset_path = out_dir / OUTPUT_DATASET_NAME
    df.to_csv(dataset_path, index=False)
    logging.info(
        "Dataset build complete: %d rows, %d columns, written to %s",
        len(df),
        len(df.columns),
        dataset_path,
    )

    if legacy_root_copy:
        df.to_csv(OUTPUT_DATASET_NAME, index=False)
        logging.info(
            "Legacy root-level copy created for compatibility: %s",
            OUTPUT_DATASET_NAME,
        )

    # Export abbreviation map for API consistency
    abbr_json_path = out_dir / "team_abbr_map.json"
    with abbr_json_path.open("w", encoding="utf-8") as f:
        json.dump(ABBR_FIX, f, indent=2)
    logging.info("Wrote team_abbr_map.json to %s", abbr_json_path)

    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for season range and output directory."""
    p = argparse.ArgumentParser(
        description="Build NFL game-level dataset (one row per game)."
    )
    p.add_argument("--start", type=int, default=2017, help="Start season (inclusive).")
    p.add_argument("--end", type=int, default=2025, help="End season (inclusive).")
    p.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory (default: ./data).",
    )
    p.add_argument(
        "--legacy-root-copy",
        action="store_true",
        help=f"Also write {OUTPUT_DATASET_NAME} to the repository root for backwards compatibility.",
    )
    p.add_argument(
        "--encode",
        choices=["onehot", "none"],
        default="onehot",
        help="Team encoding mode: onehot (create home_/away_ one-hot cols) or none.",
    )
    p.add_argument(
        "--save-dominance-matrix",
        action="store_true",
        help="Also export dominance_matrix.csv to --out-dir",
    )
    p.add_argument(
        "--no-calibration-rows",
        action="store_true",
        help="Do NOT append the 2 blank calibration rows",
    )
    p.add_argument(
        "--dominance-log",
        type=str,
        default=None,
        help="Optional path to write a textual pairwise dominance table (e.g. /path/to/dom_log.txt)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    setup_logger(out_dir)

    logging.info("=" * 60)
    logging.info("NFL Dataset Builder - PRODUCTION MODE")
    logging.info("Focus: Dataset building only (no model training)")
    logging.info("=" * 60)

    build_dataset(
        start_season=args.start,
        end_season=args.end,
        out_dir=out_dir,
        legacy_root_copy=args.legacy_root_copy,
        production_mode=True,
        include_future=True,
        encode=args.encode,
        save_dominance_matrix=args.save_dominance_matrix,
        no_calibration_rows=args.no_calibration_rows,
        dominance_log=args.dominance_log,
    )


if __name__ == "__main__":
    main()

=======
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: build_csv_datasets.py
Purpose: CLI-driven dataset builder for NFL game-level features; produces one row per game with leak-safe rolling priors, market context, dominance features, and optional encodings.
Functions: parse_args, build_dataset, add_features, load_schedules, load_team_game_metrics, load_player_game_stats, _rolling_prior_stats, _ffill_prior_features, _team_game_long, create_elo_features, create_rolling_features, build_dominance_features, setup_logger, main
Variables: ABBR_FIX (team code map), OUTPUT_DATASET_NAME, NFL_BACKEND (nflreadpy or nfl_data_py), nfl (data loader), HAS_winner_BOOL, TIME_COLS_IN_ORDER
Interacts With: nflreadpy/nfl_data_py (schedules/PBP), backend/data/ (optional pbp_cache.csv), backend/train_models.py (consumes output CSV), backend/data/ (output directory)
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Sequence, Any
from datetime import datetime, timezone

import argparse
import json
import logging
from pathlib import Path
import re
<<<<<<< HEAD
import numbers
=======
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------

ABBR_FIX: Dict[str, str] = {
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
    "WSH": "WAS",
}

# Name of the output CSV file for the generated dataset.
# Includes current date in YYYYMMDD format for traceability/versioning.
OUTPUT_DATASET_NAME = f"game_features_{datetime.now().strftime('%Y%m%d')}.csv"

# Pairwise dominance helpers
HAS_winner_BOOL = True  # if you only have scores, set False
TIME_COLS_IN_ORDER: Optional[Sequence[str]] = None  # auto-detect if None


def make_time_key(df: pd.DataFrame) -> pd.Series:
    """Return sortable integer key YYYYWW from 'season' and 'week' (assumes ints)."""
    season = df["season"].fillna(0).astype(int)
    week = df["week"].fillna(0).astype(int)
    return (season * 100) + week


<<<<<<< HEAD
# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


=======

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
def setup_logger(out_dir: Path) -> None:
    """Initialize both file and console logging so CLI users get progress feedback."""
    
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "build_csv_datasets.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info("Logger initialized → %s", log_file)


<<<<<<< HEAD
# ---------------------------------------------------------------------
# Backend selection (nflreadpy preferred, fallback to nfl_data_py)
# ---------------------------------------------------------------------

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
        import nfl_data_py as _nfl  # type: ignore[no-redef]

        nfl = _nfl
        NFL_BACKEND = "nfl_data_py"
        _note_backend(
            f"Using fallback backend '{NFL_BACKEND}' — {_fallback_reason}",
            logging.WARNING,
        )
except Exception as e:
    _fallback_reason = f"nflreadpy import failed: {e}"
    try:
        import nfl_data_py as _nfl  # type: ignore[no-redef]

        nfl = _nfl
        NFL_BACKEND = "nfl_data_py"
        _note_backend(
            f"Using fallback backend '{NFL_BACKEND}' — {_fallback_reason}",
            logging.WARNING,
        )
    except Exception as e2:
        nfl = None
        _note_backend(f"No NFL backend available: {e2}", logging.ERROR)
=======

# ---------------------------------------------------------------------
# Backend selection (nflreadpy preferred, fallback to nfl_data_py)
# ---------------------------------------------------------------------
NFL_BACKEND: Optional[str] = None
nfl: Optional[str] = None
_fallback_reason = None

def _note_backend(msg: str, level: int = logging.INFO) -> None:
    logging.log(level, msg)

try:
    import nflreadpy as nfl
    
    try:
        _probe = nfl.load_schedules(seasons=[2025])
        print("Probing nflreadpy backend...", _probe, flush=True)
        if hasattr(_probe, "to_pandas"):
            _data = _probe.head(1).to_pandas()
            nfl = "nflreadpy"
            NFL_BACKEND = nfl
        _note_backend(msg="Using backend 'nflreadpy'", level=4) 
    

    except Exception as e:
        _fallback_reason = f"nflreadpy probe failed: {e}"
        import nfl_data_py as n
        if n is not None and nfl is None:
            nfl = n
            NFL_BACKEND = "nfl_data_py"
        _note_backend(
            f"Using fallback backend '{NFL_BACKEND}' — {_fallback_reason}", logging.WARNING,
        )
    finally:
        del _probe
        del _data
       
except Exception as e:
    print(f"Failed to import nflreadpy: {e}", flush=True)

>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
<<<<<<< HEAD


=======
>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
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

    probs.loc[negative] = (-ml_numeric.loc[negative]) / (
        (-ml_numeric.loc[negative]) + 100
    )
    probs.loc[positive & ml_numeric.notna()] = 100 / (
        ml_numeric.loc[positive & ml_numeric.notna()] + 100
    )
    return probs


<<<<<<< HEAD
# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------


def load_team_game_metrics(pbp_path: Path) -> pd.DataFrame:
    """Aggregate play-by-play data to per-team, per-game advanced metrics."""
    # If nflreadpy available, load directly; otherwise use cached CSV
    if nfl is not None and NFL_BACKEND == "nflreadpy":
        try:
            seasons_to_load = list(range(2018, 2025))
            logging.info(
                "Loading play-by-play via nflreadpy for seasons %s", seasons_to_load
            )
            pbp_raw = nfl.load_pbp(seasons=seasons_to_load)
            pbp = to_pandas_safe(pbp_raw)
            logging.info("Loaded %d play-by-play rows from nflreadpy", len(pbp))
        except Exception as exc:
            logging.warning(
                "nflreadpy PBP load failed (%s); falling back to cached CSV", exc
            )
            pbp = None
    else:
        pbp = None

    if pbp is None:
        if not pbp_path.exists():
            logging.warning(
                "Cached PBP missing at %s; advanced PBP features disabled", pbp_path
            )
            return pd.DataFrame(columns=["season", "week", "game_id", "team"])
        pbp = pd.read_csv(pbp_path, low_memory=False)
        logging.info("Loaded %d play-by-play rows from CSV", len(pbp))

=======

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
pbp_cache_path = Path("../pbp_cache.csv")

def load_team_game_metrics(pbp_path: Path) -> pd.DataFrame:
    """Aggregate play-by-play data to per-team, per-game advanced metrics."""
    # If nflreadpy available, load directly; otherwise use cached CSV
    if nfl is not None and NFL_BACKEND == "nflreadpy":
        try:
            seasons_to_load: list[int] =(
                    [int(s) for s in input('Enter seasons to load (comma-separated, e.g., start=2010, end=2022):2010,2022\n').strip().split(',')]
            )

            logging.info(
                "Loading play-by-play via nflreadpy for seasons %s", seasons_to_load
            )
            pbp_raw = nfl.load_pbp(seasons=seasons_to_load)
            pbp = to_pandas_safe(pbp_raw)
            logging.info("Loaded %d play-by-play rows from nflreadpy", len(pbp))
            
        except Exception as exc:
                logging.warning(
                    "nflreadpy PBP load failed (%s); falling back to cached CSV", exc
                )
                if len(pbp.columns) == 0:
                    raise ValueError("No columns loaded from nflreadpy PBP data")
    else:
        if pbp is None:
            if not pbp_path.exists():
                logging.warning(
                    "Cached PBP missing at %s; advanced PBP features disabled", pbp_path
                )
                return pd.DataFrame(columns=["season", "week", "game_id", "team"])
            pbp = pd.read_csv(pbp_path, low_memory=False)
            logging.info("Loaded %d play-by-play rows from CSV", len(pbp))

>>>>>>> cd97fecacdc0a2f3d4ee6cd29effaa9619489d75
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

    for col in [
        "epa",
        "success",
        "pass",
        "xpass",
        "pass_attempt",
        "rush_attempt",
        "third_down_converted",
        "third_down_failed",
        "interception",
        "fumble_lost",
        "yards_gained",
    ]:
        if col in pbp.columns:
            pbp[col] = pbp[col].fillna(0.0)
        else:
            pbp[col] = 0.0

    pbp["turnover"] = pbp["interception"] + pbp["fumble_lost"]
    pbp["explosive_play"] = ((pbp["pass"] == 1.0) & (pbp["yards_gained"] >= 20)) | (
        (pbp["rush_attempt"] == 1.0) & (pbp["yards_gained"] >= 15)
    )

    off_group = ["season", "week", "game_id", "posteam"]
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
    total_def_plays = (
        def_agg["def_pass_attempts_faced"] + def_agg["def_rush_attempts_faced"]
    )
    def_agg["def_takeaway_rate"] = np.where(
        total_def_plays > 0,
        def_agg["def_takeaways"] / total_def_plays,
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

    return metrics.reset_index(drop=True)


def load_player_game_stats(seasons: List[int]) -> pd.DataFrame:
    """Load weekly player stats and aggregate to team-week level."""
    if nfl is None:
        logging.warning("No NFL backend available; player features disabled")
        return pd.DataFrame(columns=["season", "week", "team"])

    try:
        logging.info("Loading player stats via %s for seasons %s", NFL_BACKEND, seasons)
        player_stats: Optional[pd.DataFrame] = None

        if NFL_BACKEND == "nflreadpy":
            loader = getattr(nfl, "load_player_stats", None)
            if loader:
                player_stats = to_pandas_safe(
                    loader(seasons=seasons, summary_level="week")
                )
        else:
            loader = getattr(nfl, "import_weekly_data", None)
            if loader:
                player_stats = loader(seasons)

        if player_stats is None or player_stats.empty:
            logging.warning("Player stats unavailable; skipping.")
            return pd.DataFrame(columns=["season", "week", "team"])

        team_col = "recent_team" if "recent_team" in player_stats.columns else (
            "team" if "team" in player_stats.columns else None
        )
        if not team_col:
            logging.warning("Cannot determine team column in player_stats; skipping")
            return pd.DataFrame(columns=["season", "week", "team"])

        # QB aggregation
        if "position" in player_stats.columns:
            qb = player_stats[player_stats["position"] == "QB"].copy()
        else:
            qb = pd.DataFrame()

        qb_cols = {
            "passing_yards": "team_qb_pass_yards",
            "passing_tds": "team_qb_pass_tds",
            "interceptions": "team_qb_interceptions",
            "sacks": "team_qb_sacks",
            "completions": "team_qb_completions",
            "attempts": "team_qb_attempts",
        }

        qb_agg_dict = {
            tgt: (src, "sum") for src, tgt in qb_cols.items() if src in qb.columns
        }
        if qb_agg_dict and not qb.empty:
            qb_agg = qb.groupby(["season", "week", team_col], as_index=False).agg(
                **qb_agg_dict
            )
            if {
                "team_qb_completions",
                "team_qb_attempts",
            }.issubset(qb_agg.columns):
                qb_agg["team_qb_completion_pct"] = np.where(
                    qb_agg["team_qb_attempts"] > 0,
                    qb_agg["team_qb_completions"] / qb_agg["team_qb_attempts"],
                    np.nan,
                )
        else:
            qb_agg = pd.DataFrame(columns=["season", "week", team_col])

        # RB aggregation
        if "position" in player_stats.columns:
            rb = player_stats[player_stats["position"].isin(["RB"])].copy()
        else:
            rb = pd.DataFrame()

        rb_cols = {
            "rushing_yards": "team_rb_rush_yards",
            "rushing_tds": "team_rb_rush_tds",
            "receptions": "team_rb_receptions",
            "receiving_yards": "team_rb_receiving_yards",
        }
        rb_agg_dict = {
            tgt: (src, "sum") for src, tgt in rb_cols.items() if src in rb.columns
        }
        if rb_agg_dict and not rb.empty:
            rb_agg = rb.groupby(["season", "week", team_col], as_index=False).agg(
                **rb_agg_dict
            )
        else:
            rb_agg = pd.DataFrame(columns=["season", "week", team_col])

        # WR+TE aggregation
        if "position" in player_stats.columns:
            pass_catchers = player_stats[
                player_stats["position"].isin(["WR", "TE"])
            ].copy()
        else:
            pass_catchers = pd.DataFrame()

        wr_cols = {
            "targets": "team_wr_targets",
            "receptions": "team_wr_receptions",
            "receiving_yards": "team_wr_receiving_yards",
            "receiving_tds": "team_wr_receiving_tds",
        }
        wr_agg_dict = {
            tgt: (src, "sum")
            for src, tgt in wr_cols.items()
            if src in pass_catchers.columns
        }
        if wr_agg_dict and not pass_catchers.empty:
            wr_agg = pass_catchers.groupby(
                ["season", "week", team_col], as_index=False
            ).agg(**wr_agg_dict)
        else:
            wr_agg = pd.DataFrame(columns=["season", "week", team_col])

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
    """Load team-level weekly stats when available."""
    if nfl is None:
        logging.warning("No NFL backend; team-week stats disabled")
        return pd.DataFrame(columns=["season", "week", "team"])

    try:
        logging.info(
            "Loading team weekly stats via %s for seasons %s", NFL_BACKEND, seasons
        )
        if NFL_BACKEND == "nflreadpy":
            loader = getattr(nfl, "load_team_stats", None)
            if loader:
                team_stats = to_pandas_safe(
                    loader(seasons=seasons, summary_level="week")
                )
            else:
                team_stats = None
        else:
            loader = getattr(nfl, "import_team_desc", None)
            if loader:
                # nfl_data_py doesn't have a direct weekly team stats equivalent;
                # if unavailable, just skip.
                team_stats = None
            else:
                team_stats = None

        if team_stats is None or team_stats.empty:
            logging.warning("Team weekly stats unavailable; skipping")
            return pd.DataFrame(columns=["season", "week", "team"])

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
        ]
        available = [c for c in feature_cols if c in team_stats.columns]
        return team_stats[available].fillna(0).reset_index(drop=True)

    except Exception as exc:
        logging.warning("Team stats load failed (%s); team features disabled", exc)
        return pd.DataFrame(columns=["season", "week", "team"])


def load_schedules(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
    """Load schedules + final scores for given seasons."""
    sch: Optional[pd.DataFrame] = None

    if nfl is not None:
        try:
            if NFL_BACKEND == "nflreadpy":
                loader = getattr(nfl, "load_schedules", None)
                if loader:
                    sch = to_pandas_safe(loader(seasons=seasons))
            else:
                loader = getattr(nfl, "import_schedules", None)
                if loader:
                    sch = loader(seasons)
        except Exception as exc:
            logging.warning("Schedule load failed via backend (%s)", exc)
            sch = None

    if sch is None or sch.empty:
        raise RuntimeError("Could not load schedules from any backend.")

    need = [
        "season",
        "week",
        "game_id",
        "gameday",
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
    missing = [c for c in need if c not in sch.columns]
    if missing:
        raise RuntimeError(f"Missing schedule columns: {missing}")

    sch = _normalize_codes(sch, ["home_team", "away_team"])
    sch = sch.rename(columns={"gameday": "game_date"})
    sch["week"] = sch["week"].astype(int)

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

    if include_future:
        completed = sch.dropna(subset=["home_score", "away_score"]).reset_index(
            drop=True
        )
        future = sch[
            sch["home_score"].isna() | sch["away_score"].isna()
        ].copy()
        future["home_score"] = None
        future["away_score"] = None
        future = future[future["game_type"] == "REG"].reset_index(drop=True)

        logging.info(
            "Loaded %d completed games + %d future games", len(completed), len(future)
        )
        return pd.concat([completed, future], ignore_index=True)

    sch = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    logging.info("Schedules loaded: %d completed games", len(sch))
    return sch


# ---------------------------------------------------------------------
# Core feature building
# ---------------------------------------------------------------------


def _team_game_long(sch: pd.DataFrame) -> pd.DataFrame:
    """Convert per-game schedule to per-team per-game long format."""
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

    completed_mask = long["points_for"].notna() & long["points_against"].notna()
    long["win"] = np.where(
        completed_mask, (long["points_for"] > long["points_against"]).astype(float), np.nan
    )

    long["time_key"] = make_time_key(long)
    return long.sort_values(["team", "time_key", "game_id"]).reset_index(drop=True)


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
            if col not in df.columns:
                continue
            df[f"prior_{col}_{window}"] = safe_rolling_mean(df[col])

    return df


def add_features(
    sch: pd.DataFrame,
    windows: Tuple[int, ...] = (3, 5),
    advanced_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build one-row-per-game features by:
      1) creating per-team rows,
      2) computing prior rolling stats,
      3) re-pivoting to wide with home_/away_ prefixes,
      4) adding market & rest features.
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
            right=advanced_metrics, on=["season", "week", "game_id", "team"], how="left"
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

    wide = home_side.merge(
        away_side,
        left_on="home_game_id",
        right_on="away_game_id",
        how="inner",
    )
    wide = wide.rename(columns={"home_game_id": "game_id"}).drop(
        columns=["away_game_id"]
    )

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

    wide["season"] = wide["home_season"].astype(int)
    wide["week"] = wide["home_week"].astype(int)
    wide["time_key"] = make_time_key(wide)
    wide = wide.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    prior_pairs = [c for c in wide.columns if c.startswith("home_prior_")]
    for home_col in prior_pairs:
        suffix = home_col[len("home_prior_") :]
        away_col = f"away_prior_{suffix}"
        if away_col in wide.columns:
            wide[f"home_minus_away_{suffix}"] = wide[home_col] - wide[away_col]

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
    prior_feature_cols = [
        c for c in wide.columns if c.startswith(("home_prior_", "away_prior_"))
    ]
    diff_feature_cols = [c for c in wide.columns if c.startswith("home_minus_away_")]
    final_cols = ordered_cols + prior_feature_cols + diff_feature_cols

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
    wide = wide.merge(schedule_extras, on="game_id", how="left")

    wide["home_moneyline"] = pd.to_numeric(wide["home_moneyline"], errors="coerce")
    wide["away_moneyline"] = pd.to_numeric(wide["away_moneyline"], errors="coerce")
    wide["home_moneyline_prob"] = _moneyline_to_prob(wide["home_moneyline"])
    wide["away_moneyline_prob"] = _moneyline_to_prob(wide["away_moneyline"])
    wide["moneyline_prob_diff"] = (
        wide["home_moneyline_prob"] - wide["away_moneyline_prob"]
    )

    wide["spread_line"] = pd.to_numeric(wide["spread_line"], errors="coerce")
    wide["total_line"] = pd.to_numeric(wide["total_line"], errors="coerce")
    wide["home_rest"] = pd.to_numeric(wide["home_rest"], errors="coerce")
    wide["away_rest"] = pd.to_numeric(wide["away_rest"], errors="coerce")
    wide["rest_diff"] = wide["home_rest"] - wide["away_rest"]

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

    # Ensure we only select existing columns
    final_cols = [c for c in final_cols if c in wide.columns]

    return wide[final_cols].reset_index(drop=True)


def _merge_team_week_stats(
    game_df: pd.DataFrame, team_week_stats: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    """
    Merge team-week level stats into game-level dataframe for both home and away teams.
    """
    if team_week_stats is None or team_week_stats.empty:
        return game_df

    stat_cols = [c for c in team_week_stats.columns if c not in {"season", "week", "team"}]

    home_stats = team_week_stats.copy()
    home_stats.columns = ["season", "week", "home_team"] + [
        f"home_{prefix}_{c}" for c in stat_cols
    ]
    game_df = game_df.merge(home_stats, on=["season", "week", "home_team"], how="left")

    away_stats = team_week_stats.copy()
    away_stats.columns = ["season", "week", "away_team"] + [
        f"away_{prefix}_{c}" for c in stat_cols
    ]
    game_df = game_df.merge(away_stats, on=["season", "week", "away_team"], how="left")

    return game_df


# ---------------------------------------------------------------------
# Simple regression pipeline + time split (kept for reuse)
# ---------------------------------------------------------------------


def build_regression_pipeline(
    numeric_features: List[str], categorical_features: List[str], alpha: float = 1.0
) -> Pipeline:
    """Simple numeric+categorical -> Ridge regression pipeline."""
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

    model = Ridge(alpha=alpha)
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    return pipeline


def ts_split_by_season_week(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    train_end: Tuple[int, int],
):
    """Chronological split that prevents leakage; returns (X_train, y_train), df, sorted_df."""
    data = df.copy()
    required_cols = {"season", "week", *features, target}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["time_key"] = make_time_key(data)
    data = data.sort_values(["time_key"]).reset_index(drop=True)

    train_end_season, train_end_week = train_end
    is_train = (data["season"] < train_end_season) | (
        (data["season"] == train_end_season) & (data["week"] <= train_end_week)
    )

    train_df = data.loc[is_train]
    X_train, y_train = train_df[features], train_df[target]
    return (X_train, y_train), df, data


# ---------------------------------------------------------------------
# Dominance / pairwise priors
# ---------------------------------------------------------------------


def ensure_actual_winner(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'home_win' (bool) and 'actual_winner' (team name) columns exist."""
    df = df.copy()

    if "home_win" in df.columns:
        win_series = pd.Series(df["home_win"], index=df.index, dtype="boolean")
    else:
        if not HAS_winner_BOOL:
            if {"home_points_for", "away_points_for"}.issubset(df.columns):
                win_series = pd.Series(
                    df["home_points_for"] > df["away_points_for"],
                    index=df.index,
                    dtype="boolean",
                )
            else:
                raise ValueError(
                    "Need either 'home_win' bool or score columns "
                    "'home_points_for'/'away_points_for'."
                )
        else:
            if "winner" not in df.columns:
                raise ValueError("Need 'winner' column when HAS_winner_BOOL=True.")
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


def detect_time_cols(df: pd.DataFrame) -> List[str]:
    if TIME_COLS_IN_ORDER is not None:
        return list(TIME_COLS_IN_ORDER)
    if "game_date" in df.columns:
        return ["game_date"]
    if {"season", "week"}.issubset(df.columns):
        return ["season", "week"]
    return []


def make_long_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a 'long' dataframe with two rows per game:
      - (team = home, opponent = away, team_won = winner)
      - (team = away, opponent = home, team_won = not winner)
    Includes 'gid' (row index of original df) to merge features back later.
    """
    base = df.reset_index(drop=False).rename(columns={"index": "gid"}).copy()

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

    home_rows = base[["gid", "home_team", "away_team"]].copy()
    home_rows = home_rows.rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    home_rows["team_won"] = win_series.reindex(home_rows.index)
    home_rows = home_rows[["gid", "team", "opponent", "team_won"]]

    away_rows = base[["gid", "home_team", "away_team"]].copy()
    away_rows = away_rows.rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    away_rows["team_won"] = (~win_series).reindex(away_rows.index)
    away_rows = away_rows[["gid", "team", "opponent", "team_won"]]

    long = pd.concat([home_rows, away_rows], ignore_index=True)

    time_cols = detect_time_cols(df)
    if time_cols:
        long = long.merge(base[["gid"] + time_cols], on="gid", how="left")
    return long


def pairwise_table(long: pd.DataFrame) -> pd.DataFrame:
    """Static, all-time pairwise records (order matters: (team, opponent))."""
    g = long.groupby(["team", "opponent"], as_index=False, observed=True)
    agg = g.agg(games=("team", "size"), wins=("team_won", "sum"))
    agg["losses"] = agg["games"] - agg["wins"]
    agg["win_pct"] = np.where(agg["games"] > 0, agg["wins"] / agg["games"], np.nan)
    agg["dominance"] = agg["wins"] - agg["losses"]
    agg["signed_wins_style"] = agg["wins"]
    return agg


def leak_free_pregame_features(long: pd.DataFrame) -> pd.DataFrame:
    """
    For each (team, opponent), compute *prior-to-game* counts:
      prior_games, prior_wins, prior_losses, prior_dominance, prior_win_pct
    """
    time_cols = [c for c in long.columns if c in ("game_date", "season", "week")]
    sort_cols = time_cols + ["gid"] if time_cols else ["gid"]
    long = long.sort_values(sort_cols).copy()

    grp = long.groupby(["team", "opponent"], group_keys=False, observed=True)

    long["prior_games"] = grp.cumcount()
    long["prior_wins"] = grp["team_won"].cumsum() - long["team_won"]
    long["prior_losses"] = long["prior_games"] - long["prior_wins"]
    long["prior_dom"] = long["prior_wins"] - long["prior_losses"]
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


def attach_pregame_to_wide(df: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-game features back to the original wide df."""
    df = df.reset_index(drop=False).rename(columns={"index": "gid"}).copy()

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


def team_level_prediction_accuracy(df: pd.DataFrame) -> pd.Series:
    """
    Accuracy of your classifier *when a given team plays*.

    Expects:
      - df['prob_winner'] in [0,1] (probability home team wins)
      - df['home_win'] boolean
      - df['home_team'], df['away_team']
    """
    required = {"prob_winner", "home_win", "home_team", "away_team"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns for team accuracy: {required - set(df.columns)}")

    df = df.copy()
    df["pred_winner"] = np.where(
        df["prob_winner"] >= 0.5, df["home_team"], df["away_team"]
    )

    actual = pd.Series(pd.NA, index=df.index, dtype="string")
    actual.loc[df["home_win"] == True] = df.loc[df["home_win"] == True, "home_team"].astype("string")
    actual.loc[df["home_win"] == False] = df.loc[df["home_win"] == False, "away_team"].astype("string")
    df["actual_winner"] = actual

    df["pred_correct"] = (df["pred_winner"] == df["actual_winner"]).astype(int)

    home_part = df[["home_team", "pred_correct"]].rename(columns={"home_team": "team"})
    away_part = df[["away_team", "pred_correct"]].rename(columns={"away_team": "team"})
    team_games = pd.concat([home_part, away_part], ignore_index=True)

    acc = team_games.groupby("team")["pred_correct"].mean().sort_values(ascending=False)
    return acc


def build_dominance_features(dff: pd.DataFrame) -> Dict[str, Any]:
    """Compute pairwise dominance tables and leak-free pregame dominance features."""
    df = ensure_actual_winner(dff)
    long = make_long_edges(df)
    pair_table = pairwise_table(long)
    pre = leak_free_pregame_features(long)
    df_with_pregame = attach_pregame_to_wide(df, pre)

    dom_matrix = pair_table.pivot(
        index="team", columns="opponent", values="dominance"
    ).fillna(0).astype(int)

    team_acc: Optional[pd.Series] = None
    if "prob_winner" in df_with_pregame.columns:
        try:
            team_acc = team_level_prediction_accuracy(df_with_pregame)
        except Exception as exc:
            logging.warning("Failed to compute team prediction accuracy: %s", exc)

    return {
        "pair_table": pair_table.sort_values(["team", "opponent"]).reset_index(
            drop=True
        ),
        "dominance_matrix": dom_matrix,
        "df_with_pregame": df_with_pregame,
        "team_prediction_accuracy": team_acc,
    }


# ---------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------


def build_game_feature_metadata(
    df: pd.DataFrame,
    *,
    include_future: bool,
) -> Dict[str, Any]:
    """Inspect the final dataframe and build a structured metadata dict."""
    meta: Dict[str, Any] = {}

    meta["build_timestamp_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    meta["rows_total"] = int(df.shape[0])
    meta["n_features_total"] = int(df.shape[1])
    meta["include_future_games"] = bool(include_future)

    if "season" in df.columns:
        meta["season_min"] = int(df["season"].min())
        meta["season_max"] = int(df["season"].max())

    id_cols = [
        c
        for c in (
            "season",
            "week",
            "game_id",
            "game_date",
            "home_game_date",
            "home_team",
            "away_team",
        )
        if c in df.columns
    ]
    target_cols = [
        c
        for c in (
            "home_points_for",
            "away_points_for",
            "home_win",
            "winner",
            "point_diff",
        )
        if c in df.columns
    ]
    prior_cols = sorted(
        c for c in df.columns if c.startswith(("home_prior_", "away_prior_"))
    )
    prior_diff_cols = sorted(
        c for c in df.columns if c.startswith("home_minus_away_")
    )
    market_cols = sorted(
        c
        for c in df.columns
        if c
        in {
            "home_moneyline_prob",
            "away_moneyline_prob",
            "moneyline_prob_diff",
            "spread_line",
            "total_line",
        }
    )
    rest_cols = sorted(c for c in df.columns if "rest" in c)
    dominance_cols = sorted(
        c for c in df.columns if c.startswith(("home_vs_away_", "away_vs_home_"))
    )

    meta["columns"] = {
        "id": id_cols,
        "target": target_cols,
        "priors": prior_cols,
        "priors_diff": prior_diff_cols,
        "market": market_cols,
        "rest": rest_cols,
        "dominance": dominance_cols,
    }

    windows = set()
    window_pattern = re.compile(r"_(\d+)$")
    for col in prior_cols:
        m = window_pattern.search(col)
        if m:
            try:
                windows.add(int(m.group(1)))
            except ValueError:
                pass

    meta["priors_config"] = {
        "windows": sorted(windows),
    }

    has_scores = {"home_points_for", "away_points_for"}.issubset(df.columns)
    if has_scores:
        train_mask = df["home_points_for"].notna() & df["away_points_for"].notna()
    else:
        train_mask = pd.Series(True, index=df.index)

    if prior_cols:
        train_slice = df.loc[train_mask, prior_cols]
        priors_median = train_slice.median(numeric_only=True)
        priors_median_json = {
            k: (None if pd.isna(v) else float(v)) for k, v in priors_median.items()
        }
    else:
        priors_median_json = {}

    meta["priors_imputation"] = {
        "strategy": "median_over_training_rows_with_history",
        "training_rows": int(train_mask.sum()),
        "baseline_medians": priors_median_json,
    }

    return meta


# ---------------------------------------------------------------------
# Additional feature engineering helpers
# ---------------------------------------------------------------------


def create_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ELO rating features for teams with proper initialization and updates."""
    out = df.copy()

    required = {"season", "week", "home_team", "away_team", "game_id"}
    if not required.issubset(out.columns):
        logging.warning("Missing required columns for ELO features; skipping")
        return out

    elo_ratings: Dict[str, float] = {}
    K_FACTOR = 32.0

    def get_elo(team: str, default: float = 1500.0) -> float:
        return float(elo_ratings.get(team, default))

    def expected_score(elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def update_elo(team: str, actual: float, expected: float) -> float:
        current = get_elo(team)
        new_elo = current + K_FACTOR * (actual - expected)
        elo_ratings[team] = new_elo
        return new_elo

    for col in [
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff_pre",
        "home_elo_post",
        "away_elo_post",
    ]:
        out[col] = np.nan

    if "time_key" not in out.columns:
        out["time_key"] = make_time_key(out)

    out = out.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    for idx, row in out.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_elo_pre = get_elo(home_team)
        away_elo_pre = get_elo(away_team)

        out.at[idx, "home_elo_pre"] = home_elo_pre
        out.at[idx, "away_elo_pre"] = away_elo_pre
        out.at[idx, "elo_diff_pre"] = home_elo_pre - away_elo_pre

        home_score = row.get("home_points_for")
        away_score = row.get("away_points_for")

        if pd.notna(home_score) and pd.notna(away_score):
            if home_score > away_score:
                home_actual, away_actual = 1.0, 0.0
            elif home_score < away_score:
                home_actual, away_actual = 0.0, 1.0
            else:
                home_actual, away_actual = 0.5, 0.5

            home_expected = expected_score(home_elo_pre, away_elo_pre)
            away_expected = 1.0 - home_expected

            home_elo_post = update_elo(home_team, home_actual, home_expected)
            away_elo_post = update_elo(away_team, away_actual, away_expected)

            out.at[idx, "home_elo_post"] = home_elo_post
            out.at[idx, "away_elo_post"] = away_elo_post
        else:
            out.at[idx, "home_elo_post"] = home_elo_pre
            out.at[idx, "away_elo_post"] = away_elo_pre

    logging.info("Created ELO features for %d teams", len(elo_ratings))
    return out


def create_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic game-level features from schedule data."""
    out = df.copy()

    if "game_date" in out.columns:
        out["game_date_parsed"] = pd.to_datetime(out["game_date"], errors="coerce")
        out["game_day_of_week"] = out["game_date_parsed"].dt.dayofweek
        out["is_weekend"] = out["game_day_of_week"].isin([5, 6])

    if {"home_team", "away_team"}.issubset(out.columns):
        out["has_home_team"] = out["home_team"].notna() & out["away_team"].notna()

    if (
        "home_rest" in out.columns
        and "away_rest" in out.columns
        and "rest_diff" not in out.columns
    ):
        out["rest_diff"] = (
            pd.to_numeric(out["home_rest"], errors="coerce")
            - pd.to_numeric(out["away_rest"], errors="coerce")
        )

    if "game_type" in out.columns:
        out["is_regular_season"] = out["game_type"] == "REG"
        out["is_playoff"] = out["game_type"].isin(["WC", "DIV", "CON", "SB"])

    logging.info("Created basic game features")
    return out


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling window statistics for teams (points + win%)."""
    out = df.copy()

    if not {"season", "week", "home_team", "away_team", "game_id"}.issubset(out.columns):
        logging.warning("Missing required columns for rolling features; skipping")
        return out

    if "time_key" not in out.columns:
        out["time_key"] = make_time_key(out)
    out = out.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    home_games = out[
        [
            "season",
            "week",
            "game_id",
            "time_key",
            "home_team",
            "away_team",
            "home_points_for",
            "away_points_for",
        ]
    ].copy()
    home_games = home_games.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_points_for": "points_for",
            "away_points_for": "points_against",
        }
    )
    home_games["is_home"] = 1

    away_games = out[
        [
            "season",
            "week",
            "game_id",
            "time_key",
            "home_team",
            "away_team",
            "home_points_for",
            "away_points_for",
        ]
    ].copy()
    away_games = away_games.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_points_for": "points_for",
            "home_points_for": "points_against",
        }
    )
    away_games["is_home"] = 0

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(
        ["team", "time_key", "game_id"]
    ).reset_index(drop=True)

    completed = team_games["points_for"].notna() & team_games["points_against"].notna()
    team_games["win"] = np.where(
        completed, (team_games["points_for"] > team_games["points_against"]).astype(float), np.nan
    )

    def safe_rolling(series: pd.Series, window: int) -> pd.Series:
        return series.shift(1).rolling(window=window, min_periods=1).mean()

    for window in [3, 5, 10]:
        grouped = team_games.groupby("team", group_keys=False)
        team_games[f"rolling_pf_{window}"] = grouped["points_for"].apply(
            lambda x: safe_rolling(x, window)
        )
        team_games[f"rolling_pa_{window}"] = grouped["points_against"].apply(
            lambda x: safe_rolling(x, window)
        )
        team_games[f"rolling_win_pct_{window}"] = grouped["win"].apply(
            lambda x: safe_rolling(x, window)
        )

    roll_cols = [c for c in team_games.columns if c.startswith("rolling_")]

    home_rolling = team_games[team_games["is_home"] == 1][
        ["game_id", "team"] + roll_cols
    ].copy()
    home_rolling.columns = ["game_id", "home_team"] + [f"home_{c}" for c in roll_cols]

    away_rolling = team_games[team_games["is_home"] == 0][
        ["game_id", "team"] + roll_cols
    ].copy()
    away_rolling.columns = ["game_id", "away_team"] + [f"away_{c}" for c in roll_cols]

    out = out.merge(home_rolling, on=["game_id", "home_team"], how="left")
    out = out.merge(away_rolling, on=["game_id", "away_team"], how="left")

    logging.info("Created rolling window features (windows: 3, 5, 10)")
    return out


def create_qb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create quarterback-specific features if QB data is available."""
    out = df.copy()
    qb_cols = [c for c in out.columns if "qb" in c.lower()]
    if not qb_cols:
        logging.info("No QB data available; skipping QB-specific features")
        return out

    # These names only exist if you merged player stats in a compatible way.
    if {
        "home_player_team_qb_attempts",
        "home_player_team_qb_completions",
    }.issubset(out.columns):
        out["home_qb_completion_pct"] = np.where(
            out["home_player_team_qb_attempts"] > 0,
            out["home_player_team_qb_completions"]
            / out["home_player_team_qb_attempts"],
            np.nan,
        )
    if {
        "away_player_team_qb_attempts",
        "away_player_team_qb_completions",
    }.issubset(out.columns):
        out["away_qb_completion_pct"] = np.where(
            out["away_player_team_qb_attempts"] > 0,
            out["away_player_team_qb_completions"]
            / out["away_player_team_qb_attempts"],
            np.nan,
        )

    logging.info("Created QB-derived features from available stats (if any)")
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
        winner.loc[home_win == True] = out.loc[
            home_win == True, "home_team"
        ].astype("string")
        winner.loc[home_win == False] = out.loc[
            home_win == False, "away_team"
        ].astype("string")
        tie_mask = win_mask & (pf == pa)
        winner.loc[tie_mask] = "TIE"
        out["winner"] = winner
    else:
        if "point_diff" not in out.columns:
            out["point_diff"] = np.nan
        if "home_win" not in out.columns:
            out["home_win"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        if "winner" not in out.columns:
            out["winner"] = pd.Series(pd.NA, index=out.index, dtype="string")
    return out


# ---------------------------------------------------------------------
# Dataset builder (canonical)
# ---------------------------------------------------------------------


def build_dataset(
    start_season: int,
    end_season: int,
    out_dir: Path,
    legacy_root_copy: bool = False,
    production_mode: bool = True,
    include_future: bool = True,
    *,
    encode: str = "onehot",
    save_dominance_matrix: bool = False,
    no_calibration_rows: bool = True,
    dominance_log: Optional[str] = None,
) -> pd.DataFrame:
    """Build the full modeling dataset and save to CSV."""
    seasons = list(range(start_season, end_season + 1))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Building dataset for seasons %d-%d", start_season, end_season)
    logging.info("include_future=%s, encode=%s", include_future, encode)

    # PBP cache lookup
    pbp_candidates = [
        Path("pbp_cache.csv"),
        Path("backend/pbp_cache.csv"),
        Path("backend/data/pbp_cache.csv"),
        Path("data/pbp_cache.csv"),
    ]
    pbp_path = next((p for p in pbp_candidates if p.exists()), pbp_candidates[0])
    logging.info("Using PBP cache at %s (exists=%s)", pbp_path, pbp_path.exists())

    team_game_metrics = load_team_game_metrics(pbp_path)
    team_stats = load_team_weekly_stats(seasons)
    player_stats = load_player_game_stats(seasons)
    schedule = load_schedules(seasons, include_future=include_future)

    if schedule is None or schedule.empty:
        logging.error("No schedule data available. Cannot build dataset.")
        return pd.DataFrame()

    logging.info("Loaded %d games from schedule", len(schedule))

    df = add_features(
        sch=schedule,
        windows=(3, 5),
        advanced_metrics=team_game_metrics if not team_game_metrics.empty else None,
    )
    logging.info("Built base features; dataset has %d rows and %d columns", len(df), len(df.columns))

    if not team_stats.empty:
        df = _merge_team_week_stats(df, team_stats, "teamstat")
        logging.info("Merged team weekly stats; now %d columns", len(df.columns))

    if not player_stats.empty:
        df = _merge_team_week_stats(df, player_stats, "player")
        logging.info("Merged player stats; now %d columns", len(df.columns))

    df = create_game_features(df)
    df = create_elo_features(df)
    df = create_rolling_features(df)
    df = create_qb_features(df)
    df = create_target_features(df)

    # Optional encoding for teams
    try:
        if encode == "onehot" and {"home_team", "away_team"}.issubset(df.columns):
            home_ohe = pd.get_dummies(df["home_team"], prefix="home_team")
            away_ohe = pd.get_dummies(df["away_team"], prefix="away_team")
            df = pd.concat([df, home_ohe, away_ohe], axis=1)
            logging.info(
                "Applied one-hot encoding for home_team and away_team (encode=onehot)"
            )
    except Exception:
        logging.exception("Failed to apply one-hot encoding; continuing without encoding")

    # Optional calibration rows
    if not no_calibration_rows:
        try:
            calib_rows = pd.DataFrame([{c: pd.NA for c in df.columns} for _ in range(2)])
            df = pd.concat([df, calib_rows], ignore_index=True)
            logging.info(
                "Appended 2 blank calibration rows to dataset (no_calibration_rows=False)"
            )
        except Exception:
            logging.exception("Failed to append calibration rows; continuing")

    # Optional dominance features (pair table + dominance matrix)
    if save_dominance_matrix or dominance_log:
        try:
            categorical_features = ["home_team", "away_team"]
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            dff = df.copy()
            required_cols = [
                c
                for c in (categorical_features + numeric_features + ["winner", "home_win"])
                if c in dff.columns
            ]
            dff = dff[required_cols]
            result = build_dominance_features(dff)
            pair_table = result.get("pair_table")
            dom_matrix = result.get("dominance_matrix")

            if save_dominance_matrix and dom_matrix is not None:
                dom_path = out_dir / "dominance_matrix.csv"
                dom_matrix.to_csv(dom_path, index=True)
                logging.info("Wrote dominance matrix: %s", dom_path)

            if dominance_log and pair_table is not None:
                try:
                    dlog_path = Path(dominance_log)
                    dlog_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dlog_path, "w", encoding="utf-8") as fh:
                        fh.write("Pairwise dominance table\n")
                        fh.write(pair_table.to_string(index=False))
                    logging.info("Wrote dominance pairwise log: %s", dlog_path)
                except Exception:
                    logging.exception("Failed to write dominance log %s", dominance_log)
        except Exception:
            logging.exception("Failed to compute/export dominance features")

    # Write metadata
    try:
        meta = build_game_feature_metadata(df, include_future=include_future)
        meta_path = out_dir / "game_features_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logging.info("Wrote game_features_metadata.json to %s", meta_path)
    except Exception:
        logging.exception("Failed to write game_features_metadata.json")

    # Save main dataset
    dataset_path = out_dir / OUTPUT_DATASET_NAME
    df.to_csv(dataset_path, index=False)
    logging.info(
        "Dataset build complete: %d rows, %d columns, written to %s",
        len(df),
        len(df.columns),
        dataset_path,
    )

    if legacy_root_copy:
        df.to_csv(OUTPUT_DATASET_NAME, index=False)
        logging.info(
            "Legacy root-level copy created for compatibility: %s",
            OUTPUT_DATASET_NAME,
        )

    # Export abbreviation map for API consistency
    abbr_json_path = out_dir / "team_abbr_map.json"
    with abbr_json_path.open("w", encoding="utf-8") as f:
        json.dump(ABBR_FIX, f, indent=2)
    logging.info("Wrote team_abbr_map.json to %s", abbr_json_path)

    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for season range and output directory."""
    p = argparse.ArgumentParser(
        description="Build NFL game-level dataset (one row per game)."
    )
    p.add_argument("--start", type=int, default=2017, help="Start season (inclusive).")
    p.add_argument("--end", type=int, default=2025, help="End season (inclusive).")
    p.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory (default: ./data).",
    )
    p.add_argument(
        "--legacy-root-copy",
        action="store_true",
        help=f"Also write {OUTPUT_DATASET_NAME} to the repository root for backwards compatibility.",
    )
    p.add_argument(
        "--encode",
        choices=["onehot", "none"],
        default="onehot",
        help="Team encoding mode: onehot (create home_/away_ one-hot cols) or none.",
    )
    p.add_argument(
        "--save-dominance-matrix",
        action="store_true",
        help="Also export dominance_matrix.csv to --out-dir",
    )
    p.add_argument(
        "--no-calibration-rows",
        action="store_true",
        help="Do NOT append the 2 blank calibration rows",
    )
    p.add_argument(
        "--dominance-log",
        type=str,
        default=None,
        help="Optional path to write a textual pairwise dominance table (e.g. /path/to/dom_log.txt)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    setup_logger(out_dir)

    logging.info("=" * 60)
    logging.info("NFL Dataset Builder - PRODUCTION MODE")
    logging.info("Focus: Dataset building only (no model training)")
    logging.info("=" * 60)

    build_dataset(
        start_season=args.start,
        end_season=args.end,
        out_dir=out_dir,
        legacy_root_copy=args.legacy_root_copy,
        production_mode=True,
        include_future=True,
        encode=args.encode,
        save_dominance_matrix=args.save_dominance_matrix,
        no_calibration_rows=args.no_calibration_rows,
        dominance_log=args.dominance_log,
    )


if __name__ == "__main__":
    main()

>>>>>>> e908bfb18 (Prod/nfl ml vercel (#67))
