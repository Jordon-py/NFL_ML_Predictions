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
pandas, numpy, nflreadpy (preferred) or nfl_data_py (fallback)

Usage Notes
-----------
- Output: single chronologically sorted CSV ``game_features_new.csv`` written to
  the specified ``out_dir`` (optional legacy root-level copy for compatibility).
- Rolling stats use ``groupby().rolling(...)`` with shift(1) to prevent future leakage.
- Team codes are minimally normalized to limit join mismatches (LA→LAR, STL→LAR, ...).

**IMPORTANT** TO RUN:
python backend/build_csv_datasets.py --start 2016 --end 2026 --out-dir backend/data
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# --- NFL data source shim: prefer nflreadpy (fast), fallback to nfl_data_py ---
NFL_BACKEND = "nfl_data_py"
nfl = None
_fallback_reason = None
try:
    import nflreadpy as _nfl
    # probe: try to produce a tiny pandas df (Polars conversion can require pyarrow)
    try:
        _probe = _nfl.load_schedules(seasons=[2024])
        if hasattr(_probe, "to_pandas"):
            _ = _probe.head(1).to_pandas()
        NFL_BACKEND = "nflreadpy"
        nfl = _nfl
    except Exception as e:
        _fallback_reason = f"nflreadpy probe failed: {e}"
        import nfl_data_py as _nfl
        nfl = _nfl
except Exception as e:
    _fallback_reason = f"nflreadpy import failed: {e}"
    import nfl_data_py as _nfl
    nfl = _nfl

if _fallback_reason:
    print(f"[WARN] Using fallback backend '{NFL_BACKEND}' — {_fallback_reason}")
else:
    print(f"[INFO] Using backend '{NFL_BACKEND}'")

# ML imports (kept after pandas import to speed cold start a bit)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error


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
OUTPUT_DATASET_NAME = "game_features_new.csv"

# Leak-safety settings for helper section below
HAS_winner_BOOL = True            # set False if you only have scores
TIME_COLS_IN_ORDER: Optional[List[str]] = None  # e.g. ["season","week"] or ["game_date"]


def make_time_key(df: pd.DataFrame) -> pd.Series:
    """Return sortable integer key YYYYWW from 'season' and 'week' (assumes ints)."""
    return (df["season"].astype(int) * 100) + df["week"].astype(int)


# -----------------------------
# Logging
# -----------------------------

def setup_logger(out_dir: Path) -> None:
    """Initialize both file and console logging so CLI users get progress feedback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "build_csv_datasets.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info("Logger initialized. Writing to %s", log_file)


# -----------------------------
# Polars → pandas conversion (no hard pyarrow dependency)
# -----------------------------

def to_pandas_safe(obj):
    """
    Accepts either a pandas.DataFrame or a Polars DataFrame/LazyFrame (nflreadpy),
    and returns a pandas.DataFrame. Avoids hard dependency on pyarrow.
    """
    # Already pandas?
    if obj.__class__.__module__.startswith("pandas"):
        return obj

    # Polars DataFrame
    try:
        import polars as pl  # noqa
        if isinstance(obj, pl.DataFrame):
            try:
                return obj.to_pandas(use_pyarrow_extension_array=False)
            except TypeError:
                return obj.to_pandas()
    except Exception:
        pass

    # Polars LazyFrame
    try:
        import polars as pl  # noqa
        if isinstance(obj, pl.LazyFrame):
            lf = obj
            df = lf.collect()
            try:
                return df.to_pandas(use_pyarrow_extension_array=False)
            except TypeError:
                return df.to_pandas()
    except Exception:
        pass

    # Generic: object exposes .to_pandas()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()

    raise TypeError(f"Unsupported table type for to_pandas_safe: {type(obj)}")


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
    probs.loc[negative] = (-ml_numeric.loc[negative]) / ((-ml_numeric.loc[negative]) + 100)
    probs.loc[positive & ml_numeric.notna()] = 100 / (ml_numeric.loc[positive & ml_numeric.notna()] + 100)
    return probs


def load_team_game_metrics(pbp_path: Path) -> pd.DataFrame:
    """
    Aggregate play-by-play data to per-team, per-game advanced metrics.
    Uses nflreadpy when available; otherwise falls back to nfl_data_py.
    """
    try:
        seasons_to_load = list(range(2016, 2026))
        logging.info("Loading play-by-play for seasons %s via %s", seasons_to_load, NFL_BACKEND)
        if NFL_BACKEND == "nflreadpy":
            pbp = to_pandas_safe(nfl.load_pbp(seasons=seasons_to_load))
        else:
            pbp = nfl.import_pbp(seasons_to_load)
        logging.info("Loaded %d play-by-play rows", len(pbp))
    except Exception as exc:
        logging.warning("PBP load failed (%s); trying cached CSV at %s", exc, pbp_path)
        if not pbp_path.exists():
            logging.warning("Cached PBP missing; advanced features disabled")
            return pd.DataFrame(columns=["season", "week", "game_id", "team"])
        pbp = pd.read_csv(pbp_path, low_memory=False)

    required_cols = ["season", "week", "game_id", "posteam", "defteam"]
    missing = [c for c in required_cols if c not in pbp.columns]
    if missing:
        logging.warning("PBP missing columns %s; advanced features disabled", missing)
        return pd.DataFrame(columns=["season", "week", "game_id", "team"])

    mask_valid_team = pbp["posteam"].notna()
    pbp = pbp.loc[mask_valid_team].copy()
    if pbp.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "team"])

    # sanitize and fill
    pbp["season"] = pbp["season"].astype(int)
    pbp["week"] = pbp["week"].astype(int)
    for col, default in [
        ("epa", 0.0), ("success", 0.0), ("pass", 0.0), ("xpass", 0.0),
        ("pass_attempt", 0.0), ("rush_attempt", 0.0),
        ("third_down_converted", 0.0), ("third_down_failed", 0.0),
        ("interception", 0.0), ("fumble_lost", 0.0), ("yards_gained", 0.0),
    ]:
        if col in pbp.columns:
            pbp[col] = pbp[col].fillna(default)

    pbp["turnover"] = pbp.get("interception", 0.0) + pbp.get("fumble_lost", 0.0)
    pbp["explosive_play"] = (
        ((pbp.get("pass", 0.0) == 1.0) & (pbp.get("yards_gained", 0.0) >= 20))
        | ((pbp.get("rush_attempt", 0.0) == 1.0) & (pbp.get("yards_gained", 0.0) >= 15))
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

    off_agg["off_third_down_total"] = off_agg["off_third_down_conv"] + off_agg["off_third_down_fail"]
    off_agg["off_third_down_pct"] = np.where(
        off_agg["off_third_down_total"] > 0,
        off_agg["off_third_down_conv"] / off_agg["off_third_down_total"],
        np.nan,
    )
    off_agg["off_pass_over_expected"] = off_agg["off_pass_rate"] - off_agg["off_expected_pass_rate"]

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
        def_agg["def_takeaways"] / (def_agg["def_pass_attempts_faced"] + def_agg["def_rush_attempts_faced"]),
        np.nan,
    )

    metrics = off_agg.rename(columns={"posteam": "team"}).merge(
        def_agg.rename(columns={"defteam": "team"}),
        on=["season", "week", "game_id", "team"],
        how="outer",
    )

    # clean up columns that aren't needed downstream
    metrics["off_total_plays"] = metrics["off_pass_attempts"].fillna(0) + metrics["off_rush_attempts"].fillna(0)
    metrics["off_turnover_rate"] = np.where(
        metrics["off_total_plays"] > 0,
        metrics["off_turnovers"].fillna(0) / metrics["off_total_plays"],
        np.nan,
    )
    metrics = metrics.drop(
        columns=[
            "off_third_down_total", "def_epa_allowed",
            "def_pass_attempts_faced", "def_rush_attempts_faced",
            "off_pass_attempts", "off_rush_attempts", "off_turnovers",
            "off_third_down_conv", "off_third_down_fail", "def_takeaways",
            "off_pass_rate", "off_expected_pass_rate",
        ],
        errors="ignore",
    )

    return metrics.fillna(np.nan)


def load_player_game_stats(seasons: List[int]) -> pd.DataFrame:
    """Load weekly player-level stats and aggregate to team-game level."""
    try:
        logging.info("Loading player stats via %s for seasons %s", NFL_BACKEND, seasons)
        if NFL_BACKEND == "nflreadpy":
            player_stats = to_pandas_safe(nfl.load_player_stats(seasons=seasons, summary_level="week"))
        else:
            # nfl_data_py: weekly player stats
            player_stats = nfl.import_weekly_data(seasons)
        logging.info("Loaded %d player-week records", len(player_stats))

        team_col = "recent_team" if "recent_team" in player_stats.columns else ("team" if "team" in player_stats.columns else None)
        if not team_col:
            logging.warning("Cannot determine team column in player_stats; skipping")
            return pd.DataFrame(columns=["season", "week", "team"])

        # QB aggregation
        qb_stats = player_stats[player_stats.get("position").fillna("") == "QB"].copy()
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

        # RB aggregation
        rb_stats = player_stats[player_stats.get("position").isin(["RB"])].copy()
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

        # WR+TE aggregation
        pass_catchers = player_stats[player_stats.get("position").isin(["WR", "TE"])].copy()
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

        # Merge and rename
        player_team_stats = qb_agg.merge(rb_agg, on=["season", "week", team_col], how="outer")
        player_team_stats = player_team_stats.merge(wr_agg, on=["season", "week", team_col], how="outer")
        player_team_stats = player_team_stats.rename(columns={team_col: "team"}).fillna(0)
        return player_team_stats

    except Exception as exc:
        logging.warning("Failed to load player stats (%s); player features disabled", exc)
        return pd.DataFrame(columns=["season", "week", "team"])


def load_team_weekly_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Load team-level weekly stats. With nflreadpy we use summary_level='week'.
    With nfl_data_py there isn't a direct team-week API, so we skip gracefully.
    """
    if NFL_BACKEND == "nflreadpy":
        try:
            logging.info("Loading team weekly stats via nflreadpy for seasons %s", seasons)
            team_stats = to_pandas_safe(nfl.load_team_stats(seasons=seasons, summary_level="week"))
            logging.info("Loaded %d team-week records", len(team_stats))
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
        except Exception as exc:
            logging.warning("Team stats load failed (%s); team features disabled", exc)

    # fallback: unavailable
    return pd.DataFrame(columns=["season", "week", "team"])


def load_schedules(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
    """
    Load schedules + final scores for given seasons.
    Returns columns:
    ['season','week','game_id','game_date','home_team','away_team','home_score','away_score',
     'game_type','away_moneyline','home_moneyline','spread_line','total_line','away_rest','home_rest']
    """
    if NFL_BACKEND == "nflreadpy":
        sch = to_pandas_safe(nfl.load_schedules(seasons=seasons))
    else:
        sch = nfl.import_schedules(seasons)

    logging.info("Raw schedules loaded: %d games", len(sch))

    needed = [
        "season","week","game_id",
        "gameday","home_team","away_team","home_score","away_score","game_type",
        "away_moneyline","home_moneyline","spread_line","total_line","away_rest","home_rest",
    ]
    missing = [c for c in needed if c not in sch.columns]
    if missing:
        raise RuntimeError(f"Missing schedule columns: {missing}")

    sch = _normalize_codes(sch, ["home_team", "away_team"])
    sch["week"] = sch["week"].astype(int)
    sch = sch.rename(columns={"gameday": "game_date"})
    sch = sch[
        ["season","week","game_id","game_date","home_team","away_team","home_score","away_score",
         "game_type","away_moneyline","home_moneyline","spread_line","total_line","away_rest","home_rest"]
    ].copy()

    if include_future:
        completed = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        future = sch[sch["home_score"].isna() | sch["away_score"].isna()].copy()
        future[["home_score","away_score"]] = None
        future = future[future["game_type"] == "REG"].reset_index(drop=True)
        logging.info("Loaded %d completed games + %d future games", len(completed), len(future))
        return pd.concat([completed, future], ignore_index=True)

    sch = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    logging.info("Schedules loaded: %d completed games", len(sch))
    return sch


# -----------------------------
# Feature engineering (leak-free)
# -----------------------------

def _team_game_long(sch: pd.DataFrame) -> pd.DataFrame:
    """Convert per-game schedule to per-team, per-game long format (home/away rows)."""
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
        completed_mask, (long["points_for"] > long["points_against"]).astype(float), np.nan
    )

    long["time_key"] = make_time_key(long)
    return long.sort_values(["team", "time_key", "game_id"]).reset_index(drop=True)


def _rolling_prior_stats(long: pd.DataFrame, window: int = 3, advanced_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute prior rolling means and win% per team with strict leakage protection."""
    grp = long.groupby("team", group_keys=False)

    def safe_rolling_mean(s):
        shifted = s.shift(1)  # Prior games only
        return shifted.rolling(window=window, min_periods=1).mean()

    long[f"prior_pf_avg_{window}"] = grp["points_for"].apply(safe_rolling_mean)
    long[f"prior_pa_avg_{window}"] = grp["points_against"].apply(safe_rolling_mean)
    long[f"prior_win_pct_{window}"] = grp["win"].apply(safe_rolling_mean)

    if advanced_cols:
        for col in advanced_cols:
            if col in long.columns:
                long[f"prior_{col}_{window}"] = grp[col].apply(safe_rolling_mean)

    return long


def add_features(
    sch: pd.DataFrame,
    windows: Tuple[int, ...] = (3, 5),
    advanced_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build one-row-per-game features:
      1) per-team long,
      2) prior rolling stats,
      3) pivot back to wide with home_/away_ prefixes,
      4) opponent-relative deltas (home_minus_away_*),
      5) add market/rest context.
    """
    long = _team_game_long(sch)

    advanced_cols: List[str] = []
    if advanced_metrics is not None and not advanced_metrics.empty:
        advanced_cols = [c for c in advanced_metrics.columns if c not in {"season", "week", "game_id", "team"}]
        long = long.merge(advanced_metrics, on=["season", "week", "game_id", "team"], how="left")

    for w in windows:
        long = _rolling_prior_stats(long, window=w, advanced_cols=advanced_cols)

    if advanced_cols:
        long = long.drop(columns=advanced_cols, errors="ignore")

    base_cols = [
        "season","week","game_id","game_date","team","opponent","points_for","points_against","win",
    ]
    prior_cols = [c for c in long.columns if c.startswith("prior_")]
    carry = base_cols + prior_cols

    home_side = long[long["is_home"] == 1][carry].add_prefix("home_")
    away_side = long[long["is_home"] == 0][carry].add_prefix("away_")

    wide = home_side.merge(
        away_side, left_on="home_game_id", right_on="away_game_id", how="inner"
    )
    wide = wide.rename(columns={"home_game_id": "game_id"}).drop(columns=["away_game_id"])

    # outcomes
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
            np.where(wide["point_diff"].notna() & (wide["point_diff"] == 0), "TIE", "TBD"),
        ),
    )

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
    final_cols = ordered_cols + prior_feature_cols + diff_feature_cols

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

    final_cols.extend(
        ["home_moneyline_prob","away_moneyline_prob","moneyline_prob_diff",
         "spread_line","total_line","home_rest","away_rest","rest_diff"]
    )
    return wide[final_cols]


def _merge_team_week_stats(game_df: pd.DataFrame, team_week_stats: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Merge team-week level stats into game-level dataframe for both home and away teams."""
    if team_week_stats.empty:
        return game_df
    stat_cols = [c for c in team_week_stats.columns if c not in {"season", "week", "team"}]

    home_stats = team_week_stats.copy()
    home_stats.columns = ["season", "week", "home_team"] + [f"home_{prefix}_{c}" for c in stat_cols]
    game_df = game_df.merge(home_stats, on=["season", "week", "home_team"], how="left")

    away_stats = team_week_stats.copy()
    away_stats.columns = ["season", "week", "away_team"] + [f"away_{prefix}_{c}" for c in stat_cols]
    game_df = game_df.merge(away_stats, on=["season", "week", "away_team"], how="left")
    return game_df


def build_regression_pipeline(
    numeric_features: List[str], categorical_features: List[str], alpha: float = 1.0
) -> Pipeline:
    """Simple Ridge pipeline with median impute + scaling for numeric and OHE for categoricals."""
    numeric_steps = Pipeline(
        [("num_impute", SimpleImputer(strategy="median")), ("num_scale", StandardScaler(with_mean=False))]
    )
    categorical_steps = Pipeline(
        [("cat_impute", SimpleImputer(strategy="most_frequent")), ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse=True))]
    )
    preprocess = ColumnTransformer([("num", numeric_steps, numeric_features), ("cat", categorical_steps, categorical_features)])
    model = Ridge(alpha=alpha)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def ts_split_by_season_week(
    df: pd.DataFrame, features: List[str], target: str, train_end: Tuple[int, int],
):
    """Chronological split that prevents leakage. Returns (X_train, y_train), df, sorted_df."""
    data = df.copy()
    required_cols = {"season", "week", *features, target}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["time_key"] = make_time_key(data)
    data = data.sort_values(["time_key"]).reset_index(drop=True)

    train_end_season, train_end_week = train_end
    is_train = (data["season"] < train_end_season) | ((data["season"] == train_end_season) & (data["week"] <= train_end_week))
    train_df = data.loc[is_train]
    X_train, y_train = train_df[features], train_df[target]
    return (X_train, y_train), df, data


# -----------------------------
# Dominance + pregame helpers (leak-free pairwise history)
# -----------------------------

def ensure_actual_winner(df):
    df = df.copy()
    if "home_win" in df.columns:
        win_series = pd.Series(df["home_win"], index=df.index, dtype="boolean")
    else:
        if not HAS_winner_BOOL:
            if {"home_points", "away_points"}.issubset(df.columns):
                win_series = pd.Series((df["home_points"] > df["away_points"]), index=df.index, dtype="boolean")
            else:
                raise ValueError("Need either 'winner' bool or score columns 'home_points'/'away_points'.")
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


def detect_time_cols(df):
    if TIME_COLS_IN_ORDER is not None:
        return TIME_COLS_IN_ORDER
    if "game_date" in df.columns:
        return ["game_date"]
    if {"season", "week"}.issubset(df.columns):
        return ["season", "week"]
    return []


def make_long_edges(df):
    """Two rows per game (home perspective + away perspective) with team_won flag."""
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

    home_rows = base[["gid", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opponent"}).copy()
    home_rows["team_won"] = win_series.reindex(home_rows.index)
    home_rows = home_rows[["gid", "team", "opponent", "team_won"]]

    away_rows = base[["gid", "home_team", "away_team"]].rename(columns={"away_team": "team", "home_team": "opponent"}).copy()
    away_rows["team_won"] = (~win_series).reindex(away_rows.index)
    away_rows = away_rows[["gid", "team", "opponent", "team_won"]]

    long = pd.concat([home_rows, away_rows], ignore_index=True)

    time_cols = detect_time_cols(df)
    if time_cols:
        long = long.merge(base[["gid"] + time_cols], on="gid", how="left")
    return long


def pairwise_table(long):
    """Static, all-time pairwise records (order matters: (team, opponent))."""
    g = long.groupby(["team", "opponent"], as_index=False, observed=True)
    agg = g.agg(games=("team", "size"), wins=("team_won", "sum"))
    agg["losses"] = agg["games"] - agg["wins"]
    agg["win_pct"] = np.where(agg["games"] > 0, agg["wins"] / agg["games"], np.nan)
    agg["dominance"] = agg["wins"] - agg["losses"]
    agg["signed_wins_style"] = agg["wins"]
    return agg


def leak_free_pregame_features(long):
    """Prior-to-game counts within (team, opponent)."""
    time_cols = [c for c in long.columns if c in ("game_date", "season", "week")]
    sort_cols = time_cols + ["gid"] if time_cols else ["gid"]
    long = long.sort_values(sort_cols).copy()

    grp = long.groupby(["team", "opponent"], group_keys=False, observed=True)
    long["prior_games"] = grp.cumcount()
    long["prior_wins"] = grp["team_won"].cumsum() - long["team_won"]
    long["prior_losses"] = long["prior_games"] - long["prior_wins"]
    long["prior_dom"] = long["prior_wins"] - long["prior_losses"]
    long["prior_win_pct"] = np.where(long["prior_games"] > 0, long["prior_wins"] / long["prior_games"], np.nan)

    keep_cols = ["gid","team","opponent","prior_games","prior_wins","prior_losses","prior_dom","prior_win_pct"]
    return long[keep_cols]


def attach_pregame_to_wide(df, pre):
    """Attach pre-game features back to the original wide df: home_vs_away_* and away_vs_home_*."""
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


def team_level_prediction_accuracy(df):
    """Accuracy of your classifier *when a given team plays* (expects 'prob_winner' and 'winner')."""
    if "prob_winner" not in df.columns:
        raise ValueError("Need a 'prob_winner' column for prediction accuracy.")
    df = df.copy()
    df["pred_winner"] = np.where(df["prob_winner"] >= 0.5, df["home_team"], df["away_team"])
    df["actual_winner"] = np.where(df["winner"], df["home_team"], df["away_team"])
    df["pred_correct"] = (df["pred_winner"] == df["actual_winner"]).astype(int)
    home_part = df[["home_team", "pred_correct"]].rename(columns={"home_team": "team"})
    away_part = df[["away_team", "pred_correct"]].rename(columns={"away_team": "team"})
    team_games = pd.concat([home_part, away_part], ignore_index=True)
    acc = team_games.groupby("team")["pred_correct"].mean().sort_values(ascending=False)
    return acc


def build_dominance_features(dff):
    df = ensure_actual_winner(dff)
    long = make_long_edges(df)
    pair_table = pairwise_table(long)
    pre = leak_free_pregame_features(long)
    df_with_pregame = attach_pregame_to_wide(df, pre)
    dom_matrix = pair_table.pivot(index="team", columns="opponent", values="dominance").fillna(0).astype(int)
    return {
        "pair_table": pair_table.sort_values(["team","opponent"]).reset_index(drop=True),
        "dominance_matrix": dom_matrix,
        "df_with_pregame": df_with_pregame,
        # "team_prediction_accuracy": team_acc  # optional if you add prob_winner
    }


# -----------------------------
# REL: Learning Memory + Feedback Loop
# -----------------------------

class LearningMemory:
    """Persisted memory for schema aliases, feature utility, and window choices."""
    def __init__(self, path: Path):
        self.path = path
        self.data = {
            "schema_aliases": {},
            "feature_scores": {},
            "best_windows": [3,5],
            "merge_rules": {},
            "run_history": []
        }
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text()))
            except Exception:
                pass

    def bump_feature(self, name: str, auc_gain=0.0, brier_gain=0.0):
        slot = self.data["feature_scores"].setdefault(name, {"auc_gain":0.0,"brier_gain":0.0,"count":0})
        slot["auc_gain"] += float(auc_gain)
        slot["brier_gain"] += float(brier_gain)
        slot["count"] += 1

    def set_best_windows(self, windows):
        self.data["best_windows"] = list(windows)

    def remember_run(self, season_end: int, metrics: dict, kept_features: list):
        self.data["run_history"].append({"season_end": season_end, "metrics": metrics, "kept_features": kept_features})

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))


def implied_from_moneylines(df: pd.DataFrame) -> pd.Series:
    return df.get("home_moneyline_prob", pd.Series([float("nan")]*len(df), index=df.index)).astype(float)


def evaluate_binary(y_true_bool: pd.Series, p_home: pd.Series) -> dict:
    mask = y_true_bool.notna() & p_home.notna()
    if mask.sum() == 0:
        return {"auc": None, "brier": None}
    y = y_true_bool[mask].astype(int)
    p = p_home[mask].clip(1e-6, 1-1e-6)
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p))}


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"mae": None}
    return {"mae": float(mean_absolute_error(y_true[mask], y_pred[mask]))}


def train_light_models(df: pd.DataFrame, feature_cols_cls: list, feature_cols_reg: list):
    y_cls = df.get("home_win").astype("float")
    X_cls = df[feature_cols_cls].copy()
    y_reg = df.get("point_diff")
    X_reg = df[feature_cols_reg].copy()

    X_cls = X_cls.fillna(X_cls.median(numeric_only=True))
    X_reg = X_reg.fillna(X_reg.median(numeric_only=True))

    clf = LogisticRegression(max_iter=200)
    mask_cls = y_cls.notna()
    if mask_cls.sum() > 10 and len(feature_cols_cls) > 0:
        clf.fit(X_cls.loc[mask_cls], y_cls.loc[mask_cls])
        p_home = pd.Series(clf.predict_proba(X_cls)[:,1], index=df.index)
    else:
        p_home = pd.Series(np.nan, index=df.index)

    reg = Ridge(alpha=6.0)
    mask_reg = y_reg.notna()
    if mask_reg.sum() > 10 and len(feature_cols_reg) > 0:
        reg.fit(X_reg.loc[mask_reg], y_reg.loc[mask_reg])
        y_hat = pd.Series(reg.predict(X_reg), index=df.index)
    else:
        y_hat = pd.Series(np.nan, index=df.index)

    return p_home, y_hat, clf, reg


def add_momentum_and_oas(df: pd.DataFrame, windows=(3,5)) -> pd.DataFrame:
    out = df.copy()

    diff_cols = [c for c in out.columns if c.startswith("home_minus_away_")]
    preferred = [c for c in diff_cols if any(key in c for key in ["win_pct","epa","success","explosive","turnover","points","yards","first_downs"])]
    candidates = preferred if len(preferred) >= 3 else diff_cols[: min(8, len(diff_cols))]

    def rolling_slope(series: pd.Series, w: int) -> pd.Series:
        def fit_win(x):
            idx = np.arange(len(x))
            x = np.array(x, dtype=float)
            good = np.isfinite(x)
            if good.sum() < 2:
                return np.nan
            idx = idx[good]
            x = x[good]
            A = np.vstack([idx, np.ones_like(idx)]).T
            slope, _ = np.linalg.lstsq(A, x, rcond=None)[0]
            return slope
        return series.rolling(w, min_periods=2).apply(fit_win, raw=False)

    for w in windows:
        for c in candidates:
            out[f"trend_{c}_w{w}"] = rolling_slope(out[c], w)

    prior_cols = [c for c in out.columns if c.startswith(("home_prior_","away_prior_"))]
    if prior_cols and {"season","week"}.issubset(out.columns):
        grp = out.groupby(["season","week"], group_keys=False)
        z_cols = []
        for c in [c for c in prior_cols if out[c].dtype.kind in "fiu"]:
            zc = f"{c}_z"
            out[zc] = grp[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-6))
            z_cols.append(zc)
        if z_cols:
            out["oas_index"] = out[z_cols].mean(axis=1)

    return out


def feedback_iterate(df: pd.DataFrame, memory: LearningMemory, try_windows=((3,5),(2,3,5),(3,5,7))):
    best = None
    for winset in try_windows:
        cand = add_momentum_and_oas(df, windows=winset)
        priors = [c for c in cand.columns if c.startswith(("home_prior_","away_prior_","home_minus_away_"))]
        evolved = [c for c in cand.columns if c.startswith(("trend_","oas_index"))]
        market = [c for c in ["moneyline_prob_diff","spread_line","total_line","rest_diff"] if c in cand.columns]

        feature_cols_cls = [c for c in priors + evolved + market if c in cand.columns and cand[c].dtype.kind in "fiu"]
        feature_cols_reg = feature_cols_cls

        p_home, y_hat, _, _ = train_light_models(cand, feature_cols_cls, feature_cols_reg)
        market_p = implied_from_moneylines(cand)
        met_model = evaluate_binary(cand.get("home_win"), p_home)
        met_market = evaluate_binary(cand.get("home_win"), market_p)
        met_reg = evaluate_regression(cand.get("point_diff"), y_hat)

        auc_uplift = ((met_model.get("auc") or 0) - (met_market.get("auc") or 0))
        mae_penalty = (met_reg.get("mae") or 0) * 0.001
        score = auc_uplift - mae_penalty

        if (best is None) or (score > best["score"]):
            best = {
                "windows": winset,
                "df": cand,
                "metrics_model": met_model,
                "metrics_market": met_market,
                "metrics_reg": met_reg,
                "score": score,
                "feature_cols_cls": feature_cols_cls,
                "feature_cols_reg": feature_cols_reg,
            }

    memory.set_best_windows(best["windows"])
    for f in best["feature_cols_cls"]:
        memory.bump_feature(f, auc_gain=((best["metrics_model"].get("auc") or 0) - (best["metrics_market"].get("auc") or 0)))
    return best


# -----------------------------
# Orchestration (CLI)
# -----------------------------

def get_current_nfl_week_from_schedules(sch: pd.DataFrame) -> tuple[int, int]:
    """Compute current season/week from schedules using UTC 'now' cutoff."""
    from datetime import datetime, timezone
    if "game_date" not in sch.columns:
        return (int(sch["season"].max()), int(sch.loc[sch["season"] == sch["season"].max(), "week"].max()))
    now = datetime.now(timezone.utc).date()
    past = sch[pd.to_datetime(sch["game_date"]).dt.date <= now]
    if not past.empty:
        season = int(past["season"].max())
        week = int(past.loc[past["season"] == season, "week"].max())
        return season, week
    return (int(sch["season"].min()), 1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for season range and output directory."""
    p = argparse.ArgumentParser(description="Build NFL game-level dataset (one row per game).")
    p.add_argument("--start", type=int, default=2014, help="Start season (inclusive).")
    p.add_argument("--end", type=int, default=2025, help="End season (inclusive).")
    p.add_argument("--out-dir", type=str, default="backend/data", help="Output directory.")
    p.add_argument(
        "--legacy-root-copy",
        action="store_true",
        help=f"Also write {OUTPUT_DATASET_NAME} to the repository root for backwards compatibility.",
    )
    return p.parse_args()


def build_dataset(
    start: int,
    end: int,
    out_dir: Path,
    production_mode: bool = True,
    include_future: bool = True,
    legacy_root_copy: bool = True,
):
    """Build production-ready NFL dataset with completed + future scheduled games."""
    seasons = list(range(int(start), int(end) + 1))

    # Stage 1: Load base schedules with betting lines
    schedules = load_schedules(seasons, include_future=include_future)

    # Log current NFL state from schedules (more accurate than previous).
    if production_mode:
        current_season, current_week = get_current_nfl_week_from_schedules(schedules)
        logging.info("Production dataset build - Current NFL state: %dW%d", current_season, current_week)

    logging.info("Building dataset for seasons=%s (include_future=%s)", seasons, include_future)

    # Stage 2: Load advanced play-by-play metrics
    data_dir = Path(__file__).resolve().parent / "data"
    pbp_metrics = load_team_game_metrics(data_dir / "pbp_clean.csv")
    if not pbp_metrics.empty:
        pbp_metrics = pbp_metrics[pbp_metrics["season"].isin(seasons)]

    # Stage 3: Load player-level stats (QB, RB, WR aggregations)
    player_stats = load_player_game_stats(seasons)

    # Stage 4: Load team-level stats (official stats)
    team_stats = load_team_weekly_stats(seasons)

    # Stage 5: Engineer rolling features with PBP advanced metrics
    final_df = add_features(schedules, windows=(3, 5), advanced_metrics=pbp_metrics)

    # Stage 6: Merge player and team stats
    if not player_stats.empty:
        final_df = _merge_team_week_stats(final_df, player_stats, prefix="player")
        logging.info("Merged player stats: now %d columns", len(final_df.columns))

    if not team_stats.empty:
        final_df = _merge_team_week_stats(final_df, team_stats, prefix="teamstat")
        logging.info("Merged team stats: now %d columns", len(final_df.columns))

    # Robust prep: fill priors & diffs only; preserve future rows
    prior_mask = final_df.filter(regex=r"^(home|away)_prior_").columns
    diff_mask = final_df.filter(regex=r"^home_minus_away_").columns
    final_df[prior_mask] = final_df[prior_mask].fillna(final_df[prior_mask].median())
    final_df[diff_mask] = final_df[diff_mask].fillna(final_df[diff_mask].median())

    critical_feature_cols = [c for c in final_df.columns if c.startswith(("home_prior_", "away_prior_", "home_minus_away_"))]
    if include_future:
        final_df = final_df.dropna(subset=critical_feature_cols).reset_index(drop=True)
        logging.info("Kept future games - dropped only rows missing feature data")
    else:
        final_df = final_df.dropna().reset_index(drop=True)
        logging.info("Dropped all rows with any null values (training mode)")

    final_df = final_df.sort_values(by="home_game_date").reset_index(drop=True)

    # Create boolean home_win column (nullable for ties/future)
    home_win = pd.Series(pd.NA, index=final_df.index, dtype="boolean")
    home_win.loc[final_df["winner"] == final_df["home_team"]] = True
    home_win.loc[final_df["winner"] == final_df["away_team"]] = False
    final_df["home_win"] = home_win

    # Cast object→category for safety
    for col in final_df.select_dtypes(include=["object"]).columns:
        final_df[col] = final_df[col].astype("category")

    # Dominance features (writes df_with_pregame.csv for inspection)
    numeric_features = final_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = ["home_team", "away_team"]
    dff = final_df[categorical_features + list(set(numeric_features) | {"winner","home_win"}) if {"winner","home_win"}.issubset(final_df.columns) else categorical_features + numeric_features].copy()
    result = build_dominance_features(dff)
    logging.info("Pairwise dominance table:\n%s", result["pair_table"].head(10))
    logging.info("Dominance matrix:\n%s", result["dominance_matrix"].head(10))
    logging.info("DataFrame with pregame features:\n%s", result["df_with_pregame"].head(10))
    (out_dir / "df_with_pregame.csv").write_text(result["df_with_pregame"].to_csv(index=False))

    # === R.E.L.: Self-improving feedback loop (chooses best windows, adds momentum/OAS) ===
    try:
        memory_path = out_dir / "learning_memory.json"
        memory = LearningMemory(memory_path)
        fb = feedback_iterate(final_df, memory)
        final_df = fb["df"].copy()
        memory.remember_run(
            season_end=int(final_df["season"].max()) if "season" in final_df.columns else -1,
            metrics={
                "model_auc": fb["metrics_model"].get("auc"),
                "market_auc": fb["metrics_market"].get("auc"),
                "brier": fb["metrics_model"].get("brier"),
                "mae_point_diff": fb["metrics_reg"].get("mae"),
            },
            kept_features=sorted(set(fb["feature_cols_cls"]))
        )
        memory.save()
        logging.info("REL feedback: windows=%s, metrics=%s", fb["windows"], fb["metrics_model"])
    except Exception as exc:
        logging.warning("REL feedback loop failed (non-fatal): %s", exc)

    # Production output
    out_dir.mkdir(parents=True, exist_ok=True)
    main_output = out_dir / OUTPUT_DATASET_NAME
    final_df.to_csv(main_output, index=False)

    if legacy_root_copy:
        final_df.to_csv(OUTPUT_DATASET_NAME, index=False)
        logging.info("Legacy root-level copy created for compatibility across scripts.")

    logging.info("Production dataset ready: %s (%d games)", main_output, len(final_df))

    # Export team mapping for API consistency
    with (out_dir / "team_abbr_map.json").open("w") as f:
        json.dump(ABBR_FIX, f, indent=2)

    return main_output, final_df


def main() -> None:
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


if __name__ == "__main__":
    main()