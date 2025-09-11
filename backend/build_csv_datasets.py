#!/usr/bin/env python
"""
build_csv_datasets.py
=====================

Purpose
-------
Build a *single*, prediction-ready NFL game-level dataset (one row per game)
with leak-free rolling features and normalized team codes.

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
- Output: main CSV at ``<out_dir>/Nfl_data.csv`` plus a chronologically
  sorted convenience CSV ``Nfl_data_sorted.csv`` in the CWD.
- Rolling stats use ``groupby().shift(1).rolling(...)`` to prevent future leakage.
- Team codes are minimally normalized to limit join mismatches (LA→LAR, STL→LAR, ...).

**IMPORTANT** TO RUN:
python backend/build_csv_datasets.py --start 2014 --end 2025 --out-dir backend/data

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nfl_data_py as nfl  # canonical schedule source used elsewhere

# -----------------------------
# Configuration and constants
# -----------------------------

ABBR_FIX: Dict[str, str] = {
    "LA": "LAR",   # Rams short
    "STL": "LAR",  # Rams legacy
    "SD": "LAC",   # Chargers legacy
    "OAK": "LV",   # Raiders legacy
    "WSH": "WAS",  # Commanders legacy
}

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


def load_schedules(seasons: List[int]) -> pd.DataFrame:
    """
    Load schedules + final scores for given seasons; drop games without scores.

    Returns
    -------
    DataFrame with:
      ['season','week','game_id','game_date','home_team','away_team',
       'home_score','away_score']
    """
    logging.info("Loading schedules for seasons=%s", seasons)
    sch = nfl.import_schedules(seasons)

    needed = [
        "season", "week", "game_id", "gameday",  # nflverse uses 'gameday'
        "home_team", "away_team", "home_score", "away_score",
    ]
    missing = [c for c in needed if c not in sch.columns]
    if missing:
        raise RuntimeError(f"Missing schedule columns: {missing}")

    sch = _normalize_codes(sch, ["home_team", "away_team"])
    sch["week"] = sch["week"].astype(int)  # enforce int for monotonic keys
    sch = sch.rename(columns={"gameday": "game_date"})
    sch = sch[[
        "season", "week", "game_id", "game_date",
        "home_team", "away_team", "home_score", "away_score",
    ]].copy()

    # Keep only completed games
    sch = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    logging.info("Schedules loaded: %d games", len(sch))
    return sch


# -----------------------------
# Feature engineering (leak-free)
# -----------------------------

def _team_game_long(sch: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-game schedule to *per-team per-game* long format to compute priors.
    """
    # Home perspective
    home = sch.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against",
    }).copy()
    home["is_home"] = 1

    # Away perspective
    away = sch.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against",
    }).copy()
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long["win"] = (long["points_for"] > long["points_against"]).astype(int)
    long["time_key"] = make_time_key(long)

    # Sorted so that groupby().shift(1) yields strictly prior games
    return long.sort_values(["team", "time_key", "game_id"]).reset_index(drop=True)


def _rolling_prior_stats(long: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Compute prior rolling means and win% per team with strict leakage protection.
    """
    grp = long.groupby("team", group_keys=False)

    # Use s.shift(1) to exclude the *current* game from the window
    long[f"prior_pf_avg_{window}"] = grp["points_for"].apply(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    long[f"prior_pa_avg_{window}"] = grp["points_against"].apply(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    long[f"prior_win_pct_{window}"] = grp["win"].apply(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    return long


def add_features(sch: pd.DataFrame, windows: Tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """
    Build one-row-per-game features by:
      1) creating per-team rows,
      2) computing prior rolling stats,
      3) re-pivoting to wide with home_/away_ prefixes.
    """
    long = _team_game_long(sch)
    for w in windows:
        long = _rolling_prior_stats(long, window=w)

    base_cols = [
        "season", "week", "game_id", "game_date", "team", "opponent",
        "points_for", "points_against", "win",
    ]
    prior_cols = [c for c in long.columns if c.startswith("prior_")]
    carry = base_cols + prior_cols

    home_side = long[long["is_home"] == 1][carry].add_prefix("home_")
    away_side = long[long["is_home"] == 0][carry].add_prefix("away_")

    # Merge back to one row per game (home + away)
    wide = home_side.merge(
        away_side, left_on="home_game_id", right_on="away_game_id", how="inner"
    )
    wide = wide.rename(columns={"home_game_id": "game_id"}).drop(columns=["away_game_id"])

    # Convenience fields at game-level
    wide["point_diff"] = wide["home_points_for"] - wide["away_points_for"]
    wide["winner"] = np.where(
        wide["point_diff"] > 0, wide["home_team"],
        np.where(wide["point_diff"] < 0, wide["away_team"], "TIE")
    )

    # Chronological sort across seasons/weeks
    wide["season"] = wide["home_season"].astype(int)
    wide["week"] = wide["home_week"].astype(int)
    wide["time_key"] = make_time_key(wide)
    wide = wide.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    # Column ordering: identifiers + outcomes, then priors
    ordered_cols = [
        "season", "week", "game_id", "home_game_date",
        "home_team", "away_team",
        "home_points_for", "away_points_for", "point_diff", "winner",
    ]
    prior_home = sorted([c for c in wide.columns if c.startswith("home_prior_")])
    prior_away = sorted([c for c in wide.columns if c.startswith("away_prior_")])

    final = wide[ordered_cols + prior_home + prior_away].rename(
        columns={"home_game_date": "game_date"}
    )
    return final


# -----------------------------
# Orchestration (CLI)
# -----------------------------

def build_dataset(start: int, end: int, out_dir: Path):
    """
    Orchestrate the dataset build: load → features → write.
    """
    seasons = list(range(int(start), int(end) + 1))
    logging.info("Building dataset for seasons=%s", seasons)

    schedules = load_schedules(seasons)
    final_df = add_features(schedules, windows=(3, 5))

    # Fill missing prior features with column means (simple, deterministic)
    prior_mask = final_df.filter(regex=r"^(home|away)_prior_").columns
    final_df[prior_mask] = final_df[prior_mask].fillna(final_df[prior_mask].mean())

    # Sorted convenience CSV in CWD
    final_df.sort_values(by="game_date").to_csv("Nfl_data_sorted.csv", index=False)
    logging.info("Wrote sorted dataset to Nfl_data_sorted.csv")

    # Main output in <out_dir> (created if necessary)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Nfl_data.csv"
    final_df.to_csv(out_path, index=False)
    logging.info("Wrote %s with %d games", out_path, len(final_df))
    return out_path, final_df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for season range and output directory."""
    p = argparse.ArgumentParser(description="Build NFL game-level dataset (one row per game).")
    p.add_argument("--start", type=int, default=2014, help="Start season (inclusive).")
    p.add_argument("--end", type=int, default=2025, help="End season (inclusive).")
    p.add_argument("--out-dir", type=str, default="backend/data", help="Output directory.")
    return p.parse_args()


def main() -> None:
    """Entry point for CLI usage with logging setup."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    setup_logger(out_dir)
    build_dataset(args.start, args.end, out_dir)


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
