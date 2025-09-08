#!/usr/bin/env python
"""
build_csv_datasets.py
=====================
Purpose: Build a single, prediction-ready NFL game-level dataset
(one row per game), with leak-free rolling features and stable team codes.

Key properties:
- One row per game (home + away in the same row).
- Normalizes legacy abbreviations (LA→LAR, STL→LAR, SD→LAC, OAK→LV, WSH→WAS).
- Rolling features computed with shift() to prevent future leakage.
- Sorted earliest → latest for time-series work.
- Writes exactly one CSV (default: data/Nfl_data.csv).

Quick start:
  python build_csv_datasets.py --start 2014 --end 2024 --out-dir data

Requires:
  pip install nfl_data_py pandas numpy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# nfl_data_py is the canonical source used in your other scripts
import nfl_data_py as nfl


# -----------------------------
# Configuration and constants
# -----------------------------
seasons = list(range(2014, 2025))  # inclusive range of seasons to load
# Minimal, targeted normalization to avoid opponent/team join gaps.
ABBR_FIX: Dict[str, str] = {
    "LA": "LAR",   # Rams short
    "STL": "LAR",  # Rams legacy
    "SD": "LAC",   # Chargers legacy
    "OAK": "LV",   # Raiders legacy
    "WSH": "WAS",  # Commanders legacy
}

# For stable sorting and simple time slicing
def make_time_key(df: pd.DataFrame) -> pd.Series:
    """Map (season, week) to a sortable integer YYYYWW."""
    return (df["season"].astype(int) * 100) + df["week"].astype(int)


# -----------------------------
# Logging
# -----------------------------

def setup_logger(out_dir: Path) -> None:
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
    """
    Normalize team abbreviations in place to modern codes.
    Apply to any relevant team columns to avoid mismatches in joins/rollups.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace(ABBR_FIX)
    return out


def load_schedules(seasons=seasons) -> pd.DataFrame:
    """
    Load game schedules and scores. One row per game.
    Columns used: season, week, game_id, game_date, home_team, away_team, home_score, away_score.
    """
    logging.info("Loading schedules for seasons=%s", seasons)
    sch = nfl.import_schedules(seasons)

    needed = [
        "season", "week", "game_id", "gameday",  # note: nflverse uses 'gameday' for date
        "home_team", "away_team", "home_score", "away_score",
    ]
    missing = [c for c in needed if c not in sch.columns]
    if missing:
        raise RuntimeError(f"Missing schedule columns: {missing}")

    # Normalize team codes early
    sch = _normalize_codes(sch, ["home_team", "away_team"])

    # Some seasons use floats for week; enforce int weeks for monotonic keys
    sch["week"] = sch["week"].astype(int)

    # Standardize date column name
    sch = sch.rename(columns={"gameday": "game_date"})

    # Keep only the columns we need
    sch = sch[[
        "season", "week", "game_id", "game_date",
        "home_team", "away_team", "home_score", "away_score",
    ]].copy()

    # Filter out games without scores (future or canceled)
    sch = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    logging.info("Schedules loaded: %d games", len(sch))
    return sch


# -----------------------------
# Feature engineering (leak-free)
# -----------------------------

def _team_game_long(sch: pd.DataFrame) -> pd.DataFrame:
    """
    Convert schedules (one row per game) into a long team-game table
    with one row per team per game. This is a helper for rolling features.
    """
    # Home rows
    home = sch.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against"
    }).copy()
    home["is_home"] = 1

    # Away rows
    away = sch.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against"
    }).copy()
    away["is_home"] = 0

    # Stack -> long (two rows per game, one per team)
    long = pd.concat([home, away], ignore_index=True)

    # Outcomes at team perspective
    long["win"] = (long["points_for"] > long["points_against"]).astype(int)
    long["time_key"] = make_time_key(long)

    # Sort for clean groupby().shift() operations
    long = long.sort_values(["team", "time_key", "game_id"]).reset_index(drop=True)
    return long


def _rolling_prior_stats(long: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Compute simple prior rolling stats per team with leakage protection.
    Uses shift() so the current game's row NEVER uses its own info.

    Features created (prefix 'prior'):
      - avg points_for last W games
      - avg points_against last W games
      - win_pct last W games
    """
    grp = long.groupby("team", group_keys=False)

    # Rolling means of points for/against over last W games (prior only)
    long[f"prior_pf_avg_{window}"] = grp["points_for"].apply(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    long[f"prior_pa_avg_{window}"] = grp["points_against"].apply(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )

    # Rolling win percent over last W games (prior only)
    long[f"prior_win_pct_{window}"] = grp["win"].apply(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )

    return long


def add_features(sch: pd.DataFrame, windows: Tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """
    Build final, one-row-per-game dataset with leak-free team priors merged wide:
    - Start from schedules.
    - Create long team-game rows.
    - Compute prior rolling stats per team for multiple windows.
    - Pivot back to one row per game with 'home_' and 'away_' prefixes.
    """
    # Long form for rolling
    long = _team_game_long(sch)

    # Add multiple windowed priors with no leakage
    for w in windows:
        long = _rolling_prior_stats(long, window=w)

    # Split back to home/away sides and prefix columns
    home_long = long[long["is_home"] == 1].copy()
    away_long = long[long["is_home"] == 0].copy()

    # Columns to carry back (include priors and identifiers)
    base_cols = [
        "season", "week", "game_id", "game_date", "team", "opponent",
        "points_for", "points_against", "win"
    ]
    prior_cols = [c for c in long.columns if c.startswith("prior_")]
    carry = base_cols + prior_cols

    home_side = home_long[carry].add_prefix("home_")
    away_side = away_long[carry].add_prefix("away_")

    # Join back to one row per game on game_id
    wide = (home_side
            .merge(away_side, left_on="home_game_id", right_on="away_game_id", how="inner"))

    # Clean duplicate ID columns and rename to canonical
    wide = wide.rename(columns={"home_game_id": "game_id"}).drop(columns=["away_game_id"])

    # Compute helpful game-level fields
    wide["point_diff"] = wide["home_points_for"] - wide["away_points_for"]
    wide["winner"] = np.where(wide["point_diff"] > 0, wide["home_team"], np.where(
        wide["point_diff"] < 0, wide["away_team"], "TIE"
    ))

    # Final sort: earliest → latest
    wide["season"] = wide["home_season"].astype(int)
    wide["week"] = wide["home_week"].astype(int)
    wide["time_key"] = make_time_key(wide)
    wide = wide.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    # Select and order final columns for clarity
    ordered_cols = [
        "season", "week", "game_id", "home_game_date",  # identifiers
        "home_team", "away_team",
        "home_points_for", "away_points_for", "point_diff", "winner",
    ]

    # Append prior features after the core columns
    prior_home = sorted([c for c in wide.columns if c.startswith("home_prior_")])
    prior_away = sorted([c for c in wide.columns if c.startswith("away_prior_")])

    final_cols = ordered_cols + prior_home + prior_away

    # Some columns were prefixed; align names:
    final = wide[final_cols].rename(columns={"home_game_date": "game_date"})
    return final


# -----------------------------
# Main
# -----------------------------

def build_dataset(start: int, end: int, out_dir: Path) -> Path:
    """
    Orchestrates the build:
      1) Load schedules
      2) Add leak-free features
      3) Write single CSV sorted by time
    """
    seasons = list(range(int(start), int(end) + 1))
    logging.info("Building dataset for seasons=%s", seasons)

    schedules = load_schedules(seasons)

    # Final feature-augmented, one row per game
    final_df = add_features(schedules, windows=(3, 5))
    final_df[final_df.filter(regex=r'^(home|away)_prior_').columns] = final_df.filter(regex=r'^(home|away)_prior_').fillna(final_df.filter(regex=r'^(home|away)_prior_').mean())
    

    dff = final_df.copy()
    print('last 5 rows', dff.tail())
    print(dff.columns.str.strip())
    # sort by date
    dff = dff.sort_values(by='game_date')
    print(dff[['game_date', 'home_team', 'away_team', 'home_points_for', 'away_points_for']].tail(20))
    dff.to_csv('Nfl_data_sorted.csv', index=False)

    # Write once
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Nfl_data.csv"
    final_df.to_csv(out_path, index=False)
    logging.info("Wrote %s with %d games", out_path, len(final_df))
    return out_path, dff


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build NFL game-level dataset (one row per game).")
    p.add_argument("--start", type=int, default=2014, help="Start season (inclusive).")
    p.add_argument("--end", type=int, default=2025, help="End season (inclusive).")
    p.add_argument("--out-dir", type=str, default="backend/data", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    setup_logger(out_dir)
    try:
        out_path = build_dataset(args.start, args.end, out_dir)
        logging.info("Done. Dataset at %s", out_path)
    except Exception as e:
        logging.exception("Build failed: %s", e)
        raise


if __name__ == "__main__":
    main()
