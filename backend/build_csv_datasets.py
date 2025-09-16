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
- Rolling stats use ``groupby().rolling(...)`` to prevent future leakage.
- Team codes are minimally normalized to limit join mismatches (LA→LAR, STL→LAR, ...).

**IMPORTANT** TO RUN:
python backend/build_csv_datasets.py --start 2010 --end 2025 --out-dir backend/data

"""
from __future__ import annotations

from typing import List, Dict, Tuple
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


def load_schedules(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
    """
    Load schedules + final scores for given seasons.
    
    Args:
        seasons: List of seasons to load
        include_future: If True, includes scheduled games without scores for prediction
    
    Returns
    -------
    DataFrame with:
      ['season','week','game_id','game_date','home_team','away_team',
       'home_score','away_score']
    """
    logging.info("Loading schedules for seasons=%s (include_future=%s)", seasons, include_future)
    sch = nfl.import_schedules(seasons)

    needed = [
        "season", "week", "game_id", "gameday",  # nflverse uses 'gameday'
        "home_team", "away_team", "home_score", "away_score",
        "game_type", "away_moneyline", "home_moneyline"
    ]
    missing = [c for c in needed if c not in sch.columns]
    if missing:
        raise RuntimeError(f"Missing schedule columns: {missing}")

    sch = _normalize_codes(sch, ["home_team", "away_team"])
    sch["week"] = sch["week"].astype(int)  # enforce int for monotonic keys
    sch = sch.rename(columns={"gameday": "game_date"})
    sch = sch[[
        "season", "week", "game_id", "game_date",
        "home_team", "away_team", "home_score", "away_score","game_type", "away_moneyline", "home_moneyline"
    ]].copy()

    if include_future:
        # Keep both completed and scheduled games
        completed = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        
        # For future games, keep the schedule but mark scores as None
        future = sch[sch["home_score"].isna() | sch["away_score"].isna()].copy()
        future["home_score"] = None
        future["away_score"] = None
        
        # Only include regular season games for future predictions
        future = future[future["game_type"] == "REG"].reset_index(drop=True)
        
        logging.info("Loaded %d completed games + %d future games", len(completed), len(future))
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
    
    # Only compute win for completed games
    completed_mask = long["points_for"].notna() & long["points_against"].notna()
    long["win"] = np.where(
        completed_mask,
        (long["points_for"] > long["points_against"]).astype(float),
        np.nan
    )
    
    long["time_key"] = make_time_key(long)

    # Sorted so that groupby() yields strictly prior games
    return long.sort_values(["team", "time_key", "game_id"]).reset_index(drop=True)


def _rolling_prior_stats(long: pd.DataFrame, window: int = 3) -> pd.DataFrame:
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

    # Convenience fields at game-level (handle NaN scores for future games)
    wide["point_diff"] = np.where(
        wide["home_points_for"].notna() & wide["away_points_for"].notna(),
        wide["home_points_for"] - wide["away_points_for"],
        np.nan
    )
    wide["winner"] = np.where(
        wide["point_diff"].notna() & (wide["point_diff"] > 0), wide["home_team"],
        np.where(wide["point_diff"].notna() & (wide["point_diff"] < 0), wide["away_team"], 
                 np.where(wide["point_diff"].notna() & (wide["point_diff"] == 0), "TIE", "TBD"))
    )

    # Chronological sort across seasons/weeks
    wide["season"] = wide["home_season"].astype(int)
    wide["week"] = wide["home_week"].astype(int)
    wide["time_key"] = make_time_key(wide)
    wide = wide.sort_values(["time_key", "game_id"]).reset_index(drop=True)

    # Add opponent-relative (differential) features: home_minus_away_*
    for w in windows:
        # Points for differential
        wide[f"home_minus_away_pf_avg_{w}"] = wide[f"home_prior_pf_avg_{w}"] - wide[f"away_prior_pf_avg_{w}"]
        # Points against differential 
        wide[f"home_minus_away_pa_avg_{w}"] = wide[f"home_prior_pa_avg_{w}"] - wide[f"away_prior_pa_avg_{w}"]
        # Win percentage differential
        wide[f"home_minus_away_win_pct_{w}"] = wide[f"home_prior_win_pct_{w}"] - wide[f"away_prior_win_pct_{w}"]

    # Column ordering: identifiers + outcomes, then priors, then differentials
    ordered_cols = [
        "season", "week", "game_id", "home_game_date",
        "home_team", "away_team",
        "home_points_for", "away_points_for", "point_diff", "winner"]
    
    # Add all prior and differential columns to the ordered list
    prior_feature_cols = [c for c in wide.columns if c.startswith(("home_prior_", "away_prior_"))]
    diff_feature_cols = [c for c in wide.columns if c.startswith("home_minus_away_")]
    final_cols = ordered_cols + prior_feature_cols + diff_feature_cols
    
    # Return the properly ordered DataFrame
    return wide[final_cols]
    




def build_regression_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    alpha: float = 1.0
) -> Pipeline:
    """
    Returns a fit-ready sklearn Pipeline:
      - Numeric: median impute -> scale
      - Categorical: most-frequent impute -> one-hot
      - Estimator: Ridge (L2-regularized linear regression)

    Why StandardScaler(with_mean=False)?
      OneHotEncoder produces a sparse matrix; centering would densify it.
    """

    numeric_steps = Pipeline([
        ("num_impute", SimpleImputer(strategy="median")),
        ("num_scale",  StandardScaler(with_mean=False))
    ])

    categorical_steps = Pipeline([
        ("cat_impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot",    OneHotEncoder(handle_unknown="ignore", sparse=True))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_steps, numeric_features),
        ("cat", categorical_steps, categorical_features),
    ])

    model = Ridge(alpha=alpha)  # Deterministic given inputs; no random_state

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])
    return pipeline



def make_time_key(df: pd.DataFrame) -> pd.Series:
    """
    Combine season and week into a sortable integer: YYYYWW.
    Example: season=2022, week=9 -> 202209
    """
    # Defensive: ensure numeric types
    return (df["season"].astype(int) * 100) + df["week"].astype(int)

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
    # Boolean masks read like English
    is_train = data
   
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
    Returns (season, week) tuple for the most recently completed week.
    """
    from datetime import datetime
    
    current_date = datetime.now()
    current_season = current_date.year
    
    # NFL season spans Sept-Feb, adjust if in early months
    if current_date.month <= 7:
        current_season -= 1
    
    try:
        # Load existing data to determine last completed week
        import pandas as pd
        df = pd.read_csv("Nfl_data_sorted.csv")
        if not df.empty:
            latest_row = df.loc[df.index[-1]]
            return int(latest_row['season']), int(latest_row['week'])
    except (FileNotFoundError, KeyError, ImportError):
        pass
    
    # Default to Week 1 if no data available  
    return current_season, 1

def build_dataset(start: int, end: int, out_dir: Path, production_mode: bool = True, include_future: bool = True):
    """
    Build production-ready NFL dataset with completed games + future scheduled games for prediction.
    
    Args:
        start: Starting season year
        end: Ending season year  
        out_dir: Output directory path
        production_mode: If True, outputs only essential files and uses current NFL timing
        include_future: If True, includes future scheduled games for prediction
    """
    seasons = list(range(int(start), int(end) + 1))
    
    if production_mode:
        current_season, current_week = get_current_nfl_week()
        logging.info("Production dataset build - Current NFL state: %dW%d", current_season, current_week)
    
    logging.info("Building dataset for seasons=%s (include_future=%s)", seasons, include_future)
    
    # Load both completed and scheduled games
    schedules = load_schedules(seasons, include_future=include_future)
    final_df = add_features(schedules, windows=(3, 5))

    # Robust data preparation for production
    prior_mask = final_df.filter(regex=r"^(home|away)_prior_").columns
    diff_mask = final_df.filter(regex=r"^home_minus_away_").columns
    
    # Use median imputation (more robust than mean)
    final_df[prior_mask] = final_df[prior_mask].fillna(final_df[prior_mask].median())
    final_df[diff_mask] = final_df[diff_mask].fillna(final_df[diff_mask].median())

    # Ensure data integrity - only remove rows with null values in critical feature columns
    # Keep future games (which have null scores but valid feature columns)
    critical_feature_cols = [c for c in final_df.columns if c.startswith(('home_prior_', 'away_prior_', 'home_minus_away_'))]
    
    if include_future:
        # For production with future games, only drop rows missing critical features
        final_df = final_df.dropna(subset=critical_feature_cols).reset_index(drop=True)
        logging.info("Kept future games - dropped only rows missing feature data")
    else:
        # For training data, drop any rows with null values
        final_df = final_df.dropna().reset_index(drop=True)
        logging.info("Dropped all rows with any null values (training mode)")
    
    final_df = final_df.sort_values(by="home_game_date").reset_index(drop=True)
    
    # run df through pipeline to ensure no errors
    for col in final_df.select_dtypes(include=['object']).columns:
        final_df[col] = final_df[col].astype(dtype='category')

    numeric_features = final_df.select_dtypes(include=[np.number]).columns.tolist()

    categorical_features = ["home_team", "away_team"]
    
    dff = final_df.copy()
    dff = dff[categorical_features + numeric_features]
    

    # Production output
    out_dir.mkdir(parents=True, exist_ok=True)
    main_output = out_dir / "Nfl_data_sorted.csv"
    final_df.to_csv(main_output, index=False)
    
    # Root copy for backwards compatibility
    final_df.to_csv("Nfl_data_sorted.csv", index=False)
    
    logging.info("Production dataset ready: %s (%d games)", main_output, len(final_df))
    
    # Export team mapping for API consistency
    abbr_json_path = out_dir / "team_abbr_map.json"
    with open(abbr_json_path, "w") as f:
        json.dump(ABBR_FIX, f, indent=2)
    
    return main_output, final_df


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
