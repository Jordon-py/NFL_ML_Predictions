# ==========================================
# File: backend/services/live_predictor.py
# Role: Backend service module.
# Input Data: Service inputs and config.
# Output Data: Service outputs.
# Dependencies: logging, time, typing, datetime
# Notes: Used by API handlers.
# ==========================================

"""
live_predictor.py
=================

Service to handle live data fetching using nflreadpy, with in-memory caching
to prevent performance bottlenecks. Orchestrates the generation of a feature 
row for a specific matchup on-the-fly.
"""

import logging
import time
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Try to import nflreadpy
try:
    import nflreadpy as nfl
    HAS_NFLREADPY = True
except ImportError:
    HAS_NFLREADPY = False

from backend.utils.feature_engine import calculate_team_metrics, calculate_rolling_features
from backend.utils.feature_helpers import to_pandas_safe, make_time_key, process_dataset, ensure_actual_winner

log = logging.getLogger(__name__)


class LiveDataCache:
    """Simple in-memory cache with TTL."""
    def __init__(self, ttl_seconds: int = 3600):
        self._cache = {}
        self._ttl = ttl_seconds
    
    def get(self, key: str):
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return data
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self._cache[key] = (value, time.time())
        # Basic cleanup if cache grows too large? 
        # For now, we only store a few seasons of data, so memory isn't huge constraint.


# Global Cache instance
_CACHE = LiveDataCache(ttl_seconds=3600)  # 1 hour cache


def fetch_live_data(seasons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch PBP and Schedule data for the requested seasons.
    Uses nflreadpy if available.
    """
    if not HAS_NFLREADPY:
        raise RuntimeError("nflreadpy is required for live predictions but not installed.")
    
    cache_key = f"live_data_{min(seasons)}_{max(seasons)}"
    cached = _CACHE.get(cache_key)
    if cached:
        log.info(f"Using cached live data for seasons {seasons}")
        return cached

    log.info(f"Fetching live data from nflreadpy for seasons {seasons}...")
    
    # 1. Load PBP
    # We want full PBP to compute team metrics
    try:
        pbp_raw = nfl.load_pbp(seasons=seasons)
        pbp = to_pandas_safe(pbp_raw)
    except Exception as e:
        log.error(f"Failed to load PBP: {e}")
        pbp = pd.DataFrame()

    # 2. Load Schedules
    try:
        sch_raw = nfl.load_schedules(seasons=seasons)
        sch = to_pandas_safe(sch_raw)
    except Exception as e:
        log.error(f"Failed to load Schedules: {e}")
        sch = pd.DataFrame()
        
    _CACHE.set(cache_key, (pbp, sch))
    return pbp, sch


def build_live_row(
    home_team: str, 
    away_team: str, 
    season: int, 
    week: int
) -> pd.DataFrame:
    """
    Constructs the exact feature row required for inference for a specific game.
    Fetches data up to the requested week (exclusive of the game itself if it's in the future, 
    or inclusive if we are re-simulating a past game? 
    Actually, leak-free means we only use data PRIOR to the game).
    
    Strategy:
    1. Fetch data for this season (and maybe previous if early season? 
       For simplicity start with current season + previous if week < 4).
    2. Calculate team metrics (EPA etc) for all available games.
    3. Calculate rolling features.
    4. Construct the specific row for (Home vs Away) at (Season, Week).
    """
    
    # Determine window of data needed. 
    # To get rolling 5 averages, we likely need the current season plus 
    # maybe previous season if we are early in the year.
    seasons_to_load = [season]
    if week < 5:
        seasons_to_load.append(season - 1)
        
    pbp, sch = fetch_live_data(sorted(list(set(seasons_to_load))))
    
    if pbp.empty or sch.empty:
        raise RuntimeError("Could not fetch live PBP/Schedule data.")
        
    # --- 1. Prepare Features (Reusing Feature Engine) ---
    # Metric aggregation
    metrics = calculate_team_metrics(pbp)
    
    # --- 2. Convert Schedule to Long Format ---
    # We need a 'long' format schedule to merge with metrics and do rolling averages
    # We must ensure we replicate the logic from build_csv_datasetsv3
    
    # Normalize schedule
    s = sch.copy()
    s = process_dataset(s) # Coerce types
    
    # Filter to relevant columns to match builder
    keep_cols = [
        "season", "week", "game_id", "gameday", 
        "home_team", "away_team", "home_score", "away_score",
        "away_moneyline", "home_moneyline", "spread_line", "total_line",
        "away_rest", "home_rest", "game_type"
    ]
    
    # Ensure scores exist (init with NaN if missing from nflreadpy)
    if "home_score" not in s.columns:
        s["home_score"] = np.nan
    if "away_score" not in s.columns:
        s["away_score"] = np.nan
        
    # Be forgiving with columns
    avail = [c for c in keep_cols if c in s.columns]
    s = s[avail].copy()
    if "gameday" in s.columns:
        s = s.rename(columns={"gameday": "game_date"})
        
    # Create Long Format
    home_df = s.rename(columns={
        "home_team": "team", "away_team": "opponent", 
        "home_score": "points_for", "away_score": "points_against"
    })
    home_df["is_home"] = 1
    
    away_df = s.rename(columns={
        "away_team": "team", "home_team": "opponent", 
        "away_score": "points_for", "home_score": "points_against"
    })
    away_df["is_home"] = 0
    
    long = pd.concat([home_df, away_df], ignore_index=True)
    
    # Calculate "win" column (1.0 for win, 0.0 for loss, NaN for future)
    # mirroring build_csv_datasetsv3 logic
    pf = pd.to_numeric(long["points_for"], errors="coerce")
    pa = pd.to_numeric(long["points_against"], errors="coerce")
    mask_complete = pf.notna() & pa.notna()
    
    long["win"] = np.nan
    long.loc[mask_complete, "win"] = (pf[mask_complete] > pa[mask_complete]).astype(float)

    long["time_key"] = make_time_key(long)
    long = long.sort_values(["team", "time_key"])
    
    # --- 3. Merge Advanced Metrics ---
    if not metrics.empty:
        long = long.merge(metrics, on=["season", "week", "game_id", "team"], how="left")
        
    # --- 4. Rolling Calculations ---
    from backend.utils.feature_engine import calculate_rolling_features
    long = calculate_rolling_features(long, windows=(3, 5))
    
    # --- 5. Extract/Assemble the Target Row ---
    # We need the "HOME" perspective row for the specific matchup
    # But wait, the standard builder creates a WIDE row (home_... and away_...)
    
    # We need to find the specific game if it exists in schedule, or fabricate it if it's a hypothetical partial row
    
    target_game = s[
        (s["season"] == season) & 
        (s["week"] == week) & 
        (s["home_team"] == home_team) & 
        (s["away_team"] == away_team)
    ]
    
    if len(target_game) == 0:
        # Hypothetical / Future game not in schedule yet?
        # Or just didn't find it.
        # We can reconstruct it from the 'long' components of the TEAMS involved.
        log.warning(f"Matchup {home_team} vs {away_team} not found in schedule. Constructing synthetic row.")
        
        # We need the PRIOR stats for Home Team and Away Team
        # Get the latest 'long' row for home_team BEFORE this week
        
        # Filter long for correct team and time < target
        target_time = season * 100 + week
        
        home_hist = long[
            (long["team"] == home_team) & 
            (long["time_key"] < target_time)
        ].sort_values("time_key").iloc[-1:] 
        
        away_hist = long[
            (long["team"] == away_team) & 
            (long["time_key"] < target_time)
        ].sort_values("time_key").iloc[-1:]
        
        # If no history (e.g. Week 1), we might have NaNs, which is handled by imputer later
        
        # We need to pull the 'prior_X' columns
        prior_cols = [c for c in long.columns if c.startswith("prior_")]
        
        row_dict = {
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "time_key": target_time
        }
        
        # Fill Home Priors
        if not home_hist.empty:
            for c in prior_cols:
                row_dict[f"home_{c}"] = home_hist.iloc[0][c]
                
        # Fill Away Priors
        if not away_hist.empty:
            for c in prior_cols:
                row_dict[f"away_{c}"] = away_hist.iloc[0][c]
                
        final_row = pd.DataFrame([row_dict])
            
    else:
        # Game exists in schedule (e.g. it's on the schedule for this week)
        gid = target_game.iloc[0]["game_id"]
        
        # Get the calculated 'long' rows for this game
        # The 'long' df has rolling features (which are priors) already attached to this row?
        # WAIT. calculate_rolling_features attaches PRIOR stats to the current row.
        # So row N has stats from N-1, N-2...
        # So we just need to grab the row for this game_id!
        
        h_row = long[(long["game_id"] == gid) & (long["team"] == home_team)]
        a_row = long[(long["game_id"] == gid) & (long["team"] == away_team)]
        
        if h_row.empty or a_row.empty:
            raise ValueError("Could not find processed rows for game.")
            
        # Merge to Wide Format (Home + Away)
        # We manually build the wide row to match training format
        
        base_cols = [c for c in h_row.columns if c.startswith("prior_")]
        
        row_dict = {
            "season": season,
            "week": week,
            "game_id": gid,
            "home_team": home_team,
            "away_team": away_team,
            "time_key": season * 100 + week
        }

        # Copy priors
        for c in base_cols:
            if c in h_row.columns:
                row_dict[f"home_{c}"] = h_row.iloc[0][c]
            if c in a_row.columns:
                row_dict[f"away_{c}"] = a_row.iloc[0][c]
                
        # Copy other context (Moneyline, etc if in schedule)
        # The target_game row has them
        tg = target_game.iloc[0]
        for ctx in ["away_moneyline", "home_moneyline", "spread_line", "total_line", "away_rest", "home_rest"]:
            if ctx in tg:
                row_dict[ctx] = tg[ctx]
                
        final_row = pd.DataFrame([row_dict])

    # --- 6. Final Feature Engineering (Diffs, etc) ---
    # Calculate diffs (home_minus_away_...)
    prior_pairs = [c for c in final_row.columns if c.startswith("home_prior_")]
    for home_col in prior_pairs:
        suffix = home_col[len("home_prior_") :]
        away_col = f"away_prior_{suffix}"
        if away_col in final_row.columns:
             # handle NaNs gracefully
            h_val = final_row[home_col].fillna(0)
            a_val = final_row[away_col].fillna(0)
            final_row[f"home_minus_away_{suffix}"] = h_val - a_val

    # Impute missing if needed (simple fill 0 for now, actual model pipeline has imputer)
    # We leave NaNs for the pipeline to handle? Yes, pipeline has SimpleImputer.
    
    return final_row

def infer_from_row(row_df: pd.DataFrame, bundle: Any) -> Tuple[Dict[str, Any], bool]:
    """
    Run the full inference pipeline (Models + MC Simulation) on a single pre-built row.
    """
    from backend.main import MonteCarloSimulator, _predict_regressor, _predict_home_win_prob, _get_feature_columns
    
    # Validation
    if row_df is None or row_df.empty:
        raise ValueError("Cannot infer from empty row.")

    # Extract metadata before processing
    meta_row = row_df.iloc[0]
    home_team = str(meta_row.get("home_team", "UNKNOWN"))
    away_team = str(meta_row.get("away_team", "UNKNOWN"))
    season = int(meta_row.get("season", 0))
    week = int(meta_row.get("week", 0))

    numeric_cols, categorical_cols, raw_cols = _get_feature_columns(bundle)
    
    # Ensure columns exist
    row = row_df.iloc[0].copy()
    
    # Fallback / Fill missing cols with 0 or defaults if somehow missing
    for c in raw_cols:
        if c not in row:
            row[c] = 0 # Naive fill, but pipeline imputer handles it better if present as NaN?
            
    # Realign
    row_features = row.reindex(raw_cols).to_frame().T
    
    # 1) Score regressors
    home_raw = _predict_regressor(bundle.home_model, bundle.preprocessor, row_features)
    away_raw = _predict_regressor(bundle.away_model, bundle.preprocessor, row_features)

    # 2) Monte Carlo realism
    # Simple game key
    game_key = f"{int(season)}_{int(week)}_{home_team}_{away_team}"
    
    sim_engine = MonteCarloSimulator(bundle)
    sim_results = sim_engine.simulate(home_raw, away_raw, key=game_key)

    # 3) Win prob
    hist_win_prob, used_fallback = _predict_home_win_prob(bundle, row_features, float(home_raw - away_raw))

    # 4) Blend: 75% model + 25% MC
    final_win_prob = (hist_win_prob * 0.75) + (float(sim_results["sim_win_prob"]) * 0.25)
    ens_home_score = (home_raw * 0.75) + (float(sim_results["sim_home_score"]) * 0.25)
    ens_away_score = (away_raw * 0.75) + (float(sim_results["sim_away_score"]) * 0.25)
    
    # 5) Winner string
    if final_win_prob > 0.5:
        winner = home_team
    elif final_win_prob < 0.5:
        winner = away_team
    else:
        winner = "TIE"

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "season": int(season),
        "week": int(week),
        "predicted_home_score": round(ens_home_score),
        "predicted_away_score": round(ens_away_score),
        "win_probability": float(final_win_prob),
        # Helper for away win probability
        "away_win_probability": float(1.0 - float(final_win_prob)),
        "winner": winner,
        "prob_used_fallback": bool(used_fallback),
        "details": {
            **sim_results,
            "raw_home": float(home_raw),
            "raw_away": float(away_raw),
            "ensemble_weight": "75/25",
        }
    }

    return result, False
