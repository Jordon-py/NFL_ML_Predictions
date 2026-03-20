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
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

# Try to import nflreadpy
try:
    import nflreadpy as nfl
    HAS_NFLREADPY = True
except ImportError:
    HAS_NFLREADPY = False

from backend.utils.feature_engine import calculate_team_metrics, calculate_rolling_features
from backend.utils.feature_helpers import to_pandas_safe, make_time_key, process_dataset

log = logging.getLogger(__name__)


class LiveDataCache:
    """Simple in-memory cache with TTL."""
    def __init__(self, ttl_seconds: int = 3600):
        self._cache = {}
        self._ttl = ttl_seconds
    
    def get(self, key: str):
        if key not in self._cache:
            return None
        data, timestamp = self._cache[key]
        if time.time() - timestamp < self._ttl:
            return data
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
        log.info("Using cached live data for seasons %s", seasons)
        return cached

    log.info("Fetching live data from nflreadpy for seasons %s...", seasons)
    
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
    seasons_to_load = {season}
    if week < 5:
        seasons_to_load.add(season - 1)
    pbp, sch = fetch_live_data(sorted(seasons_to_load))
    
    if pbp.empty or sch.empty:
        raise RuntimeError("Could not fetch live PBP/Schedule data.")
        
    # --- 1. Prepare Features (Reusing Feature Engine) ---
    # Metric aggregation
    metrics = calculate_team_metrics(pbp)
    
    # --- 2. Convert Schedule to Long Format ---
    # We need a 'long' format schedule to merge with metrics and do rolling averages
    # We must ensure we replicate the logic from build_csv_datasetsv3
    
    # Normalize schedule
    s = process_dataset(s.copy())  # Coerce types
    
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

    prior_cols = [c for c in long.columns if c.startswith("prior_")]
    target_time = season * 100 + week
    row_dict = {
        "season": season,
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
        "time_key": target_time,
    }

    def _fill_priors(target: Dict[str, Any], prefix: str, source: pd.Series | None):
        if source is None:
            return
        for c in prior_cols:
            target[f"{prefix}_{c}"] = source.get(c)

    if target_game.empty:
        log.warning(
            "Matchup %s vs %s not found in schedule. Constructing synthetic row.",
            home_team,
            away_team,
        )
        home_hist = (
            long[(long["team"] == home_team) & (long["time_key"] < target_time)]
            .sort_values("time_key")
            .iloc[-1:]
        )
        away_hist = (
            long[(long["team"] == away_team) & (long["time_key"] < target_time)]
            .sort_values("time_key")
            .iloc[-1:]
        )
        _fill_priors(row_dict, "home", home_hist.iloc[0] if not home_hist.empty else None)
        _fill_priors(row_dict, "away", away_hist.iloc[0] if not away_hist.empty else None)
    else:
        gid = target_game.iloc[0].get("game_id")
        h_row = long[(long["game_id"] == gid) & (long["team"] == home_team)]
        a_row = long[(long["game_id"] == gid) & (long["team"] == away_team)]
        if h_row.empty or a_row.empty:
            raise ValueError("Could not find processed rows for game.")
        row_dict["game_id"] = gid
        _fill_priors(row_dict, "home", h_row.iloc[0])
        _fill_priors(row_dict, "away", a_row.iloc[0])
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
    Run inference on a single pre-built row using the bundle's models.
    """
    if row_df is None or row_df.empty:
        raise ValueError("Cannot infer from empty row.")

    meta_row = row_df.iloc[0]
    home_team = str(meta_row.get("home_team", "UNKNOWN"))
    away_team = str(meta_row.get("away_team", "UNKNOWN"))
    season = int(meta_row.get("season", 0))
    week = int(meta_row.get("week", 0))

    preprocessor = getattr(bundle, "preprocessor", None)
    raw_cols = getattr(preprocessor, "feature_names_in_", None)
    if raw_cols is None:
        raw_cols = getattr(bundle, "raw_feature_columns", None)
        if isinstance(raw_cols, dict):
            raw_cols = (raw_cols.get("numeric") or []) + (raw_cols.get("categorical") or [])
    raw_cols = list(raw_cols) if raw_cols is not None else list(row_df.columns)

    row_features = row_df.reindex(columns=raw_cols, fill_value=0)
    X = preprocessor.transform(row_features) if preprocessor is not None else row_features

    home_raw = float(bundle.home_model.predict(X)[0])
    away_raw = float(bundle.away_model.predict(X)[0])
    point_diff = home_raw - away_raw

    def _extract_home_proba(win_clf: Any, proba_row: np.ndarray) -> float | None:
        classes = getattr(win_clf, "classes_", None)
        if classes is None:
            return None
        classes_list = list(classes)
        lowered = [str(c).strip().lower() for c in classes_list]
        if "home" in lowered:
            return float(proba_row[lowered.index("home")])
        if "true" in lowered:
            return float(proba_row[lowered.index("true")])
        for key in (1, True):
            if key in classes_list:
                return float(proba_row[classes_list.index(key)])
        return None

    win_clf = getattr(bundle, "hist_win_clf", None) or getattr(bundle, "win_clf", None)
    used_fallback = True
    if win_clf is not None and hasattr(win_clf, "predict_proba"):
        probs = np.asarray(win_clf.predict_proba(X)[0], dtype=float)
        mapped = _extract_home_proba(win_clf, probs)
        if mapped is not None and np.isfinite(mapped):
            proba_home = float(mapped)
            used_fallback = False
        else:
            proba_home = float(1.0 / (1.0 + np.exp(-point_diff / 7.0)))
    else:
        proba_home = float(1.0 / (1.0 + np.exp(-point_diff / 7.0)))

    final_win_prob = float(np.clip(proba_home, 0.0, 1.0))
    winner = home_team if final_win_prob >= 0.5 else away_team

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "season": int(season),
        "week": int(week),
        "predicted_home_score": round(home_raw),
        "predicted_away_score": round(away_raw),
        "win_probability": float(final_win_prob),
        "away_win_probability": float(1.0 - float(final_win_prob)),
        "winner": winner,
        "prob_used_fallback": bool(used_fallback),
        "details": {
            "raw_home": float(home_raw),
            "raw_away": float(away_raw),
            "point_diff": float(point_diff),
        }
    }

    return result, used_fallback
