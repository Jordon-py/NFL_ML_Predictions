# ==========================================
# File: backend/services/inference_row.py
# Role: Inference row builder for live predictions.
# Input Data: Game context (teams, season, week) + Historical Dataset.
# Output Data: A single-row DataFrame ready for the ML pipeline.
# Dependencies: __future__, typing, numpy, pandas
#
# DESIGN PHILOSOPHY:
# ------------------
# This module constructs a "synthetic" feature row for future games.
# Since we don't have the actual game stats yet, we must:
# 1. Use "Prior" stats: What did these teams do in their LAST game?
# 2. Use "Rolling" stats: Averages of their last N games.
# 3. Use "Schedule" data: Vegas lines, rest days, stadium info.
# 4. Fallback/Impute: Use median values for anything else to be safe.
# ==========================================

from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

# Team Abbreviation Normalization Map
# Ensures consistent team codes across different data sources (ESPN vs NFL vs betting).
ABBR_FIX: Dict[str, str] = {
    "LA": "LAR", "STL": "LAR", "RAMS": "LAR",
    "SD": "LAC", "CHARGERS": "LAC",
    "OAK": "LV", "RAIDERS": "LV",
    "WSH": "WAS", "WAS": "WAS", "COMMANDERS": "WAS", "REDSKINS": "WAS",
    "JAX": "JAX", "JAGUARS": "JAX",
}

# Columns to EXCLUDE from the feature vector.
# These are either "target" variables (answers) or metadata we don't want the model to see.
DROP_COLS = {
    "home_points_for", "away_points_for", "point_diff",
    "winner", "home_win", "actual_winner",
    "game_date", "date", "kickoff", "time"
}

# ==============================================================================
# 1. CORE UTILITIES
# ==============================================================================

def _normalize_team(team: Any) -> str:
    """Standardize team abbreviations to uppercase standard codes."""
    if team is None:
        return ""
    s = str(team).strip().upper()
    return ABBR_FIX.get(s, s)

def _moneyline_to_prob(ml: Any) -> float:
    """Convert American Moneyline odds (-110, +200) to Implied Win Probability (0.0-1.0)."""
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    
    # Negative ML (Favorite): -150 means bet 150 to win 100.
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    # Positive ML (Underdog): +200 means bet 100 to win 200.
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return np.nan

def _infer_expected_columns(preprocessor: Any, raw_feature_columns: Optional[List[str]] = None) -> List[str]:
    """
    Determine the exact list of columns the model expects.
    This is critical to prevent "Shape Mismatch" errors during inference.
    Prioritizes the fitted preprocessor's internal state.
    """
    # 1. Best source: The fitted preprocessor knows exactly what it saw during training.
    if preprocessor is not None:
        cols = getattr(preprocessor, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
        
        # Check nested steps if top-level attribute is missing
        named_steps = getattr(preprocessor, "named_steps", None)
        if isinstance(named_steps, dict):
            for step in named_steps.values():
                cols = getattr(step, "feature_names_in_", None)
                if cols is not None:
                    return list(cols)

    # 2. Fallback: Metadata provided by the model bundle.
    if raw_feature_columns:
        return list(raw_feature_columns)
    
    return []

# ==============================================================================
# 2. HISTORY & CACHING
# ==============================================================================

def build_team_history_cache(dataset_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Optimization: Pre-sort the dataset by team and time.
    Instead of scanning the whole DataFrame for every prediction, we look up this cache.
    Returns: { "KC": DataFrame(sorted by season/week), "BUF": ... }
    """
    if dataset_df is None or dataset_df.empty:
        return {}
    
    required = {"season", "week", "home_team", "away_team"}
    if not required.issubset(dataset_df.columns):
        return {}

    # Create numeric sort keys once
    s = dataset_df.copy()
    s["season_num"] = pd.to_numeric(s["season"], errors="coerce").fillna(0)
    s["week_num"] = pd.to_numeric(s["week"], errors="coerce").fillna(0)
    
    cache: Dict[str, List[pd.Series]] = {}
    
    for _, row in s.iterrows():
        # Standardize team names so we key correctly
        h = _normalize_team(row.get("home_team"))
        a = _normalize_team(row.get("away_team"))
        if h: cache.setdefault(h, []).append(row)
        if a: cache.setdefault(a, []).append(row)
            
    # Convert lists to sorted DataFrames
    out: Dict[str, pd.DataFrame] = {}
    for team, rows in cache.items():
        df = pd.DataFrame(rows)
        out[team] = df.sort_values(["season_num", "week_num"])
        
    return out

def _get_latest_prior_game(
    team: str,
    season: int,
    week: int,
    history_cache: Optional[Dict[str, pd.DataFrame]]
) -> Optional[pd.Series]:
    """
    Find the most recent COMPLETED game for a team before the given (season, week).
    This row contains the 'rolling stats' we need to carry forward.
    """
    team = _normalize_team(team)
    if not history_cache or team not in history_cache:
        return None
        
    df = history_cache[team]
    
    # Filter for games strictly before the target date
    # Logic: (past seasons) OR (current season AND past weeks)
    mask_past = (
        (df["season_num"] < season) | 
        ((df["season_num"] == season) & (df["week_num"] < week))
    )
    
    prior_games = df.loc[mask_past]
    if prior_games.empty:
        return None
        
    # The last row is the most recent game
    return prior_games.iloc[-1]

# ==============================================================================
# 3. ROW CONSTRUCTION STEPS
# ==============================================================================

def _init_base_row(season: int, week: int, home: str, away: str) -> pd.DataFrame:
    """Create the initial 1-row DataFrame with identity columns."""
    return pd.DataFrame([{
        "season": int(season),
        "week": int(week),
        "home_team": _normalize_team(home),
        "away_team": _normalize_team(away),
        # Helper for debugging/tracking
        "api_game_id": f"{int(season)}-{int(week)}-{_normalize_team(home)}-{_normalize_team(away)}",
    }])

def _enrich_from_schedule(row_df: pd.DataFrame, schedule_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Enhance the row with data known from the schedule (Odds, Rest Days, Stadium).
    This doesn't require ML; it's factual data about the upcoming matchup.
    """
    if schedule_df is None or schedule_df.empty:
        return row_df

    season = row_df.at[0, "season"]
    week = row_df.at[0, "week"]
    home = row_df.at[0, "home_team"]
    away = row_df.at[0, "away_team"]

    # Fuzzy match logic inside _pick_schedule_row (omitted for brevity, assumed separate or inline)
    # For now, let's implement a concise finder:
    
    # We filter the schedule df
    mask = (
        (pd.to_numeric(schedule_df["season"], errors="coerce") == season) &
        (pd.to_numeric(schedule_df["week"], errors="coerce") == week) &
        (schedule_df["home_team"].apply(_normalize_team) == home) &
        (schedule_df["away_team"].apply(_normalize_team) == away)
    )
    
    if not mask.any():
        return row_df
        
    sched_row = schedule_df.loc[mask].iloc[0]
    
    # Copy relevant fields if they exist
    fields_to_copy = [
        "spread_line", "total_line", 
        "home_rest", "away_rest", 
        "home_moneyline", "away_moneyline",
        "stadium", "location", "div_game", "roof"
    ]
    
    updates = {}
    for f in fields_to_copy:
        if f in sched_row and pd.notna(sched_row[f]):
            updates[f] = sched_row[f]
            
    # Calculate Probabilities from Moneylines if available
    if "home_moneyline" in updates:
        updates["home_moneyline_prob"] = _moneyline_to_prob(updates["home_moneyline"])
    if "away_moneyline" in updates:
        updates["away_moneyline_prob"] = _moneyline_to_prob(updates["away_moneyline"])
        
    # Calculate Rest Diff
    if "home_rest" in updates and "away_rest" in updates:
        try:
            updates["rest_diff"] = float(updates["home_rest"]) - float(updates["away_rest"])
        except (ValueError, TypeError):
            pass

    if updates:
        row_df = row_df.assign(**updates)
        
    return row_df

def _roll_forward_stats(
    row_df: pd.DataFrame, 
    dataset_cols: List[str],
    history_cache: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    CRITICAL STEP: The "Expert" Logic.
    We don't know the stats for the FUTURE game.
    But we know the teams' stats from their PREVIOUS game.
    
    We take "rolling_*" and "elo_*" and "qb_*" columns from the last game 
    and copy them into this new row.
    
    Performance: Accumulates all updates in a dict then applies in one operation
    to avoid DataFrame fragmentation warnings.
    """
    season = row_df.at[0, "season"]
    week = row_df.at[0, "week"]
    home = row_df.at[0, "home_team"]
    away = row_df.at[0, "away_team"]
    
    # 1. Get last known state for both teams
    home_last = _get_latest_prior_game(home, season, week, history_cache)
    away_last = _get_latest_prior_game(away, season, week, history_cache)
    
    # 2. Accumulate all column updates in a single dict (prevents fragmentation)
    all_updates = {}

    # 3. Process Home Team stats
    if home_last is not None:
        was_home = _normalize_team(home_last["home_team"]) == home
        source_prefix = "home_" if was_home else "away_"
        
        # Find all columns we need to populate for the home team
        for tgt_col in dataset_cols:
            if not tgt_col.startswith("home_"):
                continue
            
            # Extract the core stat name (e.g., "rolling_offensive_epa")
            core_stat = tgt_col[5:]  # len("home_") == 5
            src_col = f"{source_prefix}{core_stat}"
            
            if src_col in home_last.index and pd.notna(home_last[src_col]):
                all_updates[tgt_col] = home_last[src_col]
    
    # 4. Process Away Team stats
    if away_last is not None:
        was_home = _normalize_team(away_last["home_team"]) == away
        source_prefix = "home_" if was_home else "away_"
        
        for tgt_col in dataset_cols:
            if not tgt_col.startswith("away_"):
                continue
            
            core_stat = tgt_col[5:]  # len("away_") == 5
            src_col = f"{source_prefix}{core_stat}"
            
            if src_col in away_last.index and pd.notna(away_last[src_col]):
                all_updates[tgt_col] = away_last[src_col]
    
    # 5. Apply ALL updates in a single operation (no fragmentation)
    if all_updates:
        # Create new columns DataFrame and concat with original
        # This is more efficient than repeated .assign() calls
        new_cols_df = pd.DataFrame([all_updates], index=row_df.index)
        row_df = pd.concat([row_df, new_cols_df], axis=1)
        # Remove duplicates, keeping the rightmost (newest) columns
        row_df = row_df.loc[:, ~row_df.columns.duplicated(keep='last')]
        
    return row_df

def _calculate_derived_diffs(row_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute any "_diff" columns now that we have populated the base stats.
    E.g. elo_diff = home_elo - away_elo
    """
    for col in row_df.columns:
        if col.endswith("_diff"):
            # Try to infer the components.
            # Common pattern: metric_diff comes from home_metric - away_metric
            base = col.rsplit("_diff", 1)[0] # "elo"
            h_col = f"home_{base}"
            a_col = f"away_{base}"
            
            if h_col in row_df and a_col in row_df:
                try:
                    h_val = pd.to_numeric(row_df[h_col], errors='coerce')
                    a_val = pd.to_numeric(row_df[a_col], errors='coerce')
                    # Only update if we have valid numbers and it wasn't already set
                    if pd.notna(h_val).all() and pd.notna(a_val).all():
                         row_df[col] = h_val - a_val
                except Exception:
                    pass
    return row_df

def _impute_remaining_missing(row_df: pd.DataFrame, dataset_df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """
    Final Safety Net.
    If we still have missing values for required columns (e.g. maybe it's Week 1 and there's no history),
    fill them with the Median value from the entire dataset.
    This prevents the model from crashing on NaNs.
    
    Performance: Only computes medians for columns that are actually missing.
    """
    # 1. Ensure all expected columns exist (fill with NaN if missing)
    row_df = row_df.reindex(columns=expected_cols)
    
    # 2. Identify what's still missing
    missing_cols = row_df.columns[row_df.isna().any()].tolist()
    if not missing_cols:
        return row_df
    
    # Edge case: Empty dataset
    if dataset_df is None or dataset_df.empty:
        return row_df.fillna(0)
        
    # 3. Calculate medians ONLY for the missing columns to save time
    # Drop target columns from the dataset first to avoid data leakage
    safe_ds = dataset_df.drop(columns=[c for c in DROP_COLS if c in dataset_df.columns], errors='ignore')
    
    # Only compute medians for numeric columns that are explicitly missing
    numeric_ds = safe_ds.select_dtypes(include=[np.number])
    target_fills = [c for c in missing_cols if c in numeric_ds.columns]
    
    if target_fills:
        medians = numeric_ds[target_fills].median()
        row_df = row_df.fillna(medians)
    
    # Fill any remaining non-numeric missing values with 0 or empty string
    row_df = row_df.fillna(0)
        
    return row_df

# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================

def build_model_input_row(
    *,
    dataset_df: pd.DataFrame,
    preprocessor: Any,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
    schedule_df: Optional[pd.DataFrame] = None,
    raw_feature_columns: Optional[List[str]] = None,
    team_history_cache: Optional[Dict[str, pd.DataFrame]] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, str] | Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Master orchestrator for creating a prediction row.
    
    Returns:
        (DataFrame, source_string)
        OR
        (DataFrame, source_string, debug_info_dict) if debug=True
    """
    
    # A. Check for EXACT match in dataset first.
    # If the game was already played or pre-processed in the CSV, just return it.
    # This is the "Gold Standard" source.
    home_norm = _normalize_team(home_team)
    away_norm = _normalize_team(away_team)
    
    exact_match = dataset_df[
        (dataset_df["season"] == season) & 
        (dataset_df["week"] == week) & 
        (dataset_df["home_team"].apply(_normalize_team) == home_norm) &
        (dataset_df["away_team"].apply(_normalize_team) == away_norm)
    ]
    
    if not exact_match.empty:
        # We found it! Early return.
        row_df = exact_match.head(1).copy()
        expected = _infer_expected_columns(preprocessor, raw_feature_columns)
        if expected:
            row_df = row_df.reindex(columns=expected).fillna(0) # Basic safety fill for exact matches
        
        return (row_df, "dataset_exact", {}) if debug else (row_df, "dataset_exact")

    # B. Construct Synthetic Row (The "Expert" Path)
    source = "synthetic_model_assembly"
    
    # 1. Base Identity
    row = _init_base_row(season, week, home_team, away_team)
    
    # 2. Enrich with Schedule Data (Lines, Rest, etc.)
    row = _enrich_from_schedule(row, schedule_df)
    
    # 3. Roll Forward Team Stats (The heavy lifting)
    cache = team_history_cache if team_history_cache else build_team_history_cache(dataset_df)
    row = _roll_forward_stats(row, dataset_df.columns, cache)
    
    # 4. Re-calculate Diffs (since we just updated the base values)
    row = _calculate_derived_diffs(row)
    
    # 5. One-Hot Encoding (Manual helper if not using pipeline)
    # Note: If your pipeline uses OneHotEncoder, this step might be redundant but safe.
    # We populate "home_team_KC": 1.0, etc. if they exist in the dataset.
    for col in dataset_df.columns:
        if col.startswith("home_team_") or col.startswith("away_team_"):
            # logic: home_team_KC is 1 if home_team == KC
            team_suffix = col.split("_")[-1]
            if col.startswith("home_team_"):
                row[col] = 1.0 if team_suffix == home_norm else 0.0
            else:
                row[col] = 1.0 if team_suffix == away_norm else 0.0

    # 6. Final Alignment & Imputation
    expected_cols = _infer_expected_columns(preprocessor, raw_feature_columns)
    
    debug_stats = {}
    if debug:
        debug_stats["cols_before_align"] = len(row.columns)
        debug_stats["missing_before_impute"] = int(row.isna().sum().sum())
    
    # The 'magic' fix for nan issues: Impute anything left over.
    row = _impute_remaining_missing(row, dataset_df, expected_cols)
    
    if debug:
        debug_stats["cols_final"] = len(row.columns)
        debug_stats["missing_final"] = int(row.isna().sum().sum())
        return row, source, debug_stats

    return row, source
