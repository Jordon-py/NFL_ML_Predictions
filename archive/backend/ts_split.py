# ==========================================
# File: backend/ts_split.py
# Role: Backend helper module.
# Input Data: Function inputs.
# Output Data: Module outputs.
# Dependencies: typing, pandas
# Notes: Shared utilities.
# ==========================================

# ts_split.py
from typing import Tuple, List
import pandas as pd



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
    targets: List[str],
    train_end: Tuple[int, int],  # (season, week), inclusive
    val_end: Tuple[int, int]     # (season, week), inclusive
):
    """
    Chronological split that prevents leakage.
    Train ≤ train_end, Val (train_end, val_end], Test > val_end.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    data = df.copy()

    # Validate required columns early to fail fast
    required_cols = {"season", "week"} | set(features) | set(targets)
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Always sort time so cuts behave as expected
    data["time_key"] = make_time_key(data)
    data = data.sort_values(["time_key"]).reset_index(drop=True)

    # Precompute integer cutoffs once
    train_cut = train_end[0] * 100 + train_end[1]
    val_cut   = val_end[0] * 100 + val_end[1]

    # Boolean masks read like English
    is_train = data["time_key"] <= train_cut
    is_val   = (data["time_key"] > train_cut) & (data["time_key"] <= val_cut)
    is_test  = data["time_key"] > val_cut

    # Split sets
    train_df = data.loc[is_train].copy()
    val_df   = data.loc[is_val].copy()
    test_df  = data.loc[is_test].copy()

    # Final matrices
    X_train, y_train = train_df[features], train_df[targets]
    X_val,   y_val   = val_df[features],   val_df[targets]
    # Test target may be absent during real deployment, so use safe selection
    X_test, y_test = test_df[features], (test_df[targets] if set(targets).issubset(test_df.columns) else None)

    # Clean up helper column before returning
    for d in (train_df, val_df, test_df):
        if "time_key" in d.columns:
            d.drop(columns=["time_key"], inplace=True)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



