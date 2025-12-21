#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_helpers.py
Safe, minimal helpers shared with backend/main.py and dataset builders.

This module intentionally stays lightweight:
  - Provides deterministic team/season/week normalization
  - Supplies leak-safe prior feature utilities used during dataset prep
  - Exposes resolve_model_path so env overrides work consistently

All functions are defensive and no-op when inputs are missing.
"""

import pydantic
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence
from pydantic import BaseModel, B
import numpy as np
import pandas as pd
from config import MODELS_DIR

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def resolve_model_path(key: str, filename: str) -> Path:
    """Resolve a model artifact path with optional env override MODEL_PATH_<KEY>."""
    env_val = os.getenv(f"MODEL_PATH_{key.upper()}")
    if env_val and str(env_val).strip():
        return Path(env_val).expanduser().resolve()
    p = Path(filename)
    if not p.is_absolute():
        p = MODELS_DIR / filename
    return p.resolve()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def to_team_abbr(t: str) -> str:
    fix = {"WSH": "WAS", "HST": "HOU", "CLV": "CLE", "BLT": "BAL", "ARZ": "ARI", "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV"}
    return fix.get(str(t).upper(), str(t).upper())


def coerce_season_week(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    return out


def make_time_key(df: pd.DataFrame) -> pd.Series:
    """Build a monotonic key for chronological sorting."""
    return pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int) * 100 + pd.to_numeric(
        df.get("week", 0), errors="coerce"
    ).fillna(0).astype(int)


def _normalize_feature_cols(raw: Any) -> List[str]:
    """Flatten raw_feature_columns metadata into a simple list."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        cols: List[str] = []
        for key in ("numeric", "categorical"):
            vals = raw.get(key)
            if isinstance(vals, (list, tuple, set, np.ndarray, pd.Index)):
                cols.extend([str(c) for c in vals])
        return cols
    if isinstance(raw, (list, tuple, set, np.ndarray, pd.Index)):
        return [str(c) for c in raw]
    return [str(raw)]


# ---------------------------------------------------------------------------
# Leak-safe priors (used by dataset prep and defensive runtime filling)
# ---------------------------------------------------------------------------
def _rolling_prior_stats(
    team_game_stats: pd.DataFrame,
    window: int,
    advanced_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute rolling *prior* stats per team with shift(1) to avoid leakage."""
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
            if col in df.columns:
                df[f"prior_{col}_{window}"] = safe_rolling_mean(df[col])

    return df


def _ffill_prior_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill prior_* columns per team to keep future weeks NaN-safe."""
    out = wide.copy()
    if "time_key" not in out.columns:
        out["time_key"] = make_time_key(out)

    home_prior_cols = [c for c in out.columns if c.startswith("home_prior_")]
    away_prior_cols = [c for c in out.columns if c.startswith("away_prior_")]

    if home_prior_cols and "home_team" in out.columns:
        out = out.sort_values(["home_team", "time_key", "game_id"]).copy()
        out[home_prior_cols] = out.groupby("home_team", group_keys=False)[home_prior_cols].ffill()

    if away_prior_cols and "away_team" in out.columns:
        out = out.sort_values(["away_team", "time_key", "game_id"]).copy()
        out[away_prior_cols] = out.groupby("away_team", group_keys=False)[away_prior_cols].ffill()

    return out.sort_values(["time_key", "game_id"]).reset_index(drop=True)


def _impute_remaining_prior_nans(wide: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining prior_* NaNs with neutral values (0.0; medians for QB pct)."""
    out = wide.copy()
    prior_cols = [c for c in out.columns if c.startswith(("home_prior_", "away_prior_"))]
    qb_cols = [c for c in prior_cols if "qb_completion_pct" in c]

    median_map = {c: float(out[c].median(skipna=True)) for c in qb_cols if not pd.isna(out[c].median(skipna=True))}

    if prior_cols:
        out[prior_cols] = out[prior_cols].fillna(0.0)
    for c, med in median_map.items():
        out[c] = out[c].where(out[c].notna(), med)
    return out


def ensure_actual_winner(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `home_win` boolean and `actual_winner` string columns exist."""
    out = df.copy()
    if "home_win" in out.columns:
        win_series = pd.Series(out["home_win"], index=out.index, dtype="boolean")
    elif {"home_points_for", "away_points_for"}.issubset(out.columns):
        win_series = pd.Series(out["home_points_for"] > out["away_points_for"], index=out.index, dtype="boolean")
    elif "winner" in out.columns:
        winner_col = out["winner"]
        if pd.api.types.is_bool_dtype(winner_col.dtype):
            win_series = pd.Series(winner_col, index=out.index, dtype="boolean")
        else:
            win_series = pd.Series(pd.NA, index=out.index, dtype="boolean")
            win_series.loc[winner_col == out["home_team"]] = True
            win_series.loc[winner_col == out["away_team"]] = False
    else:
        raise ValueError("Need either 'home_win', scores, or 'winner' to infer outcome.")

    out["home_win"] = win_series
    actual = pd.Series(pd.NA, index=out.index, dtype="string")
    actual.loc[out["home_win"] == True] = out.loc[out["home_win"] == True, "home_team"].astype("string")
    actual.loc[out["home_win"] == False] = out.loc[out["home_win"] == False, "away_team"].astype("string")
    out["actual_winner"] = actual
    return out


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Safe preprocessing hook: coerce season/week and return copy."""
    try:
        return coerce_season_week(df)
    except Exception as e:
        log.warning("process_dataset failed; returning input. Error: %s", e)
        return df
