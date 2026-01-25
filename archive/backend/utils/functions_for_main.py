# ==========================================
# File: backend/utils/functions_for_main.py
# Role: Backend helper module.
# Input Data: Function inputs.
# Output Data: Module outputs.
# Dependencies: pandas, logging, nflreadpy, numpy
# Notes: Shared utilities.
# ==========================================

import pandas as pd
import logging
import nflreadpy as nfl
import numpy as np
from fastapi import HTTPException
from typing import Any, Dict, List, Optional, Tuple
import os
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


# --------------------------------------------------------------------
# Constants and helper functions
# Team abbreviation normalization map (handles legacy/ambiguous codes like LA->LAR).
# --------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TEAM_ABBR_MAP: Dict[str, str] = {}
try:
    _abbr_map_path = DATA_DIR / "team_abbr_map.json"
    if _abbr_map_path.exists():
        with open(_abbr_map_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh) or {}
        if isinstance(raw, dict):
            TEAM_ABBR_MAP = {
                str(k).strip().upper(): str(v).strip().upper()
                for k, v in raw.items()
                if str(k).strip() and str(v).strip()
            }
            if TEAM_ABBR_MAP:
                logging.info("[Teams] Loaded %d abbreviation aliases from %s", len(TEAM_ABBR_MAP), _abbr_map_path)
except Exception as e:
    logging.warning("[Teams] Failed to load team_abbr_map.json: %s", e)

def _add_kickoff_utc_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: DataFrame with 'gameday' and optional 'gametime' columns.
    Returns:
        pd.DataFrame: Copy with added 'dt' column for kickoff datetime in UTC.

    Kickoff timestamp (UTC)
        ** ----------------------
        ** schedule CSV often has:
        **  - gameday (date) and
        **  - gametime (clock time, typically Eastern)
        **
        ** We combine them into a single timezone-aware UTC timestamp so "next week"
        ** election works correctly on Heroku (which compares against UTC 'now').

    """

    df["dt"] = pd.NaT

    if "gameday" in df.columns:

        if "gametime" in df.columns:
            kickoff_str = (
                df["gameday"].astype(str).str.strip()
                + " "
                + df["gametime"].astype(str).str.strip()
            )
            logging.debug("[Schedule] kickoff_str sample: %s", kickoff_str.iloc[0] if len(kickoff_str) else "")
            kickoff_naive = pd.to_datetime(kickoff_str, errors="coerce")

            try:
                df["dt"] = (
                    kickoff_naive.dt.tz_localize(
                        "America/New_York",
                        ambiguous="NaT",
                        nonexistent="shift_forward",
                    ).dt.tz_convert("UTC")
                )
            except Exception:
                # llback if tzdata is unavailable in the runtime.
                df["dt"] = kickoff_naive.dt.tz_localize(
                    "UTC",
                    ambiguous="NaT",
                    nonexistent="shift_forward",
                )
            return df
        else:
            # No gametime column: treat gameday as "upcoming" until end-of-day Eastern.
            d = pd.to_datetime(df["gameday"], errors="coerce") + pd.Timedelta(hours=23, minutes=59)
            try:
                df["dt"] = (
                    d.dt.tz_localize(
                        "America/New_York",
                        ambiguous="NaT",
                        nonexistent="shift_forward",
                    ).dt.tz_convert("UTC")
                )
            except Exception:
                df["dt"] = d.dt.tz_localize(
                    "UTC",
                    ambiguous="NaT",
                    nonexistent="shift_forward",
                )
            return df



def _coerce_season_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce season/week columns to integers when present.

    Handles both 'season'/'week' and 'season_num'/'week_num' variants.
    """
    for col in ("season", "season_num"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ("week", "week_num"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _normalize_team_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Normalize team abbreviation/name columns to uppercase strings.

    Args:
        df: DataFrame containing team columns.
        cols: List of column names to normalize.
    Returns:
        pd.DataFrame: Copy of input with normalized team columns.

    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
    return df


# -------------------------------------------------------------------
# Additions that support a simpler + more accurate /predict
# -------------------------------------------------------------------

def _normalize_team_code(team: str) -> str:
    """
    Normalize a team identifier into a stable uppercase code.

    - Trims whitespace
    - Uppercases
    - Applies TEAM_ABBR_MAP aliasing (LA -> LAR, etc.) when present
    """
    t = (team or "").strip().upper()
    if TEAM_ABBR_MAP:
        t = TEAM_ABBR_MAP.get(t, t)
    return t


def _get_game_row(
    df: pd.DataFrame,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    """
    Find the single best matching dataset row for a game.

    Matching strategies (in order):
      A) (home_team, away_team) columns if present
      B) (home_abbr, away_abbr) columns if present

    If multiple rows match, we take the first (deterministic) and log a warning.
    """
    required = {"season", "week"}
    if df is None or df.empty or not required.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded or missing required season/week columns.",
        )

    base_mask = (df["season"] == season) & (df["week"] == week)

    team_masks: List[pd.Series] = []

    if {"home_team", "away_team"}.issubset(df.columns):
        team_masks.append((df["home_team"] == home_team) & (df["away_team"] == away_team))

    if {"home_abbr", "away_abbr"}.issubset(df.columns):
        team_masks.append((df["home_abbr"] == home_team) & (df["away_abbr"] == away_team))

    if not team_masks:
        raise HTTPException(
            status_code=500,
            detail="Dataset is missing home/away team identifier columns.",
        )

    team_mask = team_masks[0]
    for m in team_masks[1:]:
        team_mask = team_mask | m

    row = df.loc[base_mask & team_mask]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail="Game data not found for given season/week/teams.",
        )

    if len(row) > 1:
        logging.warning(
            "[Predict] Duplicate rows matched (%d). Using the first row. season=%s week=%s %s vs %s",
            len(row), season, week, home_team, away_team
        )
        row = row.iloc[[0]]

    return row.copy()  # avoid SettingWithCopy surprises


def _prepare_inputs(row_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare (full_df, numeric_df) views for model inference.

    - full_df keeps all columns (for pipelines/preprocessors that expect names)
    - numeric_df keeps numeric-only columns (for estimators trained on numeric matrices)
    - inf -> NaN to prevent silent numeric explosions
    """
    full_df = row_df.drop(columns=["game_id"], errors="ignore").copy()
    full_df = full_df.replace([np.inf, -np.inf], np.nan)

    numeric_df = full_df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    return full_df, numeric_df


def _is_pipeline(model: Any) -> bool:
    """Heuristic: sklearn Pipeline objects usually have .steps or .named_steps."""
    return bool(getattr(model, "steps", None) is not None or getattr(model, "named_steps", None) is not None)


def _align_numeric_df_for_model(
    numeric_df: pd.DataFrame,
    model: Any,
    numeric_medians: Optional[pd.Series],
) -> pd.DataFrame:
    """
    Align numeric input for non-pipeline estimators.

    Key idea:
      - If the estimator exposes feature_names_in_, we respect it (order + selection).
      - We fill missing numeric features using dataset-derived medians (computed once at startup).
    """
    if numeric_df is None:
        numeric_df = pd.DataFrame()

    # Reindex into the training-time feature order when available.
    cols = list(getattr(model, "feature_names_in_", [])) if hasattr(model, "feature_names_in_") else []
    if cols:
        aligned = numeric_df.reindex(columns=cols)
    else:
        aligned = numeric_df.copy()

    # Coerce numeric just in case (single-row DFs can carry object dtype surprises).
    aligned = aligned.apply(lambda c: pd.to_numeric(c, errors="coerce"))

    if numeric_medians is not None and not numeric_medians.empty:
        fill = numeric_medians.reindex(aligned.columns)
        aligned = aligned.fillna(fill)

    # Last resort: still fill remaining NaNs (only if medians were missing or columns unknown).
    aligned = aligned.fillna(0.0)
    return aligned


def _predict_score(
    model: Any,
    full_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    *,
    preprocessor: Optional[Any],
    numeric_medians: Optional[pd.Series],
    model_name: str,
) -> float:
    """
    Predict a single score with layered fallbacks, without retraining.

    Order:
      1) If Pipeline: model.predict(full_df)
      2) If estimator: align numeric features (feature_names_in_ if present) + fill NaNs via medians
      3) If that fails and a standalone preprocessor exists: preprocessor.transform(full_df) then predict
    """
    try:
        if _is_pipeline(model):
            return float(model.predict(full_df)[0])

        aligned = _align_numeric_df_for_model(numeric_df, model, numeric_medians)

        # Many sklearn estimators accept a DataFrame, some prefer ndarray. Try DF first.
        try:
            return float(model.predict(aligned)[0])
        except Exception:
            return float(model.predict(aligned.to_numpy())[0])

    except Exception as err:
        msg = str(err) or ""
        logging.warning("[Predict] %s failed: %s", model_name, msg.splitlines()[0] if msg else "unknown")

        # Fallback: some exported bundles expect "preprocessed arrays" at inference.
        if preprocessor is not None and not _is_pipeline(model):
            try:
                X_proc = preprocessor.transform(full_df)
                return float(model.predict(X_proc)[0])
            except Exception as prep_err:
                logging.warning(
                    "[Predict] %s preprocessor fallback failed: %s",
                    model_name,
                    str(prep_err).splitlines()[0] if prep_err else "unknown",
                )

        raise  # surface the original failure


def _clamp_score(x: float) -> float:
    """Keep predicted scores in a plausible NFL range."""
    try:
        return float(np.clip(float(x), 0.0, 60.0))
    except Exception:
        return 0.0


def _sigmoid(z: float) -> float:
    z = float(z)
    return float(1.0 / (1.0 + np.exp(-z)))


def _smooth_win_probability(p_raw: float, point_diff: float, *, clf_used: bool) -> float:
    """
    Output smoothing without retraining:
      - Blend classifier p with a logistic(point_diff) baseline
      - Shrink extremes slightly toward 0.5
      - Clamp to [0.02, 0.98] to avoid UI and math edge cases
    """
    p_raw = float(np.clip(float(p_raw), 0.0, 1.0))

    # Baseline: logistic on point differential (your existing fallback concept).
    p_diff = _sigmoid(0.28 * float(point_diff))

    # If classifier was used, trust it more, but keep the diff sanity signal in the loop.
    w = 0.75 if clf_used else 0.0
    p = (w * p_raw) + ((1.0 - w) * p_diff)

    # Shrink overconfidence.
    shrink = 0.92 if clf_used else 0.97
    p = 0.5 + (p - 0.5) * shrink

    return float(np.clip(p, 0.02, 0.98))


# -------------------------------------------------------------------
# IMPORTANT FIX: do NOT overwrite row_df by reloading schedules
# (Your current helper does this and can corrupt prediction input.)
# -------------------------------------------------------------------
def _is_missing_value(v: Any) -> bool:
    """Return True if v should be treated as missing (NaN/None/empty-string)."""
    try:
        if v is None:
            return True
        if pd.isna(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
    except Exception:
        # If pd.isna raises (rare), fall back to basic checks.
        return v is None
    return False

def _last_team_game_row(
    df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
) -> Optional[pd.Series]:
    """
    Return the most recent *prior* game row for a given team before (season, week).

    This supports "roll-forward" of player/team stats into future weeks without leakage.

    Matching:
      - home_team/away_team if present
      - home_abbr/away_abbr as fallbacks

    Returns:
      pd.Series of the row (a single game) or None if not found.
    """
    if df is None or df.empty:
        return None
    if "season" not in df.columns or "week" not in df.columns:
        return None

    t = (team or "").strip().upper()

    # "Prior" means strictly earlier than the requested week in the same season,
    # or any week in an earlier season.
    prior_mask = (df["season"] < int(season)) | ((df["season"] == int(season)) & (df["week"] < int(week)))

    team_mask = None
    if {"home_team", "away_team"}.issubset(df.columns):
        team_mask = (df["home_team"].astype(str).str.upper() == t) | (df["away_team"].astype(str).str.upper() == t)
    elif {"home_abbr", "away_abbr"}.issubset(df.columns):
        team_mask = (df["home_abbr"].astype(str).str.upper() == t) | (df["away_abbr"].astype(str).str.upper() == t)
    else:
        return None

    sub = df.loc[prior_mask & team_mask]
    if sub.empty:
        return None

    # Deterministic: choose the latest (season, week), and if ties, the last row.
    sub = sub.sort_values(["season", "week"])
    return sub.iloc[-1]


def _roll_forward_missing_player_stats(
    df: pd.DataFrame,
    row_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    Fill missing player-stat-like features for future games using last known team values.

    NOTE:
      This function must operate on the provided row_df (the matched feature row).
      It should NOT reload schedules (that swaps feature columns for schedule columns).
    """
    if row_df is None or row_df.empty:
        return row_df

    idx = row_df.index[0]
    filled = 0

    last_home = _last_team_game_row(df, home_team, season, week)
    last_away = _last_team_game_row(df, away_team, season, week)

    last_home_side = None
    if last_home is not None:
        last_home_side = "home" if str(last_home.get("home_team", "")).upper() == home_team.upper() else "away"

    last_away_side = None
    if last_away is not None:
        last_away_side = "home" if str(last_away.get("home_team", "")).upper() == away_team.upper() else "away"

    # Home-side roll forward
    if last_home is not None and last_home_side:
        for col in row_df.columns:
            if not (col.startswith("home_player_team_") or col == "home_qb_completion_pct"):
                continue
            if not _is_missing_value(row_df.at[idx, col]):
                continue

            base = col[len("home_"):]
            src_col = f"{last_home_side}_{base}"
            if src_col in last_home.index and not _is_missing_value(last_home.get(src_col)):
                row_df.at[idx, col] = last_home.get(src_col)
                filled += 1

    # Away-side roll forward
    if last_away is not None and last_away_side:
        for col in row_df.columns:
            if not (col.startswith("away_player_team_") or col == "away_qb_completion_pct"):
                continue
            if not _is_missing_value(row_df.at[idx, col]):
                continue

            base = col[len("away_"):]
            src_col = f"{last_away_side}_{base}"
            if src_col in last_away.index and not _is_missing_value(last_away.get(src_col)):
                row_df.at[idx, col] = last_away.get(src_col)
                filled += 1

    if filled:
        logging.info(
            "[Predict] Rolled forward %d player-stat features for %s vs %s (season=%s week=%s)",
            filled, home_team.upper(), away_team.upper(), season, week
        )

    return row_df


