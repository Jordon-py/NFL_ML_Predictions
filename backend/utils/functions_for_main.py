"""
File: backend/utils/functions_for_main.py

What it does:
    Shared prediction helpers for backend/main.py: schedule time normalization,
    team-code normalization, dataset row lookup, model input preparation, score
    prediction, and leak-safe prior roll-forward.

Data shapes:
    - Schedule rows arrive as pandas DataFrames with season/week/team columns
      plus gameday/gametime, kickoff, or kickoff_utc timestamp fields.
    - Prediction rows leave this module as one-row DataFrames aligned to the
      trained model/preprocessor feature contract.

Syntax notes:
    - Helpers stay as module-level functions because backend/main.py imports
      them directly for FastAPI route execution.
    - Pandas timestamp parsing keeps all schedule comparisons in UTC.

Important functions (line numbers last refreshed 2026-04-30):
    - _add_kickoff_utc_datetime: around line 66
    - _get_game_row_with_source: around line 285
    - _prepare_inputs: around line 395
    - _predict_score: around line 514
    - _roll_forward_missing_player_stats: around line 895

Possible bugs:
    - Mixed naive and timezone-aware schedule strings can break parsing unless
      they are split and normalized before localization.
    - Synthetic fallback rows can mask missing exact dataset coverage.

Enhancement ideas:
    - Move schedule timestamp parsing into a dedicated tested schedule module.
    - Add typed response objects for row lookup diagnostics.
"""

import pandas as pd
import logging
import nflreadpy as nfl
import numpy as np
from fastapi import HTTPException
from typing import Any, Dict, List, Optional, Tuple, Sequence, Literal
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


def _parse_datetime_series_to_utc(values: pd.Series, *, assume_tz: str) -> pd.Series:
    """Parse a string-like Series into timezone-aware UTC timestamps."""
    raw = values.astype("string").str.strip()
    has_explicit_tz = raw.str.contains(r"(?:z|[+-]\d{2}:?\d{2})$", case=False, na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")

    if has_explicit_tz.any():
        parsed.loc[has_explicit_tz] = pd.to_datetime(
            raw.loc[has_explicit_tz],
            errors="coerce",
            utc=True,
        )

    naive_mask = ~has_explicit_tz
    if naive_mask.any():
        naive = pd.to_datetime(raw.loc[naive_mask], errors="coerce")
        try:
            localized = naive.dt.tz_localize(
                assume_tz,
                ambiguous="NaT",
                nonexistent="shift_forward",
            ).dt.tz_convert("UTC")
        except Exception:
            # Runtime fallback when local timezone data is unavailable.
            localized = naive.dt.tz_localize(
                "UTC",
                ambiguous="NaT",
                nonexistent="shift_forward",
            )
        parsed.loc[naive_mask] = localized

    return parsed


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

    df["dt"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    for kickoff_col in ("kickoff_utc", "kickoff"):
        if kickoff_col in df.columns:
            parsed_kickoff = pd.to_datetime(df[kickoff_col], errors="coerce", utc=True)
            df["dt"] = df["dt"].where(df["dt"].notna(), parsed_kickoff)

    if "gameday" in df.columns:

        if "gametime" in df.columns:
            missing_dt = df["dt"].isna()
            kickoff_str = (
                df["gameday"].astype(str).str.strip()
                + " "
                + df["gametime"].astype(str).str.strip()
            )
            logging.debug("[Schedule] kickoff_str sample: %s", kickoff_str.iloc[0] if len(kickoff_str) else "")
            if missing_dt.any():
                parsed = _parse_datetime_series_to_utc(
                    kickoff_str.loc[missing_dt],
                    assume_tz="America/New_York",
                )
                df.loc[missing_dt, "dt"] = parsed
            return df
        else:
            # No gametime column: treat gameday as "upcoming" until end-of-day Eastern.
            missing_dt = df["dt"].isna()
            if missing_dt.any():
                d = (
                    pd.to_datetime(df.loc[missing_dt, "gameday"], errors="coerce")
                    + pd.Timedelta(hours=23, minutes=59)
                )
                df.loc[missing_dt, "dt"] = _parse_datetime_series_to_utc(
                    d.astype("string"),
                    assume_tz="America/New_York",
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


def _normalize_team_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
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
    row, _ = _get_game_row_with_source(
        df=df,
        season=season,
        week=week,
        home_team=home_team,
        away_team=away_team,
    )
    return row


def _merge_duplicate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate candidate rows by preserving the first row as canonical and
    filling only missing cells from subsequent rows.
    """
    if rows is None or rows.empty:
        return rows

    merged = rows.iloc[[0]].copy()
    if len(rows) == 1:
        return merged

    out_idx = merged.index[0]
    for i in range(1, len(rows)):
        src = rows.iloc[i]
        for col in merged.columns:
            try:
                current = merged.at[out_idx, col]
                if pd.isna(current):
                    candidate = src.get(col, np.nan)
                    if not pd.isna(candidate):
                        merged.at[out_idx, col] = candidate
            except Exception:
                continue
    return merged


def _candidate_game_ids(season: int, week: int, home_team: str, away_team: str) -> List[str]:
    """
    Build canonical game_id candidates across legacy formatting variants.
    """
    candidates = [
        f"{season}_{week}_{away_team}_{home_team}",
        f"{season}_{week}_{home_team}_{away_team}",
        f"{season}-{week}-{away_team}-{home_team}",
        f"{season}-{week}-{home_team}-{away_team}",
    ]
    return [str(x).upper() for x in candidates]


def _normalized_game_id_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if "game_id" not in df.columns:
        return None
    return (
        df["game_id"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
    )


def _get_game_row_with_source(
    df: pd.DataFrame,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
) -> Tuple[pd.DataFrame, Literal["dataset_exact", "dataset_fuzzy"]]:
    """
    Find the single best matching dataset row for a game.

    Matching strategies (in order):
      A) Exact {season, week, home_team, away_team} with canonical aliases
      B) Exact {season, week, home_abbr, away_abbr} with canonical aliases
      C) Canonical game_id fallback using legacy-compatible format variants

    If multiple rows match, we merge safely by taking the first row as canonical
    and filling only missing values from duplicates.
    """
    # Hard fail early when the lookup cannot be trustworthy. A prediction row
    # without season/week filtering can silently pick the wrong matchup.
    required = {"season", "week"}
    if df is None or df.empty or not required.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded or missing required season/week columns.",
        )

    # Normalize caller input once, then compare every dataset column against the
    # same canonical team codes. This absorbs legacy aliases like LA/STL -> LAR.
    home_norm = _normalize_team_code(home_team)
    away_norm = _normalize_team_code(away_team)

    # The season/week mask is shared by every strategy so a team-code or
    # game_id match from a different week cannot leak into the prediction.
    base_mask = (df["season"] == int(season)) & (df["week"] == int(week))

    team_masks: List[pd.Series] = []

    if {"home_team", "away_team"}.issubset(df.columns):
        # Prefer full team columns when present; they are the clearest dataset
        # contract and usually come from schedule/enrichment sources.
        home_team_series = (
            df["home_team"]
            .astype(str)
            .str.upper()
            .map(lambda x: TEAM_ABBR_MAP.get(x, x))
        )
        away_team_series = (
            df["away_team"]
            .astype(str)
            .str.upper()
            .map(lambda x: TEAM_ABBR_MAP.get(x, x))
        )
        team_masks.append((home_team_series == home_norm) & (away_team_series == away_norm))

    if {"home_abbr", "away_abbr"}.issubset(df.columns):
        # Keep the abbreviation fallback separate so older generated datasets
        # can still match without changing their saved column schema.
        home_abbr_series = (
            df["home_abbr"]
            .astype(str)
            .str.upper()
            .map(lambda x: TEAM_ABBR_MAP.get(x, x))
        )
        away_abbr_series = (
            df["away_abbr"]
            .astype(str)
            .str.upper()
            .map(lambda x: TEAM_ABBR_MAP.get(x, x))
        )

        team_masks.append((home_abbr_series == home_norm) & (away_abbr_series == away_norm))

    # A/B) exact dataset match by season/week + team pair.
    if team_masks:
        # OR the available team-column contracts together while preserving the
        # season/week guard on each mask.
        exact_mask = base_mask & team_masks[0]
        for m in team_masks[1:]:
            exact_mask = exact_mask | (base_mask & m)
        exact_rows = df.loc[exact_mask]
        if not exact_rows.empty:
            if len(exact_rows) > 1:
                # Duplicate rows are recoverable because feature builders can
                # split one game across partially populated records. Merge only
                # missing values so the first row remains the source of truth.
                logging.warning(
                    "[Predict] Duplicate exact rows matched (%d). Merging rows. season=%s week=%s %s vs %s",
                    len(exact_rows),
                    season,
                    week,
                    home_norm,
                    away_norm,
                )
            return _merge_duplicate_rows(exact_rows), "dataset_exact"

    # C) fallback by canonical game_id.
    game_id_series = _normalized_game_id_series(df)
    if game_id_series is not None:
        # game_id is less explicit than home/away columns, but it protects
        # compatibility with older files that encoded matchup identity in one
        # field and sometimes swapped home/away ordering.
        candidates = _candidate_game_ids(
            season=int(season),
            week=int(week),
            home_team=home_norm,
            away_team=away_norm,
        )
        fuzzy_mask = base_mask & game_id_series.isin(candidates)
        fuzzy_rows = df.loc[fuzzy_mask]
        if not fuzzy_rows.empty:
            if len(fuzzy_rows) > 1:
                # Fuzzy duplicates are logged separately so diagnostics can
                # distinguish exact schema matches from game_id rescue matches.
                logging.warning(
                    "[Predict] Duplicate fuzzy rows matched (%d). Merging rows. season=%s week=%s %s vs %s",
                    len(fuzzy_rows),
                    season,
                    week,
                    home_norm,
                    away_norm,
                )
            return _merge_duplicate_rows(fuzzy_rows), "dataset_fuzzy"

    raise HTTPException(
        status_code=404,
        detail="Game data not found for given season/week/teams.",
    )


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

    # For prior/rolling features, avoid zero-default fallback. Use row-wise prior
    # context first, then a global numeric fallback if necessary.
    aligned = _fill_prior_feature_nans(aligned, numeric_medians=numeric_medians)

    # Last resort for non-prior features only.
    for col in aligned.columns:
        if _is_prior_like_feature(col):
            continue
        aligned[col] = aligned[col].fillna(0.0)
    return aligned


def _is_prior_like_feature(col: str) -> bool:
    c = str(col).lower()
    return ("prior_" in c) or ("rolling_" in c)


def _fill_prior_feature_nans(
    aligned: pd.DataFrame,
    *,
    numeric_medians: Optional[pd.Series],
) -> pd.DataFrame:
    if aligned is None or aligned.empty:
        return aligned
    row_idx = aligned.index[0]

    # Global fallback: prefer median of known prior columns, then overall median.
    global_prior_fallback: Optional[float] = None
    if numeric_medians is not None and not numeric_medians.empty:
        prior_med = numeric_medians[[c for c in numeric_medians.index if _is_prior_like_feature(c)]]
        if not prior_med.empty:
            try:
                global_prior_fallback = float(prior_med.median())
            except Exception:
                global_prior_fallback = None
        if global_prior_fallback is None:
            try:
                global_prior_fallback = float(numeric_medians.median())
            except Exception:
                global_prior_fallback = None
    if global_prior_fallback is None:
        global_prior_fallback = 0.5

    for col in aligned.columns:
        if not _is_prior_like_feature(col):
            continue
        if not pd.isna(aligned.at[row_idx, col]):
            continue

        side_prefix = None
        if str(col).startswith("home_"):
            side_prefix = "home_"
        elif str(col).startswith("away_"):
            side_prefix = "away_"

        side_vals: List[float] = []
        if side_prefix:
            for peer_col in aligned.columns:
                if not str(peer_col).startswith(side_prefix):
                    continue
                if not _is_prior_like_feature(peer_col):
                    continue
                val = aligned.at[row_idx, peer_col]
                try:
                    if not pd.isna(val):
                        side_vals.append(float(val))
                except Exception:
                    continue

        if side_vals:
            fill_val = float(np.mean(side_vals))
        elif numeric_medians is not None and col in numeric_medians.index:
            try:
                fill_val = float(numeric_medians[col])
            except Exception:
                fill_val = global_prior_fallback
        else:
            fill_val = global_prior_fallback

        # Ensure we never "default" priors to zero.
        if fill_val == 0.0:
            fill_val = global_prior_fallback if global_prior_fallback != 0.0 else 0.5
        aligned.at[row_idx, col] = fill_val

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
            pipeline_df = full_df.copy()
            expected_cols = (
                list(getattr(model, "feature_names_in_", []))
                if hasattr(model, "feature_names_in_")
                else []
            )

            if expected_cols:
                pipeline_df = pipeline_df.reindex(columns=expected_cols)

                # Use dataset medians for numeric expected columns when available.
                if numeric_medians is not None and not numeric_medians.empty:
                    for col in expected_cols:
                        if col in numeric_medians.index:
                            pipeline_df[col] = pd.to_numeric(
                                pipeline_df[col], errors="coerce"
                            ).fillna(float(numeric_medians[col]))

            return float(model.predict(pipeline_df)[0])

        # Preferred path for estimator-only artifacts: transform raw features
        # with the standalone preprocessor, then predict on the transformed array.
        if preprocessor is not None:
            pre_df = full_df.copy()
            expected_raw = (
                list(getattr(preprocessor, "feature_names_in_", []))
                if hasattr(preprocessor, "feature_names_in_")
                else []
            )
            if expected_raw:
                pre_df = pre_df.reindex(columns=expected_raw)
                if numeric_medians is not None and not numeric_medians.empty:
                    for col in expected_raw:
                        if col in numeric_medians.index:
                            pre_df[col] = pd.to_numeric(
                                pre_df[col], errors="coerce"
                            ).fillna(float(numeric_medians[col]))

            X_proc = preprocessor.transform(pre_df)
            return float(model.predict(X_proc)[0])

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


def _team_history_rows(
    df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    Return all prior rows for a team before (season, week), sorted in time order.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if "season" not in df.columns or "week" not in df.columns:
        return pd.DataFrame()

    t = _normalize_team_code(team)
    prior_mask = (df["season"] < int(season)) | ((df["season"] == int(season)) & (df["week"] < int(week)))

    masks: List[pd.Series] = []
    if {"home_team", "away_team"}.issubset(df.columns):
        home_series = (
            df["home_team"]
            .astype(str)
            .str.upper()
            .map(lambda x: TEAM_ABBR_MAP.get(x, x))
        )
        away_series = (
            df["away_team"]
            .astype(str)
            .str.upper()
            .map(lambda x: TEAM_ABBR_MAP.get(x, x))
        )
        masks.append((home_series == t) | (away_series == t))
    if {"home_abbr", "away_abbr"}.issubset(df.columns):
        home_series = (
            df["home_abbr"]
            .astype(str)
            .str.upper()
            .map(lambda x: TEAM_ABBR_MAP.get(x, x))
        )
        away_series = (
            df["away_abbr"]
            .astype(str)
            .str.upper()
            .map(lambda x: TEAM_ABBR_MAP.get(x, x))
        )
        masks.append((home_series == t) | (away_series == t))

    if not masks:
        return pd.DataFrame()

    team_mask = masks[0]
    for m in masks[1:]:
        team_mask = team_mask | m

    out = df.loc[prior_mask & team_mask].sort_values(["season", "week"])
    return out


def _extract_side_for_team(row: pd.Series, team: str) -> Optional[str]:
    t = _normalize_team_code(team)
    for home_col in ("home_team", "home_abbr"):
        if home_col in row.index and _normalize_team_code(str(row.get(home_col, ""))) == t:
            return "home"
    for away_col in ("away_team", "away_abbr"):
        if away_col in row.index and _normalize_team_code(str(row.get(away_col, ""))) == t:
            return "away"
    return None


def _historical_prior_value(
    df: pd.DataFrame,
    *,
    team: str,
    season: int,
    week: int,
    target_col: str,
) -> Optional[float]:
    """
    Leak-safe prior fill:
      - use only history rows before the target week
      - read same prior metric from team's side in each prior row
      - prefer latest value, fallback to rolling mean over recent history
    """
    history = _team_history_rows(df, team=team, season=season, week=week)
    if history.empty:
        return None

    if not (target_col.startswith("home_") or target_col.startswith("away_")):
        return None

    base = target_col.split("_", 1)[1]
    vals: List[float] = []
    for _, hist_row in history.iterrows():
        side = _extract_side_for_team(hist_row, team=team)
        if side is None:
            continue
        src_col = f"{side}_{base}"
        if src_col not in hist_row.index:
            continue
        raw = hist_row.get(src_col)
        try:
            if not _is_missing_value(raw):
                vals.append(float(raw))
        except Exception:
            continue

    if not vals:
        return None

    last_val = vals[-1]
    if not np.isnan(last_val):
        return float(last_val)

    window = vals[-5:]
    return float(np.mean(window)) if window else None


def _fill_missing_team_priors(
    df: pd.DataFrame,
    row_df: pd.DataFrame,
    *,
    home_team: str,
    away_team: str,
    season: int,
    week: int,
) -> int:
    if row_df is None or row_df.empty:
        return 0
    idx = row_df.index[0]
    filled = 0

    for col in row_df.columns:
        if not (str(col).startswith("home_prior_") or str(col).startswith("away_prior_")):
            continue
        if not _is_missing_value(row_df.at[idx, col]):
            continue

        team = home_team if str(col).startswith("home_") else away_team
        val = _historical_prior_value(
            df,
            team=team,
            season=int(season),
            week=int(week),
            target_col=str(col),
        )
        if val is None:
            continue
        row_df.at[idx, col] = val
        filled += 1

    return filled


def _derive_home_away_diff_features(row_df: pd.DataFrame) -> int:
    """
    Build missing home_minus_away_* features from available home_*/away_* values.
    """
    if row_df is None or row_df.empty:
        return 0
    idx = row_df.index[0]
    filled = 0
    for col in row_df.columns:
        c = str(col)
        if not c.startswith("home_minus_away_"):
            continue
        if not _is_missing_value(row_df.at[idx, c]):
            continue
        suffix = c[len("home_minus_away_"):]
        home_col = f"home_{suffix}"
        away_col = f"away_{suffix}"
        if home_col not in row_df.columns or away_col not in row_df.columns:
            continue
        hv = row_df.at[idx, home_col]
        av = row_df.at[idx, away_col]
        if _is_missing_value(hv) or _is_missing_value(av):
            continue
        try:
            row_df.at[idx, c] = float(hv) - float(av)
            filled += 1
        except Exception:
            continue
    return filled


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

    prior_filled = _fill_missing_team_priors(
        df=df,
        row_df=row_df,
        home_team=home_team,
        away_team=away_team,
        season=season,
        week=week,
    )
    diff_filled = _derive_home_away_diff_features(row_df)

    if filled:
        logging.info(
            "[Predict] Rolled forward %d player-stat features for %s vs %s (season=%s week=%s)",
            filled, home_team.upper(), away_team.upper(), season, week
        )
    if prior_filled:
        logging.info(
            "[Predict] Filled %d prior features from leak-safe history for %s vs %s (season=%s week=%s)",
            prior_filled,
            home_team.upper(),
            away_team.upper(),
            season,
            week,
        )
    if diff_filled:
        logging.info(
            "[Predict] Derived %d home_minus_away features for %s vs %s (season=%s week=%s)",
            diff_filled,
            home_team.upper(),
            away_team.upper(),
            season,
            week,
        )

    return row_df


