# ==========================================
# File: backend/services/inference_row.py
# Role: Inference row builder for live predictions.
# Input Data: Game context and dataset history.
# Output Data: Filled feature rows.
# Dependencies: __future__, typing, numpy, pandas
# Notes: Rolls forward team stats before median imputation with batched updates to avoid fragmentation.
# ==========================================

from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd


ABBR_FIX: Dict[str, str] = {        # Match your dataset builder's team normalization
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
    "WSH": "WAS",
}


DROP_COLS = {           # Targets/leaky columns you never want to feed into preprocess/estimator
    "home_points_for",
    "away_points_for",
    "point_diff",
    "winner",
    "home_win",
    "actual_winner",
}

def _normalize_team(team: Any) -> str:
    s = "" if team is None else str(team).strip().upper()
    return ABBR_FIX.get(s, s)

def _moneyline_to_prob(ml: Any) -> float:
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    if ml > 0:
        return 100.0 / (ml + 100.0)
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    return np.nan

def _infer_expected_columns(preprocessor: Any, raw_feature_columns: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Prefer preprocessor.feature_names_in_ to guarantee the raw-schema matches training.
    Fall back to bundle-provided raw_feature_columns if available.
    """
    if preprocessor is not None:
        cols = getattr(preprocessor, "feature_names_in_", None)
        if cols is not None:
            return list(cols)

        named_steps = getattr(preprocessor, "named_steps", None)
        if isinstance(named_steps, dict):
            for step in named_steps.values():
                cols = getattr(step, "feature_names_in_", None)
                if cols is not None:
                    return list(cols)
    if raw_feature_columns:
        return list(raw_feature_columns)
    return None

def _find_dataset_row(
    dataset_df: pd.DataFrame, *, season: int, week: int, home_team: str, away_team: str) -> Optional[pd.DataFrame]:
    if not {"season", "week", "home_team", "away_team"}.issubset(dataset_df.columns):
        return None
    mask = (
        (pd.to_numeric(dataset_df["season"], errors="coerce") == int(season))
        & (pd.to_numeric(dataset_df["week"], errors="coerce") == int(week))
        & (dataset_df["home_team"].astype(str).str.upper().map(_normalize_team) == home_team)
        & (dataset_df["away_team"].astype(str).str.upper().map(_normalize_team) == away_team)
    )
    if mask.any():
        return dataset_df.loc[mask].tail(1).copy()
    return None

def _apply_onehots(
    row_df: pd.DataFrame,
    *,
    home: str,
    away: str,
    dataset_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Your dataset includes home_team_* and away_team_* columns (32 each).
    Setting these makes inference compatible with numeric-only models.
    """
    # Only set if your dataset schema includes these
    home_oh = [c for c in dataset_df.columns if c.startswith("home_team_")]
    away_oh = [c for c in dataset_df.columns if c.startswith("away_team_")]

    updates: Dict[str, Any] = {}
    for c in home_oh:
        updates[c] = 1.0 if c == f"home_team_{home}" else 0.0
    for c in away_oh:
        updates[c] = 1.0 if c == f"away_team_{away}" else 0.0

    if updates:
        row_df = row_df.assign(**updates)

    return row_df

def _pick_schedule_row(
    schedule_df: pd.DataFrame, season: int, week: int, home: str, away: str
) -> Optional[dict]:
    if schedule_df is None or schedule_df.empty:
        return None

    def _colpick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in schedule_df.columns:
                return c
        return None

    c_home = _colpick(["home_team", "home", "home_abbr", "home_team_abbr"])
    c_away = _colpick(["away_team", "away", "away_abbr", "away_team_abbr"])
    c_week = _colpick(["week", "week_num", "week_number"])
    c_season = _colpick(["season", "year"])

    m = pd.Series(True, index=schedule_df.index)
    if c_season:
        m &= (pd.to_numeric(schedule_df[c_season], errors="coerce") == int(season))
    if c_week:
        m &= (pd.to_numeric(schedule_df[c_week], errors="coerce") == int(week))
    if c_home:
        m &= (schedule_df[c_home].astype(str).str.upper().map(_normalize_team) == home)
    if c_away:
        m &= (schedule_df[c_away].astype(str).str.upper().map(_normalize_team) == away)

    if m.any():
        return schedule_df.loc[m].iloc[-1].to_dict()
    return None

def _latest_team_prior_row(
    dataset_df: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    team_history_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[pd.Series]:
    """
    Grab the team's most recent row BEFORE (season, week), regardless of role.
    This is a practical approximation when you don't rebuild the whole feature pipeline live.
    """
    team = _normalize_team(team)
    if team_history_cache and team in team_history_cache:
        s = team_history_cache[team]
        if not {"season_num", "week_num"}.issubset(s.columns):
            return None
        mask_past = (s["season_num"] == int(season)) & (s["week_num"] < int(week))
        if mask_past.any():
            return s.loc[mask_past].iloc[-1]
        return None

    if not {"season", "week", "home_team", "away_team"}.issubset(dataset_df.columns):
        return None

    s = dataset_df.copy()
    s["season_num"] = pd.to_numeric(s["season"], errors="coerce")
    s["week_num"] = pd.to_numeric(s["week"], errors="coerce")

    mask_past = (s["season_num"] == int(season)) & (s["week_num"] < int(week))
    mask_team = (
        s["home_team"].astype(str).str.upper().map(_normalize_team).eq(team)
        | s["away_team"].astype(str).str.upper().map(_normalize_team).eq(team)
    )
    past = s.loc[mask_past & mask_team].sort_values(["week_num"])
    if past.empty:
        return None
    return past.iloc[-1]

def build_team_history_cache(dataset_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Precompute team-level history frames so we can reuse the latest prior game rows quickly.
    The cache is keyed by normalized team code.
    """
    if dataset_df is None or dataset_df.empty:
        return {}
    if not {"season", "week", "home_team", "away_team"}.issubset(dataset_df.columns):
        return {}

    s = dataset_df.copy()
    s["season_num"] = pd.to_numeric(s["season"], errors="coerce")
    s["week_num"] = pd.to_numeric(s["week"], errors="coerce")

    cache: Dict[str, List[pd.Series]] = {}
    for _, row in s.iterrows():
        home = _normalize_team(row.get("home_team"))
        away = _normalize_team(row.get("away_team"))
        if home:
            cache.setdefault(home, []).append(row)
        if away:
            cache.setdefault(away, []).append(row)

    out: Dict[str, pd.DataFrame] = {}
    for team, rows in cache.items():
        df = pd.DataFrame(rows)
        df = df.sort_values(["season_num", "week_num"])
        out[team] = df

    return out

def _fill_team_priors(
    row_df: pd.DataFrame,
    *,
    dataset_df: pd.DataFrame,
    season: int,
    week: int,
    home: str,
    away: str,
    team_history_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Fill home_prior_* and away_prior_* from the latest available pre-game priors for each team.
    Then compute home_minus_away_* columns if present in schema.
    """
    # Identify suffixes from schema
    home_prior_cols = [c for c in dataset_df.columns if c.startswith("home_prior_")]
    away_prior_cols = [c for c in dataset_df.columns if c.startswith("away_prior_")]

    # Build suffix list once
    suffixes = [c.replace("home_prior_", "", 1) for c in home_prior_cols]

    updates: Dict[str, Any] = {}

    def _maybe_set(dst: str, val: Any) -> None:
        if dst in updates and pd.notna(updates[dst]):
            return
        if dst in row_df.columns and pd.notna(row_df.at[0, dst]):
            return
        if pd.notna(val):
            updates[dst] = val

    # Home priors from latest row
    h_last = _latest_team_prior_row(
        dataset_df,
        season=season,
        week=week,
        team=home,
        team_history_cache=team_history_cache,
    )
    if h_last is not None:
        # Determine whether the team was home or away in that last row
        was_home = _normalize_team(h_last.get("home_team")) == home
        src_prefix = "home_prior_" if was_home else "away_prior_"
        for suf in suffixes:
            src_col = f"{src_prefix}{suf}"
            dst_col = f"home_prior_{suf}"
            if dst_col in dataset_df.columns and src_col in dataset_df.columns:
                _maybe_set(dst_col, h_last.get(src_col))

    # Away priors from latest row
    a_last = _latest_team_prior_row(
        dataset_df,
        season=season,
        week=week,
        team=away,
        team_history_cache=team_history_cache,
    )
    if a_last is not None:
        was_home = _normalize_team(a_last.get("home_team")) == away
        src_prefix = "home_prior_" if was_home else "away_prior_"
        for suf in suffixes:
            src_col = f"{src_prefix}{suf}"
            dst_col = f"away_prior_{suf}"
            if dst_col in dataset_df.columns and src_col in dataset_df.columns:
                _maybe_set(dst_col, a_last.get(src_col))

    # Derived diffs
    hma_cols = [c for c in dataset_df.columns if c.startswith("home_minus_away_")]
    for c in hma_cols:
        suf = c.replace("home_minus_away_", "", 1)
        hp = f"home_prior_{suf}"
        ap = f"away_prior_{suf}"
        hv = updates.get(hp)
        if hv is None or pd.isna(hv):
            hv = row_df.at[0, hp] if hp in row_df.columns else np.nan
        av = updates.get(ap)
        if av is None or pd.isna(av):
            av = row_df.at[0, ap] if ap in row_df.columns else np.nan
        if pd.notna(hv) and pd.notna(av):
            updates[c] = float(hv) - float(av)

    if updates:
        row_df = row_df.assign(**updates)

    return row_df

def _fill_team_rollups(
    row_df: pd.DataFrame,
    *,
    dataset_df: pd.DataFrame,
    season: int,
    week: int,
    home: str,
    away: str,
    team_history_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Roll forward team-level rolling/player/elo stats from each team's latest prior game.
    This provides stronger inference inputs than pure median imputation.
    """
    updates: Dict[str, Any] = {}

    def _maybe_set(dst: str, val: Any) -> None:
        if dst in row_df.columns:
            if pd.notna(row_df.at[0, dst]):
                return
        if pd.notna(val):
            updates[dst] = val

    def _apply_rollup(team: str, dst_prefix: str, alt_prefix: str) -> None:
        last = _latest_team_prior_row(
            dataset_df,
            season=season,
            week=week,
            team=team,
            team_history_cache=team_history_cache,
        )
        if last is None:
            return
        was_home = _normalize_team(last.get("home_team")) == team
        src_prefix = dst_prefix if was_home else alt_prefix
        dst_cols = [c for c in dataset_df.columns if c.startswith(dst_prefix)]
        for dst in dst_cols:
            suffix = dst.replace(dst_prefix, "", 1)
            src = f"{src_prefix}{suffix}"
            if src in dataset_df.columns:
                _maybe_set(dst, last.get(src))

    # Home team roll-forward
    _apply_rollup(home, "home_rolling_", "away_rolling_")
    _apply_rollup(home, "home_player_team_", "away_player_team_")
    _apply_rollup(home, "home_qb_", "away_qb_")
    _apply_rollup(home, "home_elo_", "away_elo_")

    # Away team roll-forward
    _apply_rollup(away, "away_rolling_", "home_rolling_")
    _apply_rollup(away, "away_player_team_", "home_player_team_")
    _apply_rollup(away, "away_qb_", "home_qb_")
    _apply_rollup(away, "away_elo_", "home_elo_")

    if updates:
        row_df = row_df.assign(**updates)

    return row_df

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
    Returns: (row_df_aligned_to_expected_cols, prediction_source)
    Optional: debug=True returns an extra debug dict with fill stats.

    Behavior:
      1) If dataset already has exact engineered row, reuse it (best).
      2) Else: build minimal row, optionally enrich from schedule,
         roll forward team stats/priors from latest available game rows,
         compute derived diffs, then impute remaining numeric columns from dataset medians.
    """
    home = _normalize_team(home_team)
    away = _normalize_team(away_team)

    ds_row = _find_dataset_row(dataset_df, season=season, week=week, home_team=home, away_team=away)
    if ds_row is not None and not ds_row.empty:
        row_df = ds_row.copy()
        source = "dataset_exact"
    else:
        row_df = pd.DataFrame([{
            "season": int(season),
            "week": int(week),
            "home_team": home,
            "away_team": away,
            "api_game_id": f"{int(season)}-{int(week)}-{home}-{away}",
        }])
        source = "synthetic_priors"

    # Optional schedule overwrite (safe)
    srow = _pick_schedule_row(schedule_df, season=season, week=week, home=home, away=away) if isinstance(schedule_df, pd.DataFrame) else None
    if isinstance(srow, dict):
        for k in ["spread_line", "total_line", "home_rest", "away_rest", "home_moneyline", "away_moneyline"]:
            if k in dataset_df.columns and k in srow and pd.notna(srow.get(k)):
                row_df.at[0, k] = srow.get(k)

        if "home_moneyline_prob" in dataset_df.columns and "home_moneyline" in row_df.columns:
            row_df.at[0, "home_moneyline_prob"] = _moneyline_to_prob(row_df.at[0, "home_moneyline"])
        if "away_moneyline_prob" in dataset_df.columns and "away_moneyline" in row_df.columns:
            row_df.at[0, "away_moneyline_prob"] = _moneyline_to_prob(row_df.at[0, "away_moneyline"])
        if "moneyline_prob_diff" in dataset_df.columns and {"home_moneyline_prob", "away_moneyline_prob"}.issubset(row_df.columns):
            hp = row_df.at[0, "home_moneyline_prob"]
            ap = row_df.at[0, "away_moneyline_prob"]
            if pd.notna(hp) and pd.notna(ap):
                row_df.at[0, "moneyline_prob_diff"] = float(hp) - float(ap)
        if "rest_diff" in dataset_df.columns and {"home_rest", "away_rest"}.issubset(row_df.columns):
            hr = row_df.at[0, "home_rest"]
            ar = row_df.at[0, "away_rest"]
            if pd.notna(hr) and pd.notna(ar):
                row_df.at[0, "rest_diff"] = float(hr) - float(ar)

    # Normalize team fields + onehots
    row_df["home_team"] = home
    row_df["away_team"] = away
    row_df = _apply_onehots(row_df, home=home, away=away, dataset_df=dataset_df)

    # Fill team priors + derived diffs if synthetic
    if source != "dataset_exact":
        row_df = _fill_team_priors(
            row_df,
            dataset_df=dataset_df,
            season=season,
            week=week,
            home=home,
            away=away,
            team_history_cache=team_history_cache,
        )
        row_df = _fill_team_rollups(
            row_df,
            dataset_df=dataset_df,
            season=season,
            week=week,
            home=home,
            away=away,
            team_history_cache=team_history_cache,
        )

    # Drop targets/leaky columns
    row_df = row_df.drop(columns=[c for c in DROP_COLS if c in row_df.columns], errors="ignore")

    # Align to expected raw columns
    expected = _infer_expected_columns(preprocessor, raw_feature_columns=raw_feature_columns)

    debug_info = None

    if expected:
        # Reindex once to avoid repeated column inserts (prevents DataFrame fragmentation).
        # Note: this drops any extra columns not in the model's raw feature schema.
        row_df = row_df.reindex(columns=expected)

        # Impute numeric columns from dataset medians when still missing
        # (Only for numeric schema columns; leave categorical for preprocessor imputers)
        med = dataset_df.drop(columns=[c for c in DROP_COLS if c in dataset_df.columns], errors="ignore").median(numeric_only=True)
        med_aligned = med.reindex(row_df.columns)

        if debug:
            missing_before = row_df.isna().iloc[0]
            fillable = missing_before & med_aligned.notna()
            filled_by_median_cols = list(row_df.columns[fillable])

        row_df = row_df.fillna(med_aligned)

        if debug:
            missing_after = row_df.isna().iloc[0]
            missing_after_cols = list(row_df.columns[missing_after])
            home_prior_missing = [c for c in row_df.columns if c.startswith("home_prior_") and pd.isna(row_df.at[0, c])]
            away_prior_missing = [c for c in row_df.columns if c.startswith("away_prior_") and pd.isna(row_df.at[0, c])]
            hma_missing = [c for c in row_df.columns if c.startswith("home_minus_away_") and pd.isna(row_df.at[0, c])]

            def _limit(items: List[str], limit: int = 50) -> List[str]:
                return items[:limit]

            debug_info = {
                "expected_columns": len(expected),
                "missing_before_impute": int(missing_before.sum()),
                "filled_by_median_count": len(filled_by_median_cols),
                "filled_by_median_sample": _limit(filled_by_median_cols),
                "missing_after_impute": int(missing_after.sum()),
                "missing_after_sample": _limit(missing_after_cols),
                "missing_home_prior_count": len(home_prior_missing),
                "missing_home_prior_sample": _limit(home_prior_missing),
                "missing_away_prior_count": len(away_prior_missing),
                "missing_away_prior_sample": _limit(away_prior_missing),
                "missing_home_minus_away_count": len(hma_missing),
                "missing_home_minus_away_sample": _limit(hma_missing),
            }

    if debug:
        return row_df, source, (debug_info or {})

    return row_df, source
