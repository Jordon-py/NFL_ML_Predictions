"""Compatibility imports for score-sync helpers.

The implementation lives in `backend.scripts.score_sync` after the backend
script reorganization.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from backend.utils.team_codes import normalize_team_code


def build_score_game_id(season: Any, week: Any, home_team: Any, away_team: Any) -> str:
    try:
        season_int = int(season)
        week_int = int(week)
    except (TypeError, ValueError):
        return ""

    home_code = normalize_team_code(home_team)
    away_code = normalize_team_code(away_team)
    if not home_code or not away_code:
        return ""
    return f"{season_int}_{week_int}_{home_code}_{away_code}"


def _pick_column(columns: Iterable[str], *candidates: str) -> Optional[str]:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_iso_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if parsed is None or pd.isna(parsed):
            return default
        return parsed.to_pydatetime().isoformat()
    except Exception:
        return default


def extract_score_entries_from_dataframe(
    df: Optional[pd.DataFrame],
    *,
    updated_at: Optional[str] = None,
) -> list[Dict[str, object]]:
    """Return canonical completed-game score entries from a dataset-like frame."""

    if df is None or df.empty:
        return []

    season_col = _pick_column(df.columns, "season", "season_num")
    week_col = _pick_column(df.columns, "week", "week_num")
    home_team_col = _pick_column(df.columns, "home_team", "home_abbr")
    away_team_col = _pick_column(df.columns, "away_team", "away_abbr")
    home_score_col = _pick_column(df.columns, "final_home_score", "home_points_for", "home_score")
    away_score_col = _pick_column(df.columns, "final_away_score", "away_points_for", "away_score")
    status_col = _pick_column(df.columns, "status", "game_status")
    updated_at_col = _pick_column(df.columns, "score_updated_at", "updated_at", "generated_at")

    if not all([season_col, week_col, home_team_col, away_team_col, home_score_col, away_score_col]):
        return []

    default_updated_at = _to_iso_or_default(updated_at, datetime.now(timezone.utc).isoformat())
    entries_by_game: dict[str, Dict[str, object]] = {}

    for _, row in df.iterrows():
        season = _to_int_or_none(row.get(season_col))
        week = _to_int_or_none(row.get(week_col))
        home_team = normalize_team_code(row.get(home_team_col))
        away_team = normalize_team_code(row.get(away_team_col))
        home_score = _to_int_or_none(row.get(home_score_col))
        away_score = _to_int_or_none(row.get(away_score_col))

        if None in {season, week, home_score, away_score} or not home_team or not away_team:
            continue

        game_id = build_score_game_id(season, week, home_team, away_team)
        if not game_id:
            continue

        candidate = {
            "game_id": game_id,
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "status": str(row.get(status_col) or "final").strip() or "final",
            "updated_at": _to_iso_or_default(row.get(updated_at_col), default_updated_at),
        }

        existing = entries_by_game.get(game_id)
        if existing is None or str(candidate["updated_at"]) >= str(existing["updated_at"]):
            entries_by_game[game_id] = candidate

    return sorted(
        entries_by_game.values(),
        key=lambda item: (
            int(item.get("season") or 0),
            int(item.get("week") or 0),
            str(item.get("home_team") or ""),
            str(item.get("away_team") or ""),
        ),
    )


def write_score_snapshot(path: Path, entries: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = list(entries)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
