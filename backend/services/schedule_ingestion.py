# ==========================================
# File: backend/services/schedule_ingestion.py
# Role: ESPN -> clean NFL schedule ingestion.
# Input Data: ESPN Scoreboard API responses.
# Output Data: Normalized schedule CSVs and Parquet files.
# Dependencies: requests, pandas, pydantic, argparse
# Notes: Maps postseason weeks to 19-22; ensures future rows are leak-safe.
# ==========================================

"""
backend/services/schedule_ingestion.py
--------------------------------------

ESPN -> clean NFL schedule ingestion.

Purpose:
    Fetch ESPN scoreboard events for regular season + postseason, normalize
    them into one-row-per-game schedule files, and keep future rows leak-safe.

Why this exists:
    The ESPN Core "season/types/weeks" endpoint is metadata. It helps you
    discover weeks, but it does not directly return the `events` array your
    parser expects. For actual game rows, this module uses ESPN's site
    scoreboard endpoint.

Default output is compatible with:
    - backend/build_csv_datasets_v3.py schedule expectations
    - backend/main.py schedule CSV discovery
    - backend/services/inference_row.py schedule enrichment

Example:
    python -m backend.services.schedule_ingestion --season 2025 \
      --out-csv backend/data/Nfl_schedule_2025.csv \
      --out-parquet backend/data/schedules/nfl_schedule_2025.parquet \
      --raw-dir backend/data/raw/espn/scoreboards

Notes:
    - This module intentionally keeps final scores only for completed games.
    - For scheduled/future games, home_score and away_score are None.
    - Postseason ESPN weeks are mapped to model/nflverse style weeks:
        WC=19, DIV=20, CON=21, SB=22
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import pydantic
import pandas as pd
import requests

from backend.utils.team_codes import normalize_team_code as _normalize_team_code

log = logging.getLogger(__name__)

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

REGULAR_SEASON = 2
POSTSEASON = 3

# ESPN site API postseason week numbers can differ from nflverse/model week IDs.
# Your dataset builder already uses 19-22 for playoff rounds, so normalize here.
POSTSEASON_WEEK_MAP: dict[int, tuple[int, str]] = {
    1: (19, "WC"),
    2: (20, "DIV"),
    3: (21, "CON"),
    4: (22, "SB"),
    5: (22, "SB"),
}

SCHEDULE_COLUMNS = [
    "season",
    "season_type",
    "game_type",
    "week",
    "espn_week",
    "game_id",
    "espn_game_id",
    "gameday",
    "game_date",
    "gametime",
    "kickoff",
    "kickoff_utc",
    "home_team",
    "away_team",
    "home_team_name",
    "away_team_name",
    "home_score",
    "away_score",
    "completed",
    "status",
    "status_detail",
    "neutral_site",
    "venue",
    "roof",
    "surface",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "source",
    "ingested_at",
]


@dataclass(frozen=True)
class ScheduleRow:
    """Normalized one-row-per-game schedule record.

    Data shape:
        Canonical season/week/game ids, kickoff timestamps, home/away teams,
        scores only for completed games, venue, odds, source, and ingestion
        timestamp.
    Methods:
        Dataclass container only; serialization is handled with ``asdict``.
    """

    season: int
    season_type: int
    game_type: str
    week: int
    espn_week: int
    game_id: str
    espn_game_id: str
    gameday: Optional[str]
    game_date: Optional[str]
    gametime: Optional[str]
    kickoff: Optional[str]
    kickoff_utc: Optional[str]
    home_team: str
    away_team: str
    home_team_name: Optional[str]
    away_team_name: Optional[str]
    home_score: Optional[int]
    away_score: Optional[int]
    completed: bool
    status: Optional[str]
    status_detail: Optional[str]
    neutral_site: Optional[bool]
    venue: Optional[str]
    roof: Optional[str] = None
    surface: Optional[str] = None
    spread_line: Optional[float] = None
    total_line: Optional[float] = None
    home_moneyline: Optional[int] = None
    away_moneyline: Optional[int] = None
    source: str = "espn_site_scoreboard"
    ingested_at: str = ""


def normalize_team_code(value: Any) -> str:
    """Normalize ESPN/team-abbreviation variants to your backend's canonical form."""
    if value is None:
        return ""
    code = str(value).strip().upper()
    if code in {"", "TBD", "TBA", "NA", "N/A"}:
        return ""
    return _normalize_team_code(code)


def parse_espn_datetime(value: Any) -> Optional[datetime]:
    """Parse ESPN ISO datetime into UTC datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_score(value: Any, *, completed: bool) -> Optional[int]:
    """Only preserve scores for completed games to keep future rows leak-safe."""
    if not completed:
        return None
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def canonical_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    """Match nflverse/model-style game_id: YYYY_WW_AWAY_HOME."""
    return f"{int(season)}_{int(week):02d}_{away_team}_{home_team}"


def normalize_week_and_game_type(season_type: int, espn_week: int) -> tuple[int, str]:
    """Return (model_week, game_type)."""
    if int(season_type) == REGULAR_SEASON:
        return int(espn_week), "REG"

    if int(season_type) == POSTSEASON:
        return POSTSEASON_WEEK_MAP.get(int(espn_week), (18 + int(espn_week), "POST"))

    return int(espn_week), f"TYPE_{season_type}"


def fetch_scoreboard(
    *,
    season: int,
    season_type: int,
    week: int,
    timeout: int = 20,
    limit: int = 1000,
) -> dict[str, Any]:
    """Fetch one ESPN scoreboard page for a season type + week."""
    params = {
        "dates": int(season),
        "seasontype": int(season_type),
        "week": int(week),
        "limit": int(limit),
    }
    response = requests.get(
        ESPN_SCOREBOARD_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "json" not in content_type:
        raise RuntimeError(
            f"Expected JSON, got {content_type}. Response preview: {response.text[:300]}"
        )

    return response.json()


def iter_default_weeks(season_type: int) -> range:
    """Reasonable ESPN week ranges."""
    if int(season_type) == REGULAR_SEASON:
        return range(1, 19)
    if int(season_type) == POSTSEASON:
        return range(1, 6)
    return range(1, 23)


def _competitor_by_side(competitors: Sequence[dict[str, Any]], side: str) -> Optional[dict[str, Any]]:
    for competitor in competitors:
        if str(competitor.get("homeAway", "")).lower() == side:
            return competitor
    return None


def _extract_odds(competition: dict[str, Any]) -> dict[str, Any]:
    """Best-effort odds extraction. Missing odds stay None."""
    odds = competition.get("odds") or []
    if not odds:
        return {
            "spread_line": None,
            "total_line": None,
            "home_moneyline": None,
            "away_moneyline": None,
        }

    first = odds[0] if isinstance(odds, list) and odds else {}
    return {
        "spread_line": first.get("spread"),
        "total_line": first.get("overUnder"),
        "home_moneyline": first.get("homeTeamOdds", {}).get("moneyLine")
            if isinstance(first.get("homeTeamOdds"), dict)
            else None,
        "away_moneyline": first.get("awayTeamOdds", {}).get("moneyLine")
            if isinstance(first.get("awayTeamOdds"), dict)
            else None,
    }


def normalize_event(event: dict[str, Any], *, season_type_hint: int) -> Optional[ScheduleRow]:
    """Normalize one ESPN event object into a ScheduleRow."""
    competitions = event.get("competitions") or []
    if not competitions:
        return None

    competition = competitions[0]
    competitors = competition.get("competitors") or []

    home = _competitor_by_side(competitors, "home")
    away = _competitor_by_side(competitors, "away")
    if not home or not away:
        return None

    season = int(event.get("season", {}).get("year") or 0)
    if not season:
        return None

    season_type = int(event.get("season", {}).get("type") or season_type_hint)
    espn_week = int(event.get("week", {}).get("number") or 0)
    if not espn_week:
        return None

    week, game_type = normalize_week_and_game_type(season_type, espn_week)

    status_type = (
        competition.get("status", {}).get("type")
        or event.get("status", {}).get("type")
        or {}
    )
    completed = bool(status_type.get("completed", False))

    event_date_raw = event.get("date") or competition.get("date")
    kickoff_dt = parse_espn_datetime(event_date_raw)
    kickoff_utc = kickoff_dt.isoformat().replace("+00:00", "Z") if kickoff_dt else None
    gameday = kickoff_dt.date().isoformat() if kickoff_dt else None
    gametime = kickoff_dt.strftime("%H:%MZ") if kickoff_dt else None

    home_team = home.get("team") or {}
    away_team = away.get("team") or {}
    home_abbr = normalize_team_code(home_team.get("abbreviation"))
    away_abbr = normalize_team_code(away_team.get("abbreviation"))
    if not home_abbr or not away_abbr:
        return None

    venue = competition.get("venue") or {}
    odds = _extract_odds(competition)

    return ScheduleRow(
        season=season,
        season_type=season_type,
        game_type=game_type,
        week=week,
        espn_week=espn_week,
        game_id=canonical_game_id(season, week, away_abbr, home_abbr),
        espn_game_id=str(event.get("id") or ""),
        gameday=gameday,
        game_date=gameday,
        gametime=gametime,
        kickoff=kickoff_utc,
        kickoff_utc=kickoff_utc,
        home_team=home_abbr,
        away_team=away_abbr,
        home_team_name=home_team.get("displayName"),
        away_team_name=away_team.get("displayName"),
        home_score=parse_score(home.get("score"), completed=completed),
        away_score=parse_score(away.get("score"), completed=completed),
        completed=completed,
        status=status_type.get("name") or status_type.get("description"),
        status_detail=status_type.get("detail") or status_type.get("description"),
        neutral_site=competition.get("neutralSite"),
        venue=venue.get("fullName") or venue.get("name"),
        spread_line=odds["spread_line"],
        total_line=odds["total_line"],
        home_moneyline=odds["home_moneyline"],
        away_moneyline=odds["away_moneyline"],
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )


def normalize_scoreboard(payload: dict[str, Any], *, season_type_hint: int) -> list[ScheduleRow]:
    rows: list[ScheduleRow] = []
    for event in payload.get("events", []) or []:
        row = normalize_event(event, season_type_hint=season_type_hint)
        if row is not None:
            rows.append(row)
    return rows


def ingest_schedule(
    *,
    season: int,
    season_types: Iterable[int] = (REGULAR_SEASON, POSTSEASON),
    weeks: Optional[Iterable[int]] = None,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch and normalize schedule rows for a season."""
    all_rows: list[ScheduleRow] = []

    for season_type in season_types:
        week_iterable = list(weeks) if weeks is not None else list(iter_default_weeks(season_type))

        for espn_week in week_iterable:
            try:
                payload = fetch_scoreboard(
                    season=season,
                    season_type=int(season_type),
                    week=int(espn_week),
                )
            except requests.HTTPError as exc:
                log.warning(
                    "ESPN fetch failed | season=%s season_type=%s week=%s error=%s",
                    season,
                    season_type,
                    espn_week,
                    exc,
                )
                continue

            if raw_dir is not None:
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f"season={season}_type={season_type}_week={espn_week}.json"
                raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            rows = normalize_scoreboard(payload, season_type_hint=int(season_type))
            all_rows.extend(rows)
            log.info(
                "Fetched ESPN schedule | season=%s type=%s espn_week=%s events=%s parsed=%s",
                season,
                season_type,
                espn_week,
                len(payload.get("events", []) or []),
                len(rows),
            )

    records = [asdict(row) for row in all_rows]
    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    df = df.reindex(columns=SCHEDULE_COLUMNS)
    df = clean_schedule_frame(df)
    return df


def clean_schedule_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical cleanup, typing, de-dupe, and leak guard."""
    out = df.copy()
    out = out.reindex(columns=SCHEDULE_COLUMNS)

    for col in ("season", "season_type", "week", "espn_week"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    for col in ("home_score", "away_score", "home_moneyline", "away_moneyline"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    for col in ("spread_line", "total_line"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["home_team"] = out["home_team"].map(normalize_team_code)
    out["away_team"] = out["away_team"].map(normalize_team_code)
    out["completed"] = out["completed"].fillna(False).astype(bool)

    # Future-row leak guard: no scores unless completed.
    future_mask = ~out["completed"]
    out.loc[future_mask, ["home_score", "away_score"]] = pd.NA

    out = out.dropna(subset=["season", "season_type", "week", "game_id", "home_team", "away_team"])
    out = out[out["home_team"].ne("") & out["away_team"].ne("")]
    out = out[out["home_team"].ne(out["away_team"])]

    # Prefer completed/scored rows if duplicates are returned by ESPN.
    out["_has_scores"] = out["home_score"].notna() & out["away_score"].notna()
    out = out.sort_values(
        ["season", "week", "_has_scores", "game_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    out = out.drop_duplicates(subset=["game_id"], keep="first")
    out = out.drop(columns=["_has_scores"])

    return out.sort_values(["season", "week", "kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)


def validate_schedule_frame(df: pd.DataFrame) -> dict[str, Any]:
    """Return quality stats and raise for hard schema failures."""
    required = {"season", "week", "game_id", "home_team", "away_team", "game_type"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Schedule missing required columns: {missing}")

    stats = {
        "rows": int(len(df)),
        "duplicate_game_ids": int(df["game_id"].duplicated().sum()) if "game_id" in df else None,
        "missing_home_team": int(df["home_team"].isna().sum()) if "home_team" in df else None,
        "missing_away_team": int(df["away_team"].isna().sum()) if "away_team" in df else None,
        "future_rows_with_scores": 0,
        "game_types": sorted(df["game_type"].dropna().astype(str).unique().tolist()) if "game_type" in df else [],
    }

    if {"completed", "home_score", "away_score"}.issubset(df.columns):
        future = df[~df["completed"].fillna(False).astype(bool)]
        stats["future_rows_with_scores"] = int(
            (future["home_score"].notna() | future["away_score"].notna()).sum()
        )

    hard_errors = []
    if stats["duplicate_game_ids"]:
        hard_errors.append(f"duplicate game_id rows={stats['duplicate_game_ids']}")
    if stats["missing_home_team"]:
        hard_errors.append(f"missing home_team rows={stats['missing_home_team']}")
    if stats["missing_away_team"]:
        hard_errors.append(f"missing away_team rows={stats['missing_away_team']}")
    if stats["future_rows_with_scores"]:
        hard_errors.append(f"future rows with scores={stats['future_rows_with_scores']}")

    if hard_errors:
        raise ValueError("Schedule validation failed: " + "; ".join(hard_errors))

    return stats


def save_schedule(df: pd.DataFrame, *, out_csv: Optional[Path], out_parquet: Optional[Path]) -> None:
    """Write CSV and/or parquet outputs."""
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        log.info("Wrote CSV schedule -> %s (%d rows)", out_csv, len(df))

    if out_parquet is not None:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        log.info("Wrote parquet schedule -> %s (%d rows)", out_parquet, len(df))


def parse_season_types(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and normalize NFL schedule data from ESPN.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--season-types", default="2,3", help="Comma-separated ESPN season types. Default: 2,3")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-parquet", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    df = ingest_schedule(
        season=args.season,
        season_types=parse_season_types(args.season_types),
        raw_dir=args.raw_dir,
    )
    stats = validate_schedule_frame(df)
    log.info("Schedule quality stats: %s", stats)

    save_schedule(df, out_csv=args.out_csv, out_parquet=args.out_parquet)

    if args.out_csv is None and args.out_parquet is None:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
