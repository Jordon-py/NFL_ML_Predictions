"""Debug ESPN scoreboard event parsing into canonical game IDs.

Data shape:
- Input: ESPN scoreboard `events` JSON list. Each event should include
  `season.year`, `week.number`, `competitions[0].competitors`, and final status
  metadata.
- Output: list of canonical game ID strings shaped as `YYYY_WW_AWAY_HOME`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import _build_game_id, _normalize_team_code, _parse_score_value


def debug_entries(events):
    """Convert final ESPN scoreboard events into canonical game IDs."""
    entries = []
    print(f"Num events: {len(events)}")
    for i, event in enumerate(events):
        print(f"--- Event {i} ---")
        season = event.get("season", {})
        week = event.get("week", {})
        season_value = season.get("year")
        week_value = week.get("number")
        if not season_value or not week_value:
            print("Skipping: no season_value or week_value")
            continue

        comps = event.get("competitions") or []
        if not comps:
            print("Skipping: no competitions")
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home_side = [c for c in competitors if str(c.get("homeAway")).lower() == "home"]
        away_side = [c for c in competitors if str(c.get("homeAway")).lower() == "away"]
        if not home_side or not away_side:
            print("Skipping: missing home or away side")
            continue
            
        home = home_side[0]
        away = away_side[0]
        home_team = _normalize_team_code(
            home.get("team", {}).get("abbreviation") or home.get("team", {}).get("shortDisplayName")
        )
        away_team = _normalize_team_code(
            away.get("team", {}).get("abbreviation") or away.get("team", {}).get("shortDisplayName")
        )
        _home_score = _parse_score_value(home.get("score"))
        _away_score = _parse_score_value(away.get("score"))
        
        status = event.get("status", {})
        status_type = status.get("type", {})
        state = str(status_type.get("state", "")).lower()
        is_final = (
            state in {"post", "final", "complete"}
            or status_type.get("completed") is True
            or str(status_type.get("name", "")).lower() == "final"
        )
        if not is_final:
            print(f"Skipping: not final. state={state}, completed={status_type.get('completed')}, name={status_type.get('name')}")
            continue

        game_id = _build_game_id(season_value, week_value, home_team, away_team)
        entries.append(game_id)
        print(f"Added: {game_id}")
        
    return entries


def main() -> None:
    """Fetch one sample scoreboard event and print parser diagnostics."""
    events = httpx.get(
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20250907",
        timeout=30,
    ).json().get("events", [])
    if not events:
        raise SystemExit("No ESPN events returned for sample date")
    debug_entries([events[0]])


if __name__ == "__main__":
    main()
