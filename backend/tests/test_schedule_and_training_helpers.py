import pandas as pd
from datetime import datetime, timezone

from backend.main_helpers import select_next_week_rows
from backend.train_models import _next_monthly_refresh


def test_select_next_week_rows_removes_duplicate_matchups():
    schedule_df = pd.DataFrame(
        [
            {"season": 2025, "week": 18, "home_team": "ATL", "away_team": "NO", "game_id": "2025_18_NO_ATL"},
            {"season": 2025, "week": 18, "home_team": "ATL", "away_team": "NO", "game_id": "2025_18_NO_ATL"},
            {"season": 2025, "week": 18, "home_team": "BUF", "away_team": "NYJ", "game_id": "2025_18_NYJ_BUF"},
        ]
    )

    df_next, season, week = select_next_week_rows(schedule_df)

    assert season == 2025
    assert week == 18
    assert len(df_next) == 2
    assert {
        f"{row.home_team}-{row.away_team}"
        for row in df_next.itertuples(index=False)
    } == {"ATL-NO", "BUF-NYJ"}


def test_next_monthly_refresh_stops_when_next_month_is_offseason():
    trained_at = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)

    assert _next_monthly_refresh(trained_at) is None
