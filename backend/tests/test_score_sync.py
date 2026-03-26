import pandas as pd

from backend.score_sync import build_score_game_id, extract_score_entries_from_dataframe


def test_extract_score_entries_from_dataset_uses_prediction_game_id_shape():
    df = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 22,
                "home_team": "ne",
                "away_team": "sea",
                "home_points_for": 27,
                "away_points_for": 20,
            }
        ]
    )

    entries = extract_score_entries_from_dataframe(df, updated_at="2026-03-25T12:00:00+00:00")
    assert entries == [
        {
            "game_id": build_score_game_id(2025, 22, "NE", "SEA"),
            "season": 2025,
            "week": 22,
            "home_team": "NE",
            "away_team": "SEA",
            "home_score": 27,
            "away_score": 20,
            "status": "final",
            "updated_at": "2026-03-25T12:00:00+00:00",
        }
    ]


def test_extract_score_entries_ignores_future_rows_without_scores():
    df = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 15,
                "home_team": "BUF",
                "away_team": "KC",
                "home_score": None,
                "away_score": None,
            }
        ]
    )

    assert extract_score_entries_from_dataframe(df) == []
