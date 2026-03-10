import json
from pathlib import Path

import pandas as pd

from backend.builddataset import _clean_dataset
from backend.prediction_store import (
    append_prediction_record,
    build_prediction_user_context,
    get_prediction_history,
    get_prediction_history_count,
)
from backend.schemas import PredictionRequest
from backend.services.inference_row import _infer_expected_columns


def test_clean_dataset_builds_game_id_and_dedupes():
    df = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "home_win": 1,
                "feature_value": 10.0,
            },
            {
                "season": 2025,
                "week": 1,
                "home_team": "kc",
                "away_team": "buf",
                "home_win": 1,
                "feature_value": None,
            },
            {
                "season": None,
                "week": None,
                "home_team": None,
                "away_team": None,
                "home_win": None,
                "feature_value": None,
            },
        ]
    )

    cleaned, stats = _clean_dataset(df)

    assert stats["blank_rows_removed"] == 1
    assert stats["duplicate_game_ids_removed"] == 1
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["game_id"] == "2025-1-KC-BUF"


def test_prediction_history_is_scoped_per_user(tmp_path, monkeypatch):
    import backend.prediction_store as prediction_store

    monkeypatch.setattr(prediction_store, "PREDICTION_STORE_ROOT", tmp_path)

    alice = build_prediction_user_context("alice@example.com")
    bob = build_prediction_user_context("bob@example.com")
    request = PredictionRequest(home_team="KC", away_team="BUF", season=2025, week=1)

    append_prediction_record(
        alice,
        request,
        {
            "home_score": 27.0,
            "away_score": 23.0,
            "point_diff": 4.0,
            "home_win_probability": 0.62,
            "away_win_probability": 0.38,
            "prediction_source": "unit-test",
            "win_classifier_used": True,
            "game_id": "2025-1-KC-BUF",
            "season": 2025,
            "week": 1,
            "home_team": "KC",
            "away_team": "BUF",
        },
    )
    append_prediction_record(
        bob,
        request,
        {
            "home_score": 17.0,
            "away_score": 24.0,
            "point_diff": -7.0,
            "home_win_probability": 0.33,
            "away_win_probability": 0.67,
            "prediction_source": "unit-test",
            "win_classifier_used": True,
            "game_id": "2025-1-KC-BUF",
            "season": 2025,
            "week": 1,
            "home_team": "KC",
            "away_team": "BUF",
        },
    )

    alice_history = get_prediction_history(alice, limit=10)
    bob_history = get_prediction_history(bob, limit=10)

    assert get_prediction_history_count(alice) == 1
    assert get_prediction_history_count(bob) == 1
    assert alice_history.total == 1
    assert bob_history.total == 1
    assert alice_history.entries[0].user_id == "alice@example.com"
    assert bob_history.entries[0].user_id == "bob@example.com"
    assert alice_history.entries[0].home_win_probability != bob_history.entries[0].home_win_probability

    alice_profile_path = Path(tmp_path) / alice.storage_key / "profile.json"
    alice_profile = json.loads(alice_profile_path.read_text(encoding="utf-8"))
    assert alice_profile["retained_predictions"] == 1
    assert alice_profile["total_predictions_all_time"] == 1


def test_infer_expected_columns_prefers_explicit_metadata_contract():
    class StubPreprocessor:
        feature_names_in_ = ["season", "week", "feature_a"]

    expected = _infer_expected_columns(
        StubPreprocessor(),
        raw_feature_columns=["season", "week", "feature_a", "feature_b"],
    )

    assert expected == ["season", "week", "feature_a", "feature_b"]
