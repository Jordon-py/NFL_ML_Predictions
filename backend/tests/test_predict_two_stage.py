from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend import main as main_module


class _DummyWinPipeline:
    def __init__(self, calls):
        self.calls = calls
        self.steps = [("pre", object()), ("clf", object())]

    def predict_proba(self, full_df):
        self.calls.append("win")
        assert main_module.WIN_PROBA_FEATURE not in full_df.columns
        return np.asarray([[0.1, 0.9]], dtype=float)


class _DummyScorePipeline:
    def __init__(self, calls, name: str, expected_prob: float, score: float):
        self.calls = calls
        self.name = name
        self.expected_prob = expected_prob
        self.score = score
        self.steps = [("pre", object()), ("reg", object())]
        self.feature_names_in_ = np.asarray(
            [
                "season",
                "week",
                "feature_a",
                "home_moneyline_prob",
                main_module.WIN_PROBA_FEATURE,
                "home_team",
                "away_team",
            ],
            dtype=object,
        )

    def predict(self, full_df):
        self.calls.append(self.name)
        assert main_module.WIN_PROBA_FEATURE in full_df.columns
        assert np.isclose(float(full_df.iloc[0][main_module.WIN_PROBA_FEATURE]), self.expected_prob)
        return np.asarray([self.score], dtype=float)


def test_predict_feeds_raw_win_probability_into_score_models(monkeypatch):
    calls = []
    dataset = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "BUF",
                "away_team": "KC",
                "time_key": 202501,
                "feature_a": 1.25,
                "home_moneyline_prob": 0.62,
            }
        ]
    )
    score_manifest = [
        "season",
        "week",
        "feature_a",
        "home_moneyline_prob",
        main_module.WIN_PROBA_FEATURE,
        "home_team",
        "away_team",
    ]

    monkeypatch.setattr(main_module.state, "load", lambda: None)
    monkeypatch.setattr(main_module.state, "refresh_dataset_if_changed", lambda: False)
    monkeypatch.setattr(main_module, "_roll_forward_missing_player_stats", lambda **kwargs: kwargs["row_df"])
    monkeypatch.setattr(main_module.state, "dataset", dataset)
    monkeypatch.setattr(main_module.state, "dataset_path", Path("C:/tmp/game_features_test.csv"))
    monkeypatch.setattr(main_module.state, "dataset_hash", "unit-test-dataset")
    monkeypatch.setattr(
        main_module.state,
        "numeric_medians",
        pd.Series(
            {
                "season": 2025.0,
                "week": 1.0,
                "feature_a": 1.25,
                "home_moneyline_prob": 0.62,
                main_module.WIN_PROBA_FEATURE: 0.5,
            }
        ),
    )
    monkeypatch.setattr(
        main_module.state,
        "models",
        {
            "win": _DummyWinPipeline(calls),
            "home": _DummyScorePipeline(calls, "home", expected_prob=0.9, score=20.0),
            "away": _DummyScorePipeline(calls, "away", expected_prob=0.9, score=17.0),
        },
    )
    monkeypatch.setattr(
        main_module.state,
        "models_metadata",
        {
            "raw_feature_columns": {
                "win": {
                    "numeric": ["season", "week", "feature_a", "home_moneyline_prob"],
                    "categorical": ["home_team", "away_team"],
                },
                "score": {
                    "numeric": ["season", "week", "feature_a", "home_moneyline_prob", main_module.WIN_PROBA_FEATURE],
                    "categorical": ["home_team", "away_team"],
                },
            },
            "generated_features": {
                main_module.WIN_PROBA_FEATURE: {
                    "source": "winner_model_predict_proba",
                }
            },
        },
    )
    monkeypatch.setattr(main_module.state, "feature_manifest", [])
    monkeypatch.setattr(main_module.state, "preprocessor", None)
    monkeypatch.setattr(main_module.state, "score_preprocessor", None)
    monkeypatch.setattr(main_module.state, "win_preprocessor", None)
    monkeypatch.setattr(main_module.state, "history", [])
    monkeypatch.setattr(main_module.state, "predict_cache", {})
    monkeypatch.setattr(main_module.state, "predict_cache_hits", 0)
    monkeypatch.setattr(main_module.state, "predict_cache_misses", 0)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/predict",
            json={"home_team": "BUF", "away_team": "KC", "season": 2025, "week": 1},
        )

    assert response.status_code == 200
    payload = response.json()

    assert calls == ["win", "home", "away"]
    expected_smoothed = main_module._smooth_win_probability(0.9, 3.0, clf_used=True)
    assert np.isclose(payload["home_score"], 20.0)
    assert np.isclose(payload["away_score"], 17.0)
    assert np.isclose(payload["home_win_probability"], expected_smoothed)
    assert not np.isclose(payload["home_win_probability"], 0.9)
    assert np.isclose(payload["away_win_probability"], 1.0 - expected_smoothed)
    assert payload["explanation_fields"]["dataset_hash"] == "unit-test-dataset"
    assert payload["win_classifier_used"] is True
    assert score_manifest == main_module._feature_manifest("scores")
