from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend import main as main_module


class _DummyWinPipeline:
    def __init__(self, calls):
        self.calls = calls
        self.steps = [("pre", object()), ("clf", object())]
        self.classes_ = np.asarray([0, 1], dtype=int)

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
    monkeypatch.setattr(main_module.state, "model_load_errors", {})
    monkeypatch.setattr(main_module.state, "production_warnings", [])
    monkeypatch.setattr(main_module.state, "production_blockers", [])

    with TestClient(main_module.app) as client:
        response = client.post(
            "/predict",
            json={"home_team": "BUF", "away_team": "KC", "season": 2025, "week": 1},
        )

    assert response.status_code == 200
    payload = response.json()

    assert calls == ["win", "home", "away"]
    assert np.isclose(payload["home_score"], 20.0)
    assert np.isclose(payload["away_score"], 17.0)
    assert np.isclose(payload["home_win_probability"], 0.9)
    assert np.isclose(payload["away_win_probability"], 0.1)
    assert payload["explanation_fields"]["dataset_hash"] == "unit-test-dataset"
    assert payload["win_classifier_used"] is True
    assert score_manifest == main_module._feature_manifest("scores")


def test_calculate_win_probability_uses_positive_class_label(monkeypatch):
    class _ReversedClassesWinModel:
        def __init__(self):
            self.classes_ = np.asarray([1, 0], dtype=int)

        def predict_proba(self, features):
            return np.asarray([[0.83, 0.17]], dtype=float)

    full_df = pd.DataFrame([{"home_moneyline_prob": 0.55}])
    numeric_df = pd.DataFrame([{"home_moneyline_prob": 0.55}])

    win_prob, clf_used = main_module._calculate_win_probability(
        _ReversedClassesWinModel(),
        full_df,
        numeric_df,
        preprocessor=None,
    )

    assert clf_used is True
    assert np.isclose(win_prob, 0.83)
