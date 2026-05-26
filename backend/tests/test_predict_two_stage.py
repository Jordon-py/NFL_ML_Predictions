import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import prediction_store, sqlite_store


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


def test_predict_strict_bundle_fails_loudly_when_win_classifier_falls_back(monkeypatch):
    dataset = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "BUF",
                "away_team": "KC",
                "time_key": 202501,
                "feature_a": 1.25,
            }
        ]
    )

    monkeypatch.setattr(main_module.state, "load", lambda: None)
    monkeypatch.setattr(main_module.state, "refresh_dataset_if_changed", lambda: False)
    monkeypatch.setattr(main_module, "_roll_forward_missing_player_stats", lambda **kwargs: kwargs["row_df"])
    monkeypatch.setattr(main_module.state, "dataset", dataset)
    monkeypatch.setattr(main_module.state, "dataset_path", Path("C:/tmp/game_features_test.csv"))
    monkeypatch.setattr(main_module.state, "dataset_hash", "unit-test-dataset")
    monkeypatch.setattr(
        main_module.state,
        "numeric_medians",
        pd.Series({"season": 2025.0, "week": 1.0, "feature_a": 1.25}),
    )
    monkeypatch.setattr(
        main_module.state,
        "models",
        {"win": object(), "home": object(), "away": object()},
    )
    monkeypatch.setattr(
        main_module.state,
        "models_metadata",
        {
            "serving_mode": "pipeline_primary",
            "bundle_contract_version": 2,
            "feature_manifests": {"score": {"numeric": ["season", "week"], "categorical": []}},
            "generated_features": {
                main_module.WIN_PROBA_FEATURE: {"source": "winner_model_predict_proba"}
            },
            "dataset_hash": "unit-test-dataset",
            "sklearn_version": main_module.SKLEARN_RUNTIME_VERSION or "unit-test",
            "bundle_timestamp_utc": "2026-05-21T00:00:00+00:00",
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

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["message"] == "Prediction service unavailable."
    assert "win classifier unavailable for strict model bundle" in detail["blockers"]


def test_real_2025_prediction_uses_classifier_with_verified_bundle(monkeypatch, tmp_path):
    metadata_path = main_module.BASE_DIR / "models" / "metadata.json"
    if not metadata_path.exists():
        pytest.skip("verified backend/models metadata is not available in this checkout")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_sklearn = str(metadata.get("sklearn_version") or "").strip()
    if required_sklearn and main_module.SKLEARN_RUNTIME_VERSION != required_sklearn:
        pytest.skip(
            f"verified bundle requires scikit-learn {required_sklearn}; "
            f"runtime has {main_module.SKLEARN_RUNTIME_VERSION}"
        )

    monkeypatch.setattr(main_module, "MODELS_DIR", main_module.BASE_DIR / "models")
    monkeypatch.setattr(prediction_store, "PREDICTION_STORE_ROOT", tmp_path / "Predictions" / "users")
    monkeypatch.setattr(sqlite_store, "DB_PATH", tmp_path / "predictions.db")

    with TestClient(main_module.app) as client:
        status = client.get("/status/models")
        assert status.status_code == 200
        status_payload = status.json()
        assert status_payload["ready"] is True
        assert status_payload["provenance"]["bundle_version"] == metadata["bundle_version"]

        main_module.state.predict_cache.clear()
        response = client.post(
            "/predict",
            json={"home_team": "PHI", "away_team": "DAL", "season": 2025, "week": 1},
            headers={"X-User-Id": "regression@example.invalid"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["win_classifier_used"] is True
    assert payload["explanation_fields"]["selected_row_source"] == "dataset_exact"
    assert payload["explanation_fields"]["dataset_hash"] == metadata["dataset_hash"]


def test_real_2026_synthetic_prediction_uses_classifier_with_verified_bundle(monkeypatch, tmp_path):
    metadata_path = main_module.BASE_DIR / "models" / "metadata.json"
    if not metadata_path.exists():
        pytest.skip("verified backend/models metadata is not available in this checkout")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_sklearn = str(metadata.get("sklearn_version") or "").strip()
    if required_sklearn and main_module.SKLEARN_RUNTIME_VERSION != required_sklearn:
        pytest.skip(
            f"verified bundle requires scikit-learn {required_sklearn}; "
            f"runtime has {main_module.SKLEARN_RUNTIME_VERSION}"
        )

    monkeypatch.setattr(main_module, "MODELS_DIR", main_module.BASE_DIR / "models")
    monkeypatch.setattr(prediction_store, "PREDICTION_STORE_ROOT", tmp_path / "Predictions" / "users")
    monkeypatch.setattr(sqlite_store, "DB_PATH", tmp_path / "predictions.db")

    with TestClient(main_module.app) as client:
        status = client.get("/status/models")
        assert status.status_code == 200
        status_payload = status.json()
        assert status_payload["ready"] is True
        assert status_payload["provenance"]["bundle_version"] == metadata["bundle_version"]

        main_module.state.predict_cache.clear()
        response = client.post(
            "/predict",
            json={"home_team": "CAR", "away_team": "CHI", "season": 2026, "week": 1},
            headers={"X-User-Id": "synthetic-regression@example.invalid"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["win_classifier_used"] is True
    assert payload["explanation_fields"]["selected_row_source"] == "synthetic"
    assert payload["explanation_fields"]["dataset_hash"] == metadata["dataset_hash"]
