from pathlib import Path

import pandas as pd

from backend.services import api_runtime as svc


def test_prediction_ai_payload_exposes_slate_dataset_and_model_learning(monkeypatch, tmp_path):
    dataset_path = tmp_path / "game_features.csv"
    dataset_path.write_text("season,week,home_team,away_team\n2026,1,HOU,BUF\n", encoding="utf-8")

    monkeypatch.setattr(svc.state, "dataset", pd.DataFrame({"season": [2026], "week": [1]}))
    monkeypatch.setattr(svc.state, "dataset_path", dataset_path)
    monkeypatch.setattr(svc.state, "dataset_hash", "hash123")

    prediction = {
        "season": 2026,
        "week": 1,
        "home_team": "HOU",
        "away_team": "BUF",
        "game_id": "2026_1_HOU_BUF",
        "home_score": 19.7,
        "away_score": 24.2,
        "home_win_probability": 0.298,
        "away_win_probability": 0.702,
        "point_diff": -4.5,
        "prediction_source": "pipeline_primary",
        "mode": "production",
        "win_classifier_used": True,
        "explanation_fields": {"row_quality_score": 98.0},
    }

    enriched = svc._attach_prediction_access_fields(dict(prediction), None)

    assert enriched["predicted_winner"] == "BUF"
    assert enriched["confidence_tier"] == "STRONG_EDGE"
    assert enriched["slate"]["label"] == "2026 Week 1"
    assert enriched["dataset_access"]["dataset_hash"] == "hash123"
    assert enriched["dataset_access"]["metadata_url"] == "/metadata/dataset"
    assert enriched["model_learning"]["plot"]["url"] == "/artifacts/models/training-metrics-plot.png"
    assert enriched["ai_payload"]["prediction"]["score"] == "BUF 24.2 - 19.7 HOU"


def test_slate_chat_prompt_requires_explicit_slate_line():
    context = {
        "slate": {"label": "2026 Week 1"},
        "favorites": [
            {
                "matchup": "BUF @ HOU",
                "predicted_winner": "BUF",
                "projected_score": "BUF 24.2 - 19.7 HOU",
                "favorite_win_probability": 0.702,
                "confidence_tier": "STRONG_EDGE",
            }
        ],
        "dataset_access": {"dataset_hash": "hash123"},
        "model_learning": {
            "plot": {"url": "/artifacts/models/training-metrics-plot.png"}
        },
    }

    prompt = svc._slate_chat_prompt("next week's favorites", context)

    assert "Start with this exact line: Slate: 2026 Week 1" in prompt
    assert "Do not say 'based on payloads processed in this session'" in prompt
    assert "training-metrics-plot.png" in prompt
