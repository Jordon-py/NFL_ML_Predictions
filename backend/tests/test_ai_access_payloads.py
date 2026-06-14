from pathlib import Path
import asyncio

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


def test_parse_expert_prediction_json_returns_three_sentence_contract():
    expert = svc._parse_expert_prediction_json(
        """
        {
          "home_score": 27.4,
          "away_score": 20.2,
          "home_win_probability": 0.68,
          "confidence": 0.68,
          "predicted_winner": "HOU",
          "reasoning": [
            "Houston keeps the model edge after record and injury context.",
            "Buffalo still projects close because the raw ML margin is modest.",
            "The confidence stays calibrated rather than overstated."
          ]
        }
        """,
        home_team="HOU",
        away_team="BUF",
        fallback_prediction={"home_win_probability": 0.61},
    )

    assert expert["used_llm"] is True
    assert expert["home_score"] == 27.4
    assert expert["away_score"] == 20.2
    assert expert["predicted_winner"] == "HOU"
    assert expert["confidence_percentage"] == 68
    assert len(expert["reasoning_sentences"]) == 3


def test_expert_prediction_layer_replaces_display_fields_and_preserves_ml(monkeypatch):
    async def fake_context(**_kwargs):
        return {
            "sources": ["backend dataset", "ESPN Core API"],
            "dataset_matchup_context": {"nn_home_win_proba": 0.61},
            "current_records": {},
            "previous_season_standings": {},
            "injuries": {},
        }

    async def fake_expert(**_kwargs):
        return {
            "used_llm": True,
            "home_score": 28.0,
            "away_score": 21.0,
            "home_win_probability": 0.7,
            "away_win_probability": 0.3,
            "confidence": 0.7,
            "confidence_percentage": 70,
            "predicted_winner": "HOU",
            "reasoning": (
                "Houston gets the final edge after the model and context are combined. "
                "The raw neural-assisted model still anchors the score. "
                "The confidence is capped because the matchup context is not a blowout."
            ),
            "model": "gemma4:31b-cloud",
            "host": "https://ollama.com",
        }

    monkeypatch.setattr(svc, "_build_expert_matchup_context", fake_context)
    monkeypatch.setattr(svc, "_generate_expert_prediction", fake_expert)
    monkeypatch.setattr(
        svc.state,
        "models_metadata",
        {
            "generated_features": {
                "nn_home_win_proba": {"source": "winner_model_predict_proba"}
            },
            "metrics": {
                "regression": {
                    "component_models": {
                        "home": {"mlp": {"mae": 7.1}},
                        "away": {"mlp": {"mae": 6.9}},
                    }
                },
                "classification": {"accuracy": 0.67},
                "calibration": {"expected_calibration_error": 0.07},
            },
        },
    )
    prediction = {
        "season": 2026,
        "week": 1,
        "home_team": "HOU",
        "away_team": "BUF",
        "game_id": "2026_1_HOU_BUF",
        "home_score": 24.0,
        "away_score": 23.0,
        "home_win_probability": 0.58,
        "away_win_probability": 0.42,
        "point_diff": 1.0,
        "prediction_source": "pipeline_primary",
        "win_classifier_used": True,
    }
    row_df = pd.DataFrame([{"home_prior_win_pct_5": 0.6}])
    score_df = pd.DataFrame([{"nn_home_win_proba": 0.61}])

    result = asyncio.run(
        svc._apply_expert_prediction_layer(prediction, row_df=row_df, score_full_df=score_df, request=None)
    )

    assert result["home_score"] == 28.0
    assert result["away_score"] == 21.0
    assert result["home_win_probability"] == 0.7
    assert result["prediction_source"] == "gemma_cloud_expert_calibrated"
    assert result["model_prediction"]["home_score"] == 24.0
    assert result["model_prediction"]["neural_network_used"] is True
    assert result["expert_prediction"]["used_llm"] is True
    assert result["expert_model_used"] == "gemma4:31b-cloud"
