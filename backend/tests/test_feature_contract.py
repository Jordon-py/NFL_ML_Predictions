import pandas as pd

from backend.contracts.feature_contract import (
    align_feature_frame,
    build_feature_contract,
    validate_feature_frame,
)


def test_feature_contract_detects_missing_columns_and_aligns_order():
    metadata = {
        "dataset_hash": "abc123",
        "feature_manifests": {
            "win": {
                "numeric": ["season", "week", "spread_line"],
                "categorical": ["home_team_KC"],
            }
        },
    }
    contract = build_feature_contract(metadata, "win")
    frame = pd.DataFrame([{"week": 1, "season": 2025, "extra": 99}])

    result = validate_feature_frame(frame, contract)

    assert result.ok is False
    assert result.missing_columns == ["spread_line", "home_team_KC"]
    assert result.order_mismatch is True

    aligned = align_feature_frame(frame, contract)
    assert list(aligned.columns) == ["season", "week", "spread_line", "home_team_KC"]


def test_generated_feature_can_be_required_for_prediction_frame():
    metadata = {
        "dataset_hash": "abc123",
        "feature_manifests": {
            "score": {
                "numeric": ["season", "nn_home_win_proba"],
                "categorical": [],
            }
        },
        "generated_features": {
            "nn_home_win_proba": {"source": "winner_model_predict_proba"}
        },
    }
    contract = build_feature_contract(metadata, "score")

    dataset_result = validate_feature_frame(
        pd.DataFrame([{"season": 2025}]),
        contract,
        allow_generated_missing=True,
    )
    prediction_result = validate_feature_frame(
        pd.DataFrame([{"season": 2025}]),
        contract,
        allow_generated_missing=False,
    )

    assert dataset_result.ok is True
    assert prediction_result.ok is False
    assert prediction_result.missing_columns == ["nn_home_win_proba"]
