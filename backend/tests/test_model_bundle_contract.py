from backend.contracts.model_bundle_contract import validate_model_bundle_contract


def test_model_bundle_contract_accepts_strict_bundle_with_top_level_calibration(tmp_path):
    for filename in (
        "home_pipe.joblib",
        "away_pipe.joblib",
        "win_pipe.joblib",
        "score_preprocessor.joblib",
        "win_preprocessor.joblib",
        "metadata.json",
    ):
        (tmp_path / filename).write_text("x", encoding="utf-8")

    metadata = {
        "serving_mode": "pipeline_primary",
        "bundle_contract_version": 2,
        "bundle_timestamp_utc": "2026-05-26T00:00:00+00:00",
        "dataset_hash": "abc123",
        "sklearn_version": "1.7.2",
        "feature_manifests": {
            "win": {"numeric": ["season"], "categorical": []},
            "score": {"numeric": ["season", "nn_home_win_proba"], "categorical": []},
        },
        "generated_features": {
            "nn_home_win_proba": {"source": "winner_model_predict_proba"}
        },
        "metrics": {
            "calibration": {"expected_calibration_error": 0.05}
        },
        "artifacts": {
            "reg_home": "home_pipe.joblib",
            "reg_away": "away_pipe.joblib",
            "clf_home_win": "win_pipe.joblib",
            "score_preprocessor": "score_preprocessor.joblib",
            "win_preprocessor": "win_preprocessor.joblib",
        },
    }

    result = validate_model_bundle_contract(
        models_dir=tmp_path,
        metadata=metadata,
        dataset_hash="abc123",
        sklearn_runtime_version="1.7.2",
    )

    assert result.ok is True
    assert result.strict is True
    assert result.dataset_hash_match is True
    assert result.calibration_metadata_present is True
    assert result.blockers == []


def test_model_bundle_contract_blocks_dataset_hash_mismatch(tmp_path):
    for filename in ("home_pipe.joblib", "away_pipe.joblib", "win_pipe.joblib"):
        (tmp_path / filename).write_text("x", encoding="utf-8")

    result = validate_model_bundle_contract(
        models_dir=tmp_path,
        metadata={
            "serving_mode": "pipeline_primary",
            "bundle_timestamp_utc": "2026-05-26T00:00:00+00:00",
            "dataset_hash": "trained",
            "sklearn_version": "1.7.2",
            "feature_manifests": {
                "win": {"numeric": ["season"], "categorical": []},
                "score": {"numeric": ["season"], "categorical": []},
            },
            "generated_features": {},
        },
        dataset_hash="active",
        sklearn_runtime_version="1.7.2",
    )

    assert result.ok is False
    assert "active dataset hash does not match model bundle training dataset hash" in result.blockers
