import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend import builddataset
from backend import main as main_module
from backend.utils.feature_helpers import _impute_remaining_prior_nans
from backend.utils.ops_reporting import resolve_latest_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_builddataset_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "backend/builddataset.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "canonical nfl dataset" in result.stdout.lower()


def test_clean_dataset_prefers_labeled_duplicate_rows():
    df = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "BUF",
                "away_team": "KC",
                "home_points_for": np.nan,
                "away_points_for": np.nan,
                "home_win": np.nan,
            },
            {
                "season": 2025,
                "week": 1,
                "home_team": "BUF",
                "away_team": "KC",
                "home_points_for": 24.0,
                "away_points_for": 20.0,
                "home_win": 1.0,
            },
        ]
    )

    cleaned, stats = builddataset._clean_dataset(df)

    assert len(cleaned) == 1
    assert stats["duplicate_game_ids_removed"] == 1
    assert float(cleaned.iloc[0]["home_points_for"]) == 24.0
    assert float(cleaned.iloc[0]["away_points_for"]) == 20.0


def test_impute_remaining_prior_nans_uses_completed_row_medians():
    df = pd.DataFrame(
        [
            {
                "home_points_for": 21.0,
                "away_points_for": 17.0,
                "home_prior_win_pct_3": 0.62,
                "away_prior_win_pct_3": 0.41,
            },
            {
                "home_points_for": 27.0,
                "away_points_for": 24.0,
                "home_prior_win_pct_3": 0.58,
                "away_prior_win_pct_3": 0.39,
            },
            {
                "home_points_for": np.nan,
                "away_points_for": np.nan,
                "home_prior_win_pct_3": np.nan,
                "away_prior_win_pct_3": np.nan,
            },
        ]
    )

    filled = _impute_remaining_prior_nans(df)

    assert np.isclose(filled.iloc[2]["home_prior_win_pct_3"], 0.60)
    assert np.isclose(filled.iloc[2]["away_prior_win_pct_3"], 0.40)
    assert filled.iloc[2]["home_prior_win_pct_3"] != 0.0


def test_resolve_latest_dataset_prefers_manifest_clean_dataset(tmp_path: Path):
    data_dir = tmp_path / "data"
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir(parents=True)

    older = datasets_dir / "game_features_20260123.csv"
    newer = datasets_dir / "game_features_20260317_clean.csv"
    older.write_text("season,week\n2025,1\n", encoding="utf-8")
    newer.write_text("season,week\n2025,2\n", encoding="utf-8")

    manifest = {
        "clean_dataset_path": str(newer),
        "raw_dataset_path": str(older),
    }
    (datasets_dir / "latest_dataset.json").write_text(json.dumps(manifest), encoding="utf-8")

    resolved = resolve_latest_dataset(data_dir)

    assert resolved == newer.resolve()


def test_validate_bundle_metadata_contract_rejects_missing_required_fields():
    with pytest.raises(RuntimeError, match="missing required contract fields"):
        main_module._validate_bundle_metadata_contract({"serving_mode": "pipeline_primary"})


def test_validate_bundle_metadata_contract_rejects_sklearn_mismatch(monkeypatch):
    monkeypatch.setattr(main_module, "SKLEARN_RUNTIME_VERSION", "1.5.2")
    meta = {
        "serving_mode": "pipeline_primary",
        "feature_manifests": {"score": {"numeric": ["season"], "categorical": []}},
        "generated_features": {"nn_home_win_proba": {"source": "winner_model_predict_proba"}},
        "dataset_hash": "abc123",
        "sklearn_version": "1.7.2",
        "bundle_timestamp_utc": "2026-03-20T00:00:00+00:00",
    }

    with pytest.raises(RuntimeError, match="scikit-learn 1.7.2"):
        main_module._validate_bundle_metadata_contract(meta)


def test_health_keeps_legacy_bundle_contract_as_warning(monkeypatch):
    monkeypatch.setattr(main_module.state, "dataset", pd.DataFrame([{"season": 2025, "week": 1}]))
    monkeypatch.setattr(main_module.state, "models", {"home": object(), "away": object(), "win": object()})
    monkeypatch.setattr(main_module.state, "production_blockers", [])
    monkeypatch.setattr(
        main_module.state,
        "production_warnings",
        ["legacy model bundle contract", "metadata.json missing strict contract"],
    )

    payload = main_module.health()

    assert payload.status == "healthy"
    assert payload.production_ready is True
    assert payload.reason is None
    assert payload.components.ready_for_production is True
    assert payload.components.blockers == []
    assert "legacy model bundle contract" in payload.components.warnings
