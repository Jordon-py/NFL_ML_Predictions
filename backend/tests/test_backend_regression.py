import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor

import backend.main as main
from backend.contracts.feature_contract import (
    align_feature_frame,
    build_feature_contract,
    validate_feature_frame,
)
from backend.contracts.model_bundle_contract import validate_model_bundle_contract
from backend.scripts import build_csv_datasets_v3 as csv_builder
from backend.scripts import builddataset, train_models
from backend.services.pipeline_status import build_pipeline_status
from backend.utils.ops_reporting import file_sha256, resolve_latest_dataset


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "backend" / "models"


class _PartialStatsBackend:
    def load_player_stats(self, seasons, summary_level="week"):
        if 2026 in seasons:
            raise RuntimeError("stats_player_week_2026.parquet not found")
        return pd.DataFrame(
            [
                {
                    "season": season,
                    "week": 1,
                    "recent_team": "KC",
                    "position": "QB",
                    "passing_yards": 280,
                    "passing_tds": 2,
                    "interceptions": 0,
                    "sacks": 1,
                    "completions": 24,
                    "attempts": 32,
                }
                for season in seasons
            ]
        )

    def load_team_stats(self, seasons, summary_level="week"):
        if 2026 in seasons:
            raise RuntimeError("stats_team_week_2026.parquet not found")
        return pd.DataFrame(
            [
                {
                    "season": season,
                    "week": 1,
                    "team": "KC",
                    "points_scored": 24,
                    "points_allowed": 17,
                    "total_yards": 360,
                    "total_yards_allowed": 290,
                    "turnovers": 1,
                    "turnovers_forced": 2,
                }
                for season in seasons
            ]
        )


def _freeze_readiness_refresh(monkeypatch):
    monkeypatch.setattr(main.state, "_refresh_runtime_readiness", lambda: None)


def _set_ready_basics(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "MODELS_DIR", models_dir)
    monkeypatch.setattr(main.state, "dataset", pd.DataFrame([{"season": 2026, "week": 1}]))
    monkeypatch.setattr(main.state, "dataset_path", tmp_path / "dataset.csv")
    monkeypatch.setattr(main.state, "dataset_hash", "dataset-hash")
    monkeypatch.setattr(main.state, "dataset_manifest", {"dataset_hash": "dataset-hash", "run_id": "run-1"})
    monkeypatch.setattr(main.state, "models", {"home": object(), "away": object(), "win": object()})
    monkeypatch.setattr(
        main.state,
        "models_metadata",
        {
            "serving_mode": "pipeline_primary",
            "bundle_contract_version": 2,
            "dataset_hash": "dataset-hash",
            "bundle_timestamp_utc": "2026-06-13T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(main.state, "runtime_contract_validation", {"ok": True, "blockers": [], "warnings": []})
    monkeypatch.setattr(main.state, "production_blockers", [])
    monkeypatch.setattr(main.state, "production_warnings", [])
    return models_dir


def test_player_stats_preserve_available_seasons_when_future_season_missing(monkeypatch):
    monkeypatch.setattr(csv_builder, "nfl", _PartialStatsBackend())
    monkeypatch.setattr(csv_builder, "NFL_BACKEND", "nflreadpy")

    out = csv_builder.load_player_game_stats([2025, 2026])

    assert out["season"].tolist() == [2025]
    assert out.loc[0, "team"] == "KC"
    assert out.loc[0, "team_qb_completion_pct"] == 24 / 32


def test_team_stats_preserve_available_seasons_when_future_season_missing(monkeypatch):
    monkeypatch.setattr(csv_builder, "nfl", _PartialStatsBackend())
    monkeypatch.setattr(csv_builder, "NFL_BACKEND", "nflreadpy")

    out = csv_builder.load_team_weekly_stats([2025, 2026])

    assert out["season"].tolist() == [2025]
    assert out.loc[0, "team"] == "KC"
    assert out.loc[0, "points_scored"] == 24


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
        "generated_features": {"nn_home_win_proba": {"source": "winner_model_predict_proba"}},
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
        "generated_features": {"nn_home_win_proba": {"source": "winner_model_predict_proba"}},
        "metrics": {"calibration": {"expected_calibration_error": 0.05}},
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


def test_latest_dataset_manifest_resolves_to_existing_csv():
    dataset_path = resolve_latest_dataset(ROOT / "backend" / "data")

    assert dataset_path.exists()
    assert dataset_path.name.endswith("_clean.csv")


def test_active_feature_manifest_has_no_remaining_hard_leaks():
    manifest_path = MODELS_DIR / "feature_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["hard_leak_columns_remaining"] == []
    assert payload["dropped_reason_counts"]["same_week_player_stat"] >= 1


def test_active_metadata_exposes_canonical_artifact_keys():
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    artifacts = metadata["artifacts"]

    for key in ("preprocessor", "reg_home", "reg_away", "clf_home_win"):
        artifact_path = MODELS_DIR / artifacts[key]
        assert artifact_path.exists(), f"{key} points at missing artifact {artifact_path}"

    assert metadata["gate"]["passed"] is True
    assert metadata["feature_selection"]["hard_leak_columns_remaining"] == []


def test_pipeline_status_reports_stale_dataset_and_feature_coverage(tmp_path):
    dataset = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 22,
                "home_team": "KC",
                "away_team": "BUF",
                "home_points_for": 24,
                "away_points_for": 21,
                "spread_line": -2.5,
                "total_line": 47.5,
                "home_rest": 7,
                "away_rest": 6,
                "home_moneyline_prob": 0.55,
                "away_moneyline_prob": 0.45,
                "home_prior_pf_avg_3": 28.0,
                "away_prior_pf_avg_3": 25.0,
                "home_prior_off_epa_per_play_3": 0.12,
                "away_prior_off_epa_per_play_3": 0.05,
                "home_team_KC": 1,
                "away_team_BUF": 1,
            }
        ]
    )

    status = build_pipeline_status(
        backend_dir=tmp_path,
        models_dir=tmp_path,
        dataset=dataset,
        dataset_path=tmp_path / "game_features.csv",
        dataset_hash="abc123",
        dataset_manifest={},
        dataset_metadata={},
        models_metadata={},
        loaded_models=["home", "away", "win"],
        missing_required_models=[],
        model_load_errors={},
        production_blockers=[],
        production_warnings=[],
        runtime_contract_validation={"ok": True, "blockers": [], "warnings": []},
        schedule_df=dataset,
    )

    assert status.dataset.rows == 1
    assert status.dataset.stale is True
    assert status.dataset.stale_reason == "dataset max season 2025 is older than 2026"
    assert status.dataset.critical_null_rates["spread_line"] == 0.0
    assert any(group.group == "prior_team_form" and group.coverage == 1.0 for group in status.dataset.feature_groups)
    assert status.warning_only is True


def test_prediction_readiness_payload_ready(monkeypatch, tmp_path):
    _freeze_readiness_refresh(monkeypatch)
    _set_ready_basics(monkeypatch, tmp_path)

    payload = main._prediction_readiness_payload(include_disk_snapshot=False)

    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert payload["warnings"] == []
    assert payload["dataset"]["rows"] == 1
    assert payload["model_bundle"]["loaded_models"] == ["away", "home", "win"]
    assert "disk" not in payload


def test_prediction_readiness_payload_detects_stale_process_hash_mismatch(monkeypatch, tmp_path):
    _freeze_readiness_refresh(monkeypatch)
    models_dir = _set_ready_basics(monkeypatch, tmp_path)

    dataset_path = tmp_path / "game_features_test_clean.csv"
    dataset_path.write_text("season,week\n2026,1\n", encoding="utf-8")
    disk_hash = file_sha256(dataset_path)
    (models_dir / "metadata.json").write_text(
        json.dumps({"dataset_hash": disk_hash}),
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "resolve_latest_dataset", lambda data_dir, explicit_path=None: dataset_path)
    monkeypatch.setattr(main, "load_latest_dataset_manifest", lambda data_dir: {"dataset_hash": disk_hash})
    monkeypatch.setattr(main.state, "dataset_hash", "stale-loaded-hash")
    monkeypatch.setattr(main.state, "models_metadata", {"dataset_hash": "stale-model-hash"})
    monkeypatch.setattr(
        main.state,
        "production_blockers",
        ["active dataset hash does not match model bundle training dataset hash"],
    )

    payload = main._prediction_readiness_payload()

    assert payload["ready"] is False
    assert payload["disk"]["disk_hashes_match"] is True
    assert payload["process_state_matches_disk"] is False
    assert payload["blockers"][0].startswith("Dataset/model hash mismatch:")
    assert payload["next_actions"][0] == (
        "Runtime process may be stale: disk dataset and model metadata now match. Restart Uvicorn."
    )


def test_prediction_readiness_payload_groups_feature_contract_blockers(monkeypatch, tmp_path):
    _freeze_readiness_refresh(monkeypatch)
    _set_ready_basics(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main.state,
        "runtime_contract_validation",
        {
            "ok": False,
            "blockers": ["dataset win feature contract missing 2 expected column(s)"],
            "warnings": ["win calibration metadata missing from model bundle"],
            "bundle": {"strict": True, "dataset_hash_match": True, "calibration_metadata_present": False},
            "dataset_features": {
                "win": {
                    "ok": False,
                    "expected_count": 3,
                    "observed_count": 1,
                    "missing_columns": ["spread_line", "total_line"],
                    "null_counts": {},
                    "blockers": ["win feature contract missing 2 expected column(s)"],
                    "warnings": [],
                }
            },
        },
    )
    monkeypatch.setattr(
        main.state,
        "production_blockers",
        ["dataset win feature contract missing 2 expected column(s)"],
    )
    monkeypatch.setattr(main.state, "production_warnings", ["win calibration metadata missing from model bundle"])

    payload = main._prediction_readiness_payload(include_disk_snapshot=False)

    assert payload["ready"] is False
    assert payload["blockers"] == [
        "Feature contract mismatch for win: active dataset is missing 2 expected column(s): spread_line, total_line."
    ]
    assert payload["contract"]["dataset_features"]["win"]["missing_columns"] == ["spread_line", "total_line"]
    assert "win calibration metadata missing from model bundle" in payload["warnings"]
    assert all("calibration" not in blocker.lower() for blocker in payload["blockers"])


def test_prediction_readiness_warning_only_stays_ready(monkeypatch, tmp_path):
    _freeze_readiness_refresh(monkeypatch)
    _set_ready_basics(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main.state,
        "production_warnings",
        [
            "dataset win feature frame has nulls in 78 expected column(s); strategy=dataset_numeric_median",
            "win calibration metadata missing from model bundle",
        ],
    )

    payload = main._prediction_readiness_payload(include_disk_snapshot=False)

    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert len(payload["warnings"]) == 2


def test_default_score_regressor_uses_neural_ensemble():
    rows = 12
    X = pd.DataFrame(
        {
            "season": [2024] * 6 + [2025] * 6,
            "week": list(range(1, 7)) * 2,
            "home_rest": np.linspace(5, 9, rows),
            "away_rest": np.linspace(4, 8, rows),
            "home_team": ["KC", "BUF", "DAL", "PHI"] * 3,
        }
    )
    y = np.linspace(17, 31, rows)
    groups = np.asarray([f"{season}-{week}" for season, week in zip(X["season"], X["week"])])

    model, info = train_models._fit_score_regressor(
        X,
        y,
        numeric_cols=["season", "week", "home_rest", "away_rest"],
        categorical_cols=["home_team"],
        group_labels=groups,
        random_seed=7,
        hp_n_iter=1,
        cv_splits=2,
        embargo_groups=1,
        n_jobs=1,
        fast_dev=True,
        score_model="ensemble",
        nn_weight=0.4,
    )

    assert isinstance(model, VotingRegressor)
    assert [name for name, _ in model.estimators] == ["hgb", "mlp"]
    assert info["algorithm"] == "voting_hgb_mlp"
    assert info["weights"] == {"hgb": 0.6, "mlp": 0.4}


def test_training_readiness_report_surfaces_blockers_and_warnings():
    df = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_BUF_KC",
                "home_team": "KC",
                "away_team": "BUF",
                "home_points_for": 24,
                "away_points_for": 21,
                "home_win": 1,
                "pre_game_label_signal": 0.9,
            },
            {
                "season": 2025,
                "week": 2,
                "game_id": "2025_02_DAL_PHI",
                "home_team": "PHI",
                "away_team": "DAL",
                "home_points_for": pd.NA,
                "away_points_for": pd.NA,
                "home_win": pd.NA,
                "pre_game_label_signal": pd.NA,
            },
        ]
    )

    report = builddataset._training_readiness_report(df)

    assert report["training_ready"] is False
    assert "too few completed rows for production training: 1 < 80" in report["blockers"]
    assert "pre_game_label_signal" in report["features"]["hard_leak_candidates"]
    assert report["rows"]["completed"] == 1
    assert report["rows"]["future"] == 1


def test_training_metrics_plot_is_generated_from_report(tmp_path):
    report = {
        "generated_at": "2026-06-13T00:00:00+00:00",
        "dataset_hash": "abcdef123456",
        "dataset_path": "backend/data/datasets/game_features_test_clean.csv",
        "rows": {"total": 1000, "train": 800, "holdout": 200, "embargo_excluded": 0},
        "features": {
            "win": {"count": 40, "numeric": ["season"], "categorical": ["home_team"]},
            "score": {"count": 41, "numeric": ["season", "nn_home_win_proba"], "categorical": ["home_team"]},
            "generated": ["nn_home_win_proba"],
        },
        "metrics": {
            "regression": {
                "home": {"mae": 7.8, "rmse": 10.2, "r2": 0.18},
                "away": {"mae": 8.1, "rmse": 10.7, "r2": 0.14},
                "combined_mae": 7.95,
            },
            "classification": {"accuracy": 0.61, "brier": 0.22, "roc_auc": 0.67, "log_loss": 0.63},
            "calibration": {"expected_calibration_error": 0.06},
            "score_win_agreement": {"side_conflict_rate": 0.12},
        },
        "baselines": {
            "score_train_mean": {"combined_mae": 8.9},
            "win_train_rate": {"brier": 0.25},
            "win_market_or_train_rate": {"brier": 0.23},
        },
        "train_info": {
            "home": {"algorithm": "voting_hgb_mlp"},
            "away": {"algorithm": "voting_hgb_mlp"},
            "win_base": {"algorithm": "mlp"},
            "fast_dev": False,
            "cv_splits": 5,
            "embargo_groups": 1,
        },
        "gate": {"enabled": True, "passed": True, "failures": []},
    }

    train_models._plot_training_metrics(report, tmp_path)

    plot_path = tmp_path / "training_metrics_plot.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
