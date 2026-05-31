import pandas as pd

from backend.services.pipeline_status import build_pipeline_status


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
