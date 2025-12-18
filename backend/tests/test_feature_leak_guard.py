# ==========================================
# File: backend/tests/test_feature_leak_guard.py
# Role: Backend test module.
# Input Data: Test fixtures and sample payloads.
# Output Data: Pytest assertions and results.
# Dependencies: pandas, pathlib, backend
# Notes: Run via pytest.
# ==========================================

import pandas as pd
from pathlib import Path

from backend.enhanced_pipeline import build_dataset, is_leak_feature


def test_is_leak_feature_rules():
    # Allowed engineered prefixes
    assert not is_leak_feature("home_prior_win_pct_3")
    assert not is_leak_feature("away_prior_pf_avg_5")
    assert not is_leak_feature("diff_off_turnovers_avg_3")
    assert not is_leak_feature("home_minus_away_win_pct_5")
    assert not is_leak_feature("trend_home_minus_away_pf_avg_3_w3")

    # Underscore / explicit forbidden
    assert is_leak_feature("_home_win_derived")
    assert is_leak_feature("_dom_delta_emp_home_win")
    assert is_leak_feature("_dom_delta")

    # Season home win rate can be leakage unless time-sliced; conservatively flagged
    assert is_leak_feature("season_home_win_rate")


def test_build_dataset_filters_leakage(tmp_path: Path):
    # Minimal synthetic dataset with a mix of safe and leak features
    df = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 2, 2],
            # Outcome-generating columns (build_dataset will derive target from these)
            "home_points_for": [21, 14, 28, 10],
            "away_points_for": [17, 20, 14, 13],
            # Safe engineered pregame features
            "home_prior_win_pct_3": [0.33, 0.66, 0.66, 0.33],
            "away_prior_pf_avg_5": [21.0, 24.0, 20.0, 27.0],
            # Intentionally omit any 'diff_' columns here so build_dataset takes the numeric path
            "home_minus_away_win_pct_5": [0.0, 0.2, -0.2, 0.0],
            "trend_home_minus_away_pf_avg_3_w3": [0.0, 0.5, -0.3, 0.1],
            # Leakage candidates
            "_home_win_derived": [1, 0, 1, 0],
            "_dom_delta_emp_home_win": [0.7, 0.42, 0.81, 0.38],
            "season_home_win_rate": [0.56, 0.56, 0.56, 0.56],
        }
    )

    csv_path = tmp_path / "mini.csv"
    df.to_csv(csv_path, index=False)

    X, y, groups, df_raw = build_dataset(str(csv_path))

    # Safe features should be present
    for col in [
        "home_prior_win_pct_3",
        "away_prior_pf_avg_5",
        "home_minus_away_win_pct_5",
        "trend_home_minus_away_pf_avg_3_w3",
    ]:
        assert col in X.columns, f"Expected safe feature {col} to remain in training features"

    # Leakage features should not be present
    for col in [
        "_home_win_derived",
        "_dom_delta_emp_home_win",
        "season_home_win_rate",
        "home_points_for",
        "away_points_for",
    ]:
        assert col not in X.columns, f"Leakage feature {col} should be excluded from training features"

    # Basic shapes and target checks
    assert len(X) == len(y) == len(groups) == 4
    assert set(y.unique()) <= {0, 1}
