import pandas as pd
import numpy as np
from backend import enhanced_pipeline as ep


def test_build_dataset_excludes_outcome_columns(tmp_path):
    # Construct a tiny dataset with obvious leakage columns present
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 2, 2],
            "home_team": ["SEA", "SF", "SEA", "SF"],
            "away_team": ["SF", "SEA", "SF", "SEA"],
            # Outcome columns (should NOT be used as features)
            "home_points_for": [24, 10, 17, 31],
            "away_points_for": [17, 13, 24, 7],
            "point_diff": [7, -3, -7, 24],
            # Pre-game style engineered columns (allowed)
            "home_prior_pf_avg_3": [21.0, 22.0, 21.5, 23.0],
            "away_prior_pf_avg_3": [20.0, 19.0, 20.5, 18.0],
        }
    )
    p = tmp_path / "toy.csv"
    df.to_csv(p, index=False)

    X, y, groups, df_raw = ep.build_dataset(str(p))

    used = set(X.columns)
    # Assert that outcome columns are NOT used as features
    assert "home_points_for" not in used
    assert "away_points_for" not in used
    assert "point_diff" not in used

    # But pre-game style engineered priors are present
    assert "home_prior_pf_avg_3" in used
    assert "away_prior_pf_avg_3" in used
