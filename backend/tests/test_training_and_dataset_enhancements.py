import numpy as np
import pandas as pd

from backend import build_csv_datasets_v3 as builder
from backend.train_models import _compute_balanced_sample_weights


def test_recent_margin_trend_features_are_added_for_model_inputs():
    frame = pd.DataFrame(
        [
            {
                "home_prior_pf_avg_3": 27.0,
                "home_prior_pa_avg_3": 21.0,
                "away_prior_pf_avg_3": 24.0,
                "away_prior_pa_avg_3": 19.0,
            },
            {
                "home_prior_pf_avg_3": 25.0,
                "home_prior_pa_avg_3": 20.0,
                "away_prior_pf_avg_3": 22.0,
                "away_prior_pa_avg_3": 18.0,
            },
        ]
    )

    enhanced = builder._add_recent_margin_trend_features(frame)

    assert "home_recent_margin_trend_3" in enhanced.columns
    assert "away_recent_margin_trend_3" in enhanced.columns
    assert "recent_margin_edge_3" in enhanced.columns
    assert enhanced.loc[0, "home_recent_margin_trend_3"] == 6.0
    assert enhanced.loc[0, "away_recent_margin_trend_3"] == 5.0
    assert enhanced.loc[0, "recent_margin_edge_3"] == 1.0


def test_balanced_sample_weights_prioritize_underrepresented_class():
    weights = _compute_balanced_sample_weights(np.array([1, 1, 1, 0]))

    assert weights[0] < weights[-1]
    assert np.isclose(weights.mean(), 1.0)
