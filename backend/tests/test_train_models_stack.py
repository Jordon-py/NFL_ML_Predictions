import numpy as np
import pandas as pd

from backend import train_models as tm


def test_prior_home_win_probabilities_use_only_prior_labels():
    y = np.array([1, 0, 1, 1], dtype=int)

    probs = tm._prior_home_win_probabilities(y)

    assert probs[0] == 0.5
    assert np.isclose(probs[1], 1.0)
    assert np.isclose(probs[2], 0.5)
    assert np.isclose(probs[3], 2.0 / 3.0)


def test_augment_score_features_appends_nn_probability_column():
    X = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 2],
            "home_team": ["BUF", "KC"],
        }
    )

    augmented = tm._augment_score_features(X, np.array([0.61, 0.42]))

    assert tm.WIN_PROBA_FEATURE in augmented.columns
    assert np.allclose(augmented[tm.WIN_PROBA_FEATURE].to_numpy(), [0.61, 0.42])
    assert list(augmented.columns[:-1]) == list(X.columns)
