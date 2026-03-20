import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from backend import train_models as tm


class _CloneableEstimator(BaseEstimator):
    def fit(self, X, y):
        return self


def _build_training_frame(n_rows: int = 84) -> pd.DataFrame:
    home_teams = ["BUF", "KC", "MIA", "DAL"]
    away_teams = ["PHI", "SF", "DET", "BAL"]
    rows = []
    for i in range(n_rows):
        season = 2024 + (i // 18)
        week = (i % 18) + 1
        home_points = 17 + (i % 11) + (i % 3)
        away_points = 14 + ((i * 3) % 9)
        home_win = int(home_points > away_points)
        moneyline = np.clip(0.34 + (0.28 * home_win) + (((i % 5) - 2) * 0.03), 0.05, 0.95)
        rows.append(
            {
                "season": season,
                "week": week,
                "home_team": home_teams[i % len(home_teams)],
                "away_team": away_teams[(i + 1) % len(away_teams)],
                "home_points_for": float(home_points),
                "away_points_for": float(away_points),
                "home_win": home_win,
                "home_moneyline_prob": float(moneyline),
                "elo_diff": float((i % 9) - 4),
                "home_rest_days": float(5 + (i % 4)),
                "injury_gap": float(((i * 2) % 7) - 3),
            }
        )
    return pd.DataFrame(rows)


def test_prior_home_win_probabilities_use_only_prior_labels():
    y = np.array([1, 0, 1, 1], dtype=int)

    probs = tm._prior_home_win_probabilities(y)

    assert probs[0] == 0.5
    assert np.isclose(probs[1], 1.0)
    assert np.isclose(probs[2], 0.5)
    assert np.isclose(probs[3], 2.0 / 3.0)


def test_fallback_home_win_probabilities_use_moneyline_then_neutral():
    X = pd.DataFrame(
        {
            "home_moneyline_prob": [0.72, np.nan, -0.5, 2.0],
            "feature_a": [1, 2, 3, 4],
        }
    )

    probs = tm._fallback_home_win_probabilities(X)

    assert np.isclose(probs[0], 0.72)
    assert np.isclose(probs[1], 0.5)
    assert np.isclose(probs[2], 1e-6)
    assert np.isclose(probs[3], 1 - 1e-6)


def test_generate_stacked_train_probabilities_is_chronology_safe(monkeypatch):
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    X_proc = np.arange(len(y), dtype=float).reshape(-1, 1)
    fallback = np.full(len(y), 0.41, dtype=float)
    fold_calls = []

    class _SpyCalibratedModel:
        def __init__(self, train_rows):
            self.train_rows = tuple(int(x) for x in train_rows)

        def predict_proba(self, X):
            val_rows = tuple(int(x) for x in np.asarray(X)[:, 0])
            fold_calls.append((self.train_rows, val_rows))
            probs = np.asarray([0.17 + (0.01 * row) for row in val_rows], dtype=float)
            return np.column_stack([1.0 - probs, probs])

    def _fake_calibrate_classifier(base_clf, X_train, y_train):
        return _SpyCalibratedModel(np.asarray(X_train)[:, 0]), {"mode": "spy"}

    monkeypatch.setattr(tm, "_calibrate_classifier", _fake_calibrate_classifier)

    probs, info = tm._generate_stacked_train_probabilities(
        X_proc,
        y,
        tuned_estimator=_CloneableEstimator(),
        cv_splits=3,
        fallback_probabilities=fallback,
    )

    assert fold_calls
    predicted_rows = set()
    for train_rows, val_rows in fold_calls:
        assert max(train_rows) < min(val_rows)
        predicted_rows.update(val_rows)
        for row in val_rows:
            expected = 0.17 + (0.01 * row)
            assert np.isclose(probs[row], expected)

    uncovered_rows = sorted(set(range(len(y))) - predicted_rows)
    assert uncovered_rows
    assert np.allclose(probs[uncovered_rows], fallback[uncovered_rows])
    assert not np.allclose(probs[sorted(predicted_rows)], y[sorted(predicted_rows)].astype(float))
    assert info["mode"] == "time_series_oof"
    assert info["fallback_mode"] == "home_moneyline_prob_or_neutral"


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


def test_training_main_writes_two_stage_metadata_and_report(tmp_path, monkeypatch):
    dataset_path = tmp_path / "game_features_test.csv"
    out_dir = tmp_path / "models"
    _build_training_frame().to_csv(dataset_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_models.py",
            "--data",
            str(dataset_path),
            "--out",
            str(out_dir),
            "--fast-dev",
            "--disable-gate",
            "--no-promote",
            "--n-jobs",
            "1",
        ],
    )

    rc = tm.main()

    assert rc == 0
    stage_dirs = sorted((out_dir / "staging").iterdir())
    assert stage_dirs
    stage_dir = stage_dirs[-1]

    metadata = json.loads((stage_dir / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((stage_dir / "training_report.json").read_text(encoding="utf-8"))

    assert metadata["serving_mode"] == "pipeline_primary"
    assert metadata["generated_features"][tm.WIN_PROBA_FEATURE]["source"] == "winner_model_predict_proba"
    assert metadata["raw_feature_columns"]["win"]["numeric"]
    assert tm.WIN_PROBA_FEATURE not in metadata["raw_feature_columns"]["win"]["numeric"]
    assert tm.WIN_PROBA_FEATURE in metadata["raw_feature_columns"]["score"]["numeric"]
    assert metadata["artifacts"]["home_model"] == "home_pipe.joblib"
    assert metadata["artifacts"]["away_model"] == "away_pipe.joblib"
    assert metadata["artifacts"]["win_model"] == "win_pipe.joblib"

    assert report["features"]["generated"] == [tm.WIN_PROBA_FEATURE]
    assert report["train_info"]["win_base"]["algorithm"] == "mlp"
    assert report["train_info"]["win_calibration"]["mode"] in {"prefit_tail", "cv", "uncalibrated"}
    assert report["train_info"]["score_stack"]["fallback_mode"] == "home_moneyline_prob_or_neutral"
