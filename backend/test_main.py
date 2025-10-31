"""
Tests for backend/main.py functions.

Run with: pytest backend/test_main.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from backend.main import _validate_dataset_schema, _sanity_predict


class TestValidateDatasetSchema:
    """Tests for _validate_dataset_schema function."""

    def test_valid_schema(self):
        """Test that no exception is raised when all required features are present."""
        # Mock model_objects with expected features
        model_objects = {
            "raw_feature_columns": {
                "numeric": ["home_prior_pf_avg_3", "away_prior_pf_avg_3"],
                "categorical": ["home_game_date"]
            }
        }
        # Create df with all required columns
        df = pd.DataFrame({
            "home_prior_pf_avg_3": [1.0],
            "away_prior_pf_avg_3": [2.0],
            "home_game_date": ["2025-W01"]
        })

        # Should not raise
        _validate_dataset_schema(df, model_objects)

    def test_missing_features(self):
        """Test that RuntimeError is raised when features are missing."""
        model_objects = {
            "raw_feature_columns": {
                "numeric": ["home_prior_pf_avg_3", "missing_feature"],
                "categorical": []
            }
        }
        df = pd.DataFrame({
            "home_prior_pf_avg_3": [1.0],
            # missing_feature is missing
        })

        with pytest.raises(RuntimeError, match="Dataset missing engineered features"):
            _validate_dataset_schema(df, model_objects)

    def test_empty_raw_feature_columns(self):
        """Test handling of empty raw_feature_columns."""
        model_objects = {"raw_feature_columns": {}}
        df = pd.DataFrame({"some_col": [1]})

        # Should not raise since no features expected
        _validate_dataset_schema(df, model_objects)

    def test_list_raw_feature_columns(self):
        """Test handling when raw_feature_columns is a list instead of dict."""
        model_objects = {"raw_feature_columns": ["feature1", "feature2"]}
        df = pd.DataFrame({
            "feature1": [1],
            "feature2": [2]
        })

        _validate_dataset_schema(df, model_objects)


class TestSanityPredict:
    """Tests for _sanity_predict function."""

    @patch('backend.main.log')
    def test_successful_sanity_predict(self, mock_log):
        """Test successful sanity prediction with all models present."""
        # Mock models
        preprocessor = MagicMock()
        preprocessor.transform.return_value = pd.DataFrame([[1, 2, 3]])

        home_model = MagicMock()
        home_model.predict.return_value = np.array([25.0])

        away_model = MagicMock()
        away_model.predict.return_value = np.array([22.0])

        win_model = MagicMock()
        win_model.predict_proba.return_value = np.array([[0.6, 0.4]])

        model_objects = {
            "preprocessor": preprocessor,
            "home_model": home_model,
            "away_model": away_model,
            "win_model": win_model,
            "raw_feature_columns": {
                "numeric": ["home_prior_pf_avg_3"],
                "categorical": []
            }
        }

        df = pd.DataFrame({
            "home_prior_pf_avg_3": [1.0]
        })

        # Should not raise
        _sanity_predict(model_objects, df)

        # Verify calls were made
        preprocessor.transform.assert_called_once()
        home_model.predict.assert_called_once()
        away_model.predict.assert_called_once()
        win_model.predict_proba.assert_called_once()

    @patch('backend.main.log')
    def test_missing_home_model(self, mock_log):
        """Test that RuntimeError is raised when home_model is missing."""
        model_objects = {
            "preprocessor": None,
            "home_model": None,  # Missing
            "away_model": MagicMock(),
            "win_model": None,
            "raw_feature_columns": {"numeric": [], "categorical": []}
        }
        df = pd.DataFrame()

        with pytest.raises(RuntimeError, match="home_model not present"):
            _sanity_predict(model_objects, df)

    @patch('backend.main.log')
    def test_preprocessor_transform_failure(self, mock_log):
        """Test handling of preprocessor transform failure."""
        preprocessor = MagicMock()
        preprocessor.transform.side_effect = ValueError("Transform failed")

        model_objects = {
            "preprocessor": preprocessor,
            "home_model": MagicMock(),
            "away_model": MagicMock(),
            "win_model": None,
            "raw_feature_columns": {"numeric": ["feature1"], "categorical": []}
        }
        df = pd.DataFrame({"feature1": [1.0]})

        with pytest.raises(RuntimeError, match="preprocessor.transform failed"):
            _sanity_predict(model_objects, df)

    @patch('backend.main.log')
    def test_model_predict_failure(self, mock_log):
        """Test handling of model predict failure."""
        home_model = MagicMock()
        home_model.predict.side_effect = AttributeError("No predict method")

        model_objects = {
            "preprocessor": None,
            "home_model": home_model,
            "away_model": MagicMock(),
            "win_model": None,
            "raw_feature_columns": {"numeric": [], "categorical": []}
        }
        df = pd.DataFrame()

        with pytest.raises(RuntimeError, match="home_model predict failed"):
            _sanity_predict(model_objects, df)

    @patch('backend.main.log')
    def test_empty_dataframe(self, mock_log):
        """Test handling of empty dataframe."""
        model_objects = {
            "preprocessor": None,
            "home_model": MagicMock(),
            "away_model": MagicMock(),
            "win_model": None,
            "raw_feature_columns": {"numeric": ["feature1"], "categorical": []}
        }
        df = pd.DataFrame()  # Empty

        # Should not raise, uses defaults
        _sanity_predict(model_objects, df)

    @patch('backend.main.log')
    def test_win_model_predict_proba_failure(self, mock_log):
        """Test handling of win_model predict_proba failure."""
        win_model = MagicMock()
        win_model.predict_proba.side_effect = Exception("Proba failed")

        model_objects = {
            "preprocessor": None,
            "home_model": MagicMock(),
            "away_model": MagicMock(),
            "win_model": win_model,
            "raw_feature_columns": {"numeric": [], "categorical": []}
        }
        df = pd.DataFrame()

        with pytest.raises(RuntimeError, match="win_model.predict_proba failed"):
            _sanity_predict(model_objects, df)