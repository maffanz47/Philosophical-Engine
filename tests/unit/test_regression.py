"""Unit tests for the regression module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.regression.influence_predictor import (
    predict_influence,
)


class TestPredictInfluence:
    """Tests for predict_influence function."""

    def test_predict_influence_no_model_returns_zero(self):
        # When no model is loaded, should return 0.0
        with patch(
            "ml.regression.influence_predictor._latest_model_path",
            return_value=None,
        ):
            result = predict_influence({"text": "test text"})
            assert result == 0.0

    def test_predict_influence_with_invalid_model(self):
        with patch(
            "ml.regression.influence_predictor._latest_model_path",
            return_value="nonexistent.pkl",
        ):
            result = predict_influence({"text": "test text"})
            assert result == 0.0

    @patch("ml.regression.influence_predictor._latest_model_path")
    @patch("ml.regression.influence_predictor.joblib.load")
    def test_predict_influence_with_valid_model(self, mock_load, mock_path):
        # Mock a trained model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.75]

        mock_artifact = {"model": mock_model}

        mock_path.return_value = "models/regression/influence_predictor_v123.pkl"
        mock_load.return_value = mock_artifact

        result = predict_influence({"text": "philosophy is the love of wisdom"})

        # Should return clamped value
        assert 0.0 <= result <= 1.0

    @patch("ml.regression.influence_predictor._latest_model_path")
    @patch("ml.regression.influence_predictor.joblib.load")
    def test_predict_influence_clamping(self, mock_load, mock_path):
        # Test clamping behavior
        # Value > 1 should be clamped to 1
        mock_model_high = MagicMock()
        mock_model_high.predict.return_value = [1.5]
        mock_path.return_value = "models/regression/influence_predictor_v123.pkl"
        mock_load.return_value = {"model": mock_model_high}

        result = predict_influence({"text": "test"})
        assert result <= 1.0

        # Value < 0 should be clamped to 0
        mock_model_low = MagicMock()
        mock_model_low.predict.return_value = [-0.5]
        mock_path.return_value = "models/regression/influence_predictor_v123.pkl"
        mock_load.return_value = {"model": mock_model_low}

        result = predict_influence({"text": "test"})
        assert result >= 0.0

    @patch("ml.regression.influence_predictor._latest_model_path")
    @patch("ml.regression.influence_predictor.joblib.load")
    def test_predict_influence_nan_handling(self, mock_load, mock_path):
        # Test NaN handling
        mock_model = MagicMock()
        mock_model.predict.return_value = [float("nan")]
        mock_path.return_value = "models/regression/influence_predictor_v123.pkl"
        mock_load.return_value = {"model": mock_model}

        result = predict_influence({"text": "test"})
        # Should return 0.0 for NaN
        assert result == 0.0


class TestModelPath:
    """Tests for model path discovery."""

    def test_latest_model_path_nonexistent_dir(self):
        with patch(
            "ml.regression.influence_predictor.Path.exists",
            return_value=False,
        ):
            from ml.regression.influence_predictor import _latest_model_path

            result = _latest_model_path("nonexistent")
            assert result is None
