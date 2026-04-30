"""Unit tests for the classification module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.classification.school_classifier import (
    SCHOOLS,
    predict_school,
    train_school_classifier,
)


class TestPredictSchool:
    """Tests for predict_school function."""

    def test_predict_school_no_model_returns_default(self):
        # When no model is loaded, should return default
        with patch("ml.classification.school_classifier._latest_model_path", return_value=None):
            result = predict_school("test text")
            assert result["school"] == "Other"
            assert result["confidence"] == 0.0
            assert "Other" in result["top3"]

    def test_predict_school_with_invalid_model(self):
        with patch(
            "ml.classification.school_classifier._latest_model_path",
            return_value="nonexistent.pkl",
        ):
            result = predict_school("test text")
            assert result["school"] == "Other"

    @patch("ml.classification.school_classifier._latest_model_path")
    @patch("ml.classification.school_classifier._try_load_artifact")
    def test_predict_school_with_valid_model(self, mock_load, mock_path):
        # Mock a trained model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [
            [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1]
        ]

        mock_tfidf = MagicMock()
        mock_tfidf.transform.return_value = "tfidf_vector"

        mock_artifact = {"model": mock_model, "tfidf": mock_tfidf}

        mock_path.return_value = "models/classification/school_classifier_v123.pkl"
        mock_load.return_value = mock_artifact

        result = predict_school("philosophy is the love of wisdom")

        # Should return predictions from the model
        assert "school" in result
        assert "confidence" in result
        assert "top3" in result
        assert isinstance(result["school"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["top3"], list)
        assert len(result["top3"]) == 3


class TestTrainClassifier:
    """Tests for train_school_classifier function."""

    def test_train_school_classifier_not_implemented(self):
        # Phase 1: training not yet implemented
        df = MagicMock()
        with pytest.raises(NotImplementedError):
            train_school_classifier(df)


class TestSchoolConstants:
    """Tests for school constants."""

    def test_schools_contains_expected_values(self):
        expected = [
            "Empiricism",
            "Rationalism",
            "Existentialism",
            "Stoicism",
            "Idealism",
            "Pragmatism",
            "Other",
        ]
        assert SCHOOLS == expected

    def test_schools_all_unique(self):
        assert len(SCHOOLS) == len(set(SCHOOLS))


class TestModelLoading:
    """Tests for model discovery functions."""

    @patch("ml.classification.school_classifier.models_dir", return_value=None)
    def test_latest_model_path_empty_dir(self, mock_models_dir):
        # Test when models directory doesn't exist
        with patch(
            "ml.classification.school_classifier.Path.exists",
            return_value=False,
        ):
            from ml.classification.school_classifier import _latest_model_path

            result = _latest_model_path("nonexistent")
            assert result is None
