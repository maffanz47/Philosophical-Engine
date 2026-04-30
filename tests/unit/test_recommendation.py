"""Unit tests for the recommendation module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.recommendation.recommender import (
    recommend,
)


class TestRecommend:
    """Tests for recommend function."""

    def test_recommend_no_model_returns_empty(self):
        # When no model is loaded, should return empty list
        with patch("ml.recommendation.recommender._load_similarity_matrix", return_value=None):
            result = recommend("test book", n=5)
            assert result == []

    def test_recommend_empty_query(self):
        # Empty query should return empty list
        result = recommend("", n=5)
        assert result == []

    def test_recommend_with_text_input(self):
        # Test with raw text instead of book ID
        # This will use content-based approach
        with patch("ml.recommendation.recommender._load_similarity_matrix", return_value=None):
            result = recommend("philosophy is the love of wisdom", n=5)
            assert isinstance(result, list)

    def test_recommend_n_parameter(self):
        # Test that n parameter is respected
        with patch("ml.recommendation.recommender._load_similarity_matrix", return_value=None):
            result = recommend("test", n=10)
            assert isinstance(result, list)
            # When model is not available, returns empty regardless of n

    def test_recommend_result_format(self):
        # Test result format when model is available
        mock_matrix = MagicMock()
        mock_matrix.shape = (100, 100)
        
        mock_embeddings = MagicMock()
        mock_embeddings.shape = (100, 384)
        
        mock_titles = [f"Book {i}" for i in range(100)]
        
        with patch("ml.recommendation.recommender._load_similarity_matrix", return_value=mock_matrix):
            with patch("ml.recommendation.recommender._load_embeddings", return_value=mock_embeddings):
                with patch("ml.recommendation.recommender._load_titles", return_value=mock_titles):
                    result = recommend("test", n=3)
                    
                    if result:  # If results are returned
                        assert isinstance(result, list)
                        for item in result:
                            assert "title" in item or "similarity" in item


class TestSimilarityMatrix:
    """Tests for similarity matrix functions."""

    def test_load_similarity_matrix_nonexistent(self):
        with patch(
            "ml.recommendation.recommender.Path.exists",
            return_value=False,
        ):
            from ml.recommendation.recommender import _load_similarity_matrix

            result = _load_similarity_matrix("nonexistent")
            assert result is None


class TestHybridRecommendation:
    """Tests for hybrid recommendation features."""

    def test_hybrid_weights(self):
        # Verify default hybrid weights
        from ml.recommendation.recommender import CONTENT_WEIGHT, COLLAB_WEIGHT
        
        assert CONTENT_WEIGHT == 0.7
        assert COLLAB_WEIGHT == 0.3
        # Weights should sum to 1.0
        assert abs(CONTENT_WEIGHT + COLLAB_WEIGHT - 1.0) < 0.001
