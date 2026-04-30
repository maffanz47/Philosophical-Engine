"""Unit tests for the Gutenberg scraper."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from scraper.gutenberg_scraper import (
    SUBJECT_TO_SCHOOL_KEYWORDS,
    _clean_text,
    _derive_era_label,
    _parse_int_from_text,
    _parse_year_from_release_date,
    _round_to_nearest_decade,
    _school_label_from_subjects,
    _top_noun_concepts,
)


class TestCleanText:
    """Tests for _clean_text function."""

    def test_clean_text_normalizes_whitespace(self):
        text = "Hello  \n\n\n  World"
        result = _clean_text(text)
        assert "\n\n\n" not in result
        assert "  " not in result

    def test_clean_text_strips(self):
        text = "  Hello World  \n"
        result = _clean_text(text)
        assert result == "Hello World"


class TestParseInt:
    """Tests for integer parsing functions."""

    def test_parse_int_from_text_with_commas(self):
        text = "1,234 downloads"
        assert _parse_int_from_text(text) == 1234

    def test_parse_int_from_text_simple(self):
        text = "1234"
        assert _parse_int_from_text(text) == 1234

    def test_parse_int_from_text_no_match(self):
        text = "no numbers here"
        assert _parse_int_from_text(text) is None


class TestParseYear:
    """Tests for year parsing functions."""

    def test_parse_year_from_release_date_full(self):
        text = "Release Date: March 12, 2019"
        assert _parse_year_from_release_date(text) == 2019

    def test_parse_year_from_release_date_fallback(self):
        text = "Published in 1899"
        assert _parse_year_from_release_date(text) == 1899

    def test_parse_year_from_release_date_invalid(self):
        text = "No date here"
        assert _parse_year_from_release_date(text) is None


class TestEraLabel:
    """Tests for era label derivation."""

    @pytest.mark.parametrize(
        "year,expected",
        [
            (None, "Contemporary"),
            (100, "Ancient"),
            (800, "Medieval"),
            (1500, "Renaissance"),
            (1700, "Enlightenment"),
            (1900, "Modern"),
            (2000, "Contemporary"),
        ],
    )
    def test_derive_era_label(self, year: int | None, expected: str):
        assert _derive_era_label(year) == expected


class TestDecade:
    """Tests for decade rounding."""

    @pytest.mark.parametrize(
        "year,expected",
        [
            (1904, 1900),
            (1905, 1910),
            (1950, 1950),
            (None, None),
        ],
    )
    def test_round_to_nearest_decade(self, year: int | None, expected: int | None):
        assert _round_to_nearest_decade(year) == expected


class TestSchoolLabel:
    """Tests for school label from subjects."""

    def test_school_label_empiricism(self):
        subjects = ["Empiricism", "Philosophy"]
        assert _school_label_from_subjects(subjects) == "Empiricism"

    def test_school_label_rationalism(self):
        subjects = ["Rationalism", "Metaphysics"]
        assert _school_label_from_subjects(subjects) == "Rationalism"

    def test_school_label_existentialism(self):
        subjects = ["Existentialism"]
        assert _school_label_from_subjects(subjects) == "Existentialism"

    def test_school_label_stoicism(self):
        subjects = ["Stoicism"]
        assert _school_label_from_subjects(subjects) == "Stoicism"

    def test_school_label_idealism(self):
        subjects = ["Idealism"]
        assert _school_label_from_subjects(subjects) == "Idealism"

    def test_school_label_pragmatism(self):
        subjects = ["Pragmatism"]
        assert _school_label_from_subjects(subjects) == "Pragmatism"

    def test_school_label_other(self):
        subjects = ["Mathematics", "Science"]
        assert _school_label_from_subjects(subjects) == "Other"


class TestNounConceptExtraction:
    """Tests for noun concept extraction."""

    def test_top_noun_concepts_basic(self):
        # Mock spacy nlp
        mock_doc = MagicMock()
        mock_token1 = MagicMock()
        mock_token1.is_stop = False
        mock_token1.is_alpha = True
        mock_token1.lemma_ = "philosophy"
        mock_token1.pos_ = "NOUN"

        mock_token2 = MagicMock()
        mock_token2.is_stop = False
        mock_token2.is_alpha = True
        mock_token2.lemma_ = "reason"
        mock_token2.pos_ = "NOUN"

        mock_token3 = MagicMock()
        mock_token3.is_stop = True  # Stop word
        mock_token3.is_alpha = True
        mock_token3.lemma_ = "the"
        mock_token3.pos_ = "DET"

        mock_doc.__iter__ = lambda: iter([mock_token1, mock_token2, mock_token3])
        mock_nlp = MagicMock(return_value=mock_doc)

        result = _top_noun_concepts(mock_nlp, "test text")
        # Should return top concepts sorted by frequency
        assert "philosophy" in result or "reason" in result

    def test_top_noun_concepts_filters_stopwords(self):
        # Ensure stopwords are filtered
        mock_doc = MagicMock()
        mock_token = MagicMock()
        mock_token.is_stop = True
        mock_token.is_alpha = True
        mock_token.lemma_ = "the"
        mock_token.pos_ = "DET"

        mock_doc.__iter__ = lambda: iter([mock_token])
        mock_nlp = MagicMock(return_value=mock_doc)

        result = _top_noun_concepts(mock_nlp, "test text")
        assert "the" not in result


class TestEndToEndParsing:
    """Integration tests for parsing functions."""

    def test_extract_fields_from_search_results(self):
        # Sample HTML for search results
        html = """
        <html>
            <body>
                <a href="/ebooks/1234/">Book 1</a>
                <a href="/ebooks/5678/">Book 2</a>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")
        # Test parsing - just verify links are found
        from scraper.gutenberg_scraper import _parse_search_results

        ids = _parse_search_results(soup)
        assert 1234 in ids
        assert 5678 in ids


class TestSerialization:
    """Tests for JSON serialization/deserialization."""

    def test_raw_json_roundtrip(self, tmp_path: Path, sample_raw_json: Path):
        # Load the sample JSON
        with open(sample_raw_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify expected fields
        assert "gutenberg_id" in data
        assert data["gutenberg_id"] == 12345
        assert "title" in data
        assert "school_label" in data

        # Write back and read again
        new_path = tmp_path / "test_roundtrip.json"
        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        with open(new_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["gutenberg_id"] == data["gutenberg_id"]
        assert loaded["title"] == data["title"]
