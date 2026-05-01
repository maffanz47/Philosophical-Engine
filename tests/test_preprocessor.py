from src.preprocessor import normalize_text, preprocess_text


def test_normalize_text_collapses_whitespace_and_lowercases() -> None:
    result = normalize_text("  The   SELF   and   the  World  ")
    assert result == "the self and the world"


def test_preprocess_text_returns_non_empty_lemmas() -> None:
    tokens = preprocess_text("Thinkers were thinking about universal truths.")
    assert tokens
    assert "thinker" in tokens
