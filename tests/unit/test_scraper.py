from scraper.gutenberg_scraper import clean_text

def test_clean_text():
    raw_text = "Some boilerplate *** START OF THIS PROJECT GUTENBERG EBOOK *** Actual philosophical text here *** END OF THIS PROJECT GUTENBERG EBOOK *** More boilerplate"
    cleaned = clean_text(raw_text)
    assert "Actual philosophical text here" in cleaned
    assert "START OF" not in cleaned
    assert "END OF" not in cleaned
