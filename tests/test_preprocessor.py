from src.preprocessor import preprocess_text, chunk_text

def test_preprocess_text():
    raw_text = "This is a SAMPLE sentence! With some punctuation."
    cleaned = preprocess_text(raw_text)
    
    assert "sample" in cleaned
    assert "sentence" in cleaned
    assert "!" not in cleaned
    assert "punctuation" in cleaned
    # SpaCy stopwords should be removed
    assert "this" not in cleaned
    assert "is" not in cleaned

def test_chunk_text():
    # Create a string with exactly 250 words
    words = [f"word{i}" for i in range(250)]
    text = " ".join(words)
    
    # Chunk with size 100
    chunks = chunk_text(text, chunk_size=100)
    
    # We expect 3 chunks: sizes 100, 100, 50. 
    # Since 50 >= 50, the last chunk is kept.
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 100
    assert len(chunks[1].split()) == 100
    assert len(chunks[2].split()) == 50

def test_chunk_text_skips_small_chunks():
    # Create a string with 130 words
    words = [f"word{i}" for i in range(130)]
    text = " ".join(words)
    
    # Chunk with size 100
    chunks = chunk_text(text, chunk_size=100)
    
    # We expect 1 chunk of size 100. 
    # The remaining 30 words are dropped because 30 < 50.
    assert len(chunks) == 1
    assert len(chunks[0].split()) == 100
