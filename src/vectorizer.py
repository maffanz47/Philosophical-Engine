import numpy as np

def load_tfidf():
    """Placeholder to prevent main.py from crashing.
    The application only utilizes Word2Vec embeddings.
    """
    return None

def text_to_embedding(text: str, w2v_model) -> np.ndarray:
    """Convert text to a single embedding vector by averaging word vectors.
    
    Args:
        text (str): Preprocessed text string.
        w2v_model (Word2Vec): Loaded Word2Vec model from Gensim.
        
    Returns:
        np.ndarray: A 200-dimensional vector.
    """
    words = text.split()
    vectors = []
    
    for word in words:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
            
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        # Fallback to zero vector if no words are recognized
        # PhiloClassifier expects input_dim=200
        return np.zeros(200)
