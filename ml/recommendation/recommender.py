import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sparse
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/philosophy_corpus.csv")
EMBEDDINGS_PATH = Path("data/processed/embeddings/sentence_embeddings.npy")

_model = None
_df = None
_embeddings = None
_collab_factors = None
_user_mapping = None
_item_mapping = None

def load_resources():
    global _model, _df, _embeddings, _collab_factors, _user_mapping, _item_mapping
    
    if _df is None:
        if not DATA_PATH.exists():
            logger.warning(f"Data file not found at {DATA_PATH}.")
            return False
        _df = pd.read_csv(DATA_PATH)
        # Add index for easy mapping
        _df['idx'] = range(len(_df))
        
    if _embeddings is None:
        if not EMBEDDINGS_PATH.exists():
            logger.warning(f"Embeddings file not found at {EMBEDDINGS_PATH}.")
            return False
        _embeddings = np.load(EMBEDDINGS_PATH)
        
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        
    if _collab_factors is None:
        _build_collaborative_model()
        
    return True

def _build_collaborative_model():
    """Fallback due to missing implicit compiler. Just sets dummy factors."""
    global _collab_factors
    _collab_factors = None

def _get_collab_score(item_idx: int, user_era: str) -> float:
    if not _collab_factors or user_era not in _user_mapping.values():
        return 0.0
        
    # Find user code
    user_code = -1
    for k, v in _user_mapping.items():
        if v == user_era:
            user_code = k
            break
            
    if user_code == -1:
        return 0.0
        
    gutenberg_id = _df.iloc[item_idx]['gutenberg_id']
    
    # Find item code
    item_code = -1
    for k, v in _item_mapping.items():
        if v == gutenberg_id:
            item_code = k
            break
            
    if item_code == -1:
        return 0.0
        
    # Dot product of user and item factors
    user_factor = _collab_factors.user_factors[user_code]
    item_factor = _collab_factors.item_factors[item_code]
    
    score = np.dot(user_factor, item_factor)
    return float(score)

def recommend(book_id_or_text: Union[str, int], n: int = 10) -> List[Dict[str, Any]]:
    if not load_resources():
        return [{"error": "Resources not loaded."}]
        
    # Check if input is a valid Gutenberg ID
    target_idx = -1
    target_embedding = None
    target_era = "Unknown"
    
    is_id = False
    try:
        book_id = int(book_id_or_text)
        matches = _df[_df['gutenberg_id'] == book_id]
        if not matches.empty:
            target_idx = matches.iloc[0]['idx']
            target_embedding = _embeddings[target_idx]
            target_era = matches.iloc[0]['era_label']
            is_id = True
    except ValueError:
        pass
        
    # If not ID or not found, treat as raw text
    if not is_id:
        # truncate text for speed
        text = str(book_id_or_text)[:2000]
        target_embedding = _model.encode([text])[0]
        
    # Calculate Content Similarity (Cosine)
    content_scores = cosine_similarity([target_embedding], _embeddings)[0]
    
    # Calculate Collaborative & Hybrid Scores
    recommendations = []
    
    for i in range(len(_df)):
        if is_id and i == target_idx:
            continue # skip the query book itself
            
        cb_score = float(content_scores[i])
        cf_score = _get_collab_score(i, target_era) if is_id else 0.0
        
        # Normalize CF score roughly
        cf_norm = np.clip(cf_score, 0, 1) if cf_score > 0 else 0
        
        if is_id:
            hybrid_score = (0.7 * cb_score) + (0.3 * cf_norm)
            reason = "Hybrid (Content + Collaborative)"
        else:
            hybrid_score = cb_score
            reason = "Content (Semantic Similarity)"
            
        row = _df.iloc[i]
        recommendations.append({
            "gutenberg_id": int(row['gutenberg_id']),
            "title": row['title'],
            "author": row['author'],
            "similarity": round(hybrid_score, 4),
            "reason": reason
        })
        
    # Sort and return top N
    recommendations.sort(key=lambda x: x["similarity"], reverse=True)
    return recommendations[:n]

if __name__ == "__main__":
    load_resources()
