import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path("data/processed/philosophy_corpus.csv")
EMBEDDINGS_DIR = Path("data/processed/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

NPY_PATH = EMBEDDINGS_DIR / "sentence_embeddings.npy"
UMAP_CSV_PATH = EMBEDDINGS_DIR / "umap_2d.csv"

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        logger.warning(f"Data file not found at {DATA_PATH}.")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

def generate_and_reduce_embeddings():
    df = load_data()
    if df.empty or 'full_text' not in df.columns:
        logger.error("Insufficient data for embeddings.")
        return

    # To avoid memory issues, we use the first 1000 chars of each book or avg embedding of sentences
    # For speed and simplicity, we'll embed the first 2000 chars
    texts = df['full_text'].astype(str).str[:2000].tolist()
    
    logger.info("Generating embeddings using all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Save raw embeddings
    np.save(NPY_PATH, embeddings)
    logger.info(f"Saved raw embeddings to {NPY_PATH}")

    # 1. PCA to 50 dimensions (preprocessing)
    logger.info("Applying PCA to 50 dimensions...")
    n_components = min(50, len(embeddings))
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    # 2. UMAP 2D and 3D
    logger.info("Applying UMAP...")
    umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(embeddings_pca)
    umap_3d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42).fit_transform(embeddings_pca)
    
    # 3. t-SNE 2D
    logger.info("Applying t-SNE...")
    tsne_2d = TSNE(perplexity=min(30, len(embeddings)-1), max_iter=1000, n_components=2, random_state=42).fit_transform(embeddings_pca)

    # Save 2D UMAP coordinates with metadata
    umap_df = pd.DataFrame({
        'gutenberg_id': df['gutenberg_id'],
        'title': df['title'],
        'author': df['author'],
        'school_label': df['school_label'],
        'era_label': df['era_label'],
        'x': umap_2d[:, 0],
        'y': umap_2d[:, 1],
        'tsne_x': tsne_2d[:, 0],
        'tsne_y': tsne_2d[:, 1]
    })
    
    umap_df.to_csv(UMAP_CSV_PATH, index=False)
    logger.info(f"Saved UMAP 2D map to {UMAP_CSV_PATH}")

def get_embeddings_map() -> List[Dict[str, Any]]:
    """Exposed function to return JSON array for frontend scatter plot."""
    if not UMAP_CSV_PATH.exists():
        return []
    df = pd.read_csv(UMAP_CSV_PATH)
    # Fill nan to avoid JSON errors
    df = df.fillna("")
    return df.to_dict(orient="records")

if __name__ == "__main__":
    generate_and_reduce_embeddings()
