import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import mlflow
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/philosophy_corpus.csv")
UMAP_CSV_PATH = Path("data/processed/embeddings/umap_2d.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CLUSTERS_JSON_PATH = REPORTS_DIR / "clusters.json"

MLFLOW_EXPERIMENT = "philosophical-engine-clustering"

def load_data():
    if not DATA_PATH.exists() or not UMAP_CSV_PATH.exists():
        logger.warning("Required data files not found for clustering.")
        return None, None
        
    df = pd.read_csv(DATA_PATH)
    umap_df = pd.read_csv(UMAP_CSV_PATH)
    return df, umap_df

def extract_top_terms(texts: List[str], top_n: int = 20) -> List[str]:
    if not texts:
        return []
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        X = tfidf.fit_transform(texts)
        indices = np.argsort(tfidf.idf_)[::-1]
        features = tfidf.get_feature_names_out()
        top_features = [features[i] for i in indices[:top_n]]
        return top_features
    except ValueError:
        return []

def assign_label(top_terms: List[str]) -> str:
    # A simple heuristic for "emergent school" label based on terms
    terms_str = " ".join(top_terms).lower()
    if "god" in terms_str or "faith" in terms_str or "theology" in terms_str:
        return "Theological Idealism"
    if "reason" in terms_str or "logic" in terms_str or "mind" in terms_str:
        return "Rationalist Framework"
    if "existence" in terms_str or "being" in terms_str or "angst" in terms_str:
        return "Existential Inquiry"
    if "sense" in terms_str or "experience" in terms_str or "matter" in terms_str:
        return "Empirical Materialism"
    if "state" in terms_str or "law" in terms_str or "society" in terms_str:
        return "Political Philosophy"
    return "Analytic Discourse"

def cluster_thoughts():
    df, umap_df = load_data()
    if df is None or df.empty:
        return

    # Use UMAP 2D coordinates for clustering
    X = umap_df[['x', 'y']].values
    true_labels = df['school_label'].fillna("Other")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    timestamp = int(time.time())

    # 1. K-Means sweep
    best_k = -1
    best_score = -1
    best_kmeans = None
    
    logger.info("Sweeping K-Means from k=3 to 15...")
    for k in range(3, 16):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans

    # Evaluate best K-Means
    kmeans_labels = best_kmeans.labels_
    ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)

    with mlflow.start_run(run_name=f"KMeans_Best_{timestamp}"):
        mlflow.log_params({"k": best_k, "algorithm": "K-Means"})
        mlflow.log_metrics({"silhouette_score": best_score, "adjusted_rand_index": ari_kmeans})

    # 2. HDBSCAN
    logger.info("Running HDBSCAN...")
    hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    hdb_labels = hdb.fit_predict(X)
    
    # Calculate silhouette score only for clustered points (label != -1)
    clustered_mask = hdb_labels != -1
    if np.sum(clustered_mask) > 0 and len(set(hdb_labels[clustered_mask])) > 1:
        score_hdb = silhouette_score(X[clustered_mask], hdb_labels[clustered_mask])
    else:
        score_hdb = -1.0
        
    ari_hdb = adjusted_rand_score(true_labels, hdb_labels)

    with mlflow.start_run(run_name=f"HDBSCAN_{timestamp}"):
        mlflow.log_params({"min_cluster_size": 10, "min_samples": 5, "algorithm": "HDBSCAN"})
        mlflow.log_metrics({"silhouette_score": score_hdb, "adjusted_rand_index": ari_hdb})

    # Choose best model to generate output (K-Means vs HDBSCAN by ARI)
    if ari_kmeans >= ari_hdb:
        logger.info(f"Selected K-Means (k={best_k}) as best model.")
        final_labels = kmeans_labels
    else:
        logger.info("Selected HDBSCAN as best model.")
        final_labels = hdb_labels

    # Extract info per cluster
    unique_labels = set(final_labels)
    clusters_info = []

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue # Skip noise in HDBSCAN
            
        idx = np.where(final_labels == cluster_id)[0]
        cluster_texts = df.iloc[idx]['full_text'].astype(str).tolist()
        size = len(idx)
        
        top_terms = extract_top_terms(cluster_texts, 20)
        emergent_label = assign_label(top_terms)
        
        # Get up to 5 representative books (closest to centroid for K-Means, or random for HDBSCAN)
        rep_books = []
        if ari_kmeans >= ari_hdb:
            centroid = best_kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(X[idx] - centroid, axis=1)
            closest_idx = idx[np.argsort(distances)[:5]]
        else:
            closest_idx = idx[:5]
            
        for i in closest_idx:
            row = df.iloc[i]
            rep_books.append({
                "gutenberg_id": int(row['gutenberg_id']),
                "title": row['title'],
                "author": row['author']
            })
            
        clusters_info.append({
            "cluster_id": int(cluster_id),
            "label": emergent_label,
            "size": size,
            "top_terms": top_terms,
            "representative_books": rep_books
        })
        
    with open(CLUSTERS_JSON_PATH, "w") as f:
        json.dump(clusters_info, f, indent=2)
        
    logger.info(f"Saved {len(clusters_info)} clusters to {CLUSTERS_JSON_PATH}")

def get_clusters() -> List[Dict[str, Any]]:
    if not CLUSTERS_JSON_PATH.exists():
        return [{"error": "Clusters not generated yet."}]
    with open(CLUSTERS_JSON_PATH, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    cluster_thoughts()
