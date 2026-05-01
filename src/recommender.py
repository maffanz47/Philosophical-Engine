"""Recommendation and clustering utilities for philosophical texts."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class KNNRecommender:
    """KNN recommender backed by cosine similarity."""

    def __init__(self, n_neighbors: int = 3) -> None:
        self.n_neighbors = n_neighbors
        self._knn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
        self._fitted = False
        self._corpus_texts: List[str] = []
        self._features: np.ndarray | None = None

    def fit(self, feature_matrix: np.ndarray, corpus_texts: Sequence[str]) -> None:
        self._knn.fit(feature_matrix)
        self._features = feature_matrix
        self._corpus_texts = list(corpus_texts)
        self._fitted = True

    def recommend(self, query_vector: np.ndarray, top_k: int = 3) -> List[str]:
        if not self._fitted or self._features is None:
            raise RuntimeError("KNNRecommender must be fitted before calling recommend().")

        distances, indices = self._knn.kneighbors(query_vector, n_neighbors=top_k)
        ranked_texts: List[str] = []
        for idx in indices[0]:
            ranked_texts.append(self._corpus_texts[idx])
        return ranked_texts

    def similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Cosine similarity = (A.B) / (||A|| ||B||)."""
        return float(cosine_similarity(vector_a, vector_b)[0][0])


def run_kmeans(feature_matrix: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> np.ndarray:
    """Assign latent clusters using K-Means."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return model.fit_predict(feature_matrix)


def project_pca_2d(feature_matrix: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Project high-dimensional embeddings into 2D for reporting."""
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(feature_matrix)
