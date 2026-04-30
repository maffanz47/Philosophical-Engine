"""
models_unsupervised.py — K-Means Clustering & PCA Dimensionality Reduction.

1. K-Means:
   Minimize Within-Cluster Sum of Squares (WCSS):
       J = Σ_k Σ_{x∈C_k} ‖x − μ_k‖²
   where μ_k is the centroid of cluster k (Euclidean distance).

2. PCA:
   Eigen-decomposition of the covariance matrix:
       Cov(X) = (1/n) Xᵀ X
       Cov(X) v = λ v
   Project onto the top-2 eigenvectors to get 2D coordinates (X, Y).
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def fit_kmeans(X, n_clusters: int = 7):
    """
    K-Means clustering on TF-IDF features.

    Algorithm (Lloyd's):
      1. Initialize k centroids randomly.
      2. Assign each point to nearest centroid:  argmin_k ‖xᵢ − μ_k‖²
      3. Update centroids:  μ_k = mean({xᵢ : xᵢ ∈ C_k})
      4. Repeat until convergence.

    Returns fitted KMeans model.
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(X)
    inertia = km.inertia_  # WCSS = Σ‖x − μ‖²
    print(f"  [K-Means] k={n_clusters}, WCSS (inertia)={inertia:.1f}")
    return km


def fit_pca(X, n_components: int = 2):
    """
    PCA → reduce to 2D for visualization.

    Steps:
      1. Center data: X̃ = X − mean(X)
      2. Covariance matrix: C = (1/n) X̃ᵀ X̃
      3. Eigen-decomposition: C vₖ = λₖ vₖ
      4. Project: Z = X̃ · [v₁, v₂]  → 2D coordinates

    Returns (pca_model, Z_2d array).
    """
    if hasattr(X, "toarray"):
        X = X.toarray()  # sparse → dense
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print(f"  [PCA] Explained variance: PC1={var[0]:.2%}, PC2={var[1]:.2%}")
    return pca, Z
