"""
models.py
=========
All ML model definitions and training logic for the Philosophical Text Engine.

Models implemented:
  - Tier A Baseline: TF-IDF + shallow 1-layer PyTorch ANN (classifier)
  - Tier B Pro:      DistilBERT embeddings + deep PyTorch ANN (classifier)
  - Regression:      Ridge Regression on TF-IDF features → complexity score
  - KNN Librarian:   Cosine-similarity based recommendation engine
  - KMeans:          Unsupervised clustering
  - PCA / t-SNE:     Dimensionality reduction for visualization

Cosine Similarity formula used in KNN:
    similarity = cos(θ) = (A · B) / (||A|| ||B||)
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from transformers import DistilBertTokenizer, DistilBertModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("models")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)


# ===========================================================================
# Helper — versioned model saving
# ===========================================================================
def save_model_versioned(obj, prefix: str, ext: str = "pkl") -> Path:
    """Save a model with a timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_v1.0_{timestamp}.{ext}"
    path = MODELS_DIR / filename
    if ext == "pt":
        torch.save(obj, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    logger.info("Model saved → %s", path)
    return path


# ===========================================================================
# PyTorch Dataset
# ===========================================================================
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===========================================================================
# Tier A — Baseline: Shallow 1-layer ANN
# ===========================================================================
class BaselineANN(nn.Module):
    """Shallow single-hidden-layer ANN for philosophical school classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super(BaselineANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def train_baseline_classifier(df: pd.DataFrame):
    """
    Train Tier A Baseline classifier: TF-IDF + 1-layer ANN.

    Args:
        df: Corpus DataFrame with 'lemmas' and 'school' columns.

    Returns:
        dict with model artifacts and metrics.
    """
    logger.info("=== Training Baseline Classifier (TF-IDF + 1-layer ANN) ===")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["school"].values)
    num_classes = len(le.classes_)

    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["lemmas"].values).toarray().astype(np.float32)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # DataLoaders
    train_ds = TextDataset(X_train, y_train)
    test_ds  = TextDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=64)

    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # Model
    model = BaselineANN(input_dim=X.shape[1], num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.info("Epoch %d/20 | Loss: %.4f", epoch + 1, total_loss / len(train_loader))

    # Evaluation
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    f1 = f1_score(all_true, all_preds, average="weighted")
    logger.info("Baseline Classifier F1 (weighted): %.4f", f1)

    # Save artifacts
    save_model_versioned(model.state_dict(), "ANN_Baseline", ext="pt")
    save_model_versioned(tfidf, "TFIDF_Baseline")
    save_model_versioned(le, "LabelEncoder_Baseline")

    return {
        "model": model,
        "tfidf": tfidf,
        "label_encoder": le,
        "f1_score": f1,
        "num_classes": num_classes,
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }


# ===========================================================================
# Tier B — Pro: Deep ANN with DistilBERT embeddings
# ===========================================================================
class ProANN(nn.Module):
    """
    Deep multi-layer ANN for philosophical school classification.
    Input: 768-dim DistilBERT CLS embeddings.
    """

    def __init__(self, input_dim: int = 768, num_classes: int = 5):
        super(ProANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def get_distilbert_embeddings(texts: list, batch_size: int = 16) -> np.ndarray:
    """
    Compute DistilBERT CLS-token embeddings for a list of texts.

    Args:
        texts:      List of raw text strings.
        batch_size: Batch size for inference.

    Returns:
        numpy array of shape (len(texts), 768).
    """
    logger.info("Computing DistilBERT embeddings for %d texts …", len(texts))
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    bert_model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        with torch.no_grad():
            output = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        if (i // batch_size + 1) % 10 == 0:
            logger.info("  Embedded %d/%d", min(i + batch_size, len(texts)), len(texts))

    return np.vstack(all_embeddings)


def train_pro_classifier(df: pd.DataFrame):
    """
    Train Tier B Pro classifier: DistilBERT embeddings + deep ANN.

    Args:
        df: Corpus DataFrame with 'raw_chunk' and 'school' columns.

    Returns:
        dict with model artifacts and metrics.
    """
    logger.info("=== Training Pro Classifier (DistilBERT + Deep ANN) ===")

    le = LabelEncoder()
    y = le.fit_transform(df["school"].values)
    num_classes = len(le.classes_)

    # Use first 512 chars of each chunk for speed (DistilBERT max 512 tokens)
    texts = [chunk[:512] for chunk in df["raw_chunk"].values]

    # Check for cached embeddings
    embed_cache = MODELS_DIR / "distilbert_embeddings.npy"
    if embed_cache.exists():
        logger.info("Loading cached DistilBERT embeddings …")
        X = np.load(str(embed_cache)).astype(np.float32)
        if len(X) != len(df):
            logger.warning("Cached embeddings length (%d) does not match df length (%d). Recomputing...", len(X), len(df))
            X = None
    else:
        X = None

    if X is None:
        X = get_distilbert_embeddings(texts).astype(np.float32)
        np.save(str(embed_cache), X)
        logger.info("Embeddings cached → %s", embed_cache)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    train_ds = TextDataset(X_train, y_train)
    test_ds  = TextDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=32)

    # Class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    model = ProANN(input_dim=768, num_classes=num_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/30 | Loss: %.4f", epoch + 1, total_loss / len(train_loader))

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    f1 = f1_score(all_true, all_preds, average="weighted")
    f1_per_class = f1_score(all_true, all_preds, average=None)
    logger.info("Pro Classifier F1 (weighted): %.4f", f1)
    
    # map classes to their F1 scores
    class_f1_scores = {le.classes_[i]: float(f1_per_class[i]) for i in range(num_classes)}
    logger.info("Per-class F1: %s", class_f1_scores)

    save_model_versioned(model.state_dict(), "ANN_Pro", ext="pt")
    save_model_versioned(le, "LabelEncoder_Pro")

    return {
        "model": model,
        "label_encoder": le,
        "f1_score": f1,
        "class_f1_scores": class_f1_scores,
        "X_embeddings": X,
        "y": y,
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }


# ===========================================================================
# Regression — Reading Complexity Predictor
# ===========================================================================
def train_regression_model(df: pd.DataFrame, embeddings: np.ndarray):
    """
    Train Ridge Regression to predict complexity_score from DistilBERT + scalar features.

    Args:
        df:         Corpus DataFrame.
        embeddings: DistilBERT embeddings.

    Returns:
        dict with model, RMSE, and feature data.
    """
    logger.info("=== Training Regression Model (Ridge) ===")

    y = df["complexity_score"].values.astype(np.float32)

    scalars = df[['fk_grade', 'avg_sent_len', 'avg_word_len']].values.astype(np.float32)
    X = np.hstack((embeddings, scalars))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    regressor = Ridge(alpha=1.0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    logger.info("Regression RMSE: %.4f", rmse)

    save_model_versioned(regressor, "Ridge_Regression")

    return {"model": regressor, "rmse": rmse}


# ===========================================================================
# KNN — Recommendation Engine (Cosine Similarity)
# ===========================================================================
def build_knn_recommender(X: np.ndarray, df: pd.DataFrame, n_neighbors: int = 4):
    """
    Build a KNN recommendation engine using Cosine Similarity on L2-Normalized embeddings.

    Args:
        X:            Feature matrix (DistilBERT embeddings).
        df:           Corpus DataFrame (for metadata retrieval).
        n_neighbors:  Number of neighbours to retrieve (returns top 3).

    Returns:
        dict with fitted NearestNeighbors model and the dataframe.
    """
    logger.info("=== Building KNN Recommender ===")
    from sklearn.preprocessing import normalize
    X_norm = normalize(X, norm='l2')
    
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    knn.fit(X_norm)
    save_model_versioned(knn, "KNN_Recommender")
    logger.info("KNN Recommender built on %d samples.", len(X))
    return {"model": knn, "df": df, "X": X_norm}


def get_recommendations(query_vector: np.ndarray, knn_artifacts: dict) -> list:
    """
    Return top-3 recommendations for a query vector.

    Args:
        query_vector:  1D numpy array (same feature space as training).
        knn_artifacts: dict from build_knn_recommender().

    Returns:
        List of 3 dicts: {title, author, school, chunk_preview}
    """
    knn = knn_artifacts["model"]
    df  = knn_artifacts["df"]
    q   = query_vector.reshape(1, -1)
    distances, indices = knn.kneighbors(q)
    recs = []
    for idx in indices[0][1:4]:  # skip index 0 (self-match)
        row = df.iloc[idx]
        recs.append({
            "title":         row["title"],
            "author":        row["author"],
            "school":        row["school"],
            "chunk_preview": row["raw_chunk"][:200] + "…",
        })
    return recs


# ===========================================================================
# KMeans — Unsupervised Clustering
# ===========================================================================
def run_kmeans_clustering(X: np.ndarray, df: pd.DataFrame,
                          n_clusters: int = 6) -> pd.DataFrame:
    """
    Apply KMeans clustering to discover latent thematic groupings.

    Args:
        X:          Feature matrix.
        df:         Corpus DataFrame.
        n_clusters: Number of KMeans clusters.

    Returns:
        DataFrame with an added 'cluster' column.
    """
    logger.info("=== Running KMeans Clustering (k=%d) ===", n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df = df.copy()
    df["cluster"] = labels
    save_model_versioned(km, "KMeans")
    logger.info("KMeans clustering complete.")
    return df


# ===========================================================================
# Dimensionality Reduction — PCA & t-SNE
# ===========================================================================
def reduce_dimensions(X: np.ndarray, method: str = "pca") -> np.ndarray:
    """
    Reduce high-dimensional features to 2D for visualization.

    Args:
        X:      Feature matrix (n_samples, n_features).
        method: "pca" or "tsne".

    Returns:
        2D numpy array of shape (n_samples, 2).
    """
    logger.info("=== Dimensionality Reduction via %s ===", method.upper())
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    else:
        reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(X)


# ===========================================================================
# Inference helpers
# ===========================================================================
def predict_with_baseline(text: str, artifacts: dict) -> dict:
    """
    Run full Baseline inference pipeline on a text string.

    Args:
        text:      Input text.
        artifacts: Dict from train_baseline_classifier().

    Returns:
        Dict with predicted_school, confidence_score, raw_probabilities.
    """
    model = artifacts["model"]
    tfidf = artifacts["tfidf"]
    le    = artifacts["label_encoder"]
    model.eval()

    vec = tfidf.transform([text]).toarray().astype(np.float32)
    tensor = torch.tensor(vec, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return {
        "predicted_school":  le.classes_[pred_idx],
        "confidence_score":  round(float(probs[pred_idx]), 4),
        "all_probabilities": {cls: round(float(p), 4)
                              for cls, p in zip(le.classes_, probs)},
    }


def predict_with_pro(text: str, artifacts: dict) -> dict:
    """
    Run full Pro inference pipeline on a text string.

    Args:
        text:      Input text.
        artifacts: Dict from train_pro_classifier().

    Returns:
        Dict with predicted_school, confidence_score, raw_probabilities.
    """
    model = artifacts["model"]
    le    = artifacts["label_encoder"]
    model.eval()

    embedding = get_distilbert_embeddings([text[:512]])
    tensor = torch.tensor(embedding, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return {
        "predicted_school":  le.classes_[pred_idx],
        "confidence_score":  round(float(probs[pred_idx]), 4),
        "all_probabilities": {cls: round(float(p), 4)
                              for cls, p in zip(le.classes_, probs)},
    }
