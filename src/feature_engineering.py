"""Feature extraction for baseline (TF-IDF) and pro (DistilBERT) tiers."""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_features(texts: Sequence[str], max_features: int = 1024) -> tuple[np.ndarray, TfidfVectorizer]:
    """Create sparse TF-IDF representation for baseline training."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer


def _hash_embedding(text: str, dim: int = 768) -> np.ndarray:
    """Deterministic fallback embedding when transformer weights are unavailable."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def distilbert_features(texts: Sequence[str], max_length: int = 256) -> np.ndarray:
    """
    Create dense DistilBERT embeddings for pro tier.

    Falls back to deterministic hashed embeddings if pretrained weights are not
    available in the current environment.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        model.eval()

        vectors: List[np.ndarray] = []
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                outputs = model(**encoded)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                vectors.append(cls_embedding.astype(np.float32))
        return np.vstack(vectors)
    except Exception:
        fallback = [_hash_embedding(text) for text in texts]
        return np.vstack(fallback)
