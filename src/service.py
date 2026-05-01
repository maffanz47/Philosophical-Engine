"""Shared inference service for API predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessor import preprocess_text
from src.recommender import KNNRecommender


@dataclass
class PredictionResult:
    predicted_school: str
    confidence_score: float
    complexity_index: float
    top_3_recommendations: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "predicted_school": self.predicted_school,
            "confidence_score": self.confidence_score,
            "complexity_index": self.complexity_index,
            "top_3_recommendations": self.top_3_recommendations,
        }


class PhilosophicalInferenceService:
    """Lightweight inference service for the `/predict` endpoint."""

    def __init__(self) -> None:
        self.school_labels = [
            "Stoicism",
            "Existentialism",
            "Islamic Philosophy",
            "Rationalism",
            "Empiricism",
            "Nihilism",
        ]
        self.reference_texts = [
            "Virtue and discipline shape a tranquil life aligned with nature.",
            "Existence comes first and meaning is created through action.",
            "Reason and revelation can harmonize in the search for truth.",
            "Deductive reason guides certainty and universal principles.",
            "Knowledge begins with sensory experience and observation.",
            "Traditional values collapse and the void demands reinvention.",
        ]
        self.vectorizer = TfidfVectorizer(max_features=512)
        self.reference_features = self.vectorizer.fit_transform(self.reference_texts).toarray()

        self.recommender = KNNRecommender(n_neighbors=3)
        self.recommender.fit(self.reference_features, self.reference_texts)

        self.school_prototypes = {
            label: self.reference_features[idx]
            for idx, label in enumerate(self.school_labels)
        }

    def _complexity_score(self, text: str, tokens: List[str]) -> float:
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        avg_token_len = sum(len(token) for token in tokens) / max(1, len(tokens))
        sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
        sentence_density = len(tokens) / sentence_count
        raw = 3.5 * unique_ratio + 0.45 * avg_token_len + 0.02 * sentence_density
        return float(max(0.0, min(10.0, raw)))

    def predict(self, text: str) -> PredictionResult:
        tokens = preprocess_text(text)
        processed_text = " ".join(tokens) if tokens else text.lower()

        query_vector = self.vectorizer.transform([processed_text]).toarray()

        best_label = self.school_labels[0]
        best_score = -1.0
        for label, proto in self.school_prototypes.items():
            score = self.recommender.similarity(query_vector, proto.reshape(1, -1))
            if score > best_score:
                best_score = score
                best_label = label

        recommendations = self.recommender.recommend(query_vector, top_k=3)
        complexity = self._complexity_score(text, tokens)

        return PredictionResult(
            predicted_school=best_label,
            confidence_score=round(float(np.clip(best_score, 0.0, 1.0)), 4),
            complexity_index=round(complexity, 4),
            top_3_recommendations=recommendations,
        )


inference_service = PhilosophicalInferenceService()
