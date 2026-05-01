from sklearn.linear_model import Ridge
import numpy as np
from pathlib import Path
import joblib
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.experiment import Experiment

class ComplexityScorer:
    def __init__(self):
        self.model = None
        self.models_dir = Path(__file__).parent.parent / "saved_models"

    async def train(self, db: AsyncSession, params: dict) -> dict:
        # Self-supervised: no training data needed
        # But to have a model, perhaps dummy
        # For now, no model
        # Save dummy experiment
        exp = Experiment(
            module_name="complexity_scorer",
            algorithm="ridge_regression",
            params_json=params,
            notes="Self-supervised scoring"
        )
        db.add(exp)
        await db.commit()
        return {"rmse": None, "training_size": 0}

    def predict(self, text: str) -> dict:
        # Compute features
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            sentences = [text]
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        vocab = set(text.lower().split())
        abstract_vocab_ratio = len(vocab) / len(words) if words else 0
        logical_connectors = ['because', 'therefore', 'thus', 'hence', 'consequently']
        logical_connector_count = sum(text.lower().count(conn) for conn in logical_connectors)
        type_token_ratio = len(vocab) / len(words) if words else 0
        clause_density = text.count(',') / len(sentences) if sentences else 0

        # Weighted sum (tune weights as needed)
        score = (avg_sentence_length * 0.2 + 
                 avg_word_length * 0.1 + 
                 abstract_vocab_ratio * 0.3 + 
                 logical_connector_count * 0.1 + 
                 type_token_ratio * 0.2 + 
                 clause_density * 0.1)
        score = min(max(score, 0), 100)

        return {"complexity_score": score}

    def is_trained(self) -> bool:
        return True

    def load_models(self):
        pass

# Global instance
complexity_scorer = ComplexityScorer()