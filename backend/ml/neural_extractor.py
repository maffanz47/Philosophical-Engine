from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .preprocessor import TextPreprocessor
from ..models.training_chunk import TrainingChunk
from ..models.experiment import Experiment

class NeuralExtractor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.label_encoder = None
        self.models_dir = Path(__file__).parent.parent / "saved_models"
        self.philosopher_to_category = {
            "Plato": "Metaphysics",
            "Aristotle": "Logic",
            "Kant": "Epistemology",
            "Descartes": "Epistemology",
            "Nietzsche": "Ethics",
            "Hume": "Epistemology",
            "Locke": "Epistemology",
            "Hegel": "Metaphysics",
            "Wittgenstein": "Logic",
            "Sartre": "Ethics",
            "Simone de Beauvoir": "Ethics",
            "Confucius": "Ethics"
        }

    async def train(self, db: AsyncSession, params: dict) -> dict:
        # Load data
        result = await db.execute(select(TrainingChunk))
        chunks = result.scalars().all()
        texts = [chunk.text for chunk in chunks]
        labels = [self.philosopher_to_category.get(chunk.label, "Ethics") for chunk in chunks]

        # Vectorize
        X = self.preprocessor.fit_transform(texts).toarray()
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        hidden_layers = params.get('hidden_layer_sizes', (512, 256, 128))
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=params.get('max_iter', 30))

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')

        # Save
        joblib.dump(self.model, self.models_dir / "neural_extractor_mlp.joblib")
        joblib.dump(self.preprocessor.vectorizer, self.models_dir / "tfidf_vectorizer.joblib")
        joblib.dump(self.label_encoder, self.models_dir / "label_encoder.joblib")

        # Save experiment
        exp = Experiment(
            module_name="neural_extractor",
            algorithm="mlp_classifier",
            accuracy=acc,
            f1_score=f1,
            params_json=params,
            training_size=len(X_train),
            test_size=len(X_test)
        )
        db.add(exp)
        await db.commit()

        return {"accuracy": acc, "f1": f1, "training_size": len(X_train)}

    def predict(self, text: str) -> dict:
        if not self.is_trained():
            return {"status": "not_trained", "message": "The oracle has not yet been awakened. Train the model first."}

        X = self.preprocessor.transform([text]).toarray()
        pred = self.model.predict(X)[0]
        pred_label = self.label_encoder.inverse_transform([pred])[0]
        proba = self.model.predict_proba(X)[0]
        conf = max(proba)

        return {"prediction": pred_label, "confidence": conf}

    def is_trained(self) -> bool:
        return (self.models_dir / "neural_extractor_mlp.joblib").exists() and \
               (self.models_dir / "tfidf_vectorizer.joblib").exists() and \
               (self.models_dir / "label_encoder.joblib").exists()

    def load_models(self):
        if self.is_trained():
            self.model = joblib.load(self.models_dir / "neural_extractor_mlp.joblib")
            self.preprocessor.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")
            self.label_encoder = joblib.load(self.models_dir / "label_encoder.joblib")

# Global instance
neural_extractor = NeuralExtractor()