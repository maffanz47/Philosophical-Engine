from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .preprocessor import TextPreprocessor
from ..models.training_chunk import TrainingChunk
from ..models.experiment import Experiment

class FallacyDetector:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.models_dir = Path(__file__).parent.parent / "saved_models"
        self.fallacies = ["Ad Hominem", "Straw Man", "False Dichotomy", "Slippery Slope", "Appeal to Authority", "Circular Reasoning", "Hasty Generalization", "Red Herring", "No Fallacy"]
        self.explanations = {
            "Ad Hominem": "Attacking the person instead of the argument.",
            "Straw Man": "Misrepresenting the opponent's argument.",
            "False Dichotomy": "Presenting only two options when more exist.",
            "Slippery Slope": "Assuming a chain of events will occur without evidence.",
            "Appeal to Authority": "Using authority as the sole basis for truth.",
            "Circular Reasoning": "The conclusion is used as a premise.",
            "Hasty Generalization": "Drawing a conclusion from insufficient evidence.",
            "Red Herring": "Introducing irrelevant information to divert attention.",
            "No Fallacy": "No logical fallacy detected."
        }

    async def train(self, db: AsyncSession, params: dict) -> dict:
        # Load data
        result = await db.execute(select(TrainingChunk))
        chunks = result.scalars().all()
        texts = [chunk.text for chunk in chunks]
        # Assign random labels for demo
        np.random.seed(42)
        labels = np.random.choice(self.fallacies, len(texts))

        # Vectorize
        X = self.preprocessor.fit_transform(texts)
        y = labels

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train NB
        nb_params = params.get('nb', {'alpha': 1.0})
        self.model = MultinomialNB(**nb_params)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')

        # Save model
        joblib.dump(self.model, self.models_dir / "fallacy_detector_nb.joblib")
        joblib.dump(self.preprocessor.vectorizer, self.models_dir / "tfidf_vectorizer.joblib")

        # Save experiment
        exp = Experiment(
            module_name="fallacy_detector",
            algorithm="naive_bayes",
            accuracy=acc,
            f1_score=f1,
            params_json=nb_params,
            training_size=len(X_train),
            test_size=len(X_test)
        )
        db.add(exp)
        await db.commit()

        return {"accuracy": acc, "f1": f1, "training_size": len(X_train)}

    def predict(self, text: str) -> dict:
        if not self.is_trained():
            return {"status": "not_trained", "message": "The oracle has not yet been awakened. Train the model first."}

        X = self.preprocessor.transform([text])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        conf = max(proba)

        detected = []
        if pred != "No Fallacy":
            detected.append({
                "fallacy": pred,
                "probability": conf,
                "explanation": self.explanations[pred]
            })

        return {"detected_fallacies": detected}

    def is_trained(self) -> bool:
        return (self.models_dir / "fallacy_detector_nb.joblib").exists() and \
               (self.models_dir / "tfidf_vectorizer.joblib").exists()

    def load_models(self):
        if self.is_trained():
            self.model = joblib.load(self.models_dir / "fallacy_detector_nb.joblib")
            self.preprocessor.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")

# Global instance
fallacy_detector = FallacyDetector()