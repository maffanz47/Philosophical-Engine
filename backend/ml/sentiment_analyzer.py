from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
from pathlib import Path
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .preprocessor import TextPreprocessor
from ..models.training_chunk import TrainingChunk
from ..models.experiment import Experiment

class SentimentAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.lr_model = None
        self.knn_model = None
        self.models_dir = Path(__file__).parent.parent / "saved_models"
        self.labels = ["Optimistic", "Pessimistic", "Skeptical", "Nihilistic", "Empirical", "Rationalist", "Mystical"]
        # For simplicity, assign random labels or based on philosopher, but for demo, random
        # In real, need labeled data, but since not, use random for now
        self.texts_for_knn = []  # store texts for nearest neighbors

    async def train(self, db: AsyncSession, params: dict) -> dict:
        # Load data
        result = await db.execute(select(TrainingChunk))
        chunks = result.scalars().all()
        texts = [chunk.text for chunk in chunks]
        self.texts_for_knn = texts
        # Assign random labels for demo
        np.random.seed(42)
        labels = np.random.choice(self.labels, len(texts))

        # Vectorize
        X = self.preprocessor.fit_transform(texts)
        y = labels

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train LR
        lr_params = params.get('lr', {'C': 1.0, 'max_iter': 1000})
        self.lr_model = LogisticRegression(**lr_params)
        self.lr_model.fit(X_train, y_train)
        lr_pred = self.lr_model.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        lr_f1 = f1_score(y_test, lr_pred, average='weighted')

        # Train K-NN
        knn_params = params.get('knn', {'n_neighbors': 5})
        self.knn_model = KNeighborsClassifier(**knn_params)
        self.knn_model.fit(X_train, y_train)
        knn_pred = self.knn_model.predict(X_test)
        knn_acc = accuracy_score(y_test, knn_pred)
        knn_f1 = f1_score(y_test, knn_pred, average='weighted')

        # Save models
        joblib.dump(self.lr_model, self.models_dir / "sentiment_analyzer_lr.joblib")
        joblib.dump(self.knn_model, self.models_dir / "sentiment_analyzer_knn.joblib")
        joblib.dump(self.preprocessor.vectorizer, self.models_dir / "tfidf_vectorizer.joblib")
        joblib.dump(self.texts_for_knn, self.models_dir / "sentiment_texts.joblib")

        # Save experiments
        lr_exp = Experiment(
            module_name="sentiment_analyzer",
            algorithm="logistic_regression",
            accuracy=lr_acc,
            f1_score=lr_f1,
            params_json=lr_params,
            training_size=len(X_train),
            test_size=len(X_test)
        )
        knn_exp = Experiment(
            module_name="sentiment_analyzer",
            algorithm="k_nn",
            accuracy=knn_acc,
            f1_score=knn_f1,
            params_json=knn_params,
            training_size=len(X_train),
            test_size=len(X_test)
        )
        db.add(lr_exp)
        db.add(knn_exp)
        await db.commit()

        return {"lr": {"accuracy": lr_acc, "f1": lr_f1, "training_size": len(X_train)}, 
                "knn": {"accuracy": knn_acc, "f1": knn_f1, "training_size": len(X_train)}}

    def predict(self, text: str) -> dict:
        if not self.is_trained():
            return {"status": "not_trained", "message": "The oracle has not yet been awakened. Train the model first."}

        X = self.preprocessor.transform([text])

        lr_pred = self.lr_model.predict(X)[0]
        lr_proba = self.lr_model.predict_proba(X)[0]
        lr_conf = max(lr_proba)

        knn_pred = self.knn_model.predict(X)[0]

        # Find nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=5, metric='cosine')
        neighbors.fit(self.preprocessor.transform(self.texts_for_knn))
        distances, indices = neighbors.kneighbors(X)
        top_texts = [self.texts_for_knn[i] for i in indices[0][:3]]

        return {
            "lr": {"prediction": lr_pred, "confidence": lr_conf},
            "knn": {"prediction": knn_pred, "top_matches": top_texts}
        }

    def is_trained(self) -> bool:
        return (self.models_dir / "sentiment_analyzer_lr.joblib").exists() and \
               (self.models_dir / "sentiment_analyzer_knn.joblib").exists() and \
               (self.models_dir / "tfidf_vectorizer.joblib").exists()

    def load_models(self):
        if self.is_trained():
            self.lr_model = joblib.load(self.models_dir / "sentiment_analyzer_lr.joblib")
            self.knn_model = joblib.load(self.models_dir / "sentiment_analyzer_knn.joblib")
            self.preprocessor.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")
            self.texts_for_knn = joblib.load(self.models_dir / "sentiment_texts.joblib")

# Global instance
sentiment_analyzer = SentimentAnalyzer()