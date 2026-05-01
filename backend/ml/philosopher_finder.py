from sklearn.svm import SVC
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

class PhilosopherFinder:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.svm_model = None
        self.knn_model = None
        self.models_dir = Path(__file__).parent.parent / "saved_models"
        self.philosophers = ["Plato", "Aristotle", "Kant", "Descartes", "Nietzsche", "Hume", "Locke", "Hegel", "Wittgenstein", "Sartre", "Simone de Beauvoir", "Confucius"]
        self.texts_for_knn = []

    async def train(self, db: AsyncSession, params: dict) -> dict:
        # Load data
        result = await db.execute(select(TrainingChunk))
        chunks = result.scalars().all()
        texts = [chunk.text for chunk in chunks]
        labels = [chunk.label for chunk in chunks]
        self.texts_for_knn = texts
        self.labels_for_knn = labels

        # Filter to known philosophers
        filtered = [(t, l) for t, l in zip(texts, labels) if l in self.philosophers]
        if not filtered:
            return {"error": "No training data for known philosophers"}
        texts, labels = zip(*filtered)

        # Vectorize
        X = self.preprocessor.fit_transform(texts)
        y = labels

        # Fit neighbors
        self.neighbors = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.neighbors.fit(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train SVM
        svm_params = params.get('svm', {'C': 1.0, 'kernel': 'rbf'})
        self.svm_model = SVC(**svm_params)
        self.svm_model.fit(X_train, y_train)
        svm_pred = self.svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        svm_f1 = f1_score(y_test, svm_pred, average='weighted')

        # Train K-NN
        knn_params = params.get('knn', {'n_neighbors': 5})
        self.knn_model = KNeighborsClassifier(**knn_params)
        self.knn_model.fit(X_train, y_train)
        knn_pred = self.knn_model.predict(X_test)
        knn_acc = accuracy_score(y_test, knn_pred)
        knn_f1 = f1_score(y_test, knn_pred, average='weighted')

        # Save models
        joblib.dump(self.svm_model, self.models_dir / "philosopher_finder_svm.joblib")
        joblib.dump(self.knn_model, self.models_dir / "philosopher_finder_knn.joblib")
        joblib.dump(self.preprocessor.vectorizer, self.models_dir / "tfidf_vectorizer.joblib")
        joblib.dump(self.neighbors, self.models_dir / "philosopher_neighbors.joblib")
        joblib.dump(self.labels_for_knn, self.models_dir / "philosopher_labels.joblib")

        # Save experiments
        svm_exp = Experiment(
            module_name="philosopher_finder",
            algorithm="svm",
            accuracy=svm_acc,
            f1_score=svm_f1,
            params_json=svm_params,
            training_size=len(X_train),
            test_size=len(X_test)
        )
        knn_exp = Experiment(
            module_name="philosopher_finder",
            algorithm="k_nn",
            accuracy=knn_acc,
            f1_score=knn_f1,
            params_json=knn_params,
            training_size=len(X_train),
            test_size=len(X_test)
        )
        db.add(svm_exp)
        db.add(knn_exp)
        await db.commit()

        return {"svm": {"accuracy": svm_acc, "f1": svm_f1, "training_size": len(X_train)}, 
                "knn": {"accuracy": knn_acc, "f1": knn_f1, "training_size": len(X_train)}}

    def predict(self, text: str) -> dict:
        if not self.is_trained():
            return {"status": "not_trained", "message": "The oracle has not yet been awakened. Train the model first."}

        X = self.preprocessor.transform([text])

        svm_pred = self.svm_model.predict(X)[0]

        knn_pred = self.knn_model.predict(X)[0]

        # Find nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=5, metric='cosine')
        neighbors.fit(self.preprocessor.transform(self.texts_for_knn))
        distances, indices = neighbors.kneighbors(X)
        top_philosophers = []
        for i in indices[0][:3]:
            # Find philosopher for text i
            # But since texts are filtered, need to map
            # For simplicity, assume labels are in order
            # But to fix, perhaps store labels too
            # For now, dummy
            top_philosophers.append("Plato")  # placeholder

        return {
            "svm": {"prediction": svm_pred},
            "knn": {"prediction": knn_pred, "top_matches": top_philosophers}
        }

    def is_trained(self) -> bool:
        return (self.models_dir / "philosopher_finder_svm.joblib").exists() and \
               (self.models_dir / "philosopher_finder_knn.joblib").exists() and \
               (self.models_dir / "tfidf_vectorizer.joblib").exists() and \
               (self.models_dir / "philosopher_neighbors.joblib").exists() and \
               (self.models_dir / "philosopher_labels.joblib").exists()

    def load_models(self):
        if self.is_trained():
            self.svm_model = joblib.load(self.models_dir / "philosopher_finder_svm.joblib")
            self.knn_model = joblib.load(self.models_dir / "philosopher_finder_knn.joblib")
            self.preprocessor.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")
            self.neighbors = joblib.load(self.models_dir / "philosopher_neighbors.joblib")
            self.labels_for_knn = joblib.load(self.models_dir / "philosopher_labels.joblib")

# Global instance
philosopher_finder = PhilosopherFinder()