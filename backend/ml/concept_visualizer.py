from sklearn.decomposition import PCA
import joblib
from pathlib import Path
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .preprocessor import TextPreprocessor
from ..models.training_chunk import TrainingChunk
from ..models.experiment import Experiment

class ConceptVisualizer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.pca = None
        self.models_dir = Path(__file__).parent.parent / "saved_models"
        self.reference_coords = []  # list of {"philosopher": str, "x": float, "y": float}

    async def train(self, db: AsyncSession, params: dict) -> dict:
        # Load all chunks
        result = await db.execute(select(TrainingChunk))
        chunks = result.scalars().all()
        texts = [chunk.text for chunk in chunks]
        philosophers = [chunk.label for chunk in chunks]

        # Vectorize
        X = self.preprocessor.fit_transform(texts)

        # PCA to 2D
        self.pca = PCA(n_components=2)
        coords = self.pca.fit_transform(X.toarray())

        # Create reference coords
        self.reference_coords = [
            {"philosopher": phil, "x": float(coord[0]), "y": float(coord[1])}
            for phil, coord in zip(philosophers, coords)
        ]

        # Save
        joblib.dump(self.pca, self.models_dir / "concept_visualizer_pca.joblib")
        joblib.dump(self.preprocessor.vectorizer, self.models_dir / "tfidf_vectorizer.joblib")
        joblib.dump(self.reference_coords, self.models_dir / "reference_coords.joblib")

        # Save experiment
        exp = Experiment(
            module_name="concept_visualizer",
            algorithm="pca",
            params_json=params,
            training_size=len(texts)
        )
        db.add(exp)
        await db.commit()

        return {"training_size": len(texts)}

    def predict(self, text: str) -> dict:
        if not self.is_trained():
            return {"status": "not_trained", "message": "The oracle has not yet been awakened. Train the model first."}

        X = self.preprocessor.transform([text])
        user_coord = self.pca.transform(X.toarray())[0]

        return {
            "user_coord": {"x": float(user_coord[0]), "y": float(user_coord[1])},
            "reference_coords": self.reference_coords
        }

    def is_trained(self) -> bool:
        return (self.models_dir / "concept_visualizer_pca.joblib").exists() and \
               (self.models_dir / "tfidf_vectorizer.joblib").exists() and \
               (self.models_dir / "reference_coords.joblib").exists()

    def load_models(self):
        if self.is_trained():
            self.pca = joblib.load(self.models_dir / "concept_visualizer_pca.joblib")
            self.preprocessor.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")
            self.reference_coords = joblib.load(self.models_dir / "reference_coords.joblib")

# Global instance
concept_visualizer = ConceptVisualizer()