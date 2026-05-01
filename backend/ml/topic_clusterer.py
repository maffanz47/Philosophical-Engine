from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .preprocessor import TextPreprocessor
from ..models.training_chunk import TrainingChunk
from ..models.experiment import Experiment

class TopicClusterer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.models_dir = Path(__file__).parent.parent / "saved_models"
        self.cluster_terms = {}  # cluster_id: top_terms

    async def train(self, db: AsyncSession, params: dict) -> dict:
        # Load data
        result = await db.execute(select(TrainingChunk))
        chunks = result.scalars().all()
        texts = [chunk.text for chunk in chunks]

        # Vectorize
        X = self.preprocessor.fit_transform(texts)

        # Train K-Means
        kmeans_params = params.get('kmeans', {'n_clusters': 5, 'max_iter': 300})
        self.model = KMeans(**kmeans_params)
        labels = self.model.fit_predict(X)

        # Compute top terms per cluster
        feature_names = self.preprocessor.vectorizer.get_feature_names_out()
        for cluster in range(kmeans_params['n_clusters']):
            cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster]
            if cluster_texts:
                cluster_X = self.preprocessor.transform(cluster_texts)
                mean_tfidf = np.mean(cluster_X.toarray(), axis=0)
                top_indices = np.argsort(mean_tfidf)[-10:][::-1]
                top_terms = [feature_names[i] for i in top_indices]
                self.cluster_terms[cluster] = top_terms

        # Silhouette score
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = 0

        # Save model
        joblib.dump(self.model, self.models_dir / "topic_clusterer_kmeans.joblib")
        joblib.dump(self.preprocessor.vectorizer, self.models_dir / "tfidf_vectorizer.joblib")
        joblib.dump(self.cluster_terms, self.models_dir / "cluster_terms.joblib")

        # Save experiment
        exp = Experiment(
            module_name="topic_clusterer",
            algorithm="k_means",
            params_json=kmeans_params,
            training_size=len(texts),
            notes=f"Silhouette score: {score}"
        )
        db.add(exp)
        await db.commit()

        return {"silhouette_score": score, "training_size": len(texts)}

    def predict(self, text: str) -> dict:
        if not self.is_trained():
            return {"status": "not_trained", "message": "The oracle has not yet been awakened. Train the model first."}

        X = self.preprocessor.transform([text])
        cluster = self.model.predict(X)[0]
        top_terms = self.cluster_terms.get(cluster, [])
        cluster_label = " ".join(top_terms[:3])  # auto-generated label

        return {
            "cluster_id": cluster,
            "top_terms": top_terms,
            "cluster_label": cluster_label
        }

    def is_trained(self) -> bool:
        return (self.models_dir / "topic_clusterer_kmeans.joblib").exists() and \
               (self.models_dir / "tfidf_vectorizer.joblib").exists() and \
               (self.models_dir / "cluster_terms.joblib").exists()

    def load_models(self):
        if self.is_trained():
            self.model = joblib.load(self.models_dir / "topic_clusterer_kmeans.joblib")
            self.preprocessor.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")
            self.cluster_terms = joblib.load(self.models_dir / "cluster_terms.joblib")

# Global instance
topic_clusterer = TopicClusterer()