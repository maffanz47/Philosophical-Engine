from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

class ArgumentClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.lr_model = None
        self.svm_model = None
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
        X = self.preprocessor.fit_transform(texts)
        y = np.array(labels)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train LR
        lr_params = params.get('lr', {'C': 1.0, 'max_iter': 1000})
        self.lr_model = LogisticRegression(**lr_params)
        self.lr_model.fit(X_train, y_train)
        lr_pred = self.lr_model.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        lr_f1 = f1_score(y_test, lr_pred, average='weighted')

        # Train SVM
        svm_params = params.get('svm', {'C': 1.0, 'kernel': 'rbf'})
        self.svm_model = SVC(**svm_params)
        self.svm_model.fit(X_train, y_train)
        svm_pred = self.svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        svm_f1 = f1_score(y_test, svm_pred, average='weighted')

        # Save models
        joblib.dump(self.lr_model, self.models_dir / "argument_classifier_lr.joblib")
        joblib.dump(self.svm_model, self.models_dir / "argument_classifier_svm.joblib")
        joblib.dump(self.preprocessor.vectorizer, self.models_dir / "tfidf_vectorizer.joblib")

        # Save experiments
        lr_exp = Experiment(
            module_name="argument_classifier",
            algorithm="logistic_regression",
            accuracy=lr_acc,
            f1_score=lr_f1,
            params_json=lr_params,
            training_size=len(X_train),
            test_size=len(X_test)
        )
        svm_exp = Experiment(
            module_name="argument_classifier",
            algorithm="svm",
            accuracy=svm_acc,
            f1_score=svm_f1,
            params_json=svm_params,
            training_size=len(X_train),
            test_size=len(X_test)
        )
        db.add(lr_exp)
        db.add(svm_exp)
        await db.commit()

        return {"lr": {"accuracy": lr_acc, "f1": lr_f1, "training_size": len(X_train)}, 
                "svm": {"accuracy": svm_acc, "f1": svm_f1, "training_size": len(X_train)}}

    def predict(self, text: str) -> dict:
        if not self.is_trained():
            return {"status": "not_trained", "message": "The oracle has not yet been awakened. Train the model first."}

        X = self.preprocessor.transform([text])

        lr_pred = self.lr_model.predict(X)[0]
        lr_proba = self.lr_model.predict_proba(X)[0]
        lr_conf = max(lr_proba)

        svm_pred = self.svm_model.predict(X)[0]
        if hasattr(self.svm_model, 'decision_function'):
            svm_decision = self.svm_model.decision_function(X)[0]
            svm_conf = max(svm_decision) if isinstance(svm_decision, np.ndarray) else svm_decision
        else:
            svm_conf = 0.5

        return {
            "lr": {"prediction": lr_pred, "confidence": lr_conf},
            "svm": {"prediction": svm_pred, "confidence": svm_conf},
            "winner": "lr" if lr_conf > svm_conf else "svm"
        }

    def is_trained(self) -> bool:
        return (self.models_dir / "argument_classifier_lr.joblib").exists() and \
               (self.models_dir / "argument_classifier_svm.joblib").exists() and \
               (self.models_dir / "tfidf_vectorizer.joblib").exists()

    def load_models(self):
        if self.is_trained():
            self.lr_model = joblib.load(self.models_dir / "argument_classifier_lr.joblib")
            self.svm_model = joblib.load(self.models_dir / "argument_classifier_svm.joblib")
            self.preprocessor.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")

# Global instance
argument_classifier = ArgumentClassifier()