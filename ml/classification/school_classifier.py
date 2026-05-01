import os
import json
import time
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from scipy.sparse import hstack

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path("data/processed/philosophy_corpus.csv")
MODEL_DIR = Path("models/classification")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_EXPERIMENT = "philosophical-engine-classification"

# Supported philosophical schools for classification
SCHOOLS = [
    "Empiricism",
    "Rationalism",
    "Existentialism",
    "Stoicism",
    "Idealism",
    "Pragmatism",
    "Other",
]

# Global loaded model for inference
_best_model = None
_vectorizer = None
_label_encoder = None
_is_bert = False
_tokenizer = None


def models_dir(subdir: str) -> Path:
    return Path("models") / subdir


def _latest_model_path(subdir: str) -> Optional[str]:
    model_dir = models_dir(subdir)
    if model_dir is None or not model_dir.exists():
        return None

    candidates = list(model_dir.glob("school_classifier_v*.pkl"))
    if not candidates:
        return None

    return str(max(candidates, key=os.path.getctime))


def _try_load_artifact(path: str) -> Optional[Dict[str, Any]]:
    try:
        artifact_path = Path(path)
        if not artifact_path.exists():
            return None
        return joblib.load(artifact_path)
    except Exception as e:
        logger.error(f"Failed to load artifact from {path}: {e}")
        return None


def train_school_classifier(df: Optional[pd.DataFrame] = None) -> None:
    raise NotImplementedError("School classifier training is not implemented yet.")

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        logger.warning(f"Data file not found at {DATA_PATH}. Returning empty.")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_models():
    df = load_data()
    if df.empty or 'school_label' not in df.columns:
        logger.error("Insufficient data to train classification models.")
        return

    # Drop missing text
    df = df.dropna(subset=['full_text'])
    
    # Stratified sampling for robust train/test split if possible
    # Just in case some classes are very small, we'll try standard split first
    X_text = df['full_text'].astype(str).tolist()
    X_num = df[['avg_sentence_length', 'vocab_richness', 'sentiment_polarity']].fillna(0).values
    
    le = LabelEncoder()
    y = le.fit_transform(df['school_label'])
    
    # Check if we have enough samples for stratification
    unique_labels, counts = np.unique(y, return_counts=True)
    min_class_count = min(counts)
    
    stratify_param = y if min_class_count >= 2 else None
    if stratify_param is None:
        logger.warning(f"Insufficient samples for stratification (min class count: {min_class_count}). Using random split.")
    
    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    # Combine TF-IDF with structured features
    X_train_combined = hstack([X_train_tfidf, X_train_num]).tocsr()
    X_test_combined = hstack([X_test_tfidf, X_test_num]).tocsr()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    best_f1 = 0
    best_model_name = ""
    timestamp = int(time.time())

    # 1. Baseline: Logistic Regression
    with mlflow.start_run(run_name=f"LogisticRegression_{timestamp}"):
        logger.info("Training Logistic Regression...")
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_train_combined, y_train)
        
        preds = lr.predict(X_test_combined)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        
        mlflow.log_params({"C": 1.0, "max_iter": 1000, "model": "LogisticRegression"})
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        
        # Handle case where test set might have only one class
        unique_test_labels = np.unique(y_test)
        if len(unique_test_labels) == 1:
            target_names = [le.inverse_transform([unique_test_labels[0]])[0]]
        else:
            target_names = le.classes_
            
        report = classification_report(y_test, preds, target_names=target_names, output_dict=True)
        with open("reports/lr_classification_report.json", "w") as f:
            json.dump(report, f)
        mlflow.log_artifact("reports/lr_classification_report.json")
        
        cm_path = "reports/lr_confusion_matrix.png"
        plot_confusion_matrix(y_test, preds, le.classes_, "LR Confusion Matrix", cm_path)
        mlflow.log_artifact(cm_path)
        
        mlflow.sklearn.log_model(lr, artifact_path="model")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = "LogisticRegression"
            joblib.dump({"model": lr, "vectorizer": tfidf, "label_encoder": le, "is_bert": False}, 
                        MODEL_DIR / f"school_classifier_v{timestamp}.pkl")

    # 2. Improved: XGBoost
    with mlflow.start_run(run_name=f"XGBoost_{timestamp}"):
        logger.info("Training XGBoost...")
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        xgb.fit(X_train_combined, y_train)
        
        preds = xgb.predict(X_test_combined)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        
        mlflow.log_params({"model": "XGBoost"})
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        
        # Handle case where test set might have only one class
        unique_test_labels = np.unique(y_test)
        if len(unique_test_labels) == 1:
            target_names = [le.inverse_transform([unique_test_labels[0]])[0]]
        else:
            target_names = le.classes_
            
        report = classification_report(y_test, preds, target_names=target_names, output_dict=True)
        with open("reports/xgb_classification_report.json", "w") as f:
            json.dump(report, f)
        mlflow.log_artifact("reports/xgb_classification_report.json")
        
        cm_path = "reports/xgb_confusion_matrix.png"
        plot_confusion_matrix(y_test, preds, le.classes_, "XGBoost Confusion Matrix", cm_path)
        mlflow.log_artifact(cm_path)
        
        mlflow.xgboost.log_model(xgb, artifact_path="model")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = "XGBoost"
            joblib.dump({"model": xgb, "vectorizer": tfidf, "label_encoder": le, "is_bert": False}, 
                        MODEL_DIR / f"school_classifier_v{timestamp}.pkl")

    # 3. Advanced: DistilBERT
    # Note: Training BERT on full texts takes very long. We truncate aggressively for demonstration.
    with mlflow.start_run(run_name=f"DistilBERT_{timestamp}"):
        logger.info("Training DistilBERT...")
        try:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            bert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(le.classes_))
            
            # Truncate to first 512 tokens
            train_encodings = tokenizer(X_train_text, truncation=True, padding=True, max_length=512)
            test_encodings = tokenizer(X_test_text, truncation=True, padding=True, max_length=512)
            
            train_dataset = ClassificationDataset(train_encodings, y_train)
            test_dataset = ClassificationDataset(test_encodings, y_test)
            
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch"
            )
            
            trainer = Trainer(
                model=bert_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=lambda p: {
                    "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
                    "f1_macro": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average='macro')
                }
            )
            
            trainer.train()
            
            # Evaluation
            preds_output = trainer.predict(test_dataset)
            preds = np.argmax(preds_output.predictions, axis=1)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            
            mlflow.log_params({"epochs": 3, "model": "DistilBERT"})
            mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
            
            # Handle case where test set might have only one class
            unique_test_labels = np.unique(y_test)
            if len(unique_test_labels) == 1:
                target_names = [le.inverse_transform([unique_test_labels[0]])[0]]
            else:
                target_names = le.classes_
                
            report = classification_report(y_test, preds, target_names=target_names, output_dict=True)
            with open("reports/bert_classification_report.json", "w") as f:
                json.dump(report, f)
            mlflow.log_artifact("reports/bert_classification_report.json")
            
            cm_path = "reports/bert_confusion_matrix.png"
            plot_confusion_matrix(y_test, preds, le.classes_, "DistilBERT Confusion Matrix", cm_path)
            mlflow.log_artifact(cm_path)
            
            mlflow.pytorch.log_model(bert_model, artifact_path="model")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = "DistilBERT"
                # Save BERT model
                bert_save_path = MODEL_DIR / f"school_classifier_bert_v{timestamp}"
                bert_model.save_pretrained(bert_save_path)
                tokenizer.save_pretrained(bert_save_path)
                joblib.dump({"model_path": str(bert_save_path), "label_encoder": le, "is_bert": True}, 
                            MODEL_DIR / f"school_classifier_v{timestamp}.pkl")
                
        except Exception as e:
            logger.error(f"DistilBERT training failed: {e}")

    logger.info(f"Training complete. Best model: {best_model_name} with F1-macro: {best_f1:.4f}")

def load_best_model():
    global _best_model, _vectorizer, _label_encoder, _is_bert, _tokenizer
    if _best_model is not None:
        return # Already loaded

    latest_model_path = _latest_model_path("classification")
    if not latest_model_path:
        logger.warning("No trained classification model found.")
        return

    data = _try_load_artifact(latest_model_path)
    if not data:
        logger.warning("Failed to load the classification model artifact.")
        return

    _label_encoder = data.get('label_encoder')
    _is_bert = data.get('is_bert', False)

    if _is_bert:
        model_path = data.get('model_path')
        _tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        _best_model = DistilBertForSequenceClassification.from_pretrained(model_path)
        _best_model.eval()
    else:
        _best_model = data.get('model')
        _vectorizer = data.get('vectorizer')

def predict_school(text: str, features_num: Optional[List[float]] = None) -> Dict[str, Any]:
    load_best_model()
    if _best_model is None:
        return {
            "school": "Other",
            "confidence": 0.0,
            "top3": [{"school": "Other", "confidence": 0.0}],
        }

    if _is_bert:
        inputs = _tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = _best_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    else:
        if features_num is None:
            # If no structured features provided, pad with zeros (assuming avg_sentence_length, vocab_richness, sentiment_polarity)
            features_num = [0.0, 0.0, 0.0]

        X_tfidf = _vectorizer.transform([text])
        try:
            X_combined = hstack([X_tfidf, np.array([features_num])]).tocsr()
        except Exception:
            X_combined = np.hstack(
                [np.atleast_2d(np.asarray(X_tfidf)), np.atleast_2d(np.asarray(features_num))]
            )

        probs = _best_model.predict_proba(X_combined)[0]

    # Get top 3
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [{"school": _label_encoder.inverse_transform([i])[0], "confidence": float(probs[i])} for i in top3_idx]

    return {
        "school": top3[0]["school"],
        "confidence": top3[0]["confidence"],
        "top3": top3,
    }

if __name__ == "__main__":
    # If run directly, try to train
    train_models()
