import sklearn.metrics._scorer
from sklearn.metrics import make_scorer
def _dummy_max_error(y_true, y_pred): return 0.0
if 'max_error' not in getattr(sklearn.metrics._scorer, '_SCORERS', {}):
    if hasattr(sklearn.metrics._scorer, '_SCORERS'):
        sklearn.metrics._scorer._SCORERS['max_error'] = make_scorer(_dummy_max_error)

from deepchecks.nlp import TextData
from deepchecks.nlp.checks import (
    TextPropertyOutliers,
    SpecialCharacters,
    UnknownTokens,
)
from deepchecks.nlp.suites import data_integrity
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import sys

# Add the project root to the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vectorizer import text_to_embedding
from src.model import train_model, train_sklearn_models

CLASSES = ["Existentialism", "Rationalism", "Empiricism", "Stoicism", "Ethics", "Political"]
LABEL_TO_ID = {label: i for i, label in enumerate(CLASSES)}

def run_deepchecks_validation(dataset: list):
    texts  = [d["text"]  for d in dataset]
    labels = [d["label"] for d in dataset]

    train_t, test_t, train_l, test_l = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_data = TextData(
        raw_text=train_t, label=train_l, task_type="text_classification"
    )
    test_data = TextData(
        raw_text=test_t, label=test_l, task_type="text_classification"
    )

    # Check 1: Data Integrity
    suite = data_integrity()
    result = suite.run(train_data)
    os.makedirs("reports", exist_ok=True)
    result.save_as_html("reports/data_integrity.html")
    print("✓ Data integrity report saved to reports/data_integrity.html")

    # Check 2: Train-Test Leakage
    from deepchecks.nlp.checks import TrainTestSamplesMix
    leakage_result = TrainTestSamplesMix().run(train_data, test_data)
    print(f"Leakage check completed. View reports for details if needed.")

    return train_t, test_t, train_l, test_l

def main():
    print("Loading dataset...")
    dataset_path = "data/dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Did you run the preprocessor?")
        sys.exit(1)
        
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    print("Running data validation with Deepchecks...")
    train_t, test_t, train_l, test_l = run_deepchecks_validation(dataset)
    
    print("Training Word2Vec model...")
    # Word2Vec expects list of lists of words
    sentences = [text.split() for text in train_t]
    w2v = Word2Vec(sentences=sentences, vector_size=200, window=5, min_count=1, workers=4)
    
    os.makedirs("models", exist_ok=True)
    w2v.save("models/word2vec.model")
    print("✓ Word2Vec model saved to models/word2vec.model")
    
    print("Vectorizing text data...")
    X_train = np.array([text_to_embedding(text, w2v) for text in train_t])
    X_val   = np.array([text_to_embedding(text, w2v) for text in test_t])
    
    y_train = np.array([LABEL_TO_ID[label] for label in train_l])
    y_val   = np.array([LABEL_TO_ID[label] for label in test_l])
    
    print("Training PhiloClassifier...")
    train_model(
        X_train, y_train, X_val, y_val,
        num_classes=len(CLASSES),
        save_path="models/nn_embeddings.pt",
        epochs=50
    )
    
    print("Training classical models...")
    train_sklearn_models(X_train, y_train, X_val, y_val, save_dir="models")
    
    print("✓ All models trained and saved!")

if __name__ == "__main__":
    main()