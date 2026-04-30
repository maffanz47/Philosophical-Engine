from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch, pickle, numpy as np, json, joblib
from src.model import PhiloClassifier
from src.vectorizer import load_tfidf, text_to_embedding
from src.preprocessor import preprocess_text
from src.similarity import find_similar
from gensim.models import Word2Vec

CLASSES = ["Existentialism", "Rationalism", "Empiricism",
           "Stoicism", "Ethics", "Political"]

state = {}  # holds loaded models

@asynccontextmanager
async def lifespan(app):
    # Load everything at startup
    state["tfidf"]  = load_tfidf()
    state["w2v"]    = Word2Vec.load("models/word2vec.model")
    state["model"]  = PhiloClassifier(200, len(CLASSES))
    state["model"].load_state_dict(torch.load("models/nn_embeddings.pt", map_location="cpu"))
    state["model"].eval()
    
    try:
        state["logreg"] = joblib.load("models/logreg.pkl")
        state["rf"] = joblib.load("models/rf.pkl")
    except FileNotFoundError:
        print("Warning: Scikit-learn models not found. Re-run training.")
        state["logreg"] = None
        state["rf"] = None
    with open("data/dataset.json") as f:
        ds = json.load(f)
    state["metadata"] = ds
    state["all_vecs"]  = np.array([text_to_embedding(d["text"], state["w2v"]) for d in ds])
    print("✓ All models loaded")
    yield

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Philosophical Text Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(body: TextInput):
    cleaned = preprocess_text(body.text)
    vec = text_to_embedding(cleaned, state["w2v"])
    with torch.no_grad():
        logits = state["model"](torch.FloatTensor([vec]))
        probs_nn  = torch.nn.functional.softmax(logits, dim=-1)[0]
    scores_nn = {c: round(float(p), 3) for c, p in zip(CLASSES, probs_nn)}
    dominant_nn = max(scores_nn, key=scores_nn.get)
    
    ensemble_results = {
        "Neural_Network": {
            "dominant_theme": dominant_nn,
            "confidence": scores_nn[dominant_nn],
            "scores": scores_nn
        }
    }
    
    if state.get("logreg"):
        probs_lr = state["logreg"].predict_proba([vec])[0]
        scores_lr = {c: round(float(p), 3) for c, p in zip(CLASSES, probs_lr)}
        dom_lr = max(scores_lr, key=scores_lr.get)
        ensemble_results["Logistic_Regression"] = {
            "dominant_theme": dom_lr,
            "confidence": scores_lr[dom_lr],
            "scores": scores_lr
        }
        
    if state.get("rf"):
        probs_rf = state["rf"].predict_proba([vec])[0]
        scores_rf = {c: round(float(p), 3) for c, p in zip(CLASSES, probs_rf)}
        dom_rf = max(scores_rf, key=scores_rf.get)
        ensemble_results["Random_Forest"] = {
            "dominant_theme": dom_rf,
            "confidence": scores_rf[dom_rf],
            "scores": scores_rf
        }
        
    from collections import Counter
    themes = [res["dominant_theme"] for res in ensemble_results.values()]
    consensus = Counter(themes).most_common(1)[0][0]
    import os
    suggested_books = []
    try:
        files = os.listdir(f"data/{consensus}")
        for f in files[:5]: # Return up to 5 books
            if f.endswith(".txt"):
                suggested_books.append(f)
    except Exception as e:
        pass
    
    return {
        "consensus_theme": consensus,
        "ensemble_results": ensemble_results,
        "suggested_reading": suggested_books
    }

@app.get("/similar")
async def similar(text: str, top_k: int = 5):
    cleaned = preprocess_text(text)
    vec = text_to_embedding(cleaned, state["w2v"])
    return {"results": find_similar(vec, state["all_vecs"], state["metadata"], top_k)}