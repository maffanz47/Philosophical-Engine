"""
engine_core.py — Philosophical Text Engine: Offline Core (v5).

Changes vs previous:
  • Separated Training and Usage: engine_core.py is now just a class definition.
  • Added save() and load() methods using joblib and torch.save to persist models.
  • Use `train.py` to scrape, train, and save the models.
  • Use `app.py` to load models instantly and run the console UI.
"""

import sys
import numpy as np
import torch

from taxonomy import (
    TIER1_LABELS, TIER2_LABELS,
    IDX_TO_TIER1, IDX_TO_TIER2,
)
from ingestion import ingest_all
from preprocessing import clean_and_lemmatize, build_tfidf
from models_supervised import (
    train_svm, HierarchicalMLP, train_nn, predict_nn,
)
from models_unsupervised import fit_kmeans, fit_pca
from validation import run_data_integrity_check


class PhilosophyEngine:
    """
    Modular engine: encapsulates the full ML pipeline.
    Designed so every method is independently callable for
    future FastAPI / containerised deployment.
    """

    def __init__(self):
        self.tfidf_vec = None   # fitted TfidfVectorizer

        self.svm_slow  = None   # fitted SVC (slow)
        self.svm_fast  = None   # fitted SVC (fast)
        self.mlp       = None   # trained HierarchicalMLP
        self.kmeans_slow = None # fitted KMeans (slow)
        self.kmeans_fast = None # fitted KMeans (fast)
        self.pca       = None   # fitted PCA
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Phase 1 ───────────────────────────────────────────────────────────
    def ingest(self):
        self.raw_texts, self.y_t1, self.y_t2 = ingest_all()

    # ── Phase 2 ───────────────────────────────────────────────────────────
    def preprocess(self):
        print("\n" + "=" * 60)
        print("  PHASE 2 — PREPROCESSING")
        print("=" * 60)

        print("  Cleaning & lemmatizing all texts ...")
        self.clean_texts = [clean_and_lemmatize(t) for t in self.raw_texts]
        print(f"  Cleaned {len(self.clean_texts):,} documents.")

        # Sparse TF-IDF (SVM, K-Means, PCA)
        self.X_tfidf, self.tfidf_vec = build_tfidf(self.clean_texts)



    # ── Phase 3 ───────────────────────────────────────────────────────────
    def validate(self):
        run_data_integrity_check(self.raw_texts, self.y_t1, self.y_t2)

    # ── Phase 4 ───────────────────────────────────────────────────────────
    def train_supervised(self, nn_epochs: int = 50, batch_size: int = 32):
        print("\n" + "=" * 60)
        print("  PHASE 4 — SUPERVISED TRAINING")
        print("=" * 60)

        y1 = np.array(self.y_t1)
        y2 = np.array(self.y_t2)

        # ── 4a SVM ────────────────────────────────────────────────────────
        print("\n  ── SVM Thinking (Slow) (RBF, class_weight=balanced) ──")
        self.svm_slow = train_svm(self.X_tfidf, y1)

        print("\n  ── SVM (Fast) (RBF, 25% data) ──")
        num_samples = self.X_tfidf.shape[0]
        fast_size = max(1, int(num_samples * 0.25))
        indices = np.random.choice(num_samples, fast_size, replace=False)
        self.svm_fast = train_svm(self.X_tfidf[indices], y1[indices])

        # Shared tensors
        X_dense = torch.tensor(self.X_tfidf.toarray(), dtype=torch.float32)
        y1_t    = torch.tensor(y1, dtype=torch.long)
        y2_t    = torch.tensor(y2, dtype=torch.long)

        # ── 4b MLP ────────────────────────────────────────────────────────
        print(f"\n  ── Hierarchical MLP  (epochs={nn_epochs}, bs={batch_size}) ──")
        mlp = HierarchicalMLP(input_dim=X_dense.shape[1])
        self.mlp = train_nn(mlp, X_dense, y1_t, y2_t,
                            epochs=nn_epochs, batch_size=batch_size)



    # ── Phase 5 ───────────────────────────────────────────────────────────
    def fit_unsupervised(self, n_clusters: int = 7):
        print("\n" + "=" * 60)
        print("  PHASE 5 — UNSUPERVISED MODELS")
        print("=" * 60)
        print("\n  ── K-Means Clustering Thinking (Slow) ──")
        self.kmeans_slow = fit_kmeans(self.X_tfidf, n_clusters=n_clusters)
        
        print("\n  ── K-Means Clustering (Fast) ──")
        num_samples = self.X_tfidf.shape[0]
        fast_size = max(1, int(num_samples * 0.25))
        indices = np.random.choice(num_samples, fast_size, replace=False)
        self.kmeans_fast = fit_kmeans(self.X_tfidf[indices], n_clusters=n_clusters)
        
        self.pca, self.Z_2d = fit_pca(self.X_tfidf)

    # ── Full Build ────────────────────────────────────────────────────────
    def build(self, nn_epochs: int = 50, batch_size: int = 32):
        self.ingest()
        self.preprocess()
        self.validate()
        self.train_supervised(nn_epochs=nn_epochs, batch_size=batch_size)
        self.fit_unsupervised()
        print("\n" + "=" * 60)
        print("  ✓  BUILD COMPLETE — engine ready.")
        print("=" * 60)

    # ── Persistence (Save / Load) ─────────────────────────────────────────
    def save(self, path: str = "engine_artifacts"):
        """Save all trained models and vectorizers to disk."""
        import os, joblib
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.tfidf_vec, f"{path}/tfidf_vec.pkl")
        joblib.dump(self.svm_slow, f"{path}/svm_slow.pkl")
        joblib.dump(self.svm_fast, f"{path}/svm_fast.pkl")
        joblib.dump(self.kmeans_slow, f"{path}/kmeans_slow.pkl")
        joblib.dump(self.kmeans_fast, f"{path}/kmeans_fast.pkl")
        joblib.dump(self.pca, f"{path}/pca.pkl")
        
        # Save PyTorch state dictionaries
        torch.save(self.mlp.state_dict(), f"{path}/mlp.pt")
        print(f"\n  [✓] All models successfully saved to '{path}/'")

    def load(self, path: str = "engine_artifacts"):
        """Load trained models and vectorizers from disk."""
        import os, joblib
        from models_supervised import HierarchicalMLP

        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' not found. Run train.py first.")
        
        print(f"\n  [↓] Loading models from '{path}/' ...")
        self.tfidf_vec = joblib.load(f"{path}/tfidf_vec.pkl")
        self.svm_slow  = joblib.load(f"{path}/svm_slow.pkl")
        self.svm_fast  = joblib.load(f"{path}/svm_fast.pkl")
        self.kmeans_slow = joblib.load(f"{path}/kmeans_slow.pkl")
        self.kmeans_fast = joblib.load(f"{path}/kmeans_fast.pkl")
        self.pca       = joblib.load(f"{path}/pca.pkl")
        
        # Initialize PyTorch models with correct dimensions then load weights
        input_dim = len(self.tfidf_vec.get_feature_names_out())
        self.mlp = HierarchicalMLP(input_dim=input_dim).to(self.device)
        self.mlp.load_state_dict(torch.load(f"{path}/mlp.pt", map_location=self.device, weights_only=True))
        self.mlp.eval()
        print("  [✓] Engine ready.")

    # ── Single-text Inference ─────────────────────────────────────────────
    def predict(self, text: str, mode: str = "slow") -> dict:
        """
        Classify one passage through all trained models.
        Returns a dict ready for formatted console output.
        """
        clean   = clean_and_lemmatize(text)
        x_tfidf = self.tfidf_vec.transform([clean])

        svm_model = self.svm_fast if mode == "fast" else self.svm_slow
        kmeans_model = self.kmeans_fast if mode == "fast" else self.kmeans_slow

        # SVM — deterministic Tier-1 label
        svm_idx   = int(svm_model.predict(x_tfidf)[0])
        svm_label = IDX_TO_TIER1[svm_idx]

        # MLP — hierarchical probabilities
        x_dense = torch.tensor(x_tfidf.toarray(), dtype=torch.float32)
        p1, p2  = predict_nn(self.mlp, x_dense)
        p1, p2  = p1[0], p2[0]   # single sample

        t1_idx = int(np.argmax(p1))
        t2_idx = int(np.argmax(p2))

        # K-Means cluster
        cluster_id = int(kmeans_model.predict(x_tfidf)[0])

        # PCA 2D projection
        x_2d = self.pca.transform(x_tfidf.toarray())[0]

        return {
            "svm_label":   IDX_TO_TIER1[t1_idx],   # use MLP's T1 for label too
            "svm_raw":     svm_label,
            "nn_tier1":    IDX_TO_TIER1[t1_idx],
            "nn_tier2":    IDX_TO_TIER2[t2_idx],
            "nn_probs_t1": {IDX_TO_TIER1[i]: f"{p:.1%}" for i, p in enumerate(p1)},
            "nn_probs_t2": {IDX_TO_TIER2[i]: f"{p:.1%}" for i, p in enumerate(p2)},
            "cluster_id":  cluster_id,
            "pca_coords":  (round(float(x_2d[0]), 4), round(float(x_2d[1]), 4)),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Interactive CLI
# ═══════════════════════════════════════════════════════════════════════════

def manual_test(engine: PhilosophyEngine):
    """
    Console loop: user types a philosophical passage, engine prints:
      • SVM deterministic label
      • NN hierarchical probabilities (Tier-1 → Tier-2)
      • K-Means cluster ID
      • PCA 2D coordinates
    """
    print("\n" + "═" * 62)
    print("  PHILOSOPHICAL TEXT ENGINE — Interactive Console  (v2)")
    print("  Paste any philosophical passage and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("═" * 62 + "\n")

    while True:
        try:
            text = input("φ > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("  Goodbye. ✦")
            break

        r = engine.predict(text)

        t1     = r["nn_tier1"]
        t2     = r["nn_tier2"]
        p1_top = r["nn_probs_t1"][t1]
        p2_top = r["nn_probs_t2"][t2]

        print(f"\n  ┌─ SVM Label  : {r['svm_raw']}")
        print(f"  ├─ NN Result  : {p1_top} {t1}  →  {p2_top} {t2}")
        print(f"  │")
        print(f"  │  Tier-1 probabilities:")
        for label, prob in r["nn_probs_t1"].items():
            marker = " ◀" if label == t1 else ""
            print(f"  │    {label:<18} {prob}{marker}")
        print(f"  │")
        print(f"  │  Tier-2 probabilities:")
        for label, prob in r["nn_probs_t2"].items():
            marker = " ◀" if label == t2 else ""
            print(f"  │    {label:<18} {prob}{marker}")
        print(f"  ├─ K-Means Cluster  : {r['cluster_id']}")
        print(f"  └─ PCA 2D Coords    : "
              f"x={r['pca_coords'][0]}, y={r['pca_coords'][1]}")
        print()


