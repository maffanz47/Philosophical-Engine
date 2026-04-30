"""
models_supervised.py — SVM + Hierarchical Neural Networks (PyTorch) (v6).

Changes vs v5 (anti-Idealism-Singularity update):
  • MLP: Aggressive bottleneck architecture to crush sparse noise.
         Input(10k) → 128 (GELU, Dropout 0.6)
                    → 32  (GELU, Dropout 0.6)
                    → dual heads
  • Activations: All ReLU replaced with nn.GELU() to handle
    sparse TF-IDF inputs without dying neurons.
  • Optimizer: Adam → AdamW (lr=0.0005, weight_decay=0.02).
    L2 regularisation stops the network from over-weighting
    single high-IDF Kant/Berkeley vocabulary.
  • Hierarchical Masking retained in predict_nn (unchanged).

Math Summary:
  SVM:   min ½‖w‖²  s.t. yᵢ(w·φ(xᵢ)+b) ≥ 1  (RBF kernel, balanced weights)
  MLP:   bottleneck backbone → dual linear heads
  Loss:  L_t1  = CE(logits_t1, y1)
         L_t2  = CE(logits_t2, y2, weight=T2_CLASS_W)
         L_total = L_t1 + L_t2
  Opt:   AdamW — θ_{t+1} = θ_t − α·m̂_t/(√v̂_t+ε) − αλθ_t
  Stop:  halt if L_total < EARLY_STOP_THRESH = 0.05
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from taxonomy import (
    TIER1_LABELS, TIER2_LABELS,
    TIER1_TO_IDX, TIER2_TO_IDX, TIER2_MAP,
    NIHILISM_T2_IDX,
)

N_T1 = len(TIER1_LABELS)   # 3
N_T2 = len(TIER2_LABELS)   # 7

# Early-stopping threshold
EARLY_STOP_THRESH = 0.05

# ── Nihilism Penalty Weight ────────────────────────────────────────────────
# Even on a perfectly balanced dataset, Nietzsche's aphoristic, deconstructive
# vocabulary bleeds into adjacent categories (Existentialism, Stoicism).
# We apply a penalty multiplier > 1.0 to the Nihilism class in Tier-2's
# CrossEntropyLoss so the model must work harder to emit that label.
#
# How CE class_weight works:
#   loss_i = -weight[y_i] × log(p_i)
# A weight > 1.0 magnifies the gradient signal for misclassified Nihilism
# samples, forcing sharper boundaries in the embedding space.
NIHILISM_PENALTY = 1.4   # tune between 1.2–2.0; 1.4 is a conservative start

# Build the Tier-2 weight tensor: all classes = 1.0 except Nihilism
_T2_CLASS_WEIGHTS_NP = [1.0] * N_T2
_T2_CLASS_WEIGHTS_NP[NIHILISM_T2_IDX] = NIHILISM_PENALTY

# ── Masking Matrix Setup ───────────────────────────────────────────────────
# We build a constant mask matrix M of shape (N_T2, N_T1).
# M[i, j] = 1 if Tier-2 class i belongs to Tier-1 branch j, else 0.
# This matrix maps Tier-1 probabilities to the corresponding Tier-2 categories.
_mask_np = np.zeros((N_T2, N_T1), dtype=np.float32)
for t1_label, t2_list in TIER2_MAP.items():
    t1_idx = TIER1_TO_IDX[t1_label]
    for t2_label in t2_list:
        t2_idx = TIER2_TO_IDX[t2_label]
        _mask_np[t2_idx, t1_idx] = 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  1. SVM — RBF Kernel with Balanced Class Weights
# ═══════════════════════════════════════════════════════════════════════════

def train_svm(X_tfidf, y_tier1):
    """
    Fit SVM for Tier-1 classification.

    class_weight='balanced': adjusts C per class so that each class
    contributes equally regardless of its sample count:
        C_k = C × (n_samples) / (n_classes × n_samples_k)

    RBF kernel: K(x,x') = exp(−γ‖x−x'‖²)
    Decision boundary: f(x) = Σ αᵢ yᵢ K(xᵢ,x) + b
    """
    clf = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",   # ← mathematical safeguard
    )
    clf.fit(X_tfidf, y_tier1)
    acc = clf.score(X_tfidf, y_tier1)
    print(f"  [SVM] Train accuracy: {acc:.2%}")
    return clf


# ═══════════════════════════════════════════════════════════════════════════
#  2. Hierarchical MLP — Bottleneck Architecture (anti-Singularity)
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalMLP(nn.Module):
    """
    Aggressive bottleneck design to handle sparse, high-dimensional TF-IDF.

    Input(D=10k)
      → Linear(128) → GELU → Dropout(0.6)    # crush sparse noise hard
      → Linear(32)  → GELU → Dropout(0.6)    # force abstract representation
      → Head-1: Linear(32→N_T1)               # Tier-1 logits
      → Head-2: Linear(32→N_T2)               # Tier-2 logits

    Why GELU over ReLU for sparse TF-IDF?
      ReLU hard-zeros all negative activations — killing neurons that receive
      zero TF-IDF values (which is most of the vocabulary for any given text).
      GELU uses x×Φ(x), providing a smooth gradient even near zero, keeping
      sparse-input neurons alive during backpropagation.

    Why Dropout(0.6)?
      Prevents the network from routing every TF-IDF signal through the “Idealism
      highway” neurons that dominate on high word-count Kant texts. Forces all
      neurons to contribute meaningful signal, not just the dense-vocab subset.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(0.6),
            nn.Linear(128, 32),        nn.GELU(), nn.Dropout(0.6),
        )
        self.head_t1 = nn.Linear(32, N_T1)
        self.head_t2 = nn.Linear(32, N_T2)

    def forward(self, x):
        h = self.backbone(x)
        return self.head_t1(h), self.head_t2(h)


# ═══════════════════════════════════════════════════════════════════════════
#  3. Training Loop (MLP)
# ═══════════════════════════════════════════════════════════════════════════

def train_nn(
    model,
    X_tensor: torch.Tensor,
    y1_tensor: torch.Tensor,
    y2_tensor: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,        # ← raised to 64 for 21k-sample corpus
) -> nn.Module:
    """
    Train a hierarchical NN using combined cross-entropy loss.

    L_total = CE(logits_t1, y_tier1)
            + CE(logits_t2, y_tier2, weight=T2_class_weights)

    Tier-2 loss uses a class-weight vector where Nihilism = 1.4
    and all other categories = 1.0.  This penalises Nihilism over-
    confidence without requiring additional data.

    Early stopping: halt if L_total < EARLY_STOP_THRESH (0.05)
    to prevent overfitting on the balanced 21,000-sample dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # ── Tier-1 loss: standard (all 3 branches are balanced) ───────────────
    crit_t1 = nn.CrossEntropyLoss()

    # ── Tier-2 loss: weighted (Nihilism penalty = 1.4) ────────────────────
    # weight[NIHILISM_T2_IDX] = 1.4; all others = 1.0
    # CE loss: L_i = -weight[y_i] × log(softmax(logits)[y_i])
    t2_weights = torch.tensor(_T2_CLASS_WEIGHTS_NP, dtype=torch.float32, device=device)
    crit_t2 = nn.CrossEntropyLoss(weight=t2_weights)

    # ── AdamW: L2 regularisation prevents over-weighting Kant's vocabulary ─
    # weight_decay=0.02 adds L2 penalty: loss += 0.02 * ||θ||²
    # This decouples weight decay from the adaptive learning rate
    # (unlike vanilla Adam where L2 and lr interact multiplicatively).
    # Reduced lr=0.0005 prevents jumping straight into the Idealism minimum.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)

    dataset = TensorDataset(X_tensor, y1_tensor, y2_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=False)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        for xb, y1b, y2b in loader:
            xb, y1b, y2b = xb.to(device), y1b.to(device), y2b.to(device)
            logits_t1, logits_t2 = model(xb)

            # L_total = L_tier1 + L_tier2(weighted)
            loss = crit_t1(logits_t1, y1b) + crit_t2(logits_t2, y2b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(dataset)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3d}/{epochs}  L_total={avg_loss:.4f}  "
                  f"(Nihilism penalty={NIHILISM_PENALTY}×)")

        # ── Early stopping ────────────────────────────────────────────────
        if avg_loss < EARLY_STOP_THRESH:
            print(f"    [Early Stop] L_total={avg_loss:.4f} < {EARLY_STOP_THRESH}"
                  f" at epoch {epoch}. Halting.")
            break

    model.eval()
    return model


# ── Inference Helper with Hierarchical Masking ─────────────────────────────

def predict_nn(model: nn.Module, x_tensor: torch.Tensor):
    """
    Return softmax probabilities for both heads, enforcing Hierarchical Masking:
    P(T2) = P(T2_raw) * P(T1_parent)
    
    This guarantees that P(Empiricism) drops to 0 if P(Epistemology) is 0.
    """
    device = next(model.parameters()).device
    mask   = torch.tensor(_mask_np, device=device) # [N_T2, N_T1]
    
    with torch.no_grad():
        l1, l2 = model(x_tensor.to(device)) # l1: [B, N_T1], l2: [B, N_T2]
        
        # 1. Calculate base probabilities
        p1 = torch.softmax(l1, dim=1)           # P(T1)
        p2_raw = torch.softmax(l2, dim=1)       # P(T2_raw)
        
        # 2. Hierarchical Masking: P(T2) = P(T2_raw) * P(T1_parent)
        # We project P(T1) into the T2 space. 
        # Example: if mask maps Epistemology to Empiricism, 
        # p1_expanded will have the Epistemology prob at the Empiricism index.
        p1_expanded = torch.matmul(p1, mask.t()) # [B, N_T2]
        
        p2_masked = p2_raw * p1_expanded
        
        # 3. Re-normalize P(T2) so it sums to 1.0 again
        # (Since we zeroed out cross-branch probabilities, the sum dropped)
        p2_masked_sum = p2_masked.sum(dim=1, keepdim=True)
        # Avoid div-by-zero just in case
        p2_masked = p2_masked / (p2_masked_sum + 1e-10)

    return p1.cpu().numpy(), p2_masked.cpu().numpy()
