"""
train.py — Model Training Script

Run this script ONCE to scrape the Gutenberg texts, train all models
(SVM, MLP, K-Means, PCA), and save the artifacts to disk.

Post-training steps (automatic):
  • Evaluate all 4 models (SVM Slow/Fast, K-Means Slow/Fast).
  • Send a metrics report via Discord Webhook.
  • Send the same report via legacy SMTP email.

Environment variables required for notifications (set in .env or shell):
  discord_webhook_url   — Discord Webhook URL.
  NOTIFY_EMAIL_FROM     — Sender email address.
  NOTIFY_EMAIL_TO       — Recipient email address(es), comma-separated.
  NOTIFY_SMTP_HOST      — SMTP host (default: smtp.gmail.com).
  NOTIFY_SMTP_PORT      — SMTP port (default: 587).
  NOTIFY_SMTP_PASSWORD  — SMTP password / app-password.
"""

import numpy as np
from engine_core import PhilosophyEngine
from notify import send_training_report

if __name__ == "__main__":
    print("Initializing Philosophy Engine Training Pipeline...")
    engine = PhilosophyEngine()

    # ── Phase 1–3: ingest, preprocess, validate ────────────────────────────
    engine.ingest()
    engine.preprocess()
    engine.validate()

    # ── Phase 4–5: train supervised + unsupervised models ─────────────────
    engine.train_supervised(nn_epochs=50, batch_size=64)
    engine.fit_unsupervised()

    print("\n" + "=" * 60)
    print("  ✓  BUILD COMPLETE — engine ready.")
    print("=" * 60)

    # ── Persist all model artifacts to disk ────────────────────────────────
    engine.save(path="engine_artifacts")

    # ── Post-training: evaluate metrics + send notifications ───────────────
    # We pass the full TF-IDF matrix and Tier-1 labels (already in memory)
    # so notify.py can compute accuracy / F1 without re-loading anything.
    y_tier1 = np.array(engine.y_t1)
    send_training_report(engine, engine.X_tfidf, y_tier1)

