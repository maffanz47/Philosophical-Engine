"""
notify.py — Post-Training Notification Module for the Philosophical Text Engine.

Responsibilities:
  1. Evaluate all 4 trained models (SVM Slow, SVM Fast, K-Means Slow, K-Means Fast)
     and produce a structured metrics report.
  2. Send the report via Discord Webhook (primary channel).
  3. Send the report via legacy SMTP email (secondary / backup channel).

Environment variables (loaded from .env via python-dotenv or os.environ):
  DISCORD_WEBHOOK_URL  — Full Discord webhook URL.
  NOTIFY_EMAIL_FROM    — Sender address (SMTP).
  NOTIFY_EMAIL_TO      — Recipient address (comma-separated list).
  NOTIFY_SMTP_HOST     — SMTP server host (default: smtp.gmail.com).
  NOTIFY_SMTP_PORT     — SMTP server port (default: 587).
  NOTIFY_SMTP_PASSWORD — SMTP password / app-password.

Usage (called automatically by train.py after engine.save()):
    from notify import send_training_report
    send_training_report(engine, X_tfidf, y_tier1, y_tier2)
"""

from __future__ import annotations

import os
import json
import smtplib
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone

import numpy as np
import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ── Helper: load env with optional python-dotenv support ──────────────────────

def _getenv(key: str, default: str = "") -> str:
    """Read from os.environ; optionally fall back to .env file via dotenv."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"),
                    override=False)
    except ImportError:
        pass  # dotenv not installed — rely on os.environ set by the shell
    return os.environ.get(key, default)


# ── Metrics Computation ───────────────────────────────────────────────────────

def _svm_metrics(model, X, y_true: np.ndarray) -> dict:
    """Compute classification metrics for an SVM model."""
    y_pred = model.predict(X)
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred,
                                                  average="macro",
                                                  zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred,
                                              average="macro",
                                              zero_division=0)), 4),
        "f1_macro":  round(float(f1_score(y_true, y_pred,
                                          average="macro",
                                          zero_division=0)), 4),
        "n_support_vectors": int(sum(model.n_support_)),
    }


def _kmeans_metrics(model, X) -> dict:
    """
    Compute unsupervised quality metrics for a K-Means model.

    • Inertia (WCSS)  — lower is better (tighter clusters).
    • n_clusters      — sanity check that clusters match requested k.
    """
    labels = model.predict(X)
    inertia = float(model.inertia_)
    unique_labels = len(set(labels.tolist()))
    return {
        "inertia":   round(inertia, 2),
        "n_clusters": unique_labels,
    }


def compute_all_metrics(engine, X_tfidf, y_tier1: np.ndarray) -> dict:
    """
    Evaluate all four trained models against the full training set:
      - SVM Slow   (full data, Tier-1 classifier)
      - SVM Fast   (25% data, Tier-1 classifier)
      - K-Means Slow (full data)
      - K-Means Fast (25% data)

    Args:
        engine:   A loaded/trained PhilosophyEngine instance.
        X_tfidf:  Sparse TF-IDF matrix (full training set).
        y_tier1:  Tier-1 integer labels array.

    Returns:
        dict with keys: svm_slow, svm_fast, kmeans_slow, kmeans_fast, timestamp
    """
    print("\n  [notify] Computing evaluation metrics for all 4 models …")

    metrics = {
        "svm_slow":    _svm_metrics(engine.svm_slow,    X_tfidf, y_tier1),
        "svm_fast":    _svm_metrics(engine.svm_fast,    X_tfidf, y_tier1),
        "kmeans_slow": _kmeans_metrics(engine.kmeans_slow, X_tfidf),
        "kmeans_fast": _kmeans_metrics(engine.kmeans_fast, X_tfidf),
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_samples":   int(X_tfidf.shape[0]),
    }

    print(f"  [notify] SVM Slow   — Accuracy: {metrics['svm_slow']['accuracy']:.2%}  "
          f"F1: {metrics['svm_slow']['f1_macro']:.4f}")
    print(f"  [notify] SVM Fast   — Accuracy: {metrics['svm_fast']['accuracy']:.2%}  "
          f"F1: {metrics['svm_fast']['f1_macro']:.4f}")
    print(f"  [notify] K-Means Slow — Inertia: {metrics['kmeans_slow']['inertia']:,.1f}  "
          f"Clusters: {metrics['kmeans_slow']['n_clusters']}")
    print(f"  [notify] K-Means Fast — Inertia: {metrics['kmeans_fast']['inertia']:,.1f}  "
          f"Clusters: {metrics['kmeans_fast']['n_clusters']}")

    return metrics


# ── Report Formatting ─────────────────────────────────────────────────────────

def _format_discord_embed(metrics: dict) -> dict:
    """
    Build a rich Discord embed payload containing all model metrics.
    Discord Embed docs: https://discord.com/developers/docs/resources/channel#embed-object
    """
    ss = metrics["svm_slow"]
    sf = metrics["svm_fast"]
    ks = metrics["kmeans_slow"]
    kf = metrics["kmeans_fast"]

    def bar(value: float, width: int = 10) -> str:
        """ASCII progress bar for a 0-1 accuracy value."""
        filled = round(value * width)
        return "█" * filled + "░" * (width - filled)

    fields = [
        # ── SVM Slow ──────────────────────────────────────────────
        {
            "name": "🧮 SVM — Thinking (Slow)  *(full dataset)*",
            "value": (
                f"```\n"
                f"Accuracy  {bar(ss['accuracy'])}  {ss['accuracy']:.2%}\n"
                f"Precision {bar(ss['precision'])} {ss['precision']:.4f}\n"
                f"Recall    {bar(ss['recall'])}    {ss['recall']:.4f}\n"
                f"F1 Macro  {bar(ss['f1_macro'])}  {ss['f1_macro']:.4f}\n"
                f"Support Vectors: {ss['n_support_vectors']:,}\n"
                f"```"
            ),
            "inline": False,
        },
        # ── SVM Fast ──────────────────────────────────────────────
        {
            "name": "⚡ SVM — Fast  *(25% subset)*",
            "value": (
                f"```\n"
                f"Accuracy  {bar(sf['accuracy'])}  {sf['accuracy']:.2%}\n"
                f"Precision {bar(sf['precision'])} {sf['precision']:.4f}\n"
                f"Recall    {bar(sf['recall'])}    {sf['recall']:.4f}\n"
                f"F1 Macro  {bar(sf['f1_macro'])}  {sf['f1_macro']:.4f}\n"
                f"Support Vectors: {sf['n_support_vectors']:,}\n"
                f"```"
            ),
            "inline": False,
        },
        # ── K-Means Slow ──────────────────────────────────────────
        {
            "name": "🔵 K-Means — Thinking (Slow)  *(full dataset)*",
            "value": (
                f"```\n"
                f"Inertia (WCSS): {ks['inertia']:>12,.1f}  (lower = tighter)\n"
                f"Unique Clusters: {ks['n_clusters']}\n"
                f"```"
            ),
            "inline": True,
        },
        # ── K-Means Fast ──────────────────────────────────────────
        {
            "name": "🟡 K-Means — Fast  *(25% subset)*",
            "value": (
                f"```\n"
                f"Inertia (WCSS): {kf['inertia']:>12,.1f}  (lower = tighter)\n"
                f"Unique Clusters: {kf['n_clusters']}\n"
                f"```"
            ),
            "inline": True,
        },
        # ── Dataset summary ───────────────────────────────────────
        {
            "name": "📊 Dataset",
            "value": f"`{metrics['n_samples']:,}` training samples",
            "inline": True,
        },
    ]

    return {
        "embeds": [
            {
                "title": "✅  Philosophical Engine — Training Complete",
                "description": (
                    f"All 4 models trained and saved successfully.\n"
                    f"**Timestamp:** `{metrics['timestamp']}`"
                ),
                "color": 0xF5B301,   # Amber Gold — project accent colour
                "fields": fields,
                "footer": {
                    "text": "Philosophical Text Engine · ML Pipeline v5"
                },
            }
        ]
    }


def _format_email_body(metrics: dict) -> str:
    """Plain-text email body with the same metrics data."""
    ss = metrics["svm_slow"]
    sf = metrics["svm_fast"]
    ks = metrics["kmeans_slow"]
    kf = metrics["kmeans_fast"]

    return textwrap.dedent(f"""\
    ╔══════════════════════════════════════════════════════════╗
      Philosophical Engine — Training Complete
      {metrics['timestamp']}
    ╚══════════════════════════════════════════════════════════╝

    Training samples: {metrics['n_samples']:,}

    ── SVM Thinking (Slow)  [full dataset] ─────────────────────
      Accuracy:        {ss['accuracy']:.2%}
      Precision (macro): {ss['precision']:.4f}
      Recall (macro):    {ss['recall']:.4f}
      F1 Macro:          {ss['f1_macro']:.4f}
      Support Vectors:   {ss['n_support_vectors']:,}

    ── SVM Fast  [25% subset] ───────────────────────────────────
      Accuracy:        {sf['accuracy']:.2%}
      Precision (macro): {sf['precision']:.4f}
      Recall (macro):    {sf['recall']:.4f}
      F1 Macro:          {sf['f1_macro']:.4f}
      Support Vectors:   {sf['n_support_vectors']:,}

    ── K-Means Thinking (Slow)  [full dataset] ─────────────────
      Inertia (WCSS):  {ks['inertia']:,.1f}
      Unique Clusters: {ks['n_clusters']}

    ── K-Means Fast  [25% subset] ──────────────────────────────
      Inertia (WCSS):  {kf['inertia']:,.1f}
      Unique Clusters: {kf['n_clusters']}

    ── Notes ────────────────────────────────────────────────────
      • SVM metrics are evaluated on the FULL training set for
        both Slow and Fast variants (to allow apples-to-apples
        comparison of the accuracy penalty of using 25% data).
      • K-Means inertia: lower = tighter, more cohesive clusters.

    Philosophical Text Engine · ML Pipeline v5
    """)


# ── Transport: Discord ────────────────────────────────────────────────────────

def send_discord(metrics: dict) -> bool:
    """
    POST the metrics embed to the Discord webhook.

    Returns True on success, False on any error (non-fatal — training
    artifacts are already saved before this is called).
    """
    webhook_url = _getenv("discord_webhook_url") or _getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("  [notify] ⚠  DISCORD_WEBHOOK_URL not set — skipping Discord.")
        return False

    payload = _format_discord_embed(metrics)
    try:
        resp = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if resp.status_code in (200, 204):
            print(f"  [notify] ✓  Discord notification sent (HTTP {resp.status_code}).")
            return True
        else:
            print(f"  [notify] ✗  Discord returned HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.RequestException as exc:
        print(f"  [notify] ✗  Discord request failed: {exc}")
        return False


# ── Transport: Legacy Email (SMTP) ────────────────────────────────────────────

def send_email(metrics: dict) -> bool:
    """
    Send the metrics report via SMTP (legacy notification channel).

    Reads credentials from environment variables:
      NOTIFY_EMAIL_FROM     — sender address
      NOTIFY_EMAIL_TO       — recipient(s), comma-separated
      NOTIFY_SMTP_HOST      — default: smtp.gmail.com
      NOTIFY_SMTP_PORT      — default: 587  (STARTTLS)
      NOTIFY_SMTP_PASSWORD  — SMTP password / app-password

    Returns True on success, False on any error.
    """
    from_addr  = _getenv("NOTIFY_EMAIL_FROM")
    to_raw     = _getenv("NOTIFY_EMAIL_TO")
    smtp_host  = _getenv("NOTIFY_SMTP_HOST",     "smtp.gmail.com")
    smtp_port  = int(_getenv("NOTIFY_SMTP_PORT", "587"))
    smtp_pass  = _getenv("NOTIFY_SMTP_PASSWORD")

    if not (from_addr and to_raw and smtp_pass):
        print("  [notify] ⚠  Email credentials incomplete — skipping email.")
        return False

    to_addrs = [a.strip() for a in to_raw.split(",") if a.strip()]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = (
        f"[Phil-Engine] Training Complete — "
        f"SVM Slow {metrics['svm_slow']['accuracy']:.1%} acc · "
        f"{metrics['timestamp']}"
    )
    msg["From"]    = from_addr
    msg["To"]      = ", ".join(to_addrs)

    body = _format_email_body(metrics)
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.login(from_addr, smtp_pass)
            server.sendmail(from_addr, to_addrs, msg.as_string())
        print(f"  [notify] ✓  Email sent to {to_addrs}.")
        return True
    except Exception as exc:
        print(f"  [notify] ✗  Email failed: {exc}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def send_training_report(engine, X_tfidf, y_tier1) -> None:
    """
    Master entry point called by train.py after engine.save().

    Steps:
      1. Compute metrics for all 4 models on the full training set.
      2. Fire Discord webhook.
      3. Fire legacy SMTP email.

    Both transports are independent — a failure in one does not
    block the other. Training artifacts are unaffected regardless.

    Args:
        engine:   Fully trained PhilosophyEngine (svm_slow/fast,
                  kmeans_slow/fast must all be fitted).
        X_tfidf:  Sparse TF-IDF matrix — full training corpus.
        y_tier1:  Tier-1 integer label array (numpy or list).
    """
    print("\n" + "=" * 60)
    print("  POST-TRAINING NOTIFICATIONS")
    print("=" * 60)

    y1 = np.array(y_tier1)
    metrics = compute_all_metrics(engine, X_tfidf, y1)

    discord_ok = send_discord(metrics)
    email_ok   = send_email(metrics)

    print()
    if discord_ok or email_ok:
        print("  [notify] ✓  At least one notification channel succeeded.")
    else:
        print("  [notify] ⚠  All notification channels failed or unconfigured.")
    print("=" * 60 + "\n")
