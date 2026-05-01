"""
deepchecks_suite.py
===================
Deepchecks test suite for Data Integrity, Train-Test Leakage, and Distribution Drift.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path so we can import modules
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend" / "src"))

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("deepchecks")

PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_checks():
    logger.info("Starting Deepchecks validation suite...")

    corpus_path = PROCESSED_DIR / "corpus.csv"
    if not corpus_path.exists():
        logger.error(f"Corpus file not found: {corpus_path}. Run pipeline first.")
        return

    df = pd.read_csv(corpus_path)
    logger.info(f"Loaded corpus with {len(df)} rows.")

    # Select features for deepchecks (numeric and categorical, drop high cardinality strings)
    features = ["complexity_score", "fk_grade", "avg_sent_len", "lexical_div"]
    cat_features = ["book_id", "author"]
    label = "school"

    check_df = df[features + cat_features + [label]].copy()
    
    # 1. Data Integrity Suite
    logger.info("Running Data Integrity Suite...")
    ds = Dataset(check_df, label=label, cat_features=cat_features)
    integ_suite = data_integrity()
    integ_result = integ_suite.run(ds)
    integ_result.save_as_html(str(REPORT_DIR / "data_integrity_report.html"))
    logger.info(f"Data Integrity Report saved to {REPORT_DIR / 'data_integrity_report.html'}")

    # 2. Train-Test Validation Suite (Leakage & Drift)
    logger.info("Running Train-Test Validation Suite...")
    train_df, test_df = train_test_split(check_df, test_size=0.2, random_state=42, stratify=check_df[label])
    
    train_ds = Dataset(train_df, label=label, cat_features=cat_features)
    test_ds = Dataset(test_df, label=label, cat_features=cat_features)

    val_suite = train_test_validation()
    val_result = val_suite.run(train_ds, test_ds)
    val_result.save_as_html(str(REPORT_DIR / "train_test_validation_report.html"))
    logger.info(f"Train-Test Validation Report saved to {REPORT_DIR / 'train_test_validation_report.html'}")

    logger.info("Deepchecks validation complete.")

if __name__ == "__main__":
    run_checks()
