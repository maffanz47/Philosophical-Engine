import os
import time
from pathlib import Path
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation, model_evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("data/processed/philosophy_corpus.csv")
REPORTS_DIR = Path("reports/deepchecks")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

@pytest.mark.skipif(not DATA_PATH.exists(), reason="Data not available yet")
def test_data_integrity():
    df = pd.read_csv(DATA_PATH)
    
    # Ensure sufficient rows for meaningful deepchecks
    if len(df) < 50:
        pytest.skip("Not enough data to run deepchecks properly.")
        
    ds = Dataset(df, label='school_label', cat_features=['era_label', 'author'])
    
    integ_suite = data_integrity()
    result = integ_suite.run(ds)
    
    timestamp = int(time.time())
    result.save_as_html(str(REPORTS_DIR / f"integrity_report_{timestamp}.html"))
    
    # Check for critical errors (e.g., class imbalance > 5:1)
    # Deepchecks results structure can be inspected; here we assert on the suite run passing its own conditions.
    assert result.passed(fail_if_warning=False), "Data Integrity Suite Failed"

@pytest.mark.skipif(not DATA_PATH.exists(), reason="Data not available yet")
def test_train_test_split():
    df = pd.read_csv(DATA_PATH).dropna(subset=['school_label', 'avg_sentence_length'])
    if len(df) < 50:
        pytest.skip("Not enough data.")
        
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_ds = Dataset(train_df, label='school_label', cat_features=['era_label'])
    test_ds = Dataset(test_df, label='school_label', cat_features=['era_label'])
    
    tt_suite = train_test_validation()
    result = tt_suite.run(train_ds, test_ds)
    
    timestamp = int(time.time())
    result.save_as_html(str(REPORTS_DIR / f"train_test_report_{timestamp}.html"))

@pytest.mark.skipif(not DATA_PATH.exists(), reason="Data not available yet")
def test_model_performance():
    df = pd.read_csv(DATA_PATH).dropna(subset=['school_label', 'avg_sentence_length', 'vocab_richness', 'sentiment_polarity'])
    if len(df) < 50:
        pytest.skip("Not enough data.")
        
    le = LabelEncoder()
    df['school_encoded'] = le.fit_transform(df['school_label'])
    
    features = ['avg_sentence_length', 'vocab_richness', 'sentiment_polarity']
    X = df[features]
    y = df['school_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    train_ds = Dataset(pd.concat([X_train, y_train], axis=1), label='school_encoded')
    test_ds = Dataset(pd.concat([X_test, y_test], axis=1), label='school_encoded')
    
    perf_suite = model_evaluation()
    result = perf_suite.run(train_ds, test_ds, model)
    
    timestamp = int(time.time())
    result.save_as_html(str(REPORTS_DIR / f"performance_report_{timestamp}.html"))
