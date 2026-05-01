import numpy as np

import workflows.main_flow as main_flow
from tests.deepchecks_suite import check_distribution_drift, check_train_test_leakage
from workflows.main_flow import FlowConfig, training_and_evaluation_task


def test_training_task_returns_real_metrics(monkeypatch, tmp_path) -> None:
    processed_corpus = {
        1: [["virtue", "discipline"], ["stoic", "ethics"], ["tranquility", "reason"]],
        2: [["freedom", "choice"], ["anguish", "existence"], ["meaning", "absurd"]],
    }

    def fake_distilbert_features(texts):  # type: ignore[no-untyped-def]
        rows = len(texts)
        return np.ones((rows, 8), dtype=np.float32)

    monkeypatch.setattr(main_flow, "distilbert_features", fake_distilbert_features)

    config = FlowConfig(book_ids=[1, 2], model_output_dir=str(tmp_path), num_classes=2, tfidf_max_features=32)
    result = training_and_evaluation_task.fn(processed_corpus, config)

    assert "baseline" in result
    assert "pro" in result
    assert "f1_score" in result["baseline"]
    assert "rmse" in result["pro"]
    assert result["baseline"]["model_path"]
    assert result["pro"]["regressor_path"]


def test_deepchecks_leakage_and_drift_signals() -> None:
    train_rows = [{"text": "virtue and reason"}, {"text": "discipline and ethics"}]
    test_rows = [{"text": "virtue and reason"}, {"text": "freedom and absurdity"}]
    leakage = check_train_test_leakage(train_rows, test_rows)
    assert leakage["leakage_detected"] is True

    drift = check_distribution_drift(
        [{"text": "short text"}],
        [{"text": "this is a much longer shifted distribution sample"}],
        threshold=0.2,
    )
    assert drift["drift_detected"] is True
