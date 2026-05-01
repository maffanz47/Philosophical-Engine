from pathlib import Path

from workflows.main_flow import FlowConfig, reporting_task


def test_reporting_task_writes_artifacts(tmp_path: Path) -> None:
    processed_corpus = {
        1: [["virtue", "discipline", "nature"], ["existence", "freedom", "choice"]],
        2: [["reason", "revelation", "truth"]],
    }
    training = {
        "baseline": {"f1_score": 0.5, "rmse": 1.2},
        "pro": {"f1_score": 0.7, "rmse": 1.0},
    }
    config = FlowConfig(book_ids=[1, 2], reports_output_dir=str(tmp_path))

    result = reporting_task.fn(processed_corpus, training, config)

    assert result["metrics_path"] is not None
    assert Path(result["metrics_path"]).exists()
    assert Path(result["cluster_path"]).exists()
    assert Path(result["pca_path"]).exists()
