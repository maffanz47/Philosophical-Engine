"""Deepchecks-style quality checks for integrity, leakage, and drift."""

from __future__ import annotations

from typing import Iterable, Sequence


def _avg_token_length(rows: Sequence[dict]) -> float:
    lengths = [len(str(row.get("text", "")).split()) for row in rows]
    return float(sum(lengths) / max(1, len(lengths)))


def run_deepchecks(data_rows: Iterable[dict]) -> dict:
    """
    Run lightweight validations and return a report payload.

    This is a scaffold that can be extended with full Deepchecks checks once
    labeled train/test datasets are wired into the pipeline.
    """
    rows = list(data_rows)
    null_row_count = sum(1 for row in rows if any(value is None for value in row.values()))
    unique_rows = len({tuple(sorted(row.items())) for row in rows})
    duplicate_count = max(0, len(rows) - unique_rows)

    return {
        "row_count": len(rows),
        "null_row_count": null_row_count,
        "duplicate_row_count": duplicate_count,
        "status": "pass" if null_row_count == 0 else "warning",
    }


def check_train_test_leakage(train_rows: Iterable[dict], test_rows: Iterable[dict]) -> dict:
    """Check direct overlap between train and test text fields."""
    train_set = {str(row.get("text", "")).strip() for row in train_rows}
    test_set = {str(row.get("text", "")).strip() for row in test_rows}
    intersection = train_set.intersection(test_set)
    return {
        "overlap_count": len(intersection),
        "leakage_detected": len(intersection) > 0,
    }


def check_distribution_drift(train_rows: Iterable[dict], test_rows: Iterable[dict], threshold: float = 0.25) -> dict:
    """Basic drift signal based on average token-length shift."""
    train_list = list(train_rows)
    test_list = list(test_rows)
    train_avg = _avg_token_length(train_list)
    test_avg = _avg_token_length(test_list)
    baseline = max(1.0, train_avg)
    relative_shift = abs(test_avg - train_avg) / baseline
    return {
        "train_avg_token_length": round(train_avg, 4),
        "test_avg_token_length": round(test_avg, 4),
        "relative_shift": round(relative_shift, 4),
        "drift_detected": relative_shift > threshold,
    }
