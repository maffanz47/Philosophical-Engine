"""Baseline and pro model definitions for classification/regression tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelVersion:
    """Metadata used to save timestamped model artifacts."""

    name: str
    tier: str
    extension: str = "pt"

    def filename(self) -> str:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{self.name}_{self.tier}_{timestamp}.{self.extension}"


class BaselineClassifier(nn.Module):
    """Simple 1-layer baseline classifier."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class ProClassifier(nn.Module):
    """Deep ANN classifier with ReLU and Dropout."""

    def __init__(self, input_dim: int, num_classes: int, hidden_1: int = 512, hidden_2: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ComplexityRegressor(nn.Module):
    """Single-output regressor for reading complexity score."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def save_model_checkpoint(model: nn.Module, output_dir: str, version: ModelVersion) -> str:
    """Save model state_dict with timestamped version naming."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = target_dir / version.filename()
    torch.save(model.state_dict(), artifact_path)
    return str(artifact_path)


def default_metrics_payload() -> Dict[str, float]:
    """
    Return placeholder metric keys expected by downstream orchestration.

    Real metric computation is attached during model training integration.
    """
    return {"f1_score": 0.0, "rmse": 0.0}


def _fit_classifier(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 6, lr: float = 1e-3) -> None:
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()


def _fit_regressor(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 6, lr: float = 1e-3) -> None:
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()


def train_and_evaluate_tier(
    tier: str,
    features: np.ndarray,
    labels: np.ndarray,
    complexities: np.ndarray,
    output_dir: str,
    num_classes: int,
) -> Dict[str, object]:
    """Train classifier+regressor and report F1/RMSE for one tier."""
    x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(
        features, labels, complexities, test_size=0.3, random_state=42, stratify=labels
    )

    input_dim = int(features.shape[1])
    classifier: nn.Module
    if tier == "Fast":
        classifier = BaselineClassifier(input_dim=input_dim, num_classes=num_classes)
    else:
        classifier = ProClassifier(input_dim=input_dim, num_classes=num_classes)
    regressor = ComplexityRegressor(input_dim=input_dim)

    _fit_classifier(classifier, x_train, y_train)
    _fit_regressor(regressor, x_train, c_train)

    classifier.eval()
    regressor.eval()
    with torch.no_grad():
        cls_logits = classifier(torch.tensor(x_test, dtype=torch.float32))
        cls_pred = torch.argmax(cls_logits, dim=1).cpu().numpy()
        reg_pred = regressor(torch.tensor(x_test, dtype=torch.float32)).squeeze(1).cpu().numpy()

    f1 = float(f1_score(y_test, cls_pred, average="weighted"))
    rmse = float(np.sqrt(mean_squared_error(c_test, reg_pred)))

    cls_path = save_model_checkpoint(
        model=classifier,
        output_dir=output_dir,
        version=ModelVersion(name="ANN_v1.0", tier=tier),
    )
    reg_path = save_model_checkpoint(
        model=regressor,
        output_dir=output_dir,
        version=ModelVersion(name="Regressor_v1.0", tier=tier),
    )

    return {
        "tier": tier,
        "model_path": cls_path,
        "regressor_path": reg_path,
        "f1_score": round(f1, 4),
        "rmse": round(rmse, 4),
    }
