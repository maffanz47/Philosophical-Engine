import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class PhiloClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_model(X_train, y_train, X_val, y_val,
               num_classes, save_path, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = PhiloClassifier(X_train.shape[1], num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.LongTensor(y_val).to(device)

    best_val_loss, patience_counter = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
            val_acc  = (model(Xv).argmax(1) == yv).float().mean().item()

        print(f"Epoch {epoch+1:3d} | loss {loss.item():.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stop at epoch {epoch+1}")
                break

    print(f"✓ Best model saved to {save_path}")
    return model

def train_sklearn_models(X_train, y_train, X_val, y_val, save_dir):
    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    val_acc_lr = logreg.score(X_val, y_val)
    print(f"Logistic Regression val_acc: {val_acc_lr:.3f}")
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    val_acc_rf = rf.score(X_val, y_val)
    print(f"Random Forest val_acc: {val_acc_rf:.3f}")
    
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(logreg, os.path.join(save_dir, "logreg.pkl"))
    joblib.dump(rf, os.path.join(save_dir, "rf.pkl"))
    print(f"✓ Scikit-learn models saved to {save_dir}")
    
    return logreg, rf