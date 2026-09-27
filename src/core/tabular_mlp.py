#!/usr/bin/env python3
"""Tabular Neural Network (MLP) with Swish, LayerNorm and Residual skips.

Specially designed for Tabular Kaggle competitions to provide diverse non-tree predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


class TabularResMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x0 = self.input_layer(x)
        x1 = self.block1(x0) + x0  # Residual skip
        x2 = self.block2(x1)
        return self.head(x2).squeeze(-1)


def train_tabular_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: str = "cpu"
) -> np.ndarray:
    """Train Tabular MLP and return validation probability predictions."""
    scaler = StandardScaler()
    X_tr_scaled = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    X_v_scaled = np.nan_to_num(scaler.transform(X_val), nan=0.0)

    train_ds = TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_tensor = torch.tensor(X_v_scaled, dtype=torch.float32).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TabularResMLP(in_features=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_preds = np.zeros(len(y_val))

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            auc = roc_auc_score(y_val, val_probs)
            if auc > best_auc:
                best_auc = auc
                best_preds = val_probs

    return best_preds
