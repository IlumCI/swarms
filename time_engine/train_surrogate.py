#!/usr/bin/env python3
"""
Train a tiny surrogate MLP on data generated from PDEAdapter2DCpu.
Produces a small model file`.
"""
import numpy as np
import json
import os
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None

from engine import SimulacrumEngine

def make_dataset(n_samples=100, grid_size=(4,4)):
    eng = SimulacrumEngine(db_path=":memory:")
    adapter = eng.PDEAdapter2DCpu({"alpha":1e-3})
    X = []
    Y = []
    for _ in range(n_samples):
        grid = np.random.randn(*grid_size)
        out = adapter.step(grid, 0.1, {"alpha":1e-3})
        X.append(grid.flatten())
        Y.append(np.asarray(out).flatten())
    return np.array(X), np.array(Y)

def train_and_save(model_path="surrogate.pt"):
    if torch is None:
        raise RuntimeError("PyTorch is required to train surrogate")
    X, Y = make_dataset(200, (4,4))
    inp = X.shape[1]
    model = nn.Sequential(nn.Linear(inp, 64), nn.ReLU(), nn.Linear(64, inp))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    for epoch in range(20):
        optimizer.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, Y_t)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)

if __name__ == "__main__":
    train_and_save()


