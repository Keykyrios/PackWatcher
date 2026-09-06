"""
train/train_part_b.py — Train the TribeWorldModel (Part B: Future-Sight).

Constructs (z_history, z_future) pairs from the processed data and trains
the TCN+GRU world model with Gaussian NLL loss.

Reads:
    data/splits/train/X_train.npy  [N, feature_dim]  — combined healthy + seeded-bad
    data/splits/train/y_train.npy  [N]                — labels (used to weight loss)

World model is trained on ALL data (healthy + bad) because it must predict BOTH
normal and misaligned trajectories accurately.  The order-parameter φ is what
converts predicted trajectories into danger signals.

Outputs:
    models/world_model.pt — trained TribeWorldModel weights
    models/wm_config.json — WorldModelConfig for loading

Usage:
    python -m train.train_part_b --data-dir data --model-dir models --n-epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from packwatcher.part_b.world_model import WorldModelConfig, TribeWorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_sequence_dataset(
    X:           torch.Tensor,   # [N, feature_dim]
    history_len: int,
    horizon:     int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (z_history, z_future) pairs for sequence-to-sequence training.

    z_history[i] = X[i : i+history_len]         [history_len, feature_dim]
    z_future[i]  = X[i+history_len : i+history_len+horizon]  [horizon, feature_dim]

    Returns (histories, futures) tensors.
    """
    N          = len(X)
    window     = history_len + horizon
    n_samples  = N - window + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough samples ({N}) for history_len={history_len} + horizon={horizon}={window}"
        )

    histories: list[torch.Tensor] = []
    futures:   list[torch.Tensor] = []

    for i in range(n_samples):
        histories.append(X[i : i + history_len])
        futures.append(X[i + history_len : i + window])

    return torch.stack(histories, dim=0), torch.stack(futures, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Part B: World Model")
    parser.add_argument("--data-dir",    default="data",   help="Root data directory")
    parser.add_argument("--model-dir",   default="models", help="Output model directory")
    parser.add_argument("--n-epochs",    type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--horizon",     type=int, default=5)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    data_path  = Path(args.data_dir)

    # --- Load data ---
    train_X_path = data_path / "splits" / "train" / "X_train.npy"
    val_X_path   = data_path / "splits" / "val"   / "X_val.npy"

    if not train_X_path.exists():
        log.error("X_train.npy not found. Run sim/data_gen.py first.")
        return

    train_X = torch.from_numpy(np.load(str(train_X_path))).float()
    val_X   = torch.from_numpy(np.load(str(val_X_path))).float()
    feature_dim = train_X.shape[1]
    log.info("Loaded train=%d  val=%d  feature_dim=%d", len(train_X), len(val_X), feature_dim)

    # --- Build sequence dataset ---
    train_hist, train_fut = _build_sequence_dataset(train_X, args.history_len, args.horizon)
    val_hist,   val_fut   = _build_sequence_dataset(val_X,   args.history_len, args.horizon)
    log.info("Sequence dataset: train=%d  val=%d windows", len(train_hist), len(val_hist))

    train_ds  = TensorDataset(train_hist, train_fut)
    val_ds    = TensorDataset(val_hist,   val_fut)
    train_dl  = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_dl    = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # --- Build world model ---
    wm_config = WorldModelConfig(
        z_dim          = feature_dim,
        history_len    = args.history_len,
        horizon        = args.horizon,
        tcn_channels   = 64,
        tcn_n_layers   = 2,
        gru_hidden_dim = 128,
        n_gru_layers   = 2,
        dropout        = 0.1,
    )
    model     = TribeWorldModel(wm_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    log.info("Training world model for %d epochs ...", args.n_epochs)
    best_val  = float("inf")
    best_path = str(model_path / "world_model.pt")

    for epoch in range(1, args.n_epochs + 1):
        # --- Train ---
        model.train()
        train_losses: list[float] = []
        for zh, zf in train_dl:
            zh, zf = zh.to(device), zf.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(zh, zf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Val ---
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for zh, zf in val_dl:
                zh, zf = zh.to(device), zf.to(device)
                val_losses.append(model.compute_loss(zh, zf).item())

        scheduler.step()

        train_l = float(np.mean(train_losses))
        val_l   = float(np.mean(val_losses)) if val_losses else float("inf")

        if epoch % 5 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  train_nll=%.4f  val_nll=%.4f", epoch, args.n_epochs, train_l, val_l)

        if val_l < best_val:
            best_val = val_l
            torch.save(model.state_dict(), best_path)

    log.info("Best val NLL = %.4f  saved to %s", best_val, best_path)

    # --- Save config ---
    wm_config_path = str(model_path / "wm_config.json")
    with open(wm_config_path, "w") as f:
        json.dump({
            "z_dim":          wm_config.z_dim,
            "history_len":    wm_config.history_len,
            "horizon":        wm_config.horizon,
            "tcn_channels":   wm_config.tcn_channels,
            "tcn_n_layers":   wm_config.tcn_n_layers,
            "gru_hidden_dim": wm_config.gru_hidden_dim,
            "n_gru_layers":   wm_config.n_gru_layers,
            "dropout":        wm_config.dropout,
        }, f, indent=2)
    log.info("Saved WM config to %s", wm_config_path)
    log.info("Part B training complete.")


if __name__ == "__main__":
    main()
