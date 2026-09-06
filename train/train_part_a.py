"""
train/train_part_a.py — Train the Sparse Autoencoder and fit the healthy centroid.

Phase 0 outputs:
    models/sae.pt                  — trained SAE weights
    models/order_param_centroid.npy — healthy-tribe centroid for order parameter tracker

Reads:
    data/processed/healthy_z.npy  — healthy tribe-state feature vectors [N, feature_dim]

Training SAE on healthy data teaches it to extract interpretable sparse features
from the agent activation stream. The healthy-data centroid becomes the reference
for φ computation in Part B.

Usage:
    python -m train.train_part_a --data-dir data --model-dir models --n-epochs 30
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from packwatcher.part_a.sae import SAEConfig, SparseAutoencoder, train_sae
from packwatcher.part_b.order_parameter import fit_healthy_centroid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Part A: SAE + healthy centroid")
    parser.add_argument("--data-dir",   default="data",   help="Root data directory")
    parser.add_argument("--model-dir",  default="models", help="Output model directory")
    parser.add_argument("--n-epochs",   type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    data_path  = Path(args.data_dir)

    # --- Load data ---
    healthy_z_path = data_path / "processed" / "healthy_z.npy"
    if not healthy_z_path.exists():
        log.error("healthy_z.npy not found at %s. Run sim/data_gen.py first.", healthy_z_path)
        return

    healthy_z = torch.from_numpy(np.load(str(healthy_z_path))).float()
    feature_dim = healthy_z.shape[1]
    log.info("Loaded %d healthy z-vectors, feature_dim=%d", len(healthy_z), feature_dim)

    # --- Configure and train SAE ---
    sae_config = SAEConfig(
        input_dim  = feature_dim,
        hidden_dim = feature_dim * 4,   # 4× over-complete dictionary
        l1_coeff   = 0.01,
        top_k      = None,              # Use L1 sparsity
    )
    sae = SparseAutoencoder(sae_config).to(device)

    dataset    = TensorDataset(healthy_z)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer  = torch.optim.Adam(sae.parameters(), lr=args.lr)

    log.info("Training SAE for %d epochs ...", args.n_epochs)
    history = train_sae(sae, dataloader, optimizer, n_epochs=args.n_epochs, device=device)
    log.info(
        "SAE trained. Final loss: recon=%.4f  sparsity=%.4f  total=%.4f",
        history[-1]["recon_loss"],
        history[-1]["sparsity_loss"],
        history[-1]["total_loss"],
    )

    # --- Save SAE ---
    sae_path = str(model_path / "sae.pt")
    torch.save(sae.state_dict(), sae_path)
    log.info("Saved SAE to %s", sae_path)

    # --- Fit healthy centroid ---
    healthy_z_dev = healthy_z.to(device)
    with torch.no_grad():
        # Use SAE encoded features (sparse representation) as z for centroid
        encoded_features = sae.encode(healthy_z_dev)          # [N, hidden_dim]

    centroid = fit_healthy_centroid(encoded_features.cpu())   # [hidden_dim]
    centroid_path = str(model_path / "order_param_centroid.npy")
    np.save(centroid_path, centroid.numpy())
    log.info("Saved healthy centroid to %s (dim=%d)", centroid_path, centroid.shape[0])

    # --- Save SAE config for downstream use ---
    import json
    config_path = str(model_path / "sae_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "input_dim":  sae_config.input_dim,
            "hidden_dim": sae_config.hidden_dim,
            "l1_coeff":   sae_config.l1_coeff,
            "top_k":      sae_config.top_k,
        }, f, indent=2)
    log.info("Saved SAE config to %s", config_path)
    log.info("Part A training complete.")


if __name__ == "__main__":
    main()
