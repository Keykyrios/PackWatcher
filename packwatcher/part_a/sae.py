"""
Part A — Sparse Autoencoder (SAE) for per-agent feature extraction.

Borrows architecture from mechanistic-interpretability SAE literature
(TopK variant from Anthropic/EleutherAI style SAEs).

Two sparsity modes:
  L1 mode  — l1_coeff penalises mean absolute activation (top_k=None)
  TopK mode — hard top-k masking, exactly k features active per sample

API contract (used by tests + train_part_a.py):
  sae(x)             → (x_hat, z)        2-tuple
  sae.encode(x)      → z                 sparse code
  sae.decode(z)      → x_hat             reconstruction
  sae.loss(x)        → scalar tensor     reconstruction + sparsity
  sae.normalize_decoder()                project decoder to unit norm
  train_sae(sae, dataloader, optimizer, n_epochs, device)  → list[dict]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class SAEConfig:
    input_dim: int
    hidden_dim: int
    l1_coeff: float = 1e-3
    top_k: Optional[int] = None   # If set, use TopK sparsity instead of L1


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder: encoder maps input → sparse feature code; decoder reconstructs.

    Training objective:
        L = MSE(x_hat, x) + l1_coeff * ||z||_1   (L1 mode)
        L = MSE(x_hat, x)                          (TopK mode; sparsity is structural)

    Decoder columns are kept at unit norm after each step via normalize_decoder().
    """

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse code z.  Shape: [..., hidden_dim]."""
        pre = self.encoder(x)
        if self.config.top_k is not None:
            return self._topk_activation(pre)
        return F.relu(pre)

    def _topk_activation(self, pre: torch.Tensor) -> torch.Tensor:
        k = min(self.config.top_k, pre.shape[-1])
        topk_vals, topk_idx = torch.topk(pre, k, dim=-1)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_idx, 1.0)
        return F.relu(pre) * mask

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from sparse code. Shape: [..., input_dim]."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [..., input_dim]
        Returns:
            x_hat: reconstructed input [..., input_dim]
            z:     sparse code          [..., hidden_dim]
        """
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction + sparsity loss for a batch x."""
        x_hat, z = self(x)
        recon_loss    = F.mse_loss(x_hat, x)
        sparsity_loss = self.config.l1_coeff * z.abs().mean()
        return recon_loss + sparsity_loss

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder columns back to unit norm. Call after every optimizer step."""
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    @property
    def feature_dim(self) -> int:
        return self.config.hidden_dim


# ---------------------------------------------------------------------------
# Convenience training loop — accepts a DataLoader
# ---------------------------------------------------------------------------

def train_sae(
    sae:        SparseAutoencoder,
    dataloader,                      # torch DataLoader yielding (x,) or x batches
    optimizer:  torch.optim.Optimizer,
    n_epochs:   int = 10,
    device:     str = "cpu",
) -> list[dict]:
    """
    Train a SparseAutoencoder using the provided DataLoader and optimizer.

    Returns a list of per-epoch statistics dicts:
        [{"recon_loss": float, "sparsity_loss": float, "total_loss": float, "mean_l0": float}, ...]
    """
    sae = sae.to(device)
    history: list[dict] = []

    for epoch in range(n_epochs):
        sae.train()
        ep_recon = ep_sparse = ep_total = ep_l0 = 0.0
        n_batches = 0

        for batch in dataloader:
            # DataLoader may yield a tuple (x,) or a plain tensor x
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)
            x_hat, z = sae(x)
            recon_loss    = F.mse_loss(x_hat, x)
            sparsity_loss = sae.config.l1_coeff * z.abs().mean()
            total_loss    = recon_loss + sparsity_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            sae.normalize_decoder()

            with torch.no_grad():
                l0 = (z.detach() > 0).float().sum(dim=-1).mean().item()

            ep_recon  += recon_loss.item()
            ep_sparse += sparsity_loss.item()
            ep_total  += total_loss.item()
            ep_l0     += l0
            n_batches += 1

        n_batches = max(n_batches, 1)
        history.append({
            "recon_loss":    ep_recon  / n_batches,
            "sparsity_loss": ep_sparse / n_batches,
            "total_loss":    ep_total  / n_batches,
            "mean_l0":       ep_l0     / n_batches,
        })

    return history
