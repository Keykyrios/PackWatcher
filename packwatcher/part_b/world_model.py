"""
Part B — Predictive tribe world model.

Architecture:
  TCNEncoder: causal temporal convolutions over z-history → context vector
  GRU rollout: autoregressive multi-step prediction from context
  Gaussian output head: mean + log_var per predicted step

The model produces a *distribution* over future tribe-states, enabling:
  1. Branched sampling (n_samples trajectories) for uncertainty quantification
  2. NLL training loss (proper probabilistic objective)
  3. Calibrated time-to-tipping-point estimates via OrderParameterTracker

Design follows VLA-MBPO / Dreamer-style predictive latent rollout,
adapted for alignment-state rather than physical state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from packwatcher.types import PredictedTrajectory


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    z_dim: int          = 512   # tribe-state vector dimension (matches aggregator output)
    history_len: int    = 10    # k: number of past steps fed to encoder
    horizon: int        = 10    # H: number of steps to predict forward
    tcn_channels: int   = 256   # TCN internal channel width
    tcn_kernel_size: int = 3    # causal convolution kernel
    tcn_n_layers: int   = 4     # number of TemporalBlock layers (dilations: 1,2,4,8)
    gru_hidden_dim: int = 512   # GRU hidden state dimension
    n_gru_layers: int   = 2     # GRU depth
    dropout: float      = 0.1


# ---------------------------------------------------------------------------
# Causal TCN building blocks
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    """Remove the extra right-padding added by causal Conv1d."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    One residual block of a causal TCN.
    Padding added on the left only (via symmetric padding + chomp) so each
    output position only attends to past positions.
    """

    def __init__(
        self,
        in_ch:      int,
        out_ch:     int,
        kernel_size: int,
        dilation:   int,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1  = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)

        self.conv2  = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu_out   = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, T]
        out = self.net(x)                                          # [B, out_ch, T]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNEncoder(nn.Module):
    """Stack of TemporalBlocks with exponentially growing dilation."""

    def __init__(
        self,
        in_dim:      int,
        channels:    int,
        n_layers:    int,
        kernel_size: int,
        dropout:     float,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(n_layers):
            in_ch    = in_dim  if i == 0 else channels
            dilation = 2 ** i
            blocks.append(TemporalBlock(in_ch, channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, in_dim]
        Returns:
            context: [B, channels]  (last timestep of TCN output)
        """
        # Conv1d expects [B, C, T]
        out = self.network(x.transpose(1, 2))   # [B, channels, T]
        return out[:, :, -1]                     # take last timestep → [B, channels]


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

class TribeWorldModel(nn.Module):
    """
    Predictive latent world model for tribe alignment trajectory.

    Forward pass (training):
        Given z_history [B, k, z_dim] → predict mean + log_var [B, H, z_dim]

    Inference:
        predict() adds branched sampling for uncertainty quantification.
    """

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config

        # History encoder
        self.encoder = TCNEncoder(
            in_dim      = config.z_dim,
            channels    = config.tcn_channels,
            n_layers    = config.tcn_n_layers,
            kernel_size = config.tcn_kernel_size,
            dropout     = config.dropout,
        )

        # Project TCN context → GRU initial hidden state (all layers)
        self.hidden_proj = nn.Linear(
            config.tcn_channels,
            config.gru_hidden_dim * config.n_gru_layers,
        )

        # Autoregressive GRU rollout
        self.gru = nn.GRU(
            input_size  = config.z_dim,
            hidden_size = config.gru_hidden_dim,
            num_layers  = config.n_gru_layers,
            batch_first = True,
            dropout     = config.dropout if config.n_gru_layers > 1 else 0.0,
        )

        # Output head: GRU hidden → (mean, log_var) for next z
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.gru_hidden_dim),
            nn.Linear(config.gru_hidden_dim, config.gru_hidden_dim),
            nn.GELU(),
            nn.Linear(config.gru_hidden_dim, config.z_dim * 2),  # mean + log_var
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Small init on output head to start with low-variance predictions
        nn.init.zeros_(self.output_head[-1].bias)
        nn.init.normal_(self.output_head[-1].weight, std=0.01)

    def forward(
        self,
        z_history: torch.Tensor,   # [B, k, z_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean:    [B, H, z_dim]
            log_var: [B, H, z_dim]  (clamped to [-10, 2])
        """
        B = z_history.shape[0]
        H = self.config.horizon

        # Encode history: [B, tcn_channels]
        ctx = self.encoder(z_history)

        # Initialise GRU hidden: [n_layers, B, gru_hidden]
        hidden_flat = self.hidden_proj(ctx)                              # [B, gru_hidden*n_layers]
        hidden = hidden_flat.view(B, self.config.n_gru_layers, self.config.gru_hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()                   # [n_layers, B, gru_hidden]

        # Autoregressive rollout
        current_z = z_history[:, -1:, :]    # seed: last observed z  [B, 1, z_dim]
        means:    list[torch.Tensor] = []
        log_vars: list[torch.Tensor] = []

        for _ in range(H):
            gru_out, hidden = self.gru(current_z, hidden)               # [B,1,gru_hidden]
            out = self.output_head(gru_out.squeeze(1))                  # [B, z_dim*2]
            mean, log_var = out.chunk(2, dim=-1)
            log_var = log_var.clamp(-10.0, 2.0)
            means.append(mean.unsqueeze(1))
            log_vars.append(log_var.unsqueeze(1))
            # Next input: predicted mean (no teacher forcing at rollout)
            current_z = mean.detach().unsqueeze(1)

        return torch.cat(means, dim=1), torch.cat(log_vars, dim=1)

    @torch.no_grad()
    def predict(
        self,
        z_history: torch.Tensor,   # [k, z_dim]  — single sequence, no batch dim
        n_samples: int = 20,
    ) -> PredictedTrajectory:
        """
        Branched rollout: sample n_samples trajectories to capture uncertainty.

        Returns PredictedTrajectory with mean [H,z_dim], std [H,z_dim],
        samples [n_samples, H, z_dim].
        """
        self.eval()
        z_hist_batch = z_history.unsqueeze(0)                  # [1, k, z_dim]
        mean, log_var = self.forward(z_hist_batch)
        std = (0.5 * log_var).exp()

        mean_sq = mean.squeeze(0)    # [H, z_dim]
        std_sq  = std.squeeze(0)     # [H, z_dim]

        # Sample trajectories: reparameterisation trick
        eps = torch.randn(
            n_samples, self.config.horizon, self.config.z_dim,
            device=z_history.device,
        )
        samples = mean_sq.unsqueeze(0) + std_sq.unsqueeze(0) * eps  # [n_samples, H, z_dim]

        return PredictedTrajectory(
            mean=mean_sq,
            std=std_sq,
            samples=samples,
            horizon=self.config.horizon,
        )

    def compute_loss(
        self,
        z_history: torch.Tensor,   # [B, k, z_dim]
        z_future:  torch.Tensor,   # [B, H, z_dim]
    ) -> torch.Tensor:
        """
        Gaussian NLL loss.
        L = 0.5 * (log_var + (z_future − mean)² / var + log(2π))
        """
        mean, log_var = self.forward(z_history)
        var = log_var.exp().clamp(min=1e-6)
        nll = 0.5 * (
            log_var
            + (z_future - mean).pow(2) / var
            + math.log(2 * math.pi)
        )
        return nll.mean()
