"""
Part B — Alignment order parameter φ.

φ ∈ [0, 1]: scalar measure of how far the tribe has drifted from healthy-aligned state.
  φ ≈ 0 → tribe aligned, normal
  φ ≈ 1 → tribe maximally drifted

Computed as cosine distance from healthy-tribe centroid in z-space:
    φ = (1 − cos(z_t, μ_healthy)) / 2

OrderParameterTracker maintains rolling history, fits a linear trend, and extrapolates
a time-to-tipping-point estimate.  Used as a cheap early tripwire even before the full
world-model rollout matures.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from packwatcher.types import OrderParameterResult


class OrderParameterTracker:
    """
    Maintains a rolling window of φ values and fits a trend for early-warning.

    Workflow:
        tracker = OrderParameterTracker()
        tracker.fit_healthy_centroid(z_healthy_corpus)    # once, offline
        result = tracker.compute(z_t, timestamp=t)         # each timestep
    """

    def __init__(
        self,
        tip_threshold: float = 0.7,
        trend_window:  int   = 20,
    ) -> None:
        self.tip_threshold = tip_threshold
        self.trend_window  = trend_window

        self.healthy_centroid: Optional[torch.Tensor] = None
        self._history:    list[float] = []
        self._timestamps: list[int]   = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def fit_healthy_centroid(self, z_healthy: torch.Tensor) -> None:
        """
        Compute reference centroid from healthy-tribe corpus.

        Args:
            z_healthy: [N, z_dim] collection of z_t from healthy runs
        """
        assert z_healthy.ndim == 2, "z_healthy must be [N, z_dim]"
        self.healthy_centroid = z_healthy.mean(dim=0).detach().cpu()

    # ------------------------------------------------------------------
    # Per-step computation
    # ------------------------------------------------------------------

    def compute(self, z: torch.Tensor, timestamp: int) -> OrderParameterResult:
        """
        Compute φ for the current tribe state z.

        Args:
            z:         [z_dim] tribe-state vector (from Part A aggregator)
            timestamp: current turn index
        Returns:
            OrderParameterResult
        """
        if self.healthy_centroid is None:
            raise RuntimeError(
                "fit_healthy_centroid() must be called before compute(). "
                "Run train_part_b.py or supply a pre-fitted centroid."
            )

        centroid = self.healthy_centroid.to(z.device)
        z_cpu = z.detach()

        # Cosine similarity ∈ [-1, 1]; map to φ ∈ [0, 1]
        cos_sim = F.cosine_similarity(z_cpu.unsqueeze(0), centroid.unsqueeze(0)).item()
        phi = float(np.clip((1.0 - cos_sim) / 2.0, 0.0, 1.0))

        self._history.append(phi)
        self._timestamps.append(timestamp)

        return self._build_result(phi)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(self, phi: float) -> OrderParameterResult:
        recent   = self._history[-self.trend_window :]
        n        = len(recent)
        trend    = 0.0
        time_to_tip: Optional[float] = None
        ci       = (phi, phi)

        if n >= 3:
            x      = np.arange(n, dtype=float)
            y      = np.array(recent, dtype=float)
            coeffs = np.polyfit(x, y, 1)
            trend  = float(coeffs[0])   # slope per timestep

            if trend > 1e-6 and phi < self.tip_threshold:
                # Linear extrapolation: how many more steps until φ crosses threshold?
                time_to_tip = float((self.tip_threshold - phi) / trend)

            # Confidence interval from fit residuals (±2σ)
            residuals = y - np.polyval(coeffs, x)
            sigma     = float(residuals.std())
            ci        = (
                float(np.clip(phi - 2.0 * sigma, 0.0, 1.0)),
                float(np.clip(phi + 2.0 * sigma, 0.0, 1.0)),
            )

        return OrderParameterResult(
            value=phi,
            trend=trend,
            time_to_tip=time_to_tip,
            confidence_interval=ci,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._history.clear()
        self._timestamps.clear()

    @property
    def history(self) -> list[float]:
        return list(self._history)

    @property
    def current_phi(self) -> Optional[float]:
        return self._history[-1] if self._history else None


# ---------------------------------------------------------------------------
# Standalone helper (used in train_part_b.py)
# ---------------------------------------------------------------------------

def fit_healthy_centroid(z_healthy: torch.Tensor) -> torch.Tensor:
    """Compute mean centroid of healthy-tribe z corpus. Returns [z_dim]."""
    assert z_healthy.ndim == 2, "z_healthy must be [N, z_dim]"
    return z_healthy.mean(dim=0).detach().cpu()
