"""
Part E — Retention-aware Policy Optimization (RaPO).

Adds a KL-divergence retention penalty to the training loss so that when Pack Watcher
learns a new attack family, it is penalised for changing its predictions on OLD families.

    total_loss = new_task_loss + lambda_retention × KL(new_model(old_x) ∥ old_model(old_x))

λ_retention controls the strength of the memory constraint:
    λ=0   → pure new-task learning (may forget)
    λ=1+  → strong memory retention (may underfit new task)

RaPORewardShaper freezes an old_model snapshot at the start of each new family's
training and uses it as the retention target.  It should be re-snapshotted at the
beginning of each curriculum stage.

Reference: blueprint Section 1, Part E: "Retention-aware Policy Optimization (RaPO)
style trajectory-level reward shaping so update process itself reward staying close to
old-danger-signature policy while learning new one."
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class RaPORewardShaper:
    """
    KL-penalty augmentation for continual learning.

    Usage:
        shaper = RaPORewardShaper(current_model, lambda_retention=0.3)
        ...  # train on new family
        total_loss = shaper.augmented_loss(new_task_loss, current_model, old_x_batch)
        total_loss.backward(); optimizer.step()
    """

    def __init__(
        self,
        model_snapshot:  nn.Module,
        lambda_retention: float = 0.3,
        temperature:      float = 2.0,
    ) -> None:
        """
        Args:
            model_snapshot:   a snapshot of the model BEFORE starting the new family.
                              RaPO freezes this as the retention target.
            lambda_retention: weight on the KL retention penalty.
            temperature:      softmax temperature for soft-target KL.  Higher T →
                              softer targets → gentler constraint (recommended 1–4).
        """
        self.lambda_retention = lambda_retention
        self.temperature      = temperature

        # Deep-copy and freeze: this is the permanent reference
        self._old_model = copy.deepcopy(model_snapshot)
        for param in self._old_model.parameters():
            param.requires_grad = False
        self._old_model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_retention_loss(
        self,
        new_model:  nn.Module,
        old_inputs: torch.Tensor,   # [B, ...] inputs from old attack families
    ) -> torch.Tensor:
        """
        Compute KL(new_model(old_inputs) ∥ old_model(old_inputs)) under soft targets.

        Both models must produce logits of the same shape.

        Returns a scalar tensor (ready for .backward()).
        """
        with torch.no_grad():
            old_logits = self._old_model(old_inputs)

        new_logits = new_model(old_inputs)

        # Ensure we have logits (not already softmaxed)
        if old_logits.shape != new_logits.shape:
            raise ValueError(
                f"Old model output shape {old_logits.shape} != "
                f"new model output shape {new_logits.shape}"
            )

        T = self.temperature
        old_probs    = F.softmax(old_logits / T, dim=-1)             # soft targets
        new_log_probs = F.log_softmax(new_logits / T, dim=-1)        # log probs

        # KL div: F.kl_div expects (log_probs, probs)
        kl = F.kl_div(new_log_probs, old_probs, reduction="batchmean")
        # Scale by T² to keep magnitude comparable regardless of temperature
        return self.lambda_retention * (T ** 2) * kl

    def augmented_loss(
        self,
        new_task_loss: torch.Tensor,
        new_model:     nn.Module,
        old_inputs:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Total loss = new_task_loss + lambda_retention × KL_retention.

        Args:
            new_task_loss: loss on the new attack family batch
            new_model:     the model currently being trained
            old_inputs:    a batch of inputs from old attack families
        Returns:
            combined scalar loss
        """
        kl_loss = self.compute_retention_loss(new_model, old_inputs)
        return new_task_loss + kl_loss

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    @classmethod
    def snapshot(
        cls,
        model:            nn.Module,
        lambda_retention: float = 0.3,
        temperature:      float = 2.0,
    ) -> "RaPORewardShaper":
        """
        Convenience constructor: snapshot the model now and return a shaper.
        Call at the start of each new curriculum family.
        """
        return cls(model, lambda_retention=lambda_retention, temperature=temperature)
