"""
Part E — Collaborative Neural Learning (CNL).

When Pack Watcher learns a new attack family it must NOT forget old ones.
CNL achieves this by:

1. Before updating on new task: store old-task gradient direction (snapshot_gradients).
2. When computing new-task gradient: identify parameters whose new gradient is
   anti-correlated (cosine < 0) with the old gradient direction.
3. Project the new gradient to remove the conflicting component — keeping only
   the part that doesn't hurt old-task performance.

This is the gradient-projection variant of continual learning, related to GEM and A-GEM
but operating at the parameter level rather than episodic-memory level.

Reference: blueprint Section 1, Part E: "freeze conflicting neuron (whose gradient fight
with old-knowledge gradient), only update collaborative neuron."
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CNLTrainer:
    """
    Wraps a model + optimizer and applies Collaborative Neural Learning updates.

    Workflow per new-family training step:
        trainer.snapshot_gradients()          # AFTER computing old-task loss.backward()
        trainer.step(new_task_loss)           # computes new grad, projects, steps

    If snapshot_gradients() was never called (first family), step() behaves
    identically to a plain optimizer step.
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self._old_grads: Optional[dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot_gradients(self) -> None:
        """
        Store the current parameter gradients as the 'old task' reference direction.

        Call this after running loss.backward() on the old-task data, before
        calling optimizer.zero_grad() for the new task.
        """
        self._old_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self._old_grads[name] = param.grad.detach().clone()

    def step(self, new_task_loss: torch.Tensor) -> dict[str, float]:
        """
        Compute gradients for new_task_loss, project out conflicting components,
        then step the optimizer.

        Returns a dict of stats:
            n_params_projected: how many parameter tensors were modified
            mean_conflict_cos:  mean cosine similarity (should be ≥ 0 after projection)
        """
        self.optimizer.zero_grad(set_to_none=True)
        new_task_loss.backward()

        n_projected    = 0
        cos_sims: list[float] = []

        if self._old_grads is not None:
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                old_g = self._old_grads.get(name)
                if old_g is None:
                    continue

                old_g = old_g.to(param.grad.device)
                projected, cos = self._project_gradient(old_g, param.grad.data)
                cos_sims.append(cos)

                if cos < 0:    # only modify if there WAS a conflict
                    param.grad.data = projected
                    n_projected += 1

        self.optimizer.step()

        return {
            "n_params_projected": n_projected,
            "mean_conflict_cos":  float(sum(cos_sims) / len(cos_sims)) if cos_sims else 0.0,
        }

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    @staticmethod
    def _project_gradient(
        old_grad: torch.Tensor,
        new_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Project new_grad to remove the component that conflicts with old_grad.

        If dot(old_grad, new_grad) >= 0 the gradient is already aligned or orthogonal
        — no modification needed.

        Returns (projected_grad, cosine_similarity_before_projection).
        """
        old_norm = old_grad.norm()
        if old_norm < 1e-12:
            return new_grad, 1.0   # degenerate old gradient; nothing to project

        new_norm = new_grad.norm()
        if new_norm < 1e-12:
            return new_grad, 1.0   # degenerate new gradient

        cos = float(
            torch.dot(old_grad.flatten(), new_grad.flatten())
            / (old_norm * new_norm)
        )

        if cos >= 0:
            return new_grad, cos

        # Remove the conflicting component (projection onto old_grad direction)
        old_unit     = old_grad / old_norm.clamp(min=1e-12)
        conflict     = torch.dot(new_grad.flatten(), old_unit.flatten()) * old_unit
        projected    = new_grad - conflict.view_as(new_grad)
        return projected, cos
