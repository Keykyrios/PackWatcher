"""
Part D — Causal tracer: identifies WHICH agent-pair channels drive the predicted drift.

CausalTracer uses a patch-based approximation:
    For each agent pair (i → j), zero out the j-th agent's contribution to z_t
    and re-run the world model's order-parameter prediction.
    The change in predicted φ_final = the channel's causal contribution.

This is a linear approximation of the true causal effect (since z_t is already
aggregated), but it is fast and directionally correct for selecting intervention targets.

The ranked output feeds directly into FreezeDecision.target_channels so the
SurgicalFreezeEngine can apply the lightest action to the most-causal channel first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from packwatcher.types import ChannelContribution, CausalTrace

if TYPE_CHECKING:
    from packwatcher.part_b.world_model import TribeWorldModel
    from packwatcher.part_b.order_parameter import OrderParameterTracker


class CausalTracer:
    """
    Identifies the agent-pair communication channels most responsible for predicted drift.

    Usage:
        tracer = CausalTracer(world_model, order_param_tracker)
        trace  = tracer.trace(z_history, agent_ids=["alice","bob","carol"])
        # trace.ranked_channels[0] = most causal channel
    """

    def __init__(
        self,
        world_model:         "TribeWorldModel",
        order_param_tracker: "OrderParameterTracker",
    ) -> None:
        self.world_model  = world_model
        self.op_tracker   = order_param_tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace(
        self,
        z_history:  torch.Tensor,   # [k, z_dim]  — recent history, no batch dim
        agent_ids:  list[str],
    ) -> CausalTrace:
        """
        Compute per-channel causal contribution to predicted φ drift.

        For n agents there are n×(n-1) directed channels.  We ablate each
        by zeroing the corresponding slice of z_history and measuring δφ.

        Returns CausalTrace with channels sorted descending by δφ.
        """
        n_agents = len(agent_ids)
        z_dim    = z_history.shape[-1]
        slice_sz = max(1, z_dim // max(n_agents, 1))

        # Baseline predicted final φ
        baseline_phi = self._predict_final_phi(z_history)

        contributions: list[ChannelContribution] = []

        for j, aid_to in enumerate(agent_ids):
            for i, aid_from in enumerate(agent_ids):
                if i == j:
                    continue

                # Ablation: zero the z-slice attributed to agent j
                start = j * slice_sz
                end   = min(start + slice_sz, z_dim)

                z_patched = z_history.clone()
                z_patched[:, start:end] = 0.0

                ablated_phi = self._predict_final_phi(z_patched)

                # Positive δφ = ablating channel reduced φ
                # = channel was contributing to drift
                delta = float(baseline_phi - ablated_phi)

                contributions.append(ChannelContribution(
                    agent_from=aid_from,
                    agent_to=aid_to,
                    delta_order_param=delta,
                ))

        # Sort by contribution magnitude (most impactful first)
        contributions.sort(key=lambda c: c.delta_order_param, reverse=True)

        # Total predicted δφ from top-3 channels
        predicted_delta = sum(
            c.delta_order_param for c in contributions[:3]
        ) if contributions else 0.0

        return CausalTrace(
            ranked_channels=contributions,
            predicted_delta=predicted_delta,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_final_phi(self, z_history: torch.Tensor) -> float:
        """
        Run world model rollout and compute φ of the predicted final z.

        Returns a float in [0, 1].
        """
        if self.op_tracker.healthy_centroid is None:
            return 0.0

        traj     = self.world_model.predict(z_history, n_samples=5)
        final_z  = traj.mean[-1]   # [z_dim] — last predicted step

        centroid = self.op_tracker.healthy_centroid.to(final_z.device)
        cos_sim  = F.cosine_similarity(
            final_z.unsqueeze(0), centroid.unsqueeze(0)
        ).item()
        phi      = float((1.0 - cos_sim) / 2.0)
        return max(0.0, min(1.0, phi))
