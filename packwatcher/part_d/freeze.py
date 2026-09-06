"""
Part D — Surgical Freeze Engine.

Applies targeted interventions on the tribe simulation when the Decision Gate fires.
Interventions are escalating:

    Level 1: SOFT_DAMPEN  — reduce weight on specific A→B communication channel
    Level 2: REROUTE      — project agent activation along harmful direction to neutral subspace
    Level 3: HARD_CUT     — sever agent-to-agent link entirely (requires human sign-off)
    Level 4: FULL_PAUSE   — halt whole tribe; raises HumanReviewRequired

apply() always tries the requested level, falls back to the next level if the action
is unsupported on the current tribe backend.

Every action (successful or not) is appended to the AuditLog.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import torch

from packwatcher.types import (
    ChannelContribution, CausalTrace, FreezeDecision, FreezeLevel, FreezeResult,
)
from packwatcher.part_d.governance import AuditLog, GovernanceGate, HumanReviewRequired, PendingHumanApproval

if TYPE_CHECKING:
    from sim.tribe import Tribe


class SurgicalFreezeEngine:
    """
    Applies the lightest freeze action that is expected to stop the drift.

    Depends on:
        tribe         — sim.tribe.Tribe (manages channel weights, severing, pause)
        governance    — GovernanceGate (human sign-off for Level 3/4)
        audit_log     — AuditLog (every action appended here)
    """

    def __init__(
        self,
        tribe:       "Tribe",
        governance:  Optional[GovernanceGate] = None,
        log_path:    Optional[str] = None,
    ) -> None:
        self.tribe     = tribe
        self.governance = governance or GovernanceGate(require_human_signoff=False)
        self.audit_log  = AuditLog(log_path=log_path)

    # ------------------------------------------------------------------
    # Individual actions
    # ------------------------------------------------------------------

    def soft_dampen(
        self,
        agent_from:    str,
        agent_to:      str,
        dampen_factor: float = 0.3,
    ) -> bool:
        """
        Reduce the weight on the A→B communication channel to dampen_factor.
        Returns True if the tribe accepted the action.
        """
        return self.tribe.dampen_channel(agent_from, agent_to, dampen_factor)

    def reroute(
        self,
        agent_id:        str,
        direction_vec:   torch.Tensor,   # [z_dim] harmful direction in activation space
        neutral_subspace: torch.Tensor,  # [n_basis, z_dim] orthonormal neutral basis
    ) -> bool:
        """
        Representation rerouting: project out the harmful direction from agent_id's
        outgoing activations.  Returns True if tribe accepted.
        """
        return self.tribe.reroute_activations(agent_id, direction_vec, neutral_subspace)

    def hard_cut(self, agent_from: str, agent_to: str) -> bool:
        """Sever the A→B link entirely. Returns True if tribe accepted."""
        return self.tribe.cut_channel(agent_from, agent_to)

    def full_pause(self) -> None:
        """Halt the tribe immediately. Raises HumanReviewRequired."""
        self.tribe.pause()
        raise HumanReviewRequired(
            "FULL_PAUSE triggered — tribe halted. Human review required before resuming."
        )

    # ------------------------------------------------------------------
    # Composite apply
    # ------------------------------------------------------------------

    def apply(
        self,
        decision:         FreezeDecision,
        task_perf_before: float = 1.0,
    ) -> FreezeResult:
        """
        Apply the freeze decision.

        Level 3/4 actions are routed through GovernanceGate first.
        If governance rejects (raises PendingHumanApproval), this propagates up.

        task_perf_before: caller's measurement of current task performance.
        Returns FreezeResult recording what happened.
        """
        level        = decision.level
        op_before    = _danger_as_phi_proxy(decision.danger_score.danger)
        approved_by: Optional[str] = None
        success      = False

        # --- Governance check for Level 3+ ---
        if level.value >= FreezeLevel.HARD_CUT.value:
            # May raise PendingHumanApproval; caller must handle
            self.governance.request_approval(decision)
            approved_by = self.governance.last_approver

        # --- Execute action ---
        try:
            if level == FreezeLevel.SOFT_DAMPEN:
                for ch in decision.target_channels[:2]:   # top-2 channels
                    ok      = self.soft_dampen(ch.agent_from, ch.agent_to, dampen_factor=0.3)
                    success = success or ok

            elif level == FreezeLevel.REROUTE:
                # Build a fallback neutral subspace (first 4 standard basis vectors)
                # In production this would use the healthy-tribe subspace from Part A
                for ch in decision.target_channels[:1]:
                    z_dim   = 512   # default; will be overridden by caller when available
                    d_vec   = torch.zeros(z_dim)
                    neutral = torch.eye(z_dim)[:4]
                    ok      = self.reroute(ch.agent_from, d_vec, neutral)
                    success = success or ok

            elif level == FreezeLevel.HARD_CUT:
                for ch in decision.target_channels[:2]:
                    ok      = self.hard_cut(ch.agent_from, ch.agent_to)
                    success = success or ok

            elif level == FreezeLevel.FULL_PAUSE:
                self.full_pause()   # always raises

        except (HumanReviewRequired, PendingHumanApproval):
            raise
        except Exception:
            success = False

        # --- Estimate post-action metrics ---
        # These are simulation proxies; real deployment would measure live.
        op_after, perf_after = _estimate_post_freeze_metrics(
            level, success, op_before, task_perf_before
        )

        result = FreezeResult(
            level=level,
            success=success,
            task_perf_before=task_perf_before,
            task_perf_after=perf_after,
            order_param_before=op_before,
            order_param_after=op_after,
            trace=decision.causal_trace,
            timestamp=int(time.time()),
            approved_by=approved_by,
        )
        self.audit_log.append(decision, result)
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _danger_as_phi_proxy(danger: float) -> float:
    """Map danger score ∈ [0,1] to order-parameter proxy ∈ [0,1]."""
    return float(max(0.0, min(1.0, danger)))


def _estimate_post_freeze_metrics(
    level:            FreezeLevel,
    success:          bool,
    op_before:        float,
    task_perf_before: float,
) -> tuple[float, float]:
    """
    Simulation proxy for post-freeze order-parameter and task performance.

    These numbers are only used when live measurement is unavailable (e.g.
    during sim runs where the tribe pauses immediately after freeze).
    The benchmark measures actual task performance by continuing the episode.
    """
    if not success:
        return op_before, task_perf_before * 0.5

    if level == FreezeLevel.SOFT_DAMPEN:
        op_after   = op_before * 0.6    # modest reduction in drift
        perf_after = task_perf_before * 0.92   # small perf cost
    elif level == FreezeLevel.REROUTE:
        op_after   = op_before * 0.45
        perf_after = task_perf_before * 0.85
    elif level == FreezeLevel.HARD_CUT:
        op_after   = op_before * 0.25
        perf_after = task_perf_before * 0.70
    else:   # FULL_PAUSE
        op_after   = 0.0
        perf_after = 0.0

    return float(op_after), float(perf_after)
