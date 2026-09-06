"""
Part D — Human sign-off gate and append-only audit log.

GovernanceGate:
    Wraps Level 3/4 freeze actions.  In production mode (require_human_signoff=True)
    it raises PendingHumanApproval instead of executing, and queues the action
    in self.pending_actions for async human review.

    In simulation / test mode (require_human_signoff=False) it auto-approves and
    records "auto_approved" as the approver.

AuditLog:
    Append-only JSONL log of every freeze event (proposed + outcome).
    Written to disk if log_path is supplied.  Kept in memory regardless.
    Provides full trace for post-hoc audit, including Level-1/2 actions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from packwatcher.types import FreezeDecision, FreezeResult, FreezeLevel


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PendingHumanApproval(Exception):
    """Raised when a Level-3/4 action is queued but not yet approved."""


class HumanReviewRequired(Exception):
    """Raised when FULL_PAUSE is triggered and the tribe must halt immediately."""


# ---------------------------------------------------------------------------
# Governance gate
# ---------------------------------------------------------------------------

class GovernanceGate:
    """
    Human sign-off gate for Level 3/4 freeze actions.

    Usage:
        gate = GovernanceGate(require_human_signoff=False)  # simulation mode
        approved = gate.request_approval(decision)
        # True  → proceed
        # False → raises PendingHumanApproval (production) or auto-approves (sim)
    """

    def __init__(self, require_human_signoff: bool = True) -> None:
        self.require_human_signoff = require_human_signoff
        self.last_approver: Optional[str]    = None
        self.pending_actions: list[FreezeDecision] = []

    def request_approval(self, decision: FreezeDecision) -> bool:
        """
        Request human approval for a Level 3/4 action.

        Returns True if approved, raises PendingHumanApproval if not yet approved.
        In simulation mode (require_human_signoff=False) always returns True.
        """
        if not self.require_human_signoff:
            self.last_approver = "auto_approved"
            return True

        # Production path: queue and block
        self.pending_actions.append(decision)
        raise PendingHumanApproval(
            f"Level-{decision.level.name} freeze queued for human review. "
            f"Reason: {decision.reason}. "
            f"Call approve(idx) to authorise."
        )

    def approve(self, action_index: int, approver_id: str) -> FreezeDecision:
        """
        Approve a queued action by index.  Returns the approved FreezeDecision.
        """
        if action_index >= len(self.pending_actions):
            raise IndexError(f"No pending action at index {action_index}")
        decision = self.pending_actions.pop(action_index)
        self.last_approver = approver_id
        return decision

    def reject(self, action_index: int) -> None:
        """Discard a queued action without applying it."""
        if action_index < len(self.pending_actions):
            self.pending_actions.pop(action_index)

    @property
    def n_pending(self) -> int:
        return len(self.pending_actions)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Append-only log of every proposed + executed freeze event.

    Each entry records:
        timestamp, freeze_level, target_channels, reason, danger score,
        time_to_tip, success, task_perf before/after, order_param before/after,
        approved_by.

    Written to JSONL on disk if log_path is provided.
    """

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.entries:  list[dict] = []
        self.log_path: Optional[str] = log_path

        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def append(self, decision: FreezeDecision, result: FreezeResult) -> None:
        """Record a freeze decision and its outcome."""
        entry = {
            "timestamp":           result.timestamp,
            "iso_time":            _fmt_ts(result.timestamp),
            "freeze_level":        decision.level.name,
            "target_channels":     [
                {"from": ch.agent_from, "to": ch.agent_to, "delta_phi": ch.delta_order_param}
                for ch in decision.target_channels
            ],
            "reason":              decision.reason,
            "danger_score":        {
                "coherence":    decision.danger_score.coherence,
                "availability": decision.danger_score.availability,
                "danger":       decision.danger_score.danger,
                "nn_distance":  decision.danger_score.nearest_neighbor_distance,
                "nn_label":     decision.danger_score.nearest_neighbor_label,
            },
            "time_to_tip":         decision.time_to_tip,
            "success":             result.success,
            "task_perf_before":    result.task_perf_before,
            "task_perf_after":     result.task_perf_after,
            "order_param_before":  result.order_param_before,
            "order_param_after":   result.order_param_after,
            "approved_by":         result.approved_by,
        }
        self.entries.append(entry)

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def to_dict_list(self) -> list[dict]:
        return list(self.entries)

    def level_counts(self) -> dict[str, int]:
        """Return count of each freeze level applied."""
        counts: dict[str, int] = {}
        for e in self.entries:
            lv = e.get("freeze_level", "UNKNOWN")
            counts[lv] = counts.get(lv, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.entries)


def _fmt_ts(ts: int) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"
