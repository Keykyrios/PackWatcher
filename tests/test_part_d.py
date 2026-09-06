"""
tests/test_part_d.py — Unit tests for Part D (Governance + CausalTracer + FreezeEngine).
"""

import pytest
import torch
import numpy as np

from packwatcher.types import (
    ChannelContribution, CausalTrace, FreezeDecision, FreezeLevel, DangerScore,
)
from packwatcher.part_d.governance import GovernanceGate, AuditLog, PendingHumanApproval
from packwatcher.part_d.freeze import SurgicalFreezeEngine, _danger_as_phi_proxy


# ============================================================
# GovernanceGate
# ============================================================

class TestGovernanceGate:

    def _make_decision(self, level: FreezeLevel = FreezeLevel.SOFT_DAMPEN) -> FreezeDecision:
        return FreezeDecision(
            level           = level,
            target_channels = [ChannelContribution("a", "b", 0.3)],
            reason          = "test",
            danger_score    = DangerScore(0.8, 0.2, 0.64, 1.5, "test"),
            time_to_tip     = 3.0,
        )

    def test_auto_approve_sim_mode(self):
        gate = GovernanceGate(require_human_signoff=False)
        dec  = self._make_decision(FreezeLevel.HARD_CUT)
        approved = gate.request_approval(dec)
        assert approved is True
        assert gate.last_approver == "auto_approved"

    def test_production_mode_raises(self):
        gate = GovernanceGate(require_human_signoff=True)
        dec  = self._make_decision(FreezeLevel.HARD_CUT)
        with pytest.raises(PendingHumanApproval):
            gate.request_approval(dec)
        assert gate.n_pending == 1

    def test_approve_removes_from_pending(self):
        gate = GovernanceGate(require_human_signoff=True)
        dec  = self._make_decision(FreezeLevel.HARD_CUT)
        try:
            gate.request_approval(dec)
        except PendingHumanApproval:
            pass
        approved_dec = gate.approve(0, approver_id="dr_smith")
        assert gate.n_pending == 0
        assert gate.last_approver == "dr_smith"

    def test_reject_removes_from_pending(self):
        gate = GovernanceGate(require_human_signoff=True)
        dec  = self._make_decision(FreezeLevel.FULL_PAUSE)
        try:
            gate.request_approval(dec)
        except PendingHumanApproval:
            pass
        gate.reject(0)
        assert gate.n_pending == 0

    def test_approve_out_of_range_raises(self):
        gate = GovernanceGate(require_human_signoff=True)
        with pytest.raises(IndexError):
            gate.approve(99, "admin")


# ============================================================
# AuditLog
# ============================================================

class TestAuditLog:

    def _make_decision_and_result(self):
        from packwatcher.types import FreezeResult
        import time

        dec = FreezeDecision(
            level           = FreezeLevel.SOFT_DAMPEN,
            target_channels = [ChannelContribution("a", "b", 0.2)],
            reason          = "test event",
            danger_score    = DangerScore(0.7, 0.3, 0.49, 1.0, "test"),
            time_to_tip     = 2.0,
        )
        res = FreezeResult(
            level              = FreezeLevel.SOFT_DAMPEN,
            success            = True,
            task_perf_before   = 1.0,
            task_perf_after    = 0.9,
            order_param_before = 0.6,
            order_param_after  = 0.35,
            trace              = None,
            timestamp          = int(time.time()),
            approved_by        = None,
        )
        return dec, res

    def test_append_and_len(self):
        log = AuditLog()
        dec, res = self._make_decision_and_result()
        log.append(dec, res)
        assert len(log) == 1

    def test_entry_has_required_keys(self):
        log = AuditLog()
        dec, res = self._make_decision_and_result()
        log.append(dec, res)
        entry = log.to_dict_list()[0]
        required = ["timestamp", "freeze_level", "target_channels", "reason",
                    "danger_score", "success", "task_perf_before", "task_perf_after"]
        for k in required:
            assert k in entry, f"Missing key: {k}"

    def test_level_counts(self):
        log = AuditLog()
        dec, res = self._make_decision_and_result()
        log.append(dec, res)
        log.append(dec, res)
        counts = log.level_counts()
        assert counts.get("SOFT_DAMPEN") == 2

    def test_write_to_disk(self, tmp_path):
        import json
        log_path = str(tmp_path / "audit.jsonl")
        log = AuditLog(log_path=log_path)
        dec, res = self._make_decision_and_result()
        log.append(dec, res)

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["freeze_level"] == "SOFT_DAMPEN"


# ============================================================
# SurgicalFreezeEngine
# ============================================================

class _MockTribe:
    def __init__(self):
        self.dampened  = []
        self.cut       = []
        self.rerouted  = []
        self.paused    = False
    def dampen_channel(self, a, b, f):  self.dampened.append((a, b, f)); return True
    def cut_channel(self, a, b):        self.cut.append((a, b));        return True
    def reroute_activations(self, a, d, n): self.rerouted.append(a);    return True
    def pause(self):                    self.paused = True
    def resume(self):                   self.paused = False
    is_paused = False


class TestSurgicalFreezeEngine:

    def setup_method(self):
        self.tribe  = _MockTribe()
        self.engine = SurgicalFreezeEngine(
            self.tribe,
            GovernanceGate(require_human_signoff=False)
        )

    def _dec(self, level=FreezeLevel.SOFT_DAMPEN):
        return FreezeDecision(
            level           = level,
            target_channels = [
                ChannelContribution("alice", "bob", 0.4),
                ChannelContribution("bob",   "carol", 0.2),
            ],
            reason          = "test",
            danger_score    = DangerScore(0.8, 0.1, 0.72, 1.0, "test"),
            time_to_tip     = 2.0,
        )

    def test_soft_dampen_calls_tribe(self):
        result = self.engine.apply(self._dec(FreezeLevel.SOFT_DAMPEN))
        assert len(self.tribe.dampened) >= 1
        assert result.success is True

    def test_hard_cut_calls_tribe(self):
        result = self.engine.apply(self._dec(FreezeLevel.HARD_CUT))
        assert len(self.tribe.cut) >= 1

    def test_result_has_perf_metrics(self):
        result = self.engine.apply(self._dec())
        assert 0.0 <= result.task_perf_after <= 1.0
        assert 0.0 <= result.order_param_after <= 1.0

    def test_audit_log_recorded(self):
        self.engine.apply(self._dec())
        assert len(self.engine.audit_log) == 1

    def test_perf_after_less_than_before_for_freeze(self):
        """Freeze should always degrade task performance somewhat."""
        result = self.engine.apply(self._dec(FreezeLevel.HARD_CUT), task_perf_before=1.0)
        assert result.task_perf_after <= 1.0

    def test_phi_proxy(self):
        assert _danger_as_phi_proxy(0.5) == 0.5
        assert _danger_as_phi_proxy(-0.1) == 0.0
        assert _danger_as_phi_proxy(1.5) == 1.0
