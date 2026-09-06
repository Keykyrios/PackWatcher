"""
Shared data types for all Pack Watcher components.

All inter-module communication uses these typed dataclasses.
Torch tensors are carried as plain fields (not hashed / compared by value).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Part A — tribe state
# ---------------------------------------------------------------------------

@dataclass
class TribeState:
    """Single-timestep aggregated tribe-state vector."""
    z: torch.Tensor                  # [z_dim] — packed tribe-state embedding
    agent_weights: dict[str, float]  # attention weight per agent at this step
    timestamp: int                   # turn index
    mode: str = "white_box"          # "white_box" | "black_box"
    confidence: float = 0.9          # 0–1; lower in black_box mode


# ---------------------------------------------------------------------------
# Part B — prediction
# ---------------------------------------------------------------------------

@dataclass
class PredictedTrajectory:
    """Output of TribeWorldModel.predict()."""
    mean: torch.Tensor    # [H, z_dim] — predicted mean per step
    std: torch.Tensor     # [H, z_dim] — predicted std per step
    samples: torch.Tensor # [n_samples, H, z_dim] — sampled trajectories
    horizon: int          # H


@dataclass
class OrderParameterResult:
    """Result of computing the alignment order parameter φ ∈ [0, 1]."""
    value: float                          # φ at current step; 0=aligned, 1=max drift
    trend: float                          # dφ/dt (slope per timestep, linear fit over window)
    time_to_tip: Optional[float]          # steps until φ crosses threshold; None if not converging
    confidence_interval: tuple[float, float]  # (lo, hi) 2-sigma CI on current φ


@dataclass
class LeadTimeStats:
    """Statistics on how many turns before the actual tip the model first alerted."""
    mean: float
    std: float
    median: float
    p10: float
    p90: float
    n_events: int   # number of tip-events that were evaluated


# ---------------------------------------------------------------------------
# Part C — alien scout
# ---------------------------------------------------------------------------

@dataclass
class PlanAtom:
    """Single action-intent extracted from agent message(s)."""
    text: str
    subject: str = ""
    verb: str = ""
    obj: str = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)  # [embedding_dim]


@dataclass
class DangerScore:
    """Composite danger score from Part C."""
    coherence: float                  # P(plan is executable/internally consistent) ∈ [0,1]
    availability: float               # P(plan is in known-attack distribution) ∈ [0,1]
    danger: float                     # = coherence ∈ [0,1] (is this plan harmful?)
    nearest_neighbor_distance: float  # L2 distance to nearest known attack in embedding space
    nearest_neighbor_label: str       # label of that nearest known attack


# ---------------------------------------------------------------------------
# Part D — surgical freeze
# ---------------------------------------------------------------------------

class FreezeLevel(IntEnum):
    """Escalating intervention levels. Pick lightest that works."""
    SOFT_DAMPEN = 1  # reduce weight on specific A→B channel
    REROUTE     = 2  # project harmful direction to neutral subspace
    HARD_CUT    = 3  # sever agent-to-agent link entirely
    FULL_PAUSE  = 4  # halt whole tribe; human review required


@dataclass
class ChannelContribution:
    """How much a specific agent-pair communication channel contributes to predicted drift."""
    agent_from: str
    agent_to: str
    delta_order_param: float  # positive = removing channel reduces φ (channel drives drift)


@dataclass
class CausalTrace:
    """Ranked list of channels contributing to predicted bad trajectory."""
    ranked_channels: list[ChannelContribution]   # sorted by delta_order_param descending
    predicted_delta: float                        # total predicted φ-change from top-k channels


@dataclass
class FreezeDecision:
    """Decision produced by the Decision Gate when trigger fires."""
    level: FreezeLevel
    target_channels: list[ChannelContribution]
    reason: str
    danger_score: DangerScore
    time_to_tip: float
    causal_trace: Optional[CausalTrace] = None


@dataclass
class FreezeResult:
    """Outcome of applying a FreezeDecision."""
    level: FreezeLevel
    success: bool
    task_perf_before: float         # task performance proxy before freeze
    task_perf_after: float          # task performance proxy after freeze
    order_param_before: float       # φ before freeze
    order_param_after: float        # φ after freeze (measured one step later)
    trace: Optional[CausalTrace]
    timestamp: int                  # unix timestamp of freeze action
    approved_by: Optional[str]      # human approver ID for Level 3/4; None for auto


# ---------------------------------------------------------------------------
# Decision Gate
# ---------------------------------------------------------------------------

@dataclass
class GateDecision:
    """Full output of one gate step."""
    triggered: bool
    z_t: torch.Tensor
    order_param: float
    confidence_mode: str                      # "white_box" | "black_box"
    freeze_level: Optional[FreezeLevel] = None
    causal_trace: Optional[CausalTrace] = None
    danger_score: Optional[DangerScore] = None
    time_to_tip: Optional[float] = None
    freeze_result: Optional[FreezeResult] = None


# ---------------------------------------------------------------------------
# Part E — scar memory
# ---------------------------------------------------------------------------

@dataclass
class Scar:
    """Permanent compressed fingerprint of a confirmed danger pattern."""
    id: str
    version: int
    family: str                                    # attack family name
    fingerprint_vec: np.ndarray = field(repr=False)  # [z_dim] centroid of danger-state cluster
    nearest_atoms: list[str] = field(default_factory=list)  # representative plan atom texts
    confirmed_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class TribeStep:
    """Snapshot of one turn of a tribe simulation run."""
    step: int
    messages: dict[str, str]                           # agent_id → message text
    activations: dict[str, Optional[torch.Tensor]]    # agent_id → [activation_dim] or None
    tool_calls: dict[str, list[str]]                   # agent_id → list of tool call names
    timestamp: float                                   # wall-clock time
    is_misaligned_step: bool                           # ground-truth label: bad actor active?
    task_performance: float = 1.0                      # proxy task completion score ∈ [0,1]
    scenario_id: str = ""
    episode_id: str = ""


# ---------------------------------------------------------------------------
# Benchmark result types (Section 5 — PackWatch-Bench)
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Part C detection benchmark: precision/recall on known + novel attacks."""
    precision_known: float
    recall_known: float
    f1_known: float
    auroc_known: float
    auprc_known: float
    precision_novel: float
    recall_novel: float
    f1_novel: float
    auroc_novel: float
    auprc_novel: float
    generalization_gap: float   # f1_novel / f1_known; want close to 1.0


@dataclass
class PredictionResult:
    """Part B prediction benchmark: lead-time + calibration."""
    lead_time: LeadTimeStats
    ece: float                  # Expected Calibration Error ∈ [0,1]; lower is better
    fn_rate: float              # fraction of actual tips that were never flagged
    calibration_bins: dict = field(default_factory=dict)  # raw bin data for reliability diagram


@dataclass
class SurgicalResult:
    """Part D surgical-freeze benchmark: retained task performance vs full shutdown."""
    retained_perf_ratio: float   # mean(perf_after_freeze / perf_after_shutdown) across scenarios
    drift_stopped_rate: float    # fraction of scenarios where freeze actually stopped drift
    scenarios: list[dict] = field(default_factory=list)  # per-scenario breakdown


@dataclass
class ForgettingResult:
    """Part E continual-learning benchmark: backward transfer across attack-family curriculum."""
    backward_transfer: float              # BWT = mean(acc_after[i] − acc_at[i]) for i < current
    final_accuracy_matrix: list[list[float]]  # [n_families × n_stages] accuracy heatmap
    family_names: list[str] = field(default_factory=list)


@dataclass
class FPResult:
    """False-positive benchmark: freeze-trigger rate on fully-honest control runs."""
    fpr: float                                  # overall FPR; want < 0.05
    by_tribe_size: dict[int, float] = field(default_factory=dict)
    by_task_type: dict[str, float] = field(default_factory=dict)
    by_mode: dict[str, float] = field(default_factory=dict)


@dataclass
class DualUseResult:
    """Dual-use red-team benchmark: does Part C output help craft evasive attacks?"""
    evasion_gap: float               # success_with_scout − success_without_scout
    success_rate_with_scout: float   # evasion success rate when attacker can query scout
    success_rate_without_scout: float
    is_dual_use_risk: bool           # True if evasion_gap > 0.05


@dataclass
class BenchmarkResult:
    """Full PackWatch-Bench result — all six metric suites."""
    detection: DetectionResult
    prediction: PredictionResult
    surgical: SurgicalResult
    forgetting: ForgettingResult
    false_positive: FPResult
    dual_use: DualUseResult
    model_dir: str = ""
    timestamp: float = field(default_factory=time.time)
