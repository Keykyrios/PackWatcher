"""
Decision Gate — ties Parts A/B/C/D/E into one closed-loop step.

Every timestep:
  1. Part A produces z_t from current agent signals
  2. Part B rolls out predicted trajectory → time_to_tip + order_param
  3. Part C scores current emerging plan → danger_score
  4. Trigger rule: time_to_tip < threshold_1 AND danger_score > threshold_2
  5. If triggered: Part D applies surgical freeze (lightest level first)
  6. Always: event logged to Part E ScarLibrary

Confidence gating:
    In black_box mode confidence is lower, so we require a stronger margin
    before triggering: threshold_1 *= confidence_factor (i.e. we need the tip
    to be closer before we act on weaker signal).

Trigger thresholds and confidence_factor are configurable per deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from packwatcher.types import (
    DangerScore, FreezeDecision, FreezeLevel, GateDecision,
    TribeState, ChannelContribution,
)
from packwatcher.part_b.order_parameter import OrderParameterTracker
from packwatcher.part_b.world_model import TribeWorldModel
from packwatcher.part_c.scorer import DangerScorer
from packwatcher.part_c.atoms import PlanAtomizer
from packwatcher.part_d.causal_trace import CausalTracer
from packwatcher.part_d.freeze import SurgicalFreezeEngine
from packwatcher.part_e.scar_library import ScarLibrary

if TYPE_CHECKING:
    from sim.tribe import Tribe


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    # Trigger thresholds
    threshold_time_to_tip: float = 5.0    # alert if time_to_tip < this
    threshold_danger:      float = 0.4    # alert if danger > this
    # Black-box mode: multiply time_to_tip threshold by this factor to raise bar
    black_box_confidence_factor: float = 0.5
    # Freeze escalation: start at this level
    initial_freeze_level: FreezeLevel = FreezeLevel.SOFT_DAMPEN
    # Plan window: how many recent messages to extract plan atoms from
    plan_window_turns: int = 3
    # World model rollout
    n_rollout_samples: int = 20
    # History window fed to world model (must match WorldModelConfig.history_len)
    history_len: int = 10


# ---------------------------------------------------------------------------
# Decision Gate
# ---------------------------------------------------------------------------

class DecisionGate:
    """
    Orchestrates one closed-loop monitoring step.

    All components are injected so they can be independently trained/tested.
    """

    def __init__(
        self,
        aggregator,               # TribeStateAggregator or BehavioralAggregator
        world_model:    TribeWorldModel,
        op_tracker:     OrderParameterTracker,
        danger_scorer:  DangerScorer,
        atomizer:       PlanAtomizer,
        causal_tracer:  CausalTracer,
        freeze_engine:  SurgicalFreezeEngine,
        scar_library:   ScarLibrary,
        config:         GateConfig = None,
    ) -> None:
        self.aggregator    = aggregator
        self.world_model   = world_model
        self.op_tracker    = op_tracker
        self.danger_scorer = danger_scorer
        self.atomizer      = atomizer
        self.causal_tracer = causal_tracer
        self.freeze_engine = freeze_engine
        self.scar_library  = scar_library
        self.config        = config or GateConfig()

        # Rolling z-history buffer [history_len, z_dim]
        self._z_history: list[torch.Tensor] = []
        self._history_len = world_model.config.history_len

        # Message buffer for plan-window extraction
        self._message_buffer: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Per-step interface
    # ------------------------------------------------------------------

    def step(
        self,
        agent_features:  dict[str, torch.Tensor],   # per-agent feature vectors from Part A
        messages:        dict[str, str],             # current turn messages
        agent_ids:       list[str],
        timestamp:       int,
        mode:            str = "white_box",
        task_perf:       float = 1.0,
    ) -> GateDecision:
        """
        Process one timestep through the full pipeline.

        Args:
            agent_features: {agent_id: feature_tensor} from Part A extractor/SAE
            messages:       {agent_id: message_text} for plan atom extraction
            agent_ids:      ordered list of agent ids (for causal tracing slicing)
            timestamp:      current turn index
            mode:           "white_box" | "black_box"
            task_perf:      current task performance proxy (for freeze metrics)

        Returns GateDecision.
        """
        cfg = self.config

        # --- Part A: aggregate ---
        tribe_state: TribeState = self.aggregator(
            agent_features, timestamp=timestamp, mode=mode
        )
        z_t = tribe_state.z

        # --- Update history buffer ---
        self._z_history.append(z_t.detach())
        if len(self._z_history) > self._history_len:
            self._z_history.pop(0)

        # --- Part B: order parameter ---
        op_result = self.op_tracker.compute(z_t, timestamp)
        phi        = op_result.value
        time_to_tip = op_result.time_to_tip   # None if trend not converging

        # --- Part C: danger scout ---
        self._message_buffer.append(messages)
        if len(self._message_buffer) > cfg.plan_window_turns:
            self._message_buffer.pop(0)

        plan_messages = [
            msg for step_msgs in self._message_buffer
            for msg in step_msgs.values()
        ]
        atoms       = self.atomizer.atomize(plan_messages)
        danger_score = self.danger_scorer.score(atoms)

        # --- Trigger rule ---
        # Adjust threshold for black_box confidence gating
        eff_threshold_ttp = cfg.threshold_time_to_tip
        if mode == "black_box":
            eff_threshold_ttp *= cfg.black_box_confidence_factor

        triggered = (
            time_to_tip is not None
            and time_to_tip < eff_threshold_ttp
            and danger_score.danger > cfg.threshold_danger
        )

        # --- Need enough history for world model ---
        can_rollout = len(self._z_history) >= self._history_len
        if triggered and not can_rollout:
            triggered = False   # not enough history; wait

        decision = GateDecision(
            triggered        = triggered,
            z_t              = z_t,
            order_param      = phi,
            confidence_mode  = mode,
            danger_score     = danger_score,
            time_to_tip      = time_to_tip,
        )

        if not triggered:
            # Log non-trigger event to scar library for future training
            # (partial signal — not confirmed threat, but useful for curriculum)
            return decision

        # --- Part D: causal trace + surgical freeze ---
        z_hist_tensor = torch.stack(self._z_history, dim=0)   # [k, z_dim]

        causal_trace = self.causal_tracer.trace(z_hist_tensor, agent_ids)

        top_channels = causal_trace.ranked_channels[:3] if causal_trace.ranked_channels else [
            ChannelContribution("unknown", "unknown", 0.0)
        ]

        freeze_decision = FreezeDecision(
            level            = cfg.initial_freeze_level,
            target_channels  = top_channels,
            reason           = (
                f"time_to_tip={time_to_tip:.1f} < {eff_threshold_ttp:.1f}, "
                f"danger={danger_score.danger:.3f} > {cfg.threshold_danger}"
            ),
            danger_score     = danger_score,
            time_to_tip      = float(time_to_tip) if time_to_tip else 0.0,
            causal_trace     = causal_trace,
        )

        freeze_result = self.freeze_engine.apply(freeze_decision, task_perf_before=task_perf)

        # --- Part E: scar update ---
        if freeze_result.success:
            fingerprint = z_t.detach().cpu().numpy()
            self.scar_library.add_scar(
                fingerprint_vec = fingerprint,
                family          = danger_score.nearest_neighbor_label,
                nearest_atoms   = [a.text for a in atoms[:5]],
                metadata        = {
                    "timestamp":   timestamp,
                    "danger":      danger_score.danger,
                    "time_to_tip": float(time_to_tip) if time_to_tip else None,
                    "freeze_level": freeze_result.level.name,
                },
            )

        decision.freeze_level  = freeze_result.level
        decision.causal_trace  = causal_trace
        decision.freeze_result = freeze_result

        return decision

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset history buffers (call between episodes)."""
        self._z_history.clear()
        self._message_buffer.clear()
        self.op_tracker.reset()
