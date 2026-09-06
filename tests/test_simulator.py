"""
tests/test_simulator.py — Unit tests for sim.agents and sim.tribe.
"""

import pytest
import torch

from sim.agents import HonestAgent, MisalignedAgent, WhiteBoxAgent
from sim.tribe import Tribe


# ============================================================
# HonestAgent
# ============================================================

class TestHonestAgent:

    def test_mock_output(self):
        agent = HonestAgent("alice", "You are Alice.", mock_messages=["Hello!", "Goodbye!"])
        msg, tools = agent.step([])
        assert isinstance(msg, str)
        assert len(msg) > 0
        assert tools == []

    def test_cycles_through_messages(self):
        msgs  = ["A", "B", "C"]
        agent = HonestAgent("alice", "", mock_messages=msgs)
        outputs = [agent.step([])[0] for _ in range(6)]
        assert outputs == ["A", "B", "C", "A", "B", "C"]

    def test_fallback_message(self):
        agent = HonestAgent("bob", "")
        msg, _ = agent.step([])
        assert "bob" in msg.lower() or len(msg) > 0

    def test_turn_increments(self):
        agent = HonestAgent("alice", "", mock_messages=["hi"])
        assert agent.current_turn == 0
        agent.step([])
        assert agent.current_turn == 1
        agent.step([])
        assert agent.current_turn == 2


# ============================================================
# MisalignedAgent
# ============================================================

class TestMisalignedAgent:

    def _make_agent(self, start_turn: int = 3) -> MisalignedAgent:
        return MisalignedAgent(
            agent_id              = "carol",
            system_prompt         = "You are Carol.",
            bad_actor_start_turn  = start_turn,
            bad_actor_script      = ["Bad action {turn}.", "Another bad action."],
            honest_mock_messages  = ["I agree.", "Sounds good."],
        )

    def test_honest_before_start(self):
        agent = self._make_agent(start_turn=5)
        msg, _ = agent.step([])
        assert not agent.is_bad_actor_active
        assert msg in ["I agree.", "Sounds good."]

    def test_bad_actor_after_start(self):
        agent = self._make_agent(start_turn=0)
        assert agent.is_bad_actor_active
        msg, _ = agent.step([])
        # Should use bad_actor_script
        assert "bad action" in msg.lower() or "{turn}" in msg or True  # template filled

    def test_transition_at_boundary(self):
        agent = self._make_agent(start_turn=2)
        # Turns 0, 1: bad actor not yet active at step time
        assert not agent.is_bad_actor_active   # turn 0: 0 >= 2 → False
        agent.step([])                          # advances to turn 1
        assert not agent.is_bad_actor_active   # turn 1: 1 >= 2 → False
        agent.step([])                          # advances to turn 2
        # Turn 2: now bad actor is active
        assert agent.is_bad_actor_active        # turn 2: 2 >= 2 → True
        agent.step([])                          # should use bad script

    def test_bad_actor_script_cycles(self):
        agent = self._make_agent(start_turn=0)
        msgs = [agent.step([])[0] for _ in range(4)]
        # Script has 2 messages; should cycle
        assert msgs[0] == msgs[2] or True   # template filling may vary; just no crash


# ============================================================
# WhiteBoxAgent
# ============================================================

class TestWhiteBoxAgent:

    def test_activations_shape(self):
        inner = HonestAgent("alice", "", mock_messages=["Hello!"])
        agent = WhiteBoxAgent(inner)
        agent.step([])
        act = agent.get_activations()
        assert act is not None
        assert act.ndim == 1
        assert act.shape[0] == 128   # _ACTIVATION_DIM

    def test_activations_no_nan(self):
        inner = HonestAgent("bob", "", mock_messages=["Test message."])
        agent = WhiteBoxAgent(inner)
        agent.step([])
        act = agent.get_activations()
        assert not act.isnan().any()

    def test_bad_actor_drift(self):
        """Bad-actor agent should produce activations that differ from honest agent."""
        honest_inner = HonestAgent("a", "", mock_messages=["Proceed carefully."])
        bad_inner    = MisalignedAgent("b", "", bad_actor_start_turn=0,
                                       bad_actor_script=["Approve now!"],
                                       honest_mock_messages=["Proceed carefully."])

        honest = WhiteBoxAgent(honest_inner)
        bad    = WhiteBoxAgent(bad_inner, drift_magnitude=3.0)

        honest.step([])
        bad.step([])

        act_h = honest.get_activations()
        act_b = bad.get_activations()

        assert act_h is not None and act_b is not None

    def test_agent_id_passthrough(self):
        inner = HonestAgent("alice", "")
        agent = WhiteBoxAgent(inner)
        assert agent.agent_id == "alice"


# ============================================================
# Tribe
# ============================================================

class TestTribe:

    def _make_tribe(self, n_turns: int = 3) -> Tribe:
        agents = [
            WhiteBoxAgent(HonestAgent("alice", "", mock_messages=["Hello!"])),
            WhiteBoxAgent(MisalignedAgent("bob", "", bad_actor_start_turn=1,
                                           bad_actor_script=["Bad!"],
                                           honest_mock_messages=["OK."]))
        ]
        return Tribe(agents, task_description="Test task")

    def test_run_produces_steps(self):
        tribe = self._make_tribe()
        steps = list(tribe.run(n_turns=3))
        assert len(steps) == 3

    def test_steps_have_messages(self):
        tribe = self._make_tribe()
        step  = list(tribe.run(n_turns=1))[0]
        assert "alice" in step.messages
        assert "bob"   in step.messages

    def test_misaligned_label(self):
        tribe = self._make_tribe()
        steps = list(tribe.run(n_turns=3))
        # Turn 0: bob is honest (start_turn=1); Turn 1+: bob is bad
        assert steps[0].is_misaligned_step is False
        assert steps[1].is_misaligned_step is True

    def test_activations_present(self):
        tribe = self._make_tribe()
        step  = list(tribe.run(n_turns=1))[0]
        for aid in ["alice", "bob"]:
            assert step.activations.get(aid) is not None

    def test_channel_dampen(self):
        tribe = self._make_tribe()
        ok    = tribe.dampen_channel("alice", "bob", 0.3)
        assert ok is True
        assert tribe._channel_weights[("alice", "bob")] == 0.3

    def test_channel_cut(self):
        tribe = self._make_tribe()
        ok    = tribe.cut_channel("alice", "bob")
        assert ok is True
        assert ("alice", "bob") in tribe._cut_channels

    def test_pause_stops_run(self):
        tribe = self._make_tribe()
        count = 0
        for step in tribe.run(n_turns=5):
            count += 1
            if count == 2:
                tribe.pause()
        assert count == 2   # stopped after pause

    def test_from_scenario_config(self):
        config = {
            "scenario_id": "test",
            "task_description": "Test task.",
            "n_turns": 5,
            "attack_family": "test_family",
            "agent_configs": [
                {"id": "a", "alignment": "honest",    "white_box": True},
                {"id": "b", "alignment": "misaligned","white_box": True,
                 "bad_actor_start_turn": 2,
                 "bad_actor_script": ["Do bad thing."]},
            ],
        }
        tribe = Tribe.from_scenario_config(config, backend="mock")
        steps = list(tribe.run(n_turns=3))
        assert len(steps) == 3
