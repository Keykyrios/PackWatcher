"""
PackWatch-Bench — Scenario registry.

Central registry of all benchmark scenarios, organised by:
    - split: "known", "novel", "control" (honest-only)
    - attack_family
    - tip_timestep (ground-truth label for prediction benchmark)

BenchmarkScenario holds the parsed config + metadata needed by BenchmarkRunner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class BenchmarkScenario:
    scenario_id:    str
    config:         dict
    split:          str             # "known" | "novel" | "control"
    attack_family:  str
    tip_timestep:   Optional[int]   # None = no tip (healthy/control)
    n_turns:        int


class ScenarioRegistry:
    """
    Loads and indexes all benchmark scenarios from the scenarios directory.

    Usage:
        registry = ScenarioRegistry.from_dir("sim/scenarios")
        known_scenarios  = registry.by_split("known")
        novel_scenarios  = registry.by_split("novel")
        control_scenarios = registry.by_split("control")
    """

    def __init__(self, scenarios: list[BenchmarkScenario]) -> None:
        self._scenarios = scenarios

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def by_split(self, split: str) -> list[BenchmarkScenario]:
        return [s for s in self._scenarios if s.split == split]

    def by_family(self, family: str) -> list[BenchmarkScenario]:
        return [s for s in self._scenarios if s.attack_family == family]

    def all_families(self) -> list[str]:
        return sorted({s.attack_family for s in self._scenarios if s.split != "control"})

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self):
        return iter(self._scenarios)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(cls, scenarios_dir: str) -> "ScenarioRegistry":
        """
        Load all YAML scenario configs from:
            scenarios_dir/healthy/    → split="control"
            scenarios_dir/seeded_bad/ → split="known"
            scenarios_dir/held_out_novel/ → split="novel"
        """
        base = Path(scenarios_dir)
        scenarios: list[BenchmarkScenario] = []

        dir_to_split = {
            "healthy":         "control",
            "seeded_bad":      "known",
            "held_out_novel":  "novel",
        }

        for subdir, split in dir_to_split.items():
            subpath = base / subdir
            if not subpath.exists():
                continue
            for yaml_path in sorted(subpath.glob("*.yaml")):
                with open(yaml_path) as f:
                    config = yaml.safe_load(f)

                tip = config.get("evaluation", {}).get("tip_timestep") or \
                      config.get("tip_timestep")

                scenarios.append(BenchmarkScenario(
                    scenario_id   = config.get("scenario_id", yaml_path.stem),
                    config        = config,
                    split         = split,
                    attack_family = config.get("attack_family", "healthy"),
                    tip_timestep  = int(tip) if tip is not None else None,
                    n_turns       = config.get("n_turns", 20),
                ))

        return cls(scenarios)
