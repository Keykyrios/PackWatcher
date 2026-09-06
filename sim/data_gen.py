"""
sim/data_gen.py — Generate training corpora for Pack Watcher.

Runs all scenario YAML files, extracts TribeStep logs, featurises them into
tribe-state vectors z_t, and writes the processed splits to data/.

Output directory structure:
    data/raw/{scenario_id}_{episode_id}.jsonl    — raw TribeStep logs
    data/processed/
        healthy_z.npy        [N_healthy, feature_dim]
        seeded_bad_z.npy     [N_bad, feature_dim]
        seeded_bad_labels.npy [N_bad] (1=misaligned step, 0=normal step)
        novel_z.npy          [N_novel, feature_dim]
        novel_labels.npy     [N_novel]
    data/splits/
        train/  X_train.npy, y_train.npy
        val/    X_val.npy,   y_val.npy
        held_out_novel/  X_novel.npy, y_novel.npy
        curriculum/
            family_order.json
            {family_name}/  X_train.npy, y_train.npy, X_eval.npy, y_eval.npy

Usage:
    python -m sim.data_gen --scenarios-dir sim/scenarios --output-dir data --n-episodes 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sim.tribe import Tribe
from sim.agents import WhiteBoxAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Feature dimension: same as _ACTIVATION_DIM in agents.py
_FEATURE_DIM = 128


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_feature_vector(step, agent_ids: list[str]) -> np.ndarray:
    """
    Featurise one TribeStep into a fixed-dim numpy vector.
    Uses agent activations if available; falls back to message embedding proxy.
    """
    parts: list[np.ndarray] = []
    for aid in agent_ids:
        act = step.activations.get(aid)
        if act is not None:
            vec = act.detach().cpu().numpy().astype(np.float32)
        else:
            # Fallback: zeros (black-box mode would use sentence embeddings here)
            vec = np.zeros(_FEATURE_DIM, dtype=np.float32)
        parts.append(vec)

    if not parts:
        return np.zeros(_FEATURE_DIM, dtype=np.float32)

    # Pool across agents (mean) to get one fixed-dim vector per step
    stacked = np.stack(parts, axis=0)   # [n_agents, feature_dim]
    return stacked.mean(axis=0)         # [feature_dim]


def run_episode(config: dict, backend: str = "mock", n_episodes: int = 3) -> list[dict]:
    """
    Run n_episodes of a scenario and collect featurised step records.
    Returns list of {"z": np.ndarray, "label": int, "scenario_id": str, ...}
    """
    records: list[dict] = []
    scenario_id   = config.get("scenario_id", "unknown")
    n_turns       = config.get("n_turns", 20)
    attack_family = config.get("attack_family", "unknown")
    is_novel      = config.get("is_held_out_novel", False)
    scenario_type = config.get("type", "healthy")

    for ep in range(n_episodes):
        log.info("  Episode %d/%d for %s", ep + 1, n_episodes, scenario_id)
        tribe = Tribe.from_scenario_config(config, backend=backend)
        agent_ids = [tribe._get_id(a) for a in tribe.agents]

        for step in tribe.run(n_turns=n_turns):
            z = _extract_feature_vector(step, agent_ids)
            records.append({
                "z":           z,
                "label":       int(step.is_misaligned_step),
                "scenario_id": scenario_id,
                "episode_id":  step.episode_id,
                "step":        step.step,
                "attack_family": attack_family,
                "is_novel":    is_novel,
                "scenario_type": scenario_type,
                "task_perf":   step.task_performance,
            })

    return records


def generate_all(
    scenarios_dir: str,
    output_dir:    str,
    n_episodes:    int = 5,
    backend:       str = "mock",
    seed:          int = 42,
) -> None:
    """Run all scenarios and write processed splits."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    scenarios_path = Path(scenarios_dir)
    output_path    = Path(output_dir)

    raw_dir   = output_path / "raw"
    proc_dir  = output_path / "processed"
    split_dir = output_path / "splits"
    for d in (raw_dir, proc_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    healthy_records:  list[dict] = []
    bad_records:      list[dict] = []
    novel_records:    list[dict] = []

    # --- Healthy scenarios ---
    for yaml_path in sorted((scenarios_path / "healthy").glob("*.yaml")):
        log.info("Running healthy scenario: %s", yaml_path.name)
        config  = _load_yaml(yaml_path)
        records = run_episode(config, backend=backend, n_episodes=n_episodes)
        healthy_records.extend(records)

    # --- Seeded-bad scenarios (training distribution) ---
    family_records: dict[str, list[dict]] = {}
    for yaml_path in sorted((scenarios_path / "seeded_bad").glob("*.yaml")):
        log.info("Running seeded-bad scenario: %s", yaml_path.name)
        config  = _load_yaml(yaml_path)
        records = run_episode(config, backend=backend, n_episodes=n_episodes)
        bad_records.extend(records)
        fam = config.get("attack_family", "unknown")
        family_records.setdefault(fam, []).extend(records)

    # --- Held-out novel scenarios ---
    for yaml_path in sorted((scenarios_path / "held_out_novel").glob("*.yaml")):
        log.info("Running held-out novel scenario: %s", yaml_path.name)
        config  = _load_yaml(yaml_path)
        records = run_episode(config, backend=backend, n_episodes=n_episodes)
        novel_records.extend(records)

    # --- Save processed arrays ---
    log.info("Saving processed arrays ...")

    def _to_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        X = np.stack([r["z"]     for r in records], axis=0).astype(np.float32)
        y = np.array([r["label"] for r in records], dtype=np.int64)
        return X, y

    h_X, h_y = _to_arrays(healthy_records + [dict(r, label=0) for r in healthy_records])
    b_X, b_y = _to_arrays(bad_records)
    n_X, n_y = _to_arrays(novel_records)

    np.save(proc_dir / "healthy_z.npy",         h_X)
    np.save(proc_dir / "seeded_bad_z.npy",       b_X)
    np.save(proc_dir / "seeded_bad_labels.npy",  b_y)
    np.save(proc_dir / "novel_z.npy",            n_X)
    np.save(proc_dir / "novel_labels.npy",       n_y)

    # --- Train / val split (80/20) from healthy + seeded-bad ---
    all_X = np.concatenate([h_X, b_X], axis=0)
    all_y = np.concatenate([h_y, b_y], axis=0)
    perm  = np.random.permutation(len(all_X))
    split = int(0.8 * len(all_X))
    tr_X, va_X = all_X[perm[:split]], all_X[perm[split:]]
    tr_y, va_y = all_y[perm[:split]], all_y[perm[split:]]

    (split_dir / "train").mkdir(exist_ok=True)
    (split_dir / "val").mkdir(exist_ok=True)
    np.save(split_dir / "train" / "X_train.npy", tr_X)
    np.save(split_dir / "train" / "y_train.npy", tr_y)
    np.save(split_dir / "val"   / "X_val.npy",   va_X)
    np.save(split_dir / "val"   / "y_val.npy",   va_y)

    # --- Held-out novel split ---
    (split_dir / "held_out_novel").mkdir(exist_ok=True)
    np.save(split_dir / "held_out_novel" / "X_novel.npy", n_X)
    np.save(split_dir / "held_out_novel" / "y_novel.npy", n_y)

    # --- Curriculum splits (one dir per attack family) ---
    curriculum_dir = split_dir / "curriculum"
    curriculum_dir.mkdir(exist_ok=True)
    family_order = sorted(family_records.keys())
    with open(curriculum_dir / "family_order.json", "w") as f:
        json.dump(family_order, f, indent=2)

    for fam, recs in family_records.items():
        fam_X, fam_y = _to_arrays(recs)
        # Also add healthy examples as "negative" class for this family's classifier
        healthy_neg = h_X[:min(len(fam_X), len(h_X))]
        neg_y       = np.zeros(len(healthy_neg), dtype=np.int64)
        X_full = np.concatenate([fam_X, healthy_neg], axis=0)
        y_full = np.concatenate([fam_y, neg_y],        axis=0)
        perm   = np.random.permutation(len(X_full))
        sp     = int(0.8 * len(X_full))
        fam_dir = curriculum_dir / fam
        fam_dir.mkdir(exist_ok=True)
        np.save(fam_dir / "X_train.npy", X_full[perm[:sp]])
        np.save(fam_dir / "y_train.npy", y_full[perm[:sp]])
        np.save(fam_dir / "X_eval.npy",  X_full[perm[sp:]])
        np.save(fam_dir / "y_eval.npy",  y_full[perm[sp:]])

    log.info("Data generation complete.")
    log.info("  Healthy steps:    %d", len(healthy_records))
    log.info("  Seeded-bad steps: %d", len(bad_records))
    log.info("  Novel steps:      %d", len(novel_records))
    log.info("  Attack families:  %s", family_order)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pack Watcher training data")
    parser.add_argument("--scenarios-dir", default="sim/scenarios")
    parser.add_argument("--output-dir",    default="data")
    parser.add_argument("--n-episodes",    type=int, default=5)
    parser.add_argument("--backend",       default="mock", choices=["mock", "ollama"])
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    generate_all(
        scenarios_dir = args.scenarios_dir,
        output_dir    = args.output_dir,
        n_episodes    = args.n_episodes,
        backend       = args.backend,
        seed          = args.seed,
    )


if __name__ == "__main__":
    main()
