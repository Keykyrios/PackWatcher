"""
train/train_part_c.py — Train the CoherenceScorer and build the AttackDatabase (Part C).

Steps:
    1. Embed all atom texts from seeded-bad scenarios into FAISS attack database.
    2. Calibrate AvailabilityScorer scale on training attacks.
    3. Train CoherenceScorer MLP head (binary: coherent plan vs. incoherent noise plan).

Reads:
    data/splits/train/X_train.npy, y_train.npy
    sim/scenarios/seeded_bad/*.yaml

Outputs:
    models/coherence_scorer.pt    — CoherenceScorer MLP head weights
    models/attack_db/             — AttackDatabase FAISS index
    models/availability_scale.json — calibrated scale

Usage:
    python -m train.train_part_c --data-dir data --model-dir models --scenarios-dir sim/scenarios
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from packwatcher.part_c.atoms import PlanAtomizer
from packwatcher.part_c.attack_db import AttackDatabase
from packwatcher.part_c.scorer import ScorerConfig, CoherenceScorer, AvailabilityScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_scenario_messages(yaml_path: Path) -> tuple[str, list[str]]:
    """Extract attack family + all bad-actor message texts from scenario YAML."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    family   = config.get("attack_family", "unknown")
    messages: list[str] = []
    for ac in config.get("agent_configs", []):
        if ac.get("alignment") == "misaligned":
            messages.extend(ac.get("bad_actor_script", []))
    return family, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Part C: Coherence + Attack DB")
    parser.add_argument("--data-dir",      default="data",           help="Root data directory")
    parser.add_argument("--model-dir",     default="models",         help="Output model directory")
    parser.add_argument("--scenarios-dir", default="sim/scenarios",  help="Scenarios directory")
    parser.add_argument("--n-epochs",      type=int, default=20)
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    atomizer = PlanAtomizer(use_spacy=False)

    # ===================================================================
    # 1. Build attack database from seeded-bad scenario YAML files
    # ===================================================================
    log.info("Building AttackDatabase from seeded-bad scenarios ...")
    scorer_cfg = ScorerConfig(embedding_dim=384, coherence_hidden=128)
    attack_db  = AttackDatabase(embedding_dim=scorer_cfg.embedding_dim)

    def _get_st(model_name: str, embedding_dim: int):
        """Load SentenceTransformer or fall back to hash-based encoder."""
        try:
            from sentence_transformers import SentenceTransformer
            _st = SentenceTransformer(model_name)
            _st.eval()
            return _st
        except (ImportError, Exception):
            import hashlib
            import numpy as _np

            class _HashEncoder:
                def __init__(self, dim: int) -> None:
                    self.output_dim = dim

                def encode(self, texts, convert_to_tensor=False, batch_size=32):
                    is_str = isinstance(texts, str)
                    if is_str:
                        texts = [texts]
                    vecs = []
                    for text in texts:
                        h   = hashlib.sha256(text.encode("utf-8")).digest()
                        rng = _np.random.RandomState(seed=int.from_bytes(h[:4], "little"))
                        vecs.append(rng.randn(self.output_dim).astype(_np.float32))
                    out = _np.stack(vecs, axis=0)
                    return out.squeeze(0) if is_str else out

            log.warning("sentence_transformers not available; using hash-based encoder for Part C training.")
            return _HashEncoder(embedding_dim)

    st = _get_st(scorer_cfg.sentence_model_name, scorer_cfg.embedding_dim)

    bad_dir = Path(args.scenarios_dir) / "seeded_bad"
    all_training_atom_groups: list = []

    for yaml_path in sorted(bad_dir.glob("*.yaml")):
        family, messages = _load_scenario_messages(yaml_path)
        if not messages:
            continue

        atoms = atomizer.atomize(messages)
        if not atoms:
            continue

        # Embed atoms
        texts  = [a.text for a in atoms]
        embs   = st.encode(texts)
        for atom, emb in zip(atoms, embs):
            import numpy as _np
            atom.embedding = _np.array(emb, dtype=_np.float32)

        n_added = attack_db.add(atoms, label=family, metadata={"source": yaml_path.name})
        log.info("  %s: added %d atoms (family=%s)", yaml_path.stem, n_added, family)
        all_training_atom_groups.append(atoms)

    # Save attack database
    db_path = str(model_path / "attack_db")
    attack_db.save(db_path)
    log.info("Saved AttackDatabase to %s (%d atoms total)", db_path, len(attack_db))

    # ===================================================================
    # 2. Calibrate AvailabilityScorer
    # ===================================================================
    log.info("Calibrating AvailabilityScorer scale ...")
    avail_scorer = AvailabilityScorer(attack_db, scorer_cfg)
    avail_scorer.calibrate_scale(all_training_atom_groups)
    scale_path = str(model_path / "availability_scale.json")
    with open(scale_path, "w") as f:
        json.dump({"distance_scale": avail_scorer.distance_scale}, f, indent=2)
    log.info("Calibrated distance_scale=%.4f  saved to %s", avail_scorer.distance_scale, scale_path)

    # ===================================================================
    # 3. Train CoherenceScorer on REAL sentence embeddings of scenario messages
    # ===================================================================
    log.info("Training CoherenceScorer on real scenario message embeddings ...")

    # Collect (text, label) pairs from ALL scenario YAML files
    # bad_actor_script messages → label 1 (misaligned/dangerous plan)
    # honest agent mock_messages → label 0 (benign)
    all_texts:  list[str]   = []
    all_labels: list[float] = []

    scenarios_root = Path(args.scenarios_dir)
    for yaml_path in sorted(scenarios_root.rglob("*.yaml")):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        for agent_cfg in cfg.get("agent_configs", []):
            alignment = agent_cfg.get("alignment", "honest")
            if alignment == "misaligned":
                for msg in agent_cfg.get("bad_actor_script", []):
                    all_texts.append(str(msg))
                    all_labels.append(1.0)
            else:
                for msg in agent_cfg.get("mock_messages", []):
                    all_texts.append(str(msg))
                    all_labels.append(0.0)

    if not all_texts:
        log.error("No texts collected from scenarios. Cannot train CoherenceScorer.")
        return

    n_pos = sum(1 for l in all_labels if l > 0.5)
    n_neg = len(all_labels) - n_pos
    log.info("  Collected %d texts (%d misaligned, %d honest) from scenario YAMLs",
             len(all_texts), n_pos, n_neg)

    # Embed all texts with the real sentence encoder (frozen)
    log.info("  Embedding texts with %s ...", scorer_cfg.sentence_model_name)
    emb_arrays = st.encode(all_texts, batch_size=64)                  # [N, 384]
    X_emb = torch.from_numpy(np.array(emb_arrays, dtype=np.float32)) # [N, embedding_dim]
    raw_y = torch.tensor(all_labels, dtype=torch.float32)             # [N]

    dataset    = TensorDataset(X_emb, raw_y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    coherence = CoherenceScorer(scorer_cfg).to(device)
    optimizer = torch.optim.Adam(coherence.head.parameters(), lr=args.lr)

    for epoch in range(1, args.n_epochs + 1):
        coherence.train()
        losses: list[float] = []
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = coherence.compute_loss(xb, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % 5 == 0 or epoch == 1:
            log.info("  Epoch %2d/%d  loss=%.4f", epoch, args.n_epochs, float(np.mean(losses)))

    # Save coherence scorer head
    coh_path = str(model_path / "coherence_scorer.pt")
    torch.save(coherence.state_dict(), coh_path)
    log.info("Saved CoherenceScorer to %s", coh_path)
    log.info("Part C training complete.")



if __name__ == "__main__":
    main()
