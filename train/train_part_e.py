"""
train/train_part_e.py — Train Part E: sequential curriculum with CNL + RaPO.

Runs the full attack-family curriculum, applying Collaborative Neural Learning
and RaPO retention penalty to prevent catastrophic forgetting.

For each family in order:
    1. Snapshot current model for RaPO
    2. Snapshot current gradients for CNL
    3. Train on new family data with augmented loss
    4. Evaluate on ALL families (to track forgetting)

Outputs:
    models/curriculum_final.pt           — final model weights after all families
    models/curriculum_accuracy_matrix.json — [n_stages × n_families] accuracy matrix
    models/scar_library.db               — populated scar library

Usage:
    python -m train.train_part_e --data-dir data --model-dir models
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from packwatcher.part_e.cnl import CNLTrainer
from packwatcher.part_e.rapo import RaPORewardShaper
from packwatcher.part_e.scar_library import ScarLibrary
from packwatcher.part_e.curriculum import AttackFamilyCurriculum, FamilyDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_FEATURE_DIM = 128


def _build_simple_classifier(feature_dim: int) -> nn.Module:
    """Simple MLP classifier for binary misalignment detection."""
    return nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2),   # 2 classes: aligned / misaligned
    )


def _evaluate_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Binary accuracy on a dataset."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds  = logits.argmax(dim=-1)
    return float((preds == y).float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Part E: curriculum continual learning")
    parser.add_argument("--data-dir",        default="data",   help="Root data directory")
    parser.add_argument("--model-dir",       default="models", help="Output model directory")
    parser.add_argument("--n-epochs",        type=int, default=10)
    parser.add_argument("--batch-size",      type=int, default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--lambda-retention", type=float, default=0.3,
                        help="RaPO retention loss weight")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # --- Load curriculum data ---
    curriculum_dir = Path(args.data_dir) / "splits" / "curriculum"
    if not (curriculum_dir / "family_order.json").exists():
        log.error("Curriculum data not found. Run sim/data_gen.py first.")
        return

    curriculum = AttackFamilyCurriculum.from_data_dir(str(curriculum_dir))
    log.info("Curriculum: %d families: %s", len(curriculum), curriculum.family_names)

    # --- Build initial model ---
    model     = _build_simple_classifier(_FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cnl       = CNLTrainer(model, optimizer)
    rapo: RaPORewardShaper | None = None

    # Hold the first-seen family data for RaPO retention
    first_family_data: torch.Tensor | None = None

    accuracy_matrix: list[list[float]] = []

    scar_lib = ScarLibrary(str(model_path / "scar_library.db"))

    # --- Sequential curriculum ---
    for stage_idx, family_ds in enumerate(curriculum):
        log.info("=== Stage %d: training on family '%s' ===", stage_idx, family_ds.family)

        X_train = family_ds.X_train.to(device)
        y_train = family_ds.y_train.to(device)
        X_eval  = family_ds.X_eval.to(device)
        y_eval  = family_ds.y_eval.to(device)

        if first_family_data is None:
            first_family_data = X_train[:32]   # hold 32 samples from first family for RaPO

        # --- Snapshot for RaPO at start of each new family ---
        rapo = RaPORewardShaper.snapshot(
            model, lambda_retention=args.lambda_retention, temperature=2.0
        )

        # --- Snapshot gradients for CNL (compute on previous families' eval) ---
        if stage_idx > 0 and first_family_data is not None:
            model.train()
            old_logits = model(first_family_data)
            # Synthetic old-task loss: cross-entropy with identity labels
            old_labels = torch.zeros(len(first_family_data), dtype=torch.long, device=device)
            old_loss   = torch.nn.functional.cross_entropy(old_logits, old_labels)
            optimizer.zero_grad(set_to_none=True)
            old_loss.backward()
            cnl.snapshot_gradients()
            optimizer.zero_grad(set_to_none=True)

        # --- Train on new family ---
        dataset    = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        ce_loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, args.n_epochs + 1):
            model.train()
            epoch_losses: list[float] = []

            for xb, yb in dataloader:
                # New task loss (cross-entropy on this family)
                new_loss = ce_loss_fn(model(xb), yb.long())

                # RaPO retention penalty on first-family data (if we have it)
                if rapo is not None and first_family_data is not None and stage_idx > 0:
                    try:
                        total_loss = rapo.augmented_loss(new_loss, model, first_family_data)
                    except Exception:
                        total_loss = new_loss
                else:
                    total_loss = new_loss

                # CNL step (projects out conflicting gradients)
                stats = cnl.step(total_loss)
                epoch_losses.append(float(total_loss.item()))

            if epoch % 5 == 0 or epoch == 1:
                acc = _evaluate_accuracy(model, X_eval, y_eval)
                log.info(
                    "  Epoch %2d/%d  loss=%.4f  this-family-acc=%.3f",
                    epoch, args.n_epochs,
                    float(np.mean(epoch_losses)),
                    acc,
                )

        # --- Evaluate ALL families after this stage ---
        def eval_fn(X_ev, y_ev):
            return _evaluate_accuracy(model, X_ev.to(device), y_ev.to(device))

        stage_accs = curriculum.evaluate_all(eval_fn)
        accuracy_matrix.append(stage_accs)
        log.info(
            "Stage %d complete. Accuracies across all families: %s",
            stage_idx,
            " ".join(f"{a:.3f}" for a in stage_accs),
        )

        # --- Add confirmed family fingerprint to scar library ---
        with torch.no_grad():
            # Fingerprint = mean of training embeddings
            fingerprint = X_train.cpu().mean(dim=0).numpy()
        scar_lib.add_scar(
            fingerprint_vec = fingerprint,
            family          = family_ds.family,
            nearest_atoms   = [],
            metadata        = {"stage": stage_idx, "n_samples": len(X_train)},
        )

    # --- Compute forgetting metrics ---
    curriculum_result = curriculum.build_result()
    log.info("Curriculum BWT = %.4f", curriculum_result.backward_transfer)
    log.info("Pass? BWT ≥ -0.10: %s", curriculum_result.backward_transfer >= -0.10)

    # --- Save outputs ---
    final_path = str(model_path / "curriculum_final.pt")
    torch.save(model.state_dict(), final_path)
    log.info("Saved final curriculum model to %s", final_path)

    matrix_path = str(model_path / "curriculum_accuracy_matrix.json")
    with open(matrix_path, "w") as f:
        json.dump({
            "family_names":     curriculum.family_names,
            "accuracy_matrix":  accuracy_matrix,
            "backward_transfer": curriculum_result.backward_transfer,
        }, f, indent=2)
    log.info("Saved accuracy matrix to %s", matrix_path)
    log.info("Part E training complete.")


if __name__ == "__main__":
    main()
